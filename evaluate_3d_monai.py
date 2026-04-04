import argparse
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import albumentations as albu
import cv2
import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torch.utils import data
from tqdm import tqdm

from models import sam_feat_seg_model_registry


FILENAME_RE = re.compile(r"^(P\d+)_(T\d+)_(\d+)\.[^.]+$")


def _natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def _ensure_multiclass_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if logits.shape[1] == num_classes:
        return logits
    if num_classes == 2 and logits.shape[1] == 1:
        return torch.cat([-logits, logits], dim=1)
    return logits


def _find_label_path(labels_dir: str, basename_no_ext: str) -> str:
    pattern = os.path.join(labels_dir, basename_no_ext + ".*")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Missing label for {basename_no_ext} in {labels_dir}")
    return candidates[0]


def _load_image_any(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        if path.lower().endswith(".tif") or path.lower().endswith(".tiff"):
            import tifffile

            arr = tifffile.imread(path)
        else:
            raise FileNotFoundError(f"Cannot read image: {path}")

    if arr.ndim == 2:
        arr = arr[..., None]

    if arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    return arr


def _to_input_channels(img: np.ndarray, input_channels: int) -> np.ndarray:
    if input_channels == 1:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)[..., None]
        elif img.shape[2] != 1:
            img = img.mean(axis=2, keepdims=True)
    elif input_channels == 3:
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
    else:
        raise ValueError(f"Unsupported input_channels={input_channels}. Use 1 or 3.")

    img = img.astype(np.float32)

    # Keep behavior aligned with train_dataset normalization expectations.
    v_min, v_max = img.min(), img.max()
    if v_max > 255.0 or v_min < 0.0:
        if v_max - v_min > 1e-8:
            img = (img - v_min) / (v_max - v_min) * 255.0
        else:
            img = np.zeros_like(img, dtype=np.float32)

    return img


def _build_transform(img_size: int, input_channels: int):
    if input_channels == 1:
        mean, std = [0.5], [0.5]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    return albu.Compose(
        [
            albu.LongestMaxSize(max_size=img_size),
            albu.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            albu.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ],
        is_check_shapes=False,
    )


def _parse_volume_info(filename: str) -> Tuple[str, str, int]:
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    m = FILENAME_RE.match(base)
    if m is None:
        # Retry using stem, useful when there are multiple dots in extension variants.
        m = FILENAME_RE.match(stem + os.path.splitext(base)[1])
    if m is None:
        raise ValueError(
            f"Filename '{base}' does not match expected pattern P<id>_T<id>_<slice>.<ext>"
        )
    patient, timepoint, slice_idx = m.group(1), m.group(2), int(m.group(3))
    return patient, timepoint, slice_idx


class SliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels_dir: str,
        transform,
        input_channels: int,
    ):
        self.image_paths = image_paths
        self.labels_dir = labels_dir
        self.transform = transform
        self.input_channels = input_channels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        basename = os.path.basename(img_path)
        stem = os.path.splitext(basename)[0]

        patient, timepoint, slice_idx = _parse_volume_info(basename)

        lbl_path = _find_label_path(self.labels_dir, stem)

        img = _load_image_any(img_path)
        img = _to_input_channels(img, self.input_channels)

        mask = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {lbl_path}")
        mask = (mask > 127).astype(np.uint8)
        mask = mask[..., None]

        augmented = self.transform(image=img, mask=mask)
        img = augmented["image"].astype(np.float32)
        mask = augmented["mask"]
        mask = (mask[..., 0] > 0).astype(np.uint8)

        img = np.transpose(img, (2, 0, 1))

        return {
            "img": img,
            "mask": mask,
            "patient": patient,
            "timepoint": timepoint,
            "slice_idx": slice_idx,
            "basename": basename,
        }


@dataclass
class VolumeResult:
    patient: str
    timepoint: str
    n_slices: int
    min_slice_idx: int
    max_slice_idx: int
    dice: float
    ppv: float
    tpr: float
    hd95: float
    tp: int
    fp: int
    fn: int
    tn: int
    status: str


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return numer / denom


def _compute_volume_metrics(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray,
    dice_metric: DiceMetric,
    hd95_metric: HausdorffDistanceMetric,
) -> Tuple[float, float, float, float, int, int, int, int, str]:
    pred_fg = pred_3d.astype(bool)
    gt_fg = gt_3d.astype(bool)

    tp = int(np.logical_and(pred_fg, gt_fg).sum())
    fp = int(np.logical_and(pred_fg, np.logical_not(gt_fg)).sum())
    fn = int(np.logical_and(np.logical_not(pred_fg), gt_fg).sum())
    tn = int(np.logical_and(np.logical_not(pred_fg), np.logical_not(gt_fg)).sum())

    ppv = _safe_div(tp, tp + fp)
    tpr = _safe_div(tp, tp + fn)

    gt_any = bool(gt_fg.any())
    pred_any = bool(pred_fg.any())

    # Explicit policies for edge cases, with status tracking.
    if (not gt_any) and (not pred_any):
        return 1.0, ppv, tpr, float("nan"), tp, fp, fn, tn, "both_empty"

    pred_oh = np.stack([np.logical_not(pred_fg), pred_fg], axis=0).astype(np.float32)
    gt_oh = np.stack([np.logical_not(gt_fg), gt_fg], axis=0).astype(np.float32)

    pred_t = torch.from_numpy(pred_oh).unsqueeze(0)
    gt_t = torch.from_numpy(gt_oh).unsqueeze(0)

    dice_metric.reset()
    dice_val_t = dice_metric(y_pred=pred_t, y=gt_t)
    dice_val = float(dice_val_t.squeeze().item())

    hd95_val = float("nan")
    if gt_any and pred_any:
        hd95_metric.reset()
        hd95_val_t = hd95_metric(y_pred=pred_t, y=gt_t)
        hd95_raw = float(hd95_val_t.squeeze().item())
        if math.isfinite(hd95_raw):
            hd95_val = hd95_raw

    status = "ok"
    if (not gt_any) and pred_any:
        status = "gt_empty"
    elif gt_any and (not pred_any):
        status = "pred_empty"

    return dice_val, ppv, tpr, hd95_val, tp, fp, fn, tn, status


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="3D volume-wise MONAI evaluation from 2D slices (MSLesSeg naming)."
    )
    p.add_argument("--test_img_path", type=str, required=True, help="Directory with test slices")
    p.add_argument("--test_label_path", type=str, required=True, help="Directory with test labels")
    p.add_argument("--model_path", type=str, required=True, help="Path to checkpoint .pth")
    p.add_argument(
        "--model_type",
        type=str,
        default="vit_l_hardnet",
        choices=["vit_b", "vit_l", "vit_h", "vit_b_hardnet", "vit_l_hardnet", "vit_h_hardnet"],
    )
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--input_channels", type=int, default=1, choices=[1, 3])
    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--iter_2stage", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--img_glob", type=str, default="*.tif", help="Glob pattern for test images")
    p.add_argument(
        "--eval_stage",
        type=str,
        default="final",
        choices=["stage1", "final"],
        help="Evaluate first-stage output or final refined output.",
    )
    p.add_argument("--output_csv", type=str, default="metrics_per_volume.csv")
    p.add_argument("--output_json", type=str, default="metrics_summary.json")
    p.add_argument("--device", type=str, default="cuda")
    return p


def main(args):
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_paths = sorted(glob.glob(os.path.join(args.test_img_path, args.img_glob)))
    if not image_paths:
        raise RuntimeError(f"No images found in {args.test_img_path} with pattern {args.img_glob}")

    transform = _build_transform(args.img_size, args.input_channels)
    ds = SliceDataset(
        image_paths=image_paths,
        labels_dir=args.test_label_path,
        transform=transform,
        input_channels=args.input_channels,
    )
    loader = data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = sam_feat_seg_model_registry[args.model_type](
        num_classes=args.num_classes,
        checkpoint=args.model_path,
        img_size=args.img_size,
        iter_2stage=args.iter_2stage,
        input_channels=args.input_channels,
    )
    model.to(device)
    model.eval()

    volume_data: Dict[Tuple[str, str], Dict[str, Dict[int, np.ndarray]]] = {}

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc="Inference 2D"):
            img = batch["img"].to(device)
            mask = batch["mask"].numpy().astype(np.uint8)
            patients = batch["patient"]
            timepoints = batch["timepoint"]
            slice_indices = batch["slice_idx"].tolist()

            out1, out2, _ = model(img)
            logits = out2 if args.eval_stage == "final" else out1
            logits = _ensure_multiclass_logits(logits, args.num_classes)
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            pred = (pred > 0).astype(np.uint8)

            for i in range(pred.shape[0]):
                key = (patients[i], timepoints[i])
                if key not in volume_data:
                    volume_data[key] = {"pred": {}, "gt": {}}

                s_idx = int(slice_indices[i])
                volume_data[key]["pred"][s_idx] = pred[i]
                volume_data[key]["gt"][s_idx] = mask[i]

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, directed=False, reduction="mean")

    results: List[VolumeResult] = []
    global_tp = global_fp = global_fn = global_tn = 0

    sorted_keys = sorted(volume_data.keys(), key=lambda k: (_natural_key(k[0]), _natural_key(k[1])))

    for patient, timepoint in tqdm(sorted_keys, desc="Metrics 3D"):
        pred_map = volume_data[(patient, timepoint)]["pred"]
        gt_map = volume_data[(patient, timepoint)]["gt"]

        all_indices = sorted(pred_map.keys())
        pred_stack = np.stack([pred_map[s] for s in all_indices], axis=-1)
        gt_stack = np.stack([gt_map[s] for s in all_indices], axis=-1)

        (
            dice_val,
            ppv_val,
            tpr_val,
            hd95_val,
            tp,
            fp,
            fn,
            tn,
            status,
        ) = _compute_volume_metrics(
            pred_3d=pred_stack,
            gt_3d=gt_stack,
            dice_metric=dice_metric,
            hd95_metric=hd95_metric,
        )

        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn

        results.append(
            VolumeResult(
                patient=patient,
                timepoint=timepoint,
                n_slices=len(all_indices),
                min_slice_idx=min(all_indices),
                max_slice_idx=max(all_indices),
                dice=dice_val,
                ppv=ppv_val,
                tpr=tpr_val,
                hd95=hd95_val,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                status=status,
            )
        )

    def _nanmean(vals: List[float]) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return float("nan")
        return float(np.nanmean(arr))

    macro = {
        "dice": _nanmean([r.dice for r in results]),
        "ppv": _nanmean([r.ppv for r in results]),
        "tpr": _nanmean([r.tpr for r in results]),
        "hd95": _nanmean([r.hd95 for r in results]),
        "num_volumes": len(results),
        "num_hd95_valid": int(np.sum(np.isfinite([r.hd95 for r in results]))),
    }

    micro = {
        "dice": _safe_div(2.0 * global_tp, 2.0 * global_tp + global_fp + global_fn),
        "ppv": _safe_div(global_tp, global_tp + global_fp),
        "tpr": _safe_div(global_tp, global_tp + global_fn),
        "tp": int(global_tp),
        "fp": int(global_fp),
        "fn": int(global_fn),
        "tn": int(global_tn),
    }

    out_csv_dir = os.path.dirname(args.output_csv)
    if out_csv_dir:
        os.makedirs(out_csv_dir, exist_ok=True)

    out_json_dir = os.path.dirname(args.output_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient",
                "timepoint",
                "n_slices",
                "min_slice_idx",
                "max_slice_idx",
                "dice",
                "ppv",
                "tpr",
                "hd95",
                "tp",
                "fp",
                "fn",
                "tn",
                "status",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "patient": r.patient,
                    "timepoint": r.timepoint,
                    "n_slices": r.n_slices,
                    "min_slice_idx": r.min_slice_idx,
                    "max_slice_idx": r.max_slice_idx,
                    "dice": r.dice,
                    "ppv": r.ppv,
                    "tpr": r.tpr,
                    "hd95": r.hd95,
                    "tp": r.tp,
                    "fp": r.fp,
                    "fn": r.fn,
                    "tn": r.tn,
                    "status": r.status,
                }
            )

    summary = {
        "config": {
            "test_img_path": args.test_img_path,
            "test_label_path": args.test_label_path,
            "model_path": args.model_path,
            "model_type": args.model_type,
            "input_channels": args.input_channels,
            "img_size": args.img_size,
            "img_glob": args.img_glob,
            "eval_stage": args.eval_stage,
        },
        "macro": macro,
        "micro": micro,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== 3D MONAI Evaluation Summary ===")
    print(f"Volumes: {len(results)}")
    print(
        f"Macro -> Dice: {macro['dice']:.6f}, PPV: {macro['ppv']:.6f}, "
        f"TPR: {macro['tpr']:.6f}, HD95: {macro['hd95']:.6f}"
    )
    print(
        f"Micro -> Dice: {micro['dice']:.6f}, PPV: {micro['ppv']:.6f}, "
        f"TPR: {micro['tpr']:.6f}"
    )
    print(f"Saved per-volume metrics: {args.output_csv}")
    print(f"Saved summary metrics: {args.output_json}")


if __name__ == "__main__":
    parsed_args = build_argparser().parse_args()
    main(parsed_args)
