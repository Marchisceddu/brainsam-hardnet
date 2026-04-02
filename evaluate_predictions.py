"""Evaluate segmentation predictions against ground-truth masks (2D PNG/TIF).

This script is designed for the Brain-SAM pipeline in this repo:
  1) Run inference.py to produce predicted masks (values are class indices: 0/1 for binary).
  2) Run this script to compute DSC (Dice) and F1 score.

Notes:
- For binary segmentation with positive class=1, Dice and F1 are numerically identical:
    Dice = F1 = 2TP / (2TP + FP + FN)
  We report both for convenience.
- Ground-truth masks produced by convert_mslesseg_to_brainsam.py are 0/1.
  Many image viewers render value 1 almost black; that is expected.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def dice(self, empty_score: float = 1.0) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        if denom == 0:
            return float(empty_score)
        return float(2 * self.tp / denom)

    def f1(self, empty_score: float = 1.0) -> float:
        # Same as Dice for binary segmentation
        return self.dice(empty_score=empty_score)

    def precision(self, empty_score: float = 1.0) -> float:
        denom = self.tp + self.fp
        if denom == 0:
            return float(empty_score)
        return float(self.tp / denom)

    def recall(self, empty_score: float = 1.0) -> float:
        denom = self.tp + self.fn
        if denom == 0:
            return float(empty_score)
        return float(self.tp / denom)


def _list_images(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def _load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        # if RGB by mistake, use first channel
        arr = arr[..., 0]
    return arr


def _binarize(arr: np.ndarray, positive_mode: str) -> np.ndarray:
    if positive_mode == "gt_gt1":
        return (arr > 0)
    if positive_mode == "eq1":
        return (arr == 1)
    if positive_mode == "gt127":
        return (arr > 127)
    raise ValueError(f"Unknown positive_mode: {positive_mode}")


def _accumulate(m: Metrics, pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch pred={pred.shape}, gt={gt.shape}")
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    m.tp += int(np.logical_and(pred, gt).sum())
    m.fp += int(np.logical_and(pred, np.logical_not(gt)).sum())
    m.fn += int(np.logical_and(np.logical_not(pred), gt).sum())
    m.tn += int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute DSC (Dice) and F1 from predicted vs GT masks")
    p.add_argument("--pred_dir", type=str, required=True, help="Folder with predicted masks (e.g., from inference.py)")
    p.add_argument("--gt_dir", type=str, required=True, help="Folder with ground-truth masks")
    p.add_argument(
        "--exts",
        type=str,
        nargs="+",
        default=[".png"],
        help="Mask file extensions to include (default: .png)",
    )
    p.add_argument(
        "--positive_mode",
        type=str,
        default="gt_gt1",
        choices=["gt_gt1", "eq1", "gt127"],
        help=(
            "How to convert masks to binary. "
            "gt_gt1: value>0 (recommended for 0/1 labels and 0/255 viz labels). "
            "eq1: value==1 (strict for 0/1). "
            "gt127: value>127 (for 0/255)."
        ),
    )
    p.add_argument(
        "--empty_score",
        type=float,
        default=1.0,
        help="Score returned when both pred and gt are empty (default 1.0)",
    )
    p.add_argument(
        "--per_image",
        action="store_true",
        help="Also print per-image DSC/F1",
    )
    p.add_argument(
        "--resize_pred_to_gt",
        action="store_true",
        help="If shapes differ, resize prediction to GT size using nearest-neighbor",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    exts = tuple(e.lower() for e in args.exts)

    if not pred_dir.exists():
        raise SystemExit(f"pred_dir not found: {pred_dir}")
    if not gt_dir.exists():
        raise SystemExit(f"gt_dir not found: {gt_dir}")

    pred_files = _list_images(pred_dir, exts)
    if not pred_files:
        raise SystemExit(f"No prediction files found in {pred_dir} with exts={exts}")

    total = Metrics()
    per_img_scores: List[Tuple[str, float]] = []

    missing = 0
    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            missing += 1
            continue

        pred_arr = _load_mask(pred_path)
        gt_arr = _load_mask(gt_path)

        if pred_arr.shape != gt_arr.shape:
            if args.resize_pred_to_gt:
                pred_img = Image.fromarray(pred_arr.astype(np.uint8), mode="L")
                pred_img = pred_img.resize((gt_arr.shape[1], gt_arr.shape[0]), resample=Image.NEAREST)
                pred_arr = np.array(pred_img)
            else:
                raise SystemExit(
                    f"Shape mismatch for {pred_path.name}: pred={pred_arr.shape}, gt={gt_arr.shape}. "
                    "Use --resize_pred_to_gt to auto-fix."
                )

        pred_bin = _binarize(pred_arr, args.positive_mode)
        gt_bin = _binarize(gt_arr, args.positive_mode)

        m = Metrics()
        _accumulate(m, pred_bin, gt_bin)
        _accumulate(total, pred_bin, gt_bin)

        if args.per_image:
            per_img_scores.append((pred_path.name, m.dice(empty_score=args.empty_score)))

    if missing:
        print(f"[WARN] Missing GT for {missing} prediction files (ignored)")

    dice = total.dice(empty_score=args.empty_score)
    f1 = total.f1(empty_score=args.empty_score)
    prec = total.precision(empty_score=args.empty_score)
    rec = total.recall(empty_score=args.empty_score)

    print("=== Metrics (global pixel-aggregated) ===")
    print(f"Dice/DSC:   {dice:.6f}")
    print(f"F1 score:   {f1:.6f}")
    print(f"Precision:  {prec:.6f}")
    print(f"Recall:     {rec:.6f}")
    print(f"TP={total.tp} FP={total.fp} FN={total.fn} TN={total.tn}")

    if args.per_image:
        print("=== Per-image DSC ===")
        for name, s in per_img_scores:
            print(f"{name}\t{s:.6f}")


if __name__ == "__main__":
    main()
