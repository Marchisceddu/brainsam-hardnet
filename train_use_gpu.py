import argparse
import os, glob
import sys
import shutil
import warnings
import contextlib
import cv2
import albumentations as albu
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
from monai.losses import DiceLoss, DiceCELoss
from models import sam_feat_seg_model_registry
from train_dataset import Dataset

try:
    import wandb
except Exception:
    wandb = None


parser = argparse.ArgumentParser(description='PyTorch Brain-SAM Training')

parser.add_argument('--epochs', default=50, type=int, required=False,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=30, type=int, required=False,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=4, type=int, required=False,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.0003, type=float, required=False,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, required=False,
                    metavar='W', help='weight decay (default: 1e-4)',)
parser.add_argument('--num_workers', default=2, type=int, required=False,
                    help='Number of DataLoader workers. Use <=2 on constrained environments.')
parser.add_argument('--resume', default='', type=str, required=False,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=7, type=int, required=False,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int, required=False,
                    help='GPU id to use (single-GPU mode only).')
parser.add_argument("--iter_2stage", type=int, default=1, required=False,
                    help='Number of second-stage iterations (fixed to 1 for training)')
parser.add_argument("--num_classes", type=int, default=2, required=False,)
parser.add_argument("--save_dir", type=str, default='', required=False,)
parser.add_argument("--load_saved_model", action='store_true',
                    help='whether freeze encoder of the segmenter')
parser.add_argument('--model_type', type=str, default="vit_l", required=False,
                    help='Model key: vit_b, vit_l, vit_h (legacy) or vit_b_hardnet, vit_l_hardnet, vit_h_hardnet (HardNet first stage).')
parser.add_argument('--input_channels', type=int, default=1, choices=[1, 3],
                    help='Input channels: 1 for grayscale MRI (recommended), 3 for RGB.')
parser.add_argument('--format_img', type=str, default='.tif', required=False, help='')
parser.add_argument('--format_img_glob', type=str, default='*.tif', required=False, help='')
parser.add_argument("--img_size", type=int, default=1024, required=False,
                    help='Original image size')
parser.add_argument('--img_path', type=str, required=False, default=r'',
                    help='(Legacy) Single image directory for auto 80/20 split')
parser.add_argument('--label_path', type=str, required=False, default=r'',
                    help='(Legacy) Single label directory for auto 80/20 split')
parser.add_argument('--model_checkpoint', type=str, required=False, default=r'')

# Separate train/val directories (recommended — no data leakage)
parser.add_argument('--train_img_path', type=str, default='',
                    help='Directory with training images')
parser.add_argument('--train_label_path', type=str, default='',
                    help='Directory with training labels')
parser.add_argument('--val_img_path', type=str, default='',
                    help='Directory with validation images')
parser.add_argument('--val_label_path', type=str, default='',
                    help='Directory with validation labels')

# Loss function selection
parser.add_argument('--loss_type', type=str, default='DiceCE',
                    choices=['CE', 'Dice', 'DiceCE'],
                    help='Loss function to use: '
                         'CE = CrossEntropy only, '
                         'Dice = Dice only (from MONAI), '
                         'DiceCE = Dice + CrossEntropy combined (from MONAI, default). '
                         'DiceCE is recommended for small-lesion segmentation (e.g. MS).')

# Gradient accumulation
parser.add_argument('--accum_steps', default=1, type=int,
                    help='Number of gradient accumulation steps. '
                         'Effective batch size = batch_size × accum_steps (× num_gpus if distributed). '
                         'Increase this to simulate larger batches without extra GPU memory. '
                         'Default: 1 (no accumulation).')

# Multi-GPU / Distributed arguments
parser.add_argument('--distributed', action='store_true',
                    help='Use DistributedDataParallel for multi-GPU training. '
                         'Launch with: torchrun --nproc_per_node=NUM_GPUS train_use_gpu.py --distributed ...')

# W&B logging
parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                    help='Disable Weights & Biases logging.')
parser.set_defaults(use_wandb=True)
parser.add_argument('--wandb_entity', type=str,
                default='marcopilia02-university-of-cagliari',
                help='W&B entity/team name.')
parser.add_argument('--wandb_project', type=str,
                default='brain-hardnet-sam-mslesseg',
                help='W&B project name.')
parser.add_argument('--wandb_mode', type=str, default='online',
                choices=['online', 'offline', 'disabled'],
                help='W&B execution mode.')
parser.add_argument('--wandb_run_name', type=str, default='',
                help='Optional custom W&B run name.')
parser.add_argument('--wandb_log_images_every', type=int, default=1,
                help='Log qualitative validation images every N epochs.')
parser.add_argument('--wandb_max_images', type=int, default=2,
                help='Max number of validation samples to log per image logging epoch.')
parser.add_argument('--use_tensorboard', action='store_true',
                help='Enable TensorBoard logging (disabled by default to avoid TensorFlow startup warnings).')


# ──────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────

def iou_score(mask, logits, eps=1e-5):
    """Compute mean IoU over a batch (foreground only).

    Args:
        mask:   (B, H, W) ground-truth integer labels
        logits: (B, C, H, W) raw model logits
        eps:    smoothing to avoid division by zero
    Returns:
        scalar IoU averaged over the batch
    """
    mask = mask.detach().cpu().numpy()
    preds = logits.argmax(dim=1).detach().cpu().numpy()

    iou = 0.0
    for i in range(mask.shape[0]):
        gt = mask[i] > 0
        pr = preds[i] > 0
        intersection = (gt & pr).sum()
        union = (gt | pr).sum()
        iou += (intersection + eps) / (union + eps)
    return iou / mask.shape[0]


def dsc_score(mask, logits, eps=1e-5):
    """Compute mean Dice Similarity Coefficient over a batch (foreground only).

    DSC = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        mask:   (B, H, W) ground-truth integer labels
        logits: (B, C, H, W) raw model logits
        eps:    smoothing to avoid division by zero
    Returns:
        scalar DSC averaged over the batch
    """
    mask = mask.detach().cpu().numpy()
    preds = logits.argmax(dim=1).detach().cpu().numpy()

    dsc = 0.0
    for i in range(mask.shape[0]):
        gt = mask[i] > 0
        pr = preds[i] > 0
        intersection = (gt & pr).sum()
        dsc += (2.0 * intersection + eps) / (gt.sum() + pr.sum() + eps)
    return dsc / mask.shape[0]


def _ensure_multiclass_logits(logits, num_classes):
    """Convert single-channel logits to 2-class logits when needed."""
    if logits.shape[1] == num_classes:
        return logits
    if num_classes == 2 and logits.shape[1] == 1:
        # Two-class equivalent logits from a binary logit.
        return torch.cat([-logits, logits], dim=1)
    return logits


# ──────────────────────────────────────────────
#  Loss builder
# ──────────────────────────────────────────────

def build_criterion(loss_type, num_classes, device):
    """Build the loss function based on --loss_type.

    Args:
        loss_type: one of 'CE', 'Dice', 'DiceCE'
        num_classes: number of segmentation classes
        device: torch device to place the loss on
    Returns:
        A callable criterion(logits, target) that accepts
        logits of shape (B, C, H, W) and target of shape (B, H, W).
    """
    if loss_type == 'CE':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif loss_type == 'Dice':
        # softmax=True because model outputs raw logits;
        # include_background=False so background class doesn't dominate the Dice
        criterion = DiceLoss(
            softmax=True,
            to_onehot_y=True,
            include_background=False,
            reduction='mean',
        )
    elif loss_type == 'DiceCE':
        # MONAI DiceCELoss: combines Dice + CE for better small-lesion focus
        criterion = DiceCELoss(
            softmax=True,
            to_onehot_y=True,
            include_background=False,
            reduction='mean',
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return criterion.to(device)


# ──────────────────────────────────────────────
#  Distributed helpers
# ──────────────────────────────────────────────

def setup_distributed():
    """Initialize the distributed process group.
    Expects environment variables set by `torchrun`:
        LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    """
    # Ensure NCCL works on Kaggle / multi-GPU setups
    os.environ.setdefault('NCCL_DEBUG', 'WARN')
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Returns True if this is rank 0 (or if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def reduce_tensor(tensor, device, op=dist.ReduceOp.SUM):
    """All-reduce a Python scalar across all ranks and return the result.

    Args:
        tensor: a Python float/int scalar
        device: torch.device for the temporary tensor
        op: reduction operation (default: SUM)
    Returns:
        The all-reduced value as a Python float.
    """
    t = torch.tensor(tensor, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=op)
    return t.item()


def _tensor_to_vis_image(img_tensor):
    """Convert normalized CHW tensor to uint8 HWC image for visualization."""
    img = img_tensor.detach().cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)

    # De-normalize ImageNet stats used in the dataset pipeline.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img = (img * std + mean).clip(0.0, 1.0)
    img = (img.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return img


def _logits_to_fg_mask(logits_tensor):
    """Convert logits tensor [C,H,W] to foreground binary mask [H,W]."""
    logits = logits_tensor.detach().cpu().float()
    if logits.ndim == 2:
        return (logits > 0).numpy().astype(np.uint8)
    if logits.shape[0] == 1:
        return (torch.sigmoid(logits[0]) > 0.5).numpy().astype(np.uint8)
    return (logits.argmax(dim=0) > 0).numpy().astype(np.uint8)


def _dice_from_masks(gt_mask, pred_mask, eps=1e-5):
    gt = gt_mask.astype(bool)
    pr = pred_mask.astype(bool)
    inter = (gt & pr).sum()
    return (2.0 * inter + eps) / (gt.sum() + pr.sum() + eps)


def _draw_contours(base_img, gt_mask, pred_mask):
    """Draw GT in red and prediction in cyan with alpha blend for readability."""
    canvas = base_img.copy()
    gt_u8 = (gt_mask.astype(np.uint8) * 255)
    pr_u8 = (pred_mask.astype(np.uint8) * 255)

    # Soft filled overlays to make small lesions easier to see.
    gt_fill = np.zeros_like(canvas)
    gt_fill[..., 0] = gt_u8 # Red channel (RGB format)
    pr_fill = np.zeros_like(canvas)
    pr_fill[..., 1] = pr_u8 # Green
    pr_fill[..., 2] = pr_u8 # Blue -> Cyan (Green + Blue)

    canvas = cv2.addWeighted(canvas, 1.0, gt_fill, 0.35, 0)
    canvas = cv2.addWeighted(canvas, 1.0, pr_fill, 0.35, 0)

    # Crisp contours on top (using RGB colors since canvas and wandb expect RGB)
    gt_cnts, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pr_cnts, _ = cv2.findContours(pr_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, gt_cnts, -1, (255, 0, 0), 2)  # Red
    cv2.drawContours(canvas, pr_cnts, -1, (0, 255, 255), 2)  # Cyan
    return canvas


def _build_wandb_qualitative_panel(img, gt, pred1, pred2):
    """Create a 2x2 panel: raw image, GT+stage1, GT+stage2, error map."""
    h, w = img.shape[:2]

    raw = img.copy()
    stage1 = _draw_contours(img, gt, pred1)
    stage2 = _draw_contours(img, gt, pred2)

    fp = np.logical_and(pred2 == 1, gt == 0)
    fn = np.logical_and(pred2 == 0, gt == 1)
    err = img.copy()
    err[fp] = [255, 255, 0]   # False Positive: Yellow (Red+Green) in RGB
    err[fn] = [255, 0, 255]   # False Negative: Magenta (Red+Blue) in RGB

    cv2.putText(raw, 'Image', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(stage1, 'Stage1: GT(red) Pred(cyan)', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(stage2, 'Final: GT(red) Pred(cyan)', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(err, 'Error map: FP(yellow) FN(magenta)', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    top = np.concatenate([raw, stage1], axis=1)
    bottom = np.concatenate([stage2, err], axis=1)
    panel = np.concatenate([top, bottom], axis=0)
    return panel


def _log_wandb_validation_images(wandb_run, epoch, vis_dict, max_images):
    if wandb_run is None or vis_dict is None:
        return

    imgs = vis_dict['img']
    gts = vis_dict['mask']
    out1 = vis_dict['out1']
    out2 = vis_dict['out2']

    log_items = []
    n = min(max_images, imgs.shape[0])
    for i in range(n):
        img_np = _tensor_to_vis_image(imgs[i])
        gt_np = (gts[i].detach().cpu().numpy() > 0).astype(np.uint8)
        p1_np = _logits_to_fg_mask(out1[i])
        p2_np = _logits_to_fg_mask(out2[i])

        d1 = _dice_from_masks(gt_np, p1_np)
        d2 = _dice_from_masks(gt_np, p2_np)
        panel = _build_wandb_qualitative_panel(img_np, gt_np, p1_np, p2_np)
        log_items.append(
            wandb.Image(
                panel,
                caption=f"epoch={epoch} sample={i} dice_stage1={d1:.4f} dice_final={d2:.4f}",
            )
        )

    wandb_run.log({"val/qualitative_panels": log_items, "epoch": epoch})


# ──────────────────────────────────────────────
#  Main worker
# ──────────────────────────────────────────────

def main_worker(args):
    # ── Setup device ──
    if args.distributed:
        local_rank = setup_distributed()
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process():
            print(f"Distributed training on {dist.get_world_size()} GPUs")
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f"Use GPU: {args.gpu} for training")

    # ── Build model ──
    model = sam_feat_seg_model_registry[args.model_type](
        num_classes=args.num_classes,
        checkpoint=args.model_checkpoint,
        img_size=args.img_size,
        iter_2stage=args.iter_2stage,
        input_channels=args.input_channels,
    )
    model.to(device)

    # freeze image_encoder
    for name, param in model.named_parameters():
        if "image_encoder" in name:
            param.requires_grad = False

    # ── Wrap with DDP if distributed ──
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            if is_main_process():
                print(f"=> loading checkpoint '{args.resume}'")
            if args.distributed:
                map_location = {'cuda:0': f'cuda:{local_rank}'}
            else:
                map_location = device
            checkpoint = torch.load(args.resume, map_location=map_location,
                                    weights_only=False)
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            # handle DDP state_dict prefix
            if args.distributed and not any(k.startswith('module.') for k in state_dict):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif not args.distributed and any(k.startswith('module.') for k in state_dict):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if is_main_process():
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            if is_main_process():
                print(f"=> no checkpoint found at '{args.resume}'")

    cudnn.benchmark = True

    # ── Data loading ──
    # Resize preserving aspect ratio, then pad to square with black borders.
    # Normalization uses ImageNet mean/std to match SAM's frozen encoder pretraining.
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    if args.input_channels == 1:
        norm_mean, norm_std = [0.5], [0.5]
    else:
        norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = albu.Compose([
        albu.LongestMaxSize(max_size=args.img_size),
        albu.PadIfNeeded(
            min_height=args.img_size,
            min_width=args.img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Normalize(
            mean=norm_mean,
            std=norm_std,
            max_pixel_value=255.0,
        ),
    ], is_check_shapes=False)

    val_transform = albu.Compose([
        albu.LongestMaxSize(max_size=args.img_size),
        albu.PadIfNeeded(
            min_height=args.img_size,
            min_width=args.img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
        ),
        albu.Normalize(
            mean=norm_mean,
            std=norm_std,
            max_pixel_value=255.0,
        ),
    ], is_check_shapes=False)

    use_separate_dirs = bool(args.train_img_path and args.val_img_path)

    if use_separate_dirs:
        # ── Separate train/val directories (recommended) ──
        train_img_dir = args.train_img_path
        train_lbl_dir = args.train_label_path
        val_img_dir = args.val_img_path
        val_lbl_dir = args.val_label_path

        train_names = [os.path.basename(x).split('.')[0]
                       for x in glob.glob(os.path.join(train_img_dir, args.format_img_glob))]
        val_names = [os.path.basename(x).split('.')[0]
                     for x in glob.glob(os.path.join(val_img_dir, args.format_img_glob))]

        if is_main_process():
            print(f"Using separate directories:")
            print(f"  Train: {len(train_names)} images from {train_img_dir}")
            print(f"  Val:   {len(val_names)} images from {val_img_dir}")

        train_set = Dataset(args=args, names=train_names, transform=train_transform,
                            img_dir=train_img_dir, mask_dir=train_lbl_dir)
        val_set = Dataset(args=args, names=val_names, transform=val_transform,
                          img_dir=val_img_dir, mask_dir=val_lbl_dir)
    else:
        # ── Legacy mode: single directory with 80/20 split ──
        from sklearn.model_selection import train_test_split
        img_names = glob.glob(os.path.join(args.img_path, args.format_img_glob))
        base_names = [os.path.basename(x).split('.')[0] for x in img_names]
        train_names, val_names = train_test_split(base_names, train_size=0.8, random_state=42)

        if is_main_process():
            print(f"Using legacy single-directory mode with 80/20 split:")
            print(f"  Train: {len(train_names)} / Val: {len(val_names)}")

        train_set = Dataset(args=args, names=train_names, transform=train_transform)
        val_set = Dataset(args=args, names=val_names, transform=val_transform)

    # Distributed samplers (or None for single-GPU)
    train_sampler = DistributedSampler(train_set, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if args.distributed else None

    train_loader = data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # shuffle only when not using sampler
        sampler=train_sampler,
        drop_last=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False, num_workers=args.num_workers, pin_memory=True,
    )

    # TensorBoard — only rank 0 writes
    writer = None
    wandb_run = None
    args.save_dir = "./output_experiment/" + args.save_dir
    if is_main_process():
        print(args.save_dir)
        if args.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard'))
        if args.use_wandb:
            if args.wandb_mode == 'disabled':
                warnings.warn("W&B is explicitly disabled via --wandb_mode disabled.")
            if wandb is None:
                warnings.warn("W&B requested but package 'wandb' is not installed. Disabling W&B logging.")
            else:
                run_name = args.wandb_run_name if args.wandb_run_name else None
                ### ELIMINARE ###
                try:
                    wandb.login(key="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")
                except Exception as e:
                    warnings.warn(f"W&B login failed: {e}")
                ### --------- ###
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=run_name,
                    mode=args.wandb_mode,
                    config=vars(args),
                    reinit="finish_previous",
                    settings=wandb.Settings(console="wrap"),
                )
                wandb_run.define_metric("epoch")
                wandb_run.define_metric("train/*", step_metric="epoch")
                wandb_run.define_metric("val/*", step_metric="epoch")
                print(f"W&B initialized: mode={args.wandb_mode} run_id={wandb_run.id}")
                if getattr(wandb_run, "url", None):
                    print(f"W&B run url: {wandb_run.url}")

        if args.use_wandb and wandb_run is None:
            raise RuntimeError(
                "W&B logging was requested but could not be initialized. "
                "Check: 1) pip install wandb 2) wandb login 3) --wandb_mode online."
            )

    # ── Build loss function ──
    criterion = build_criterion(args.loss_type, args.num_classes, device)
    if is_main_process():
        print(f"Loss function: {args.loss_type} → {criterion.__class__.__name__}")
        if args.accum_steps > 1:
            eff_bs = args.batch_size * args.accum_steps
            if args.distributed:
                eff_bs *= dist.get_world_size()
            print(f"Gradient accumulation: {args.accum_steps} steps  "
                  f"→ effective batch size = {eff_bs}")

    best_loss = float('inf')

    for epoch in range(args.start_epoch, args.epochs):
        # set epoch for distributed sampler (ensures proper shuffling)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_metrics = train(
            train_loader,
            model,
            optimizer,
            epoch,
            args,
            writer,
            device,
            criterion,
            accum_steps=args.accum_steps,
            wandb_run=wandb_run,
        )

        val_loss, val_metrics, vis_dict = validate(val_loader, model, epoch, args, writer, device, criterion,
                            wandb_run=wandb_run)

        # ── Consolidated W&B logging (once per epoch cycle) ──
        if is_main_process() and wandb_run is not None:
            total_payload = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "val/loss": val_loss,
            }
            total_payload.update(train_metrics)
            total_payload.update(val_metrics)

            if val_loss < best_loss: # Use current cycle's best
                total_payload["val/best_loss"] = val_loss

            wandb_run.log(total_payload)
            # Update summary to specifically highlight latest metrics in the table
            wandb_run.summary.update({k: v for k, v in total_payload.items() if k != "epoch"})

            print(f"--- Epoch {epoch} Metrics Logged to W&B ---")

            # Qualitative visual logging
            should_log_images = args.wandb_log_images_every > 0 and (epoch % args.wandb_log_images_every == 0)
            if should_log_images:
                print(f"Logging qualitative images to W&B...")
                _log_wandb_validation_images(
                    wandb_run=wandb_run,
                    epoch=epoch,
                    vis_dict=vis_dict,
                    max_images=max(1, args.wandb_max_images),
                )

        scheduler.step()

        # save checkpoint (only rank 0)
        if is_main_process() and val_loss < best_loss:
            best_loss = val_loss
            filename = f'all_model_{epoch}.pth'
            print(f'save model: val_loss = {val_loss}')
            # save the underlying model state_dict (without DDP wrapper)
            model_state = model.module.state_dict() if args.distributed else model.state_dict()
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_state,
                'optimizer': optimizer.state_dict(),
            }, is_best=True, filename=filename)

            # (Best loss already included in payload above if found)

    if args.distributed:
        cleanup_distributed()

    if is_main_process() and writer is not None:
        writer.close()
    if is_main_process() and wandb_run is not None:
        wandb_run.finish()


# ──────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────

def train(train_loader, model, optimizer, epoch, args, writer, device, criterion,
          accum_steps=1, wandb_run=None):
    model.train()
    train_loss_sum = 0.0
    iou_stage1_sum = 0.0
    iou_stage2_sum = 0.0
    dsc_stage1_sum = 0.0
    dsc_stage2_sum = 0.0
    num_batches = 0
    use_ce = isinstance(criterion, nn.CrossEntropyLoss)

    optimizer.zero_grad()  # zero once at the start

    for step, (img, label) in enumerate(tqdm(train_loader,
                                              total=len(train_loader),
                                              disable=not is_main_process())):
        img = img.to(device)
        mask = label.to(device, dtype=torch.long).squeeze(dim=1)

        # ── Optionally skip DDP gradient sync on non-update steps ──
        # When using DDP, gradient all-reduce is expensive.  We can
        # defer it until the actual optimizer step by wrapping
        # non-update iterations in model.no_sync().
        is_update_step = ((step + 1) % accum_steps == 0) or \
                         ((step + 1) == len(train_loader))

        ctx = model.no_sync() if (args.distributed and not is_update_step) else \
              contextlib.nullcontext()

        with ctx:
            img_out1, img_out, prompt_embedding = model(img)
            img_out1_for_loss = _ensure_multiclass_logits(img_out1, args.num_classes)
            img_out_for_loss = _ensure_multiclass_logits(img_out, args.num_classes)

            if use_ce:
                loss = criterion(img_out_for_loss, mask) * 0.3 + criterion(img_out1_for_loss, mask) * 0.7
            else:
                mask_monai = mask.unsqueeze(1).float()
                loss = criterion(img_out_for_loss, mask_monai) * 0.3 + \
                       criterion(img_out1_for_loss, mask_monai) * 0.7

            # Scale loss so that gradient magnitude stays consistent
            # regardless of the number of accumulation steps.
            loss = loss / accum_steps
            loss.backward()

        iou_stage1_sum += iou_score(mask, img_out1_for_loss)
        iou_stage2_sum += iou_score(mask, img_out_for_loss)
        dsc_stage1_sum += dsc_score(mask, img_out1_for_loss)
        dsc_stage2_sum += dsc_score(mask, img_out_for_loss)

        # Accumulate the *unscaled* loss for logging
        train_loss_sum += loss.item() * accum_steps
        num_batches += 1

        # ── Update weights every accum_steps, or on the last batch ──
        if is_update_step:
            optimizer.step()
            optimizer.zero_grad()

        # ── Intermediate W&B logging every 50 batches (optional but helpful) ──
        if is_main_process() and wandb_run is not None and (step + 1) % 50 == 0:
            wandb_run.log({
                "train/batch_loss": loss.item() * accum_steps,
                "epoch": epoch + (step + 1) / len(train_loader)
            })

    # ── Synchronize train loss across all GPUs ──
    if args.distributed:
        train_loss_sum = reduce_tensor(train_loss_sum, device)
        num_batches = reduce_tensor(num_batches, device)
        iou_stage1_sum = reduce_tensor(iou_stage1_sum, device)
        iou_stage2_sum = reduce_tensor(iou_stage2_sum, device)
        dsc_stage1_sum = reduce_tensor(dsc_stage1_sum, device)
        dsc_stage2_sum = reduce_tensor(dsc_stage2_sum, device)

    mean_train_loss = train_loss_sum / max(num_batches, 1)
    train_iou_stage1 = iou_stage1_sum / max(num_batches, 1)
    train_iou_stage2 = iou_stage2_sum / max(num_batches, 1)
    train_dsc_stage1 = dsc_stage1_sum / max(num_batches, 1)
    train_dsc_stage2 = dsc_stage2_sum / max(num_batches, 1)
    if is_main_process():
        print(f'epoch [{epoch}]: mean_train_loss={mean_train_loss:.6f}')
        if writer is not None:
            writer.add_scalar("train_loss", mean_train_loss, global_step=epoch)

    return mean_train_loss, {
        "train/iou_stage1": train_iou_stage1,
        "train/iou_stage2": train_iou_stage2,
        "train/dsc_stage1": train_dsc_stage1,
        "train/dsc_stage2": train_dsc_stage2,
    }


# ──────────────────────────────────────────────
#  Validation loop (FIXED: no gradient updates)
# ──────────────────────────────────────────────

def validate(val_loader, model, epoch, args, writer, device, criterion, wandb_run=None):
    if is_main_process():
        print('VALIDATE')

    model.eval()  # ← BatchNorm uses running stats, Dropout disabled
    use_ce = isinstance(criterion, nn.CrossEntropyLoss)

    val_loss_sum = 0.0
    iou_stage1_sum = 0.0   # IoU for img_out1 (first decoder)
    iou_stage2_sum = 0.0   # IoU for img_out  (refined decoder)
    dsc_stage1_sum = 0.0   # DSC for img_out1
    dsc_stage2_sum = 0.0   # DSC for img_out
    num_batches = 0
    vis_dict = None

    with torch.no_grad():  # ← no gradient computation during validation
        for img, label in tqdm(val_loader, total=len(val_loader),
                               disable=not is_main_process()):
            img = img.to(device)
            mask = label.to(device, dtype=torch.long).squeeze(dim=1)

            img_out1, img_out, prompt_embedding = model(img)
            img_out1_for_loss = _ensure_multiclass_logits(img_out1, args.num_classes)
            img_out_for_loss = _ensure_multiclass_logits(img_out, args.num_classes)

            if use_ce:
                loss = criterion(img_out_for_loss, mask) * 0.3 + criterion(img_out1_for_loss, mask) * 0.7
            else:
                mask_monai = mask.unsqueeze(1).float()
                loss = criterion(img_out_for_loss, mask_monai) * 0.3 + criterion(img_out1_for_loss, mask_monai) * 0.7

            val_loss_sum += loss.item()

            # Separate metrics for each decoder stage
            iou_stage1_sum += iou_score(mask, img_out1_for_loss)
            iou_stage2_sum += iou_score(mask, img_out_for_loss)
            dsc_stage1_sum += dsc_score(mask, img_out1_for_loss)
            dsc_stage2_sum += dsc_score(mask, img_out_for_loss)
            num_batches += 1

            if vis_dict is None and is_main_process() and args.wandb_log_images_every > 0:
                max_n = max(1, args.wandb_max_images)
                vis_dict = {
                    'img': img[:max_n].detach().cpu(),
                    'mask': mask[:max_n].detach().cpu(),
                    'out1': img_out1[:max_n].detach().cpu(),
                    'out2': img_out[:max_n].detach().cpu(),
                }

    # ── Synchronize validation metrics across all GPUs ──
    if args.distributed:
        val_loss_sum = reduce_tensor(val_loss_sum, device)
        iou_stage1_sum = reduce_tensor(iou_stage1_sum, device)
        iou_stage2_sum = reduce_tensor(iou_stage2_sum, device)
        dsc_stage1_sum = reduce_tensor(dsc_stage1_sum, device)
        dsc_stage2_sum = reduce_tensor(dsc_stage2_sum, device)
        num_batches = reduce_tensor(num_batches, device)

    # Average over total batches across all GPUs
    num_batches = max(num_batches, 1)
    val_loss = val_loss_sum / num_batches
    iou_stage1 = iou_stage1_sum / num_batches
    iou_stage2 = iou_stage2_sum / num_batches
    dsc_stage1 = dsc_stage1_sum / num_batches
    dsc_stage2 = dsc_stage2_sum / num_batches

    if is_main_process():
        print(f'epoch[{epoch}]: val_loss     = {val_loss:.6f}')
        print(f'epoch[{epoch}]: val_iou_stage1 (img_out1) = {iou_stage1:.4f}')
        print(f'epoch[{epoch}]: val_iou_stage2 (img_out)  = {iou_stage2:.4f}')
        print(f'epoch[{epoch}]: val_dsc_stage1 (img_out1) = {dsc_stage1:.4f}')
        print(f'epoch[{epoch}]: val_dsc_stage2 (img_out)  = {dsc_stage2:.4f}')

        if writer is not None:
            writer.add_scalar("val_loss", val_loss, global_step=epoch)
            writer.add_scalar("val_iou_stage1", iou_stage1, global_step=epoch)
            writer.add_scalar("val_iou_stage2", iou_stage2, global_step=epoch)
            writer.add_scalar("val_dsc_stage1", dsc_stage1, global_step=epoch)
            writer.add_scalar("val_dsc_stage2", dsc_stage2, global_step=epoch)

    return val_loss, {
        "val/iou_stage1": iou_stage1,
        "val/iou_stage2": iou_stage2,
        "val/dsc_stage1": dsc_stage1,
        "val/dsc_stage2": dsc_stage2,
    }, vis_dict


# ──────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'all_model_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

if __name__ == '__main__':

    # Flush stdout/stderr immediately (important for torchrun / Kaggle)
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    args = parser.parse_args()

    if not args.distributed and args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                       'disable data parallelism.')

    try:
        main_worker(args)
    except Exception as e:
        print(f"\n[RANK {os.environ.get('LOCAL_RANK', '?')}] ERROR: {e}",
              file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        raise