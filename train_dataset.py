import os
import re
from typing import Tuple
import numpy as np
import torch
import torch.utils.data
from PIL import Image


VOLUME_FILENAME_RE = re.compile(r"^(P\d+)_(T\d+)_(\d+)\.[^.]+$")


def parse_volume_info(filename: str) -> Tuple[str, str, int]:
    """Extract (patient, timepoint, slice_idx) from filename pattern P<id>_T<id>_<slice>.<ext>."""
    base = os.path.basename(filename)
    m = VOLUME_FILENAME_RE.match(base)
    if m is None:
        # Fallback: return None to indicate unparseable (for legacy datasets)
        return None, None, None
    patient, timepoint, slice_idx = m.group(1), m.group(2), int(m.group(3))
    return patient, timepoint, slice_idx


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, names, transform=None, img_dir=None, mask_dir=None):
        """
        Args:
            args: Configuration object (used for format_img and as fallback for paths).
            names: List of image base names (without extension).
            transform: Albumentations transform pipeline.
            img_dir: Override for image directory. If None, uses args.img_path.
            mask_dir: Override for mask directory. If None, uses args.label_path.
        """
        self.config = args
        self.img_ids = names
        self.img_dir = img_dir if img_dir is not None else args.img_path
        self.mask_dir = mask_dir if mask_dir is not None else args.label_path
        self.num_classes = 1
        self.transform = transform
        self.format_img = args.format_img
        self.input_channels = int(getattr(args, 'input_channels', 3))
    
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id+self.format_img)
        
        # Le maschere sono ora sempre salvate originariamente in png dal preprocessing
        mask_ext = '.png'
        
        # Lettura immagine supportando .tif senza causare troncamenti e cast forzati
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            if img_path.endswith('.tif'):
                import tifffile
                img = tifffile.imread(img_path)
            else:
                img = np.array(Image.open(img_path))
                
        if len(img.shape) == 2:
            img = img[..., None]

        if len(img.shape) == 3 and img.shape[2] == 3 and img.dtype == np.uint8:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_channels == 1:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)[..., None]
            elif img.shape[2] != 1:
                img = img.mean(axis=2, keepdims=True)
        elif self.input_channels == 3:
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
        else:
            raise ValueError(f"Unsupported input_channels={self.input_channels}. Use 1 or 3.")
            
        img = img.astype(np.float32)

        # Rescale raw float TIF values to [0, 255] for the augmentation pipeline.
        # PNG uint8 images are already in [0, 255] and won't be affected.
        v_min, v_max = img.min(), img.max()
        if v_max > 255.0 or v_min < 0.0:
            if v_max - v_min > 1e-8:
                img = (img - v_min) / (v_max - v_min) * 255.0
            else:
                img = np.zeros_like(img, dtype=np.float32)

        mask = []
        for i in range(self.num_classes):
            mask_path = os.path.join(self.mask_dir, img_id+mask_ext)
            temp_mask = np.array(Image.open(mask_path).convert('L'))
            mask.append(temp_mask)
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32')
        img = np.transpose(img,(2,0,1))
        # mask = mask/255

        mask = np.transpose(mask,(2,0,1))
        mask = (mask > 127).astype('float32')  # Binarize: {0,255} → {0,1}

        # Parse volume metadata if available (for 3D validation)
        patient, timepoint, slice_idx = parse_volume_info(img_id + self.format_img)

        return {
            'img': img,
            'mask': mask,
            'img_id': img_id,
            'patient': patient,
            'timepoint': timepoint,
            'slice_idx': slice_idx,
        }
