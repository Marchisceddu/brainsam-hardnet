import argparse
import glob
import os

import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from tqdm import tqdm

from models import sam_feat_seg_model_registry
from test_dataset import Dataset

class Config(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = ''
    img_out_path = ''
    model_path = ''
    img_type = 'png'
    model_type = 'vit_b'
    img_size = 1024
    num_classes = 2
    batch_size = 1

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Brain-SAM inference')
    p.add_argument('--img_path', type=str, required=True, help='Directory with input 2D images')
    p.add_argument('--model_path', type=str, required=True, help='Path to trained model weights (.pth)')
    p.add_argument('--img_out_path', type=str, required=True, help='Output directory for predicted masks')
    p.add_argument('--img_type', type=str, default='png', help='Input image extension without dot (e.g., png, tif)')
    p.add_argument('--model_type', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'], help='Backbone type')
    p.add_argument('--img_size', type=int, default=1024, help='Resize size for model input')
    p.add_argument('--num_classes', type=int, default=2, help='Number of classes (including background)')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size (suggest 1-2 on 16GB VRAM)')
    return p


def test(configs: Config):
    suffix = '*.' + configs.img_type
    img_paths = glob.glob(os.path.join(configs.img_path, suffix))
    if not img_paths:
        raise RuntimeError(f"No images found in {configs.img_path} with pattern {suffix}")
    img_names = [os.path.basename(p) for p in img_paths]

    test_transform = albu.Compose(
        [
            albu.Resize(configs.img_size, configs.img_size),
            albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
        is_check_shapes=False,
    )
    test_set = Dataset(configs=configs, names=img_names, transform=test_transform)
    test_loader = data.DataLoader(test_set, batch_size=configs.batch_size, shuffle=False, drop_last=False)

    model = sam_feat_seg_model_registry[configs.model_type](num_classes=configs.num_classes, checkpoint=configs.model_path, img_size=configs.img_size, iter_2stage=1)

    model.to(configs.device)

    model.eval()

    os.makedirs(configs.img_out_path, exist_ok=True)

    with torch.no_grad():
        for img, name in tqdm(test_loader, total=len(test_loader)):
            img = img.to(configs.device)
            img_out_initial, img_out_end, prompt_embedding = model(img)

            pred_intial = img_out_initial.argmax(dim=1).unsqueeze(dim=1)
            pred_end = img_out_end.argmax(dim=1).unsqueeze(dim=1)
            for i in range(img.size(0)):

                img_end = pred_end.detach()[i].cpu().numpy()
                img_end = Image.fromarray(img_end.astype(np.uint8).squeeze(axis=0), mode='L')
                img_end = img_end.resize((configs.img_size, configs.img_size), Image.NEAREST)
                img_end.save(os.path.join(configs.img_out_path, name['img_id'][i]))


if __name__ == '__main__':
    args = build_argparser().parse_args()
    cfg = Config()
    cfg.img_path = args.img_path
    cfg.model_path = args.model_path
    cfg.img_out_path = args.img_out_path
    cfg.img_type = args.img_type
    cfg.model_type = args.model_type
    cfg.img_size = args.img_size
    cfg.num_classes = args.num_classes
    cfg.batch_size = args.batch_size

    test(cfg)

    