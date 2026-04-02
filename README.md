# Brain-SAM: A General Automatic SAM-based Segmentation Model for Brain Science Images

## Introduction
Brain-SAM is a general automatic segmentation model based on the Segment Anything Model (SAM) designed for the segmentation and analysis of microscopic optical images in brain science. We introduce an Automatic Prompt Encoder to enable high-throughput automated segmentation and a Segmentation Optimizer to enhance segmentation performance. This repository contains the code and datasets used in the study.

## Features
- **Automatic Prompt Encoder**: Automatically acquires positional encoding of segmentation targets for high-throughput segmentation.
- **Segmentation Optimizer**: Refines segmentation results by integrating embedding information from the SAM image encoder.
- **High Performance**: Achieves superior segmentation results on various brain science image datasets compared to specialized models.
- **Public Datasets**: Provides a series of rich publicly available brain science image datasets created using fluorescence microscopic optical tomography (fMOST) technology.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Brainsmatics/Brain-SAM.git
2. Install the required dependencies (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
## Usage

### MSLesSeg Preprocessing (Subset)
Use this script when you want a restricted subset while preserving the original preprocessing logic.

```bash
python preprocess_nifti_subset.py \
  --input "MSLesSeg Dataset" \
  --output out_subset \
  --n_train 10 \
  --n_val 5 \
  --slice_stride 4 \
  --non_lesion_ratio 0.15
```

Default behavior of `preprocess_nifti_subset.py`:
- Processes `train` and `val` only (`test` is skipped unless `--include_test` is used).
- Uses natural patient order (`P1, P2, ..., P10`).
- Uses only the last timepoint per patient (use `--all_timepoints` to disable).
- Keeps one slice every 4 (`--slice_stride 4`).
- Preserves the same output layout (`split/images`, `split/labels`) and mask format (`.png`).

### Training
To train the Brain-SAM model, use the following command:

```bash
python train_use_gpu.py \
  --img_path /path/to/images \
  --label_path /path/to/labels \
  --model_checkpoint checkpoints/sam_vit_l.pth \
  --save_dir experiment_1 \
  --batch_size 4 \
  --epochs 30
```
### Testing
To run inference on your test images, use the following command:
```bash
python inference.py \
  --img_path /path/to/test_images \
  --model_path checkpoints/all_model_best.pth \
  --img_out_path results/
```
## Datasets
The following datasets are used in this study:
- A𝛽 Dataset: A𝛽 Plaque, contains images with diffuse boundary structures.
- Louble Dataset: Liver Sinusoidal, features fuzzy boundaries and complex targets.
- Soma Dataset: Cell Body, dense and strongly adhering targets.
- Brain Dataset: Mouse Brain Contour, Segmentation contours for mouse brain.
- Neuron Dataset: Linear neuronal structures.
- Lectin2d, Lectin3d, Tek Datasets: Tubular vascular data and mesh-like structures.

These datasets and pretrained checkpoint model are publicly available and can be downloaded from [google drive](https://drive.google.com/drive/folders/1MnWVS8i4pzO781JeqMkdtT8X1q9-GQ9B?usp=drive_link).

## Results
The experimental results demonstrate that Brain-SAM outperforms several state-of-the-art segmentation models, including U-Net, Deeplab v3+, ResUNet, UNet++, and AutoSAM. Key metrics used for evaluation include IoU, Dice, F1-score, and Accuracy. Detailed results are provided in the Experiments section.

## Contact
For any questions or issues, please contact the corresponding authors:
- Chi Xiao: xiaochi@hainanu.edu.cn
- Zhao Feng: fengzhao@hainnau.edu.cn





