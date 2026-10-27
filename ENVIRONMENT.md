# Experimental Environment

## 1. WHU Formal Training and Evaluation

- **Platform:** Kaggle
- **Hardware:** NVIDIA Tesla T4, 15 GB VRAM
- **Python:** 3.10–3.12 across different runs
- **PyTorch:** 2.x with CUDA 12.x; exact versions vary by run and are not yet fully archived
- **Dataset path in the original environment:** `/kaggle/input/datasets/xiaomaoperson/whu-data/whu_data`
- **Dataset split sizes:** 4,736 training images, 1,036 validation images, and 2,416 test images
- **Input size:** 256 × 256 pixels
- **Image processing:** PIL image loading, bilinear interpolation for images, nearest-neighbor interpolation for labels, followed by `ToTensor`. No mean–standard-deviation normalization was applied.
- **Training configuration:** Batch size 8, 50 epochs, Adam optimizer, learning rate 1e-3
- **Data augmentation and post-processing:** None
- **Model selection:** Best checkpoint selected using validation Building IoU
- **Model initialization: ImageNet-pretrained weights were enabled in the displayed notebook configuration (USE_IMAGENET_WEIGHTS=True).

A local verification environment used Python 3.9.25 and PyTorch 2.8.0+cpu. The recorded verification values matched the corresponding Kaggle results at the reported precision.

## 2. Inria, X2-region, and X2-shallow Experiments

- **Platform:** Kaggle
- **Python:** 3.12.13
- **PyTorch:** 2.10.0+cu128
- **CUDA runtime:** 12.8
- **GPU:** NVIDIA Tesla T4, 15 GB VRAM
- **Driver:** 580.159.04; the driver reports CUDA 13.0 as its maximum supported CUDA version
- **Inria data:** 180 tiles, split into 150 training tiles and 30 validation tiles
- **Patch generation:** 512 × 512 windows, stride 1536, and 9 patches per tile, producing 1,620 patches
- **Image processing:** The X1, X2, and X2-shallow implementations used OpenCV-based image loading consistently within this experimental group.

## 3. Additional Evaluation Inputs

The original Kaggle environment used the following mounted inputs:

- WHU three-split dataset: `/kaggle/input/datasets/xiaomaoperson/whu-data/whu_data`
- CBRNet weights: mounted under the Kaggle input directory identified as `cbrnet2`
- Nine-checkpoint evaluation weights: mounted under the Kaggle input directory identified as `SeedTest`
- Inria data: mounted under a Kaggle input directory

These are environment-specific paths and are not portable download links. Users must configure the corresponding dataset and checkpoint paths in their own environment.

## 4. Dependency Version Status

The recorded September 2026 environment includes PyTorch 2.10.0+cu128 and a compatible torchvision installation. Other identified dependencies include NumPy, OpenCV, Pillow, and THOP for model complexity measurement.

A complete `pip freeze` export from each original Kaggle notebook has not yet been archived. Therefore, the exact package versions for all formal training and evaluation runs remain to be verified. The dependency list should not be interpreted as a fully tested, version-pinned environment specification.

## 5. Reproducibility Notes

The experiments were conducted across multiple cloud and local environments. Python and PyTorch versions were not identical across all runs.

The original Kaggle input paths must be adapted before execution. Exact environment reproduction requires the corresponding source code, configuration, dataset split, checkpoint, and dependency versions for each experiment.

This document records the currently available environment information; it does not claim that every experiment can be reproduced from a clean environment without further verification.
