# -*- coding: utf-8 -*-
"""
CBRNet unified paper-metric re-evaluation script for Kaggle
============================================================

Purpose
-------
Load an already trained CBRNet_best.pth and re-evaluate WHU validation/test
sets using EXACTLY the paper's aggregation convention:

Regional metrics:
    - accumulate TP / FP / FN / TN over the whole dataset
    - then compute Building IoU, two-class mIoU, Precision, Recall, F1

Boundary metrics:
    - for EACH 256x256 image separately:
        1) boundary = dilate(mask, 3x3) - erode(mask, 3x3)
        2) Boundary F1 uses tolerance=2 pixels (5x5 dilation)
        3) Boundary IoU is strict, with no tolerance
    - finally take np.mean over image-wise Boundary F1 / Boundary IoU

This is intentionally different from the previous CBRNet "pooled boundary
pixel" aggregation. It is designed to match the paper's X0/X1/X2 and baseline
evaluation convention.

What you need to add to Kaggle
------------------------------
1. The SAME WHU dataset used by the paper:
       train/image, train/label
       val/image,   val/label
       test/image,  test/label
   Expected sample counts: 4736 / 1036 / 2416.

2. CBRNet_best.pth as a Kaggle Input.
   The script auto-searches /kaggle/input/**/CBRNet_best.pth.

No training is performed. No VGG ImageNet download is needed.

Outputs
-------
/kaggle/working/CBRNet_paper_metric_reeval/
    CBRNet_val_paper_metrics.json
    CBRNet_test_paper_metrics.json
    CBRNet_val_per_image_boundary.csv
    CBRNet_test_per_image_boundary.csv
    CBRNet_paper_metric_reeval_report.txt
"""

import os
import csv
import json
import glob
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 0. Configuration
# ============================================================
CFG = {
    "IMG_SIZE": 256,
    "BATCH_SIZE": 8,
    "NUM_WORKERS": 2,
    "EXPECTED_COUNTS": (4736, 1036, 2416),
    "OUT_DIR": "/kaggle/working/CBRNet_paper_metric_reeval",

    # Safety switch:
    # If more than one CBRNet_best.pth is mounted, stop instead of guessing.
    "FAIL_IF_MULTIPLE_CHECKPOINTS": True,
}

os.makedirs(CFG["OUT_DIR"], exist_ok=True)

K3 = np.ones((3, 3), np.uint8)


# ============================================================
# 1. Find WHU dataset and checkpoint
# ============================================================
def find_whu_root():
    """
    Find a directory containing:
        train/image, train/label,
        val/image,   val/label,
        test/image,  test/label
    """
    candidates = []
    patterns = [
        "/kaggle/input/**/whu_data/train/image",
        "/kaggle/input/**/WHU_BUILDING/train/image",
        "/kaggle/input/**/train/image",
    ]

    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))

    valid = []
    for train_img in candidates:
        root = os.path.dirname(os.path.dirname(train_img))
        needed = [
            os.path.join(root, "train", "image"),
            os.path.join(root, "train", "label"),
            os.path.join(root, "val", "image"),
            os.path.join(root, "val", "label"),
            os.path.join(root, "test", "image"),
            os.path.join(root, "test", "label"),
        ]
        if all(os.path.isdir(p) for p in needed):
            valid.append(root)

    valid = sorted(
        set(valid),
        key=lambda p: ("whu" not in p.lower(), len(p), p)
    )

    if not valid:
        raise FileNotFoundError(
            "Cannot find WHU dataset under /kaggle/input. "
            "Please add the same WHU dataset used in the paper."
        )

    # Prefer an exact sample-count match if multiple WHU-like roots exist.
    exact = []
    for root in valid:
        try:
            counts = tuple(
                len(build_pairs(
                    os.path.join(root, split, "image"),
                    os.path.join(root, split, "label")
                ))
                for split in ("train", "val", "test")
            )
            if counts == CFG["EXPECTED_COUNTS"]:
                exact.append(root)
        except Exception:
            pass

    if len(exact) == 1:
        return exact[0]

    if len(exact) > 1:
        print("[WARN] Multiple WHU roots have the expected 4736/1036/2416 split:")
        for p in exact:
            print("       ", p)
        print("[WARN] Using the first exact-match root:", exact[0])
        return exact[0]

    print("[WARN] No dataset root exactly matches 4736/1036/2416.")
    print("[WARN] Candidate roots:")
    for p in valid:
        print("       ", p)
    print("[WARN] Using:", valid[0])
    return valid[0]


def find_best_checkpoint():
    candidates = sorted(set(
        glob.glob("/kaggle/input/**/CBRNet_best.pth", recursive=True)
    ))

    if not candidates:
        # Also allow a manually copied checkpoint in /kaggle/working.
        candidates = sorted(set(
            glob.glob("/kaggle/working/**/CBRNet_best.pth", recursive=True)
        ))

    if not candidates:
        raise FileNotFoundError(
            "CBRNet_best.pth was not found. "
            "Please add it as a Kaggle Input."
        )

    if len(candidates) > 1 and CFG["FAIL_IF_MULTIPLE_CHECKPOINTS"]:
        msg = [
            "More than one CBRNet_best.pth was found.",
            "To prevent loading the wrong experiment, keep only the intended "
            "checkpoint mounted or set FAIL_IF_MULTIPLE_CHECKPOINTS=False.",
            "Candidates:"
        ] + [f"  - {p}" for p in candidates]
        raise RuntimeError("\n".join(msg))

    ckpt_path = candidates[0]
    print("[CHECKPOINT]", ckpt_path)
    return ckpt_path


def build_pairs(img_dir, label_dir):
    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

    imgs = {
        Path(p).stem: p
        for p in glob.glob(os.path.join(img_dir, "*"))
        if Path(p).suffix.lower() in exts
    }
    labs = {
        Path(p).stem: p
        for p in glob.glob(os.path.join(label_dir, "*"))
        if Path(p).suffix.lower() in exts
    }

    keys = sorted(set(imgs) & set(labs))
    if not keys:
        raise RuntimeError(
            f"No paired image/label files found:\n"
            f"  image: {img_dir}\n"
            f"  label: {label_dir}"
        )

    return [(imgs[k], labs[k], k) for k in keys]


# ============================================================
# 2. Dataset — ToTensor only, no additional normalization
# ============================================================
def load_binary_mask(path, size=256):
    mask = Image.open(path).convert("L").resize(
        (size, size),
        Image.Resampling.NEAREST
    )
    arr = np.asarray(mask)
    return (arr > 0).astype(np.uint8)


class WHUEvalDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lab_path, stem = self.pairs[idx]

        img = Image.open(img_path).convert("RGB").resize(
            (CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
            Image.Resampling.BILINEAR
        )
        mask_np = load_binary_mask(
            lab_path,
            CFG["IMG_SIZE"]
        )

        # Same preprocessing as the latest unified CBRNet run:
        # ToTensor only; no mean/std normalization.
        img_t = TF.to_tensor(img).float()
        mask_t = torch.from_numpy(mask_np.astype(np.uint8))

        return {
            "image": img_t,
            "mask": mask_t,
            "name": stem,
        }


# ============================================================
# 3. CBRNet architecture
#    IMPORTANT: weights=None because the full trained checkpoint
#    will overwrite all model parameters. No Internet/download needed.
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=True
            )
            self.conv = DoubleConv(
                in_channels,
                out_channels,
                in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                2,
                stride=2
            )
            self.conv = DoubleConv(
                in_channels,
                out_channels
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        return self.conv(
            torch.cat([x2, x1], dim=1)
        )


class FixSeg(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(
            1, 8, 3, 1, 1,
            bias=False
        )

        w = torch.tensor(
            [
                [[[0, 0, 0], [1, 0, 0], [0, 0, 0]]],
                [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]],
                [[[0, 1, 0], [0, 0, 0], [0, 0, 0]]],
                [[[0, 0, 1], [0, 0, 0], [0, 0, 0]]],
                [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
                [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                [[[0, 0, 0], [0, 0, 0], [0, 1, 0]]],
                [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]],
            ],
            dtype=torch.float32,
        )

        self.conv0.weight = nn.Parameter(w)

    def forward(
        self,
        direc_pred,
        masks_pred,
        edge_pred
    ):
        direc_pred = direc_pred.softmax(1)

        edge_mask = (
            torch.sigmoid(edge_pred).detach()
            > 0.5
        ).float()

        refined = (
            (self.conv0(masks_pred) * direc_pred)
            .sum(1)
            .unsqueeze(1)
            * edge_mask
            + masks_pred * (1 - edge_mask)
        )

        return refined


class CBRNet(nn.Module):
    def __init__(self):
        super().__init__()

        bilinear = True
        state = [64, 128, 256, 512, 1024]

        # Architecture only. We deliberately do NOT download ImageNet weights.
        # The trained CBRNet_best.pth is loaded immediately afterwards.
        vgg = models.vgg16_bn(weights=None)

        self.inc = vgg.features[:5]
        self.down1 = vgg.features[5:12]
        self.down2 = vgg.features[12:22]
        self.down3 = vgg.features[22:32]
        self.down4 = vgg.features[32:42]

        del vgg

        factor = 2 if bilinear else 1

        self.classifier1 = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        self.up1 = Up(
            state[4],
            state[3] // factor,
            bilinear
        )

        self.classifier2 = nn.Sequential(
            nn.Conv2d(256 + 1, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        self.classifier2_2 = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        self.up2 = Up(
            state[3],
            state[2] // factor,
            bilinear
        )

        self.classifier3 = nn.Sequential(
            nn.Conv2d(128 + 1, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        self.classifier3_2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        self.up3 = Up(
            state[2],
            state[1] // factor,
            bilinear
        )

        self.classifier4 = nn.Sequential(
            nn.Conv2d(64 + 1, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        self.classifier4_2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        self.up4 = Up(
            state[1],
            state[0],
            bilinear
        )

        self.classifier5 = nn.Sequential(
            nn.Conv2d(64 + 1, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        self.classifier5_2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )

        self.dir_head = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, 1),
        )

        self.fixer = FixSeg()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        seg5 = self.classifier1(x5)

        x = self.up1(x5, x4)
        edge4 = self.classifier2_2(x)
        seg4 = self.classifier2(
            torch.cat(
                (
                    x,
                    F.interpolate(
                        seg5,
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ),
                1,
            )
        )

        x = self.up2(x, x3)
        edge3 = self.classifier3_2(x)
        seg3 = self.classifier3(
            torch.cat(
                (
                    x,
                    F.interpolate(
                        seg4,
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ),
                1,
            )
        )

        x = self.up3(x, x2)
        edge2 = self.classifier4_2(x)
        seg2 = self.classifier4(
            torch.cat(
                (
                    x,
                    F.interpolate(
                        seg3,
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ),
                1,
            )
        )

        x = self.up4(x, x1)
        edge1 = self.classifier5_2(x)
        seg1 = self.classifier5(
            torch.cat(
                (
                    x,
                    F.interpolate(
                        seg2,
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ),
                1,
            )
        )

        direction = self.dir_head(x)
        refined = self.fixer(
            direction,
            seg1,
            edge1
        )

        return (
            refined,
            seg1,
            seg2,
            seg3,
            seg4,
            seg5,
            edge1,
            edge2,
            edge3,
            edge4,
            direction,
        )


# ============================================================
# 4. EXACT paper boundary metrics
#    Copied in logic from the paper's WHU X0/X1/X2 evaluation.
# ============================================================
def get_boundary(mask):
    """
    Paper convention:
      binary mask -> uint8*255
      3x3 dilation and erosion
      boundary = (dilated - eroded) > 0
    """
    mask_u8 = mask.astype(np.uint8) * 255

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    dilated = cv2.dilate(
        mask_u8,
        kernel,
        iterations=1
    )

    eroded = cv2.erode(
        mask_u8,
        kernel,
        iterations=1
    )

    return (
        (dilated - eroded) > 0
    ).astype(np.uint8)


def boundary_metrics(
    gt,
    pred,
    tolerance=2
):
    """
    EXACT paper convention.

    Important empty-boundary behavior:
    denominators use +1e-8, so if GT and prediction both have
    no boundary, Boundary F1 and Boundary IoU are 0 for that image.
    This is intentionally preserved to match the paper tables.
    """
    gt_boundary = get_boundary(gt)
    pred_boundary = get_boundary(pred)

    intersection = np.logical_and(
        gt_boundary,
        pred_boundary
    ).sum()

    union = np.logical_or(
        gt_boundary,
        pred_boundary
    ).sum()

    # Strict Boundary IoU: no tolerance.
    boundary_iou = (
        intersection / (union + 1e-8)
    )

    kernel = np.ones(
        (
            2 * tolerance + 1,
            2 * tolerance + 1,
        ),
        np.uint8
    )

    gt_dilated = cv2.dilate(
        gt_boundary,
        kernel
    )

    pred_dilated = cv2.dilate(
        pred_boundary,
        kernel
    )

    matched_pred = np.logical_and(
        pred_boundary == 1,
        gt_dilated == 1
    ).sum()

    matched_gt = np.logical_and(
        gt_boundary == 1,
        pred_dilated == 1
    ).sum()

    precision = matched_pred / (
        (pred_boundary == 1).sum()
        + 1e-8
    )

    recall = matched_gt / (
        (gt_boundary == 1).sum()
        + 1e-8
    )

    boundary_f1 = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    return (
        float(boundary_f1),
        float(boundary_iou),
        int(gt_boundary.sum()),
        int(pred_boundary.sum()),
    )


# ============================================================
# 5. Paper-consistent evaluation
# ============================================================
@torch.no_grad()
def evaluate_paper_metrics(
    model,
    loader,
    device,
    split_name
):
    model.eval()

    # Whole-dataset confusion counts for regional metrics.
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    # Per-image lists for boundary metrics.
    boundary_f1_list = []
    boundary_iou_list = []
    per_image_rows = []

    num_gt_empty_boundary = 0
    num_pred_empty_boundary = 0
    num_both_empty_boundary = 0

    seen = 0

    for batch_idx, batch in enumerate(loader, 1):
        images = batch["image"].to(
            device,
            non_blocking=True
        )

        gt = (
            batch["mask"]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        names = list(batch["name"])

        # Final CBRNet output used by the original experiment.
        refined = model(images)[0]

        # refined is a logit-like output; original CBRNet evaluation
        # used threshold > 0, equivalent to sigmoid > 0.5.
        pred = (
            refined.squeeze(1) > 0
        ).cpu().numpy().astype(np.uint8)

        # Regional metrics: aggregate pixels over entire split.
        tp += np.logical_and(
            pred == 1,
            gt == 1
        ).sum()

        fp += np.logical_and(
            pred == 1,
            gt == 0
        ).sum()

        fn += np.logical_and(
            pred == 0,
            gt == 1
        ).sum()

        tn += np.logical_and(
            pred == 0,
            gt == 0
        ).sum()

        # Boundary metrics: compute EACH IMAGE first, then mean.
        for name, p, g in zip(
            names,
            pred,
            gt
        ):
            bf1, biou, gt_b_px, pred_b_px = boundary_metrics(
                g,
                p,
                tolerance=2
            )

            boundary_f1_list.append(bf1)
            boundary_iou_list.append(biou)

            gt_empty = (gt_b_px == 0)
            pred_empty = (pred_b_px == 0)

            num_gt_empty_boundary += int(gt_empty)
            num_pred_empty_boundary += int(pred_empty)
            num_both_empty_boundary += int(
                gt_empty and pred_empty
            )

            per_image_rows.append(
                {
                    "name": name,
                    "BoundaryF1": bf1 * 100.0,
                    "BoundaryIoU": biou * 100.0,
                    "GTBoundaryPixels": gt_b_px,
                    "PredBoundaryPixels": pred_b_px,
                    "GTBoundaryEmpty": int(gt_empty),
                    "PredBoundaryEmpty": int(pred_empty),
                }
            )

        seen += len(names)

        if (
            batch_idx % 50 == 0
            or batch_idx == len(loader)
        ):
            print(
                f"[{split_name}] "
                f"{seen}/{len(loader.dataset)} images"
            )

    # Same formulas as the paper.
    building_iou = tp / (
        tp + fp + fn + 1e-8
    )

    background_iou = tn / (
        tn + fp + fn + 1e-8
    )

    precision = tp / (
        tp + fp + 1e-8
    )

    recall = tp / (
        tp + fn + 1e-8
    )

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    metrics = {
        "BuildingIoU": float(building_iou),
        "True_mIoU": float(
            (building_iou + background_iou)
            / 2
        ),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "BoundaryF1": float(
            np.mean(boundary_f1_list)
        ),
        "BoundaryIoU": float(
            np.mean(boundary_iou_list)
        ),
    }

    diagnostics = {
        "num_images": int(seen),
        "num_gt_empty_boundary": int(
            num_gt_empty_boundary
        ),
        "num_pred_empty_boundary": int(
            num_pred_empty_boundary
        ),
        "num_both_empty_boundary": int(
            num_both_empty_boundary
        ),
        "gt_empty_boundary_rate": float(
            num_gt_empty_boundary
            / max(seen, 1)
        ),
        "pred_empty_boundary_rate": float(
            num_pred_empty_boundary
            / max(seen, 1)
        ),
        "both_empty_boundary_rate": float(
            num_both_empty_boundary
            / max(seen, 1)
        ),
    }

    return metrics, diagnostics, per_image_rows


# ============================================================
# 6. I/O helpers
# ============================================================
def safe_torch_load(path, map_location):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(
            path,
            map_location=map_location
        )


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # Allow a raw state_dict checkpoint.
        state = ckpt
    else:
        raise TypeError(
            "Unsupported checkpoint format."
        )

    # Strip possible DataParallel prefix.
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    return cleaned


def print_checkpoint_metadata(ckpt):
    if not isinstance(ckpt, dict):
        print(
            "[WARN] Raw state_dict checkpoint; "
            "no metadata available."
        )
        return

    print("\n[CHECKPOINT METADATA]")

    if "epoch" in ckpt:
        print("  epoch:", ckpt["epoch"])

    cfg = ckpt.get("cfg")
    if isinstance(cfg, dict):
        print(
            "  experiment:",
            cfg.get("EXPERIMENT", "<missing>")
        )
        print(
            "  preprocessing:",
            cfg.get("PREPROCESSING", "<missing>")
        )

        prep = str(
            cfg.get("PREPROCESSING", "")
        ).lower()

        if prep and "totensor" not in prep:
            print(
                "  [WARNING] Checkpoint metadata does not "
                "look like the ToTensor-only experiment."
            )

    if "val_metrics" in ckpt:
        vm = ckpt["val_metrics"]
        if isinstance(vm, dict):
            print(
                "  saved val BuildingIoU:",
                vm.get("BuildingIoU", "<missing>")
            )
            print(
                "  saved val BoundaryF1 "
                "(old CBRNet aggregation may differ):",
                vm.get("BoundaryF1", "<missing>")
            )
            print(
                "  saved val BoundaryIoU "
                "(old CBRNet aggregation may differ):",
                vm.get("BoundaryIoU", "<missing>")
            )


def save_metrics_json(
    path,
    metrics,
    diagnostics
):
    payload = {
        "aggregation": {
            "regional_metrics":
                "global TP/FP/FN/TN over entire split",
            "boundary_metrics":
                "per-image BoundaryF1/BoundaryIoU then arithmetic mean",
            "boundary_kernel":
                "3x3 dilation minus erosion",
            "boundary_f1_tolerance_pixels": 2,
            "strict_boundary_iou_tolerance_pixels": 0,
            "empty_boundary_behavior":
                "preserved from paper code: +1e-8 denominators, "
                "therefore both-empty image contributes 0",
        },
        "metrics_percent": {
            k: float(v * 100.0)
            for k, v in metrics.items()
        },
        "diagnostics": diagnostics,
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2
        )


def save_per_image_csv(
    path,
    rows
):
    fields = [
        "name",
        "BoundaryF1",
        "BoundaryIoU",
        "GTBoundaryPixels",
        "PredBoundaryPixels",
        "GTBoundaryEmpty",
        "PredBoundaryEmpty",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


def format_metrics(metrics):
    order = [
        "BuildingIoU",
        "True_mIoU",
        "Precision",
        "Recall",
        "F1",
        "BoundaryF1",
        "BoundaryIoU",
    ]

    return "\n".join(
        f"  {k}: {metrics[k] * 100:.4f}%"
        for k in order
    )


def compare_with_saved_val(
    ckpt,
    new_val_metrics
):
    if not (
        isinstance(ckpt, dict)
        and isinstance(
            ckpt.get("val_metrics"),
            dict
        )
    ):
        return []

    old = ckpt["val_metrics"]
    lines = []

    # These regional metrics should be numerically almost identical.
    for key in (
        "BuildingIoU",
        "True_mIoU",
        "Precision",
        "Recall",
        "F1",
    ):
        if key in old:
            diff_pp = (
                new_val_metrics[key]
                - float(old[key])
            ) * 100.0

            lines.append(
                f"{key}: new={new_val_metrics[key]*100:.6f}% "
                f"saved={float(old[key])*100:.6f}% "
                f"diff={diff_pp:+.6f} percentage points"
            )

    return lines


# ============================================================
# 7. Main
# ============================================================
def main():
    print("=" * 72)
    print("CBRNet — unified PAPER metric re-evaluation")
    print("=" * 72)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("[DEVICE]", device)

    whu_root = find_whu_root()
    ckpt_path = find_best_checkpoint()

    print("[WHU ROOT]", whu_root)

    train_pairs = build_pairs(
        os.path.join(
            whu_root,
            "train",
            "image"
        ),
        os.path.join(
            whu_root,
            "train",
            "label"
        ),
    )

    val_pairs = build_pairs(
        os.path.join(
            whu_root,
            "val",
            "image"
        ),
        os.path.join(
            whu_root,
            "val",
            "label"
        ),
    )

    test_pairs = build_pairs(
        os.path.join(
            whu_root,
            "test",
            "image"
        ),
        os.path.join(
            whu_root,
            "test",
            "label"
        ),
    )

    counts = (
        len(train_pairs),
        len(val_pairs),
        len(test_pairs)
    )

    print(
        "[DATA COUNTS] "
        f"train={counts[0]} "
        f"val={counts[1]} "
        f"test={counts[2]}"
    )

    if counts != CFG["EXPECTED_COUNTS"]:
        raise RuntimeError(
            "Dataset sample counts do not match the paper split "
            f"{CFG['EXPECTED_COUNTS']}. Found {counts}. "
            "Stop here to avoid an invalid comparison."
        )

    val_ds = WHUEvalDataset(
        val_pairs
    )
    test_ds = WHUEvalDataset(
        test_pairs
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(
            CFG["NUM_WORKERS"] > 0
        ),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CFG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(
            CFG["NUM_WORKERS"] > 0
        ),
    )

    print("\n[MODEL] Building architecture...")
    model = CBRNet().to(device)

    ckpt = safe_torch_load(
        ckpt_path,
        map_location=device
    )
    print_checkpoint_metadata(ckpt)

    state = extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(
        state,
        strict=False
    )

    if missing or unexpected:
        print(
            "\n[ERROR] Checkpoint/model key mismatch."
        )
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        raise RuntimeError(
            "The mounted CBRNet_best.pth does not match "
            "the CBRNet architecture used by this script."
        )

    model.eval()

    params_m = (
        sum(
            p.numel()
            for p in model.parameters()
        )
        / 1e6
    )
    print(
        f"[MODEL] Parameters: {params_m:.6f} M"
    )

    # Quick output-shape sanity check.
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            3,
            CFG["IMG_SIZE"],
            CFG["IMG_SIZE"],
            device=device
        )
        refined = model(dummy)[0]

    print(
        "[MODEL] Refined output shape:",
        tuple(refined.shape)
    )

    print(
        "\n[VAL] Re-evaluating with paper metric convention..."
    )
    val_metrics, val_diag, val_rows = (
        evaluate_paper_metrics(
            model,
            val_loader,
            device,
            "VAL"
        )
    )

    print(
        "\n[TEST] Re-evaluating with paper metric convention..."
    )
    test_metrics, test_diag, test_rows = (
        evaluate_paper_metrics(
            model,
            test_loader,
            device,
            "TEST"
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("VALIDATION METRICS — PAPER CONVENTION")
    print("=" * 72)
    print(format_metrics(val_metrics))
    print(
        "  GT-empty-boundary images:",
        f"{val_diag['num_gt_empty_boundary']}/"
        f"{val_diag['num_images']} "
        f"({val_diag['gt_empty_boundary_rate']*100:.2f}%)"
    )
    print(
        "  Both-empty-boundary images:",
        f"{val_diag['num_both_empty_boundary']}/"
        f"{val_diag['num_images']} "
        f"({val_diag['both_empty_boundary_rate']*100:.2f}%)"
    )

    print("\n" + "=" * 72)
    print("INDEPENDENT TEST METRICS — PAPER CONVENTION")
    print("=" * 72)
    print(format_metrics(test_metrics))
    print(
        "  GT-empty-boundary images:",
        f"{test_diag['num_gt_empty_boundary']}/"
        f"{test_diag['num_images']} "
        f"({test_diag['gt_empty_boundary_rate']*100:.2f}%)"
    )
    print(
        "  Both-empty-boundary images:",
        f"{test_diag['num_both_empty_boundary']}/"
        f"{test_diag['num_images']} "
        f"({test_diag['both_empty_boundary_rate']*100:.2f}%)"
    )

    # Regional-metric consistency check against values stored
    # in the best checkpoint, if present.
    consistency_lines = compare_with_saved_val(
        ckpt,
        val_metrics
    )

    if consistency_lines:
        print(
            "\n[CHECK] New vs checkpoint-saved validation "
            "REGIONAL metrics:"
        )
        for line in consistency_lines:
            print(" ", line)

        print(
            "\n[NOTE] Boundary metrics are EXPECTED to change "
            "because this script uses per-image mean aggregation."
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    val_json = os.path.join(
        CFG["OUT_DIR"],
        "CBRNet_val_paper_metrics.json"
    )
    test_json = os.path.join(
        CFG["OUT_DIR"],
        "CBRNet_test_paper_metrics.json"
    )
    val_csv = os.path.join(
        CFG["OUT_DIR"],
        "CBRNet_val_per_image_boundary.csv"
    )
    test_csv = os.path.join(
        CFG["OUT_DIR"],
        "CBRNet_test_per_image_boundary.csv"
    )

    save_metrics_json(
        val_json,
        val_metrics,
        val_diag
    )
    save_metrics_json(
        test_json,
        test_metrics,
        test_diag
    )
    save_per_image_csv(
        val_csv,
        val_rows
    )
    save_per_image_csv(
        test_csv,
        test_rows
    )

    report_path = os.path.join(
        CFG["OUT_DIR"],
        "CBRNet_paper_metric_reeval_report.txt"
    )

    report = []
    report.append(
        "CBRNet unified paper-metric re-evaluation"
    )
    report.append(
        "=" * 72
    )
    report.append(
        f"Checkpoint: {ckpt_path}"
    )
    report.append(
        f"WHU root: {whu_root}"
    )
    report.append(
        f"Device: {device}"
    )
    report.append(
        f"Parameters (M): {params_m:.6f}"
    )
    report.append(
        f"Samples: train={counts[0]}, "
        f"val={counts[1]}, test={counts[2]}"
    )
    report.append("")
    report.append(
        "Metric convention:"
    )
    report.append(
        "- Region: global TP/FP/FN/TN over full split."
    )
    report.append(
        "- Boundary: compute each image separately, "
        "then arithmetic mean."
    )
    report.append(
        "- Boundary extraction: 3x3 dilation - erosion."
    )
    report.append(
        "- Boundary F1 tolerance: 2 pixels (5x5 neighborhood)."
    )
    report.append(
        "- Boundary IoU: strict, no tolerance."
    )
    report.append(
        "- Empty-boundary behavior: paper code preserved; "
        "both-empty image contributes 0."
    )
    report.append("")
    report.append(
        "Validation metrics (%)"
    )
    for k, v in val_metrics.items():
        report.append(
            f"{k}: {v*100:.4f}"
        )
    report.append(
        "Validation diagnostics: "
        + json.dumps(
            val_diag,
            ensure_ascii=False
        )
    )
    report.append("")
    report.append(
        "Independent test metrics (%)"
    )
    for k, v in test_metrics.items():
        report.append(
            f"{k}: {v*100:.4f}"
        )
    report.append(
        "Test diagnostics: "
        + json.dumps(
            test_diag,
            ensure_ascii=False
        )
    )

    if consistency_lines:
        report.append("")
        report.append(
            "Regional consistency check "
            "(new vs checkpoint saved val)"
        )
        report.extend(
            consistency_lines
        )
        report.append(
            "Boundary metrics are not expected to match "
            "the old pooled-boundary aggregation."
        )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "\n".join(report)
        )

    print("\n[SAVED]")
    for p in (
        val_json,
        test_json,
        val_csv,
        test_csv,
        report_path,
    ):
        print(" ", p)

    print(
        "\nDONE. Use the values in "
        "CBRNet_test_paper_metrics.json "
        "for the paper comparison table."
    )


if __name__ == "__main__":
    main()
