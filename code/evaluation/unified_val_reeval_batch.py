# -*- coding: utf-8 -*-
"""UNIFIED VALIDATION RE-EVALUATION BATCH — 8个NEEDS_REEVALUATION checkpoint
与 calibration_x2region_seed42_val.py 同一evaluator实现（已通过calibration PASS）。
只推理。不训练/不选择/不改checkpoint/不评test。
"""
import os, sys, json, csv, hashlib, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels
from PIL import Image
import cv2
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
from pathlib import Path

ROOT = r'D:\KaiFa\study\one\paper_freeze_v4_final_20261004'
VAL_IMG = r'D:\KaiFa\study\one\shuju\whu_data\val\image'
VAL_LBL = r'D:\KaiFa\study\one\shuju\whu_data\val\label'
OUTD = ROOT + r'\13_FINAL_AUDIT\UNIFIED_VAL_REEVAL'
os.makedirs(OUTD, exist_ok=True)

JOBS = [
    ('X0_seed42',        r'D:\KaiFa\study\one\results\X0_Baseline_artifacts\WHU_X0_Baseline_final\checkpoints\X0_Baseline_best.pth', False),
    ('X1_seed42',        r'D:\KaiFa\study\one\results\X1_Edge_NoSupervision_artifacts\WHU_X1_Edge_NoSupervision_final\checkpoints\X1_Edge_NoSupervision_best.pth', True),
    ('X1_seed123',       r'D:\KaiFa\study\one\2\X1_Edge_NoSupervision_seed123_artifacts\WHU_X1_Edge_NoSupervision_seed123_final\checkpoints\X1_Edge_NoSupervision_seed123_best.pth', True),
    ('X1_seed2026',      r'D:\KaiFa\study\one\2\X1_Edge_NoSupervision_seed2026_artifacts\WHU_X1_Edge_NoSupervision_seed2026_final\checkpoints\X1_Edge_NoSupervision_seed2026_best.pth', True),
    ('X2_seed42',        r'D:\KaiFa\study\one\1\checkpoints\X2_Edge_Supervision_best.pth', True),
    ('X2_seed123',       r'D:\KaiFa\study\one\2\X2_Edge_Supervision_seed123_artifacts\WHU_X2_Edge_Supervision_seed123_final\checkpoints\X2_Edge_Supervision_seed123_best.pth', True),
    ('X2_seed2026',      r'D:\KaiFa\study\one\2\X2_Edge_Supervision_seed2026_artifacts\WHU_X2_Edge_Supervision_seed2026_final\checkpoints\X2_Edge_Supervision_seed2026_best.pth', True),
    ('X2_edge_lambda010_seed42', r'D:\KaiFa\study\one\results\X2_Lambda010_artifacts\WHU_X2_Lambda010_final\checkpoints\X2_Lambda010_best.pth', True),
]

# ---- 模型（与calibration逐字一致）----
class EdgeBranch(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.edge_head = nn.Conv2d(out_channels, 1, 1)
    def forward(self, x):
        f = self.conv(x); return f, self.edge_head(f)

class SemanticBranch(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        r = tvmodels.resnet18(weights=None)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.smooth = nn.Sequential(nn.Conv2d(512, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        f = self.smooth(self.backbone(x))
        return F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)

class EdgeEnhancedNet(nn.Module):
    def __init__(self, num_classes=2, use_edge=True):
        super().__init__()
        self.use_edge = use_edge
        self.semantic_branch = SemanticBranch(64)
        fin = 64
        if use_edge:
            self.edge_branch = EdgeBranch(3, 32); fin = 96
        self.fusion = nn.Sequential(nn.Conv2d(fin, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, num_classes, 1))
    def forward(self, x):
        s = self.semantic_branch(x)
        if self.use_edge:
            e, _ = self.edge_branch(x)
            if s.shape[2:] != e.shape[2:]:
                s = F.interpolate(s, size=e.shape[2:], mode='bilinear', align_corners=False)
            s = torch.cat([s, e], dim=1)
        return self.fusion(s)

from pathlib import Path
exts = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}
imgs = {Path(p).stem: p for p in Path(VAL_IMG).iterdir() if p.suffix.lower() in exts}
labs = {Path(p).stem: p for p in Path(VAL_LBL).iterdir() if p.suffix.lower() in exts}
pairs = [(imgs[k], labs[k], k) for k in sorted(set(imgs) & set(labs))]
assert len(pairs) == 1036, f'val数量异常: {len(pairs)}'

def load(i, l):
    img = Image.open(i).convert('RGB').resize((256, 256), Image.BILINEAR)
    lbl = Image.open(l).convert('L').resize((256, 256), Image.NEAREST)
    mask = (np.asarray(lbl) > 127).astype(np.uint8)
    return TF.to_tensor(img).float(), mask

K3 = np.ones((3, 3), np.uint8); K5 = np.ones((5, 5), np.uint8)
def get_boundary(m):
    m = m.astype(np.uint8)
    return ((cv2.dilate(m, K3) - cv2.erode(m, K3)) > 0).astype(np.uint8)
def boundary_metrics(g, p, tol=2):
    gb, pb = get_boundary(g), get_boundary(p)
    biou = np.logical_and(gb, pb).sum() / (np.logical_or(gb, pb).sum() + 1e-8)
    gd, pd = cv2.dilate(gb, K5), cv2.dilate(pb, K5)
    prec = np.logical_and(pb == 1, gd == 1).sum() / (pb.sum() + 1e-8)
    rec = np.logical_and(gb == 1, pd == 1).sum() / (gb.sum() + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8)), float(biou)

SCRIPT_SHA = hashlib.sha256(os.path.abspath(__file__).encode('utf-8')).hexdigest()
ALL = []
t00 = time.time()
for name, ckpt, use_edge in JOBS:
    ck_sha = hashlib.sha256(open(ckpt, 'rb').read()).hexdigest()
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict') if isinstance(ck, dict) and 'state_dict' in ck else ck
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    model = EdgeEnhancedNet(num_classes=2, use_edge=use_edge)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0, f'{name} 加载失败: {len(missing)}/{len(unexpected)}'
    model.eval()
    tp = fp = fn = tn = 0
    bf1s, bios = [], []
    per_rows = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(pairs), 8):
            batch = [load(*pairs[j][:2]) for j in range(i, min(i + 8, len(pairs)))]
            x = torch.stack([b[0] for b in batch])
            gts = [b[1] for b in batch]
            preds = torch.argmax(model(x), dim=1).cpu().numpy().astype(np.uint8)
            for p, g, (ip, lp, stem) in zip(preds, gts, batch):
                tp += int(((p == 1) & (g == 1)).sum()); fp += int(((p == 1) & (g == 0)).sum())
                fn += int(((p == 0) & (g == 1)).sum()); tn += int(((p == 0) & (g == 0)).sum())
                bf1, bio = boundary_metrics(g, p)
                bf1s.append(bf1); bios.append(bio)
                per_rows.append([stem, round(bio * 100, 6), round(bf1 * 100, 6)])
    biou = tp / (tp + fp + fn + 1e-8)
    bg = tn / (tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    row = {'job': name, 'checkpoint': ckpt, 'checkpoint_sha256': ck_sha,
           'BuildingIoU': biou * 100, 'True_mIoU': (biou + bg) / 2 * 100,
           'Precision': prec * 100, 'Recall': rec * 100, 'F1': f1 * 100,
           'BoundaryF1': float(np.mean(bf1s)) * 100, 'BoundaryIoU': float(np.mean(bios)) * 100,
           'n_images': len(pairs), 'evaluator_script_sha256': SCRIPT_SHA, 'elapsed_s': round(time.time() - t0, 1)}
    ALL.append(row)
    od = os.path.join(OUTD, name)
    os.makedirs(od, exist_ok=True)
    json.dump(row, open(os.path.join(od, 'metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    with open(os.path.join(od, 'per_image.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['image', 'BuildingIoU', 'BoundaryF1'])
        w.writerows(per_rows)
    print(f"[{name}] BuildingIoU={row['BuildingIoU']:.4f} F1={row['F1']:.4f} BF1={row['BoundaryF1']:.4f} BIoU={row['BoundaryIoU']:.4f} ({row['elapsed_s']}s)", flush=True)

with open(OUTD + r'\UNIFIED_VAL_REEVAL_ALL.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['job', 'checkpoint', 'checkpoint_sha256', 'BuildingIoU', 'True_mIoU', 'Precision', 'Recall', 'F1', 'BoundaryF1', 'BoundaryIoU', 'n_images', 'evaluator_script_sha256'])
    for r in ALL:
        w.writerow([r['job'], r['checkpoint'], r['checkpoint_sha256']] + [round(r[k], 6) for k in ['BuildingIoU', 'True_mIoU', 'Precision', 'Recall', 'F1', 'BoundaryF1', 'BoundaryIoU']] + [r['n_images'], r['evaluator_script_sha256']])
print(f'全部完成, 总耗时 {time.time()-t00:.0f}s', flush=True)
