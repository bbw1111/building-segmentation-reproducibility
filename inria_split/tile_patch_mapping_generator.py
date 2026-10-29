# -*- coding: utf-8 -*-
# ============================================================================
# Inria 干净重训实验 —— Step 1/3：一次性切片 + 预训练权重缓存（CPU 即可，~10 分钟）
# ----------------------------------------------------------------------------
# 为什么拆这一步：
#   原整段脚本每次重跑都要花 ~10 分钟从 180 个 5000×5000 tile 重切 1620 个
#   patch，而预训练权重一旦下载失败，整个训练直接崩掉、前功尽弃。
#   本步把"准备工作"单独跑一次并保存为 Notebook Output，之后 Step2/Step3
#   挂载它即可完全离线训练，中断重跑也不再重复切片。
#
# Kaggle 用法：
#   1. 新建 Notebook（无需 GPU），Add Input 挂载 Inria 原始数据集
#      （含 AerialImageDataset/train/{images,gt}）
#   2. 本文件整段粘贴到一个 cell 运行（或 Save & Run All 提交）
#   3. 运行结束 Save Version；在 Step2/Step3 的 Notebook 里把本 Notebook
#      的 Output 添加为 Input（内含切片与 resnet18 权重）
#
# 输出 /kaggle/working/inria_step1_out/：
#   inria_clean_patches/images/*.png, gt/*.png   180 tile × 3×3 = 1620 个 patch
#   pretrained/resnet18-f37072fd.pth             ImageNet 预训练权重（离线用）
#   inria_tile_split.txt                         tile 级无泄漏划分记录
#
# 确定性：切片与划分逻辑与原整段脚本完全一致，产物可复现。
# ============================================================================
import os, re, glob, time
import numpy as np
import cv2
import torch

try:                                    # 静音 GeoTIFF 未知标签的无害警告刷屏
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_ERROR)
except Exception:
    pass

CFG = dict(
    PATCH         = 512,     # 从 5000×5000 tile 上切 512×512
    STRIDE        = 1536,    # 每 tile 3×3=9 个 patch，共 180×9=1620
    VAL_TILE_STEP = 6,       # 每 6 个 tile 取 1 个作验证（180/6=30，城市均衡）
    OUT_DIR       = '/kaggle/working/inria_step1_out',
)
RESNET18_URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'

# ---------------------------------------------------------------- 基础工具
def find_inria_root():
    cands = glob.glob('/kaggle/input/**/AerialImageDataset/train/images', recursive=True)
    if not cands:
        raise FileNotFoundError('未找到 AerialImageDataset/train/images，请检查 Inria 数据集是否已 Add Input')
    return os.path.dirname(os.path.dirname(cands[0]))

def list_tiles(root):
    tiles = sorted({re.match(r'(.+?)\.tif$', os.path.basename(p)).group(1)
                    for p in glob.glob(os.path.join(root, 'train', 'images', '*.tif'))})
    return tiles

def generate_patches(root, img_dir, gt_dir):
    """512×512 / stride 1536 确定性切分；patch 命名 {tile}_{y}_{x}.png 记录来源"""
    tiles = list_tiles(root)
    n = 0
    for t in tiles:
        img = cv2.imread(os.path.join(root, 'train', 'images', t + '.tif'), cv2.IMREAD_COLOR)
        gt  = cv2.imread(os.path.join(root, 'train', 'gt',     t + '.tif'), cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            print(f'[WARN] 读取失败，跳过 {t}'); continue
        H, W = img.shape[:2]; S = CFG['PATCH']; st = CFG['STRIDE']
        for y in range(0, H - S + 1, st):
            for x in range(0, W - S + 1, st):
                cv2.imwrite(os.path.join(img_dir, f'{t}_{y}_{x}.png'), img[y:y+S, x:x+S])
                cv2.imwrite(os.path.join(gt_dir,  f'{t}_{y}_{x}.png'), gt[y:y+S, x:x+S])
                n += 1
    return n, tiles

def tile_level_split(img_dir, val_step=CFG['VAL_TILE_STEP'], out_txt=None):
    """tile 级无泄漏划分：排序后每 val_step 个取 1 个为验证 tile（城市均衡）"""
    tiles = sorted({re.match(r'(.+?)_\d+_\d+\.png$', f).group(1) for f in os.listdir(img_dir)})
    val_tiles   = sorted(t for i, t in enumerate(tiles) if i % val_step == val_step - 1)
    train_tiles = sorted(t for t in tiles if t not in set(val_tiles))
    train_files = sorted(f for f in os.listdir(img_dir)
                         if re.match(r'(.+?)_\d+_\d+\.png$', f).group(1) in set(train_tiles))
    val_files   = sorted(f for f in os.listdir(img_dir)
                         if re.match(r'(.+?)_\d+_\d+\.png$', f).group(1) in set(val_tiles))
    if out_txt:
        with open(out_txt, 'w', encoding='utf-8') as fp:
            fp.write(f'train tiles ({len(train_tiles)}): {train_tiles}\n\n')
            fp.write(f'val tiles   ({len(val_tiles)}): {val_tiles}\n\n')
            fp.write(f'train patches: {len(train_files)} | val patches: {len(val_files)}\n')
    return train_files, val_files, train_tiles, val_tiles

def load_resnet18_sd():
    """预训练权重：挂载里已有就直接用（零网络）；否则带 5 次重试下载并缓存"""
    hits = glob.glob('/kaggle/input/**/resnet18*.pth', recursive=True)
    if hits:
        print(f'[WEIGHTS] 使用挂载的预训练权重: {hits[0]}')
        return torch.load(hits[0], map_location='cpu', weights_only=False)
    cache = os.path.join(CFG['OUT_DIR'], 'pretrained'); os.makedirs(cache, exist_ok=True)
    for i in range(1, 6):
        try:
            print(f'[WEIGHTS] 下载 ResNet18 ImageNet 权重 (第 {i}/5 次)…')
            return torch.hub.load_state_dict_from_url(RESNET18_URL, model_dir=cache, progress=False)
        except Exception as ex:
            print(f'[WARN] 下载失败: {ex}'); time.sleep(min(60, 10 * i))
    raise RuntimeError('权重多次下载失败；请重跑本 Step，或手动上传 resnet18-f37072fd.pth 作数据集挂载')

# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(CFG['OUT_DIR'], exist_ok=True)

    img_dir = os.path.join(CFG['OUT_DIR'], 'inria_clean_patches', 'images')
    gt_dir  = os.path.join(CFG['OUT_DIR'], 'inria_clean_patches', 'gt')
    if os.path.isdir(img_dir) and len(os.listdir(img_dir)) >= 1000:
        print(f'[DATA] 已有 {len(os.listdir(img_dir))} 个 patch，跳过切片')
    else:
        root = find_inria_root()
        os.makedirs(img_dir, exist_ok=True); os.makedirs(gt_dir, exist_ok=True)
        t0 = time.time()
        n, tiles = generate_patches(root, img_dir, gt_dir)
        print(f'[DATA] 从 {len(tiles)} 个 tile 生成 {n} 个 patch，用时 {time.time()-t0:.0f}s')

    tr_f, va_f, tr_t, va_t = tile_level_split(
        img_dir, out_txt=os.path.join(CFG['OUT_DIR'], 'inria_tile_split.txt'))
    print(f'[SPLIT] train tiles {len(tr_t)} / val tiles {len(va_t)} | '
          f'train patches {len(tr_f)} / val patches {len(va_f)}')

    load_resnet18_sd()
    print(f'Step1 完成。结果目录: {CFG["OUT_DIR"]}')
    print('>>> 现在 Save Version，然后在 Step2/Step3 的 Notebook 里把本 Notebook 的 Output 添加为 Input')

if __name__ == '__main__':
    main()
