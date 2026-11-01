# 论文唯一边界口径(与Kaggle实现逐字一致)
import cv2, numpy as np
K3 = np.ones((3,3), np.uint8)
def get_boundary(mask):
    m = mask.astype(np.uint8)
    return ((cv2.dilate(m, K3) - cv2.erode(m, K3)) > 0).astype(np.uint8)
def boundary_metrics(gt, pred, tolerance=2):
    gb, pb = get_boundary(gt), get_boundary(pred)
    biou = np.logical_and(gb, pb).sum() / (np.logical_or(gb, pb).sum() + 1e-8)
    k = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    gd, pd = cv2.dilate(gb, k), cv2.dilate(pb, k)
    prec = np.logical_and(pb==1, gd==1).sum() / (pb.sum() + 1e-8)
    rec = np.logical_and(gb==1, pd==1).sum() / (gb.sum() + 1e-8)
    return float(2*prec*rec/(prec+rec+1e-8)), float(biou)
# 聚合: 逐图计算后取算术平均; 双空图块因+1e-8分母计0
