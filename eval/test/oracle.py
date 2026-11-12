"""Hidden-test ground-truth helpers: label hashing and the per-template IoU oracle.

The oracle defines the true class experts of the paper's Eq. (4): template m is
a true expert of class k when the single-template classifier W(T_m) reaches a
higher IoU on class k than the averaged classifier, measured on the hidden
validation labels with the same bilinear upsampling used for scoring.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return array_sha256(np.load(path, mmap_mode="r"))


@torch.no_grad()
def compute_oracle(feats: np.ndarray, index, labels: np.ndarray, text: np.ndarray) -> dict:
    """Per-class IoU of the 80 single-template classifiers (rows 0..79) and the
    averaged classifier (row 80) over the labelled images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, k, d = text.shape
    t = torch.from_numpy(np.asarray(text, dtype=np.float32)).to(device)
    classifiers = torch.cat([t, F.normalize(t.mean(0), dim=-1)[None]], dim=0)
    inter = torch.zeros(m + 1, k, dtype=torch.float64, device=device)
    pred_area = torch.zeros(m + 1, k, dtype=torch.float64, device=device)
    gt_area = torch.zeros(k, dtype=torch.float64, device=device)
    for i in range(len(index["ids"])):
        h, w = (int(v) for v in index["hw"][i])
        lh, lw = (int(v) for v in index["label_hw"][i])
        f = torch.from_numpy(np.asarray(feats[index["offsets"][i] : index["offsets"][i + 1]], dtype=np.float32))
        f = f.to(device).view(h, w, d)
        gt = torch.from_numpy(np.asarray(labels[index["label_offsets"][i] : index["label_offsets"][i + 1]]))
        gt = gt.to(device).long().view(lh, lw)
        valid = gt != 255
        g = gt[valid]
        gt_area += torch.bincount(g, minlength=k).double()
        for s in range(0, m + 1, 27):
            logits = torch.einsum("hwd,mkd->mkhw", f, classifiers[s : s + 27])
            pred = F.interpolate(logits, size=(lh, lw), mode="bilinear", align_corners=False).argmax(1)[:, valid]
            offs = torch.arange(pred.shape[0], device=device)[:, None] * k
            n = pred.shape[0] * k
            pred_area[s : s + 27] += torch.bincount((pred + offs).flatten(), minlength=n).view(-1, k).double()
            inter[s : s + 27] += torch.bincount((pred + offs)[pred == g[None]], minlength=n).view(-1, k).double()
    union = pred_area + gt_area[None] - inter
    iou = torch.where(union > 0, inter / union.clamp_min(1), torch.full_like(inter, float("nan")))
    return {
        "iou": iou.cpu().numpy(),
        "intersection": inter.cpu().numpy(),
        "union": union.cpu().numpy(),
        "gt_area": gt_area.cpu().numpy(),
    }


def true_experts(iou: np.ndarray) -> np.ndarray:
    """[M, K] bool: template m beats the averaged classifier (last row) on class k."""
    single, avg = iou[:-1], iou[-1]
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(single, nan=-1.0) > np.nan_to_num(avg, nan=np.inf)[None, :]
