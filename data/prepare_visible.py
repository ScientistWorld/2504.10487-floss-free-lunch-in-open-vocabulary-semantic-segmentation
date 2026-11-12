"""Build the method-visible ADE20K artifacts (GPU recommended).

Writes data/downloads/naclip/:
  text_embeddings.npy        [80, 150, 512] single-template class embeddings
  adapt_feats.npy / adapt_index.npz    2000 unlabeled adaptation images
  dev_feats.npy / dev_index.npz        500 labeled dev images
  dev_labels.npy             flat uint8 labels at label resolution (255 = ignore)
  dev_oracle.npz             per-template class IoU on the dev set (rows 0..79
                             single templates, row 80 the averaged classifier)

Idempotent: skips artifacts that already exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from naclip_features import (  # noqa: E402
    ADE20K_CLASSES,
    IMAGENET_TEMPLATES,
    NaclipEncoder,
    default_device,
    extract_split,
    read_ids,
    resized_hw,
)

SRC = ROOT / "data" / "downloads" / "ade20k"
OUT = ROOT / "data" / "downloads" / "naclip"
MODEL = ROOT / "data" / "downloads" / "clip-vit-base-patch16"


def load_ade_label(path: Path, height: int, width: int) -> np.ndarray:
    """ADE20K label -> [height/2, width/2] uint8 with 0..149 classes and 255 ignore."""
    lab = Image.open(path)
    lab = lab.resize((width // 2, height // 2), Image.NEAREST)
    arr = np.asarray(lab, dtype=np.int16)
    return np.where(arr == 0, 255, arr - 1).astype(np.uint8)


def build_labels(images_dir: Path, ann_dir: Path, index_path: Path, out_path: Path) -> None:
    index = np.load(index_path)
    chunks = []
    for i, image_id in enumerate(index["ids"]):
        with Image.open(images_dir / f"{image_id}.jpg") as img:
            height, width = resized_hw(img.width, img.height)
        lab = load_ade_label(ann_dir / f"{image_id}.png", height, width)
        assert lab.shape == tuple(index["label_hw"][i]), (image_id, lab.shape, index["label_hw"][i])
        chunks.append(lab.reshape(-1))
    np.save(out_path, np.concatenate(chunks))


@torch.no_grad()
def template_iou_table(feats_path: Path, index_path: Path, labels_path: Path, text: np.ndarray) -> dict:
    device = default_device()
    feats = np.load(feats_path, mmap_mode="r")
    index = np.load(index_path)
    labels = np.load(labels_path, mmap_mode="r")
    m, k, d = text.shape
    t = torch.from_numpy(text).to(device)
    avg = F.normalize(t.mean(0), dim=-1)
    classifiers = torch.cat([t, avg[None]], dim=0)  # [M+1, K, D]
    inter = torch.zeros(m + 1, k, dtype=torch.float64, device=device)
    pred_area = torch.zeros(m + 1, k, dtype=torch.float64, device=device)
    gt_area = torch.zeros(k, dtype=torch.float64, device=device)
    chunk = 27
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
        for s in range(0, m + 1, chunk):
            cls = classifiers[s : s + chunk]
            logits = torch.einsum("hwd,mkd->mkhw", f, cls)
            up = F.interpolate(logits, size=(lh, lw), mode="bilinear", align_corners=False)
            pred = up.argmax(1)[:, valid]  # [m, n_valid]
            offs = torch.arange(pred.shape[0], device=device)[:, None] * k
            pred_area[s : s + chunk] += torch.bincount((pred + offs).flatten(), minlength=pred.shape[0] * k).view(-1, k).double()
            hit = pred == g[None]
            inter[s : s + chunk] += torch.bincount((pred + offs)[hit], minlength=pred.shape[0] * k).view(-1, k).double()
    union = pred_area + gt_area[None] - inter
    iou = torch.where(union > 0, inter / union.clamp_min(1), torch.full_like(inter, float("nan")))
    return {
        "iou": iou.cpu().numpy(),
        "intersection": inter.cpu().numpy(),
        "union": union.cpu().numpy(),
        "gt_area": gt_area.cpu().numpy(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    encoder = None

    def enc() -> NaclipEncoder:
        nonlocal encoder
        if encoder is None:
            encoder = NaclipEncoder(MODEL, default_device())
        return encoder

    text_path = OUT / "text_embeddings.npy"
    if not text_path.exists():
        np.save(text_path, enc().encode_text(ADE20K_CLASSES, IMAGENET_TEMPLATES))
    for split, ids_file in (("adapt", "ade20k_adapt_ids.txt"), ("dev", "ade20k_dev_ids.txt")):
        if not (OUT / f"{split}_index.npz").exists():
            extract_split(enc(), SRC / split / "images", read_ids(ROOT / "data" / "splits" / ids_file), OUT / split)
    if not (OUT / "dev_labels.npy").exists():
        build_labels(SRC / "dev" / "images", SRC / "dev" / "annotations", OUT / "dev_index.npz", OUT / "dev_labels.npy")
    if not (OUT / "dev_oracle.npz").exists():
        text = np.load(text_path)
        table = template_iou_table(OUT / "dev_feats.npy", OUT / "dev_index.npz", OUT / "dev_labels.npy", text)
        np.savez(OUT / "dev_oracle.npz", **table)
        iou = table["iou"]
        print(f"dev averaged-classifier mIoU: {np.nanmean(iou[-1]) * 100:.2f}", flush=True)
    print("visible artifacts ready", flush=True)


if __name__ == "__main__":
    main()
