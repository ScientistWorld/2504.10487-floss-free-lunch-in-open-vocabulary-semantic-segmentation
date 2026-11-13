"""Build the hidden test artifacts (GPU recommended). Test answers live only here.

Writes eval/test/data/naclip/:
  text_embeddings.npy                 the evaluator's own copy of the template pool
  pool_feats.npy / pool_index.npz     2000 hidden unlabeled adaptation images
  val_feats.npy / val_index.npz       the 2000 ADE20K validation images
  val_labels.npy                      flat uint8 labels (255 = ignore)
  val_oracle.npz                      per-template class IoU on the validation set
and eval/test/label_hashes.json (committed) so the evaluator can detect edits.

Idempotent: skips artifacts that already exist.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(HERE))

from naclip_features import (  # noqa: E402
    ADE20K_CLASSES,
    IMAGENET_TEMPLATES,
    NaclipEncoder,
    default_device,
    extract_split,
    read_ids,
    resized_hw,
)
from oracle import array_sha256, compute_oracle, file_sha256  # noqa: E402

SRC = HERE / "data" / "ade20k"
OUT = HERE / "data" / "naclip"
MODEL = ROOT / "data" / "downloads" / "clip-vit-base-patch16"
HASHES = HERE / "label_hashes.json"


def _label(path: Path, height: int, width: int) -> np.ndarray:
    lab = Image.open(path).resize((width // 2, height // 2), Image.NEAREST)
    arr = np.asarray(lab, dtype=np.int16)
    return np.where(arr == 0, 255, arr - 1).astype(np.uint8)


def _labels(index_path: Path, out_path: Path) -> None:
    index = np.load(index_path)
    chunks = []
    for i, image_id in enumerate(index["ids"]):
        with Image.open(SRC / "val" / "images" / f"{image_id}.jpg") as img:
            height, width = resized_hw(img.width, img.height)
        lab = _label(SRC / "val" / "annotations" / f"{image_id}.png", height, width)
        assert lab.shape == tuple(index["label_hw"][i]), (image_id, lab.shape)
        chunks.append(lab.reshape(-1))
    np.save(out_path, np.concatenate(chunks))


def _unseal() -> None:
    """The evaluator makes answers unreadable while a procedure runs; undo a seal a killed run left behind."""
    for path in (OUT / "val_labels.npy", OUT / "val_oracle.npz", SRC / "val" / "annotations"):
        if path.exists():
            need = 0o770 if path.is_dir() else 0o660
            mode = path.stat().st_mode & 0o7777
            if mode & need != need:
                os.chmod(path, mode | need)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _unseal()
    encoder = None

    def enc() -> NaclipEncoder:
        nonlocal encoder
        if encoder is None:
            encoder = NaclipEncoder(MODEL, default_device())
        return encoder

    if not (OUT / "text_embeddings.npy").exists():
        np.save(OUT / "text_embeddings.npy", enc().encode_text(ADE20K_CLASSES, IMAGENET_TEMPLATES))
    for split, ids_file in (("pool", "ade20k_pool_ids.txt"), ("val", "ade20k_val_ids.txt")):
        if not (OUT / f"{split}_index.npz").exists():
            extract_split(enc(), SRC / split / "images", read_ids(HERE / "splits" / ids_file), OUT / split)
    if not (OUT / "val_labels.npy").exists():
        _labels(OUT / "val_index.npz", OUT / "val_labels.npy")
    if not (OUT / "val_oracle.npz").exists():
        table = compute_oracle(
            np.load(OUT / "val_feats.npy", mmap_mode="r"),
            np.load(OUT / "val_index.npz"),
            np.load(OUT / "val_labels.npy", mmap_mode="r"),
            np.load(OUT / "text_embeddings.npy"),
        )
        np.savez(OUT / "val_oracle.npz", **table)
        print(f"val averaged-classifier mIoU: {np.nanmean(table['iou'][-1]) * 100:.2f}", flush=True)
    hashes = {
        "val_labels.npy": file_sha256(OUT / "val_labels.npy"),
        "val_oracle_iou": array_sha256(np.load(OUT / "val_oracle.npz")["iou"]),
    }
    if HASHES.exists():
        old = json.loads(HASHES.read_text())
        if old.get("val_labels.npy") != hashes["val_labels.npy"]:
            print("WARNING: regenerated val labels differ from the committed hash", flush=True)
    else:
        HASHES.write_text(json.dumps(hashes, indent=2) + "\n")
    print("hidden test artifacts ready", flush=True)


if __name__ == "__main__":
    main()
