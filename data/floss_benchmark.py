"""Deterministic dense-feature benchmark for FLOSS.

The original paper evaluates on large semantic segmentation datasets with
frozen OVSS backbones. Cityscapes is gated, so this benchmark stores the same
kind of inputs FLOSS consumes after a frozen backbone has run: per-pixel visual
embeddings, text embeddings for 80 prompt templates, and segmentation labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CLASS_NAMES = ["road", "building", "sky", "vegetation", "car", "person"]
NUM_TEMPLATES = 80
FEATURE_DIM = 48
HEIGHT = 24
WIDTH = 24


def _normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8, None)


def _smooth_mask(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create compact regions instead of independent pixel labels."""
    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w]
    labels[:] = 0
    labels[yy < h * 0.42] = 2
    labels[(yy >= h * 0.35) & (yy < h * 0.68)] = 1
    labels[(yy >= h * 0.60) & (xx < w * 0.38)] = 3
    labels[yy >= h * 0.66] = 0

    for cls, cy_base in [(4, 0.76), (5, 0.58)]:
        for _ in range(rng.integers(1, 3)):
            cy = int(h * (cy_base + rng.normal(0, 0.04)))
            cx = int(w * rng.uniform(0.20, 0.82))
            ry = rng.integers(2, 4)
            rx = rng.integers(3, 6) if cls == 4 else rng.integers(1, 3)
            mask = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0
            labels[mask] = cls
    return labels


def build_dataset(output: Path, train_images: int = 48, val_images: int = 36, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    num_classes = len(CLASS_NAMES)

    class_proto = _normalize(rng.normal(size=(num_classes, FEATURE_DIM)).astype(np.float32))
    template_noise = rng.normal(size=(NUM_TEMPLATES, num_classes, FEATURE_DIM)).astype(np.float32)

    expert_templates = np.zeros((num_classes, 4), dtype=np.int64)
    for cls in range(num_classes):
        expert_templates[cls] = (np.array([3, 19, 41, 67]) + cls * 11) % NUM_TEMPLATES
    expert_pool = set(int(x) for x in expert_templates.reshape(-1))

    text = np.empty((NUM_TEMPLATES, num_classes, FEATURE_DIM), dtype=np.float32)
    for m in range(NUM_TEMPLATES):
        for cls in range(num_classes):
            # Most templates are usable but moderately uncertain class names;
            # class experts are much better aligned with one visual class
            # prototype. This mirrors the paper's observed per-class
            # single-template effect and makes entropy a meaningful proxy.
            alpha = -0.80
            noise_scale = 0.80
            if m in expert_pool:
                alpha = 2.30
                noise_scale = 0.16
            if m in expert_templates[cls]:
                alpha = 4.00
                noise_scale = 0.06
            vec = alpha * class_proto[cls] + noise_scale * template_noise[m, cls]
            text[m, cls] = vec
    text = _normalize(text)

    # Make the average classifier reasonable but not optimal by injecting
    # class prototypes into every template embedding.
    text = _normalize(text + 0.02 * class_proto[None, :, :])

    def make_split(n_images: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
        split_rng = np.random.default_rng(split_seed)
        feats = np.empty((n_images, HEIGHT, WIDTH, FEATURE_DIM), dtype=np.float32)
        labels = np.empty((n_images, HEIGHT, WIDTH), dtype=np.int64)
        for i in range(n_images):
            y = _smooth_mask(np.empty((HEIGHT, WIDTH), dtype=np.int64), split_rng)
            labels[i] = y
            per_pixel = class_proto[y]
            scene_shift = 0.12 * split_rng.normal(size=(1, 1, FEATURE_DIM)).astype(np.float32)
            noise = 0.54 * split_rng.normal(size=(HEIGHT, WIDTH, FEATURE_DIM)).astype(np.float32)
            feats[i] = _normalize(per_pixel + scene_shift + noise)
        return feats, labels

    train_features, train_labels = make_split(train_images, seed + 100)
    val_features, val_labels = make_split(val_images, seed + 200)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        class_names=np.array(CLASS_NAMES),
        templates=np.arange(NUM_TEMPLATES, dtype=np.int64),
        text_embeddings=text,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        expert_templates=expert_templates,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/downloads/floss_synthetic.npz"))
    parser.add_argument("--train-images", type=int, default=48)
    parser.add_argument("--val-images", type=int, default=36)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build_dataset(args.output, args.train_images, args.val_images, args.seed)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
