"""Template-averaging OVSS baseline used by FLOSS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def run(dataset_path: Path, output_dir: Path, temperature: float = 12.0) -> None:
    data = np.load(dataset_path, allow_pickle=False)
    text = data["text_embeddings"].astype(np.float32)
    features = data["val_features"].astype(np.float32)
    classifier = text.mean(axis=0)
    classifier = classifier / np.clip(np.linalg.norm(classifier, axis=-1, keepdims=True), 1e-8, None)
    probs = softmax(np.einsum("...d,kd->...k", features, classifier) * temperature)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predictions.npy", probs.argmax(axis=-1).astype(np.int64))
    np.save(output_dir / "probabilities.npy", probs.astype(np.float32))
    (output_dir / "metadata.json").write_text(
        json.dumps({"method": "template_average", "dataset": str(dataset_path)}, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/downloads/floss_synthetic.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/template_average"))
    parser.add_argument("--temperature", type=float, default=12.0)
    args = parser.parse_args()
    run(args.dataset, args.output_dir, args.temperature)


if __name__ == "__main__":
    main()
