"""Visible train-slice scorer for saved OVSS segmentation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SLICE = "train"


def _slice_indices(num_images: int) -> np.ndarray:
    """Use a fixed even-index half of the validation pool."""
    return np.arange(0, num_images, 2, dtype=np.int64)


def mean_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> tuple[float, list[float]]:
    ious: list[float] = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_mask, target_mask).sum()
        ious.append(float(inter / union))
    return float(np.mean(ious) * 100.0), [x * 100.0 for x in ious]


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean() * 100.0)


def mean_class_accuracy(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    recalls: list[float] = []
    for cls in range(num_classes):
        mask = target == cls
        if mask.any():
            recalls.append(float((pred[mask] == cls).mean()))
    return float(np.mean(recalls) * 100.0)


def evaluate_run(dataset_path: Path, labels_path: Path, output_dir: Path) -> dict[str, float]:
    data = np.load(dataset_path, allow_pickle=False)
    labels_data = np.load(labels_path, allow_pickle=False)
    labels = labels_data["val_labels"]
    indices = _slice_indices(labels.shape[0])
    num_classes = len(data["class_names"])

    pred = np.load(output_dir / "predictions.npy")
    if pred.shape != labels.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match labels {labels.shape}")

    pred_slice = pred[indices]
    label_slice = labels[indices]
    miou, _per_class = mean_iou(pred_slice, label_slice, num_classes)
    return {
        "mIoU": miou,
        "pixel_accuracy": pixel_accuracy(pred_slice, label_slice),
        "mean_class_accuracy": mean_class_accuracy(pred_slice, label_slice, num_classes),
        "num_templates": float(len(data["templates"])),
        "num_experts_per_class": float(_num_experts(output_dir)),
    }


def evaluate_expert_quality(dataset_path: Path, labels_path: Path, output_dir: Path) -> dict[str, float]:
    data = np.load(dataset_path, allow_pickle=False)
    predicted_path = output_dir / "experts.npy"
    if not predicted_path.exists():
        raise ValueError(f"{output_dir} does not contain experts.npy")

    predicted = np.load(predicted_path)
    oracle = np.load(labels_path, allow_pickle=False)["expert_templates"]
    if predicted.shape != oracle.shape:
        raise ValueError(f"Expert shape {predicted.shape} does not match oracle shape {oracle.shape}")

    per_class = []
    for pred_row, oracle_row in zip(predicted, oracle):
        overlap = len(set(int(x) for x in pred_row) & set(int(x) for x in oracle_row))
        per_class.append(overlap / len(pred_row))

    return {
        "expert_precision_pct": float(np.mean(per_class) * 100.0),
        "num_experts_per_class": float(predicted.shape[1]),
        "num_templates": float(len(data["templates"])),
    }


def _num_experts(output_dir: Path) -> int:
    experts = output_dir / "experts.npy"
    if not experts.exists():
        return 80
    arr = np.load(experts)
    return int(arr.shape[1])


def _add_experiment(results: dict, name: str) -> None:
    results["experiments"][name] = {"slice": SLICE, "results": {}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/downloads/floss_synthetic_inputs.npz"))
    parser.add_argument("--labels", type=Path, default=Path("data/downloads/floss_synthetic_eval.npz"))
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--low-data-runs", nargs="*", default=[])
    parser.add_argument("--expert-quality-runs", nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_train.json"))
    args = parser.parse_args()

    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {"experiments": {}}
    _add_experiment(results, "synthetic_ovss")
    for spec in args.runs:
        name, path = spec.split("=", 1)
        metrics = evaluate_run(args.dataset, args.labels, Path(path))
        results["experiments"]["synthetic_ovss"]["results"][name] = metrics

    if args.low_data_runs:
        _add_experiment(results, "low_data_synthetic_ovss")
        for spec in args.low_data_runs:
            name, path = spec.split("=", 1)
            metrics = evaluate_run(args.dataset, args.labels, Path(path))
            results["experiments"]["low_data_synthetic_ovss"]["results"][name] = metrics

    if args.expert_quality_runs:
        _add_experiment(results, "expert_quality_synthetic_ovss")
        for spec in args.expert_quality_runs:
            name, path = spec.split("=", 1)
            metrics = evaluate_expert_quality(args.dataset, args.labels, Path(path))
            results["experiments"]["expert_quality_synthetic_ovss"]["results"][name] = metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
