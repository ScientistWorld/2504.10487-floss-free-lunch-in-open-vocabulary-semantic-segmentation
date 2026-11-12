"""Visible-slice (train) scorer for the ADE20K template-selection task.

Usage (no arguments; this is what scripts/evaluate_train.sh runs):

    python eval/train/evaluate.py

Same protocol and interface as the hidden test, on method-visible data: every
directory under outputs/ is a method whose outputs/<name>/procedure.py is run
by a separate worker (eval/train/worker.py) that adapts on the visible
unlabeled pool and predicts the 500 labeled dev images. This process recomputes
every number from the per-pixel class maps and the dev labels; a failed run
receives the worst value in every metric of every experiment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402

BUILTIN = "template_average"


def _discover(outputs_root: Path) -> list[tuple[str, Path | None]]:
    methods: list[tuple[str, Path | None]] = [(BUILTIN, None)]
    if outputs_root.is_dir():
        for path in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
            methods.append((path.name if path.name != BUILTIN else f"{BUILTIN}@outputs", path))
    return methods


def _confusion(pred: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros(k * k, dtype=np.int64)
    step = 1 << 24
    for s in range(0, len(gt), step):
        g = np.asarray(gt[s : s + step])
        p = np.asarray(pred[s : s + step]).astype(np.int64)
        v = g != 255
        cm += np.bincount(g[v].astype(np.int64) * k + p[v], minlength=k * k)
    return cm.reshape(k, k)


def _miou(cm: np.ndarray) -> float:
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou) * 100.0)


def _load_pred(path: Path, size: int, k: int) -> np.ndarray:
    pred = np.load(path, mmap_mode="r")
    if pred.dtype != np.uint8 or pred.shape != (size,):
        raise ValueError(f"{path.name}: expected uint8[{size}], got {pred.dtype}{list(pred.shape)}")
    if size and int(pred.max()) >= k:
        raise ValueError(f"{path.name}: class id out of range")
    return pred


class Truth:
    def __init__(self) -> None:
        for p in (P.TEXT, P.LABELS, P.ORACLE, Path(f"{P.EVAL_PREFIX}_index.npz"), Path(f"{P.POOL_PREFIX}_index.npz")):
            if not p.exists():
                raise FileNotFoundError(f"visible data missing: {p}; run scripts/prepare_data.sh")
        self.labels = np.load(P.LABELS, mmap_mode="r")
        index = np.load(f"{P.EVAL_PREFIX}_index.npz")
        self.label_hw = index["label_hw"]
        self.label_offsets = index["label_offsets"]
        if int(self.label_offsets[-1]) != len(self.labels):
            raise RuntimeError("dev label index is inconsistent with the label file")
        text = np.load(P.TEXT)
        self.m, self.k, _ = text.shape
        self.n_pool = len(np.load(f"{P.POOL_PREFIX}_index.npz")["hw"])
        iou = np.load(P.ORACLE)["iou"]
        single, avg = iou[:-1], iou[-1]
        with np.errstate(invalid="ignore"):
            self.true_experts = np.nan_to_num(single, nan=-1.0) > np.nan_to_num(avg, nan=np.inf)[None, :]
        self.eligible = np.flatnonzero(self.true_experts.any(axis=0))
        self.low_ids = P.low_eval_ids(len(self.label_hw))
        self.low_gt = np.concatenate(
            [self.labels[self.label_offsets[i] : self.label_offsets[i + 1]] for i in self.low_ids])
        self.n_low_runs = len(P.low_data_runs(self.n_pool))

    def expert_precision(self, ids: np.ndarray) -> float:
        hits = [len(set(ids[c].tolist()) & set(np.flatnonzero(self.true_experts[:, c]).tolist()))
                for c in self.eligible]
        return float(np.mean(hits) / P.NUM_EXPERTS * 100.0)

    def chance_precision(self) -> float:
        return float(np.mean(self.true_experts[:, self.eligible].sum(0) / self.m) * 100.0)


def _score(name: str, method_dir: Path | None, truth: Truth) -> dict:
    scratch = P.ROOT / "tmp" / "eval_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="train_", dir=scratch) as tmp:
        tmp_path = Path(tmp)
        cmd = [sys.executable, str(HERE / "worker.py"), "--out", str(tmp_path)]
        if method_dir is None:
            cmd.append("--builtin")
        else:
            procedure = method_dir / "procedure.py"
            if not procedure.is_file():
                raise FileNotFoundError(f"{method_dir} has no procedure.py")
            cmd += ["--procedure", str(procedure.resolve())]
        result = subprocess.run(cmd, cwd=str(P.ROOT), timeout=P.WORKER_TIMEOUT_S)
        if result.returncode != 0:
            raise RuntimeError(f"worker exited with code {result.returncode}")
        status_file = tmp_path / "worker_status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if not status.get("completed") or status.get("failed"):
            raise RuntimeError("worker did not complete")
        core = _load_pred(tmp_path / "core.npy", len(truth.labels), truth.k)
        metrics = {P.EXP_CORE: {"mIoU": _miou(_confusion(core, truth.labels, truth.k))}}
        low = [
            _miou(_confusion(_load_pred(tmp_path / f"low_{r}.npy", len(truth.low_gt), truth.k), truth.low_gt, truth.k))
            for r in range(truth.n_low_runs)
        ]
        metrics[P.EXP_LOW] = {"mIoU": float(np.mean(low))}
        precision = 0.0
        if method_dir is None:
            precision = truth.chance_precision()
        elif (tmp_path / "experts.npy").exists():
            ids = np.load(tmp_path / "experts.npy")
            if (ids.shape == (truth.k, P.NUM_EXPERTS) and np.issubdtype(ids.dtype, np.integer)
                    and ids.min() >= 0 and ids.max() < truth.m
                    and all(len(set(row.tolist())) == P.NUM_EXPERTS for row in ids)):
                precision = truth.expert_precision(ids)
        else:
            print(f"[eval-train] {name}: expert diagnostic scored 0 ({status.get('experts_error')})", flush=True)
        metrics[P.EXP_EXPERT] = {"expert_precision_pct": precision}
        print(f"[eval-train] {name}: low-data runs {[round(x, 2) for x in low]}", flush=True)
        return metrics


def _write(path: Path, per_method: dict[str, dict]) -> None:
    out = {"experiments": {exp: {"slice": "train", "results": {}} for exp in P.WORST}}
    for name, metrics in per_method.items():
        for exp in P.WORST:
            out["experiments"][exp]["results"][name] = metrics[exp]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=P.ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=P.ROOT / "scoring" / "scores_train.json")
    args = parser.parse_args()
    args.output.unlink(missing_ok=True)
    methods = _discover(args.outputs_root)
    try:
        truth = Truth()
    except Exception as exc:  # noqa: BLE001
        print(f"[eval-train] cannot score: {exc}", flush=True)
        _write(args.output, {name: {e: dict(v) for e, v in P.WORST.items()} for name, _ in methods})
        sys.exit(1)
    per_method: dict[str, dict] = {}
    for name, method_dir in methods:
        print(f"[eval-train] scoring {name}", flush=True)
        try:
            per_method[name] = _score(name, method_dir, truth)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval-train] {name}: FAILED ({exc}); worst value in every metric", flush=True)
            per_method[name] = {e: dict(v) for e, v in P.WORST.items()}
        print(f"[eval-train] {name}: {json.dumps(per_method[name])}", flush=True)
    _write(args.output, per_method)
    print(json.dumps(json.loads(args.output.read_text()), indent=2))


if __name__ == "__main__":
    main()
