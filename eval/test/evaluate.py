"""Hidden-test scorer for the ADE20K template-selection task.

Usage (no arguments; this is what scripts/evaluate_test.sh runs):

    python eval/test/evaluate.py

Every directory under outputs/ is a method and is scored; a directory without a
working outputs/<name>/procedure.py is scored as a failed run. The frozen
model's averaged-template classifier is scored as the built-in baseline
`template_average`. Each run of a method's procedure (the core run and every
low-data run) executes in its own fresh worker process (eval/test/worker.py) on
inputs only eval/test/ holds; while a worker runs, the hidden answers (labels,
oracle table, raw annotations) are unreadable. This process then recomputes
every number from the per-pixel class maps the worker wrote and the hidden
labels. A failed run receives the worst value in every metric of every
experiment. The previous scores file is deleted before scoring starts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402
from oracle import array_sha256, compute_oracle, file_sha256, true_experts  # noqa: E402

BUILTIN = "template_average"


def _discover(outputs_root: Path) -> list[tuple[str, Path | None]]:
    methods: list[tuple[str, Path | None]] = [(BUILTIN, None)]
    if outputs_root.is_dir():
        for path in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
            name = path.name if path.name != BUILTIN else f"{BUILTIN}@outputs"
            methods.append((name, path))
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
        missing = [p for p in (P.TEXT, P.LABELS, Path(f"{P.EVAL_PREFIX}_index.npz"),
                               Path(f"{P.POOL_PREFIX}_index.npz")) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"hidden test data missing: {missing}; run scripts/prepare_data.sh")
        if not P.HASHES.exists():
            raise RuntimeError("eval/test/label_hashes.json is missing")
        hashes = json.loads(P.HASHES.read_text())
        self.labels = np.load(P.LABELS, mmap_mode="r")
        if file_sha256(P.LABELS) != hashes.get("val_labels.npy"):
            raise RuntimeError("hidden labels do not match eval/test/label_hashes.json")
        index = np.load(f"{P.EVAL_PREFIX}_index.npz")
        self.label_hw = index["label_hw"]
        self.label_offsets = index["label_offsets"]
        if int(self.label_offsets[-1]) != len(self.labels):
            raise RuntimeError("hidden label index is inconsistent with the label file")
        self.text = np.load(P.TEXT)
        self.m, self.k, _ = self.text.shape
        self.n_pool = len(np.load(f"{P.POOL_PREFIX}_index.npz")["hw"])
        iou = np.load(P.ORACLE)["iou"] if P.ORACLE.exists() else None
        if iou is None or array_sha256(iou) != hashes.get("val_oracle_iou"):
            print("[eval] oracle table missing or modified; recomputing from verified labels", flush=True)
            iou = compute_oracle(np.load(f"{P.EVAL_PREFIX}_feats.npy", mmap_mode="r"), index,
                                 self.labels, self.text)["iou"]
        self.true_experts = true_experts(iou)  # [M, K]
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

    def per_image(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, k = len(self.label_hw), self.k
        inter = np.zeros((n, k), dtype=np.int64)
        union = np.zeros((n, k), dtype=np.int64)
        for i in range(n):
            s, e = self.label_offsets[i], self.label_offsets[i + 1]
            cm = _confusion(pred[s:e], self.labels[s:e], k)
            inter[i] = np.diag(cm)
            union[i] = cm.sum(0) + cm.sum(1) - inter[i]
        return inter, union


class _Sealed:
    """Make the hidden answers unreadable for the duration of a worker run."""

    def __enter__(self):
        self.saved = []
        for path in P.ANSWERS:
            if path.exists():
                mode = stat.S_IMODE(path.stat().st_mode)
                try:
                    os.chmod(path, 0)
                    self.saved.append((path, mode))
                except OSError as exc:
                    print(f"[eval] warning: cannot seal {path}: {exc}", flush=True)
        return self

    def __exit__(self, *exc):
        for path, mode in self.saved:
            os.chmod(path, mode)
        return False


def _unseal_stale() -> None:
    """Restore owner access to answers a killed earlier run may have left sealed."""
    for path in P.ANSWERS:
        if path.exists():
            need = 0o770 if path.is_dir() else 0o660
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & need != need:
                os.chmod(path, mode | need)


def _worker_run(cmd: list[str], out: Path, deadline: float) -> dict:
    """One worker process for one run; retried once if it fails (e.g. a transient GPU error)."""
    for attempt in (1, 2):
        for f in out.iterdir():
            shutil.rmtree(f) if f.is_dir() and not f.is_symlink() else f.unlink()
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("method exceeded its time budget")
        try:
            with _Sealed():
                result = subprocess.run(cmd, cwd=str(P.ROOT), timeout=remaining)
            status_file = out / "worker_status.json"
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            if result.returncode == 0 and status.get("completed") and not status.get("failed"):
                return status
            error = f"worker exited with code {result.returncode}"
        except subprocess.TimeoutExpired:
            raise TimeoutError("method exceeded its time budget") from None
        if attempt == 1:
            print(f"[eval] {error}; retrying once", flush=True)
    raise RuntimeError(error)


def _score(name: str, method_dir: Path | None, truth: Truth, details_dir: Path | None) -> dict:
    scratch = P.ROOT / "tmp" / "eval_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(HERE / "worker.py")]
    if method_dir is None:
        base.append("--builtin")
    else:
        procedure = method_dir / "procedure.py"
        if not procedure.is_file():
            raise FileNotFoundError(f"{method_dir} has no procedure.py")
        base += ["--procedure", str(procedure.resolve())]
    deadline = time.time() + P.WORKER_TIMEOUT_S
    core = None
    status: dict = {}
    ids = None
    low = []
    for run in range(1 + truth.n_low_runs):  # run 0: core; runs 1..n: low-data
        with tempfile.TemporaryDirectory(prefix="test_", dir=scratch) as tmp:
            tmp_path = Path(tmp)
            st = _worker_run(base + ["--out", str(tmp_path), "--run", str(run)], tmp_path, deadline)
            if run == 0:
                status = st
                core = np.array(_load_pred(tmp_path / "pred.npy", len(truth.labels), truth.k))
                if (tmp_path / "experts.npy").exists():
                    ids = np.load(tmp_path / "experts.npy")
            else:
                pred = _load_pred(tmp_path / "pred.npy", len(truth.low_gt), truth.k)
                low.append(_miou(_confusion(pred, truth.low_gt, truth.k)))
    metrics = {P.EXP_CORE: {"mIoU": _miou(_confusion(core, truth.labels, truth.k))}}
    metrics[P.EXP_LOW] = {"mIoU": float(np.mean(low))}
    precision = 0.0
    if method_dir is None:
        precision = truth.chance_precision()
    elif ids is not None:
        if (ids.shape == (truth.k, P.NUM_EXPERTS) and np.issubdtype(ids.dtype, np.integer)
                and ids.min() >= 0 and ids.max() < truth.m
                and all(len(set(row.tolist())) == P.NUM_EXPERTS for row in ids)):
            precision = truth.expert_precision(ids)
    else:
        print(f"[eval] {name}: expert diagnostic scored 0 ({status.get('experts_error')})", flush=True)
    metrics[P.EXP_EXPERT] = {"expert_precision_pct": precision}
    print(f"[eval] {name}: low-data runs {[round(x, 2) for x in low]}", flush=True)
    if details_dir is not None:
        inter, union = truth.per_image(core)
        details_dir.mkdir(parents=True, exist_ok=True)
        np.savez(details_dir / f"{name}.npz", inter=inter, union=union, low_miou=np.array(low))
    return metrics


def _write(path: Path, per_method: dict[str, dict]) -> None:
    out = {"experiments": {exp: {"slice": "test", "results": {}} for exp in P.WORST}}
    for name, metrics in per_method.items():
        for exp in P.WORST:
            out["experiments"][exp]["results"][name] = metrics[exp]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=P.ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=P.ROOT / "scoring" / "scores_test.json")
    parser.add_argument("--details-dir", type=Path, default=None, help="optional per-image diagnostics")
    parser.add_argument("--skip-builtin", action="store_true", help="tooling: do not score template_average")
    args = parser.parse_args()
    args.output.unlink(missing_ok=True)
    methods = _discover(args.outputs_root)
    if args.skip_builtin:
        methods = [(n, d) for n, d in methods if d is not None]
    worst = {exp: dict(v) for exp, v in P.WORST.items()}
    try:
        _unseal_stale()
        truth = Truth()
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] cannot score: {exc}", flush=True)
        _write(args.output, {name: worst for name, _ in methods})
        sys.exit(1)
    per_method: dict[str, dict] = {}
    for name, method_dir in methods:
        print(f"[eval] scoring {name}", flush=True)
        try:
            per_method[name] = _score(name, method_dir, truth, args.details_dir)
        except Exception as exc:  # noqa: BLE001 - any failure is a failed run
            print(f"[eval] {name}: FAILED ({exc}); worst value in every metric", flush=True)
            per_method[name] = {exp: dict(v) for exp, v in P.WORST.items()}
        print(f"[eval] {name}: {json.dumps(per_method[name])}", flush=True)
    _write(args.output, per_method)
    print(json.dumps(json.loads(args.output.read_text()), indent=2))


if __name__ == "__main__":
    main()
