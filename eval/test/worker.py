"""Run one method's procedure on the hidden test inputs (separate process).

The parent evaluator (evaluate.py) launches this worker once per method. The
worker loads `procedure.py`, calls

    state  = adapt(features_list, text_embeddings)
    scores = predict(state, features)          # per image, [h, w, K]
    ids    = experts(state)                    # optional, [K, 4] template ids

on the hidden adaptation pool and validation images, with template and class
order shuffled per run, and writes per-pixel class maps (uint8) to --out. It
never sees labels and never computes a score; the parent does that.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P  # noqa: E402


class _TemplateAverage:
    """Built-in baseline: the frozen model's default averaged-template classifier."""

    @staticmethod
    def adapt(features_list, text_embeddings):
        cls = np.asarray(text_embeddings, dtype=np.float32).mean(axis=0)
        return cls / np.linalg.norm(cls, axis=-1, keepdims=True)

    @staticmethod
    def predict(state, features):
        return np.asarray(features, dtype=np.float32) @ state.T


def _load_procedure(path: Path):
    spec = importlib.util.spec_from_file_location("method_procedure", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["method_procedure"] = module
    sys.path.insert(0, str(path.parent))  # helper files shipped next to procedure.py
    spec.loader.exec_module(module)
    for name in ("adapt", "predict"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"{path} does not define {name}()")
    return module


class _Split:
    def __init__(self, prefix: Path):
        self.feats = np.load(f"{prefix}_feats.npy", mmap_mode="r")
        index = np.load(f"{prefix}_index.npz")
        self.hw = index["hw"]
        self.offsets = index["offsets"]
        self.label_hw = index["label_hw"]

    def __len__(self) -> int:
        return len(self.hw)

    def image(self, i: int) -> np.ndarray:
        h, w = (int(v) for v in self.hw[i])
        return self.feats[self.offsets[i] : self.offsets[i + 1]].reshape(h, w, -1)


def _as_scores(raw, h: int, w: int, k: int) -> np.ndarray:
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().float().cpu().numpy()
    scores = np.asarray(raw, dtype=np.float32)
    if scores.shape != (h, w, k):
        raise ValueError(f"predict() returned shape {scores.shape}, expected {(h, w, k)}")
    if not np.isfinite(scores).all():
        raise ValueError("predict() returned non-finite scores")
    return scores


def _experts(proc, state, perm_t: np.ndarray, perm_c: np.ndarray, m: int, k: int):
    fn = getattr(proc, "experts", None)
    if not callable(fn):
        return None, "procedure defines no experts()"
    try:
        raw = fn(state)
    except Exception as exc:  # noqa: BLE001 - only the expert diagnostic fails
        return None, f"experts() raised {exc!r}"
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()
    ids = np.asarray(raw)
    if ids.shape != (k, P.NUM_EXPERTS) or not np.issubdtype(ids.dtype, np.integer):
        return None, f"experts() must return an integer array of shape {(k, P.NUM_EXPERTS)}, got {ids.shape} {ids.dtype}"
    if ids.min() < 0 or ids.max() >= m:
        return None, "experts() returned template ids out of range"
    if any(len(set(row.tolist())) != P.NUM_EXPERTS for row in ids):
        return None, "experts() returned duplicate template ids within a class"
    original = np.empty_like(ids, dtype=np.int64)
    original[perm_c] = perm_t[ids]
    return original, None


def _run(proc, run_index, text, pool, adapt_ids, evals, eval_ids, out_path, want_experts, device):
    m, k, _ = text.shape
    perm_t, perm_c = P.permutation(run_index, m, k)
    text_p = np.ascontiguousarray(text[perm_t][:, perm_c])
    state = proc.adapt([pool.image(int(i)) for i in adapt_ids], text_p)
    experts = None
    if want_experts:
        experts = _experts(proc, state, perm_t, perm_c, m, k)
    sizes = [int(evals.label_hw[i][0] * evals.label_hw[i][1]) for i in eval_ids]
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(sum(sizes),))
    pos = 0
    for i, size in zip(eval_ids, sizes):
        feats = evals.image(int(i))
        h, w = feats.shape[:2]
        scores = _as_scores(proc.predict(state, feats), h, w, k)
        original = np.empty_like(scores)
        original[..., perm_c] = scores
        lh, lw = (int(v) for v in evals.label_hw[i])
        t = torch.from_numpy(original).to(device).permute(2, 0, 1)[None]
        lab = F.interpolate(t, size=(lh, lw), mode="bilinear", align_corners=False)[0].argmax(0)
        out[pos : pos + size] = lab.to(torch.uint8).cpu().numpy().reshape(-1)
        pos += size
    out.flush()
    del out
    return experts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--procedure", type=Path)
    group.add_argument("--builtin", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status: dict = {}
    try:
        proc = _TemplateAverage() if args.builtin else _load_procedure(args.procedure)
        text = np.load(P.TEXT)
        pool, evals = _Split(P.POOL_PREFIX), _Split(P.EVAL_PREFIX)
        t0 = time.time()
        experts = _run(proc, 0, text, pool, np.arange(len(pool)), evals, np.arange(len(evals)),
                       args.out / "core.npy", not args.builtin, device)
        if experts is not None:
            ids, err = experts
            if ids is not None:
                np.save(args.out / "experts.npy", ids)
            status["experts_error"] = err
        print(f"[worker] core done in {time.time() - t0:.0f}s", flush=True)
        low_ids = P.low_eval_ids(len(evals))
        for r, run in enumerate(P.low_data_runs(len(pool))):
            _run(proc, 1 + r, text, pool, run["adapt_ids"], evals, low_ids, args.out / f"low_{r}.npy", False, device)
        print(f"[worker] low-data done in {time.time() - t0:.0f}s", flush=True)
    except BaseException:  # noqa: BLE001 - any failure (including sys.exit) fails the run
        traceback.print_exc()
        (args.out / "worker_status.json").write_text(json.dumps({"failed": True}))
        sys.exit(1)
    status["completed"] = True
    (args.out / "worker_status.json").write_text(json.dumps(status))


if __name__ == "__main__":
    main()
