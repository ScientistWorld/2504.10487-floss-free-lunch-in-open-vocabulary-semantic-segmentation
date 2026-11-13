"""Run one method's procedure for ONE run of the visible (train-slice) protocol.

Same interface and isolation as the hidden test's worker, on method-visible
data: the parent evaluator (evaluate.py) launches a fresh worker for every run.
Run 0 is the core run (adapt on the whole visible adaptation pool, predict
every dev image, report experts); runs 1..12 are the low-data runs. The worker
loads `procedure.py`, calls

    state  = adapt(features_list, text_embeddings)
    scores = predict(state, features)          # per image, [h, w, K]
    ids    = experts(state)                    # core run only, optional, [K, 4]

with template and class order shuffled per run, and writes per-pixel class maps
(uint8) to --out. It never sees labels and never computes a score.

Before the procedure is imported, an audit hook restricts the process: the
procedure may read files only inside its own directory (plus Python source
files), may write only inside this run's --out directory, and may not start
other processes. Everything else under the workspace is off limits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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

_SPAWN_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "os.forkpty", "os.startfile", "pty.spawn",
})
_MUTATE_EVENTS = frozenset({
    "os.chmod", "os.chown", "os.chflags", "os.lchflags", "os.rename", "os.remove", "os.rmdir",
    "os.mkdir", "os.symlink", "os.link", "os.truncate", "os.utime", "os.setxattr", "os.removexattr",
    "os.mkfifo", "os.mknod", "shutil.rmtree", "shutil.move", "shutil.copyfile", "shutil.copymode",
    "shutil.copystat", "shutil.copytree", "shutil.chown", "shutil.make_archive", "shutil.unpack_archive",
})
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND


def _install_guard(root: Path, proc_dir: Path | None, out_dir: Path) -> None:
    """Restrict file access and process creation for the rest of this process."""
    root_s = os.path.realpath(root)
    out_s = os.path.realpath(out_dir)
    readable = [out_s] + ([os.path.realpath(proc_dir)] if proc_dir is not None else [])

    def inside(path: str, base: str) -> bool:
        return path == base or path.startswith(base + os.sep)

    def real(obj) -> str | None:
        if isinstance(obj, int) or obj is None:
            return None
        try:
            return os.path.realpath(os.fsdecode(obj))
        except (TypeError, ValueError):
            return None

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path = real(args[0])
            if path is None:
                return
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
            if writing:
                if inside(path, out_s) or path == os.devnull:
                    return
                raise PermissionError(f"evaluator: a procedure may not write {path}")
            if inside(path, root_s) and not path.endswith(".py") and not any(inside(path, b) for b in readable):
                raise PermissionError(f"evaluator: a procedure may only read files in its own directory, not {path}")
        elif event in _MUTATE_EVENTS:
            for a in args:
                path = real(a) if isinstance(a, (str, bytes, os.PathLike)) else None
                if path is not None and not inside(path, out_s):
                    raise PermissionError(f"evaluator: {event} on {path} is not allowed")
        elif event in _SPAWN_EVENTS:
            raise PermissionError(f"evaluator: a procedure may not start processes ({event})")

    sys.dont_write_bytecode = True
    sys.addaudithook(hook)


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
    parser.add_argument("--run", type=int, required=True, help="0 = core run, r >= 1 = low-data run r")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--procedure", type=Path)
    group.add_argument("--builtin", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status: dict = {}
    out_dir = args.out.resolve()
    try:
        text = np.load(P.TEXT)
        pool, evals = _Split(P.POOL_PREFIX), _Split(P.EVAL_PREFIX)
        runs = P.low_data_runs(len(pool))
        if not 0 <= args.run <= len(runs):
            raise ValueError(f"run index {args.run} out of range")
        if args.run == 0:
            adapt_ids, eval_ids = np.arange(len(pool)), np.arange(len(evals))
        else:
            adapt_ids, eval_ids = runs[args.run - 1]["adapt_ids"], P.low_eval_ids(len(evals))
        torch.zeros(1, device=device)  # initialise the device before the guard
        proc_dir = None if args.builtin else args.procedure.resolve().parent
        _install_guard(P.ROOT, proc_dir, out_dir)
        proc = _TemplateAverage() if args.builtin else _load_procedure(args.procedure.resolve())
        t0 = time.time()
        experts = _run(proc, args.run, text, pool, adapt_ids, evals, eval_ids, out_dir / "pred.npy",
                       args.run == 0 and not args.builtin, device)
        if experts is not None:
            ids, err = experts
            if ids is not None:
                np.save(out_dir / "experts.npy", ids)
            status["experts_error"] = err
        print(f"[worker] run {args.run} done in {time.time() - t0:.0f}s", flush=True)
    except BaseException:  # noqa: BLE001 - any failure (including sys.exit) fails the run
        traceback.print_exc()
        try:
            (out_dir / "worker_status.json").write_text(json.dumps({"failed": True}))
        finally:
            os._exit(1)
    status["completed"] = True
    (out_dir / "worker_status.json").write_text(json.dumps(status))


if __name__ == "__main__":
    main()
