"""Shared helpers for the hardening shortcut probes (copied next to each probe)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()  # evaluator workers run from the workspace root
VISIBLE = ROOT / "data" / "downloads" / "naclip"


def load_floss():
    spec = importlib.util.spec_from_file_location("floss_reference_copy", ROOT / "method" / "floss.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recover_permutation(text: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undo the evaluator's anonymisation by matching against the visible template pool.

    Returns (perm_t, perm_c): shown template i is canonical perm_t[i], shown class j is canonical perm_c[j].
    """
    canon = np.load(VISIBLE / "text_embeddings.npy")

    def match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        return (a @ b.T).argmax(1)

    return match(text.mean(1), canon.mean(1)), match(text.mean(0), canon.mean(0))


def floss_state(floss, text: np.ndarray, sel: np.ndarray) -> dict:
    """FLOSS expert classifiers / fusion state for given per-class expert ids [K, N] (shown index space)."""
    m, k, _ = text.shape
    t = torch.from_numpy(np.ascontiguousarray(text, dtype=np.float32)).to(floss.DEVICE)
    avg = floss._norm(t.mean(0))
    experts = torch.stack([floss._norm(t[torch.from_numpy(sel[c]).to(floss.DEVICE)].mean(0)) for c in range(k)])
    order = np.zeros((m, k), dtype=np.int64)
    order[: sel.shape[1]] = sel.T
    return {"default": avg, "experts": experts, "order": order, "chosen": list(sel)}


def oracle_top4(path: Path) -> np.ndarray:
    """[K, 4] canonical template ids with the highest labelled single-template class IoU."""
    iou = np.load(path)["iou"][:-1]
    return np.argsort(-np.nan_to_num(iou, nan=-1.0), axis=0, kind="stable")[:4].T
