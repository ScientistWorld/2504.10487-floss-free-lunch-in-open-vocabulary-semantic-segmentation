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


def state_from_selection(floss, text: np.ndarray, sel) -> dict:
    """FLOSS state for per-class expert id lists of any length (shown index space); empty -> W_CLIP."""
    m, k, d = text.shape
    t = torch.from_numpy(np.ascontiguousarray(text, dtype=np.float32)).to(floss.DEVICE)
    avg = floss._norm(t.mean(0))
    experts = torch.empty(k, k, d, device=floss.DEVICE)
    order = np.zeros((m, k), dtype=np.int64)
    for c in range(k):
        s = np.asarray(sel[c], dtype=np.int64)
        experts[c] = avg if len(s) == 0 else floss._norm(t[torch.from_numpy(s).to(floss.DEVICE)].mean(0))
        taken = set(s.tolist())
        order[:, c] = np.concatenate([s, [i for i in range(m) if i not in taken]])
    return {"default": avg, "experts": experts, "order": order, "chosen": [np.asarray(x) for x in sel]}


def match_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every row of a, the index of the most similar row of b (cosine)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a @ b.T).argmax(1)


def remap_selection(text: np.ndarray, cached_text: np.ndarray, cached_sel) -> list:
    """Map per-class expert ids chosen under one template/class order to the order of `text`."""
    cur_to_old_t = match_rows(text.mean(1), cached_text.mean(1))
    cur_to_old_c = match_rows(text.mean(0), cached_text.mean(0))
    old_to_cur_t = np.argsort(cur_to_old_t)
    return [old_to_cur_t[np.asarray(cached_sel[cur_to_old_c[j]], dtype=np.int64)] for j in range(text.shape[1])]
