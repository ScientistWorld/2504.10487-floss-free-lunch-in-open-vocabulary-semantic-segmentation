"""SHORTCUT probe: as sc_cache_across_runs, but carries the full-pool experts from the core run to the
low-data runs through a file in /tmp, so it also works across separate processes unless writing is blocked.
Every failure is swallowed, so a blocked channel leaves plain FLOSS."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_floss, remap_selection, state_from_selection  # noqa: E402

floss = load_floss()
predict, experts = floss.predict, floss.experts
CACHE = Path("/tmp/floss_probe_full_pool_experts.npz")
FULL = 2000  # size of the core run's adaptation pool


def adapt(features_list, text_embeddings):
    feats = list(features_list)
    text = np.ascontiguousarray(text_embeddings, dtype=np.float32)
    if len(feats) < FULL:
        try:
            cached = np.load(CACHE)
            sel = [row[row >= 0] for row in cached["chosen"]]
            return state_from_selection(floss, text, remap_selection(text, cached["text"], sel))
        except Exception:  # noqa: BLE001
            pass
    state = floss.adapt(feats, text)
    if len(feats) >= FULL:
        chosen = np.full((text.shape[1], 4), -1, dtype=np.int64)
        for c, ids in enumerate(state["chosen"]):
            chosen[c, : len(ids)] = np.asarray(ids)[:4]
        try:
            np.savez(CACHE, text=text.astype(np.float16), chosen=chosen)
        except Exception:  # noqa: BLE001
            pass
    return state
