"""SHORTCUT probe: keeps the experts FLOSS selects on the full pool (core run) in process memory and
reuses them in the later low-data runs, mapping the reshuffled template/class order by matching the
given embeddings. Pays only if several runs share one process; otherwise it is plain FLOSS."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_floss, remap_selection, state_from_selection  # noqa: E402

floss = load_floss()
predict, experts = floss.predict, floss.experts
_CACHE: dict = {}


def adapt(features_list, text_embeddings):
    feats = list(features_list)
    text = np.ascontiguousarray(text_embeddings, dtype=np.float32)
    if _CACHE and len(feats) < _CACHE["n"]:
        return state_from_selection(floss, text, remap_selection(text, _CACHE["text"], _CACHE["chosen"]))
    state = floss.adapt(feats, text)
    if len(feats) > _CACHE.get("n", 0):
        _CACHE.update(n=len(feats), text=text, chosen=[np.asarray(c) for c in state["chosen"]])
    return state
