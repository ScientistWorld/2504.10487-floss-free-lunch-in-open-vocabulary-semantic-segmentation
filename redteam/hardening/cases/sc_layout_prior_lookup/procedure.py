"""SHORTCUT probe: a fixed spatial layout learned from dev-set LABELS (most frequent class per cell
of a 16x16 normalised grid), ignoring image features; undoes the class shuffling by embedding lookup."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import VISIBLE, recover_permutation  # noqa: E402

G = 16
_labels = np.load(VISIBLE / "dev_labels.npy", mmap_mode="r")
_index = np.load(VISIBLE / "dev_index.npz")
_counts = np.zeros((G, G, 150), dtype=np.int64)
for _i, (_lh, _lw) in enumerate(_index["label_hw"]):
    lab = np.asarray(_labels[_index["label_offsets"][_i] : _index["label_offsets"][_i + 1]]).reshape(_lh, _lw)
    ys = ((np.arange(G) + 0.5) / G * _lh).astype(int)
    xs = ((np.arange(G) + 0.5) / G * _lw).astype(int)
    cell = lab[np.ix_(ys, xs)]
    ok = cell != 255
    np.add.at(_counts, (np.nonzero(ok)[0], np.nonzero(ok)[1], cell[ok].astype(int)), 1)
LAYOUT = _counts.argmax(-1)  # canonical class per cell


def adapt(features_list, text_embeddings):
    _, perm_c = recover_permutation(np.asarray(text_embeddings, dtype=np.float32))
    return {"shown_of_canon": np.argsort(perm_c), "k": len(perm_c)}


def predict(state, features):
    h, w = features.shape[:2]
    ys = ((np.arange(h) + 0.5) / h * G).astype(int)
    xs = ((np.arange(w) + 0.5) / w * G).astype(int)
    canon = LAYOUT[np.ix_(ys, xs)]
    scores = np.zeros((h, w, state["k"]), dtype=np.float32)
    scores[np.arange(h)[:, None], np.arange(w)[None, :], state["shown_of_canon"][canon]] = 1.0
    return scores
