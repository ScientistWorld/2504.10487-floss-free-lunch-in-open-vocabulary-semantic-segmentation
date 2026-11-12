"""UPPER BOUND (not a method): the paper's oracle experiment -- the top-4 true class-experts by
labelled IoU on the hidden validation set itself, with FLOSS fusion. Reads hidden test answers."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, floss_state, load_floss, oracle_top4, recover_permutation  # noqa: E402

floss = load_floss()
_hidden = ROOT / "eval" / "test" / "data" / "naclip" / "val_oracle.npz"
CANON = oracle_top4(_hidden if _hidden.exists() else ROOT / "data" / "downloads" / "naclip" / "dev_oracle.npz")
predict, experts = floss.predict, floss.experts


def adapt(features_list, text_embeddings):
    text = np.asarray(text_embeddings, dtype=np.float32)
    perm_t, perm_c = recover_permutation(text)
    shown_of_canon_t = np.argsort(perm_t)
    sel = np.stack([shown_of_canon_t[CANON[perm_c[j]]] for j in range(text.shape[1])])
    return floss_state(floss, text, sel)
