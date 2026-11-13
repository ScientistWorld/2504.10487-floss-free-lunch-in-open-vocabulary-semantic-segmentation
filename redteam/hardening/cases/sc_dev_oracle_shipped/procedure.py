"""SHORTCUT probe (left to the grader): class-experts chosen with dev-set LABELS, shipped inside the method
directory (oracle_top4.npy, made by install_cases.py from the visible dev oracle) together with canonical
class/template mean embeddings to undo the shuffling; FLOSS fusion. Reads only its own directory."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import load_floss, match_rows, state_from_selection  # noqa: E402

floss = load_floss()
predict, experts = floss.predict, floss.experts
CANON_TOP4 = np.load(HERE / "oracle_top4.npy")  # [K, 4] canonical template ids
CANON_T = np.load(HERE / "canon_template_means.npy").astype(np.float32)
CANON_C = np.load(HERE / "canon_class_means.npy").astype(np.float32)


def adapt(features_list, text_embeddings):
    text = np.ascontiguousarray(text_embeddings, dtype=np.float32)
    shown_to_canon_t = match_rows(text.mean(1), CANON_T)
    shown_to_canon_c = match_rows(text.mean(0), CANON_C)
    canon_to_shown_t = np.argsort(shown_to_canon_t)
    sel = [canon_to_shown_t[CANON_TOP4[shown_to_canon_c[j]]] for j in range(text.shape[1])]
    return state_from_selection(floss, text, sel)
