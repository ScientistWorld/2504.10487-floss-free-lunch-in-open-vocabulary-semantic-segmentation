"""SHORTCUT probe: class-experts chosen with dev-set LABELS (top-4 single-template IoU),
hard-coded by canonical template/class index, then FLOSS fusion. Ignores the adaptation images."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import VISIBLE, floss_state, load_floss, oracle_top4  # noqa: E402

floss = load_floss()
CANON = oracle_top4(VISIBLE / "dev_oracle.npz")
predict, experts = floss.predict, floss.experts


def adapt(features_list, text_embeddings):
    return floss_state(floss, text_embeddings, CANON)  # assumes the canonical order
