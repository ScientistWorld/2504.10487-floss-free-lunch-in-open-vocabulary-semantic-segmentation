"""SHORTCUT probe: FLOSS, but adapt() ignores the images it is given and reads the whole visible
unlabeled pool from disk instead (defeats the low-data setting)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import VISIBLE, load_floss  # noqa: E402

floss = load_floss()
predict, experts = floss.predict, floss.experts


def adapt(features_list, text_embeddings):
    feats = np.load(VISIBLE / "adapt_feats.npy", mmap_mode="r")
    index = np.load(VISIBLE / "adapt_index.npz")
    pool = [feats[index["offsets"][i] : index["offsets"][i + 1]].reshape(int(h), int(w), -1)
            for i, (h, w) in enumerate(index["hw"])]
    return floss.adapt(pool, text_embeddings)
