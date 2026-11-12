"""SHORTCUT probe: reference FLOSS segmentation, but reports an empty expert set (zero budget)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_floss  # noqa: E402

floss = load_floss()
adapt, predict = floss.adapt, floss.predict


def experts(state):
    return np.zeros((state["default"].shape[0], 0), dtype=np.int64)
