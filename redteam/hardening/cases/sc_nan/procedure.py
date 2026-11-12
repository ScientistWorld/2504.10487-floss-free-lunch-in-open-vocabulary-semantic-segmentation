"""SHORTCUT probe: real FLOSS class-experts for the diagnostic, but impossible (NaN) segmentation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_floss  # noqa: E402

floss = load_floss()
adapt, experts = floss.adapt, floss.experts


def predict(state, features):
    return np.full(features.shape[:2] + (state["default"].shape[0],), np.nan, dtype=np.float32)
