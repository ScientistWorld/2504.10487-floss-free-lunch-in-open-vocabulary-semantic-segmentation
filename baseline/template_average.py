"""The paper's baseline as a gym procedure: CLIP's averaged-template classifier.

Both evaluators already score this baseline as the built-in `template_average`
entry; this file is the smallest complete example of the procedure interface
(see briefing/evaluation.md). Copy it to outputs/<name>/procedure.py to run it
as a method.
"""

from __future__ import annotations

import numpy as np


def adapt(features_list, text_embeddings):
    """Ignores the unlabeled images; averages the M template embeddings per class."""
    cls = np.asarray(text_embeddings, dtype=np.float32).mean(axis=0)  # [K, D]
    return cls / np.linalg.norm(cls, axis=-1, keepdims=True)


def predict(state, features):
    """[h, w, D] patch features -> [h, w, K] class scores (cosine logits)."""
    return np.asarray(features, dtype=np.float32) @ state.T
