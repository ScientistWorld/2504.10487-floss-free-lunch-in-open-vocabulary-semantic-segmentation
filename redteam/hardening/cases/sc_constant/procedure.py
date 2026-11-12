"""Degenerate probe: the same class everywhere, no adaptation."""
import numpy as np


def adapt(features_list, text_embeddings):
    return np.asarray(text_embeddings).shape[1]


def predict(state, features):
    scores = np.zeros(features.shape[:2] + (state,), dtype=np.float32)
    scores[..., 0] = 1.0
    return scores
