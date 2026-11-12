"""SHORTCUT probe (stale-file analogue): adapts, then crashes during prediction."""
import numpy as np


def adapt(features_list, text_embeddings):
    return np.asarray(text_embeddings).shape


def predict(state, features):
    raise RuntimeError("deliberate failure")
