"""FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation (Benigmim et al.).

Gym procedure (installed as outputs/<name>/procedure.py by scripts/method.sh):

    adapt(features_list, text_embeddings) -> state
        Unsupervised identification of class-experts (paper Sec. 4.1): every
        single-template classifier W(T_m) labels the unlabeled adaptation
        patches; for each class k the mean softmax entropy H_{m,k} over the
        patches W(T_m) assigns to k is computed (Eq. 5); the N templates with
        the lowest entropy among those predicting k at least once become the
        class-experts (Eq. 6), and the expert classifier W(E_k) averages their
        embeddings for all classes.
    predict(state, features) -> [h, w, K] probabilities
        Expert fusion (Sec. 4.2, Eq. 7): at each location, among the experts
        that predict their own class, the most confident one decides; if none
        does, the averaged-template classifier W_CLIP decides.
    experts(state) -> [K, 4] template ids (the selected class-experts)

Defaults reproduce the paper (N = 4, entropy, "Highest" fusion, CLIP logit
scale 100 as in the official NACLIP integration). An optional config.json next
to the installed file overrides them; the alternatives below exist only for
the paper's ablations and for the gym's variant checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CONFIG = {
    "top_n": 4,  # experts per class (paper N = 4)
    "logit_scale": 100.0,  # softmax temperature on cosine logits
    "selection": "entropy",  # entropy (paper) | avg_prob | random
    "region": "own",  # own (paper): patches the template predicts as k | default_region: patches W_CLIP predicts as k
    "min_support": 0.0,  # 0 (paper: >0 predicted patches); else fraction of the class's median valid count
    "min_count": 0,  # 0 (paper); else a template needs >= this many predicted patches to be a candidate expert
    "fusion": "highest",  # highest (paper) | default_conflict | average_conflict | average_all
    "shrink": 0.0,  # 0 (paper); >0 adds the averaged class embedding to each expert classifier
    "pool_fraction": 1.0,  # <1 adapts on a random subset of the given images (sensitivity runs)
    "seed": 0,
    "chunk": 4096,  # patches per batch when scoring templates
}
_CONFIG_FILE = Path(__file__).with_name("config.json")
if _CONFIG_FILE.exists():
    CONFIG.update(json.loads(_CONFIG_FILE.read_text()))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _template_statistics(features_list, text: np.ndarray):
    """Per (template, class): patch count, mean entropy and mean class probability."""
    m, k, d = text.shape
    weights = torch.from_numpy(text.reshape(m * k, d)).to(DEVICE).T.contiguous()  # [D, M*K]
    avg = _norm(torch.from_numpy(text).to(DEVICE).mean(0))  # [K, D]
    offs = torch.arange(m, device=DEVICE) * k
    count = torch.zeros(m * k, dtype=torch.float64, device=DEVICE)
    ent_sum = torch.zeros_like(count)
    prob_sum = torch.zeros_like(count)
    scale = float(CONFIG["logit_scale"])

    def consume(buf: list[np.ndarray]) -> None:
        f = torch.from_numpy(np.concatenate(buf).astype(np.float32)).to(DEVICE)
        logits = (f @ weights).view(-1, m, k) * scale
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)  # [P, M]
        if CONFIG["region"] == "default_region":
            target = (f @ avg.T).argmax(-1)  # [P]
            prob = p.gather(-1, target[:, None, None].expand(-1, m, 1)).squeeze(-1)
            idx = target[:, None] + offs[None, :]
        else:
            prob, pred = p.max(-1)  # [P, M]
            idx = pred + offs[None, :]
        idx = idx.reshape(-1)
        count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))
        ent_sum.index_add_(0, idx, ent.reshape(-1).double())
        prob_sum.index_add_(0, idx, prob.reshape(-1).double())

    buf: list[np.ndarray] = []
    size = 0
    for feats in features_list:
        arr = np.asarray(feats).reshape(-1, d)
        buf.append(arr)
        size += len(arr)
        if size >= CONFIG["chunk"]:
            consume(buf)
            buf, size = [], 0
    if buf:
        consume(buf)
    count_np = count.view(m, k).cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        entropy = (ent_sum.view(m, k).cpu().numpy() / count_np)
        avg_prob = (prob_sum.view(m, k).cpu().numpy() / count_np)
    return count_np, entropy, avg_prob


def _rank(count: np.ndarray, entropy: np.ndarray, avg_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Templates sorted per class (best first) and the [M, K] mask of eligible templates."""
    m, k = count.shape
    valid = count >= max(1, int(CONFIG["min_count"]))  # paper: the template predicts the class at least once
    if CONFIG["selection"] == "random":
        rng = np.random.default_rng(CONFIG["seed"])
        return np.argsort(rng.random((m, k)), axis=0, kind="stable"), np.ones((m, k), dtype=bool)
    if CONFIG["min_support"] > 0:
        for c in range(k):
            if valid[:, c].any():
                valid[:, c] &= count[:, c] >= CONFIG["min_support"] * np.median(count[valid[:, c], c])
    key = entropy if CONFIG["selection"] == "entropy" else -avg_prob
    key = np.where(valid, key, np.inf)
    return np.argsort(key, axis=0, kind="stable"), valid


@torch.no_grad()
def adapt(features_list, text_embeddings) -> dict:
    text = np.ascontiguousarray(text_embeddings, dtype=np.float32)
    m, k, d = text.shape
    feats = list(features_list)
    if CONFIG["pool_fraction"] < 1.0:
        rng = np.random.default_rng(CONFIG["seed"])
        n = max(1, int(round(len(feats) * CONFIG["pool_fraction"])))
        feats = [feats[i] for i in np.sort(rng.choice(len(feats), size=n, replace=False))]
    count, entropy, avg_prob = _template_statistics(feats, text)
    order, valid = _rank(count, entropy, avg_prob)
    t = torch.from_numpy(text).to(DEVICE)
    avg_emb = _norm(t.mean(0))  # W_CLIP: [K, D]
    expert_cls = torch.empty(k, k, d, device=DEVICE)
    chosen = []
    for c in range(k):
        sel = order[: min(int(CONFIG["top_n"]), int(valid[:, c].sum())), c]
        chosen.append(sel)
        if len(sel) == 0:  # no template ever predicts c: fall back to W_CLIP for this class
            expert_cls[c] = avg_emb
        else:
            emb = _norm(t[torch.from_numpy(sel).to(DEVICE)].mean(0))  # W(E_c): [K, D]
            expert_cls[c] = _norm(emb + float(CONFIG["shrink"]) * avg_emb)
    return {"default": avg_emb, "experts": expert_cls, "order": order, "chosen": chosen}


@torch.no_grad()
def predict(state: dict, features) -> np.ndarray:
    h, w, d = features.shape
    f = torch.from_numpy(np.asarray(features, dtype=np.float32).reshape(-1, d)).to(DEVICE)
    scale = float(CONFIG["logit_scale"])
    probs_default = torch.softmax((f @ state["default"].T) * scale, dim=-1)  # [P, K]
    probs = torch.softmax(torch.einsum("pd,ecd->epc", f, state["experts"]) * scale, dim=-1)  # [K, P, K]
    k, n = probs.shape[0], probs.shape[1]
    classes = torch.arange(k, device=DEVICE)
    own = probs.argmax(-1) == classes[:, None]  # expert e predicts its own class: [K, P]
    conf = probs[classes, :, classes]  # [K, P]
    n_own = own.sum(0)
    fusion = CONFIG["fusion"]
    if fusion == "highest":
        best = torch.where(own, conf, torch.full_like(conf, -1.0)).argmax(0)
        out = torch.where((n_own > 0)[:, None], probs[best, torch.arange(n, device=DEVICE)], probs_default)
    elif fusion == "default_conflict":
        best = own.float().argmax(0)
        out = torch.where((n_own == 1)[:, None], probs[best, torch.arange(n, device=DEVICE)], probs_default)
    elif fusion == "average_conflict":
        mean = (probs * own.float()[..., None]).sum(0) / n_own.clamp_min(1)[:, None]
        out = torch.where((n_own > 0)[:, None], mean, probs_default)
    elif fusion == "average_all":
        out = probs.mean(0)
    else:
        raise ValueError(f"unknown fusion {fusion}")
    return out.view(h, w, k).cpu().numpy()


def experts(state: dict) -> np.ndarray:
    """Top-4 ranked templates per class (the class-experts used when N = 4)."""
    return np.ascontiguousarray(state["order"][:4].T).astype(np.int64)
