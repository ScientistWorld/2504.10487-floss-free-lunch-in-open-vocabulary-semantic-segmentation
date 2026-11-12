"""Fixed visible-slice (train) protocol: inputs, splits, seeds and operating points.

Mirrors the hidden test protocol on method-visible data: adaptation on the
visible unlabeled pool (data/downloads/naclip/adapt_*) and scoring on the
500-image labeled dev set (data/downloads/naclip/dev_*), with its own seeds
and template/class permutations.

The method artifact itself (outputs/<name>/procedure.py) is loaded by worker.py;
the seeded generators below only pin which inputs each run receives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "downloads" / "naclip"
POOL_PREFIX = DATA / "adapt"  # visible unlabeled adaptation pool (2000 ADE20K training images)
EVAL_PREFIX = DATA / "dev"  # labeled dev images (500 ADE20K training images)
TEXT = DATA / "text_embeddings.npy"
LABELS = DATA / "dev_labels.npy"
ORACLE = DATA / "dev_oracle.npz"

EXP_CORE = "ade20k_naclip_core"
EXP_LOW = "ade20k_naclip_low_data"
EXP_EXPERT = "ade20k_naclip_expert_quality"

NUM_EXPERTS = 4
LOW_DATA_K = (4, 16, 64)
LOW_DATA_SEEDS = (0, 1, 2, 3)
LOW_EVAL_IMAGES = 500
PERM_SEED = 1312
SUBSET_SEED = 2203
WORKER_TIMEOUT_S = 4 * 3600

WORST = {
    EXP_CORE: {"mIoU": 0.0},
    EXP_LOW: {"mIoU": 0.0},
    EXP_EXPERT: {"expert_precision_pct": 0.0},
}


def low_data_runs(n_pool: int) -> list[dict]:
    runs = []
    for k in LOW_DATA_K:
        for seed in LOW_DATA_SEEDS:
            rng = np.random.default_rng([SUBSET_SEED, k, seed])
            runs.append({"k": k, "seed": seed, "adapt_ids": np.sort(rng.choice(n_pool, size=k, replace=False))})
    return runs


def low_eval_ids(n_eval: int) -> np.ndarray:
    rng = np.random.default_rng([SUBSET_SEED, 999])
    return np.sort(rng.choice(n_eval, size=min(LOW_EVAL_IMAGES, n_eval), replace=False))


def permutation(run_index: int, num_templates: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([PERM_SEED, run_index])
    return rng.permutation(num_templates), rng.permutation(num_classes)
