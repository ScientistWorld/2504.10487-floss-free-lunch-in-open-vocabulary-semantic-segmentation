"""Fixed hidden-test protocol: inputs, splits, seeds and operating points.

Everything on the evaluator's side of the comparison is pinned here: which
unlabeled images a method may adapt on, which images it is scored on, the
low-data subset sizes and seeds, the expert budget of the diagnostic, and the
anonymising template/class permutations.

The method artifact itself (outputs/<name>/procedure.py) is loaded by worker.py;
the seeded generators below only pin which inputs each run receives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data" / "naclip"
POOL_PREFIX = DATA / "pool"  # hidden unlabeled adaptation pool (2000 ADE20K training images)
EVAL_PREFIX = DATA / "val"  # ADE20K validation images (2000)
TEXT = DATA / "text_embeddings.npy"
LABELS = DATA / "val_labels.npy"
ORACLE = DATA / "val_oracle.npz"
HASHES = HERE / "label_hashes.json"
# Test answers and everything that regenerates them; unreadable while a procedure runs.
ANSWERS = (LABELS, ORACLE, HERE / "data" / "ade20k" / "val" / "annotations")

EXP_CORE = "ade20k_naclip_core"
EXP_LOW = "ade20k_naclip_low_data"
EXP_EXPERT = "ade20k_naclip_expert_quality"

NUM_EXPERTS = 4  # experts per class reported for the diagnostic (paper's N)
LOW_DATA_K = (4, 16, 64)  # unlabeled adaptation images per low-data run
LOW_DATA_SEEDS = (0, 1, 2, 3)
LOW_EVAL_IMAGES = 500  # validation images scored in every low-data run
PERM_SEED = 7121
SUBSET_SEED = 5507
WORKER_TIMEOUT_S = 4 * 3600  # total budget per method over all of its runs

WORST = {
    EXP_CORE: {"mIoU": 0.0},
    EXP_LOW: {"mIoU": 0.0},
    EXP_EXPERT: {"expert_precision_pct": 0.0},
}


def low_data_runs(n_pool: int) -> list[dict]:
    """[{k, seed, adapt_ids}] for every low-data run, in a fixed order."""
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
    """Template and class order shown to the method in a given run."""
    rng = np.random.default_rng([PERM_SEED, run_index])
    return rng.permutation(num_templates), rng.permutation(num_classes)
