"""Install every hardening case as its own outputs root: redteam/hardening/outputs/cases/<name>/<name>/.

One outputs root per case lets run_hardening.sh score (and resume) the cases one at a time through the
unmodified evaluators. Also prepares redteam/hardening/outputs/before/ (probes re-scored with the
pre-hardening evaluator) and the single-method roots used by the integrity scenarios.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLOSS = ROOT / "method" / "floss.py"
VISIBLE = ROOT / "data" / "downloads" / "naclip"

# FLOSS configurations: the reference, honest variants and adaptation-pool seed reruns.
FLOSS_CASES = {
    "floss": {},  # reference (paper settings)
    # candidate improvements (label-free)
    "v_avg_prob": {"selection": "avg_prob"},  # paper Fig. 4a alternative proxy
    "v_min_support": {"min_support": 0.25},
    "v_min_count25": {"min_count": 25},  # low-data robustness: experts need >= 25 predicted patches
    "v_min_count100": {"min_count": 100},
    # weaker variants (the paper's ablations / matched-budget random experts)
    "w_random_experts": {"selection": "random", "seed": 0},
    "w_average_all": {"fusion": "average_all"},
    "w_default_conflict": {"fusion": "default_conflict"},
    "w_top1": {"top_n": 1},
    # reference rerun on random halves of the adaptation pool (seed spread)
    "seed_pool50_s0": {"pool_fraction": 0.5, "seed": 0},
    "seed_pool50_s1": {"pool_fraction": 0.5, "seed": 1},
    "seed_pool50_s2": {"pool_fraction": 0.5, "seed": 2},
}
# Probes scored again with the pre-hardening evaluator to measure their "before" score.
BEFORE = ("sc_cache_across_runs", "sc_file_cache_across_runs", "sc_hidden_labels_raw_read")


def _install_probe(case: Path, dest: Path) -> None:
    shutil.copytree(case, dest, ignore=shutil.ignore_patterns("__pycache__"))
    if (dest / "procedure.py").exists():
        shutil.copy(HERE / "cases" / "_common.py", dest / "_common.py")
    if case.name == "sc_dev_oracle_shipped":
        iou = np.load(VISIBLE / "dev_oracle.npz")["iou"][:-1]
        top4 = np.argsort(-np.nan_to_num(iou, nan=-1.0), axis=0, kind="stable")[:4].T
        text = np.load(VISIBLE / "text_embeddings.npy").astype(np.float32)
        np.save(dest / "oracle_top4.npy", top4.astype(np.int64))
        np.save(dest / "canon_template_means.npy", text.mean(1).astype(np.float16))
        np.save(dest / "canon_class_means.npy", text.mean(0).astype(np.float16))


def main() -> None:
    out = HERE / "outputs"
    for sub in ("cases", "before", "ref_only", "crash_only"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for name, cfg in FLOSS_CASES.items():
        d = out / "cases" / name / name
        d.mkdir(parents=True)
        shutil.copy(FLOSS, d / "procedure.py")
        if cfg:
            (d / "config.json").write_text(json.dumps(cfg) + "\n")
    for case in sorted((HERE / "cases").iterdir()):
        if not case.is_dir() or case.name.startswith(("_", ".")):
            continue
        _install_probe(case, out / "cases" / case.name / case.name)
        if case.name in BEFORE:
            _install_probe(case, out / "before" / case.name)
    (out / "ref_only" / "floss").mkdir(parents=True)
    shutil.copy(FLOSS, out / "ref_only" / "floss" / "procedure.py")
    shutil.copytree(HERE / "cases" / "sc_crash", out / "crash_only" / "sc_crash",
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"installed {len(list((out / 'cases').iterdir()))} cases under {out / 'cases'}")


if __name__ == "__main__":
    main()
