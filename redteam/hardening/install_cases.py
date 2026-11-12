"""Install every hardening case as a method directory under redteam/hardening/outputs/all/."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLOSS = ROOT / "method" / "floss.py"

# FLOSS configurations: the reference, honest variants and adaptation-pool seed reruns.
FLOSS_CASES = {
    "floss": {},  # reference (paper settings)
    # candidate improvements (label-free)
    "v_default_region": {"region": "default_region"},
    "v_min_support": {"min_support": 0.25},
    "v_shrink": {"shrink": 1.0},
    "v_default_region_shrink": {"region": "default_region", "shrink": 1.0},
    "v_top8": {"top_n": 8},
    "v_avg_prob": {"selection": "avg_prob"},
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


def main() -> None:
    root = HERE / "outputs" / "all"
    if root.exists():
        shutil.rmtree(root)
    for name, cfg in FLOSS_CASES.items():
        d = root / name
        d.mkdir(parents=True)
        shutil.copy(FLOSS, d / "procedure.py")
        if cfg:
            (d / "config.json").write_text(json.dumps(cfg) + "\n")
    for case in sorted((HERE / "cases").iterdir()):
        if not case.is_dir() or case.name.startswith(("_", ".")):
            continue
        d = root / case.name
        shutil.copytree(case, d, ignore=shutil.ignore_patterns("__pycache__"))
        if (d / "procedure.py").exists():
            shutil.copy(HERE / "cases" / "_common.py", d / "_common.py")
    ref = HERE / "outputs" / "ref_only"
    if ref.exists():
        shutil.rmtree(ref)
    (ref / "floss").mkdir(parents=True)
    shutil.copy(FLOSS, ref / "floss" / "procedure.py")
    crash = HERE / "outputs" / "crash_only"
    if crash.exists():
        shutil.rmtree(crash)
    shutil.copytree(HERE / "cases" / "sc_crash", crash / "sc_crash", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"installed {len(list(root.iterdir()))} cases under {root}")


if __name__ == "__main__":
    main()
