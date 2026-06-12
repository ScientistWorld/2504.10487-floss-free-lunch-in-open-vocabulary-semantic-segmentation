#!/bin/bash
# Held-out test-slice evaluation for the same saved method outputs scored on train.
#
# Argument contract mirrors scripts/evaluate.sh:
#   bash scripts/evaluate_test.sh
#   bash scripts/evaluate_test.sh my_method=outputs/my_method
#   bash scripts/evaluate_test.sh my_method=outputs/my_method \
#     --low-data-runs my_method_low=outputs/my_method_low \
#     --expert-quality-runs my_method=outputs/my_method

set -euo pipefail

cd /home/user

mkdir -p scoring

RUNS=()
LOW_DATA_RUNS=()
EXPERT_QUALITY_RUNS=()

read_train_scored_runs() {
  python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


scores_path = Path("scoring/scores_train.json")
if not scores_path.exists():
    raise SystemExit(
        "scoring/scores_train.json is required when no explicit runs are passed; "
        "run scripts/evaluate_train.sh first or pass NAME=outputs/path specs."
    )

scores = load_json(scores_path)
manifest = load_json(Path("scoring/eval_train_manifest.json"))
manifest_exps = manifest.get("experiments", {})
experiments = scores.get("experiments", scores)

sections = [
    ("runs", "clip_dinoiser_cityscapes_core"),
    ("low_data", "clip_dinoiser_cityscapes_low_data"),
    ("expert_quality", "expert_quality_clip_dinoiser_cityscapes_core"),
]

for section, exp_name in sections:
    exp = experiments.get(exp_name)
    if not isinstance(exp, dict):
        continue
    results = exp.get("results", exp)
    if not isinstance(results, dict):
        continue
    paths = manifest_exps.get(exp_name, {}).get("runs", {})
    for name in sorted(results):
        if name.startswith("_"):
            continue
        path = paths.get(name, f"outputs/{name}")
        print(f"{section}\t{name}={path}")
PY
}

write_manifest() {
  python - "$@" <<'PY'
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_specs(specs: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        parsed[name] = path
    return parsed


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--slice", required=True)
parser.add_argument("--runs", nargs="*", default=[])
parser.add_argument("--low-data-runs", nargs="*", default=[])
parser.add_argument("--expert-quality-runs", nargs="*", default=[])
args = parser.parse_args()

manifest = {
    "slice": args.slice,
    "experiments": {
        "clip_dinoiser_cityscapes_core": {"runs": parse_specs(args.runs)},
        "clip_dinoiser_cityscapes_low_data": {"runs": parse_specs(args.low_data_runs)},
        "expert_quality_clip_dinoiser_cityscapes_core": {"runs": parse_specs(args.expert_quality_runs)},
    },
}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
}

if [ "$#" -eq 0 ]; then
  while IFS=$'\t' read -r section spec; do
    case "$section" in
      runs) RUNS+=("$spec") ;;
      low_data) LOW_DATA_RUNS+=("$spec") ;;
      expert_quality) EXPERT_QUALITY_RUNS+=("$spec") ;;
    esac
  done < <(read_train_scored_runs)
else
  section="runs"
  for arg in "$@"; do
    case "$arg" in
      --low-data-runs)
        section="low_data"
        ;;
      --expert-quality-runs)
        section="expert_quality"
        ;;
      --runs)
        section="runs"
        ;;
      *)
        case "$section" in
          runs) RUNS+=("$arg") ;;
          low_data) LOW_DATA_RUNS+=("$arg") ;;
          expert_quality) EXPERT_QUALITY_RUNS+=("$arg") ;;
        esac
        ;;
    esac
  done
fi

if [ "${#RUNS[@]}" -eq 0 ]; then
  echo "Error: no runs to score. Provide NAME=outputs/path specs or run scripts/evaluate_train.sh first." >&2
  exit 2
fi

MANIFEST_ARGS=(--output scoring/eval_test_manifest.json --slice test --runs "${RUNS[@]}")
if [ "${#LOW_DATA_RUNS[@]}" -gt 0 ]; then
  MANIFEST_ARGS+=(--low-data-runs "${LOW_DATA_RUNS[@]}")
fi
if [ "${#EXPERT_QUALITY_RUNS[@]}" -gt 0 ]; then
  MANIFEST_ARGS+=(--expert-quality-runs "${EXPERT_QUALITY_RUNS[@]}")
fi
write_manifest "${MANIFEST_ARGS[@]}"

EXTRA_ARGS=()
if [ "${#LOW_DATA_RUNS[@]}" -gt 0 ]; then
  EXTRA_ARGS+=(--low-data-runs "${LOW_DATA_RUNS[@]}")
fi
if [ "${#EXPERT_QUALITY_RUNS[@]}" -gt 0 ]; then
  EXTRA_ARGS+=(--expert-quality-runs "${EXPERT_QUALITY_RUNS[@]}")
fi

python eval/test/evaluate.py \
  --dataset data/downloads/floss_synthetic_inputs.npz \
  --labels data/downloads/floss_synthetic_eval.npz \
  --runs "${RUNS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --output scoring/scores_test.json
