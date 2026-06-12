#!/bin/bash
# Visible train-slice evaluation for saved method outputs.
#
# Argument contract mirrors scripts/evaluate.sh:
#   bash scripts/evaluate_train.sh
#   bash scripts/evaluate_train.sh my_method=outputs/my_method
#   bash scripts/evaluate_train.sh my_method=outputs/my_method \
#     --low-data-runs my_method_low=outputs/my_method_low \
#     --expert-quality-runs my_method=outputs/my_method

set -euo pipefail

cd /home/user

mkdir -p scoring

RUNS=()
LOW_DATA_RUNS=()
EXPERT_QUALITY_RUNS=()

discover_runs() {
  local pred dir name
  for pred in outputs/*/predictions.npy; do
    [ -f "$pred" ] || continue
    dir="${pred%/predictions.npy}"
    name="${dir#outputs/}"
    printf '%s=%s\n' "$name" "$dir"
  done | sort
}

discover_low_data_runs() {
  local pred dir name
  for pred in outputs/*/predictions.npy; do
    [ -f "$pred" ] || continue
    dir="${pred%/predictions.npy}"
    name="${dir#outputs/}"
    case "$name" in
      *low_data*) printf '%s=%s\n' "$name" "$dir" ;;
    esac
  done | sort
}

discover_expert_quality_runs() {
  local experts dir name
  for experts in outputs/*/experts.npy; do
    [ -f "$experts" ] || continue
    dir="${experts%/experts.npy}"
    name="${dir#outputs/}"
    printf '%s=%s\n' "$name" "$dir"
  done | sort
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
        "synthetic_ovss": {"runs": parse_specs(args.runs)},
        "low_data_synthetic_ovss": {"runs": parse_specs(args.low_data_runs)},
        "expert_quality_synthetic_ovss": {"runs": parse_specs(args.expert_quality_runs)},
    },
}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
}

if [ "$#" -eq 0 ]; then
  mapfile -t RUNS < <(discover_runs)
  mapfile -t LOW_DATA_RUNS < <(discover_low_data_runs)
  mapfile -t EXPERT_QUALITY_RUNS < <(discover_expert_quality_runs)
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
  echo "Error: no runs found. Provide NAME=outputs/path or create outputs/*/predictions.npy." >&2
  exit 2
fi

MANIFEST_ARGS=(--output scoring/eval_train_manifest.json --slice train --runs "${RUNS[@]}")
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

python eval/train/evaluate.py \
  --dataset data/downloads/floss_synthetic_inputs.npz \
  --labels data/downloads/floss_synthetic_eval.npz \
  --runs "${RUNS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --output scoring/scores_train.json
