#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   bash scripts/evaluate.sh
#   bash scripts/evaluate.sh my_method=outputs/my_method
#   bash scripts/evaluate.sh my_method=outputs/my_method \
#     --low-data-runs my_method_low=outputs/my_method_low \
#     --expert-quality-runs my_method=outputs/my_method
#
# Called from run.sh after training, or independently.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -eq 0 ]; then
  RUNS=(template_average=outputs/template_average floss=outputs/floss)
  LOW_DATA_RUNS=(template_average=outputs/template_average floss_low_data=outputs/floss_low_data)
  EXPERT_QUALITY_RUNS=(floss=outputs/floss floss_low_data=outputs/floss_low_data)
else
  RUNS=()
  LOW_DATA_RUNS=()
  EXPERT_QUALITY_RUNS=()
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
  if [ "${#RUNS[@]}" -eq 0 ]; then
    echo "Error: provide at least one NAME=outputs/path run before optional sections." >&2
    exit 2
  fi
fi

EXTRA_ARGS=()
if [ "${#LOW_DATA_RUNS[@]}" -gt 0 ]; then
  EXTRA_ARGS+=(--low-data-runs "${LOW_DATA_RUNS[@]}")
fi
if [ "${#EXPERT_QUALITY_RUNS[@]}" -gt 0 ]; then
  EXTRA_ARGS+=(--expert-quality-runs "${EXPERT_QUALITY_RUNS[@]}")
fi

python eval/segmentation_metrics.py \
  --dataset data/downloads/floss_synthetic_inputs.npz \
  --labels data/downloads/floss_synthetic_eval.npz \
  --runs "${RUNS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --output scoring/scores.json
