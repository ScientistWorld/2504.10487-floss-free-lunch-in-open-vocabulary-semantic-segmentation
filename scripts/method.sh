#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

bash scripts/download.sh
python method/floss.py \
  --dataset data/downloads/floss_synthetic_inputs.npz \
  --output-dir outputs/floss \
  --top-n 4 \
  --temperature 12.0

python method/floss.py \
  --dataset data/downloads/floss_synthetic_inputs.npz \
  --output-dir outputs/floss_low_data \
  --top-n 4 \
  --temperature 12.0 \
  --train-limit 4

bash scripts/evaluate.sh \
  floss=outputs/floss \
  --low-data-runs floss_low_data=outputs/floss_low_data \
  --expert-quality-runs floss=outputs/floss floss_low_data=outputs/floss_low_data
