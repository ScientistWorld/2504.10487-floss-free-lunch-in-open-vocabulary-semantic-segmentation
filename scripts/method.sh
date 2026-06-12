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
  --dataset data/downloads/floss_synthetic.npz \
  --output-dir outputs/floss \
  --top-n 4 \
  --temperature 12.0

bash scripts/evaluate.sh
