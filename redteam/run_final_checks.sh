#!/bin/bash
set -euo pipefail
cd /home/user
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
bash redteam/weaker_random_experts/run.sh
bash redteam/forged_scores_shortcut/run.sh
bash redteam/stale_failure_shortcut/run.sh
python redteam/summarize.py
