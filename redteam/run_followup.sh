#!/bin/bash
set -euo pipefail
cd /home/user
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
bash redteam/layout_shortcut/run.sh
bash redteam/diagnostic_only_positive_shortcut/run.sh
python redteam/summarize.py
