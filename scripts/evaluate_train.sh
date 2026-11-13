#!/bin/bash
# Visible-slice (train) evaluation. Takes no arguments.
#
# Scores every directory under outputs/ (each must hold a procedure.py; see
# briefing/evaluation.md) plus the built-in averaged-template baseline, by
# running each procedure on the visible unlabeled adaptation pool and the
# 500-image labeled dev set, one fresh file-restricted worker process per run.
# Writes scoring/scores_train.json.

set -euo pipefail
cd "$(dirname "$0")/.."
python eval/train/evaluate.py
