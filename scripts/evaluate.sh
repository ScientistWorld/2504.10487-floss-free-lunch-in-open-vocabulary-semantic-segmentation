#!/bin/bash
# Standard evaluation of everything under outputs/ on the visible slice.
#
# Same protocol as scripts/evaluate_train.sh (visible unlabeled pool + 500
# labeled dev images, every outputs/<name>/procedure.py plus the built-in
# averaged-template baseline). Writes scoring/scores.json. Uses only eval/;
# never imports method/.
#
# Usage: bash scripts/evaluate.sh

set -euo pipefail
cd "$(dirname "$0")/.."
python eval/train/evaluate.py --output scoring/scores.json
