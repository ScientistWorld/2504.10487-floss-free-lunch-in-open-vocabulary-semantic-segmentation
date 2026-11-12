#!/bin/bash
# Install the paper's method (FLOSS) as a gym procedure and score it.
#
# The method needs no training: outputs/floss/procedure.py is a copy of
# method/floss.py, run by the evaluators on the unlabeled adaptation images
# they choose. Default settings are the paper's (N = 4 entropy-selected
# class-experts, highest-confidence fusion).
#
# Usage: bash scripts/method.sh

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/floss
cp method/floss.py outputs/floss/procedure.py
rm -f outputs/floss/config.json
bash scripts/evaluate.sh
