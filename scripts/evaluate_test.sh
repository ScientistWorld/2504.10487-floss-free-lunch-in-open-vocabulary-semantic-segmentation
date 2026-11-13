#!/bin/bash
# Hidden-test evaluation. Takes no arguments.
#
# Scores every directory under outputs/ (each must hold a procedure.py; see
# briefing/evaluation.md) plus the built-in averaged-template baseline, by
# running each procedure on inputs only eval/test/ holds: a hidden unlabeled
# adaptation pool and the 2000 ADE20K validation images. Every run (core and
# each low-data run) is a fresh, file-restricted worker process, and the hidden
# labels are unreadable while it runs. A method that fails gets the worst value
# in every metric. Writes scoring/scores_test.json.

set -euo pipefail
cd "$(dirname "$0")/.."
python eval/test/evaluate.py
