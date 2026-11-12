#!/bin/bash
# The paper's baseline: the frozen NACLIP model with CLIP's default
# averaged-template classifier (all 80 templates averaged per class).
#
# It needs no adaptation and is built into both evaluators, which score it as
# `template_average` alongside every method under outputs/. This script runs
# the visible-slice evaluation so the baseline (and any installed method) is
# scored into scoring/scores.json.
#
# Usage: bash scripts/baseline.sh

set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/evaluate.sh
