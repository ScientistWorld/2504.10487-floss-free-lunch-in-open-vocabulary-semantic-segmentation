#!/bin/bash
# Reproduce the paper's results end-to-end (inside the compute container).
#
# Assumes scripts/download.sh has run on a node with internet. Builds the
# frozen-backbone features (GPU), installs FLOSS under outputs/floss/, and
# scores it with the built-in averaged-template baseline on the visible slice
# (scoring/scores.json).

set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/prepare_data.sh
bash scripts/baseline.sh
bash scripts/method.sh
