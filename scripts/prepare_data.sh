#!/bin/bash
# Build frozen-backbone features, labels and ground-truth tables (GPU job).
#
# Requires scripts/download.sh to have run (ADE20K images + CLIP ViT-B/16).
# Writes method-visible artifacts to data/downloads/naclip/ and hidden test
# artifacts to eval/test/data/naclip/. Idempotent. About 10 minutes on one GPU.

set -euo pipefail
cd "$(dirname "$0")/.."
python data/prepare_visible.py
python eval/test/prepare_test_data.py
