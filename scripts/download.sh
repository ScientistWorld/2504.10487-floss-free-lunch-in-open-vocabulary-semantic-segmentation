#!/bin/bash
# Download all data and models needed for this environment (needs internet).
#
# Idempotent: skips anything already in place.
#
# Estimated download: ~0.97 GB (ADE20K challenge zip) + ~0.6 GB (CLIP ViT-B/16)
# Disk after placement: ~0.3 GB images + 0.6 GB weights; features built later
# by scripts/prepare_data.sh add ~4 GB (data/downloads/naclip, eval/test/data/naclip).
#
# The ADE20K zip is split by data/ade20k_layout.py: method-visible images go
# to data/downloads/ade20k/, the hidden test images and validation labels to
# eval/test/data/ade20k/. Nothing else from the archive is kept.

set -euo pipefail
cd "$(dirname "$0")/.."

CLIP_DIR=data/downloads/clip-vit-base-patch16
mkdir -p "$CLIP_DIR"
for f in config.json merges.txt preprocessor_config.json special_tokens_map.json \
         tokenizer.json tokenizer_config.json vocab.json pytorch_model.bin; do
  if [ ! -s "$CLIP_DIR/$f" ]; then
    wget -q -c "https://huggingface.co/openai/clip-vit-base-patch16/resolve/main/$f" -O "$CLIP_DIR/$f"
  fi
done

if ! python3 data/ade20k_layout.py --check; then
  ZIP=data/downloads/ADEChallengeData2016.zip
  mkdir -p data/downloads
  wget -q -c https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip -O "$ZIP"
  python3 data/ade20k_layout.py --zip "$ZIP"
  rm -f "$ZIP"
fi
echo "downloads ready; build features with: bash scripts/prepare_data.sh (GPU)"
