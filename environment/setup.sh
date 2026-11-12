#!/bin/bash
# Install Python packages into the container overlay.
#
# This script is run by the system on the login node (which has internet)
# inside the container with a persistent overlay. Changes persist across
# job submissions without rebuilding the container.
#
# Put ALL pip installs here, NOT in container.def.
# Only use container.def for the base Docker image and apt-get packages.
#
# IMPORTANT: Install into the SYSTEM Python, not a virtual environment.
# Use: uv pip install --system <packages>
#
# torch: frozen CLIP backbone (feature extraction) and method procedures (GPU or CPU).
# transformers: loads the public CLIP ViT-B/16 weights (pytorch_model.bin needs torch >= 2.6).
# pillow is pinned: hidden labels are hashed after nearest-neighbour resizing.

set -e

uv pip install --system "numpy==2.2.6" "torch==2.6.0" "transformers==4.51.3" "pillow==11.1.0"
