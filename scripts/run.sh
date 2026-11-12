#!/bin/bash
# Job entry point: build features, score the reference end to end, run the hardening checks.
set -euo pipefail
cd /home/user
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python -c "import torch, transformers, PIL, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'pillow', PIL.__version__, 'numpy', numpy.__version__)"
nproc || true

# 1. Frozen-backbone features, labels and ground-truth tables (idempotent).
bash scripts/prepare_data.sh

# 2. outputs/ holds only the paper's method; the averaged-template baseline is built into the evaluators.
rm -rf outputs
bash scripts/method.sh          # installs outputs/floss/procedure.py, writes scoring/scores.json
bash scripts/evaluate_train.sh  # scoring/scores_train.json
bash scripts/evaluate_test.sh   # scoring/scores_test.json

# 3. Variants, seed reruns and shortcut probes through the same evaluators (GPU only; slow on CPU).
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  bash redteam/hardening/run_hardening.sh
else
  echo "no GPU: skipping hardening cases"
fi

python validate.py --compare || true
