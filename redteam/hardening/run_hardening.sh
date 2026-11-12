#!/bin/bash
# Run every hardening case through the unmodified train and hidden-test evaluators,
# plus three integrity scenarios. Results go to redteam/hardening/results/.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=redteam/hardening
R=$H/results
mkdir -p "$R"
python $H/install_cases.py

if [ ! -s "$R/test_all.json" ]; then
  python eval/test/evaluate.py --outputs-root $H/outputs/all --output "$R/test_all.json" \
    --details-dir $H/outputs/details 2>&1 | tee "$R/test_all.log"
fi
if [ ! -s "$R/train_all.json" ]; then
  python eval/train/evaluate.py --outputs-root $H/outputs/all --output "$R/train_all.json" 2>&1 | tee "$R/train_all.log"
fi

D=eval/test/data/naclip
# Restore anything a killed previous run left tampered.
[ -f $D/val_labels.npy.orig ] && mv -f $D/val_labels.npy.orig $D/val_labels.npy
[ -f $D/val_oracle.npz.orig ] && mv -f $D/val_oracle.npz.orig $D/val_oracle.npz
# Scenario A: hidden labels edited on disk -> every method must be scored as failed, exit != 0.
if [ ! -s "$R/scenario_label_tamper.json" ]; then
  cp $D/val_labels.npy $D/val_labels.npy.orig
  python - <<'PY'
import numpy as np
p = "eval/test/data/naclip/val_labels.npy"
a = np.load(p); a[a != 255] = 0; np.save(p, a)
PY
  python eval/test/evaluate.py --outputs-root $H/outputs/ref_only --output "$R/scenario_label_tamper.json" \
    > "$R/scenario_label_tamper.log" 2>&1
  echo "exit=$?" >> "$R/scenario_label_tamper.log"
  mv -f $D/val_labels.npy.orig $D/val_labels.npy
fi
# Scenario B: oracle table edited on disk -> recomputed from verified labels, scores unchanged.
if [ ! -s "$R/scenario_oracle_tamper.json" ]; then
  cp $D/val_oracle.npz $D/val_oracle.npz.orig
  python - <<'PY'
import numpy as np
p = "eval/test/data/naclip/val_oracle.npz"
t = dict(np.load(p)); t["iou"] = np.random.default_rng(0).random(t["iou"].shape); np.savez(p, **t)
PY
  python eval/test/evaluate.py --outputs-root $H/outputs/ref_only --output "$R/scenario_oracle_tamper.json" \
    > "$R/scenario_oracle_tamper.log" 2>&1
  echo "exit=$?" >> "$R/scenario_oracle_tamper.log"
  mv -f $D/val_oracle.npz.orig $D/val_oracle.npz
fi
# Scenario C: forged 100-valued score file left behind, then a crashing method -> file regenerated.
if [ ! -s "$R/scenario_stale.json" ]; then
  python - <<'PY'
import json
forged = {"experiments": {e: {"results": {"sc_crash": {m: 100.0}}} for e, m in [
    ("ade20k_naclip_core", "mIoU"), ("ade20k_naclip_low_data", "mIoU"),
    ("ade20k_naclip_expert_quality", "expert_precision_pct")]}}
open("redteam/hardening/results/scenario_stale.json", "w").write(json.dumps(forged))
PY
  python eval/test/evaluate.py --outputs-root $H/outputs/crash_only --output "$R/scenario_stale.json" \
    > "$R/scenario_stale.log" 2>&1
  echo "exit=$?" >> "$R/scenario_stale.log"
fi
python $H/analyze.py
