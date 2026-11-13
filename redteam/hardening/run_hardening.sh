#!/bin/bash
# Run every hardening case through the unmodified train and hidden-test evaluators, one case at a time
# (resumable: a case whose result file exists is skipped), re-score the new cross-run / raw-read probes
# with the pre-hardening evaluator (tmp/old_eval_test, from git), and run the integrity scenarios.
# Results go to redteam/hardening/results/.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=redteam/hardening
R=$H/results
mkdir -p "$R/test" "$R/train" "$R/before" "$R/scenarios"
python $H/install_cases.py
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null || true

# Built-in baseline once per slice (no outputs root -> only template_average).
[ -s "$R/test/template_average.json" ] || python eval/test/evaluate.py --outputs-root $H/outputs/none \
  --output "$R/test/template_average.json" --details-dir $H/outputs/details > "$R/test/template_average.log" 2>&1
[ -s "$R/train/template_average.json" ] || python eval/train/evaluate.py --outputs-root $H/outputs/none \
  --output "$R/train/template_average.json" > "$R/train/template_average.log" 2>&1

for dir in $H/outputs/cases/*/; do
  c=$(basename "$dir")
  if [ ! -s "$R/test/$c.json" ]; then
    python eval/test/evaluate.py --skip-builtin --outputs-root "$dir" --output "$R/test/$c.json" \
      --details-dir $H/outputs/details > "$R/test/$c.log" 2>&1
    echo "test  $c: $(grep -h "\[eval\] $c: {" "$R/test/$c.log" | tail -1)"
  fi
  case "$c" in sc_*|ub_*) continue ;; esac  # shortcut probes: hidden test only
  if [ ! -s "$R/train/$c.json" ]; then
    python eval/train/evaluate.py --skip-builtin --outputs-root "$dir" --output "$R/train/$c.json" \
      > "$R/train/$c.log" 2>&1
    echo "train $c: $(grep -h "\[eval-train\] $c: {" "$R/train/$c.log" | tail -1)"
  fi
done

# "Before": the new probes through the evaluator as it was before this hardening (single worker process
# for all runs, no file guard, answers readable). Snapshot taken from git by the hardening iteration.
if [ -f tmp/old_eval_test/evaluate.py ] && [ ! -s "$R/before/test_probes.json" ]; then
  python tmp/old_eval_test/evaluate.py --outputs-root $H/outputs/before --output "$R/before/test_probes.json" \
    > "$R/before/test_probes.log" 2>&1
  rm -f /tmp/floss_probe_full_pool_experts.npz
fi

D=eval/test/data/naclip
A=eval/test/data/ade20k/val/annotations
[ -f $D/val_labels.npy.orig ] && mv -f $D/val_labels.npy.orig $D/val_labels.npy
[ -f $D/val_oracle.npz.orig ] && mv -f $D/val_oracle.npz.orig $D/val_oracle.npz
S=$R/scenarios
# Scenario A: hidden labels edited on disk -> every method scored as failed, exit != 0.
if [ ! -s "$S/label_tamper.json" ]; then
  cp $D/val_labels.npy $D/val_labels.npy.orig
  python - <<'PY'
import numpy as np
p = "eval/test/data/naclip/val_labels.npy"
a = np.load(p); a[a != 255] = 0; np.save(p, a)
PY
  python eval/test/evaluate.py --skip-builtin --outputs-root $H/outputs/ref_only --output "$S/label_tamper.json" \
    > "$S/label_tamper.log" 2>&1
  echo "exit=$?" >> "$S/label_tamper.log"
  mv -f $D/val_labels.npy.orig $D/val_labels.npy
fi
# Scenario B: oracle table edited on disk -> recomputed from verified labels, scores unchanged.
if [ ! -s "$S/oracle_tamper.json" ]; then
  cp $D/val_oracle.npz $D/val_oracle.npz.orig
  python - <<'PY'
import numpy as np
p = "eval/test/data/naclip/val_oracle.npz"
t = dict(np.load(p)); t["iou"] = np.random.default_rng(0).random(t["iou"].shape); np.savez(p, **t)
PY
  python eval/test/evaluate.py --skip-builtin --outputs-root $H/outputs/ref_only --output "$S/oracle_tamper.json" \
    > "$S/oracle_tamper.log" 2>&1
  echo "exit=$?" >> "$S/oracle_tamper.log"
  mv -f $D/val_oracle.npz.orig $D/val_oracle.npz
fi
# Scenario C: forged 100-valued score file left behind and the answers left sealed by a killed run,
# then a crashing method -> answers unsealed, file regenerated with worst values.
if [ ! -s "$S/stale.json" ] || grep -q '100.0' "$S/stale.json"; then
  python - <<'PY'
import json
forged = {"experiments": {e: {"results": {"sc_crash": {m: 100.0}}} for e, m in [
    ("ade20k_naclip_core", "mIoU"), ("ade20k_naclip_low_data", "mIoU"),
    ("ade20k_naclip_expert_quality", "expert_precision_pct")]}}
open("redteam/hardening/results/scenarios/stale.json", "w").write(json.dumps(forged))
PY
  chmod 000 $D/val_labels.npy $D/val_oracle.npz $A
  python eval/test/evaluate.py --skip-builtin --outputs-root $H/outputs/crash_only --output "$S/stale.json" \
    > "$S/stale.log" 2>&1
  echo "exit=$?" >> "$S/stale.log"
fi
# Answers must be readable again after every evaluation.
stat -c '%a %n' $D/val_labels.npy $D/val_oracle.npz $A | tee "$S/answer_modes.txt"
python $H/analyze.py
