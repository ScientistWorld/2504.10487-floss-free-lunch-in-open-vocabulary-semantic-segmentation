# Evaluation Targets

All targets are scored by `scripts/evaluate_train.sh` (visible slice: 2000-image unlabeled pool, 500 labeled dev images) and `scripts/evaluate_test.sh` (hidden slice: a different 2000-image unlabeled pool, 2000 ADE20K validation images). Both run every `outputs/<name>/procedure.py` and the built-in `template_average` baseline.

## Primary: ade20k_naclip_core (weight 0.6)

Paper Table 1, NACLIP on ADE20K: 17.4 → 18.4 mIoU. The method adapts label-free on the full unlabeled pool and is scored on the held-out images.

- **Metric**: `mIoU` over 150 classes (higher is better).

## Secondary: ade20k_naclip_low_data (weight 0.25)

Paper Fig. 3 / Table 4: adaptation from very few unlabeled images still helps. 12 evaluator-drawn runs with 4, 16 and 64 adaptation images, each scored on 500 fixed held-out images.

- **Metric**: mean `mIoU` over the 12 runs (higher is better).

## Diagnostic: ade20k_naclip_expert_quality (weight 0.15)

Paper Table 2 (Eq. 8), NACLIP on ADE20K: 57%. This is the share of the 4 per-class templates a method reports that are true experts (paper Eq. 4) on the held-out labels.

- **Metric**: `expert_precision_pct` (higher is better).

## Constraints (enforced by the protocol rather than scored)

The paper's claim is an improvement *without labels, without training, and with a frozen backbone and the fixed 80-template pool*. The protocol enforces these structurally:

- A procedure receives only unlabeled features and the 80×150 template embeddings.
- It runs on scored images it never sees, under shuffled template and class order.
- Every run (core and each low-data run) is a fresh process. The procedure can read only its own directory, cannot write lasting files and cannot start processes. The hidden labels are unreadable while it runs.
- Any invalid output fails the whole run (worst value in every metric).

Rules that the protocol cannot enforce mechanically, such as no reading of label files and no data beyond what `adapt` is given, are listed in `scoring/CONSTRAINTS.md` and are checked separately.
