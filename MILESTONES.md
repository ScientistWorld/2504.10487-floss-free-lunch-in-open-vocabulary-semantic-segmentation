# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-06-12 08:16] - secondary_claims
- Made `scripts/reproduce.sh` call `scripts/evaluate.sh` explicitly at the end, so the top-level reproduction script always refreshes `scoring/scores.json`.
- Ran lightweight local checks: shell syntax validation for all main scripts and `python3 validate.py --compare`, both passing.
- Kept milestone at `secondary_claims`; real-dataset majority coverage remains blocked by gated or unavailable benchmark datasets.

### [2026-06-12 08:16] - secondary_claims
- Addressed read-only audit findings before final signoff: renamed scoring experiment keys to the paper targets they reference and documented the synthetic dense-feature substitute in the target descriptions.
- Made `scripts/baseline.sh` and `scripts/method.sh` score their own outputs without depending on each other.
- Submitted a CPU test job to verify the renamed scoring contract and standalone script changes in the managed container.

### [2026-06-12 08:11] - secondary_claims
- Completed final verification of workspace structure, scoring files, method/problem separation, and reusable evaluation scripts.
- Refreshed `scoring/scores.json` from saved method outputs so the reproduced scores include all benefit and constraint metrics declared in `scoring/reference.json`.
- Cleaned method-agnostic and high-level scoring documentation to avoid leaking solution-specific implementation details.

### [2026-06-12 08:06] - secondary_claims
- Audited the default evaluator against `scoring/reference.json`.
- Added the full benefit and constraint metric set to `eval/segmentation_metrics.py` so `scoring/scores.json` includes mIoU, pixel accuracy, mean class accuracy, rare-class mIoU, boundary accuracy, and template-budget metrics.
- Updated `scripts/run.sh` to regenerate baseline and FLOSS outputs with `scripts/reproduce.sh` before running validation and split scoring in the managed container.

### [2026-06-12 07:42] - secondary_claims
- Managed CPU test job `08371469-b67` completed with exit code 0 after the continuation-audit fixes.
- The job regenerated split method-input and eval-label bundles, reran `scripts/reproduce.sh`, and passed `python validate.py --compare`.
- Verified reproduced scores remained stable: template averaging 33.073 mIoU, FLOSS 37.596 mIoU, low-data FLOSS 38.249 mIoU, and expert precision 37.500% / 41.667%.
- Restored `secondary_claims`; real-dataset majority coverage remains blocked by gated or unavailable benchmark datasets.

### [2026-06-12 07:35] - core_claim_plus
- Continued-agent audit reviewed the previous `secondary_claims` claim by checking `method/`, `eval/`, scripts, scoring files, and prior managed job logs.
- Confirmed the committed method is not a surrogate: it implements FLOSS single-template predictions, entropy-based expert selection, top-4 class experts, and highest-confidence fusion.
- Fixed audit issues before restoring `secondary_claims`: split method-visible inputs from eval-only labels/oracle experts and generalized `scripts/evaluate.sh` for arbitrary future method outputs.
- Submitted those fixes from milestone `core_claim_plus` for managed-container verification before restoring `secondary_claims`.

### [2026-06-12 07:29] - secondary_claims
- Implemented the paper's FLOSS computation: averaged-template baseline, entropy-based top-4 class-expert selection from unlabeled features, and highest-confidence expert fusion with default fallback.
- Verified the reusable environment in a managed container job. On the synthetic dense-feature OVSS benchmark, template averaging reached 33.073 mIoU and FLOSS reached 37.596 mIoU.
- Verified the low-data variant using only 4 unlabeled selection images: FLOSS reached 38.249 mIoU.
- Added expert-selection quality diagnostics: selected experts overlap nontrivially with oracle class experts without using labels during selection.

## Stop Justification
- Completed at milestone `secondary_claims`.
