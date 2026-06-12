# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-12 08:00] - core_claim_plus
- Continued-agent audit reviewed the previous `secondary_claims` claim by checking `method/`, `eval/`, scripts, scoring files, and prior managed job logs.
- Confirmed the committed method is not a surrogate: it implements FLOSS single-template predictions, entropy-based expert selection, top-4 class experts, and highest-confidence fusion.
- Fixed audit issues before restoring `secondary_claims`: split method-visible inputs from eval-only labels/oracle experts and generalized `scripts/evaluate.sh` for arbitrary future method outputs.
- Current unverified code changes are submitted from milestone `core_claim_plus`; restore to `secondary_claims` after the managed job confirms the patched pipeline.

### [2026-06-12 07:29] - secondary_claims
- Implemented the paper's FLOSS computation: averaged-template baseline, entropy-based top-4 class-expert selection from unlabeled features, and highest-confidence expert fusion with default fallback.
- Verified the reusable environment in a managed container job. On the synthetic dense-feature OVSS benchmark, template averaging reached 33.073 mIoU and FLOSS reached 37.596 mIoU.
- Verified the low-data variant using only 4 unlabeled selection images: FLOSS reached 38.249 mIoU.
- Added expert-selection quality diagnostics: selected experts overlap nontrivially with oracle class experts without using labels during selection.

## Stop Justification
- Not stopped. A CPU test job is being submitted to verify the continuation-audit fixes in the managed container before restoring the previous `secondary_claims` milestone.
