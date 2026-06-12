# Reproduction Milestones

**Current: core_claim**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-12 07:14] - method_runs
- Read the FLOSS paper and checked the released implementation details.
- Implemented the paper's entropy-based class-expert selection and highest-confidence expert fusion on cached dense segmentation features.
- Local end-to-end reproduction runs baseline, FLOSS, and evaluation without errors.

### [2026-06-12 07:18] - core_claim
- CPU compute job `a289cd39-9cc` completed successfully in the container.
- Verified the core FLOSS claim on `synthetic_ovss`: template averaging scored 33.073 mIoU and FLOSS scored 37.596 mIoU with 4 experts per class.
- Added a local low-data variant for the next submitted job to verify.

## Stop Justification
- Not stopped. A CPU compute job is being submitted to verify the expanded low-data experiment.
