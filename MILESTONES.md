# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-06-12 07:29] - secondary_claims
- Implemented the paper's FLOSS computation: averaged-template baseline, entropy-based top-4 class-expert selection from unlabeled features, and highest-confidence expert fusion with default fallback.
- Verified the reusable environment in a managed container job. On the synthetic dense-feature OVSS benchmark, template averaging reached 33.073 mIoU and FLOSS reached 37.596 mIoU.
- Verified the low-data variant using only 4 unlabeled selection images: FLOSS reached 38.249 mIoU.
- Added expert-selection quality diagnostics: selected experts overlap nontrivially with oracle class experts without using labels during selection.

## Stop Justification
- Completed at milestone `secondary_claims`. Real Cityscapes/ACDC/BDD/Mapillary reproduction remains out of scope because the required segmentation datasets are gated or unavailable in the shared cache, but the packaged benchmark preserves the FLOSS computational procedure and evaluates the core and low-data claims on fixed dense OVSS-style features.
