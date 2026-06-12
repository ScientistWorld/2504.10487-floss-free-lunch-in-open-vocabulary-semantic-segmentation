# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-06-12 07:22] - core_claim
- Audited the inherited FLOSS implementation and confirmed it implements the paper's actual dense-feature computation: average-template baseline, entropy-based class-expert selection, top-4 experts, and highest-confidence expert fusion.
- Confirmed evaluation is separate from `method/` and reads saved prediction artifacts.

### [2026-06-12 07:22] - core_claim
- Added an expert-quality diagnostic experiment to score the paper's secondary claim that entropy-selected experts overlap with oracle class experts.
- Submitting a CPU container job to regenerate `scoring/scores.json` and validate the expanded benchmark before claiming a higher milestone.

### [2026-06-12 07:28] - secondary_claims
- CPU container job `77ce9f9e-466` completed successfully and `python validate.py --compare` passes.
- Reproduced the core mIoU improvement, the low-data improvement, and a diagnostic expert-quality result showing nontrivial overlap between selected and oracle class experts.
