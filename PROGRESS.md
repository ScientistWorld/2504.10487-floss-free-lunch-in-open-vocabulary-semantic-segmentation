# Progress

## What Works

- Implemented FLOSS on dense OVSS-style cached features: default averaged-template classifier, per-template entropy ranking on unlabeled training features, top-4 class expert selection, per-class expert classifier construction, and expert fusion with default fallback.
- Implemented a template-averaging baseline and reusable evaluator that reads saved predictions from `outputs/`.
- Added an expert-quality diagnostic evaluator that reads `outputs/<method>/experts.npy` and compares selected class experts to oracle class-specific template sets stored in the benchmark data.
- `bash scripts/reproduce.sh && python validate.py --compare` runs in the managed container; the login-node Python lacks NumPy, so local reruns are intentionally left to the container setup.

## Results

- Paper reference used for the core claim: CLIP-DINOiser on Cityscapes improves from 31.3 to 34.6 mIoU with FLOSS and uses 4 experts per class from the 80-template pool.
- Verified container result (`synthetic_ovss`, job `a289cd39-9cc`): template averaging gets 33.073 mIoU; FLOSS gets 37.596 mIoU with 4 experts per class. This supports the core claim on a synthetic dense-feature substitute.
- Verified container result (`low_data_synthetic_ovss`, job `77ce9f9e-466`): template averaging gets 33.073 mIoU; FLOSS with expert selection from only 4 unlabeled images gets 38.249 mIoU. This supports the low-data secondary claim.
- Verified container result (`expert_quality_synthetic_ovss`, job `77ce9f9e-466`): FLOSS gets 37.500% expert precision and FLOSS low-data gets 41.667% expert precision against oracle class-expert sets. This is below the paper's 50% average reference but demonstrates nontrivial recovery of true class experts without labels.

## Remaining

- Port the benchmark to real Cityscapes/CLIP-DINOiser outputs if the gated dataset and backbone artifacts become available.
- Add shifted-domain real-data evaluation for ACDC/BDD/Mapillary if those datasets become available.

## Issues

- Cityscapes is gated and was not present in shared datasets, so the exact main benchmark cannot be downloaded automatically.
- The workspace root `README.md` is a root-owned character device and git reports it as modified; it is unrelated to this reproduction and is not staged.

## Deviations from Paper

<!-- Do not edit this section unless you deviated from the paper's original setup.
     For each deviation, document: what the paper used, what you substituted, and why.

     ACCEPTABLE deviations (data/resource/scale substitutions — milestones still valid):
       - Replaced a proprietary dataset with a public one
       - Used a smaller model of the same architecture
       - Scaled down training (fewer epochs, smaller batch)
       - Used a different library for the same algorithm

     UNACCEPTABLE deviations (algorithm substitutions — milestones above 'none' are void):
       - Replaced the paper's algorithm with a fundamentally different one
       - Wrote a generic model trained to approximate the method's outputs
         instead of implementing the method's computational steps

     The test: could someone reading your code identify which algorithm from
     the paper you implemented? If not, you replaced the method. -->

- Data/resource deviation: the paper evaluates on Cityscapes, PASCAL VOC, PASCAL Context, ADE20K, and COCO-Stuff using CLIP-based OVSS backbones. This reproduction uses a deterministic synthetic dense-feature benchmark because the central Cityscapes benchmark is gated and unavailable in the shared cache. The substitute stores the same type of inputs FLOSS consumes after a frozen OVSS backbone has run: per-pixel normalized visual embeddings, per-template normalized text embeddings, and held-out segmentation labels for scoring.
- Scale deviation: the benchmark uses 6 classes, 80 templates, 48 unlabeled selection images, and 36 validation images. The FLOSS procedure itself is unchanged: no labels for expert selection, entropy ranking over single-template predictions, top-4 templates per class, expert classifier construction, and highest-confidence fusion.
