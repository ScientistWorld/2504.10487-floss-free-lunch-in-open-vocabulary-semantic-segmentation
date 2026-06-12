## Stage 2 Evaluation Design Handoff

Checked:
- Read `eval/train/evaluate.py`, `eval/test/evaluate.py`, `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, and `scoring/reference.json` against the train/test anti-shortcut requirements.
- Confirmed both shell scripts enumerate method outputs from `outputs/*` or explicit `NAME=path` specs instead of hardcoding a method name.
- Confirmed `scripts/evaluate_test.sh` invokes `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Confirmed the evaluators score saved method artifacts (`predictions.npy`, `experts.npy`) rather than regenerating the method inline.
- Grepped for train/test cross-imports and forbidden paper-agreement metric names.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` are absent.
- Ran `bash -n` on both train/test evaluation scripts.

Changed:
- Fixed the expert-quality leakage pattern: train now scores overlap on even class ids, while test scores held-out odd class ids. The prior design used all classes in both slices, so a method could optimize the visible expert-quality metric directly.
- Added `rare_class_mIoU` to both segmentation experiments to penalize background/common-region shortcuts.
- Added `boundary_pixel_accuracy` to both segmentation experiments to penalize coarse mask shortcuts that can keep reasonable global mIoU or pixel accuracy.
- Updated `mean_iou` to return a stable class-indexed per-class vector so class-specific triangulation metrics cannot accidentally read the wrong class.
- Added required `higher_is_better`, `coefficient`, and `description` metadata for every scored metric in `scoring/reference.json`, including the existing constraint metrics.

Runtime concerns for the next iteration:
- The new metrics were only statically checked, per this iteration's no-experiment rule. The next runner should verify that `rare_class_mIoU` and `boundary_pixel_accuracy` are present in both score files and are neither saturated nor constant across baseline/proposed outputs.
- The validation pool is compact: train/test split 36 validation images into 18/18. This is acceptable for a lightweight gym, but score variance should be watched if future changes make the synthetic benchmark noisier.
- The expert-quality split uses 3 visible and 3 held-out classes in the current 6-class synthetic benchmark. If the dataset class count changes, confirm both class subsets remain non-empty and comparable.
