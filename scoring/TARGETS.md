# Evaluation Targets

## Primary: clip_dinoiser_cityscapes_core

Tests the paper's CLIP-DINOiser Cityscapes improvement target using this workspace's held-out synthetic dense-feature substitute.

- **Metric**: mIoU (higher is better)
- **How to run evaluation**: run `bash scripts/reproduce.sh`, then `bash scripts/evaluate.sh`.

## Constraint: template_budget

Tests whether the adapted system preserves the benchmark constraint: it uses only a small class-specific artifact budget drawn from the fixed 80-item pool rather than adding a larger prompt set or a trained model.

- **Metrics**: num_experts_per_class and num_templates (lower is better subject to maintaining mIoU)
- **How to run evaluation**: the evaluator reads the saved selected-artifact file and the dataset pool size.

## Primary: clip_dinoiser_cityscapes_low_data

Tests the paper's low-data CLIP-DINOiser Cityscapes target using this workspace's synthetic dense-feature substitute with only a few unlabeled selection images.

- **Metric**: mIoU (higher is better)
- **How to run evaluation**: run `bash scripts/reproduce.sh`, then inspect the `clip_dinoiser_cityscapes_low_data` entry written by `scripts/evaluate.sh`.

## Diagnostic: expert_quality_clip_dinoiser_cityscapes_core

Tests the paper's expert-quality target using hidden oracle artifact sets in the synthetic substitute. This is a diagnostic target for class-specific artifact quality, not a permission to use validation labels during adaptation.

- **Metric**: expert_precision_pct (higher is better)
- **How to run evaluation**: run `bash scripts/reproduce.sh`, then inspect the `expert_quality_clip_dinoiser_cityscapes_core` entry written by `scripts/evaluate.sh`.
