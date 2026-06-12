# Evaluation Targets

## Primary: synthetic_ovss

Tests whether a text-driven dense segmentation system improves over the default averaged-template classifier on a held-out dense-feature benchmark.

- **Metric**: mIoU (higher is better)
- **How to run evaluation**: run `bash scripts/reproduce.sh`, then `bash scripts/evaluate.sh`.

## Constraint: template_budget

Tests whether the adapted system preserves the paper's text-side constraint: it uses a small number of class experts selected from the fixed 80-template pool rather than adding a larger prompt set or a trained model.

- **Metrics**: num_experts_per_class and num_templates (lower is better subject to maintaining mIoU)
- **How to run evaluation**: the evaluator reads the saved expert-index artifact and the dataset template count.

## Primary: low_data_synthetic_ovss

Tests whether the same segmentation improvement remains when the adaptation step has access to only a few unlabeled selection images.

- **Metric**: mIoU (higher is better)
- **How to run evaluation**: run `bash scripts/reproduce.sh`, then inspect the `low_data_synthetic_ovss` entry written by `scripts/evaluate.sh`.

## Diagnostic: expert_quality_synthetic_ovss

Tests whether the selected class-specific template sets overlap with oracle class-specific template sets in the benchmark. This is a diagnostic target for the paper's expert-identification claim, not a permission to use validation labels during adaptation.

- **Metric**: expert_precision_pct (higher is better)
- **How to run evaluation**: run `bash scripts/reproduce.sh`, then inspect the `expert_quality_synthetic_ovss` entry written by `scripts/evaluate.sh`.
