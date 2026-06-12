# Experiment Overview

## synthetic_ovss

Tests whether an adapted dense open-vocabulary-style segmentation system improves held-out segmentation quality while using the same fixed feature inputs and class vocabulary as the baseline. The experiment exists to validate the central claim in a portable benchmark where all data can be regenerated without external gated datasets.

## low_data_synthetic_ovss

Tests whether the reproduced system still improves held-out dense segmentation quality when adaptation is restricted to a very small unlabeled selection subset. This mirrors the paper's low-data question without exposing the internal mechanism used to adapt.

## expert_quality_synthetic_ovss

Tests whether optional adaptation artifacts align with hidden oracle artifact sets in the benchmark. The experiment is diagnostic: it helps future agents understand whether gains are accompanied by meaningful class-specific adaptation decisions rather than only final-mask changes.
