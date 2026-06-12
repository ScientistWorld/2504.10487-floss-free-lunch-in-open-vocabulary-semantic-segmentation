# Experiment Overview

## synthetic_ovss

Tests whether the reproduced system improves dense open-vocabulary-style segmentation quality on held-out images while using the same frozen feature and class-embedding inputs as the baseline. The experiment exists to validate the paper's central claim in a portable benchmark where all data can be regenerated without external gated datasets.

## low_data_synthetic_ovss

Tests whether the reproduced system still improves held-out dense segmentation quality when adaptation is restricted to a very small unlabeled selection subset. This mirrors the paper's low-data question without exposing the internal mechanism used to adapt.
