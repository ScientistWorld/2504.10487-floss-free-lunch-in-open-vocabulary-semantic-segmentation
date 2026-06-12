# Experiment Overview

## clip_dinoiser_cityscapes_core

Tracks the paper's CLIP-DINOiser Cityscapes core claim while evaluating on this workspace's held-out synthetic dense-feature substitute. It tests whether adaptation improves segmentation quality while using the same fixed feature inputs and class vocabulary as the baseline.

## clip_dinoiser_cityscapes_low_data

Tracks the paper's low-data CLIP-DINOiser Cityscapes claim on the synthetic dense-feature substitute. It tests whether the reproduced system still improves held-out dense segmentation quality when adaptation is restricted to a very small unlabeled selection subset.

## expert_quality_clip_dinoiser_cityscapes_core

Tracks the paper's expert-quality diagnostic on hidden oracle artifact sets in the synthetic substitute. It helps future agents understand whether gains are accompanied by meaningful class-specific adaptation decisions rather than only final-mask changes.
