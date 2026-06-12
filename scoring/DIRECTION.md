# Research Direction

## Research Problem

This benchmark is about label-free, training-free adaptation for frozen CLIP-based open-vocabulary semantic segmentation. The core question is how to improve dense segmentation for a target class set and domain when only unlabeled images are available for adaptation.

## Core Contribution

The paper contributes a text-side adaptation strategy for a fixed OVSS model. Future work should stay within that research area: improving how the text-conditioned classifier or prediction aggregation is adapted from unlabeled images while preserving the frozen base segmenter and no-label setting.

## Approach Scope

Scientists may change how unlabeled images are used to select, weight, generate, rank, or combine text-side class representations. They may introduce alternative unsupervised confidence, consistency, diversity, calibration, or domain-fit criteria, provided these criteria do not consume segmentation labels. They may also modify prediction fusion or fallback behavior as long as the method remains a plug-in adaptation of the fixed OVSS backbone.

## Out of Bounds

- Replacing the fixed OVSS backbone with a supervised segmentation model or a different architecture family.
- Training or fine-tuning the image encoder, text encoder, segmentation head, or learned prompts during the adaptation phase.
- Using target validation labels, oracle per-class IoU, or ground-truth masks to select adaptation decisions.
- Improving results by changing the benchmark class vocabulary, dataset split, preprocessing, or ignore-label policy.
- Treating the task as fully supervised, weakly supervised, or test-time training segmentation instead of label-free adaptation of a frozen open-vocabulary model.
