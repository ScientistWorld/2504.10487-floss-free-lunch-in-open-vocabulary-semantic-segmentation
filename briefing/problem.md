# Problem Definition

## Research Question

Can a frozen, text-driven open-vocabulary semantic segmentation system be improved for a target class set using only unlabeled images from that target domain, while preserving the ability to evaluate on unseen images and without adding supervised training?

## Why It Matters

OVSS systems are attractive because class vocabularies can be changed at inference time, but dense labels are expensive and domain-specific prompt or model tuning can be brittle. A successful method would adapt segmentation behavior to a new dataset or domain with minimal annotation and compute cost.

## Success Criteria

A successful solution should raise semantic segmentation quality on held-out images, work across multiple CLIP-based OVSS backbones and datasets, avoid using ground-truth masks during adaptation, require no parameter training for the adaptation step, and keep preprocessing and evaluation consistent with the baseline model.

## Scope of This Gym

The gym instantiates the problem on one of the paper's own settings: a frozen NACLIP (CLIP ViT-B/16) model on ADE20K (150 classes), with CLIP's 80 ImageNet prompt templates as the only text-side material. A method receives unlabeled image features and the per-template class embeddings. It is judged on held-out validation images it never sees, through the procedure interface described in `briefing/evaluation.md`.
