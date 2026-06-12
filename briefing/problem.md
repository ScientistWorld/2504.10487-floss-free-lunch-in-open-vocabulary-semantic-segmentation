# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here -- only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

Can a frozen, text-driven open-vocabulary semantic segmentation system be improved for a target class set using only unlabeled images from that target domain, while preserving the ability to evaluate on unseen images and without adding supervised training?

## Why It Matters

OVSS systems are attractive because class vocabularies can be changed at inference time, but dense labels are expensive and domain-specific prompt or model tuning can be brittle. A successful method would adapt segmentation behavior to a new dataset or domain with minimal annotation and compute cost.

## Success Criteria

A successful solution should raise semantic segmentation quality on held-out images, work across multiple CLIP-based OVSS backbones and datasets, avoid using ground-truth masks during adaptation, require no parameter training for the adaptation step, and keep preprocessing and evaluation consistent with the baseline model.
