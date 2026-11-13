# FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening). The runs listed below were made on that earlier version.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2504.10487"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2504.10487-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2504.10487](https://arxiv.org/abs/2504.10487) |
|:---|:---|
| **Domain** | Computer Vision / Open-Vocabulary Semantic Segmentation |
| **TL;DR** | FLOSS improves frozen CLIP-based open-vocabulary semantic segmentation by selecting class-specific prompt templates from unlabeled images and fusing their predictions without labels or training. |

## Summary

The paper studies a common OVSS practice: representing each class by averaging CLIP text embeddings from the standard 80 ImageNet prompt templates. It observes that individual templates can be much better for particular classes than the averaged prompt classifier. FLOSS estimates these class-specific template experts using unlabeled images, then combines their predictions into a final segmentation map. The method is presented as plug-and-play for existing frozen CLIP-based OVSS models and is evaluated across in-domain, shifted-domain, and low-data settings.

## Problem

Can a frozen, text-driven open-vocabulary semantic segmentation system be improved for a target class set using only unlabeled images from that target domain, while preserving the ability to evaluate on unseen images and without adding supervised training?

## The paper's method

FLOSS introduces training-free, label-free class-wise prompt template selection for open-vocabulary semantic segmentation. Its central idea is that each semantic class may be better segmented by a small subset of individual CLIP prompt templates than by the standard average over all templates.

## Key results of the paper

- FLOSS improves mIoU for MaskCLIP, NACLIP, and CLIP-DINOiser on Cityscapes, PASCAL VOC 20, PASCAL Context 59, ADE20K, and COCO-Stuff.
- Experts selected on Cityscapes transfer to shifted driving datasets including ACDC weather/night subsets, BDD-100K, and Mapillary.
- The method remains useful when expert selection uses very few unlabeled images, including single-image selection in the driving-domain shift experiment.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-terra-xhigh` | +198.4 |
| `gpt-5.6-sol-xhigh` | +101.4 |
| `gpt-5.6-luna-xhigh` | +70.1 |
| `gpt-5.5-xhigh` | +30.1 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2504.10487 --mode innovate --agent claude-code
sw evaluate 2504.10487
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

