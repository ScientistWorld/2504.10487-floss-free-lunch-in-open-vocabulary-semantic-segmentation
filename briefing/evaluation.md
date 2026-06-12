# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure -- do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

The primary task metric is mean Intersection over Union (mIoU) over the evaluated semantic classes; higher is better. Per-class IoU is useful for diagnosing whether gains are broad or concentrated in a few categories.

Constraint metrics should report whether the adaptation used ground-truth masks, whether model parameters were trained, the number of unlabeled images used for adaptation, and inference overhead relative to the corresponding baseline. For the paper's own mechanism analyses, expert-selection quality is measured as the overlap between estimated class experts and oracle class experts, but that requires labels and should be treated as diagnostic rather than the main benchmark score.

## Baselines and Targets

Main reported baselines are MaskCLIP, NACLIP, and CLIP-DINOiser with the standard averaged CLIP prompt-template classifier. FLOSS is reported as an add-on to each baseline.

Headline validation-set mIoU targets from the paper include:

- MaskCLIP to MaskCLIP+FLOSS average across Cityscapes, VOC20, PC59, ADE20K, and COCO-Stuff: 28.7 to 29.3.
- NACLIP to NACLIP+FLOSS average across those datasets: 38.2 to 39.0.
- CLIP-DINOiser to CLIP-DINOiser+FLOSS average across those datasets: 38.5 to 39.7.
- On Cityscapes with CLIP-DINOiser, the main reported gain is 31.3 to 34.6 mIoU.

Robustness targets include Cityscapes-selected adaptation evaluated on ACDC night/fog/rain/snow, BDD-100K, and Mapillary with the Cityscapes label set. Low-data targets vary the number of unlabeled selection images and evaluate on held-out validation images.

## Evaluation Protocol

Evaluate on the validation split of each dataset, using the class names and ignore-label handling expected by the OVSS benchmark. The main datasets are Cityscapes, PASCAL VOC 20, PASCAL Context 59, ADE20K, and COCO-Stuff. Domain-shift evaluation selects adaptation data from Cityscapes and tests on ACDC conditions, BDD-100K, and Mapillary with the shared semantic classes.

For fair comparison, hold the base OVSS model, CLIP backbone, image resolution, sliding-window settings, and text class list fixed between the baseline and adapted system. The paper uses ViT-B/16 for the main MaskCLIP, NACLIP, and CLIP-DINOiser results, with additional ViT-L/14 checks for NACLIP. Any low-data evaluation should average over multiple random selections of adaptation images and keep validation labels unavailable to the adaptation procedure.
