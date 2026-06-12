# Overview

- **Paper ID:** 2504.10487
- **Title:** FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation
- **Domain:** Computer Vision / Open-Vocabulary Semantic Segmentation
- **TL;DR:** FLOSS improves frozen CLIP-based open-vocabulary semantic segmentation by selecting class-specific prompt templates from unlabeled images and fusing their predictions without labels or training.

## Short Summary

The paper studies a common OVSS practice: representing each class by averaging CLIP text embeddings from the standard 80 ImageNet prompt templates. It observes that individual templates can be much better for particular classes than the averaged prompt classifier. FLOSS estimates these class-specific template experts using unlabeled images, then combines their predictions into a final segmentation map. The method is presented as plug-and-play for existing frozen CLIP-based OVSS models and is evaluated across in-domain, shifted-domain, and low-data settings.

## Key Results

- FLOSS improves mIoU for MaskCLIP, NACLIP, and CLIP-DINOiser on Cityscapes, PASCAL VOC 20, PASCAL Context 59, ADE20K, and COCO-Stuff.
- Experts selected on Cityscapes transfer to shifted driving datasets including ACDC weather/night subsets, BDD-100K, and Mapillary.
- The method remains useful when expert selection uses very few unlabeled images, including single-image selection in the driving-domain shift experiment.
