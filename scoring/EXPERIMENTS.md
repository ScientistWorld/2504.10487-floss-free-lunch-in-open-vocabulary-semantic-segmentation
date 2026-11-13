# Experiment Overview

All three experiments use the same frozen open-vocabulary segmentation model, NACLIP features on ADE20K with 150 classes and CLIP's 80 prompt templates. They score the same method procedure, which the evaluator runs on unlabeled images it chooses. No method sees the scored images or any label.

## ade20k_naclip_core

Can a frozen model be made to segment a target dataset better using only that dataset's unlabeled images, with no training and no labels? The method adapts on a pool of 2000 unlabeled images and is scored by mIoU on held-out images (hidden test: the 2000 ADE20K validation images). The comparison point is the model's default classifier (`template_average`). This is the paper's headline claim.

## ade20k_naclip_low_data

Does the adaptation still help when only a handful of unlabeled images is available, and how quickly does it approach the full-pool result? The evaluator gives the method 4, 16 or 64 unlabeled images (four random draws each) and scores each run on the same 500 held-out images. The metric is the mean mIoU over the 12 runs. Each run starts from scratch, so knowledge from the full pool cannot be carried over. A method that cannot handle tiny adaptation sets, or that is unstable across draws, loses here even if it is strong with the full pool.

## ade20k_naclip_expert_quality

Are the class-specific decisions a method makes actually the right ones? For every class the method names the 4 prompt templates it trusts most. A template counts as correct when, used alone, it segments that class better than the default classifier on the held-out labels. The score is the percentage of correct choices. It checks that segmentation gains come from identifying genuinely better class-wise text representations, and it rises only with better label-free identification. The `template_average` row is the chance level of a random choice.
