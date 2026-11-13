# Evaluation

## Setting

The paper evaluates on Cityscapes, PASCAL VOC 20, PASCAL Context 59, ADE20K and COCO-Stuff with three frozen CLIP-based OVSS models (MaskCLIP, NACLIP, CLIP-DINOiser). This gym uses one of the paper's own settings: **NACLIP (CLIP ViT-B/16, no refinement) on ADE20K (150 classes)** with CLIP's 80 ImageNet prompt templates. The backbone is frozen and its outputs are precomputed:

- **Visual features.** For each image, a grid of 512-d patch embeddings `[h, w, 512]` (float16). Images are resized to a short side of 336 and processed with 224x224 windows at stride 112 (long side rounded to 224 + 112j); each patch feature is the average of the L2-normalised window embeddings covering it. `features @ text_embedding` is the backbone's cosine logit for that class.
- **Text embeddings.** `[80, 150, 512]`: one L2-normalised embedding per (template, class name). The baseline classifier averages the 80 templates per class and renormalises.
- **Data.** Method-visible: 2000 unlabeled ADE20K training images (`data/downloads/naclip/adapt_*`) and 500 labeled ADE20K training images used as the dev set (`dev_*`). Hidden: a different set of 2000 unlabeled training images and the 2000 ADE20K validation images with labels; these exist only under `eval/test/` and are used by the hidden test.

## What a method provides

A method is a directory `outputs/<name>/` containing `procedure.py` (plus any files it needs next to it). The evaluators import it in a separate process and call:

```python
state  = adapt(features_list, text_embeddings)
#   features_list: list of [h_i, w_i, 512] float16 arrays (unlabeled images)
#   text_embeddings: [80, K, 512] float32
scores = predict(state, features)          # features [h, w, 512] -> scores [h, w, K]
ids    = experts(state)                    # optional: int array [K, 4]
```

- `adapt` receives only unlabeled features chosen by the evaluator. No labels exist at test time.
- `predict` is called once per scored image and must return finite class scores on the feature grid. The evaluator upsamples them bilinearly to the label resolution (8 pixels per patch side) and takes the argmax.
- `experts` names, for every class, the 4 templates (indices into the given template axis, distinct within a row) the method considers the most reliable for that class. It is used only by the expert-quality experiment.
- **Template and class order are shuffled in every evaluator call.** Indices refer to the order passed to `adapt`; the evaluator maps them back. A method must not depend on the identity of a template or class index.

### Isolation of each run

- **Every run is a fresh process.** The core run and each of the 12 low-data runs import `procedure.py` anew in their own worker process, so nothing computed in one run (module-level state, caches) is available in another.
- **File access.** While a procedure runs, it may read files only inside its own `outputs/<name>/` directory (and Python source files). It may not read anything else under the workspace (`data/`, `eval/`, other method directories, ...), may not write files (except into the run's own scratch directory, which is deleted when the run ends), and may not start other processes. A blocked access raises `PermissionError`; if the procedure does not handle it, the run fails. On the hidden test the evaluation labels are additionally unreadable while any procedure runs.

`baseline/template_average.py` is a complete minimal example. The frozen model's default averaged-template classifier is built into both evaluators and scored as `template_average`.

## Experiments and metrics

| Experiment | What the evaluator does | Metric (higher is better) |
|---|---|---|
| `ade20k_naclip_core` | `adapt` on the full unlabeled pool (2000 images), `predict` every scored image | mIoU over 150 classes |
| `ade20k_naclip_low_data` | 12 runs: `adapt` on 4, 16 or 64 pool images (4 seeds each), `predict` 500 fixed scored images | mean mIoU over the 12 runs |
| `ade20k_naclip_expert_quality` | `experts(state)` after the core `adapt` | % of the reported experts that are true experts |

- **mIoU.** For each class, IoU = intersection / union accumulated over all scored images. Pixels labelled "other" (ADE20K label 0) are ignored. Classes absent from both the labels and the predictions are skipped, and the 150-class mean is reported, as in the paper.
- **True experts** (paper Eq. 4): template m is a true expert of class k when the single-template classifier of m alone reaches a higher IoU on class k than the averaged classifier, on the scored labels. Expert precision (paper Eq. 8) is 100 × |reported ∩ true| / 4, averaged over the classes that have at least one true expert. The `template_average` row reports the chance level of a uniformly random selection.

## Train vs hidden test

`scripts/evaluate_train.sh` runs the protocol on visible data: adaptation on the visible pool, scoring on the 500 dev images. It writes `scoring/scores_train.json`. `scripts/evaluate_test.sh` runs the same protocol on the hidden pool and the 2000 validation images, with different seeds and shuffles, and writes `scoring/scores_test.json`. Both take no arguments and score every directory under `outputs/`.

## Failure rules

A method whose directory has no `procedure.py`, whose procedure raises in any run (including on a blocked file access), times out (4 h per method over all of its runs), or returns scores of the wrong shape or with non-finite values is a **failed run**. A run that fails is retried once before the method is failed. It gets mIoU 0 and expert precision 0 in every experiment. An invalid or missing `experts()` output scores 0 in the expert-quality experiment only. The hidden test deletes the previous score file before it starts, and refuses to score (all methods failed) if the hidden labels no longer match their recorded hash.

## Paper reference numbers

NACLIP on ADE20K: 17.4 mIoU with the averaged classifier, 18.4 after adaptation. Expert quality is 57%. This gym is scaled relative to the paper: 2000 instead of 20,210 adaptation images, scoring at half of the 336-short-side resolution, and a patch grid shared across windows. So its absolute numbers are close to the paper's, but not identical.
