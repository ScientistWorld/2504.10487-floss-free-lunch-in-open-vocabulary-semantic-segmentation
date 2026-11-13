# Scientist Constraints

## Model and Features

Use the provided frozen NACLIP patch features and the 80 × 150 template embedding pool. Do not recompute features with another image model, train or fine-tune any encoder or segmentation head, or learn prompts. Do not add text embeddings from new prompts. The task is text-side adaptation of a fixed model from unlabeled images.

## Data

- **No labels in adaptation.** `adapt` may use only the unlabeled features it is given. The dev labels (`data/downloads/naclip/dev_labels.npy`, `dev_oracle.npz`) exist to measure progress and to choose global settings such as a hyperparameter. They must not drive per-class or per-template decisions: hard-coding template lists, class priors or layouts derived from labels is out of bounds.
- **Only the images you are given.** Each run must adapt only on the features passed to `adapt` in that run. Do not bring in features or statistics from other images, whether read from disk, shipped in the method directory (for example per-template entropy tables precomputed on the visible pool and reused in low-data runs) or carried between runs. `predict` must not use anything about the scored images beyond the features passed to it, and must not adapt on the scored images as it goes.
- **Enforced by the evaluator:** every run is a fresh process; a procedure may read only files inside its own directory (and Python source), cannot write files that outlive the run and cannot start processes. On the hidden test the labels are unreadable while it runs. Data shipped *inside* the method directory is not inspected by the evaluator, so the label and only-the-given-images rules above still apply to it and are checked separately.
- **Nothing under `eval/test/`** may be read or modified by a method. No external labeled segmentation data.

## Procedure

- Deliver `outputs/<name>/procedure.py` with `adapt(features_list, text_embeddings)`, `predict(state, features) -> [h, w, K]` and `experts(state) -> int [K, 4]`, as described in `briefing/evaluation.md`. Keep any helper files inside that directory.
- Template and class order are shuffled in every call. Do not undo the shuffling by matching the given embeddings against stored copies of the embeddings (on disk or shipped in the method directory), and do not rely on index identity.
- Keep the class vocabulary, template pool, splits, resolution and evaluators fixed. Score with `scripts/evaluate_train.sh` while developing; the hidden test uses `scripts/evaluate_test.sh` unchanged.

## Compliance Checklist

- [ ] Uses only the provided frozen features and the 80-template embedding pool
- [ ] `adapt` uses only the unlabeled features it receives in that run; no labels, no other files, no shipped statistics from other images
- [ ] No per-class or per-template choices derived from dev labels or oracle tables
- [ ] Never reads or modifies `eval/test/`
- [ ] `outputs/<name>/procedure.py` implements `adapt`, `predict` and `experts`
- [ ] Evaluators and data left unchanged
