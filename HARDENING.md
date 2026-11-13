# Hardening report

`REDTEAM.md` audited the earlier gym, which used synthetic 6-class dense features with a public generator. This iteration did the work in two steps:

1. **Rebuild** (commit `0b1dbb8`, job `ec864c51-dcb`). The gym now uses one of the paper's own settings: frozen NACLIP (OpenAI CLIP ViT-B/16, no PAMR) features of ADE20K (150 classes), with CLIP's 80 ImageNet templates.
   - A method is a `procedure.py` (`adapt` / `predict` / `experts`). Both evaluators run it on inputs they choose. The hidden adaptation pool and the 2000 validation images exist only under `eval/test/`.
   - The evaluator recomputes every number from per-pixel class maps.
   - FLOSS was restored to the paper's and the official code's settings: N = 4, per-class entropy on the patches each template assigns to the class, highest-confidence fusion with the W_CLIP fallback, and logit scale 100.
2. **Audit and fixes** (commit `c7c6195`, verification job `60d5c371-451`).
   - The rebuilt gym was probed with 24 cases.
   - Three shortcuts still paid, and two further ones were constructed. All were fixed.
   - Everything was rerun through the final evaluators.

All numbers below are from real runs of the unmodified `eval/test/evaluate.py`. Per-case logs and JSON:
- **After:** `redteam/hardening/results/` (`summary.md`, `summary.json`, `test/`, `train/`, `before/`, `scenarios/`).
- **Rebuilt gym before this pass:** `redteam/hardening/results_before_fix/`.
- **Probes:** `redteam/hardening/cases/`.

**ΔR** below is the illustrative rank change against `floss`. It sums, over experiments and metrics, weight × |coefficient| × (improvement in the better direction) / (std of that metric over the honest cohort of this run). The cohort std is core 0.510, low-data 0.146 and expert 8.00. The ranking system normalises by the future cohort, so magnitudes will differ. Signs and orderings of componentwise gains will not.

## Does a better result score higher?

Yes. Honest label-free variants that improve the method rank above the reference. The paper's ablations and degraded variants rank below it. Simple baselines rank clearly lower.

| Variant (hidden test) | core mIoU | low-data mIoU | expert % | core Δ, paired bootstrap 95% CI | low Δ (runs better /12) | ΔR before this pass | ΔR after |
|---|---:|---:|---:|---|---|---:|---:|
| **floss (reference)** | 18.58 | 17.35 | 60.5 | — | — | 0 | 0 |
| v_avg_prob: paper Fig. 4a proxy | 18.91 | 17.26 | 63.7 | +0.32 [+0.20, +0.44] | −0.08 (3) | +0.305 | **+0.305** |
| v_min_support: ignore templates with tiny class support | 18.68 | 17.37 | 62.0 | +0.10 [+0.06, +0.14] | +0.02 (7) | +0.179 | **+0.179** |
| v_min_count25 (new, low-data aimed) | 18.59 | 17.19 | 61.0 | +0.01 [−0.00, +0.02] | −0.16 (4) | — | −0.251 |
| v_min_count100 (new) | 18.72 | 17.05 | 62.7 | +0.13 [+0.08, +0.18] | −0.30 (3) | — | −0.311 |
| w_average_all: paper Tab. 6 ablation | 18.19 | 17.46 | 60.5 | −0.38 [−0.49, −0.27] | +0.11 (7) | −0.274 | −0.274 |
| w_top1: N = 1 | 18.47 | 17.19 | 60.5 | −0.11 [−0.30, +0.09] | −0.16 (2) | −0.407 | −0.407 |
| w_default_conflict: paper Tab. 6 ablation | 18.29 | 17.25 | 60.5 | −0.28 [−0.35, −0.20] | −0.10 (3) | −0.512 | −0.512 |
| template_average: paper baseline | 17.47 | 17.14 | 39.4* | −1.07 [−1.30, −0.83] | −0.21 (3) | −2.059 | −2.059 |
| w_random_experts: matched 4-expert budget | 17.04 | 16.87 | 38.7 | −1.51 [−1.76, −1.27] | −0.48 (1) | −3.040 | −3.040 |

\* chance level of a random 4-template selection.

- "Before this pass" is the rebuilt gym's evaluator, rescored here with the same cohort std. Honest variants are deterministic and get identical outputs from both evaluators.
- On `REDTEAM.md`'s synthetic gym the stronger variant scored +0.66 and the random-expert ablation −0.99. That ordering was right, but a fixed spatial mask beat both (see below), so the score did not reward better research there.
- The core improvements of v_avg_prob and v_min_support exceed the paired noise: their bootstrap CIs exclude 0.
- Seed spread: FLOSS on three random halves of the pool gives core 18.51–18.64 (range 0.13) and expert 58.3–60.7.
- Train and test agree on direction for every variant. For example v_avg_prob scores 17.16 vs 17.06 on train and 18.91 vs 18.58 on test.

## Shortcuts: fix and score before → after

### Found on the rebuilt gym, fixed in this pass

| Shortcut probe | What it does | Before (core / low / expert, ΔR) | Fix | After |
|---|---|---|---|---|
| `ub_val_oracle_lookup` | reads the hidden `val_oracle.npz` from inside `adapt`; picks true top-4 experts per class | 20.74 / 19.93 / 100, **+7.69** | File guard (1) and sealed answers (2) | import blocked → failed run: 0 / 0 / 0 |
| `sc_hidden_labels_raw_read` (new) | reads hidden `val_labels.npy` through `torch.from_file`, a C-level read a Python hook cannot see; predicts the labels in the core run | 78.32 / 17.35 / 60.5, **+70.2** (pre-fix evaluator) | Sealed answers (2) | read fails → plain FLOSS: 18.58 / 17.35 / 60.5, ΔR 0 |
| `sc_dev_oracle_lookup` | reads visible dev labels/oracle and the visible text-embedding file to undo the shuffle; per-class label-chosen experts | 19.74 / 19.14 / 84.5, **+4.89** | File guard (1) | blocked → failed run: 0 / 0 / 0 |
| `sc_full_pool_lowdata` | in low-data runs, reads the 2000-image visible pool from disk instead of the images given | 18.56 / 17.87 / 59.7, **+0.86** | File guard (1) | blocked → failed run: 0 / 0 / 0 |
| `sc_cache_across_runs` (new) | keeps the full-pool experts from the core run in module memory and reuses them in the 12 low-data runs | 18.58 / 18.01 / 60.5, **+1.13** (pre-fix evaluator) | Fresh process per run (3) | plain FLOSS, ΔR 0 |
| `sc_file_cache_across_runs` (new) | same, carried through `/tmp` | 18.58 / 18.01 / 60.5, **+1.13** (pre-fix evaluator) | Process per run (3) and write guard (1) | plain FLOSS, ΔR 0 |
| `sc_dev_oracle_hardcoded`, `sc_layout_prior_lookup` | label-derived choices / layout read from dev files | 17.11 and 0.49 core (already non-paying) | File guard (1) | failed runs |

The fixes (`eval/test/worker.py`, `eval/test/evaluate.py`; the train side mirrors 1 and 3):

1. **File/process guard.** Before `procedure.py` is imported, the worker installs a Python audit hook.
   - The procedure may read only files inside its own `outputs/<name>/` directory, plus `.py` source files.
   - It may write only to its run's scratch directory, which is deleted after the run.
   - It may not chmod, rename or link workspace files, and may not start processes.
   - A blocked access raises `PermissionError`; if unhandled, it fails the run. The guard logic was unit-tested on 21 cases on the login node and behaved as designed in the container: honest variants were unaffected and blocked probes failed.
2. **Sealed hidden answers.** While any worker runs, the evaluator sets `val_labels.npy`, `val_oracle.npz` and the raw validation annotation directory to mode 000. It restores them after each run, and at start-up it repairs seals that a killed run left behind.
   - Scenario C left the answers sealed and a forged 100-valued score file behind. The next evaluation unsealed the answers, overwrote the forged file and scored the crashing method 0.
   - Modes after the job: 660 / 660 / 2770.
3. **One fresh worker process per run.** The core run and each of the 12 low-data runs are separate processes, so nothing computed on the full pool reaches a low-data run. Failed runs are retried once, which absorbs transient GPU errors (the job's GPU was shared, with 71 GB in use by another process).

Rechecked integrity scenarios (`results/scenarios/`):
- Edited hidden labels → every method is scored as failed and the exit code is 1.
- Edited oracle table → recomputed from the hash-verified labels, scores unchanged.
- Stale forged score file plus crashing method → file regenerated with worst values.

### From `REDTEAM.md` (synthetic gym), closed by the rebuild

| Shortcut | Before (synthetic gym) | After (current gym) |
|---|---|---|
| rebuild answers from the public generator | 100 / 100 mIoU, ΔR +3.27 | no generator; real ADE20K labels exist only in `eval/test/`, unreadable during runs |
| fixed spatial layout mask, no image inference | 56.61 vs 37.84 ref, ΔR +0.72 | `sc_layout_prior_lookup` scored 0.49 core before this pass and fails now |
| planted expert IDs written as diagnostic | expert 100%, ΔR +0.40 | true experts are Eq. 4 on hidden labels; no formula to guess |
| diagnostic-only registration / omitted experiments | ΔR +0.40 / 0 | one procedure produces all three experiments; a failure anywhere gives worst values everywhere |
| empty expert array for budget credit | ΔR +0.01 | no budget metric; `sc_empty_experts` gets expert 0, ΔR −1.13 |
| shrunken template-pool metadata | ΔR +0.18 | the evaluator supplies the template pool; no metadata metric |
| rewrite ground truth in `data/downloads` | 100 / 100 | labels hash-checked (scenario A), oracle recomputed (scenario B) |
| oracle brute force over 80 templates using rebuilt answers | 41.61, ΔR +0.60 | needs hidden labels; `ub_val_oracle_lookup` now fails |
| bad method left unregistered on disk | dropped silently | every directory under `outputs/` is scored |
| forged / stale score file | tie / undefined | score file deleted first and always rewritten |
| constant / NaN predictions | 4.3 / 0 | 0.0004 / failed run (worst values) |

## Shortcuts left (for the grader)

- **Label-derived choices shipped inside the method directory** (`sc_dev_oracle_shipped`). The method stores per-class top-4 templates chosen with the visible dev labels, plus canonical class/template mean embeddings to undo the shuffle, all in its own folder.
  - It scores 19.74 / 19.14 / 84.5, **ΔR +4.89**.
  - Why it stays open:
    - The template strings, class names and CLIP weights are public, so the shuffle can always be undone.
    - The train slice must carry labels to score methods.
    - Only reading the code or data can show that a shipped table came from labels.
  - It is explicitly prohibited in `scoring/CONSTRAINTS.md` and `DIRECTION.md`.
- **Shipped statistics from other unlabeled images**, for example an entropy table precomputed on the visible pool and reused in low-data runs. It works the same way; its payoff is bounded by what the cross-run cache earned (low-data 18.01, ΔR +1.13). It was not run as a separate probe. Prohibited in `CONSTRAINTS.md`.
- **Adapting on the scored images inside `predict`** (transductive / online). Not probed. The paper reports no gain from transductive selection (34.4 vs 34.6 inductive). Out of bounds in `DIRECTION.md`.
- **In-process tampering with the worker or evaluator** (ctypes, editing `eval/`). These are environment attacks, out of scope.

The expert diagnostic is self-reported by `experts()`, but every report is label-free at test time, so it can only rise with better label-free identification. This is not a loophole.

## Reference score before → after

| | core mIoU | low-data mIoU | expert % |
|---|---:|---:|---:|
| `REDTEAM.md` synthetic gym (different population) | 37.84 | 38.52 | 33.3 |
| rebuilt gym, before this pass (hidden test) | 18.580 | 17.348 | 60.5 |
| **final (hidden test, `scoring/scores_test.json`)** | **18.580** | **17.348** | **60.5** |
| final train slice (`scores_train.json`) | 17.057 | 16.505 | 57.8 |
| paper, NACLIP ADE20K (Tab. 1 / Tab. 2) | 18.4 (baseline 17.4) | — | 57 |

The baseline is 17.47 on test and 16.35 on train, so FLOSS gains +1.11 on test (+0.71 on train), against the paper's +1.0. `scoring/scores*.json` were regenerated by the final job from the reference alone (`outputs/floss` plus the built-in baseline).

## Headroom check

- **No scored metric is at its bound.** Core is 18.58: the oracle with true experts reaches 20.74, and label-free v_avg_prob reaches 18.91. Low-data is 17.35: full-pool experts give 18.01 and the oracle gives 19.93. Expert precision is 60.5% out of 100: v_avg_prob gets 63.7, the dev-label oracle 84.5.
- **Variants above the reference:** v_avg_prob (+0.305) and v_min_support (+0.179), both beyond paired noise on core.
- **Low-data limitation.** No honest variant tried beats the reference on low-data by more than its noise:
  - v_min_support: +0.02 ± 0.02.
  - w_average_all: +0.11 ± 0.07, but −0.38 on core.
  - The new min-count variants: −0.16 and −0.30.

  The 0.66-point gap to full-pool experts shows room for better few-image estimation; no experiment was added. FLOSS is below the baseline at 4 images (−0.21), and above it at 16 (+0.23) and 64 (+0.60). Its 12-run mean is +0.21 against a draw SE of 0.04.
- **Brute force** (choosing templates by labelled IoU) requires the hidden labels, which are unreadable during runs. With dev labels it is the shortcut left above.
- **Baselines:** template_average is −1.07 core (ΔR −2.06), random experts −3.04. No simple rule outranks the reference.

## Weights

Unchanged:
- **0.6 core**: the paper's headline, Tab. 1.
- **0.25 low-data**: a secondary claim, Fig. 3 / Tab. 4.
- **0.15 expert quality**: a diagnostic, Tab. 2.

The diagnostic moves most easily without a segmentation gain, so it keeps the smallest weight.

## Other changes

- `method/floss.py` gains an optional `min_count` knob, used only by the variants. The default reproduces the paper.
- `redteam/hardening/` now installs and scores cases one at a time and resumes after interruption. It also rescores new probes with the pre-fix evaluator, a git snapshot in `tmp/old_eval_test/`, rebuilt only if present.
- Docs updated to match the enforced rules: `briefing/evaluation.md` (run isolation, file access, retry), `scoring/CONSTRAINTS.md`, `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, `scripts/evaluate_*.sh`.

## Not fixed / limitations

- The label-derived shipped-table shortcut above.
- Coverage of the paper:
  - The gym covers one of the paper's 15 model × dataset cells: NACLIP + ADE20K.
  - Cityscapes is gated, and the domain-shift (Tab. 3/4) and cross-dataset (Tab. 5) claims are not covered.
  - Adaptation uses 2000 of 20,210 training images, and scoring is at half the 336-short-side resolution.
- `validate.py` passes. Its two warnings flag the seeded protocol files as "detached from the method"; they only pin the evaluator's input draws.
- Rank magnitudes depend on the future cohort's spread. The directions above do not.
