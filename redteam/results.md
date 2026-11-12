| Run | Core mIoU | Low mIoU | Expert % | Rank delta floss* | Rank delta floss_low_data* | Test exit |
|---|---:|---:|---:|---:|---:|---:|
| brute_force_shortcut | 41.6078 | 41.6078 | 91.6667 | 0.6035 | 0.3394 | 0 |
| constant_shortcut | 4.3355 | 4.3355 | 33.3333 | -1.8145 | -0.8458 | 0 |
| diagnostic_only_positive_shortcut | missing | missing | 100.0000 | 0.4030 | 0.0000 | 0 |
| diagnostic_only_shortcut | missing | missing | 33.3333 | 0.0000 | 0.0000 | 0 |
| empty_budget_shortcut | 37.8363 | 38.5212 | missing | 0.0135 | 0.0062 | 0 |
| forged_scores_shortcut | 37.8363 | 38.5212 | 33.3333 | 0.0000 | 0.0000 | 0 |
| layout_shortcut | 56.6144 | 56.6144 | 33.3333 | 0.7247 | 0.3022 | 0 |
| nan_shortcut | 0.0000 | 0.0000 | 33.3333 | -2.3155 | -1.0719 | 0 |
| omitted_disk_method_shortcut | 37.8363 | 38.5212 | 33.3333 | 0.0000 | 0.0000 | 0 |
| rebuild_answers_shortcut | 100.0000 | 100.0000 | 100.0000 | 3.2743 | 1.5748 | 0 |
| reference | 37.8363 | 38.5212 | 33.3333 | 0.0000 | 0.0000 | 0 |
| rewrite_ground_truth_shortcut | 100.0000 | 100.0000 | 100.0000 | 3.2743 | 1.5748 | 0 |
| seed_0 | 37.8363 | 38.7687 | 33.3333 | 0.0000 | -0.0953 | 0 |
| seed_1 | 37.8363 | 37.7299 | 33.3333 | 0.0000 | -0.1177 | 0 |
| seed_2 | 37.8363 | 30.9609 | 33.3333 | 0.0000 | -0.3780 | 0 |
| shrink_pool_shortcut | 37.8363 | 38.5212 | 33.3333 | 0.1761 | 0.1057 | 0 |
| stale_failure_shortcut | 100.0000 | 100.0000 | 100.0000 | missing | missing | 1 |
| stronger_consensus | 41.9500 | 41.9500 | 100.0000 | 0.6634 | 0.3941 | 0 |
| stronger_consensus_fusion | 41.6899 | 41.6899 | 100.0000 | 0.6559 | 0.3907 | 0 |
| unlinked_experts_shortcut | 37.8363 | 38.5212 | 100.0000 | 0.4030 | 0.3023 | 0 |
| weaker_no_experts | 33.2851 | 33.2851 | missing | -0.4352 | -0.2232 | 0 |
| weaker_one_image | 40.6422 | 40.6422 | 25.0000 | 0.1372 | -0.0922 | 0 |
| weaker_random_experts | 22.3883 | 22.3883 | 0.0000 | -0.9896 | -0.6843 | 0 |

*Audit-cohort standard deviations; the future cohort changes the numerical size. Reference delta is zero. Literal method names are scored separately, as actually written under results. The extra combined profile in summary.json uses core floss, low-data floss_low_data, and expert-quality floss for descriptive pipeline comparisons only. Failed tests have no accepted rank; inspect the stale-file field in summary.json separately.

```json
{
  "clip_dinoiser_cityscapes_core.mIoU": {
    "values": [
      37.83629851870367,
      37.83629851870367,
      37.83629851870367
    ],
    "min": 37.83629851870367,
    "max": 37.83629851870367,
    "range": 0.0,
    "sample_sd": 0.0
  },
  "clip_dinoiser_cityscapes_low_data.mIoU": {
    "values": [
      38.7686878566728,
      37.72985500293483,
      30.960930632623924
    ],
    "min": 30.960930632623924,
    "max": 38.7686878566728,
    "range": 7.807757224048874,
    "sample_sd": 4.2398620987535605
  },
  "expert_quality_clip_dinoiser_cityscapes_core.expert_precision_pct": {
    "values": [
      33.33333333333333,
      33.33333333333333,
      33.33333333333333
    ],
    "min": 33.33333333333333,
    "max": 33.33333333333333,
    "range": 0.0,
    "sample_sd": 0.0
  },
  "low_data_expert_precision_pct": {
    "values": [
      33.33333333333333,
      33.33333333333333,
      25.0
    ],
    "range": 8.333333333333329,
    "sample_sd": 4.811252243246878
  },
  "combined_rank": {
    "values": [
      0.005449292727427842,
      -0.01693164136729386,
      -0.2269089432321801
    ],
    "range": 0.23235823595960794
  },
  "literal_rank_floss": {
    "values": [
      0.0,
      0.0,
      0.0
    ],
    "range": 0.0,
    "sample_sd": 0.0
  },
  "literal_rank_floss_low_data": {
    "values": [
      -0.09531041144845548,
      -0.11769134554317717,
      -0.37804849949600505
    ],
    "range": 0.2827380880475496,
    "sample_sd": 0.15717695985637373
  }
}
```
