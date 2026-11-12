| Run | Core mIoU | Low mIoU | Expert % | Rank delta* | Test exit |
|---|---:|---:|---:|---:|---:|
| brute_force_shortcut | 41.6078 | 41.6078 | 91.6667 | 0.7043 | 0 |
| constant_shortcut | 4.3355 | 4.3355 | 33.3333 | -2.6374 | 0 |
| diagnostic_only_positive_shortcut | missing | missing | 100.0000 | 0.4215 | 0 |
| diagnostic_only_shortcut | missing | missing | 33.3333 | 0.0000 | 0 |
| empty_budget_shortcut | 37.8363 | 38.5212 | missing | 0.0192 | 0 |
| forged_scores_shortcut | 37.8363 | 38.5212 | 33.3333 | 0.0000 | 0 |
| layout_shortcut | 56.6144 | 56.6144 | 33.3333 | 1.0197 | 0 |
| nan_shortcut | 0.0000 | 0.0000 | 33.3333 | -3.3588 | 0 |
| omitted_disk_method_shortcut | 37.8363 | 38.5212 | 33.3333 | 0.0000 | 0 |
| rebuild_answers_shortcut | 100.0000 | 100.0000 | 100.0000 | 4.5279 | 0 |
| reference | 37.8363 | 38.5212 | 33.3333 | 0.0000 | 0 |
| rewrite_ground_truth_shortcut | 100.0000 | 100.0000 | 100.0000 | 4.5279 | 0 |
| seed_0 | 37.8363 | 38.7687 | 33.3333 | 0.0054 | 0 |
| seed_1 | 37.8363 | 37.7299 | 33.3333 | -0.0168 | 0 |
| seed_2 | 37.8363 | 30.9609 | 33.3333 | -0.2249 | 0 |
| shrink_pool_shortcut | 37.8363 | 38.5212 | 33.3333 | 0.2294 | 0 |
| stale_failure_shortcut | 100.0000 | 100.0000 | 100.0000 | 4.7765 | 1 |
| stronger_consensus | 41.9500 | 41.9500 | 100.0000 | 0.7707 | 0 |
| stronger_consensus_fusion | 41.6899 | 41.6899 | 100.0000 | 0.7598 | 0 |
| unlinked_experts_shortcut | 37.8363 | 38.5212 | 100.0000 | 0.4215 | 0 |
| weaker_no_experts | 33.2851 | 33.2851 | missing | -0.6474 | 0 |
| weaker_one_image | 40.6422 | 40.6422 | 25.0000 | 0.1918 | 0 |

*Audit-cohort standard deviations; the future cohort changes the numerical size. Reference delta is zero.

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
  "combined_rank": {
    "values": [
      0.005410939886300385,
      -0.016773495166447883,
      -0.22487983062545094
    ],
    "range": 0.23029077051175134
  }
}
```
