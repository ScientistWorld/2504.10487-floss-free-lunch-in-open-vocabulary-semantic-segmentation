"""Summarize real hidden-test files using the stated rank's cohort scaling.

The later ranking cohort is unknowable at audit time. These are reproducible
illustrative ranks with population standard deviations over successful audit
runs plus the saved reference. Metrics are also reported without normalization.
"""
from pathlib import Path
import json
import math
import statistics

ROOT = Path('/home/user/redteam')
CORE = 'clip_dinoiser_cityscapes_core'
LOW = 'clip_dinoiser_cityscapes_low_data'
EXPERT = 'expert_quality_clip_dinoiser_cityscapes_core'
SECTIONS = [CORE, LOW, EXPERT]
config = json.loads((ROOT / 'snapshot/reference.json').read_text())['experiments']
reference = json.loads((ROOT / 'snapshot/scores_test.json').read_text())['experiments']


def load_profile(experiments, primary=True):
    profile = {}
    for experiment in SECTIONS:
        method = 'floss_low_data' if experiment == LOW else 'floss'
        profile[experiment] = experiments.get(experiment, {}).get('results', {}).get(method, {})
    return profile


base = load_profile(reference)
runs = {}
for path in sorted(ROOT.glob('*/run_status.json')):
    case = path.parent.name
    status = json.loads(path.read_text())
    score_file = path.parent / 'scores_test.json'
    scores = json.loads(score_file.read_text())['experiments'] if score_file.exists() else {}
    profile = load_profile(scores)
    runs[case] = {'status': status, 'profile': profile, 'raw_experiments': scores}

scales = {}
for experiment in SECTIONS:
    scales[experiment] = {}
    for metric in config[experiment]['metrics']:
        values = [base[experiment][metric]]
        for run in runs.values():
            value = run['profile'][experiment].get(metric)
            if run['status']['test_exit_code'] == 0 and isinstance(value, (int, float)) and math.isfinite(value):
                values.append(value)
        scales[experiment][metric] = statistics.pstdev(values)


def named_profile(experiments, method):
    return {experiment: experiments.get(experiment, {}).get('results', {}).get(method, {}) for experiment in SECTIONS}


def rank(profile, only=None, anchor=None):
    anchor = base if anchor is None else anchor
    value = 0.
    for experiment in SECTIONS:
        if only and experiment not in only:
            continue
        for metric, details in config[experiment]['metrics'].items():
            observed = profile.get(experiment, {}).get(metric)
            if not isinstance(observed, (int, float)) or not math.isfinite(observed):
                continue
            reference_value = anchor.get(experiment, {}).get(metric)
            if reference_value is None:
                continue
            spread = scales[experiment][metric]
            if spread == 0:
                continue
            sign = 1 if details['higher_is_better'] else -1
            value += config[experiment]['weight'] * abs(details['coefficient']) * sign * (observed - reference_value) / spread
    return value

summary = {'rank_definition': 'weighted abs(coefficient) * direction * (value - saved reference) / population_std; missing metrics skipped; audit-cohort normalization, not unknown future cohort',
           'reference_profile': base, 'cohort_standard_deviations': scales, 'runs': {}}
for case, run in runs.items():
    profile = run['profile']
    summary['runs'][case] = {
        'test_exit_code': run['status']['test_exit_code'],
        'core_mIoU': profile[CORE].get('mIoU'),
        'low_data_mIoU': profile[LOW].get('mIoU'),
        'expert_precision_pct': profile[EXPERT].get('expert_precision_pct'),
        'low_data_expert_precision_pct': run['raw_experiments'].get(EXPERT, {}).get('results', {}).get('floss_low_data', {}).get('expert_precision_pct'),
        'core_experts': profile[CORE].get('num_experts_per_class'),
        'core_templates': profile[CORE].get('num_templates'),
        'illustrative_combined_rank_delta': rank(profile) if run['status']['test_exit_code'] == 0 else None,
        'literal_floss_rank_delta_core_plus_diagnostic': rank(profile, [CORE, EXPERT]) if run['status']['test_exit_code'] == 0 else None,
        'literal_floss_low_data_rank_delta_low_plus_diagnostic': rank(named_profile(run['raw_experiments'], 'floss_low_data'), anchor=named_profile(reference, 'floss_low_data')) if run['status']['test_exit_code'] == 0 else None,
        'stale_file_rank_delta_not_accepted': rank(profile) if run['status']['test_exit_code'] != 0 else None,
    }
seed_summary = {}
for experiment, metric in [(CORE, 'mIoU'), (LOW, 'mIoU'), (EXPERT, 'expert_precision_pct')]:
    values = [runs[f'seed_{seed}']['profile'][experiment][metric] for seed in range(3)]
    seed_summary[f'{experiment}.{metric}'] = {'values': values, 'min': min(values), 'max': max(values), 'range': max(values) - min(values), 'sample_sd': statistics.stdev(values)}
low_expert_values = [runs[f'seed_{seed}']['raw_experiments'][EXPERT]['results']['floss_low_data']['expert_precision_pct'] for seed in range(3)]
seed_summary['low_data_expert_precision_pct'] = {'values': low_expert_values, 'range': max(low_expert_values) - min(low_expert_values), 'sample_sd': statistics.stdev(low_expert_values)}
seed_summary['combined_rank'] = {'values': [rank(runs[f'seed_{seed}']['profile']) for seed in range(3)]}
seed_summary['combined_rank']['range'] = max(seed_summary['combined_rank']['values']) - min(seed_summary['combined_rank']['values'])
for method, key in [('floss', 'literal_floss_rank_delta_core_plus_diagnostic'), ('floss_low_data', 'literal_floss_low_data_rank_delta_low_plus_diagnostic')]:
    values = [summary['runs'][f'seed_{seed}'][key] for seed in range(3)]
    seed_summary[f'literal_rank_{method}'] = {'values': values, 'range': max(values) - min(values), 'sample_sd': statistics.stdev(values)}
summary['seed_spread'] = seed_summary
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['| Run | Core mIoU | Low mIoU | Expert % | Rank delta floss* | Rank delta floss_low_data* | Test exit |', '|---|---:|---:|---:|---:|---:|---:|']
for case, row in summary['runs'].items():
    def fmt(x):
        return 'missing' if x is None else f'{x:.4f}'
    lines.append(f"| {case} | {fmt(row['core_mIoU'])} | {fmt(row['low_data_mIoU'])} | {fmt(row['expert_precision_pct'])} | {fmt(row['literal_floss_rank_delta_core_plus_diagnostic'])} | {fmt(row['literal_floss_low_data_rank_delta_low_plus_diagnostic'])} | {row['test_exit_code']} |")
lines += ['', '*Audit-cohort standard deviations; the future cohort changes the numerical size. Reference delta is zero. Literal method names are scored separately, as actually written under results. The extra combined profile in summary.json uses core floss, low-data floss_low_data, and expert-quality floss for descriptive pipeline comparisons only. Failed tests have no accepted rank; inspect the stale-file field in summary.json separately.', '', '```json', json.dumps(seed_summary, indent=2), '```']
(ROOT / 'results.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
