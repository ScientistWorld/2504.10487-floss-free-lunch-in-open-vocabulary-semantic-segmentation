"""Lightweight evidence checks; does not run models, generators, or scorers."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
start = (AUDIT / 'snapshot/start_commit.txt').read_text().strip()
checks = {}


def check(name, condition):
    checks[name] = bool(condition)
    if not condition:
        raise AssertionError(name)


module = ast.parse((AUDIT / 'harness.py').read_text())
cases = next(ast.literal_eval(node.value) for node in module.body
             if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'CASES' for t in node.targets))
check('23_distinct_cases', len(cases) == len(set(cases)) == 23)
summary = json.loads((AUDIT / 'summary.json').read_text())
check('summary_covers_all_cases', set(summary['runs']) == set(cases))
successes = 0
for case in cases:
    folder = AUDIT / case
    required = ['run.sh', 'run_status.json', 'scores_train.json', 'scores_test.json',
                'eval_train_manifest.json', 'eval_test_manifest.json', 'train.log', 'test.log']
    check(f'{case}_evidence_present', all((folder / name).is_file() for name in required))
    check(f'{case}_entrypoint', case in (folder / 'run.sh').read_text())
    status = json.loads((folder / 'run_status.json').read_text())
    check(f'{case}_visible_success', status['train_exit_code'] == 0)
    check(f'{case}_no_argument_hidden_test', status['hidden_command'] == 'bash scripts/evaluate_test.sh')
    expected = 1 if case == 'stale_failure_shortcut' else 0
    check(f'{case}_expected_hidden_exit', status['test_exit_code'] == expected)
    successes += status['test_exit_code'] == 0
check('22_successes_one_deliberate_failure', successes == 22)

for name in ['scores_train.json', 'scores_test.json', 'eval_train_manifest.json', 'eval_test_manifest.json']:
    check(f'{name}_restored', (ROOT / 'scoring' / name).read_bytes() == (AUDIT / 'snapshot' / name).read_bytes())
data_hashes = {}
for filename, cached in [('floss_synthetic_inputs.npz', 'inputs.npz'), ('floss_synthetic_eval.npz', 'labels.npz')]:
    actual = hashlib.sha256((ROOT / 'data/downloads' / filename).read_bytes()).hexdigest()
    original = hashlib.sha256((AUDIT / 'cache' / cached).read_bytes()).hexdigest()
    check(f'{filename}_original_bytes_restored', actual == original)
    data_hashes[filename] = actual

core = 'clip_dinoiser_cityscapes_core'
low = 'clip_dinoiser_cityscapes_low_data'
expert = 'expert_quality_clip_dinoiser_cityscapes_core'
base = json.loads((AUDIT / 'snapshot/scores_test.json').read_text())['experiments']
rerun = json.loads((AUDIT / 'reference/scores_test.json').read_text())['experiments']
for section, method in [(core, 'floss'), (low, 'floss_low_data'), (expert, 'floss'), (expert, 'floss_low_data')]:
    check(f'reference_exact_{section}_{method}', base[section]['results'][method] == rerun[section]['results'][method])

runs = summary['runs']
check('core_stronger_gain_exceeds_seed_range', runs['stronger_consensus']['core_mIoU'] - runs['reference']['core_mIoU'] > summary['seed_spread'][core + '.mIoU']['range'])
check('core_weaker_loss_exceeds_seed_range', runs['reference']['core_mIoU'] - runs['weaker_random_experts']['core_mIoU'] > summary['seed_spread'][core + '.mIoU']['range'])
check('low_weaker_loss_exceeds_seed_range', runs['reference']['low_data_mIoU'] - runs['weaker_random_experts']['low_data_mIoU'] > summary['seed_spread'][low + '.mIoU']['range'])
check('low_stronger_miou_gain_is_below_seed_range', runs['stronger_consensus']['low_data_mIoU'] - runs['reference']['low_data_mIoU'] < summary['seed_spread'][low + '.mIoU']['range'])
check('failing_task_can_rank_positive', runs['diagnostic_only_positive_shortcut']['core_mIoU'] is None and runs['diagnostic_only_positive_shortcut']['literal_floss_rank_delta_core_plus_diagnostic'] > 0)
check('failure_not_counted_as_accepted_rank', runs['stale_failure_shortcut']['literal_floss_rank_delta_core_plus_diagnostic'] is None)
check('answer_reconstruction_perfect', runs['rebuild_answers_shortcut']['core_mIoU'] == runs['rebuild_answers_shortcut']['low_data_mIoU'] == runs['rebuild_answers_shortcut']['expert_precision_pct'] == 100)
status = json.loads((AUDIT / 'rebuild_answers_shortcut/run_status.json').read_text())
check('generator_inputs_match_bit_for_bit', status['generated_input_matches_original_sha256'] and status['generated_visible_arrays_match'])
check('successful_forgery_recomputed', runs['forged_scores_shortcut']['core_mIoU'] == runs['reference']['core_mIoU'])

for path in [AUDIT / 'harness.py', AUDIT / 'summarize.py', AUDIT / 'validate_audit.py', *sorted((AUDIT / 'support').glob('*.py'))]:
    ast.parse(path.read_text())
check('audit_python_syntax_valid', True)
for path in [*sorted(AUDIT.glob('*/run.sh')), *sorted(AUDIT.glob('run_*.sh'))]:
    subprocess.run(['bash', '-n', str(path)], check=True)
check('audit_shell_syntax_valid', True)

# Initial unstaged README/manifest/trajectory changes are recorded in the
# snapshot. The action signal is the one required operational exception.
diff = subprocess.check_output([
    'git', 'diff', '--name-only', start, '--', '.', ':!redteam', ':!REDTEAM.md',
    ':!action.yaml', ':!README.md', ':!manifest.json', ':!trajectory'
], cwd=ROOT, text=True).splitlines()
check('no_audit_owned_changes_outside_allowed_paths', not diff)

report = (ROOT / 'REDTEAM.md').read_text()
links = re.findall(r'\]\(([^)]+)\)', report)
check('report_links_exist', all((ROOT / link).exists() or link == 'redteam/validation.json' for link in links))
out = {'passed': True, 'case_count': len(cases), 'successful_hidden_tests': successes,
       'intentional_failed_hidden_tests': 1, 'start_commit': start,
       'restored_data_sha256': data_hashes, 'checks': checks,
       'scope': 'Lightweight artifact, syntax, score and restoration checks; computation ran in managed jobs.'}
(AUDIT / 'validation.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k: out[k] for k in ['passed', 'case_count', 'successful_hidden_tests', 'intentional_failed_hidden_tests']}))
