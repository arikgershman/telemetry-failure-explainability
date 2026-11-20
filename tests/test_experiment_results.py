"""CPU return-trip verification with synthetic completions, never weights."""

import builtins
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from telemetry_reasoning import experiment_results as er
from telemetry_reasoning import experiments as exp
from telemetry_reasoning.io import write_metadata_csv
from telemetry_reasoning.models.llm import MEMORY_POLICY, Generation, LLMConfig, StubBackend, run_inference
from telemetry_reasoning.models.prompts import TEMPLATE_FILENAMES, DEFAULT_TEMPLATE_DIR
from telemetry_reasoning.models.rule_based import load_rule_config, predict
from telemetry_reasoning.parser.divergence_metrics import enrich_paired_bt, load_divergence_config
from telemetry_reasoning.schemas import PairedBehaviorTree
from scripts.run_llm import publish_run

ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path, monkeypatch, *, backend='stub', refusals=False,
             response_text=None, invalid_output_metadata=False):
    bundle, results = tmp_path / 'bundle', tmp_path / 'results'
    (bundle / 'configs').mkdir(parents=True)
    (bundle / 'prompts').mkdir()
    results.mkdir()
    for name in TEMPLATE_FILENAMES.values():
        shutil.copyfile(DEFAULT_TEMPLATE_DIR / name, bundle / 'prompts' / name)
    for name in ('eval.yaml', 'rules.yaml'):
        shutil.copyfile(ROOT / 'configs' / name, bundle / 'configs' / name)
    config = LLMConfig(context_window_tokens=128, max_new_tokens=16,
                       prompt_dir=bundle / 'prompts', device='cuda')
    exp._write(bundle / 'configs/llm.yaml', {'llm': config.model_dump(mode='json')})

    class FixtureBackend(StubBackend):
        name = backend
        token_count_method = 'tokenizer' if backend == 'hf' else 'fixture_counter'
        def count_tokens(self, prompt):
            return 128 if refusals == 'all' or refusals and prompt.variant.value == 'main' else 5
        def generate(self, prompt):
            text = json.dumps({'predicted_failure_mode': None, 'predicted_subtask': None,
                               'explanation': 'Synthetic fixture, no diagnosis',
                               'evidence': [], 'status': 'abstain'})
            if response_text is not None and prompt.variant.value == 'baseline_2':
                text = response_text(prompt) if callable(response_text) else response_text
            output_tokens = True if invalid_output_metadata and prompt.variant.value == 'baseline_2' else 5
            return Generation(text, self.count_tokens(prompt), output_tokens, self.token_count_method)
        def provenance(self):
            return {'weights_loaded': backend == 'hf', 'device': 'cuda',
                    'tokenizer_loaded': True, 'implementation': 'transformers_batched_greedy_v2',
                    'token_count_method': self.token_count_method, 'dtype': config.dtype,
                    'batch_size': config.batch_size, 'quantization': config.quantization,
                    'native_context_window_tokens': 131072,
                    'resources': {'policy': MEMORY_POLICY, 'host_admissible': True},
                    'runtime': {'torch': 'fixture-only', 'transformers': 'fixture-only'},
                    'model': {'id': config.model_id, 'requested_revision': config.revision,
                              'resolved_revision': config.revision},
                    'tokenizer': {'id': config.model_id, 'requested_revision': config.revision,
                                  'resolved_revision': config.revision}}

    token_budget = []
    original = json.loads((ROOT / 'data/sample/pair_000_bt.json').read_text())
    for task in exp.TASKS:
        value = copy.deepcopy(original)
        if task == 'transport_object':
            names = ('maintain_grasp', 'move_along_path', 'reach_target')
            value['metadata'].update(task=task, failed_subtask='move_along_path')
            value['subtasks'] = value['subtasks'][:3]
            for node, name in zip(value['subtasks'], names):
                node['subtask'] = name
                for field in ('success_samples', 'failure_samples'):
                    for row in node[field]:
                        row['subtask'] = name
            value['subtasks'][-1]['failure_segment_end'] = 'trace_end'
        tree = enrich_paired_bt(PairedBehaviorTree.model_validate(value),
                                load_divergence_config(ROOT / 'configs/divergence.yaml'))
        task_dir = bundle / 'tasks' / task
        (task_dir / 'bt').mkdir(parents=True)
        (task_dir / 'bt/pair_000_bt.json').write_text(tree.model_dump_json())
        write_metadata_csv(task_dir / 'metadata.csv', [tree.metadata])
        exp._write(task_dir / 'task_manifest.json', {'pairs': {'000': {'split': 'dev', 'initial_condition_id': 0}}})
        exp.write_predictions(task_dir / 'baseline_1.jsonl', [predict(tree.to_failure_only(), load_rule_config())])
        exp._write(task_dir / 'baseline_1.manifest.json', {'model_variant': 'baseline_1', 'backend': 'rules'})
        run = run_inference([tree], exp.VARIANTS, config, backend=FixtureBackend(config))
        # A fixture backend subclasses StubBackend, so production batch behavior
        # is represented by explicitly retaining its actual single-request size.
        if backend == 'hf':
            run.manifest['decoding']['batch_size'] = config.batch_size
        publish_run(results / task / 'inference', run.predictions, exp.VARIANTS,
                    run.manifest, completions=run.completions)
        (results / task / 'inference.log').write_text('Synthetic fixture inference only\n')
        token_budget.extend({'task': task, **row,
                             'fits': row['input_tokens'] + config.max_new_tokens <= config.context_window_tokens}
                            for row in run.manifest['requests'])
    source_files = {'fixture-source': {'sha256': 'a' * 64, 'size_bytes': 1}}
    monkeypatch.setattr(exp, '_sources', lambda: copy.deepcopy(source_files))
    manifest = {'bundle_version': 1, 'status': 'ready_for_gpu', 'split': 'dev',
                'source_files': source_files, 'token_budget': token_budget,
                'pair_counts': {task: 1 for task in exp.TASKS},
                'dataset_release_status': 'parser_ready_certification_failed',
                'files': exp._inventory(bundle)}
    exp._write(bundle / 'bundle_manifest.json', manifest)
    receipt = {'run_version': 1, 'backend': backend, 'status': 'synthetic' if backend == 'stub' else 'passed',
               'bundle_manifest_sha256': exp._fingerprint(bundle / 'bundle_manifest.json')['sha256'],
               'pair_counts': manifest['pair_counts'], 'split': 'dev',
               'stages': [{'task': task, 'stage': 'inference', 'exit_code': 1 if refusals else 0}
                          for task in exp.TASKS], 'files': exp._inventory(results)}
    exp._write(results / 'experiment_receipt.json', receipt)
    return bundle, results


def _rebind(results):
    path = results / 'experiment_receipt.json'
    receipt = exp._json(path)
    receipt['files'] = exp._inventory(results, ('experiment_receipt.json',))
    exp._write(path, receipt)


def test_cpu_return_trip_runs_public_four_variant_metrics_and_plots(tmp_path, monkeypatch):
    bundle, results = _fixture(tmp_path, monkeypatch)
    before = exp._inventory(results)
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformers', 'huggingface_hub', 'trl', 'peft', 'bitsandbytes'}:
            pytest.fail('CPU scoring imported optional model runtime: ' + name)
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    report = er.evaluate_results(bundle=bundle, results=results, output_dir=tmp_path / 'evaluation')
    assert report['status'] == 'synthetic'
    assert not report['real_model_acceptance'] and not report['real_generation_acceptance']
    assert report['gpu_results_unchanged'] and before == exp._inventory(results)
    for task in exp.TASKS:
        root = tmp_path / 'evaluation' / task / 'evaluation'
        metrics = exp._json(root / 'metrics.json')
        assert set(metrics['variants']) == {'baseline_1', *exp.VARIANTS}
        assert len(list((root / 'plots').glob('*.png'))) >= 5
    receipt = exp._json(tmp_path / 'evaluation/evaluation_receipt.json')
    assert exp._inventory(tmp_path / 'evaluation', ('evaluation_receipt.json',)) == receipt['files']


def test_context_refusals_are_validated_but_never_full_acceptance(tmp_path, monkeypatch):
    bundle, results = _fixture(tmp_path, monkeypatch, backend='hf', refusals=True)
    verified = er.verify_gpu_results(bundle, results)
    assert verified['real_generation_acceptance'] and not verified['real_model_acceptance']
    assert all(row['context_refused_requests'] == 1 for row in verified['request_coverage'].values())


def test_all_context_refusals_never_establish_real_generation(tmp_path, monkeypatch):
    bundle, results = _fixture(tmp_path, monkeypatch, backend='hf', refusals='all')
    verified = er.verify_gpu_results(bundle, results)
    assert not verified['real_generation_acceptance'] and not verified['real_model_acceptance']


@pytest.mark.parametrize('field', ['resources', 'dtype', 'tokenizer_loaded', 'native_context_window_tokens', 'runtime'])
def test_hf_runtime_provenance_is_verified_independently(tmp_path, monkeypatch, field):
    bundle, results = _fixture(tmp_path, monkeypatch, backend='hf')
    path = results / 'pick_up_object/inference/run_manifest.json'
    record = exp._json(path)
    record['backend_provenance'][field] = None
    exp._write(path, record)
    _rebind(results)
    with pytest.raises(ValueError, match='recorded CUDA'):
        er.verify_gpu_results(bundle, results)


@pytest.mark.parametrize('mutation', ['changed', 'added', 'deleted', 'symlink'])
def test_returned_inventory_tampering_is_refused_before_output(tmp_path, monkeypatch, mutation):
    bundle, results = _fixture(tmp_path, monkeypatch)
    path = results / 'pick_up_object/inference.log'
    if mutation == 'changed':
        path.write_text('changed')
    elif mutation == 'added':
        (results / 'extra').write_text('added')
    elif mutation == 'deleted':
        path.unlink()
    else:
        (results / 'link').symlink_to(path)
    with pytest.raises(ValueError, match='inventory|Symlink'):
        er.evaluate_results(bundle=bundle, results=results, output_dir=tmp_path / 'evaluation')
    assert not (tmp_path / 'evaluation').exists()


@pytest.mark.parametrize('mutation', ['bundle_hash', 'backend', 'source'])
def test_binding_failures_prevent_scoring(tmp_path, monkeypatch, mutation):
    bundle, results = _fixture(tmp_path, monkeypatch)
    if mutation == 'source':
        monkeypatch.setattr(exp, '_sources', lambda: {})
    else:
        path = results / 'experiment_receipt.json'
        receipt = exp._json(path)
        receipt['bundle_manifest_sha256' if mutation == 'bundle_hash' else 'backend'] = 'invalid'
        exp._write(path, receipt)
    with pytest.raises(ValueError, match='bind|implementation differs'):
        er.verify_gpu_results(bundle, results)


@pytest.mark.parametrize('score_response_errors', [False, True])
@pytest.mark.parametrize('mutation', ['duplicate', 'unknown', 'missing', 'bad_completion', 'token_count', 'decoding'])
def test_semantic_failures_are_fully_accounted_and_never_scored(
    tmp_path, monkeypatch, mutation, score_response_errors,
):
    bundle, results = _fixture(tmp_path, monkeypatch)
    task = 'pick_up_object'
    directory = results / task / 'inference'
    if mutation in ('duplicate', 'unknown', 'missing'):
        path = directory / 'baseline_2.jsonl'
        row = json.loads(path.read_text())
        if mutation == 'duplicate':
            path.write_text(path.read_text() * 2)
        elif mutation == 'unknown':
            row['pair_id'] = 'unknown'
            path.write_text(json.dumps(row) + '\n')
        else:
            path.write_text('')
    elif mutation == 'bad_completion':
        path = directory / 'completions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['text'] = 'invalid JSON'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    else:
        path = directory / 'run_manifest.json'
        row = exp._json(path)
        if mutation == 'token_count':
            row['requests'][0]['output_tokens'] += 1
        else:
            row['decoding']['do_sample'] = True
        exp._write(path, row)
    _rebind(results)
    def fake_score(*args, **kwargs):
        return type('Completed', (), {'returncode': 0})()
    monkeypatch.setattr(er.subprocess, 'run', fake_score)
    report = er.evaluate_results(bundle=bundle, results=results, output_dir=tmp_path / 'evaluation',
                                 score_response_errors=score_response_errors)
    assert report['status'] == 'failed' and not report['real_model_acceptance']
    assert task not in report['request_coverage']
    accounting = report['request_accounting'][task]['predictions']
    if mutation == 'duplicate':
        assert accounting['duplicates'] == [{'key': ['000', 'baseline_2'], 'occurrences': 2}]
    elif mutation == 'unknown':
        assert accounting['unexpected'] == [['unknown', 'baseline_2']]
        assert accounting['missing'] == [['000', 'baseline_2']]
    elif mutation == 'missing':
        assert accounting['missing'] == [['000', 'baseline_2']]
    assert all(row['task'] != task for row in report['stages'])


def test_no_overwrite_or_output_inside_inputs(tmp_path, monkeypatch):
    bundle, results = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='absent'):
        er.evaluate_results(bundle=bundle, results=results, output_dir=results)
    with pytest.raises(ValueError, match='separate'):
        er.evaluate_results(bundle=bundle, results=results, output_dir=results / 'evaluation')


def test_cpu_import_does_not_load_hf_or_model_runtime(tmp_path):
    code = '''
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'huggingface_hub', 'trl', 'peft', 'bitsandbytes'}:
        raise AssertionError('Forbidden optional runtime import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from telemetry_reasoning.experiment_results import evaluate_results, verify_gpu_results
'''
    completed = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_cli_errors_are_clean(tmp_path, capsys):
    from scripts.evaluate_gpu_results import main
    assert main(['--bundle', str(tmp_path / 'missing'), '--results', str(tmp_path / 'missing-results'),
                 '--output-dir', str(tmp_path / 'evaluation')]) == 2
    stderr = capsys.readouterr().err
    assert 'error:' in stderr and 'Traceback' not in stderr


def test_response_errors_remain_strictly_rejected_without_scoring_opt_in(tmp_path, monkeypatch):
    bundle, results = _fixture(tmp_path, monkeypatch, backend='hf', response_text='not JSON')
    with pytest.raises(ValueError, match='failed inference|response validation'):
        er.verify_gpu_results(bundle, results)


def test_opt_in_scores_reproduced_response_errors_for_every_variant_without_acceptance(tmp_path, monkeypatch):
    bundle, results = _fixture(tmp_path, monkeypatch, backend='hf', response_text='not JSON')
    before = exp._inventory(results)
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformers', 'huggingface_hub', 'trl', 'peft', 'bitsandbytes'}:
            pytest.fail('Response-error scoring imported optional model runtime: ' + name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    verified = er.verify_gpu_results(bundle, results, score_response_errors=True)
    assert verified['response_error_requests'] == 2
    assert not verified['real_generation_acceptance'] and not verified['real_model_acceptance']
    assert all(row['validated_fit_requests'] == 2 and row['response_error_requests'] == 1
               and not row['full_request_acceptance'] for row in verified['request_coverage'].values())
    report = er.evaluate_results(bundle=bundle, results=results, output_dir=tmp_path / 'evaluation',
                                 score_response_errors=True)
    assert report['status'] == 'failed' and report['metrics_generated'] is True
    assert report['score_response_errors'] and report['response_error_requests'] == 2
    assert not report['real_generation_acceptance'] and not report['real_model_acceptance']
    assert all('response-validation' in failure for failure in report['failures'])
    assert before == exp._inventory(results) and report['gpu_results_unchanged']
    for task in exp.TASKS:
        metrics = exp._json(tmp_path / 'evaluation' / task / 'evaluation/metrics.json')
        assert set(metrics['variants']) == {'baseline_1', *exp.VARIANTS}
        assert metrics['variants']['baseline_2']['slices']['all']['n_error'] == 1


def test_opt_in_reproduces_ungrounded_evidence_error_without_weakening_parser(tmp_path, monkeypatch):
    def ungrounded(prompt):
        node = next(node for node in prompt.payload['subtasks'] if node['failure_samples'])
        return json.dumps({'predicted_failure_mode': 'position_error', 'predicted_subtask': node['subtask'],
                           'explanation': 'Synthetic unobserved evidence', 'status': 'ok',
                           'evidence': [{'subtask': node['subtask'], 'timestamp': 1000000,
                                         'feature': 'gripper_force', 'value': 0}]})

    bundle, results = _fixture(tmp_path, monkeypatch, backend='hf', response_text=ungrounded)
    verified = er.verify_gpu_results(bundle, results, score_response_errors=True)
    assert verified['response_error_requests'] == 2 and not verified['real_model_acceptance']
    for task in exp.TASKS:
        completion = _jsonl_rows(results / task / 'inference/completions.jsonl')[0]
        assert completion['status'] == 'error' and completion['error_type'] == 'ValueError'
        assert '1000000' in completion['text']


def _jsonl_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize('mutation', ['error_type', 'error_message', 'completion_only', 'missing_completion',
                                      'runtime_error', 'raw_text', 'metadata_error_fields'])
def test_opt_in_refuses_tampered_or_non_response_errors(tmp_path, monkeypatch, mutation):
    bundle, results = _fixture(tmp_path, monkeypatch, backend='hf', response_text='not JSON')
    directory = results / 'pick_up_object/inference'
    manifest = exp._json(directory / 'run_manifest.json')
    completions = _jsonl_rows(directory / 'completions.jsonl')
    request, completion = manifest['requests'][0], completions[0]
    if mutation in ('error_type', 'error_message'):
        request[mutation] = completion[mutation] = 'fabricated exception'
    elif mutation == 'completion_only':
        completion['error_message'] = 'fabricated exception'
    elif mutation == 'missing_completion':
        completions.pop(0)
    elif mutation == 'runtime_error':
        request['error_type'] = completion['error_type'] = 'ResourceSafetyError'
        request['error_message'] = completion['error_message'] = 'CUDA loading refused'
    elif mutation == 'raw_text':
        completion['text'] = 'tampered raw JSON'
    else:
        completion['metadata_error_fields'] = ['output_tokens']
    exp._write(directory / 'run_manifest.json', manifest)
    (directory / 'completions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in completions))
    _rebind(results)
    with pytest.raises((ValueError, TypeError)):
        er.verify_gpu_results(bundle, results, score_response_errors=True)


def test_opt_in_does_not_score_invalid_generation_metadata_as_response_error(tmp_path, monkeypatch):
    bundle, results = _fixture(tmp_path, monkeypatch, backend='hf', invalid_output_metadata=True)
    with pytest.raises(ValueError):
        er.verify_gpu_results(bundle, results, score_response_errors=True)


def test_opt_in_does_not_misclassify_valid_json_model_error_as_parser_exception(tmp_path, monkeypatch):
    text = json.dumps({'predicted_failure_mode': None, 'predicted_subtask': None, 'evidence': [],
                       'status': 'error', 'explanation': 'Synthetic model declined diagnosis'})
    bundle, results = _fixture(tmp_path, monkeypatch, backend='hf', response_text=text)
    with pytest.raises(ValueError):
        er.verify_gpu_results(bundle, results, score_response_errors=True)


def test_cli_scoring_opt_in_retains_failed_receipt_exit_status(tmp_path, monkeypatch, capsys):
    from scripts import evaluate_gpu_results
    captured = {}

    def evaluate(**kwargs):
        captured.update(kwargs)
        return {'status': 'failed', 'real_model_acceptance': False}

    monkeypatch.setattr(evaluate_gpu_results, 'evaluate_results', evaluate)
    assert evaluate_gpu_results.main([
        '--bundle', str(tmp_path / 'bundle'), '--results', str(tmp_path / 'results'),
        '--output-dir', str(tmp_path / 'evaluation'), '--score-response-errors',
    ]) == 1
    assert captured['score_response_errors'] is True
    assert 'failed' in capsys.readouterr().out
