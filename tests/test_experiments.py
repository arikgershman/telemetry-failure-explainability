"""CPU-only bundle, tokenizer-count, split, and publication boundary checks."""

import copy
import json
from pathlib import Path

import pytest

from telemetry_reasoning import experiments as exp
from telemetry_reasoning.io import read_metadata_csv
from telemetry_reasoning.models.prompts import build_prompt
from telemetry_reasoning.schemas import PairedBehaviorTree

ROOT = Path(__file__).resolve().parents[1]


def _bundle(tmp_path, monkeypatch, **updates):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'configs').mkdir()
    (bundle / 'configs/llm.yaml').write_text((exp.ROOT / 'configs/llm.yaml').read_text())
    (bundle / 'input.txt').write_text('immutable input')
    source = {'test_source': {'sha256': '0' * 64, 'size_bytes': 0}}
    monkeypatch.setattr(exp, '_sources', lambda: copy.deepcopy(source))
    manifest = {'bundle_version': 1, 'status': 'ready_for_gpu',
                'source_files': source, 'files': exp._inventory(bundle),
                'token_budget': [{'fits': True}], 'pair_counts': {},
                'dataset_release_status': 'parser_ready_certification_failed', 'split': 'dev',
                **updates}
    exp._write(bundle / 'bundle_manifest.json', manifest)
    return bundle


def test_select_pairs_preserves_groups_and_test_holdout():
    row = read_metadata_csv(ROOT / 'data/sample/metadata.csv')[0]
    metadata = [row.model_copy(update={'pair_id': str(i)}) for i in range(3)]
    manifest = {'pairs': {str(i): {'split': split, 'initial_condition_id': i}
                          for i, split in enumerate(('train', 'dev', 'test'))}}
    assert [r.pair_id for r in exp.select_pairs(metadata, manifest)] == ['1']
    with pytest.raises(ValueError, match='final-study'):
        exp.select_pairs(metadata, manifest, split='test')
    assert [r.pair_id for r in exp.select_pairs(metadata, manifest, split='test', final_study=True)] == ['2']
    manifest['pairs']['2']['initial_condition_id'] = 1
    with pytest.raises(ValueError, match='overlaps'):
        exp.select_pairs(metadata, manifest)


def test_smoke_selects_one_pair_per_mode_without_changing_inputs():
    rows = read_metadata_csv(ROOT / 'data/sample/metadata.csv')
    row = rows[0]
    metadata = [row.model_copy(update={'pair_id': str(i)}) for i in range(3)]
    manifest = {'pairs': {str(i): {'split': 'dev', 'initial_condition_id': i} for i in range(3)}}
    assert exp.select_pairs(metadata, manifest, smoke=True) == [metadata[0]]
    assert len(metadata) == 3


def test_token_count_handles_modern_batch_encoding_without_counting_dict_keys():
    tree = PairedBehaviorTree.model_validate_json((ROOT / 'data/sample/pair_000_bt.json').read_bytes())
    prompt = build_prompt(tree, 'baseline_2')
    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            assert messages == prompt.messages
            assert options['truncation'] is False
            assert options['return_tensors'] is None
            assert options['add_generation_prompt'] is True
            return {'input_ids': [1, 2, 3, 4], 'attention_mask': [1, 1, 1, 1]}
    assert exp.prompt_token_count(Tokenizer(), prompt) == 4


@pytest.mark.parametrize('encoded', [
    {'input_ids': [], 'attention_mask': []},
    {'input_ids': [[1, 2]], 'attention_mask': [[1, 1]]},
    {'input_ids': [1, 2], 'attention_mask': [0, 1]},
    {'input_ids': [True], 'attention_mask': [1]},
])
def test_unexpected_tokenizer_shape_is_refused(encoded):
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return encoded
    with pytest.raises(ValueError, match='tokenizer result'):
        exp.prompt_token_count(Tokenizer(), type('Prompt', (), {'messages': []})())


def test_tokenizer_hash_gate_precedes_optional_import(tmp_path, monkeypatch):
    lock = tmp_path / 'lock.json'
    exp._write(lock, {'files': {'tokenizer.json': {'sha256': '0' * 64, 'size_bytes': 0}}})
    (tmp_path / 'tokenizer.json').write_text('corrupt')
    monkeypatch.setattr(exp.importlib, 'import_module', lambda _: pytest.fail('no optional imports'))
    with pytest.raises(ValueError, match='Pinned tokenizer file changed'):
        exp.load_cpu_tokenizer(tmp_path, lock)


@pytest.mark.parametrize('mutation', ['changed', 'added', 'deleted', 'symlink'])
def test_bundle_inventory_detects_artifact_changes(tmp_path, monkeypatch, mutation):
    bundle = _bundle(tmp_path, monkeypatch)
    assert exp.verify_experiment(bundle)['status'] == 'ready_for_gpu'
    if mutation == 'changed':
        (bundle / 'input.txt').write_text('same sized edits')
    elif mutation == 'added':
        (bundle / 'unexpected').write_text('extra')
    elif mutation == 'deleted':
        (bundle / 'input.txt').unlink()
    else:
        (bundle / 'link').symlink_to(bundle / 'input.txt')
    with pytest.raises(ValueError, match='inventory|Symlink'):
        exp.verify_experiment(bundle)


def test_source_changes_are_refused_before_output_creation(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(exp, '_sources', lambda: {'changed': {}})
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='implementation differs'):
        exp.run_experiment(bundle, output)
    assert not output.exists()


def test_failed_cpu_budget_never_starts_subprocess_or_creates_output(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, monkeypatch, status='blocked_context_budget')
    monkeypatch.setattr(exp.subprocess, 'run', lambda *a, **k: pytest.fail('must not start inference'))
    output = tmp_path / 'run'
    with pytest.raises(ValueError, match='preflight failed'):
        exp.run_experiment(bundle, output)
    assert not output.exists()


@pytest.mark.parametrize('relative', ['../escape', '/absolute', 'foo\\bar', './file', 'a//b'])
def test_unsafe_artifact_paths_are_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match='Unsafe'):
        exp._safe(tmp_path, relative)


def test_partial_run_writes_failure_receipt_without_claiming_acceptance(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, monkeypatch)
    def failure(*args, **kwargs):
        raise OSError('synthetic process startup failure')
    monkeypatch.setattr(exp.subprocess, 'run', failure)
    with pytest.raises(OSError, match='startup'):
        exp.run_experiment(bundle, tmp_path / 'run', backend='stub')
    receipt = json.loads((tmp_path / 'run/experiment_receipt.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['real_model_acceptance'] is False
    assert 'synthetic process startup failure' in receipt['failures'][0]


def test_inference_only_does_not_run_cpu_metrics_or_plots(tmp_path, monkeypatch):
    from types import SimpleNamespace
    bundle = _bundle(tmp_path, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(exp.subprocess, 'run', run)
    monkeypatch.setattr(exp, '_check_task_run', lambda *a: {'context_refused_requests': 0})
    output = tmp_path / 'inference-only'
    assert exp.run_experiment(bundle, output, backend='stub', inference_only=True) == 0
    assert len(calls) == len(exp.TASKS)
    assert all(command[1] == 'scripts/run_llm.py' for command in calls)
    assert not list(output.rglob('evaluation'))
    receipt = json.loads((output / 'experiment_receipt.json').read_text())
    assert receipt['inference_only'] is True
    assert receipt['evaluation'] == 'deferred_to_cpu'
    assert receipt['real_model_acceptance'] is False
    assert all(stage['stage'] == 'inference' for stage in receipt['stages'])


def test_public_cli_forwards_inference_only(tmp_path, monkeypatch):
    from scripts import gpu_experiment
    calls = []
    monkeypatch.setattr(gpu_experiment, 'run_experiment', lambda **kwargs: calls.append(kwargs) or 0)
    assert gpu_experiment.main(['run', '--bundle', str(tmp_path / 'bundle'),
                                '--output-dir', str(tmp_path / 'out'), '--inference-only']) == 0
    assert calls[0]['inference_only'] is True
    assert calls[0]['backend'] == 'hf'


def test_modified_production_observations_cannot_enter_an_experiment(tmp_path):
    production = tmp_path / 'production'
    production.mkdir()
    tree = production / 'pair_000_bt.json'
    tree.write_text('original observations')
    handoff_manifest = tmp_path / 'dataset_manifest.json'
    handoff_manifest.write_text('frozen source receipt')
    manifest = {'source_inventory_sha256': 'frozen raw data', 'validation': {'accepted_pairs': 1}}
    exp._write(production / 'production_report.json', {
        'production_version': 1, 'status': 'validated_parser_publication',
        'sources_unchanged': True, 'source_inventory_sha256': 'frozen raw data',
        'input_sha256': {str(handoff_manifest.resolve()): exp._fingerprint(handoff_manifest)['sha256']},
        'pair_count': 1, 'outputs': {tree.name: exp._fingerprint(tree)},
    })
    exp._verify_production(production, manifest, handoff_manifest)
    tree.write_text('modified observations')
    with pytest.raises(ValueError, match='Production BT/report inventory changed'):
        exp._verify_production(production, manifest, handoff_manifest)


def test_exact_tokenizer_stub_remains_synthetic_and_measures_completion(tmp_path, monkeypatch):
    from telemetry_reasoning.models.llm import LLMConfig
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}
        def __call__(self, text, **kwargs):
            assert kwargs['add_special_tokens'] is False
            return {'input_ids': [4, 5]}
    config = LLMConfig()
    monkeypatch.setattr(exp, 'load_cpu_tokenizer', lambda _: (Tokenizer(), {
        'model_id': config.model_id, 'revision': config.revision, 'files': {},
    }))
    backend = exp.TokenizerStubBackend(config, tmp_path)
    generated = backend.generate(type('Prompt', (), {'messages': []})())
    assert generated.input_tokens == 3 and generated.output_tokens == 2
    assert generated.token_count_method == 'pinned_tokenizer_cpu'
    assert json.loads(generated.text)['status'] == 'abstain'
    assert backend.provenance()['weights_loaded'] is False


def test_cpu_tokenizer_flag_cannot_override_real_backend(tmp_path, capsys):
    from scripts.run_llm import main
    assert main(['--backend', 'hf', '--stub-tokenizer-dir', str(tmp_path),
                 '--bt-dir', str(ROOT / 'data/sample'), '--output-dir', str(tmp_path / 'run')]) == 2
    assert 'only with --backend stub' in capsys.readouterr().err
    assert not (tmp_path / 'run').exists()


def _saved_context_run(tmp_path):
    from telemetry_reasoning.models.llm import Generation, LLMConfig, StubBackend, run_inference
    from telemetry_reasoning.parser.divergence_metrics import enrich_paired_bt, load_divergence_config
    tree = PairedBehaviorTree.model_validate_json((ROOT / 'data/sample/pair_000_bt.json').read_bytes())
    tree = enrich_paired_bt(tree, load_divergence_config(ROOT / 'configs/divergence.yaml'))
    config = LLMConfig(context_window_tokens=100, max_new_tokens=16)
    class SmallStub(StubBackend):
        token_count_method = 'test_counter'
        def count_tokens(self, prompt):
            return 5 if prompt.variant.value == 'baseline_2' else 100
        def generate(self, prompt):
            return Generation(json.dumps({'predicted_failure_mode': None, 'predicted_subtask': None,
                                          'explanation': 'Synthetic test only', 'evidence': [], 'status': 'abstain'}),
                              self.count_tokens(prompt), 5, self.token_count_method)
    run = run_inference([tree], exp.VARIANTS, config, backend=SmallStub(config))
    bundle, target = tmp_path / 'bundle', tmp_path / 'output'
    (bundle / 'configs').mkdir(parents=True)
    (bundle / 'tasks/pick_up_object/bt').mkdir(parents=True)
    exp._write(bundle / 'configs/llm.yaml', {'llm': config.model_dump(mode='json')})
    (bundle / 'tasks/pick_up_object/bt/pair_000_bt.json').write_text(tree.model_dump_json())
    (target / 'inference').mkdir(parents=True)
    for variant in exp.VARIANTS:
        exp.write_predictions(target / 'inference' / f'{variant}.jsonl',
                              [r for r in run.predictions if r['model_variant'] == variant])
    (target / 'inference/completions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in run.completions))
    exp._write(target / 'inference/run_manifest.json', run.manifest)
    manifest = {'token_budget': [{'task': 'pick_up_object', **r,
                                 'fits': r['input_tokens'] + config.max_new_tokens <= config.context_window_tokens}
                                for r in run.manifest['requests']]}
    return bundle, target, manifest


def test_context_refusals_remain_explicit_and_all_other_responses_are_reparsed(tmp_path):
    bundle, target, manifest = _saved_context_run(tmp_path)
    result = exp._check_task_run(bundle, 'pick_up_object', target, manifest, 'stub')
    assert result == {'expected_requests': 3, 'validated_fit_requests': 1,
                      'context_refused_requests': 2, 'full_request_acceptance': False}
    path = target / 'inference/run_manifest.json'
    record = json.loads(path.read_text())
    record['requests'][1]['error_type'] = 'ResourceSafetyError'
    exp._write(path, record)
    with pytest.raises(ValueError, match='expected explicit context refusal'):
        exp._check_task_run(bundle, 'pick_up_object', target, manifest, 'stub')


def test_fitting_generation_failures_and_missing_predictions_are_not_context_exceptions(tmp_path):
    bundle, target, manifest = _saved_context_run(tmp_path)
    (target / 'inference/baseline_2.jsonl').write_text('')
    with pytest.raises(ValueError, match='inference coverage'):
        exp._check_task_run(bundle, 'pick_up_object', target, manifest, 'stub')


def test_cli_expected_errors_are_clean(capsys, tmp_path):
    from scripts.gpu_experiment import main
    assert main(['verify', '--bundle', str(tmp_path / 'missing')]) == 2
    assert 'error:' in capsys.readouterr().err
