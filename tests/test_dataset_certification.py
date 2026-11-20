"""Release gates must distinguish reproducible evidence from plausible labels."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv, write_telemetry_csv
from telemetry_reasoning.sim.dataset_certification import (
    compare_pre_injection, inventory_sha256, inventory_sources, load_evidence,
    producer_certification, replay_verified,
)
from telemetry_reasoning.sim.freeze_robot_dataset import (
    freeze_dataset, is_release_certified, publish_handoff, task_manifests,
    verify_dataset, verify_handoff, write_manifest,
)
from telemetry_reasoning.sim.validate_dataset import validate_dataset
from test_dataset_acceptance import tiny_batch  # noqa: F401


def evidence_for(batch):
    task = 'pick_up_object'
    files = inventory_sources(batch)
    meta = read_metadata_csv(batch / task / 'metadata.csv')[0]
    pair = {'task': task, 'pair_id': meta.pair_id}
    for kind, key in (('success', 'nominal'), ('failure', 'injected')):
        relative = f'{task}/{getattr(meta, f"{kind}_file")}'
        rows = read_telemetry_csv(batch / relative)
        comparison = {'matched': True, 'all_rows': True, 'all_fields': True,
                      'row_count': len(rows), 'replayed_row_count': len(rows),
                      'atol': 1e-8, 'rtol': 0, 'identity_matched': True,
                      'duration_matched': True, 'injection_diagnostics_matched': True}
        pair[key] = {
            'file': relative, 'source_sha256': files[relative]['sha256'],
            'origin': {'timestamp_s': 0.0, 'basis': 'verified_exact_replay', 'verified': True},
            'replay': {**comparison, 'repeats': 2, 'fresh_repeatability_matched': True,
                       'repeat_comparisons': [dict(comparison), dict(comparison)]},
        }
    return {'schema_version': 1, 'source_files': files,
            'source_inventory_sha256': inventory_sha256(files), 'pairs': [pair]}


def test_absent_evidence_never_certifies_controller_labels(tiny_batch):
    result = validate_dataset(tiny_batch)
    record = result['pairs'][0]
    assert record['accepted']
    assert record['certification']['independent_physical_outcomes']['status'] == 'pending'
    assert record['certification']['deterministic_replay']['status'] == 'pending'
    assert record['certification']['complete_pre_injection_agreement']['status'] == 'pending'
    assert 'injection_timestamp' not in record
    assert record['injection_time_basis'] == 'absolute_simulator_time'


@pytest.mark.parametrize('missing', [
    'repeats', 'repeat_comparisons', 'fresh_repeatability_matched',
])
def test_single_receipt_booleans_cannot_certify_repeatability(tiny_batch, missing):
    evidence = evidence_for(tiny_batch)
    del evidence['pairs'][0]['injected']['replay'][missing]
    record = validate_dataset(tiny_batch, evidence=evidence)['pairs'][0]
    assert record['certification']['deterministic_replay']['status'] == 'pending'
    assert 'injection_timestamp' not in record


@pytest.mark.parametrize('field', ['identity_matched', 'duration_matched', 'injection_diagnostics_matched'])
@pytest.mark.parametrize('value', [None, False])
def test_every_repeat_requires_identity_duration_and_injection_evidence(tiny_batch, field, value):
    evidence = evidence_for(tiny_batch)
    comparison = evidence['pairs'][0]['injected']['replay']['repeat_comparisons'][1]
    if value is None:
        comparison.pop(field)
    else:
        comparison[field] = value
    record = validate_dataset(tiny_batch, evidence=evidence)['pairs'][0]
    assert record['certification']['deterministic_replay']['status'] == ('failed' if value is False else 'pending')
    assert not record['clock_origins']['injected']['verified']
    assert 'injection_timestamp' not in record
    assert record['injection_time_basis'] == 'absolute_simulator_time'


def test_replay_binding_enables_only_verified_clock_conversion(tiny_batch):
    evidence = evidence_for(tiny_batch)
    result = validate_dataset(tiny_batch, evidence=evidence)
    record = result['pairs'][0]
    assert record['certification']['deterministic_replay']['status'] == 'passed'
    assert record['injection_timestamp'] == 5.6
    assert record['injection_conversion_verified'] is True
    assert result['certification']['status'] != 'certified'  # tiny/incomplete and no producer


def test_nominal_replay_never_verifies_injected_clock(tiny_batch):
    evidence = evidence_for(tiny_batch)
    evidence['pairs'][0]['injected']['origin'] = {
        'timestamp_s': 0, 'basis': 'published_nominal_replay', 'verified': True}
    record = validate_dataset(tiny_batch, evidence=evidence)['pairs'][0]
    assert record['clock_origins']['nominal']['verified'] is True
    assert record['clock_origins']['injected']['verified'] is False
    assert record['certification']['complete_pre_injection_agreement']['status'] == 'pending'
    assert 'injection_timestamp' not in record


def test_legacy_aligned_diagnostics_cannot_masquerade_as_recorded_clocks(tiny_batch):
    path = tiny_batch / 'pick_up_object/diagnostics.jsonl'
    diagnostic = json.loads(path.read_text())
    diagnostic['clock_alignment_source'] = 'verified_nominal_replay'
    for key in ('nominal', 'injected'):
        diagnostic[key]['telemetry_start_sim_s'] = 0.0
    path.write_text(json.dumps(diagnostic) + '\n')
    record = validate_dataset(tiny_batch)['pairs'][0]
    assert not record['clock_origins']['injected']['verified']
    assert record['clock_origins']['injected']['basis'] == 'verified_nominal_replay'
    assert 'injection_timestamp' not in record
    # A new independently verified repeated replay can still establish each
    # origin, without modifying or trusting the old aligned diagnostics.
    record = validate_dataset(tiny_batch, evidence=evidence_for(tiny_batch))['pairs'][0]
    assert record['clock_origins']['injected']['verified']
    assert record['clock_origins']['injected']['basis'] == 'verified_exact_replay'


@pytest.mark.parametrize('change', [
    {'matched': False}, {'all_rows': False}, {'all_fields': False},
    {'row_count': -1}, {'atol': 1e-6}, {'rtol': 1e-7}, {'atol': float('nan')},
])
def test_incomplete_or_loose_replay_is_pending(tiny_batch, change):
    evidence = evidence_for(tiny_batch)
    evidence['pairs'][0]['injected']['replay'].update(change)
    record = validate_dataset(tiny_batch, evidence=evidence)['pairs'][0]
    assert record['certification']['deterministic_replay']['status'] == (
        'failed' if any(value is False for value in change.values()) else 'pending')
    assert not record['clock_origins']['injected']['verified']
    assert 'injection_timestamp' not in record


@pytest.mark.parametrize('mutation', ['trace', 'metadata', 'diagnostics', 'digest'])
def test_evidence_source_tampering_fails_before_use(tiny_batch, mutation):
    evidence = evidence_for(tiny_batch)
    relative = {'trace': 'pick_up_object/failure_000.csv',
                'metadata': 'pick_up_object/metadata.csv',
                'diagnostics': 'pick_up_object/diagnostics.jsonl'}.get(mutation)
    if relative:
        evidence['source_files'][relative]['sha256'] = '0' * 64
    else:
        evidence['source_inventory_sha256'] = '0' * 64
    result = validate_dataset(tiny_batch, evidence=evidence)
    assert any('certification evidence' in issue for issue in result['fatal_errors'])
    assert result['pairs'][0]['certification']['deterministic_replay']['status'] == 'pending'


def test_missing_metadata_identity_binding_is_rejected(tiny_batch):
    evidence = evidence_for(tiny_batch)
    del evidence['source_files']['pick_up_object/metadata.csv']
    evidence.pop('source_inventory_sha256')
    assert 'identity binding' in load_evidence(tiny_batch, evidence)[2][0]


def test_duplicate_evidence_pairs_are_rejected(tiny_batch):
    evidence = evidence_for(tiny_batch)
    evidence['pairs'].append(copy.deepcopy(evidence['pairs'][0]))
    assert 'duplicate evidence pair' in load_evidence(tiny_batch, evidence)[2][0]


def test_wrong_pair_trace_binding_cannot_pass_replay(tiny_batch):
    evidence = evidence_for(tiny_batch)
    evidence['pairs'][0]['injected']['file'] = evidence['pairs'][0]['nominal']['file']
    record = validate_dataset(tiny_batch, evidence=evidence)['pairs'][0]
    assert record['certification']['deterministic_replay']['status'] == 'pending'


def test_complete_prefix_checks_later_rows_and_stage_labels(tiny_batch):
    rows = read_telemetry_csv(tiny_batch / 'pick_up_object/success_000.csv')
    nominal = [row.model_copy(deep=True) for row in rows]
    failure = [row.model_copy(deep=True) for row in rows]
    origin = {'verified': True, 'timestamp_s': 1.0}
    result = compare_pre_injection(nominal, failure, origin, origin, 1.2)
    assert result['status'] == 'passed'
    assert result['compared_rows'] == 10
    failure[5].joint_positions[0] += 1e-4
    result = compare_pre_injection(nominal, failure, origin, origin, 1.2)
    assert result['status'] == 'failed'
    assert result['first_mismatch'] == {'row': 5, 'field': 'joint_positions', 'timestamp_s': .1}
    failure[5] = nominal[5].model_copy(update={'subtask': 'align'})
    assert compare_pre_injection(nominal, failure, origin, origin, 1.2)['status'] == 'failed'


def test_prefix_does_not_include_post_activation_changes(tiny_batch):
    rows = read_telemetry_csv(tiny_batch / 'pick_up_object/success_000.csv')
    failure = [row.model_copy(deep=True) for row in rows]
    failure[10].joint_positions[0] += 1.0
    origin = {'verified': True, 'timestamp_s': 1.0}
    assert compare_pre_injection(rows, failure, origin, origin, 1.2)['status'] == 'passed'
    assert compare_pre_injection(rows, failure, origin, origin, 1.0)['status'] == 'pending'


def test_origin_offset_does_not_hide_missing_prefix(tiny_batch):
    rows = read_telemetry_csv(tiny_batch / 'pick_up_object/success_000.csv')
    assert compare_pre_injection(rows, rows, {'verified': True, 'timestamp_s': 1.0},
                                 {'verified': True, 'timestamp_s': 1.02}, 1.2)['status'] == 'failed'


def test_physical_flags_are_not_substitutes_for_terminal_measurements(tiny_batch):
    evidence = evidence_for(tiny_batch)
    for key, status in (('nominal', 'succeeded'), ('injected', 'failed')):
        evidence['pairs'][0][key]['physical_outcome'] = {'status': status}
    record = validate_dataset(tiny_batch, evidence=evidence)['pairs'][0]
    assert record['certification']['independent_physical_outcomes']['status'] == 'pending'


def test_physical_objective_is_recomputed_for_exact_replay(tiny_batch):
    evidence = evidence_for(tiny_batch)
    for key, rise in (('nominal', .2), ('injected', 0)):
        initial = read_telemetry_csv(tiny_batch / evidence['pairs'][0][key]['file'])[0].obj_pos
        final = list(initial)
        final[2] += rise
        evidence['pairs'][0][key]['terminal_state'] = {
            'initial_position': initial, 'final_position': final,
            'final_velocity': [0, 0, 0], 'terminal_observed': True}
    record = validate_dataset(tiny_batch, evidence=evidence)['pairs'][0]
    assert record['certification']['independent_physical_outcomes']['status'] == 'passed'
    evidence['pairs'][0]['injected']['terminal_state']['final_position'][2] += .2
    record = validate_dataset(tiny_batch, evidence=evidence)['pairs'][0]
    assert record['certification']['independent_physical_outcomes']['status'] == 'failed'


def test_retrospective_revision_is_not_producer_provenance():
    claimed = {'revision': 'a' * 40, 'tree_clean': False,
               'source_files': {'runner.py': 'b' * 64}, 'model_files': {'model.xml': 'c' * 64}}
    assert producer_certification({'producer_provenance': claimed})['status'] == 'pending'
    claimed['recorded_at_generation'] = True
    assert producer_certification({'producer_provenance': claimed})['status'] == 'passed'


def test_relative_injection_is_exposed_only_after_verified_conversion(tiny_batch):
    manifest = handoff_manifest()
    record = validate_dataset(tiny_batch, evidence=evidence_for(tiny_batch))['pairs'][0]
    pair = {'task': record['task'], 'pair_id': record['pair_id']}
    manifest['validation']['pairs'] = [record]
    manifest['splits']['splits']['train'] = [pair]
    relative = task_manifests(manifest)['pick_up_object']['pairs'][record['pair_id']]
    assert relative['injection_timestamp'] == 5.6
    assert relative['injection_time_basis'] == 'absolute_simulator_time'
    record.pop('injection_conversion_verified')
    absolute_only = task_manifests(manifest)['pick_up_object']['pairs'][record['pair_id']]
    assert 'injection_timestamp' not in absolute_only
    assert absolute_only['injection_timestamp_s'] == 5.6


def test_headline_marker_does_not_override_incomplete_certification():
    assert not is_release_certified({'certification': {'status': 'certified'}})


def test_freeze_strict_gate_and_input_mutation(tmp_path, monkeypatch):
    from telemetry_reasoning.sim import freeze_robot_dataset as module
    from telemetry_reasoning.sim import validate_dataset as validator
    source = tmp_path / 'pick_up_object/source.csv'
    source.parent.mkdir()
    source.write_text('original source')
    validation = {'accepted_pairs': 1000, 'quarantined_pairs': 0, 'fatal_errors': [],
                  'certification': {'status': 'pending', 'gates': {
                      'producer_provenance': {'status': 'pending'}}}}
    monkeypatch.setattr(validator, 'validate_dataset', lambda *args, **kwargs: validation)
    monkeypatch.setattr(module, 'create_splits', lambda *args, **kwargs: {})
    monkeypatch.setattr(module, 'source_provenance', lambda: {})
    assert freeze_dataset(tmp_path)['release_status'] == 'parser_ready_pending_certification'
    with pytest.raises(ValueError, match='not release-certified: producer_provenance'):
        freeze_dataset(tmp_path, require_certified=True)
    def mutate(*args, **kwargs):
        source.write_text('source changed during validation')
        return validation
    monkeypatch.setattr(validator, 'validate_dataset', mutate)
    with pytest.raises(ValueError, match='changed during validation'):
        freeze_dataset(tmp_path)


def test_public_freeze_cli_cleans_strict_certification_refusal(tmp_path, monkeypatch, capsys):
    from telemetry_reasoning.sim import freeze_robot_dataset as module
    from telemetry_reasoning.sim import validate_dataset as validator
    validation = {'accepted_pairs': 1000, 'quarantined_pairs': 0, 'fatal_errors': [],
                  'certification': {'status': 'pending', 'gates': {
                      'producer_provenance': {'status': 'pending'}}}}
    monkeypatch.setattr(validator, 'validate_dataset', lambda *args, **kwargs: validation)
    monkeypatch.setattr(module, 'create_splits', lambda *args, **kwargs: {})
    output = tmp_path / 'certified-handoff'
    code = module.main([str(tmp_path), '--require-certified', '--handoff-dir', str(output)])
    captured = capsys.readouterr()
    assert code == 2
    assert 'not release-certified: producer_provenance' in captured.err
    assert 'Traceback' not in captured.err
    assert not output.exists()


@pytest.mark.parametrize('content,expected_code', [
    ('not json', 2), ('[]', 1), ('{"files":[]}', 1),
    ('{"files":{},"certification":null}', 1),
    ('{"files":{},"certification":{"gates":null}}', 1),
    ('{"files":{},"validation":null}', 1),
])
def test_public_freeze_cli_handles_malformed_manifest_cleanly(tmp_path, content, expected_code):
    path = tmp_path / 'malformed.json'
    path.write_text(content)
    result = subprocess.run([sys.executable, '-m', 'telemetry_reasoning.sim.freeze_robot_dataset',
                             str(tmp_path), '--verify', str(path), '--require-certified'], capture_output=True, text=True)
    assert result.returncode == expected_code
    assert 'Traceback' not in result.stderr + result.stdout
    assert 'failed' in (result.stderr + result.stdout).lower()


def handoff_manifest():
    return {'metadata_layout': {task: f'{task}/metadata.csv'
                                for task in ('pick_up_object', 'transport_object')},
            'split_seed': 723, 'verification_scope': {},
            'release_status': 'parser_ready_pending_certification',
            'validation': {'pairs': []},
            'splits': {'splits': {'train': [], 'dev': [], 'test': []}}}


def test_complete_handoff_is_published_atomically_and_refuses_overwrite(tmp_path, monkeypatch):
    from telemetry_reasoning.sim import freeze_robot_dataset as module
    destination = tmp_path / 'handoff'
    real_rename = module.os.rename
    def checked_rename(source, target):
        assert not destination.exists()
        assert sorted(path.relative_to(source).as_posix() for path in source.rglob('*.json')) == [
            'dataset_manifest.json', 'dataset_splits.json',
            'publication_receipt.json',
            'tasks/pick_up_object.json', 'tasks/transport_object.json']
        return real_rename(source, target)
    monkeypatch.setattr(module.os, 'rename', checked_rename)
    path = publish_handoff(destination, handoff_manifest())
    original = path.read_bytes()
    assert json.loads(original)['release_status'] == 'parser_ready_pending_certification'
    assert verify_handoff(destination) == []
    with pytest.raises(FileExistsError):
        publish_handoff(destination, handoff_manifest())
    assert path.read_bytes() == original


def test_handoff_detects_task_artifact_tampering(tmp_path):
    destination = tmp_path / 'handoff'
    publish_handoff(destination, handoff_manifest())
    (destination / 'tasks/pick_up_object.json').write_text('{}')
    assert verify_handoff(destination) == ['Handoff artifact changed: tasks/pick_up_object.json']


def test_failed_publication_leaves_no_partial_handoff(tmp_path, monkeypatch):
    from telemetry_reasoning.sim import freeze_robot_dataset as module
    def fail(*args):
        raise OSError('simulated write failure')
    monkeypatch.setattr(module, '_write_json', fail)
    with pytest.raises(OSError, match='simulated'):
        publish_handoff(tmp_path / 'handoff', handoff_manifest())
    assert list(tmp_path.iterdir()) == []


def test_single_manifest_write_refuses_overwrite(tmp_path):
    path = tmp_path / 'manifest.json'
    write_manifest(path, {'original': True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_manifest(path, {'original': False})
    assert path.read_bytes() == original


def test_symlinked_source_is_rejected(tiny_batch):
    source = tiny_batch / 'pick_up_object/failure_000.csv'
    alternate = source.with_suffix('.original')
    source.rename(alternate)
    source.symlink_to(alternate.name)
    result = validate_dataset(tiny_batch)
    assert result['quarantined_pairs'] == 1
    assert any('symlink source path' in issue for issue in result['pairs'][0]['issues'])
    with pytest.raises(ValueError, match='symlink'):
        inventory_sources(tiny_batch)


def test_reused_hardlink_is_not_a_distinct_paired_episode(tiny_batch):
    import os
    success = tiny_batch / 'pick_up_object/success_000.csv'
    failure = tiny_batch / 'pick_up_object/failure_000.csv'
    failure.unlink()
    os.link(success, failure)
    assert any('reused telemetry file inodes' in error for error in validate_dataset(tiny_batch)['fatal_errors'])


def test_legacy_windows_inventory_is_verified_without_changing_sources(tmp_path):
    folder = tmp_path / 'pick_up_object'
    folder.mkdir()
    (folder / 'file.csv').write_text('source')
    files = inventory_sources(tmp_path)
    manifest = {'manifest_version': 1, 'files': {key.replace('/', '\\'): value for key, value in files.items()}}
    assert verify_dataset(tmp_path, manifest) == []
