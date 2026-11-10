"""Offline regressions for the strict dataset handoff gate."""

import json
from pathlib import Path
import shutil

import pytest

from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv, write_telemetry_csv
from telemetry_reasoning.sim.freeze_robot_dataset import freeze_dataset
from telemetry_reasoning.sim.validate_dataset import validate_dataset


@pytest.fixture
def tiny_batch(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'data/sample'
    task = tmp_path / 'pick_up_object'
    task.mkdir()
    for name in ('metadata.csv', 'success_000.csv', 'failure_000.csv'):
        shutil.copyfile(source / name, task / name)
    meta = read_metadata_csv(task / 'metadata.csv')[0]
    identity = dict(task='pick_up_object', seed=meta.seed, initial_condition_id=0,
                    initial_offset_xy=[0, 0])
    record = dict(pair_id=meta.pair_id, initial_condition_id=0,
                  failure_config=dict(mode=meta.failure_mode.value, subtask=meta.failed_subtask.value),
                  nominal=dict(**identity, controller_outcome='success'),
                  injected=dict(**identity, controller_outcome='failed',
                                failure=dict(activated=True, activation_steps=1, first_activation_s=5.6)))
    (task / 'diagnostics.jsonl').write_text(json.dumps(record) + '\n', encoding='utf-8')
    return tmp_path


def test_complete_pair_passes_and_missing_task_is_fatal(tiny_batch):
    result = validate_dataset(tiny_batch)
    assert result['accepted_pairs'] == 1
    assert result['fatal_errors']  # incomplete full release cannot pass freeze


def test_partial_trace_is_quarantined_without_changing_source(tiny_batch):
    trace = tiny_batch / 'pick_up_object/failure_000.csv'
    rows = read_telemetry_csv(trace)
    write_telemetry_csv(trace, [r for r in rows if r.subtask == rows[0].subtask])
    original = trace.read_bytes()
    result = validate_dataset(tiny_batch)
    assert result['quarantined_pairs'] == 1
    assert 'failure: incomplete or unordered subtasks' in result['pairs'][0]['issues']
    assert trace.read_bytes() == original


def test_freeze_rejects_incomplete_release(tiny_batch):
    with pytest.raises(ValueError, match='cannot be frozen'):
        freeze_dataset(tiny_batch)


def test_duplicate_diagnostics_rejected(tiny_batch):
    path = tiny_batch / 'pick_up_object/diagnostics.jsonl'
    path.write_text(path.read_text() * 2, encoding='utf-8')
    assert 'duplicate diagnostic ID' in validate_dataset(tiny_batch)['fatal_errors'][0]


def test_missing_outcome_is_rejected(tiny_batch):
    path = tiny_batch / 'pick_up_object/diagnostics.jsonl'
    record = json.loads(path.read_text())
    del record['injected']['controller_outcome']
    path.write_text(json.dumps(record), encoding='utf-8')
    assert 'controller outcomes not success/failure' in validate_dataset(tiny_batch)['pairs'][0]['issues']
