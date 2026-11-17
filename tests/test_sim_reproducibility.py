"""Opt-in real physics checks: TRACK2_REAL_SIM=1 python -m pytest this_file."""

import os
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TRACK2_REAL_SIM') != '1',
                                reason='Set TRACK2_REAL_SIM=1 for real MuJoCo checks')


@pytest.mark.parametrize('task', ['pick_up_object', 'transport_object'])
def test_reset_and_nominal_replay(task):
    pytest.importorskip('mujoco')
    from telemetry_reasoning.sim.runner import MODEL_PATH, run_episode
    if not Path(MODEL_PATH).is_file():
        pytest.skip('MuJoCo scene assets absent')
    first = run_episode(task=task, seed=723, initial_condition_id=0)
    second = run_episode(task=task, seed=723, initial_condition_id=0)
    assert first.task_succeeded and second.task_succeeded
    assert first.diagnostics == second.diagnostics
    assert len(first.rows) == len(second.rows)
    for a, b in zip(first.rows, second.rows):
        assert a.subtask == b.subtask
        for field in type(a).model_fields:
            if field != 'subtask':
                np.testing.assert_allclose(getattr(a, field), getattr(b, field), atol=1e-8, rtol=0)


@pytest.mark.parametrize('task', ['pick_up_object', 'transport_object'])
def test_nominal_success_all_initial_conditions(task):
    pytest.importorskip('mujoco')
    from telemetry_reasoning.sim.runner import run_episode
    for condition in range(50):
        result = run_episode(task=task, seed=723, initial_condition_id=condition)
        assert result.task_succeeded, (task, condition, result.diagnostics)
        assert result.diagnostics['complete_subtasks']
        start, end = result.rows[0], result.rows[-1]
        if task == 'pick_up_object':
            assert end.obj_pos[2] - start.obj_pos[2] > 0.12
        else:
            expected_xy = np.array(start.obj_pos[:2]) + [0, 0.15]
            assert np.linalg.norm(np.array(end.obj_pos[:2]) - expected_xy) < 0.04


@pytest.mark.parametrize('task', ['pick_up_object', 'transport_object'])
def test_injected_replay_representative_modes(task):
    pytest.importorskip('mujoco')
    from telemetry_reasoning.schemas import FailureMode, Subtask
    from telemetry_reasoning.sim.failures import FailureConfig
    from telemetry_reasoning.sim.runner import run_episode
    dataset = Path(__file__).resolve().parents[1] / 'outputs/dataset/batch_20261030_152125_e5ca72de'
    path = dataset / task / 'diagnostics.jsonl'
    if not path.exists():
        pytest.skip('Released diagnostics absent')
    representatives = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        representatives.setdefault(record['failure_config']['mode'], record)
    assert len(representatives) == 4
    for record in representatives.values():
        config = dict(record['failure_config'])
        config['mode'] = FailureMode(config['mode'])
        config['subtask'] = Subtask(config['subtask'])
        kwargs = dict(task=task, seed=record['nominal']['seed'],
                      initial_condition_id=record['initial_condition_id'], failure=FailureConfig(**config))
        first, second = run_episode(**kwargs), run_episode(**kwargs)
        assert not first.task_succeeded and not second.task_succeeded
        assert first.diagnostics == second.diagnostics
        assert first.diagnostics['failure']['activated']
        assert len(first.rows) == len(second.rows)
        for a, b in zip(first.rows, second.rows):
            assert a.subtask == b.subtask
            for field in type(a).model_fields:
                if field != 'subtask':
                    np.testing.assert_allclose(getattr(a, field), getattr(b, field), atol=1e-8, rtol=0)
