
"""Tests for failure injection diagnostics and engine behavior."""

import json

import numpy as np
import pytest

from telemetry_reasoning.schemas import FailureMode, Subtask
from telemetry_reasoning.sim.failures import FailureConfig, FailureEngine
from test_dataset_validation import dataset



@pytest.mark.parametrize("task", ("pick_up_object", "transport_object"))
def test_released_failure_diagnostics(dataset, task):
    """Validate paired outcomes and activated failure injections."""
    path = dataset / task / "diagnostics.jsonl"

    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]

    assert len(records) == 500

    pair_ids = [str(record["pair_id"]) for record in records]
    assert len(set(pair_ids)) == 500

    for record in records:
        pair_id = record["pair_id"]
        nominal = record["nominal"]
        injected = record["injected"]
        failure = injected["failure"]

        # The unmodified controller must succeed.
        assert nominal["controller_outcome"] == "success", pair_id

        # The injected controller must report failure.
        assert injected["controller_outcome"] == "failed", pair_id

        # The failure injection must actually activate.
        assert failure["activated"] is True, pair_id
        assert failure["activation_steps"] > 0, pair_id
        assert failure["first_activation_s"] is not None, pair_id
        assert np.isfinite(float(failure["first_activation_s"])), pair_id

        # Both runs must start from the same initial condition.
        assert nominal["seed"] == injected["seed"], pair_id
        assert (
            nominal["initial_condition_id"]
            == injected["initial_condition_id"]
        ), pair_id
        assert (
            nominal["initial_offset_xy"]
            == injected["initial_offset_xy"]
        ), pair_id

        # Both runs must correspond to the correct task.
        assert nominal["task"] == task, pair_id
        assert injected["task"] == task, pair_id



def test_position_error_changes_target_only_when_active():
    """Position perturbation applies only after its configured delay."""
    config = FailureConfig(
        mode=FailureMode.POSITION_ERROR,
        subtask=Subtask.APPROACH,
        magnitude=0.05,
        delay_s=0.1,
        axis=0,
    )
    engine = FailureEngine(config)

    position = np.array([0.3, 0.2, 0.1])
    rotation = np.eye(3)

    engine.set_context("approach", 0.0)
    before, _, _ = engine.transform_target(position, rotation, 1.0)
    np.testing.assert_allclose(before, position)

    engine.set_context("approach", 0.1)
    after, _, _ = engine.transform_target(position, rotation, 1.0)
    np.testing.assert_allclose(after, position + [0.05, 0, 0])

    assert engine.diagnostics()["activated"] is True
    assert engine.diagnostics()["activation_steps"] == 1

    engine.set_context("grasp", 0.2)
    inactive, _, _ = engine.transform_target(position, rotation, 1.0)
    np.testing.assert_allclose(inactive, position)


def test_early_release_opens_gripper():
    """Early release commands the gripper to open when active."""
    config = FailureConfig(
        mode=FailureMode.EARLY_GRIPPER_RELEASE,
        subtask=Subtask.GRASP,
        magnitude=1.0,
    )
    engine = FailureEngine(config)

    class FakeData:
        def __init__(self):
            self.ctrl = np.zeros(8)

    data = FakeData()

    engine.set_context("grasp", 0.0)
    engine.transform_commands(data, "grasp")

    assert data.ctrl[7] == 255.0
    assert engine.diagnostics()["activated"] is True


@pytest.mark.parametrize('mode', list(FailureMode))
def test_failure_isolation_and_boundaries(mode):
    """All five mechanisms leave inputs untouched outside their target window."""
    config = FailureConfig(mode=mode, subtask=Subtask.GRASP, magnitude=2.0, delay_s=0.1)
    engine = FailureEngine(config)
    position, rotation = np.array([0.3, 0.2, 0.1]), np.eye(3)

    class Data:
        ctrl = np.zeros(8)

    data = Data()
    for stage, time in [('approach', 0.0), ('grasp', 1.0), ('grasp', 1.099)]:
        engine.set_context(stage, time)
        pos, rot, speed = engine.transform_target(position, rotation, 1.0)
        engine.transform_commands(data, stage)
        np.testing.assert_array_equal(pos, position)
        np.testing.assert_array_equal(rot, rotation)
        assert speed == 1.0 and np.all(data.ctrl == 0)
    engine.set_context('grasp', 1.1)
    pos, rot, speed = engine.transform_target(position, rotation, 1.0)
    engine.transform_commands(data, 'grasp')
    assert engine.diagnostics()['activated']
    if mode == FailureMode.POSITION_ERROR:
        np.testing.assert_allclose(pos, position + [2, 0, 0])
        np.testing.assert_array_equal(rot, rotation)
    elif mode == FailureMode.ORIENTATION_ERROR:
        np.testing.assert_array_equal(pos, position)
        assert not np.allclose(rot, rotation)
        np.testing.assert_allclose(rot.T @ rot, rotation, atol=1e-12)
    elif mode == FailureMode.EXCESSIVE_SPEED:
        assert speed == 2.0
    else:
        assert data.ctrl[7] == (255 if mode == FailureMode.EARLY_GRIPPER_RELEASE else 2)
    np.testing.assert_array_equal(position, [0.3, 0.2, 0.1])
    np.testing.assert_array_equal(rotation, np.eye(3))
