"""Alignment contracts for Oct. 30; divergence scoring is a later milestone."""

import numpy as np
import pytest

from telemetry_reasoning.parser import align_subtask_samples
from telemetry_reasoning.schemas import (
    SUBTASK_METADATA, RawTelemetryRow, Subtask, SubtaskBT, TelemetryFeature,
)


def observation(timestamp, *, value=0.0, closure=1.0, quaternion=None, joints=7):
    quaternion = quaternion if quaternion is not None else [1.0, 0.0, 0.0, 0.0]
    return RawTelemetryRow(
        timestamp=timestamp, subtask=Subtask.MOVE_ALONG_PATH,
        joint_positions=[value] * joints, joint_velocities=[value] * joints,
        joint_torques=[value] * joints, gripper_state=closure, gripper_force=value,
        ee_pos=[value, 2 * value, -value], ee_quat=quaternion,
        obj_pos=[value, 2 * value, -value], obj_quat=quaternion,
        obj_vel=[value, 2 * value, -value], contact_force=value,
    )


def node(success, failure):
    return SubtaskBT(
        subtask=Subtask.MOVE_ALONG_PATH,
        metadata=SUBTASK_METADATA[Subtask.MOVE_ALONG_PATH].model_copy(deep=True),
        success_samples=success, failure_samples=failure,
    )


def test_union_alignment_removes_duration_and_start_shift_without_index_pairing():
    pair = node(
        [observation(0.0, value=0.0), observation(2.0, value=20.0)],
        [observation(10.0, value=0.0), observation(11.0, value=5.0),
         observation(13.0, value=15.0), observation(14.0, value=20.0)],
    )
    before = pair.model_dump_json()
    aligned = align_subtask_samples(pair)
    np.testing.assert_array_equal(aligned.progress, [0.0, 0.25, 0.75, 1.0])
    np.testing.assert_array_equal(aligned.failure_timestamps, [10.0, 11.0, 13.0, 14.0])
    assert set(aligned.success_values) == set(aligned.failure_values) == set(TelemetryFeature)
    for feature in TelemetryFeature:
        np.testing.assert_allclose(aligned.success_values[feature], aligned.failure_values[feature])
    np.testing.assert_allclose(aligned.success_values[TelemetryFeature.GRIPPER_FORCE], [0, 5, 15, 20])
    np.testing.assert_allclose(aligned.success_values[TelemetryFeature.OBJ_POS],
                               [[0, 0, 0], [5, 10, -5], [15, 30, -15], [20, 40, -20]])
    assert aligned.success_values[TelemetryFeature.JOINT_POSITIONS].shape == (4, 7)
    assert pair.model_dump_json() == before
    assert pair.deterministic_divergence is None
    # Returned comparisons are detached from the retained observations.
    aligned.success_values[TelemetryFeature.JOINT_POSITIONS][0, 0] = 999
    assert pair.success_samples[0].joint_positions[0] == 0.0


def test_grid_contains_both_traces_and_maps_interpolated_failure_times():
    pair = node(
        [observation(0.0), observation(0.5), observation(2.0)],
        [observation(10.0), observation(13.0), observation(14.0)],
    )
    aligned = align_subtask_samples(pair)
    np.testing.assert_array_equal(aligned.progress, [0.0, 0.25, 0.75, 1.0])
    np.testing.assert_array_equal(aligned.failure_timestamps, [10.0, 11.0, 13.0, 14.0])
    assert 11.0 not in [sample.timestamp for sample in pair.failure_samples]


def test_gripper_closure_uses_previous_value_including_exact_transition():
    pair = node(
        [observation(0.0, closure=0.0), observation(1.0, closure=1.0),
         observation(2.0, closure=0.0)],
        [observation(10.0), observation(10.5), observation(11.0),
         observation(11.5), observation(12.0)],
    )
    aligned = align_subtask_samples(pair)
    np.testing.assert_array_equal(aligned.success_values[TelemetryFeature.GRIPPER_STATE],
                                  [0, 0, 1, 1, 0])


@pytest.mark.parametrize("offset", [0.1, 1.0, 3.0, 10.0])
def test_shifted_decimal_timestamps_do_not_create_false_gripper_transitions(offset):
    success = [observation(round(i * 0.02, 8), value=float(i), closure=float(i >= 1))
               for i in range(4)]
    failure = [observation(round(offset + i * 0.02, 8), value=float(i), closure=float(i >= 1))
               for i in range(4)]
    aligned = align_subtask_samples(node(success, failure))
    for feature in TelemetryFeature:
        np.testing.assert_array_equal(aligned.success_values[feature], aligned.failure_values[feature])


def test_previous_value_does_not_snap_a_distinct_nearby_transition():
    pair = node(
        [observation(0.0, closure=0.0), observation(0.5, closure=1.0), observation(1.0)],
        [observation(0.0), observation(0.5 - 1e-8), observation(1.0)],
    )
    aligned = align_subtask_samples(pair)
    np.testing.assert_array_equal(aligned.success_values[TelemetryFeature.GRIPPER_STATE], [0, 0, 1, 1])


@pytest.mark.parametrize("singleton_side", ["success", "failure", "both"])
def test_singletons_are_constant_and_two_singletons_compare_once(singleton_side):
    success = [observation(3.0, value=5.0)]
    failure = [observation(8.0, value=5.0)]
    if singleton_side == "success":
        failure += [observation(8.5, value=7.0), observation(10.0, value=9.0)]
    elif singleton_side == "failure":
        success += [observation(3.5, value=7.0), observation(5.0, value=9.0)]
    aligned = align_subtask_samples(node(success, failure))
    expected_grid = [0.0] if singleton_side == "both" else [0.0, 0.25, 1.0]
    np.testing.assert_array_equal(aligned.progress, expected_grid)
    for samples, values in ((success, aligned.success_values), (failure, aligned.failure_values)):
        if len(samples) == 1:
            np.testing.assert_array_equal(values[TelemetryFeature.GRIPPER_FORCE],
                                          np.full(len(expected_grid), 5.0))
    expected_times = [row.timestamp for row in failure] if len(failure) > 1 else [8.0] * len(expected_grid)
    np.testing.assert_array_equal(aligned.failure_timestamps, expected_times)


def test_quaternions_are_normalized_sign_aligned_and_follow_shortest_path():
    # +170 to -170 degrees about z must pass through 180, rather than zero.
    angle = np.deg2rad(85)
    start = [np.cos(angle), 0.0, 0.0, np.sin(angle)]
    end = [np.cos(angle), 0.0, 0.0, -np.sin(angle)]
    pair = node(
        [observation(0.0, quaternion=(2 * np.array(start)).tolist()),
         observation(2.0, quaternion=end)],
        [observation(10.0, quaternion=(-3 * np.array(start)).tolist()),
         observation(11.0, quaternion=[0.0, 0.0, 0.0, -1.0]),
         observation(12.0, quaternion=(-np.array(end)).tolist())],
    )
    aligned = align_subtask_samples(pair)
    for feature in (TelemetryFeature.EE_QUAT, TelemetryFeature.OBJ_QUAT):
        success, failure = aligned.success_values[feature], aligned.failure_values[feature]
        np.testing.assert_allclose(np.linalg.norm(success, axis=1), 1.0)
        np.testing.assert_allclose(success, failure, atol=1e-14)
        np.testing.assert_allclose(success[1], [0, 0, 0, 1], atol=1e-14)


def test_quaternion_slerp_uses_angular_fraction_and_deterministic_orthogonal_tie():
    pair = node(
        [observation(0.0), observation(4.0, quaternion=[0.0, 0.0, 0.0, -1.0])],
        [observation(10.0), observation(11.0), observation(14.0)],
    )
    aligned = align_subtask_samples(pair)
    np.testing.assert_allclose(aligned.success_values[TelemetryFeature.EE_QUAT][1],
                               [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], atol=1e-14)
    again = align_subtask_samples(pair)
    for feature in TelemetryFeature:
        np.testing.assert_array_equal(aligned.success_values[feature], again.success_values[feature])
        np.testing.assert_array_equal(aligned.failure_values[feature], again.failure_values[feature])


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.04, 2.0])
def test_quaternion_interpolation_preserves_angular_fraction_even_for_small_rotations(angle):
    def quaternion(progress):
        return [np.cos(progress * angle / 2), np.sin(progress * angle / 2), 0.0, 0.0]

    pair = node(
        [observation(0.0, quaternion=quaternion(0)), observation(4.0, quaternion=quaternion(1))],
        [observation(10.0, quaternion=quaternion(0)), observation(11.0, quaternion=quaternion(.25)),
         observation(14.0, quaternion=quaternion(1))],
    )
    aligned = align_subtask_samples(pair)
    for feature in (TelemetryFeature.EE_QUAT, TelemetryFeature.OBJ_QUAT):
        np.testing.assert_allclose(aligned.success_values[feature][1], quaternion(.25), rtol=0, atol=1e-14)
        np.testing.assert_allclose(aligned.success_values[feature], aligned.failure_values[feature],
                                   rtol=0, atol=1e-14)


def test_seeded_nonuniform_grids_match_analytic_motion_for_every_continuous_feature():
    rng = np.random.default_rng(723)
    for _ in range(100):
        success_progress = np.r_[0.0, np.sort(rng.uniform(.01, .99, int(rng.integers(0, 15)))), 1.0]
        failure_progress = np.r_[0.0, np.sort(rng.uniform(.01, .99, int(rng.integers(0, 15)))), 1.0]
        start, duration = rng.uniform(0, 10), rng.uniform(.1, 8)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(.001, 2.8)

        def sample(timestamp, progress, sign):
            rotation = np.r_[np.cos(progress * angle / 2), axis * np.sin(progress * angle / 2)]
            return observation(float(timestamp), value=float(3 + 5 * progress),
                               quaternion=(sign * rotation).tolist())

        pair = node(
            [sample(u * 3, u, 1) for u in success_progress],
            [sample(start + u * duration, u, int(rng.choice([-1, 1]))) for u in failure_progress],
        )
        before = pair.model_dump_json()
        aligned = align_subtask_samples(pair)
        for feature in TelemetryFeature:
            np.testing.assert_allclose(aligned.success_values[feature], aligned.failure_values[feature],
                                       rtol=0, atol=2e-12)
        expected_force = 3 + 5 * aligned.progress
        expected_quat = np.column_stack((np.cos(aligned.progress * angle / 2),
                                         np.sin(aligned.progress * angle / 2)[:, None] * axis))
        np.testing.assert_allclose(aligned.success_values[TelemetryFeature.GRIPPER_FORCE], expected_force,
                                   rtol=0, atol=2e-12)
        np.testing.assert_allclose(aligned.success_values[TelemetryFeature.EE_QUAT], expected_quat,
                                   rtol=0, atol=2e-12)
        assert pair.model_dump_json() == before


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_quaternion_normalization_handles_finite_extreme_scales(scale):
    quaternion = [scale, 0.0, 0.0, scale]
    pair = node([observation(0.0, quaternion=quaternion)], [observation(1.0, quaternion=quaternion)])
    aligned = align_subtask_samples(pair)
    for values in (aligned.success_values, aligned.failure_values):
        np.testing.assert_allclose(values[TelemetryFeature.EE_QUAT][0],
                                   [np.sqrt(.5), 0, 0, np.sqrt(.5)], rtol=0, atol=1e-14)


@pytest.mark.parametrize("bad_side", ["success_samples", "failure_samples"])
@pytest.mark.parametrize("problem,match", [
    ("empty", "at least one"), ("duplicate", "strictly increasing"),
    ("reordered", "strictly increasing"), ("subtask", "one matching subtask"),
    ("joint_count", "same joint dimensions"), ("zero_quaternion", "nonzero quaternions"),
])
def test_alignment_rejects_invalid_segments(bad_side, problem, match):
    pair = node([observation(0.0), observation(1.0)], [observation(3.0), observation(4.0)])
    # Exercise API boundary validation even if an already-validated node was mutated.
    samples = getattr(pair, bad_side)
    if problem == "empty":
        samples.clear()
    elif problem == "duplicate":
        samples[1] = samples[0]
    elif problem == "reordered":
        samples.reverse()
    elif problem == "subtask":
        samples[0] = samples[0].model_copy(update={"subtask": Subtask.REACH_TARGET})
    elif problem == "joint_count":
        samples[:] = [observation(sample.timestamp, joints=2) for sample in samples]
    else:
        samples[0] = observation(samples[0].timestamp, quaternion=[0.0] * 4)
    with pytest.raises(ValueError, match=match):
        align_subtask_samples(pair)
