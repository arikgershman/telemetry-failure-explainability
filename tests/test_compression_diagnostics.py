"""Compression quality uses raw events, independent of candidate thresholds."""

import json
from math import cos, sin

import pytest

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.parser.compressor import compress_rows
from telemetry_reasoning.parser.diagnostics import combine_diagnostics, compression_diagnostics
from telemetry_reasoning.schemas import (
    SUBTASK_METADATA, TASK_GOALS, TASK_SUBTASKS, FailureMode, MetadataRow,
    PairedBehaviorTree, RawTelemetryRow, SubtaskBT, Task, TelemetryFeature,
)


def observation(index, subtask, *, joints=7, **updates):
    values = {
        "timestamp": round(index * 0.02, 8), "subtask": subtask,
        "joint_positions": [0.0] * joints, "joint_velocities": [0.0] * joints,
        "joint_torques": [0.0] * joints, "gripper_state": 1.0, "gripper_force": 10.0,
        "ee_pos": [0.0] * 3, "ee_quat": [1.0, 0.0, 0.0, 0.0],
        "obj_pos": [0.0] * 3, "obj_quat": [1.0, 0.0, 0.0, 0.0],
        "obj_vel": [0.0] * 3, "contact_force": 10.0,
    }
    values.update(updates)
    return RawTelemetryRow.model_validate(values)


def raw_pair(task=Task.PICK_UP_OBJECT, lengths=None, *, spike=False, joints=7):
    stages = TASK_SUBTASKS[task]
    lengths = lengths if lengths is not None else [9] * len(stages)
    success, failure = [], []
    index = 0
    for stage_index, (subtask, length) in enumerate(zip(stages, lengths)):
        for local_index in range(length):
            success.append(observation(index, subtask, joints=joints))
            force = 1.0 if spike and stage_index == 0 and local_index == 4 else 10.0
            failure.append(observation(index, subtask, joints=joints, gripper_force=force))
            index += 1
    return success, failure


def tree_for(success, failure, config, task=Task.PICK_UP_OBJECT):
    return PairedBehaviorTree(
        metadata=MetadataRow(
            pair_id="opaque", task=task, success_file="success.csv", failure_file="failure.csv",
            failed_subtask=TASK_SUBTASKS[task][0], failure_mode=FailureMode.POSITION_ERROR,
            perturbation_param="private label", perturbation_value=1.0, seed=723,
        ),
        overall_goal=TASK_GOALS[task],
        subtasks=[SubtaskBT(
            subtask=subtask, metadata=SUBTASK_METADATA[subtask].model_copy(deep=True),
            success_samples=success_stage, failure_samples=failure_stage,
        ) for subtask, success_stage, failure_stage in zip(
            TASK_SUBTASKS[task], compress_rows(success, config), compress_rows(failure, config),
        )],
    )


@pytest.mark.parametrize("task,joints", [(Task.PICK_UP_OBJECT, 2), (Task.TRANSPORT_OBJECT, 7)])
def test_fixed_raw_reference_detects_events_lost_by_candidate(task, joints):
    success, failure = raw_pair(task, spike=True, joints=joints)
    lossy = CompressionConfig(k_uniform_samples=0, major_event_thresholds={"gripper_force": 20.0})
    retaining = CompressionConfig(k_uniform_samples=0, major_event_thresholds={"gripper_force": 2.0})
    reference = {"gripper_force": 2.0}
    lost = compression_diagnostics(success, failure, tree_for(success, failure, lossy, task), lossy, reference)
    kept = compression_diagnostics(success, failure, tree_for(success, failure, retaining, task), retaining, reference)
    lost_events = lost["totals"]["event_retention"]["gripper_force"]
    kept_events = kept["totals"]["event_retention"]["gripper_force"]
    assert lost_events["raw_events"] == kept_events["raw_events"] == 2
    assert lost_events["retained_event_rows"] == lost_events["retained_event_neighborhoods"] == 0
    assert lost_events["raw_neighbor_rows"] == 4
    assert lost_events["retained_neighbor_rows"] == 0
    assert lost_events["event_retention_fraction"] == 0.0
    assert kept_events["retained_event_rows"] == kept_events["retained_event_neighborhoods"] == 2
    assert kept_events["neighbor_retention_fraction"] == 1.0
    assert lost["runs"]["failure"]["stages"][0]["reconstruction_error"]["gripper_force"]["max_abs_error"] == 9.0
    assert kept["totals"]["reconstruction_error"]["gripper_force"]["max_abs_error"] == 0.0
    # Reusing each candidate's thresholds as the reference would hide this loss.
    candidate_reference = compression_diagnostics(success, failure, tree_for(success, failure, lossy, task), lossy)
    assert candidate_reference["reference_thresholds"] == {"gripper_force": 20.0}
    assert candidate_reference["totals"]["event_retention"]["gripper_force"]["raw_events"] == 0


def test_event_centers_and_complete_neighborhoods_are_distinct_metrics():
    success, failure = raw_pair(spike=True)
    config = CompressionConfig(k_uniform_samples=1)
    report = compression_diagnostics(success, failure, tree_for(success, failure, config), config,
                                     {"gripper_force": 2.0})
    events = report["totals"]["event_retention"]["gripper_force"]
    assert events["raw_events"] == 2
    assert events["retained_event_rows"] == 1  # uniform center catches the spike
    assert events["retained_event_neighborhoods"] == 0
    assert events["retained_neighbor_rows"] == 1
    assert events["neighbor_retention_fraction"] == 0.25


def test_strict_component_events_never_cross_stage_boundaries():
    success, failure = raw_pair(lengths=[3, 3, 3, 3])
    for rows in (success, failure):
        for index, row in enumerate(rows):
            local = index % 3
            rows[index] = row.model_copy(update={
                "gripper_force": float(index // 3 * 100 + min(local, 1) * 2),
                "joint_positions": [0.0] * 6 + [float(local * 3)],
            })
    config = CompressionConfig(k_uniform_samples=0)
    report = compression_diagnostics(success, failure, tree_for(success, failure, config), config,
                                     {"gripper_force": 2.0, "joint_positions": 2.0})
    events = report["totals"]["event_retention"]
    assert events["gripper_force"]["raw_events"] == 0  # equality and 100 N boundaries excluded
    assert events["gripper_force"]["event_retention_fraction"] is None
    assert events["joint_positions"]["raw_events"] == 16  # any joint component qualifies
    assert events["joint_positions"]["retained_event_neighborhoods"] == 0


@pytest.mark.parametrize("task,lengths", [
    (Task.PICK_UP_OBJECT, [1, 2, 3, 1]), (Task.TRANSPORT_OBJECT, [2, 1, 2]),
])
def test_short_stages_and_empty_event_denominators(task, lengths):
    success, failure = raw_pair(task, lengths)
    config = CompressionConfig(k_uniform_samples=5, major_event_thresholds={"ee_pos": 0.03})
    tree = tree_for(success, failure, config, task)
    report = compression_diagnostics(success, failure, tree, config)
    assert report["totals"]["raw_rows"] == report["totals"]["retained_rows"] == 2 * sum(lengths)
    assert report["totals"]["compression_ratio"] == report["totals"]["retained_fraction"] == 1.0
    assert [stage["raw_rows"] for stage in report["runs"]["success"]["stages"]] == lengths
    assert report["totals"]["event_retention"]["ee_pos"]["neighbor_retention_fraction"] is None
    assert all(metric["max_abs_error"] == 0.0 for metric in report["totals"]["reconstruction_error"].values())
    json.dumps(report, allow_nan=False)


def test_reconstruction_reuses_linear_previous_value_and_quaternion_semantics():
    success, failure = raw_pair()
    for rows in (success, failure):
        for index, row in enumerate(rows):
            local = index % 9
            angle = local / 8.0
            rows[index] = row.model_copy(update={
                "joint_positions": [float(local)] * 7,
                "ee_pos": [float(local * local), 0.0, 0.0],
                "gripper_state": float(local >= 4),
                "ee_quat": [cos(angle / 2), 0.0, 0.0, sin(angle / 2)],
            })
    config = CompressionConfig(k_uniform_samples=0)
    report = compression_diagnostics(success, failure, tree_for(success, failure, config), config)
    errors = report["runs"]["failure"]["stages"][0]["reconstruction_error"]
    assert errors["joint_positions"]["max_abs_error"] == pytest.approx(0.0, abs=1e-14)
    assert errors["joint_positions"]["component_observations"] == 9 * 7
    assert errors["ee_pos"]["max_abs_error"] == 16.0
    assert errors["gripper_state"]["max_abs_error"] == 1.0
    assert errors["gripper_state"]["mean_abs_error"] == pytest.approx(4 / 9)
    assert errors["ee_quat"]["max_abs_error"] == pytest.approx(0.0, abs=1e-14)


def test_zero_quaternion_marks_only_its_reconstruction_feature_unavailable():
    success, failure = raw_pair(spike=True)
    failure[4] = failure[4].model_copy(update={"ee_quat": [0.0] * 4})
    config = CompressionConfig(k_uniform_samples=0)
    report = compression_diagnostics(success, failure, tree_for(success, failure, config), config)
    errors = report["totals"]["reconstruction_error"]
    assert errors["ee_quat"]["max_abs_error"] is None
    assert errors["ee_quat"]["mean_abs_error"] is None
    assert errors["ee_quat"]["unavailable_stages"] == 1
    assert errors["obj_quat"]["max_abs_error"] == 0.0
    assert errors["gripper_force"]["max_abs_error"] == 9.0
    assert combine_diagnostics([report])["totals"]["reconstruction_error"]["ee_quat"]["max_abs_error"] is None
    json.dumps(report, allow_nan=False)


def test_canonical_prompt_payload_size_excludes_labels_ids_and_templates_without_mutation():
    success, failure = raw_pair(spike=True)
    config = CompressionConfig(k_uniform_samples=0)
    tree = tree_for(success, failure, config)
    tree.overall_goal = "Pick up café object"
    reference = {TelemetryFeature.GRIPPER_FORCE: 2.0}
    before = ([row.model_dump_json() for row in success], [row.model_dump_json() for row in failure],
              tree.model_dump_json(), config.model_dump_json(), dict(reference))
    report = compression_diagnostics(success, failure, tree, config, reference)
    for name, view in (("failure_only", tree.to_failure_only()),
                       ("paired_without_divergence", tree.to_paired_without_divergence()),
                       ("full_paired", tree.to_full_paired())):
        view.pop("pair_id")
        text = json.dumps(view, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        assert report["prompt_payload_sizes"][name] == {"characters": len(text), "utf8_bytes": len(text.encode("utf-8"))}
        assert len(text.encode("utf-8")) > len(text)
    changed = tree.model_copy(deep=True)
    changed.metadata = changed.metadata.model_copy(update={
        "pair_id": "a very different and longer join ID", "success_file": "secret success file",
        "failure_file": "private failure file", "seed": 1024,
        "failure_mode": FailureMode.EXCESSIVE_SPEED, "perturbation_param": "secret perturbation",
        "perturbation_value": 99999.0,
    })
    assert compression_diagnostics(success, failure, changed, config, reference)["prompt_payload_sizes"] == report["prompt_payload_sizes"]
    after = ([row.model_dump_json() for row in success], [row.model_dump_json() for row in failure],
             tree.model_dump_json(), config.model_dump_json(), dict(reference))
    assert before == after
    assert all(node.deterministic_divergence is None for node in tree.subtasks)
    assert "not model tokens" in report["definitions"]["prompt_payload_sizes"]


def test_aggregation_recomputes_counts_and_weights_reconstruction():
    config = CompressionConfig(k_uniform_samples=0)
    reports = []
    for length in (9, 5):
        success, failure = raw_pair(lengths=[length] * 4, spike=True)
        reports.append(compression_diagnostics(success, failure, tree_for(success, failure, config),
                                               config, {"gripper_force": 2.0}))
    combined = combine_diagnostics(reports)
    assert combined["pair_count"] == 2
    assert combined["totals"]["raw_rows"] == 2 * 4 * 14
    assert combined["totals"]["retained_rows"] == 2 * 2 * 4 * 2
    assert combined["totals"]["compression_ratio"] == 3.5
    metrics = [report["totals"]["reconstruction_error"]["gripper_force"] for report in reports]
    expected_mean = sum(metric["mean_abs_error"] * metric["component_observations"] for metric in metrics) / sum(metric["component_observations"] for metric in metrics)
    assert combined["totals"]["reconstruction_error"]["gripper_force"]["mean_abs_error"] == pytest.approx(expected_mean)
    for view in combined["prompt_payload_sizes"]:
        assert combined["prompt_payload_sizes"][view]["utf8_bytes"] == sum(report["prompt_payload_sizes"][view]["utf8_bytes"] for report in reports)
    empty = combine_diagnostics([])
    assert empty["pair_count"] == empty["totals"]["raw_rows"] == 0
    assert empty["totals"]["compression_ratio"] is None
    json.dumps(empty, allow_nan=False)
    changed = dict(reports[0], reference_thresholds={"gripper_force": 4.0})
    with pytest.raises(ValueError, match="different compression/reference settings"):
        combine_diagnostics([reports[0], changed])


@pytest.mark.parametrize("problem", ["empty", "cadence", "template", "edited", "endpoint", "duplicate", "dimensions"])
def test_diagnostics_rejects_invalid_raw_runs_or_nonoriginal_retained_rows(problem):
    success, failure = raw_pair()
    config = CompressionConfig(k_uniform_samples=0)
    tree = tree_for(success, failure, config)
    if problem == "empty":
        success = []
    elif problem == "cadence":
        success[1] = success[1].model_copy(update={"timestamp": 0.03})
    elif problem == "template":
        success = success[:9]
    elif problem == "edited":
        tree.subtasks[0].failure_samples.insert(1, failure[4].model_copy(update={"gripper_force": 8.0}))
    elif problem == "endpoint":
        tree.subtasks[0].failure_samples.pop(0)
    elif problem == "duplicate":
        tree.subtasks[0].failure_samples.insert(1, tree.subtasks[0].failure_samples[0])
    else:
        failure = [observation(index, row.subtask, joints=2) for index, row in enumerate(failure)]
    with pytest.raises(ValueError):
        compression_diagnostics(success, failure, tree, config)


@pytest.mark.parametrize("thresholds", [{"unknown_feature": 2.0}, {"gripper_force": -1.0}, {"gripper_force": float("nan")}])
def test_reference_thresholds_use_existing_strict_configuration_validation(thresholds):
    success, failure = raw_pair()
    config = CompressionConfig()
    with pytest.raises(ValueError):
        compression_diagnostics(success, failure, tree_for(success, failure, config), config, thresholds)
