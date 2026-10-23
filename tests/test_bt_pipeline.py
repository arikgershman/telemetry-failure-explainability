"""Synthetic batch/transport contracts; no physics or model dependencies."""

import csv
from pathlib import Path

import pytest

from telemetry_reasoning.config import CompressionConfig, load_config
from telemetry_reasoning.io import read_metadata_csv, write_telemetry_csv
from telemetry_reasoning.parser import build_paired_bt, build_paired_bts
from telemetry_reasoning.parser import pipeline
from telemetry_reasoning.schemas import (
    SUBTASK_METADATA, TASK_GOALS, TASK_SUBTASKS, MetadataRow, PairedBehaviorTree,
    RawTelemetryRow, Task,
)

ROOT = Path(__file__).resolve().parents[1]


def synthetic_run(stages, counts, *, start=0.0, joints=7):
    rows = []
    for subtask, count in zip(stages, counts):
        for _ in range(count):
            rows.append(RawTelemetryRow(
                timestamp=round(start + len(rows) * 0.02, 8), subtask=subtask,
                joint_positions=[0.0] * joints, joint_velocities=[0.0] * joints,
                joint_torques=[0.0] * joints, gripper_state=1.0, gripper_force=10.0,
                ee_pos=[0.0, 0.0, 0.2], ee_quat=[1.0, 0.0, 0.0, 0.0],
                obj_pos=[0.0, 0.0, 0.1], obj_quat=[1.0, 0.0, 0.0, 0.0],
                obj_vel=[0.0, 0.0, 0.0], contact_force=10.0,
            ))
    return rows


def write_pair(directory, pair_id, task=Task.TRANSPORT_OBJECT):
    stages = TASK_SUBTASKS[task]
    metadata = MetadataRow(
        pair_id=pair_id, task=task,
        success_file=f"success_{pair_id}.csv", failure_file=f"failure_{pair_id}.csv",
        failed_subtask=stages[1], failure_mode="position_error",
        perturbation_param="offset_m", perturbation_value=0.1, seed=723,
    )
    for name in ("success", "failure"):
        write_telemetry_csv(directory / getattr(metadata, f"{name}_file"),
                            synthetic_run(stages, [5] * len(stages)))
    return metadata


def test_existing_sample_batch_matches_versioned_tree():
    data_dir = ROOT / "data/sample"
    config = load_config(ROOT / "configs/default.yaml")
    result = build_paired_bts(read_metadata_csv(data_dir / "metadata.csv"), data_dir, config.parser)
    assert not result.errors
    tree, = result.trees
    assert tree == PairedBehaviorTree.model_validate_json((data_dir / "pair_000_bt.json").read_text())
    assert all(node.deterministic_divergence is None for node in tree.subtasks)


def test_transport_preserves_semantics_and_unequal_shifted_stage_timestamps(tmp_path):
    metadata = write_pair(tmp_path, "transport_000")
    stages = TASK_SUBTASKS[metadata.task]
    success = synthetic_run(stages, [3, 9, 1])
    failure = synthetic_run(stages, [6, 4, 2], start=3.0)
    write_telemetry_csv(tmp_path / metadata.success_file, success)
    write_telemetry_csv(tmp_path / metadata.failure_file, failure)
    config = CompressionConfig(k_uniform_samples=1)
    result = build_paired_bts([metadata], tmp_path, config)
    assert not result.errors
    tree, = result.trees
    assert tree == build_paired_bt(metadata, tmp_path, config)
    assert tree.overall_goal == TASK_GOALS[metadata.task]
    assert tuple(node.subtask for node in tree.subtasks) == stages
    for node in tree.subtasks:
        assert node.metadata == SUBTASK_METADATA[node.subtask]
        assert node.deterministic_divergence is None
        for name, raw in (("success", success), ("failure", failure)):
            segment = [row for row in raw if row.subtask == node.subtask]
            samples = getattr(node, f"{name}_samples")
            assert samples[0] == segment[0] and samples[-1] == segment[-1]
            assert all(sample in segment for sample in samples)
            assert all(len(sample.joint_positions) == 7 for sample in samples)
    assert [len(node.success_samples) for node in tree.subtasks] == [3, 3, 1]
    assert [len(node.failure_samples) for node in tree.subtasks] == [3, 3, 2]
    assert PairedBehaviorTree.model_validate_json(tree.model_dump_json()) == tree


@pytest.mark.parametrize("problem", ["missing_file", "invalid_csv", "joint_mismatch"])
def test_batch_reports_pair_errors_and_continues_in_input_order(tmp_path, problem):
    first = write_pair(tmp_path, "pickup_000", Task.PICK_UP_OBJECT)
    bad = write_pair(tmp_path, "bad")
    last = write_pair(tmp_path, "transport_000")
    path = tmp_path / bad.failure_file
    if problem == "missing_file":
        path.unlink()
    elif problem == "invalid_csv":
        path.write_text("timestamp,subtask\n0.0,maintain_grasp\n")
    else:
        write_telemetry_csv(path, synthetic_run(TASK_SUBTASKS[bad.task], [5, 5, 5], joints=2))
    metadata = [first, bad, last]
    before = [row.model_dump_json() for row in metadata]
    config = CompressionConfig()
    result = build_paired_bts(metadata, tmp_path, config)
    assert [tree.metadata.pair_id for tree in result.trees] == [first.pair_id, last.pair_id]
    error, = result.errors
    assert (error.pair_id, error.success_file, error.failure_file) == (
        bad.pair_id, bad.success_file, bad.failure_file,
    )
    assert error.error_type and error.message
    assert len(result.trees) + len(result.errors) == len(metadata)
    assert [row.model_dump_json() for row in metadata] == before
    assert build_paired_bts(metadata, tmp_path, config) == result


@pytest.mark.parametrize("task", list(Task))
@pytest.mark.parametrize("problem", ["missing", "repeated", "reordered"])
@pytest.mark.parametrize("side", ["success", "failure"])
def test_batch_rejects_incomplete_or_revisited_stages(tmp_path, task, problem, side):
    metadata = write_pair(tmp_path, "bad_order", task)
    stages = list(TASK_SUBTASKS[task])
    if problem == "missing":
        stages.pop(1)
    elif problem == "repeated":
        stages.append(stages[0])
    else:
        stages[0], stages[1] = stages[1], stages[0]
    write_telemetry_csv(tmp_path / getattr(metadata, f"{side}_file"),
                        synthetic_run(stages, [3] * len(stages)))
    result = build_paired_bts([metadata], tmp_path, CompressionConfig())
    assert not result.trees
    error, = result.errors
    assert error.pair_id == metadata.pair_id
    assert f"{side} run must contain the complete ordered task template" in error.message


def test_empty_batch_and_duplicate_ids(tmp_path):
    config = CompressionConfig()
    empty = build_paired_bts([], tmp_path, config)
    assert empty.trees == empty.errors == ()
    metadata = write_pair(tmp_path, "duplicate")
    with pytest.raises(ValueError, match="pair_id values must be unique"):
        build_paired_bts([metadata, metadata], tmp_path, config)


@pytest.mark.parametrize("side", ["success", "failure"])
@pytest.mark.parametrize("problem,match", [
    ("empty", "CSV contains no observations"), ("cadence", "0.02 seconds"),
    ("nonfinite", "finite"), ("bad_vector", "3 items"),
    ("bad_json", "invalid telemetry row"), ("duplicate_time", "strictly increasing"),
])
def test_raw_validation_failures_are_reported_without_losing_valid_pairs(tmp_path, side, problem, match):
    bad = write_pair(tmp_path, "bad_raw")
    good = write_pair(tmp_path, "good")
    path = tmp_path / getattr(bad, f"{side}_file")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, records = reader.fieldnames, list(reader)
    if problem == "empty":
        records.clear()
    elif problem == "cadence":
        records[1]["timestamp"] = "0.021"
    elif problem == "nonfinite":
        records[1]["gripper_force"] = "nan"
    elif problem == "bad_vector":
        records[1]["ee_pos"] = "[0.0,0.0]"
    elif problem == "bad_json":
        records[1]["ee_quat"] = "[broken"
    else:
        records[1]["timestamp"] = records[0]["timestamp"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    result = build_paired_bts([bad, good], tmp_path, CompressionConfig())
    assert [tree.metadata.pair_id for tree in result.trees] == [good.pair_id]
    error, = result.errors
    assert error.pair_id == bad.pair_id
    assert match in error.message


def test_all_rejected_pairs_keep_error_order(tmp_path):
    metadata = [write_pair(tmp_path, pair_id) for pair_id in ("z", "a", "m")]
    for row in metadata:
        (tmp_path / row.failure_file).unlink()
    result = build_paired_bts(metadata, tmp_path, CompressionConfig())
    assert not result.trees
    assert [error.pair_id for error in result.errors] == [row.pair_id for row in metadata]
    assert all(error.error_type == "FileNotFoundError" for error in result.errors)


def test_duplicate_preflight_precedes_file_reads_and_programming_errors_propagate(tmp_path, monkeypatch):
    metadata = write_pair(tmp_path, "duplicate")

    def broken_builder(*args):
        raise RuntimeError("unexpected builder defect")

    monkeypatch.setattr(pipeline, "build_paired_bt", broken_builder)
    with pytest.raises(ValueError, match="pair_id values must be unique"):
        build_paired_bts([metadata, metadata], tmp_path, CompressionConfig())
    with pytest.raises(RuntimeError, match="unexpected builder defect"):
        build_paired_bts([metadata], tmp_path, CompressionConfig())
