"""Contract and compression checks; these are not model evaluation tests."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from telemetry_reasoning.config import CompressionConfig, load_config
from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.parser import build_paired_bt, compress_rows, compress_subtask
from telemetry_reasoning.schemas import (
    TASK_SUBTASKS, DeterministicDivergence, FailureMode, PairedBehaviorTree,
    RawTelemetryRow, Subtask, Task,
)

ROOT = Path(__file__).resolve().parents[1]


def observation(index: int, **updates) -> RawTelemetryRow:
    data = {
        "timestamp": round(index * 0.02, 8), "subtask": "grasp",
        "joint_positions": [0.0, 0.0], "joint_velocities": [0.0, 0.0],
        "joint_torques": [0.0, 0.0], "gripper_state": 1.0, "gripper_force": 10.0,
        "ee_pos": [0.0, 0.0, 0.0], "ee_quat": [1.0, 0.0, 0.0, 0.0],
        "obj_pos": [0.0, 0.0, 0.0], "obj_quat": [1.0, 0.0, 0.0, 0.0],
        "obj_vel": [0.0, 0.0, 0.0], "contact_force": 10.0,
    }
    data.update(updates)
    return RawTelemetryRow.model_validate(data)


@pytest.fixture(scope="module")
def generated_dir(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("mock_pair")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_mock_pair.py"),
         "--output-dir", str(output_dir)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "400 rows per run" in result.stdout
    return output_dir


def test_mock_csvs_compress_into_valid_bt(generated_dir):
    config = load_config(ROOT / "configs/default.yaml")
    metadata, = read_metadata_csv(generated_dir / "metadata.csv")
    assert metadata.pair_id == "000"
    assert metadata.task == Task.PICK_UP_OBJECT
    assert metadata.failed_subtask == Subtask.GRASP
    assert metadata.failure_mode == FailureMode.INADEQUATE_GRIP_STRENGTH
    tree = build_paired_bt(metadata, generated_dir, config.parser)
    stored = PairedBehaviorTree.model_validate_json((generated_dir / "pair_000_bt.json").read_text())
    assert stored == tree
    assert tuple(node.subtask for node in tree.subtasks) == TASK_SUBTASKS[metadata.task]

    for name in ("success", "failure"):
        rows = read_telemetry_csv(generated_dir / getattr(metadata, f"{name}_file"))
        assert len(rows) == 400
        assert rows[0].timestamp == 0.0 and rows[-1].timestamp == 7.98
        all_timestamps = []
        for node in tree.subtasks:
            raw = [row for row in rows if row.subtask == node.subtask]
            samples = getattr(node, f"{name}_samples")
            assert samples[0] == raw[0] and samples[-1] == raw[-1]
            assert 7 <= len(samples) < len(raw)
            assert all(sample in raw for sample in samples)
            all_timestamps.extend(sample.timestamp for sample in samples)
        assert all_timestamps == sorted(set(all_timestamps))
    grasp = tree.subtasks[2]
    # Force drops at row 255 (5.10 s), deliberately off the uniform grid.
    by_time = {row.timestamp: row for row in grasp.failure_samples}
    assert {5.08, 5.10, 5.12} <= set(by_time)
    assert by_time[5.08].gripper_force == 15.0
    assert by_time[5.10].gripper_force == 3.0
    assert tree.subtasks[3].success_samples[-1].obj_pos[2] > 0.05
    assert tree.subtasks[3].failure_samples[-1].obj_pos[2] == 0.05
    assert all(node.deterministic_divergence is None for node in tree.subtasks)


def test_versioned_sample_is_reproducible(generated_dir):
    for name in ("success_000.csv", "failure_000.csv", "metadata.csv", "pair_000_bt.json"):
        assert (ROOT / "data/sample" / name).read_bytes() == (generated_dir / name).read_bytes()


def test_baseline_views_omit_ground_truth_and_excluded_inputs(generated_dir):
    tree = PairedBehaviorTree.model_validate_json((generated_dir / "pair_000_bt.json").read_text())
    # Populate only a test fixture, not a divergence implementation.
    tree.subtasks[2].deterministic_divergence = DeterministicDivergence(
        first_divergence_timestamp=5.1, max_abs_deltas={"gripper_force": 12.0},
    )
    failure = tree.to_failure_only()
    paired = tree.to_paired_without_divergence()
    full = tree.to_full_paired()
    for view in (failure, paired, full):
        for label in ("failed_subtask", "failure_mode", "perturbation_param",
                      "perturbation_value", "success_file", "failure_file", "seed"):
            assert label not in json.dumps(view)
        assert len(view["subtasks"]) == 4
        assert all(node["failure_samples"] and node["metadata"] for node in view["subtasks"])
    assert all("success_samples" not in node and "deterministic_divergence" not in node
               for node in failure["subtasks"])
    assert all(node["success_samples"] and "deterministic_divergence" not in node
               for node in paired["subtasks"])
    assert full["subtasks"][2]["deterministic_divergence"]["max_abs_deltas"] == {"gripper_force": 12.0}
    # Exported dictionaries are detached from the storage schema.
    full["subtasks"][0]["failure_samples"][0]["joint_positions"][0] = 999.0
    assert tree.subtasks[0].failure_samples[0].joint_positions[0] != 999.0


@pytest.mark.parametrize("n,k,expected", [
    (1, 5, [0]), (2, 5, [0, 1]), (4, 5, [0, 1, 2, 3]),
    (13, 5, [0, 2, 4, 6, 8, 10, 12]), (13, 0, [0, 12]),
])
def test_endpoints_and_uniform_samples(n, k, expected):
    rows = [observation(i) for i in range(n)]
    result = compress_subtask(rows, CompressionConfig(k_uniform_samples=k))
    assert [row.timestamp for row in result] == [rows[i].timestamp for i in expected]


@pytest.mark.parametrize("event", [1, 3, 11])
def test_events_keep_neighbors_clip_boundaries_and_deduplicate(event):
    rows = [observation(i, gripper_force=10.0 if i < event else 3.0,
                        ee_pos=[0.0, 0.0, 0.0 if i < event else 0.5]) for i in range(12)]
    config = CompressionConfig(k_uniform_samples=0,
                               major_event_thresholds={"gripper_force": 2.0, "ee_pos": 0.1})
    result = compress_subtask(rows, config)
    expected = {0, 11} | set(range(max(0, event - 1), min(12, event + 2)))
    assert [row.timestamp for row in result] == [rows[i].timestamp for i in sorted(expected)]


def test_threshold_is_strictly_exceeded():
    rows = [observation(i, gripper_force=10.0 if i < 4 else 8.0) for i in range(12)]
    config = CompressionConfig(k_uniform_samples=0, major_event_thresholds={"gripper_force": 2.0})
    assert compress_subtask(rows, config) == [rows[0], rows[-1]]


def test_consecutive_grouping_does_not_merge_revisited_subtasks_or_cross_boundaries():
    rows = [observation(i, subtask="align" if 6 <= i < 12 else "approach",
                        gripper_force=100.0 if 6 <= i < 12 else 0.0) for i in range(18)]
    segments = compress_rows(rows, CompressionConfig(
        k_uniform_samples=0, major_event_thresholds={"gripper_force": 2.0},
    ))
    assert [[row.timestamp for row in segment] for segment in segments] == [
        [0.0, 0.1], [0.12, 0.22], [0.24, 0.34],
    ]


@pytest.mark.parametrize("updates", [
    {"ee_pos": [0.0, 0.0]}, {"ee_quat": [1.0, 0.0, 0.0]},
    {"joint_velocities": [0.0]}, {"joint_positions": []},
    {"timestamp": -0.02}, {"gripper_state": 2.0}, {"gripper_force": float("nan")},
    {"obj_vel": [0.0, float("inf"), 0.0]}, {"subtask": "unknown"}, {"typo": 1},
])
def test_invalid_rows_are_rejected(updates):
    with pytest.raises(ValidationError):
        observation(0, **updates)


@pytest.mark.parametrize("rows", [
    [], [observation(0), observation(0)], [observation(1), observation(0)],
    [observation(0), observation(2)],
    [observation(0), observation(1, joint_positions=[0.0], joint_velocities=[0.0], joint_torques=[0.0])],
])
def test_invalid_run_order_cadence_and_dimensions_are_rejected(rows):
    with pytest.raises(ValueError):
        compress_rows(rows, CompressionConfig())


@pytest.mark.parametrize("updates", [
    {"k_uniform_samples": -1}, {"k_uniform_samples": 1.5},
    {"major_event_thresholds": {"gripper_force": -1.0}},
    {"major_event_thresholds": {"gripper_forc": 1.0}},
])
def test_invalid_compression_config_is_rejected(updates):
    with pytest.raises(ValidationError):
        CompressionConfig(**updates)


def test_bt_rejects_misaligned_or_duplicate_samples(generated_dir):
    data = json.loads((generated_dir / "pair_000_bt.json").read_text())
    data["subtasks"][0]["success_samples"].append(data["subtasks"][0]["success_samples"][0])
    with pytest.raises(ValidationError, match="strictly increasing"):
        PairedBehaviorTree.model_validate(data)
    metadata, = read_metadata_csv(generated_dir / "metadata.csv")
    metadata = metadata.model_copy(update={"task": Task.TRANSPORT_OBJECT, "failed_subtask": Subtask.MAINTAIN_GRASP})
    with pytest.raises(ValueError, match="ordered task template"):
        build_paired_bt(metadata, generated_dir, CompressionConfig())
