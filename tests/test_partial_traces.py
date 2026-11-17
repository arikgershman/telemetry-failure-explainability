"""Track 3 contracts for authentic failure prefixes; no simulator/model inputs."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.io import write_metadata_csv, write_telemetry_csv
from telemetry_reasoning.parser import (
    build_bt_from_segments, build_paired_bt, build_paired_bts, parse_dataset,
    validate_subtask_sequence,
)
from telemetry_reasoning.schemas import (
    TASK_SUBTASKS, DeterministicDivergence, MetadataRow, PairedBehaviorTree,
    RawTelemetryRow, SubtaskBT, Task, TelemetryFeature,
)

ROOT = Path(__file__).resolve().parents[1]


def raw_run(stages, count=9, *, joints=7):
    rows = []
    for subtask in stages:
        for _ in range(count):
            rows.append(RawTelemetryRow(
                timestamp=round(len(rows) * 0.02, 8), subtask=subtask,
                joint_positions=[0.0] * joints, joint_velocities=[0.0] * joints,
                joint_torques=[0.0] * joints, gripper_state=1.0, gripper_force=10.0,
                ee_pos=[0.0, 0.0, 0.2], ee_quat=[1.0, 0.0, 0.0, 0.0],
                obj_pos=[0.0, 0.0, 0.1], obj_quat=[1.0, 0.0, 0.0, 0.0],
                obj_vel=[0.0, 0.0, 0.0], contact_force=10.0,
            ))
    return rows


def pair_files(directory, task=Task.PICK_UP_OBJECT, prefix_length=2, pair_id="partial"):
    stages = TASK_SUBTASKS[task]
    metadata = MetadataRow(
        pair_id=pair_id, task=task, success_file=f"{pair_id}_success.csv",
        failure_file=f"{pair_id}_failure.csv", failed_subtask=stages[0],
        failure_mode="position_error", perturbation_param="offset_m",
        perturbation_value=0.1, seed=723,
    )
    success = raw_run(stages)
    # End within the final observed stage; do not invent later observations.
    failure = raw_run(stages[:prefix_length])[:(prefix_length - 1) * 9 + 3]
    write_telemetry_csv(directory / metadata.success_file, success)
    write_telemetry_csv(directory / metadata.failure_file, failure)
    return metadata, success, failure


@pytest.mark.parametrize("task,length", [
    (task, length) for task in Task for length in range(1, len(TASK_SUBTASKS[task]) + 1)
])
def test_every_failure_prefix_preserves_complete_success_structure(tmp_path, task, length):
    metadata, success, failure = pair_files(tmp_path, task, length)
    originals = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    config = CompressionConfig(k_uniform_samples=1)
    result = build_paired_bts([metadata], tmp_path, config)
    assert not result.errors
    tree, = result.trees
    expected = TASK_SUBTASKS[task]
    assert tuple(node.subtask for node in tree.subtasks) == expected
    assert "schema_version" not in tree.model_dump()
    for index, node in enumerate(tree.subtasks):
        success_segment = [row for row in success if row.subtask == node.subtask]
        assert node.success_samples[0] == success_segment[0]
        assert node.success_samples[-1] == success_segment[-1]
        failure_segment = [row for row in failure if row.subtask == node.subtask]
        assert bool(node.failure_samples) == (index < length)
        assert all(row in failure_segment for row in node.failure_samples)
        if failure_segment:
            assert node.failure_samples[0] == failure_segment[0]
            assert node.failure_samples[-1] == failure_segment[-1]
        assert isinstance(node, SubtaskBT)
        assert node.failure_observation_status == ("observed" if index < length else "not_observed")
        expected_end = None if index >= length else "trace_end" if index == length - 1 else "transitioned"
        assert node.failure_segment_end == expected_end
        assert node.divergence_status == ("uncomputed" if index < length else "not_comparable")
        assert node.deterministic_divergence is None
    assert PairedBehaviorTree.model_validate_json(tree.model_dump_json()) == tree
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == originals


@pytest.mark.parametrize("task", list(Task))
def test_incomplete_success_is_rejected_even_when_failure_is_valid_prefix(tmp_path, task):
    metadata, _, failure = pair_files(tmp_path, task, 1)
    write_telemetry_csv(tmp_path / metadata.success_file, failure)
    with pytest.raises(ValueError, match="success run must contain the complete ordered task template"):
        build_paired_bt(metadata, tmp_path, CompressionConfig())


@pytest.mark.parametrize("task", list(Task))
def test_prefix_validation_rejects_gaps_revisits_reorderings_and_empty_sequences(task):
    expected = TASK_SUBTASKS[task]
    invalid = [(), expected[1:], (expected[0], expected[2]),
               (expected[0], expected[0]), expected + (expected[0],),
               (expected[1], expected[0])]
    for observed in invalid:
        with pytest.raises(ValueError, match="nonempty ordered prefix"):
            validate_subtask_sequence(observed, task, allow_partial=True, run_name="failure")
    for length in range(1, len(expected)):
        with pytest.raises(ValueError, match="complete ordered task template"):
            validate_subtask_sequence(expected[:length], task, run_name="success")


def test_segment_builder_refuses_empty_segments_instead_of_creating_observations(tmp_path):
    metadata, _, _ = pair_files(tmp_path)
    with pytest.raises(ValueError, match="at least one observation"):
        build_bt_from_segments(metadata, [[]], [[]])
    with pytest.raises(ValueError, match="complete ordered task template"):
        build_bt_from_segments(metadata, [], [])


@pytest.mark.parametrize("task", list(Task))
def test_complete_failure_coverage_uses_explicit_states_and_marks_terminal_stage(tmp_path, task):
    metadata, _, _ = pair_files(tmp_path, task, len(TASK_SUBTASKS[task]))
    tree = build_paired_bt(metadata, tmp_path, CompressionConfig())
    assert all(node.failure_observation_status == "observed" for node in tree.subtasks)
    assert all(node.failure_segment_end == "transitioned" for node in tree.subtasks[:-1])
    assert tree.subtasks[-1].failure_segment_end == "trace_end"
    assert all(node.divergence_status == "uncomputed" for node in tree.subtasks)
    assert PairedBehaviorTree.model_validate_json(tree.model_dump_json()) == tree
    assert build_paired_bts([metadata], tmp_path, CompressionConfig()).trees == (tree,)


@pytest.mark.parametrize("task,length", [
    (task, length) for task in Task for length in range(1, len(TASK_SUBTASKS[task]) + 1)
])
def test_single_sample_terminal_stage_aligns_once_and_supplies_valid_evidence(tmp_path, task, length):
    from telemetry_reasoning.eval.contracts import EvidenceItem
    from telemetry_reasoning.eval.metrics import EvidenceConfig, index_view, verify_evidence_item
    from telemetry_reasoning.parser import align_subtask_samples

    metadata, _, failure = pair_files(tmp_path, task, length)
    failure = failure[:(length - 1) * 9 + 1]
    write_telemetry_csv(tmp_path / metadata.failure_file, failure)
    tree = build_paired_bt(metadata, tmp_path, CompressionConfig())
    terminal = tree.subtasks[length - 1]
    aligned = align_subtask_samples(terminal)
    assert len(terminal.failure_samples) == 1
    assert aligned.alignment_mode == "common_elapsed_time"
    assert aligned.elapsed_seconds.tolist() == [0.0]
    assert aligned.failure_timestamps.tolist() == [failure[-1].timestamp]
    assert aligned.success_timestamps.tolist() == [terminal.success_samples[0].timestamp]

    evidence_index = index_view(tree.to_failure_only())
    config = EvidenceConfig()
    # Check every feature's scalar/vector contract against the evaluator's
    # independently defined evidence schema and actual input lookup.
    for feature in TelemetryFeature:
        citation = EvidenceItem(
            subtask=terminal.subtask.value, timestamp=failure[-1].timestamp,
            feature=feature.value, value=getattr(failure[-1], feature.value),
        )
        assert verify_evidence_item(citation, evidence_index, config) is None
    for absent in tree.subtasks[length:]:
        citation = EvidenceItem(
            subtask=absent.subtask.value, timestamp=failure[-1].timestamp,
            feature="gripper_force", value=0.0,
        )
        assert verify_evidence_item(citation, evidence_index, config) == "subtask_not_in_input"


@pytest.fixture
def partial_data(tmp_path):
    metadata, _, _ = pair_files(tmp_path)
    return build_paired_bt(metadata, tmp_path, CompressionConfig()).model_dump(mode="json")


@pytest.mark.parametrize("node_index", [0, -1], ids=["observed", "not_observed"])
@pytest.mark.parametrize("field", [
    "failure_observation_status", "failure_segment_end", "divergence_status",
])
def test_each_node_requires_every_observation_and_comparison_state(partial_data, node_index, field):
    partial_data["subtasks"][node_index].pop(field)
    with pytest.raises(ValidationError):
        PairedBehaviorTree.model_validate(partial_data)


@pytest.mark.parametrize("problem", [
    "missing_states", "missing_one_node_states", "missing_success",
    "empty_failure", "interior_gap", "state_contradiction", "wrong_terminal_stage",
    "missing_terminal_state", "invented_divergence", "computed_without_object",
    "object_without_computed", "noncomparable_observed", "joint_dimensions",
    "cross_stage_timestamps", "wrong_sample_subtask", "null_telemetry",
])
def test_validation_rejects_inconsistent_or_fabricated_structure(partial_data, problem):
    data = partial_data
    nodes = data["subtasks"]
    if problem in ("missing_states", "missing_one_node_states"):
        for node in nodes if problem == "missing_states" else nodes[:1]:
            for field in ("failure_observation_status", "failure_segment_end", "divergence_status"):
                node.pop(field)
    elif problem == "missing_success":
        nodes[-1]["success_samples"] = []
    elif problem in ("empty_failure", "interior_gap"):
        for node in nodes if problem == "empty_failure" else nodes[:1]:
            node.update(failure_samples=[], failure_observation_status="not_observed",
                        failure_segment_end=None, divergence_status="not_comparable")
    elif problem == "state_contradiction":
        nodes[-1]["failure_observation_status"] = "observed"
    elif problem == "wrong_terminal_stage":
        nodes[0]["failure_segment_end"] = "trace_end"
    elif problem == "missing_terminal_state":
        nodes[1]["failure_segment_end"] = None
    elif problem == "invented_divergence":
        nodes[-1].update(divergence_status="computed", deterministic_divergence={"max_abs_deltas": {}})
    elif problem == "computed_without_object":
        nodes[0]["divergence_status"] = "computed"
    elif problem == "object_without_computed":
        nodes[0]["deterministic_divergence"] = {"max_abs_deltas": {}}
    elif problem == "noncomparable_observed":
        nodes[0]["divergence_status"] = "not_comparable"
    elif problem == "joint_dimensions":
        for sample in nodes[1]["failure_samples"]:
            for field in ("joint_positions", "joint_velocities", "joint_torques"):
                sample[field] = [0.0, 0.0]
    elif problem == "cross_stage_timestamps":
        nodes[1]["failure_samples"][0]["timestamp"] = nodes[0]["failure_samples"][-1]["timestamp"]
    elif problem == "wrong_sample_subtask":
        nodes[0]["failure_samples"][0]["subtask"] = nodes[1]["subtask"]
    else:
        nodes[0]["failure_samples"][0]["gripper_force"] = None
    with pytest.raises(ValidationError):
        PairedBehaviorTree.model_validate(data)


def test_computed_observed_nodes_roundtrip_while_absent_nodes_remain_unavailable(partial_data):
    partial_data["subtasks"][0].update(
        divergence_status="computed",
        deterministic_divergence=DeterministicDivergence(
            first_divergence_timestamp=None, max_abs_deltas={"gripper_force": 0.0},
        ).model_dump(mode="json"),
    )
    tree = PairedBehaviorTree.model_validate(partial_data)
    assert PairedBehaviorTree.model_validate_json(tree.model_dump_json()) == tree
    assert tree.subtasks[0].divergence_status == "computed"
    assert tree.subtasks[-1].divergence_status == "not_comparable"


def test_views_preserve_coverage_exclude_comparisons_and_do_not_leak_labels(tmp_path):
    metadata, _, _ = pair_files(tmp_path)
    tree = build_paired_bt(metadata, tmp_path, CompressionConfig())
    before = tree.model_dump_json()
    views = [tree.to_failure_only(), tree.to_paired_without_divergence(), tree.to_full_paired()]
    for view in views:
        assert "schema_version" not in view
        assert len(view["subtasks"]) == 4
        assert view["subtasks"][-1]["failure_samples"] == []
        assert view["subtasks"][-1]["failure_observation_status"] == "not_observed"
        for forbidden in ("failed_subtask", "failure_mode", "perturbation_param",
                          "perturbation_value", "seed", "success_file", "failure_file"):
            assert forbidden not in json.dumps(view)
    for view in views[:2]:
        assert all("deterministic_divergence" not in node and "divergence_status" not in node
                   for node in view["subtasks"])
    assert all("success_samples" not in node for node in views[0]["subtasks"])
    assert views[2]["subtasks"][-1]["divergence_status"] == "not_comparable"
    views[2]["subtasks"][0]["failure_samples"][0]["joint_positions"][0] = 999.0
    assert tree.model_dump_json() == before
    # Injection labels do not establish where execution or recording stopped.
    other_labels = metadata.model_copy(update={
        "failed_subtask": TASK_SUBTASKS[metadata.task][-1], "failure_mode": "orientation_error",
    })
    other = build_paired_bt(other_labels, tmp_path, CompressionConfig())
    assert other.to_full_paired() == tree.to_full_paired()


def test_mixed_coverage_batch_serialization_reports_coverage_and_source_integrity(tmp_path):
    partial, _, _ = pair_files(tmp_path)
    complete, _, _ = pair_files(tmp_path, Task.TRANSPORT_OBJECT, 3, "complete")
    invalid, _, _ = pair_files(tmp_path, Task.TRANSPORT_OBJECT, 2, "invalid")
    write_telemetry_csv(tmp_path / invalid.failure_file, raw_run(TASK_SUBTASKS[invalid.task][1:]))
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.glob("*.csv")}
    result = parse_dataset([partial, invalid, complete], tmp_path, CompressionConfig(), tmp_path / "bt")
    assert result.report["counts"] == {"input_pairs": 3, "written_pairs": 2, "rejected_pairs": 1}
    assert "report_version" not in result.report
    assert not any(key.startswith("bt_schema") for key in result.report)
    assert result.report["failure_coverage"] == {"complete_template_pairs": 1, "partial_prefix_pairs": 1}
    assert result.report["divergence"]["node_counts"] == {"uncomputed": 5, "not_comparable": 2}
    assert [pair["pair_id"] for pair in result.report["pairs"]] == ["partial", "complete"]
    assert result.report["pairs"][0]["failure_coverage"]["termination_verified"] is False
    assert result.report["errors"][0]["pair_id"] == "invalid"
    trees = [PairedBehaviorTree.model_validate_json(path.read_text()) for path in result.files]
    assert [tree.metadata.pair_id for tree in trees] == ["partial", "complete"]
    assert all("schema_version" not in tree.model_dump() for tree in trees)
    assert all(not any(key.startswith("bt_schema") for key in pair)
               for pair in result.report["pairs"])
    # Check the predictor handoff without changing Track 4's implementation or
    # asserting scientific correctness of these synthetic diagnoses.
    from scripts.run_baseline import run
    from telemetry_reasoning.models.rule_based import load_rule_config
    predictions = run(result.report_path.parent, load_rule_config(ROOT / "configs/rules.yaml"))
    assert {record["pair_id"] for record in predictions} == {"partial", "complete"}
    assert all(record["status"] != "error" for record in predictions)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.glob("*.csv")} == sources
    before = {path.name: path.read_bytes() for path in result.report_path.parent.iterdir()}
    for path in result.report_path.parent.iterdir():
        path.unlink()
    repeated = parse_dataset([partial, invalid, complete], tmp_path, CompressionConfig(), tmp_path / "bt")
    assert {path.name: path.read_bytes() for path in repeated.report_path.parent.iterdir()} == before


def test_parser_cli_accepts_partial_failure_without_simulation_or_label_inference(tmp_path):
    metadata, _, _ = pair_files(tmp_path, Task.TRANSPORT_OBJECT, 1)
    write_metadata_csv(tmp_path / "metadata.csv", [metadata])
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/parse_dataset.py"),
        "--metadata", str(tmp_path / "metadata.csv"), "--data-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "bt"),
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads((tmp_path / "bt/pair_partial_bt.json").read_text())
    PairedBehaviorTree.model_validate(data)
    assert "schema_version" not in data
    assert data["subtasks"][1]["failure_samples"] == []


def test_parser_cli_serializes_complete_fixture_with_explicit_states(tmp_path):
    sample = ROOT / "data/sample"
    expected = (sample / "pair_000_bt.json").read_bytes()
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/parse_dataset.py"),
        "--metadata", str(sample / "metadata.csv"), "--data-dir", str(sample),
        "--output-dir", str(tmp_path / "bt"),
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    tree = PairedBehaviorTree.model_validate_json((tmp_path / "bt/pair_000_bt.json").read_text())
    assert "schema_version" not in tree.model_dump()
    assert all(node.failure_observation_status == "observed" for node in tree.subtasks)
    assert tree.subtasks[-1].failure_segment_end == "trace_end"
    report = json.loads((tmp_path / "bt/parse_report.json").read_text())
    assert "report_version" not in report
    assert not any(key.startswith("bt_schema") for key in report)
    assert report["failure_coverage"]["partial_prefix_pairs"] == 0
    assert (sample / "pair_000_bt.json").read_bytes() == expected


def test_parser_cli_rejects_removed_schema_selector(tmp_path):
    sample = ROOT / "data/sample"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/parse_dataset.py"),
        "--metadata", str(sample / "metadata.csv"), "--data-dir", str(sample),
        "--schema-version", "2.0", "--output-dir", str(tmp_path / "bt"),
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "unrecognized arguments: --schema-version 2.0" in result.stderr
    assert not (tmp_path / "bt").exists()
