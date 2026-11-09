"""Synthetic batch/transport contracts; no physics or model dependencies."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from telemetry_reasoning.config import CompressionConfig, load_config
from telemetry_reasoning.io import read_metadata_csv, write_telemetry_csv
from telemetry_reasoning.parser import build_paired_bt, build_paired_bts, parse_dataset
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


def test_serialized_batch_integrates_both_tasks_and_reports_rejections(tmp_path):
    rows = [write_pair(tmp_path, "pickup", Task.PICK_UP_OBJECT),
            write_pair(tmp_path, "bad"), write_pair(tmp_path, "transport")]
    (tmp_path / rows[1].failure_file).unlink()
    source_hashes = {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")}
    result = parse_dataset(rows, tmp_path, CompressionConfig(), tmp_path / "bt")
    assert [path.name for path in result.files] == ["pair_pickup_bt.json", "pair_transport_bt.json"]
    report = json.loads(result.report_path.read_text())
    assert report == result.report
    assert report["counts"] == {"input_pairs": 3, "written_pairs": 2, "rejected_pairs": 1}
    assert report["errors"][0]["pair_id"] == "bad"
    assert report["divergence"]["status"] == "not_computed"
    assert report["summary"]["pair_count"] == 2
    for path, detail in zip(result.files, report["pairs"]):
        tree = PairedBehaviorTree.model_validate_json(path.read_text())
        assert path.name == f"pair_{tree.metadata.pair_id}_bt.json"
        assert tree == build_paired_bt(tree.metadata, tmp_path, CompressionConfig())
        assert all(node.deterministic_divergence is None for node in tree.subtasks)
        assert detail["output_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert detail["diagnostics"]["totals"]["raw_rows"] == 10 * len(tree.subtasks)
        for side in ("success", "failure"):
            source = tmp_path / getattr(tree.metadata, f"{side}_file")
            assert detail["sources"][side]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.csv")} == source_hashes


def test_serialization_preserves_sample_bytes_and_event_neighborhood(tmp_path):
    sample = ROOT / "data/sample"
    config = load_config(ROOT / "configs/parser.yaml")
    assert config == load_config(ROOT / "configs/default.yaml")
    rows = read_metadata_csv(sample / "metadata.csv")
    result = parse_dataset(rows, sample, config.parser, tmp_path / "bt")
    path, = result.files
    assert path.read_bytes() == (sample / path.name).read_bytes()
    node = PairedBehaviorTree.model_validate_json(path.read_text()).subtasks[2]
    assert {5.08, 5.10, 5.12} <= {row.timestamp for row in node.failure_samples}
    assert result.report["summary"]["totals"]["retained_rows"] == 65
    # The report has no runtime fields; deterministic reruns in the same path
    # after removing only our own output reproduce every byte.
    before = {p.name: p.read_bytes() for p in result.report_path.parent.iterdir()}
    for p in result.report_path.parent.iterdir():
        p.unlink()
    repeated = parse_dataset(rows, sample, config.parser, tmp_path / "bt")
    assert {p.name: p.read_bytes() for p in repeated.report_path.parent.iterdir()} == before


def test_serialization_preflight_protects_existing_outputs_and_rejects_unsafe_ids(tmp_path):
    row = write_pair(tmp_path, "good")
    output = tmp_path / "bt"
    output.mkdir()
    stale = output / "pair_stale_bt.json"
    stale.write_text("keep this")
    with pytest.raises(ValueError, match="absent or empty"):
        parse_dataset([row], tmp_path, CompressionConfig(), output)
    assert stale.read_text() == "keep this"
    for pair_id in ("../escape", "a/b", "a\\b", "line\nbreak"):
        unsafe = row.model_copy(update={"pair_id": pair_id})
        with pytest.raises(ValueError, match="flat BT filename"):
            parse_dataset([unsafe], tmp_path, CompressionConfig(), tmp_path / "new")
    with pytest.raises(ValueError, match="must be unique"):
        parse_dataset([row, row], tmp_path, CompressionConfig(), tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_all_rejected_and_empty_serialization_have_explicit_reports(tmp_path):
    row = write_pair(tmp_path, "bad")
    (tmp_path / row.success_file).unlink()
    for name, rows, rejected in (("bad", [row], 1), ("empty", [], 0)):
        result = parse_dataset(rows, tmp_path, CompressionConfig(), tmp_path / name)
        assert result.files == ()
        assert result.report["counts"]["rejected_pairs"] == rejected
        assert result.report["summary"]["pair_count"] == 0
        assert list(result.report_path.parent.iterdir()) == [result.report_path]


def test_output_failure_never_publishes_partial_batch(tmp_path, monkeypatch):
    row = write_pair(tmp_path, "good")
    original = Path.write_text

    def fail_report(path, *args, **kwargs):
        if path.name == "parse_report.json":
            raise OSError("synthetic storage failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_report)
    with pytest.raises(OSError, match="storage failure"):
        parse_dataset([row], tmp_path, CompressionConfig(), tmp_path / "bt")
    assert not (tmp_path / "bt").exists()
    assert not list(tmp_path.glob(".parse-*"))


def test_parse_cli_exit_codes_provenance_and_downstream_loading(tmp_path):
    sample = ROOT / "data/sample"
    command = [sys.executable, str(ROOT / "scripts/parse_dataset.py"),
               "--metadata", str(sample / "metadata.csv"), "--data-dir", str(sample),
               "--config", str(ROOT / "configs/parser.yaml"), "--output-dir", str(tmp_path / "bt")]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    report = json.loads((tmp_path / "bt/parse_report.json").read_text())
    assert report["provenance"]["metadata_sha256"] == hashlib.sha256((sample / "metadata.csv").read_bytes()).hexdigest()
    assert report["provenance"]["dataset_validation"] == "not_asserted_by_parser"
    # Real Track 4 consumer: report files must not look like BT inputs.
    from scripts.run_baseline import run
    from telemetry_reasoning.models.rule_based import load_rule_config

    predictions = run(tmp_path / "bt", load_rule_config(ROOT / "configs/rules.yaml"))
    assert len(predictions) == 1 and predictions[0]["pair_id"] == "000"
    assert predictions[0]["status"] == "ok"
    repeated = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert repeated.returncode == 2 and "fresh directory" in repeated.stderr


def test_parse_cli_partial_rejection_and_malformed_metadata(tmp_path):
    from telemetry_reasoning.io import write_metadata_csv

    good, bad = write_pair(tmp_path, "good"), write_pair(tmp_path, "bad")
    (tmp_path / bad.failure_file).unlink()
    metadata_file = tmp_path / "metadata.csv"
    write_metadata_csv(metadata_file, [good, bad])
    command = [sys.executable, str(ROOT / "scripts/parse_dataset.py"),
               "--metadata", str(metadata_file), "--data-dir", str(tmp_path),
               "--output-dir", str(tmp_path / "bt")]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert process.returncode == 1, process.stderr
    assert (tmp_path / "bt/pair_good_bt.json").exists()
    metadata_file.write_text("bad,header\n1,2\n")
    process = subprocess.run(command[:-1] + [str(tmp_path / "bad-output")],
                             cwd=ROOT, text=True, capture_output=True)
    assert process.returncode == 2
    assert not (tmp_path / "bad-output").exists()
