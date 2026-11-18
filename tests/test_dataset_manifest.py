
"""Tests for dataset checksum verification."""

import json
from pathlib import Path

from telemetry_reasoning.eval.metrics import load_manifest
from telemetry_reasoning.parser.tuning import validate_tuning_manifest
from telemetry_reasoning.schemas import MetadataRow
from telemetry_reasoning.sim.freeze_robot_dataset import (
    sha256_file,
    task_manifests,
    verify_dataset,
)


def make_test_dataset(tmp_path: Path):
    """Create a tiny fake dataset without touching the real one."""
    batch = tmp_path / "batch"
    task_dir = batch / "pick_up_object"
    task_dir.mkdir(parents=True)

    trace = task_dir / "sample.csv"
    trace.write_text("timestamp,subtask\n0.0,approach\n", encoding="utf-8")

    relative = "pick_up_object/sample.csv"
    manifest = {
        "files": {
            relative: {
                "sha256": sha256_file(trace),
                "size_bytes": trace.stat().st_size,
            }
        }
    }
    return batch, trace, manifest


def test_unchanged_dataset_passes(tmp_path):
    batch, _, manifest = make_test_dataset(tmp_path)

    assert verify_dataset(batch, manifest) == []


def test_modified_file_detected(tmp_path):
    batch, trace, manifest = make_test_dataset(tmp_path)

    trace.write_text("timestamp,subtask\n0.0,grasp\n", encoding="utf-8")

    errors = verify_dataset(batch, manifest)

    assert any("changed" in error.lower() for error in errors)


def test_missing_file_detected(tmp_path):
    batch, trace, manifest = make_test_dataset(tmp_path)

    trace.unlink()

    errors = verify_dataset(batch, manifest)

    assert any("Missing:" in error for error in errors)


def test_unexpected_file_detected(tmp_path):
    batch, _, manifest = make_test_dataset(tmp_path)

    extra = batch / "pick_up_object" / "extra.csv"
    extra.write_text("unexpected data", encoding="utf-8")

    errors = verify_dataset(batch, manifest)

    assert any("Unexpected:" in error for error in errors)


def test_same_size_corruption_detected(tmp_path):
    batch, trace, manifest = make_test_dataset(tmp_path)
    original = trace.read_bytes()
    trace.write_bytes(original.replace(b'0.0', b'1.0'))
    assert any('Checksum changed:' in error for error in verify_dataset(batch, manifest))


def test_inventory_path_escape_rejected(tmp_path):
    batch, _, manifest = make_test_dataset(tmp_path)
    manifest['files']['../outside.csv'] = {'sha256': 'x', 'size_bytes': 0}
    assert any('Unsafe inventory path:' in error for error in verify_dataset(batch, manifest))


def test_task_manifests_load_in_tuner_and_evaluator(tmp_path):
    tasks = {"pick_up_object": "grasp", "transport_object": "maintain_grasp"}
    splits = {"train": [], "dev": []}
    records = []
    for task, split in zip(tasks, splits):
        pair = {
            "task": task, "pair_id": "000000", "initial_condition_id": "0",
            "failure_mode": "inadequate_grip_strength",
            "success_file": f"{task}/success.csv", "failure_file": f"{task}/failure.csv",
        }
        splits[split].append(pair)
        records.append({
            **pair, "injection_timestamp_s": 5.0,
            "injection_time_basis": "absolute_simulator_time",
        })
    manifests = task_manifests({
        "validation": {"pairs": records},
        "metadata_layout": {task: f"{task}/metadata.csv" for task in tasks},
        "split_seed": 723,
        "verification_scope": {},
        "splits": {"splits": splits},
    })
    for (task, subtask), split in zip(tasks.items(), splits):
        row = MetadataRow(
            task=task, pair_id="000000", success_file="success.csv", failure_file="failure.csv",
            failed_subtask=subtask, failure_mode="inadequate_grip_strength",
            perturbation_param="force", perturbation_value=0.2, seed=723,
        )
        assert validate_tuning_manifest([row], manifests[task])["000000"]["split"] == split
        path = tmp_path / f"{task}.json"
        path.write_text(json.dumps(manifests[task]), encoding="utf-8")
        loaded = load_manifest(path)
        assert loaded.splits == {"000000": split}
        assert loaded.injection_times == {}

    aligned = {
        "clock_alignment": {
            "pick_up_object": {"pairs": {"000000": {"injection_timestamp": 0.0}}},
            "transport_object": {"pairs": {"000000": {"injection_timestamp": 1.25}}},
        },
        "validation": {"pairs": records},
        "metadata_layout": {task: f"{task}/metadata.csv" for task in tasks},
        "split_seed": 723,
        "verification_scope": {},
        "splits": {"splits": splits},
    }
    for task, manifest in task_manifests(aligned).items():
        path = tmp_path / f"{task}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        expected = aligned["clock_alignment"][task]["pairs"]["000000"]["injection_timestamp"]
        assert load_manifest(path).injection_times == {"000000": expected}
