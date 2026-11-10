
"""Tests for dataset checksum verification."""

from pathlib import Path

from telemetry_reasoning.sim.freeze_robot_dataset import (
    sha256_file,
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
