"""Production orchestration, using a small explicitly synthetic handoff."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from telemetry_reasoning.io import write_metadata_csv
from telemetry_reasoning.parser import production
from telemetry_reasoning.schemas import PairedBehaviorTree, Task
from telemetry_reasoning.sim.dataset_certification import inventory_sha256, inventory_sources
from telemetry_reasoning.sim.freeze_robot_dataset import publish_handoff
from test_bt_pipeline import write_pair

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path):
    batch = tmp_path / "batch"
    pairs, records = [], []
    for task in Task:
        directory = batch / task.value
        directory.mkdir(parents=True)
        row = write_pair(directory, "same-local-id", task)
        write_metadata_csv(directory / "metadata.csv", [row])
        pair = {"task": task.value, "pair_id": row.pair_id, "initial_condition_id": "0",
                "success_file": f"{task.value}/{row.success_file}",
                "failure_file": f"{task.value}/{row.failure_file}"}
        pairs.append(pair)
        records.append({**pair, "injection_timestamp_s": 5.0,
                        "injection_time_basis": "absolute_simulator_time"})
    files = inventory_sources(batch)
    manifest = {
        "manifest_version": 1, "split_seed": 723,
        "metadata_layout": {task.value: f"{task.value}/metadata.csv" for task in Task},
        "validation": {"pairs": records},
        "splits": {"splits": {"train": pairs, "dev": [], "test": []}},
        "files": files, "source_inventory_sha256": inventory_sha256(files),
        "verification_scope": {"stored_trace_integrity_and_parser_compatibility": "passed",
                               "deterministic_replay": "failed"},
        "release_status": "parser_ready_certification_failed",
    }
    handoff = tmp_path / "handoff"
    publish_handoff(handoff, manifest)
    return batch, handoff


def run(source, destination):
    return production.produce_dataset(*source, ROOT / "configs/parser.yaml",
                                      ROOT / "configs/divergence.yaml", destination)


def test_production_preserves_task_identity_states_receipts_and_deterministic_bytes(source, tmp_path):
    before = {p: p.read_bytes() for parent in source for p in parent.rglob("*") if p.is_file()}
    first = run(source, tmp_path / "first")
    second = run(source, tmp_path / "second")
    assert first["pair_count"] == 2 and first["sources_unchanged"]
    assert first["release_status"] == "parser_ready_certification_failed"
    assert set(first["tasks"]) == {task.value for task in Task}
    for task in Task:
        relative = f"{task.value}/enriched-bt/pair_same-local-id_bt.json"
        left, right = tmp_path / "first" / relative, tmp_path / "second" / relative
        assert left.read_bytes() == right.read_bytes()
        assert first["outputs"][relative] == second["outputs"][relative]
        tree = PairedBehaviorTree.model_validate_json(left.read_bytes())
        assert tree.metadata.task == task
        assert all(node.divergence_status == "computed" for node in tree.subtasks)
        assert first["outputs"][relative]["sha256"] == hashlib.sha256(left.read_bytes()).hexdigest()
        assert first["tasks"][task.value]["split_counts"] == {"train": 1}
    assert all(path.read_bytes() == payload for path, payload in before.items())
    published_bytes = {p.relative_to(tmp_path / "first"): p.read_bytes()
                       for p in (tmp_path / "first").rglob("*") if p.is_file()}
    shutil.rmtree(tmp_path / "first")
    run(source, tmp_path / "first")
    assert {p.relative_to(tmp_path / "first"): p.read_bytes()
            for p in (tmp_path / "first").rglob("*") if p.is_file()} == published_bytes
    assert not list(tmp_path.glob(".production-bt-*"))


def test_changed_source_or_handoff_refuses_publication(source, tmp_path):
    batch, handoff = source
    trace = next((batch / Task.PICK_UP_OBJECT.value).glob("success*.csv"))
    trace.write_bytes(trace.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source verification"):
        run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    (handoff / "tasks/pick_up_object.json").write_text("{}")
    with pytest.raises(ValueError, match="handoff"):
        run(source, tmp_path / "output")


def test_midrun_mutation_fails_atomically(source, tmp_path, monkeypatch):
    original = production.enrich_paired_bt
    def changing(tree, config):
        (source[0] / tree.metadata.task.value / tree.metadata.success_file).write_text("corrupted")
        return original(tree, config)
    monkeypatch.setattr(production, "enrich_paired_bt", changing)
    with pytest.raises(ValueError):
        run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".production-bt-*"))


def test_existing_and_overlapping_destinations_are_protected(source, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    for output in (existing, source[0] / "new", source[1] / "new", tmp_path):
        with pytest.raises(ValueError):
            run(source, output)
    assert not (source[0] / "new").exists()


def test_cli_reports_errors_without_traceback(source, tmp_path):
    command = [sys.executable, str(ROOT / "scripts/produce_bts.py"),
               "--batch", str(source[0]), "--handoff", str(source[1]),
               "--config", str(ROOT / "configs/parser.yaml"),
               "--output-dir", str(tmp_path / "output")]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    again = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert again.returncode == 2
    assert "must be absent" in again.stderr and "Traceback" not in again.stderr


def test_production_rejects_unbound_or_test_tuned_selection(source, tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection_version": 1, "test_data_used_for_selection": True}))
    with pytest.raises(ValueError, match="selection receipt"):
        production.produce_dataset(*source, ROOT / "configs/parser.yaml", ROOT / "configs/divergence.yaml",
                                   tmp_path / "output", selection_path=selection)
    assert not (tmp_path / "output").exists()
