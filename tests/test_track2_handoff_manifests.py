"""Tests against Track 2's REAL handoff manifests, read from the released batch under outputs/dataset/.

    <batch>/handoff/pick_up_object.json, <batch>/handoff/transport_object.json   per-task manifests
    <batch>/manifest.json                                                         batch-level SHA-256 inventory and validation record

where <batch> is outputs/dataset/batch_20261030_152125_e5ca72de. The batch is not committed; when the
manifests are absent every test in this module is skipped, with the missing path named in the report.

These files carry real pair IDs, splits, failure modes, initial conditions and injection times. They do
NOT carry the failed stage (that lives only in the batch's metadata.csv / diagnostics.jsonl) and they do
NOT carry the evaluator's CSV-clock ``injection_timestamp``. Both gaps are asserted here, because the
evaluator and the real smokes must not paper over them.

Where a test needs per-pair labels the manifests lack (the failed stage for ``evaluate.py``), they are
placeholders and the test says so; only IDs, splits, modes and conditions come from the real files.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import math
from pathlib import Path

import pytest

from telemetry_reasoning.eval.contracts import ContractError, TASK_SUBTASKS, write_predictions
from telemetry_reasoning.eval.metrics import load_manifest
from telemetry_reasoning.io import write_metadata_csv
from telemetry_reasoning.schemas import FailureMode, MetadataRow, Subtask, Task

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs" / "dataset" / "batch_20261030_152125_e5ca72de"
HANDOFF = BATCH / "handoff"
BATCH_MANIFEST = BATCH / "manifest.json"
TASKS = ("pick_up_object", "transport_object")
_MISSING = [str(path) for path in (*(HANDOFF / f"{task}.json" for task in TASKS), BATCH_MANIFEST) if not path.is_file()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="released Track 2 manifests not found: " + ", ".join(_MISSING))
MODES = ("position_error", "orientation_error", "inadequate_grip_strength", "early_gripper_release")


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load_script("integration_smoke")


@pytest.fixture(scope="module")
def manifests():
    return {task: json.loads((HANDOFF / f"{task}.json").read_text()) for task in TASKS}


@pytest.fixture(scope="module")
def batch():
    return json.loads((BATCH_MANIFEST).read_text())


# --------------------------------------------------------------------------- #
# What the handoff document promises
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_the_smoke_loader_accepts_the_real_task_manifest(task):
    manifest = smoke.load_task_manifest(HANDOFF / f"{task}.json", task)
    assert manifest["dataset_kind"] == "mujoco" and manifest["task"] == task
    assert manifest["metadata"] == f"{task}/metadata.csv" and manifest["split_seed"] == 723
    assert sorted(manifest["pairs"]) == [f"{i:06d}" for i in range(500)]


@pytest.mark.parametrize("task", TASKS)
def test_each_task_has_350_50_100_pairs_in_35_5_10_disjoint_initial_conditions(manifests, task):
    pairs = manifests[task]["pairs"].values()
    assert collections.Counter(p["split"] for p in pairs) == {"train": 350, "dev": 50, "test": 100}
    conditions = {s: {p["initial_condition_id"] for p in pairs if p["split"] == s} for s in ("train", "dev", "test")}
    assert {s: len(c) for s, c in conditions.items()} == {"train": 35, "dev": 5, "test": 10}
    assert not (conditions["train"] & conditions["dev"] or conditions["train"] & conditions["test"] or conditions["dev"] & conditions["test"])
    # grouped split: a pair's split is a pure function of its initial condition
    assert all(len({p["split"] for p in pairs if p["initial_condition_id"] == c}) == 1 for c in {p["initial_condition_id"] for p in pairs})


@pytest.mark.parametrize("task", TASKS)
def test_every_split_contains_all_four_released_modes_and_never_excessive_speed(manifests, task):
    pairs = list(manifests[task]["pairs"].values())
    for split in ("train", "dev", "test"):
        assert {p["failure_mode"] for p in pairs if p["split"] == split} == set(MODES)
    assert collections.Counter(p["failure_mode"] for p in pairs) == {m: 125 for m in MODES}
    assert "excessive_speed" not in {p["failure_mode"] for p in pairs}


def test_pair_ids_overlap_across_tasks_so_identity_is_task_and_pair_id(manifests, batch):
    shared = set(manifests["pick_up_object"]["pairs"]) & set(manifests["transport_object"]["pairs"])
    assert len(shared) == 500
    assert batch["pair_identity"].startswith("(task, pair_id)") and "task-local" in batch["pair_identity"]


@pytest.mark.parametrize("task", TASKS)
def test_pair_files_follow_the_released_layout(manifests, task):
    for pair_id, pair in manifests[task]["pairs"].items():
        assert pair["success_file"] == f"{task}/success/{pair_id}.csv"
        assert pair["failure_file"] == f"{task}/failure/{pair_id}.csv"


# --------------------------------------------------------------------------- #
# The two gaps the smokes must respect
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_injection_times_are_absolute_simulator_time_with_no_evaluator_clock(manifests, task):
    pairs = manifests[task]["pairs"].values()
    assert {p["injection_time_basis"] for p in pairs} == {"absolute_simulator_time"}
    assert all(isinstance(p["injection_timestamp_s"], float) and math.isfinite(p["injection_timestamp_s"]) and p["injection_timestamp_s"] >= 0 for p in pairs)
    # generated without the clock sidecars: no CSV-clock timestamp, so no onset reference exists
    assert not any("injection_timestamp" in p for p in pairs)
    assert smoke.has_evaluator_onset_clock(manifests[task]) is False


@pytest.mark.parametrize("task", TASKS)
def test_the_failed_stage_is_in_neither_manifest(manifests, batch, task):
    assert not any(k in p for p in manifests[task]["pairs"].values() for k in ("failed_subtask", "subtask", "failure_config"))
    entries = [e for split in batch["splits"]["splits"].values() for e in split if e["task"] == task]
    assert not any(k in e for e in entries for k in ("failed_subtask", "subtask", "failure_config"))


def test_a_manifest_that_does_add_the_evaluator_clock_is_recognized(manifests):
    marked = json.loads(json.dumps(manifests["pick_up_object"]))
    marked["pairs"]["000001"]["injection_timestamp"] = 1.25
    assert smoke.has_evaluator_onset_clock(marked) is True


# --------------------------------------------------------------------------- #
# The three files agree with each other
# --------------------------------------------------------------------------- #
def test_task_manifests_match_the_batch_manifest_splits_and_validation(manifests, batch):
    batch_split = {(e["task"], e["pair_id"]): (split, e) for split, entries in batch["splits"]["splits"].items() for e in entries}
    validated = {(p["task"], p["pair_id"]): p for p in batch["validation"]["pairs"]}
    assert len(batch_split) == len(validated) == 1000
    for task, manifest in manifests.items():
        for pair_id, pair in manifest["pairs"].items():
            split, entry = batch_split[(task, pair_id)]
            assert (pair["split"], pair["failure_mode"], str(pair["initial_condition_id"])) == (split, entry["failure_mode"], str(entry["initial_condition_id"]))
            assert (pair["success_file"], pair["failure_file"]) == (entry["success_file"], entry["failure_file"])
            check = validated[(task, pair_id)]
            assert check["accepted"] is True and not check["issues"]
            assert check["injection_timestamp_s"] == pair["injection_timestamp_s"]
            assert check["injection_time_basis"] == pair["injection_time_basis"]


def test_batch_manifest_records_a_clean_validation_and_the_excluded_mode(batch):
    validation = batch["validation"]
    assert (validation["accepted_pairs"], validation["quarantined_pairs"], validation["fatal_errors"]) == (1000, 0, [])
    assert set(batch["excluded_failure_modes"]) == {"excessive_speed"}
    assert batch["hash_algorithm"] == "sha256" and batch["split_seed"] == 723 and batch["splits"]["split_unit"] == "initial_condition_id"
    assert batch["verification_scope"]["independent_physical_outcomes"] == "not_certified_by_this_manifest"


def test_batch_inventory_lists_every_file_the_smoke_reads(manifests, batch):
    files = batch["files"]
    for task, manifest in manifests.items():
        assert batch["metadata_layout"][task] == manifest["metadata"] and manifest["metadata"] in files
        assert f"{task}/diagnostics.jsonl" in files  # where the failed stage and injection config live
        for pair in manifest["pairs"].values():
            assert pair["success_file"] in files and pair["failure_file"] in files
    assert all(set(entry) == {"sha256", "size_bytes"} and len(entry["sha256"]) == 64 for entry in files.values())


# --------------------------------------------------------------------------- #
# The evaluator against real IDs, splits and modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_evaluator_manifest_loader_reads_real_splits_but_finds_no_onset_reference(manifests, task):
    loaded = load_manifest(HANDOFF / f"{task}.json")
    assert collections.Counter(loaded.splits.values()) == {"train": 350, "dev": 50, "test": 100}
    assert dict(loaded.injection_times) == {}  # injection_timestamp_s is deliberately NOT read as an onset


def test_evaluator_manifest_loader_refuses_the_batch_level_manifest():
    with pytest.raises(ContractError, match="expected a 'pairs' mapping"):
        load_manifest(BATCH_MANIFEST)
    with pytest.raises(smoke.SmokeError, match="batch-level manifest"):
        smoke.load_task_manifest(BATCH_MANIFEST, "pick_up_object")


def _write_truth_and_predictions(tmp_path, task, manifest):
    """Metadata and predictions for all 500 REAL pair IDs. Failed stages are placeholders (not in the manifest)."""
    stage = Subtask(TASK_SUBTASKS[task][0])
    rows, records = [], []
    for pair_id, pair in manifest["pairs"].items():
        rows.append(MetadataRow(pair_id=pair_id, task=Task(task), success_file=f"success/{pair_id}.csv", failure_file=f"failure/{pair_id}.csv",
                                failed_subtask=stage, failure_mode=FailureMode(pair["failure_mode"]),
                                perturbation_param="placeholder", perturbation_value=1.0, seed=723))
        records.append({"pair_id": pair_id, "model_variant": "baseline_1", "predicted_failure_mode": pair["failure_mode"],
                        "predicted_subtask": stage.value, "explanation": "placeholder", "evidence": [], "status": "ok", "backend": "rules"})
    write_metadata_csv(tmp_path / "metadata.csv", rows)
    write_predictions(tmp_path / "predictions.jsonl", records)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("split, expected", [("dev", 50), ("test", 100), ("train", 350)])
def test_evaluate_script_scopes_scoring_to_the_real_split_and_reports_no_onset(tmp_path, manifests, task, split, expected):
    _write_truth_and_predictions(tmp_path, task, manifests[task])
    out = tmp_path / "eval"
    code = load_script("evaluate").main([
        "--metadata", str(tmp_path / "metadata.csv"), "--predictions", str(tmp_path / "predictions.jsonl"),
        "--manifest", str(HANDOFF / f"{task}.json"), "--split", split, "--output-dir", str(out)])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    entry = metrics["variants"]["baseline_1"]
    assert metrics["split"] == split and entry["slices"]["all"]["n"] == expected
    assert entry["slices"]["all"]["failure_mode_accuracy"] == 1.0
    assert (entry["onset"].get("n_with_reference") or 0) == 0  # no CSV-clock timestamps in the real manifests
