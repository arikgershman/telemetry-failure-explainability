"""Replay trust boundaries using schema-valid synthetic traces, without MuJoCo."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from telemetry_reasoning.config import load_config
from telemetry_reasoning.io import write_metadata_csv, write_telemetry_csv
from telemetry_reasoning.schemas import MetadataRow, Subtask, Task
from telemetry_reasoning.sim.replay import compare_injection_diagnostics, compare_telemetry, recover_dataset_evidence
from telemetry_reasoning.sim.runner import EpisodeResult
from scripts.generate_mock_pair import make_mock_rows

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_and_unique_activation_counts_have_explicit_comparison_bases():
    legacy = {"activated": True, "first_activation_s": 0.5, "activation_steps": 10}
    current = {**legacy, "activation_steps": 5, "activation_context_calls": 10,
               "activation_steps_basis": "unique_active_simulator_timestamps"}
    comparison = compare_injection_diagnostics(legacy, current)
    assert comparison == {"matched": True, "activation_count_comparison_basis": "archived_unversioned_context_calls"}
    assert legacy == {"activated": True, "first_activation_s": 0.5, "activation_steps": 10}
    assert compare_injection_diagnostics(current, current)["matched"] is True
    bad = {**current, "activation_steps_basis": "unknown_future_basis"}
    assert compare_injection_diagnostics(legacy, bad)["matched"] is False
    assert compare_injection_diagnostics(bad, current)["matched"] is False
    assert compare_injection_diagnostics(current, legacy)["matched"] is False
    assert not compare_injection_diagnostics(legacy, {**current, "activation_context_calls": 11})["matched"]


@pytest.fixture
def replay_batch(tmp_path):
    pickup_rows = make_mock_rows(load_config(ROOT / "configs/default.yaml"), 19)
    batch, cases = tmp_path / "batch", {}
    for task in Task:
        folder = batch / task.value
        folder.mkdir(parents=True)
        traces = deepcopy(pickup_rows)
        if task is Task.TRANSPORT_OBJECT:
            mapping = {Subtask.APPROACH: Subtask.MAINTAIN_GRASP,
                       Subtask.ALIGN: Subtask.MOVE_ALONG_PATH, Subtask.GRASP: Subtask.REACH_TARGET}
            traces = tuple([row.model_copy(update={"subtask": mapping[row.subtask]})
                            for row in rows if row.subtask in mapping] for rows in traces)
        meta = MetadataRow(pair_id="000", task=task, success_file="success.csv", failure_file="failure.csv",
                           failed_subtask="grasp" if task is Task.PICK_UP_OBJECT else "reach_target",
                           failure_mode="inadequate_grip_strength", perturbation_param="force",
                           perturbation_value=100.0, seed=19)
        write_metadata_csv(folder / "metadata.csv", [meta])
        diagnostic = {"pair_id": "000", "initial_condition_id": 0,
                      "failure_config": {"mode": "inadequate_grip_strength", "subtask": meta.failed_subtask.value,
                                         "magnitude": 100.0, "delay_s": 0.0, "axis": 0}}
        for kind, rows, name, origin in (("nominal", traces[0], "success.csv", 0.5),
                                         ("injected", traces[1], "failure.csv", 0.7)):
            write_telemetry_csv(folder / name, rows)
            data = {"task": task.value, "seed": 19, "initial_condition_id": 0,
                    "initial_offset_xy": [0, 0], "duration_s": rows[-1].timestamp + origin + 0.002,
                    "controller_outcome": "success" if kind == "nominal" else "failed"}
            data["failure"] = (None if kind == "nominal" else
                               {"activated": True, "first_activation_s": origin + 4.4, "activation_steps": 100})
            diagnostic[kind] = deepcopy(data)  # Archived records have no clock origin.
            data.update(telemetry_start_sim_s=origin,
                        terminal_state={"terminal_observed": True, "initial_position": rows[0].obj_pos,
                                        "final_position": rows[-1].obj_pos, "final_velocity": rows[-1].obj_vel},
                        producer={"mujoco_version": "synthetic_test_double"})
            cases[task.value, kind] = EpisodeResult(rows, kind == "nominal", data)
        (folder / "diagnostics.jsonl").write_text(json.dumps(diagnostic) + "\n")
    return batch, cases


def fake_runner(cases, calls):
    def run(*, task, seed, initial_condition_id, failure, spread):
        kind = "injected" if failure is not None else "nominal"
        calls.append((task.value, kind))
        return deepcopy(cases[task.value, kind])
    return run


def test_each_run_clock_requires_its_own_complete_repeated_replay(replay_batch, tmp_path):
    batch, cases = replay_batch
    calls = []
    originals = {str(path): path.read_bytes() for path in batch.rglob("*") if path.is_file()}
    report = recover_dataset_evidence(batch, tmp_path / "replay", episode_runner=fake_runner(cases, calls))
    assert report["status"] == "matched" and len(calls) == 8
    assert report["summary"]["matched_run_records"] == 4
    for pair in report["pairs"]:
        assert pair["nominal"]["origin"] == {"timestamp_s": 0.5, "basis": "verified_exact_replay", "verified": True}
        assert pair["injected"]["origin"] == {"timestamp_s": 0.7, "basis": "verified_exact_replay", "verified": True}
        assert pair["injected"]["runtime_provenance"]["mujoco_version"] == "synthetic_test_double"
        assert pair["injected"]["replay"]["fresh_repeatability_matched"] is True
    assert report["producer_provenance"]["recorded_at_generation"] is False
    assert {name: Path(name).read_bytes() for name in originals} == originals


def test_injected_mismatch_cannot_inherit_a_matched_nominal_clock(replay_batch, tmp_path):
    batch, cases = replay_batch
    cases["pick_up_object", "injected"].rows[-1].gripper_force += 0.1
    report = recover_dataset_evidence(batch, tmp_path / "replay", episode_runner=fake_runner(cases, []))
    pair = report["pairs"][0]
    assert report["status"] == "mismatched"
    assert pair["nominal"]["origin"]["verified"] is True
    assert pair["injected"]["origin"]["verified"] is False
    assert pair["injected"]["origin"]["basis"] == "unverified_replay_candidate"
    assert "gripper_force" in pair["injected"]["replay"]["mismatched_fields"]


def test_identity_and_terminal_duration_must_also_match(replay_batch, tmp_path):
    batch, cases = replay_batch
    cases["pick_up_object", "nominal"].diagnostics["duration_s"] += 0.02
    report = recover_dataset_evidence(batch, tmp_path / "replay", episode_runner=fake_runner(cases, []))
    run = report["pairs"][0]["nominal"]
    assert run["replay"]["duration_matched"] is False
    assert run["origin"]["verified"] is False


def test_changed_activation_diagnostics_cannot_certify_clocks(replay_batch, tmp_path):
    batch, cases = replay_batch
    cases["pick_up_object", "injected"].diagnostics["failure"]["first_activation_s"] += 0.02
    report = recover_dataset_evidence(batch, tmp_path / "replay", episode_runner=fake_runner(cases, []))
    run = report["pairs"][0]["injected"]
    assert run["replay"]["injection_diagnostics_matched"] is False
    assert run["origin"]["verified"] is False


def test_cached_equal_csvs_cannot_hide_different_archived_duration(replay_batch, tmp_path):
    from shutil import copyfile
    from telemetry_reasoning.io import read_metadata_csv

    batch, cases = replay_batch
    task = batch / "pick_up_object"
    original = read_metadata_csv(task / "metadata.csv")[0]
    repeated = original.model_copy(update={"pair_id": "001", "success_file": "success2.csv", "failure_file": "failure2.csv"})
    write_metadata_csv(task / "metadata.csv", [original, repeated])
    copyfile(task / "success.csv", task / "success2.csv")
    copyfile(task / "failure.csv", task / "failure2.csv")
    first = json.loads((task / "diagnostics.jsonl").read_text())
    later = deepcopy(first)
    later["pair_id"] = "001"
    later["nominal"]["duration_s"] += 0.02
    (task / "diagnostics.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(later) + "\n")
    report = recover_dataset_evidence(batch, tmp_path / "replay", episode_runner=fake_runner(cases, []))
    first, later = report["pairs"][:2]
    assert first["nominal"]["origin"]["verified"] is True
    assert later["nominal"]["origin"]["verified"] is False
    assert later["nominal"]["replay"]["duration_matched"] is False


def test_source_mutation_invalidates_receipt(replay_batch, tmp_path):
    batch, cases = replay_batch
    wrapped = fake_runner(cases, [])
    def run(**kwargs):
        result = wrapped(**kwargs)
        (batch / "pick_up_object/metadata.csv").write_text("mutated")
        return result
    with pytest.raises(ValueError, match="sources changed"):
        recover_dataset_evidence(batch, tmp_path / "replay", episode_runner=run)
    report = json.loads((tmp_path / "replay/evidence.json").read_text())
    assert report["status"] == "error"


def test_refuses_source_directory_outputs_and_single_replay(replay_batch, tmp_path):
    batch, cases = replay_batch
    with pytest.raises(ValueError, match="outside"):
        recover_dataset_evidence(batch, batch / "evidence", episode_runner=fake_runner(cases, []))
    with pytest.raises(ValueError, match="two repeat"):
        recover_dataset_evidence(batch, tmp_path / "replay", repeats=1)
    assert not (batch / "evidence").exists() and not (tmp_path / "replay").exists()


def test_numeric_comparison_is_absolute_and_checks_all_features(replay_batch):
    _, cases = replay_batch
    rows = cases["pick_up_object", "nominal"].rows
    near = deepcopy(rows)
    near[-1].joint_torques[0] += 1e-9
    assert compare_telemetry(rows, near)["matched"]
    near[-1].joint_torques[0] += 1e-4
    comparison = compare_telemetry(rows, near)
    assert not comparison["matched"] and "joint_torques" in comparison["mismatched_fields"]
    assert not compare_telemetry(rows, near[:-1])["all_rows"]
