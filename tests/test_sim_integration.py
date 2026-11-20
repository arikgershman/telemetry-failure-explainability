"""Offline regressions for the real-producer acceptance orchestrator."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import integration_smoke
from telemetry_reasoning import integration
from telemetry_reasoning.io import write_telemetry_csv
from telemetry_reasoning.schemas import RawTelemetryRow, TASK_SUBTASKS, Task
from telemetry_reasoning.sim.outcomes import assess_physical_outcome
from telemetry_reasoning import sim_integration as smoke


def fake_producer_inventory():
    return {kind: {name: hashlib.sha256((integration.ROOT / name).read_bytes()).hexdigest()}
            for kind, name in (("source_files", "src/telemetry_reasoning/schemas.py"),
                               ("model_files", "configs/default.yaml"))}


@pytest.fixture(autouse=True)
def offline_producer_dependencies(monkeypatch):
    monkeypatch.setattr(smoke, "_producer_inventory", fake_producer_inventory)


def candidate_cases():
    return [{"pair_id": f"{task.value}_{index:03d}", "task": task.value,
             "seed": 723, "initial_condition_id": index,
             "failure_config": {"mode": mode, "subtask": TASK_SUBTASKS[task][0].value,
                                "magnitude": 2.0, "delay_s": 0.04, "axis": 0},
             "input_origin": "unit_test_fake_only"}
            for task in Task for index, mode in enumerate(sorted(smoke.RELEASED_MODES))]


def fake_runner(*, task, seed, initial_condition_id, failure, output):
    initial = [0.5, 0.0, 0.425]
    succeeded = failure is None or failure.mode.value == "excessive_speed"
    final = ([0.5, 0, 0.6] if task == "pick_up_object" else [0.5, 0.15, 0.427]) if succeeded else initial
    rows = []
    for subtask in TASK_SUBTASKS[Task(task)]:
        for _ in range(3):
            rows.append(RawTelemetryRow(timestamp=len(rows) * 0.02, subtask=subtask,
                                        joint_positions=[0.0] * 7, joint_velocities=[0.0] * 7,
                                        joint_torques=[0.0] * 7, gripper_state=1.0, gripper_force=15.0,
                                        ee_pos=[0.5, 0.0, 0.5], ee_quat=[1.0, 0.0, 0.0, 0.0],
                                        obj_pos=initial, obj_quat=[1.0, 0.0, 0.0, 0.0], obj_vel=[0.0] * 3,
                                        contact_force=10.0))
    write_telemetry_csv(output, rows)
    physical = assess_physical_outcome(task, initial, final, [0.0] * 3,
                                       source="simulator_terminal_state")
    inventory = fake_producer_inventory()
    producer = {"recorded_at_generation": True, "mujoco_version": "fake_test_only", **inventory}
    for kind, summary in (("source_files", "source_sha256"), ("model_files", "model_sha256")):
        producer[summary] = hashlib.sha256("\n".join(f"{name}:{value}" for name, value in sorted(inventory[kind].items())).encode()).hexdigest()
    diagnostics = {"task": task, "seed": seed, "initial_condition_id": initial_condition_id,
                   "duration_s": 1.0, "controller_outcome": "success" if succeeded else "failed",
                   "complete_subtasks": True,
                   "failure_config": ({"mode": failure.mode.value, "subtask": failure.subtask.value,
                                       "magnitude": failure.magnitude, "delay_s": failure.delay_s,
                                       "axis": failure.axis} if failure else None),
                   "telemetry_start_sim_s": 0.5, "observed_subtasks": [s.value for s in TASK_SUBTASKS[Task(task)]],
                   "physical_outcome": physical,
                   "terminal_state": {"initial_position": initial, "final_position": final,
                                      "final_velocity": [0.0] * 3, "terminal_observed": True,
                                      "simulator_time_s": 1.0},
                   "producer": producer,
                   "failure": ({"activated": True, "activation_steps": 2,
                                "first_activation_s": 0.5 + failure.delay_s} if failure else None)}
    return SimpleNamespace(rows=rows, task_succeeded=succeeded, diagnostics=diagnostics)


def write_catalogue(batch):
    catalogue = {"catalogue_version": 1, "tasks": {}}
    for task in Task:
        directory = batch / task.value
        directory.mkdir(parents=True)
        cases = [case for case in candidate_cases() if case["task"] == task.value]
        records = [{"pair_id": case["pair_id"], "nominal": {"seed": case["seed"]},
                    "initial_condition_id": case["initial_condition_id"], "failure_config": case["failure_config"]}
                   for case in cases]
        diagnostics = directory / "diagnostics.jsonl"
        diagnostics.write_text("".join(json.dumps(record) + "\n" for record in records))
        catalogue["tasks"][task.value] = {
            "diagnostics_sha256": hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
            "configurations": [{"representative_pair_id": case["pair_id"], "failure_config": case["failure_config"]}
                               for case in cases]}
    (batch / "provenance").mkdir()
    (batch / "provenance/recovered_failure_configs.json").write_text(json.dumps(catalogue))


def test_import_is_pure_without_optional_physics_or_model_runtime(tmp_path):
    result = subprocess.run([sys.executable, "-c", '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'torch', 'transformers'}:
            raise AssertionError('optional native runtime imported: ' + fullname)
sys.meta_path.insert(0, Guard())
import telemetry_reasoning.sim_integration
'''], env={**os.environ, "PYTHONPATH": str(integration.ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_catalogue_preserves_input_identity_and_checks_source_hash(tmp_path):
    write_catalogue(tmp_path)
    candidates, sources = smoke.released_candidates(tmp_path)
    assert len(candidates) == 8 and len(sources) == 3
    assert {case["seed"] for case in candidates} == {723}
    diagnostics = tmp_path / "pick_up_object/diagnostics.jsonl"
    diagnostics.write_text(diagnostics.read_text() + "\n")
    with pytest.raises(integration.SmokeFailure, match="checksum"):
        smoke.released_candidates(tmp_path)


def test_catalogue_cannot_omit_an_accepted_configuration_with_the_same_mode(tmp_path):
    write_catalogue(tmp_path)
    path = tmp_path / "pick_up_object/diagnostics.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    record["pair_id"] = "additional"
    record["failure_config"]["magnitude"] = 3.0
    path.write_text(path.read_text() + json.dumps(record) + "\n")
    catalogue_path = tmp_path / "provenance/recovered_failure_configs.json"
    catalogue = json.loads(catalogue_path.read_text())
    catalogue["tasks"]["pick_up_object"]["diagnostics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    catalogue_path.write_text(json.dumps(catalogue))
    with pytest.raises(integration.SmokeFailure, match="every unique"):
        smoke.released_candidates(tmp_path)


def test_recorded_producer_hashes_must_match_actual_dependency_bytes(tmp_path):
    def wrong_hash(**kwargs):
        result = fake_runner(**kwargs)
        result.diagnostics["producer"]["source_files"]["src/telemetry_reasoning/schemas.py"] = "0" * 64
        return result
    with pytest.raises(integration.SmokeFailure, match="producer source/model inventory"):
        smoke.generate_fresh_pairs(candidate_cases(), tmp_path, episode_runner=wrong_hash)


def test_fresh_generation_requires_physical_failure_and_keeps_speed_as_probe(tmp_path):
    result = smoke.generate_fresh_pairs(candidate_cases(), tmp_path, episode_runner=fake_runner)
    assert result["accepted_pair_count"] == 8
    assert result["repeat_executions"] == 36
    assert result["speed_probe_count"] == 2
    assert all(set(modes) == smoke.RELEASED_MODES for modes in result["coverage"].values())
    assert result["pre_injection_status_counts"] == {"passed": 8, "not_observed": 0}
    probes = json.loads((tmp_path / "producer/speed_probes.json").read_text())
    assert all(probe["physical_status"] == "succeeded" and not probe["required_physical_failure"] for probe in probes)
    metadata = integration.read_metadata_csv(tmp_path / "input/metadata.csv")
    assert len(metadata) == 8 and all(row.failure_mode.value != "excessive_speed" for row in metadata)
    assert (tmp_path / "fresh_manifest.json").is_file()


def test_claimed_outcome_cannot_replace_observed_terminal_goal(tmp_path):
    def dishonest(**kwargs):
        result = fake_runner(**kwargs)
        result.diagnostics["physical_outcome"]["status"] = "failed"
        return result
    with pytest.raises(integration.SmokeFailure, match="recomputed terminal"):
        smoke.generate_fresh_pairs(candidate_cases(), tmp_path, episode_runner=dishonest)


def test_controller_failure_alone_does_not_create_a_failed_pair(tmp_path):
    def controller_only_failure(**kwargs):
        result = fake_runner(**kwargs)
        if kwargs["failure"]:
            state = result.diagnostics["terminal_state"]
            state["final_position"] = [0.5, 0, 0.6] if kwargs["task"] == "pick_up_object" else [0.5, 0.15, 0.427]
            result.diagnostics["physical_outcome"] = assess_physical_outcome(
                kwargs["task"], state["initial_position"], state["final_position"], state["final_velocity"],
                source="simulator_terminal_state")
        return result
    with pytest.raises(integration.SmokeFailure, match="coverage is incomplete"):
        smoke.generate_fresh_pairs(candidate_cases(), tmp_path, episode_runner=controller_only_failure)
    attempts = json.loads((tmp_path / "producer/attempts.json").read_text())
    assert len(attempts) == 8 and all(not attempt["accepted_for_paired_analysis"] for attempt in attempts)
    assert not (tmp_path / "input/metadata.csv").exists()


def test_repeated_execution_checks_every_observation(tmp_path):
    def drift(**kwargs):
        result = fake_runner(**kwargs)
        if str(kwargs["output"]).endswith("nominal_1.csv"):
            result.rows[-1].joint_torques[0] = 0.01
            write_telemetry_csv(kwargs["output"], result.rows)
        return result
    with pytest.raises(integration.SmokeFailure, match="did not repeat"):
        smoke.generate_fresh_pairs(candidate_cases(), tmp_path, episode_runner=drift)


def test_pre_injection_checks_all_prior_rows_and_zero_onset_is_explicit(tmp_path):
    case = candidate_cases()[0]
    nominal = fake_runner(task=case["task"], seed=723, initial_condition_id=0, failure=None, output=tmp_path / "nominal.csv")
    diagnostics = deepcopy(nominal.diagnostics)
    diagnostics["failure"] = {"activated": True, "activation_steps": 1, "first_activation_s": 0.5}
    none = smoke.pre_injection_agreement(nominal.rows, nominal.rows, nominal.diagnostics, diagnostics)
    assert none["status"] == "not_observed" and none["row_count"] == 0
    diagnostics["failure"]["first_activation_s"] = 0.56
    changed = deepcopy(nominal.rows)
    changed[1].gripper_force += 0.01
    with pytest.raises(integration.SmokeFailure, match="pre-injection"):
        smoke.pre_injection_agreement(nominal.rows, changed, nominal.diagnostics, diagnostics)


def test_setup_refuses_archive_output_and_populated_destination(tmp_path):
    batch = tmp_path / "batch"
    write_catalogue(batch)
    with pytest.raises(ValueError, match="outside"):
        smoke.run_mujoco_cpu_smoke(batch / "run", batch=batch, episode_runner=fake_runner)
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.txt").write_text("preserve")
    with pytest.raises(ValueError, match="fresh"):
        smoke.run_mujoco_cpu_smoke(output, batch=batch, episode_runner=fake_runner)
    assert (output / "existing.txt").read_text() == "preserve"


def test_producer_failure_retains_honest_failed_receipt(tmp_path):
    batch = tmp_path / "batch"
    write_catalogue(batch)
    def fail(**kwargs):
        raise RuntimeError("injected simulator failure")
    output = tmp_path / "run"
    with pytest.raises(integration.SmokeFailure, match="simulator failure"):
        smoke.run_mujoco_cpu_smoke(output, batch=batch, episode_runner=fail)
    receipt = json.loads((output / integration.MANIFEST_NAME).read_text())
    assert receipt["status"] == "failed" and receipt["archive_certification"] == "not_asserted"
    assert receipt["stages"][0]["status"] == "failed"
    assert (output / "producer/pick_up_object_000/nominal_0.stdout.log").is_file()


def test_interrupted_producer_cannot_leave_a_running_receipt(tmp_path):
    batch = tmp_path / "batch"
    write_catalogue(batch)
    def interrupt(**kwargs):
        raise KeyboardInterrupt()
    output = tmp_path / "run"
    with pytest.raises(integration.SmokeFailure):
        smoke.run_mujoco_cpu_smoke(output, batch=batch, episode_runner=interrupt)
    receipt = json.loads((output / integration.MANIFEST_NAME).read_text())
    assert receipt["status"] == "failed" and receipt["error"]["type"] == "KeyboardInterrupt"
    assert receipt["stages"][0]["status"] == "failed"


def test_public_cli_selects_real_producer_explicitly(tmp_path, monkeypatch, capsys):
    seen = {}
    def run(output_dir, **kwargs):
        seen.update(output_dir=output_dir, **kwargs)
        return {"output_dir": str(output_dir), "checks": {"prediction_count": 32, "plot_files": ["plot.png"]}}
    monkeypatch.setattr(smoke, "run_mujoco_cpu_smoke", run)
    assert integration_smoke.main(["--sim-backend", "mujoco", "--batch", str(tmp_path / "batch"),
                                   "--output-dir", str(tmp_path / "run")]) == 0
    assert seen["batch"] == tmp_path / "batch"
    assert "real physics, synthetic stub" in capsys.readouterr().out


def refusal_fixture():
    record = SimpleNamespace(backend="stub", evidence=[], status="error",
                             explanation=f"Inference rejected (ValueError): {smoke.CONTEXT_REFUSAL}")
    request = {"status": "error", "input_tokens": 1000, "output_tokens": None,
               "error_type": "ValueError", "error_message": smoke.CONTEXT_REFUSAL,
               "token_count_method": "pinned_tokenizer_cpu"}
    config = {"max_new_tokens": 100, "context_window_tokens": 1000}
    return record, request, config


def test_exact_proven_context_refusal_is_an_explicit_negative_case():
    assert smoke._validate_stub_outcome(*refusal_fixture()) is True


@pytest.mark.parametrize("problem", ["fits_budget", "byte_estimate", "runtime_error", "evidence", "wrong_status"])
def test_context_refusal_cannot_hide_other_errors_or_generated_evidence(problem):
    record, request, config = refusal_fixture()
    if problem == "fits_budget":
        request["input_tokens"] = 100
    elif problem == "byte_estimate":
        request["token_count_method"] = "utf8_bytes_upper_bound"
    elif problem == "runtime_error":
        request["error_type"] = "RuntimeError"
    elif problem == "evidence":
        record.evidence = ["fabricated"]
    else:
        request["status"] = "abstain"
    with pytest.raises(integration.SmokeFailure):
        smoke._validate_stub_outcome(record, request, config)


def test_stub_abstention_still_requires_context_fit():
    record, request, config = refusal_fixture()
    record.status = request["status"] = "abstain"
    with pytest.raises(integration.SmokeFailure, match="oversized"):
        smoke._validate_stub_outcome(record, request, config)


@pytest.mark.parametrize("arguments", [["--batch", "unused"], ["--parser-config", "unused"],
                                      ["--stub-tokenizer-dir", "unused"],
                                      ["--sim-backend", "mujoco", "--seed", "7"]])
def test_cli_rejects_ignored_input_arguments(tmp_path, arguments):
    assert integration_smoke.main(["--output-dir", str(tmp_path / "run"), *arguments]) == 2
    assert not (tmp_path / "run").exists()
