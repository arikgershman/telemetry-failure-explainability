"""CPU-only publication and verification of fresh physical inference inputs."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from telemetry_reasoning import experiments as exp
from telemetry_reasoning import fresh_experiments as fresh
from telemetry_reasoning.config import load_config
from telemetry_reasoning.eval.contracts import write_predictions
from telemetry_reasoning.io import write_metadata_csv, write_telemetry_csv
from telemetry_reasoning.models.llm import load_llm_config
from telemetry_reasoning.parser.compressor import build_paired_bt
from telemetry_reasoning.parser.divergence_metrics import enrich_paired_bt, load_divergence_config
from telemetry_reasoning.schemas import MetadataRow, RawTelemetryRow, TASK_SUBTASKS, Task
from telemetry_reasoning.sim.outcomes import assess_physical_outcome


@pytest.fixture
def native(tmp_path, monkeypatch):
    """Small deterministic stored fake, never claimed as a real acceptance run."""
    root = tmp_path / "native"
    for directory in ("input", "producer", "raw-bt", "bt", "predictions"):
        (root / directory).mkdir(parents=True)
    producer_inputs = {"source_files": {"src/telemetry_reasoning/io.py": exp._fingerprint(exp.ROOT / "src/telemetry_reasoning/io.py")["sha256"]},
                       "model_files": {"configs/default.yaml": exp._fingerprint(exp.ROOT / "configs/default.yaml")["sha256"]}}
    parser_path = exp.ROOT / "configs/parser-production-v1.yaml"
    parser_config = load_config(parser_path).parser
    divergence = load_divergence_config(exp.ROOT / "configs/divergence.yaml")
    metadata, attempts, references, predictions = [], [], {}, []
    for task in Task:
        modes = sorted(fresh.RELEASED_MODES)
        # Two position-error candidates let the smoke reject an oversized first
        # candidate while keeping fully fitting task/mode coverage.
        for index, mode in enumerate(modes + ["position_error"]):
            pair_id = f"{task.value}_{index:03d}"
            subtask = TASK_SUBTASKS[task][0]
            failure = {"mode": mode, "subtask": subtask.value, "magnitude": 2.0, "delay_s": 0.04, "axis": 0}
            row = MetadataRow(pair_id=pair_id, task=task,
                success_file=f"{pair_id}_success.csv", failure_file=f"{pair_id}_failure.csv",
                failed_subtask=subtask, failure_mode=mode,
                perturbation_param="magnitude", perturbation_value=2.0, seed=723)
            metadata.append(row)
            directory = root / "producer" / pair_id
            directory.mkdir()
            rows = [RawTelemetryRow(timestamp=i * 0.02, subtask=stage,
                    joint_positions=[0.0] * 7, joint_velocities=[0.0] * 7, joint_torques=[0.0] * 7,
                    gripper_state=1.0, gripper_force=15.0, ee_pos=[0.5, 0.0, 0.5], ee_quat=[1.0, 0.0, 0.0, 0.0],
                    obj_pos=[0.5, 0.0, 0.425], obj_quat=[1.0, 0.0, 0.0, 0.0], obj_vel=[0.0] * 3, contact_force=10.0)
                    for i, stage in enumerate(stage for stage in TASK_SUBTASKS[task] for _ in range(3))]
            for kind, side in (("nominal", "success"), ("injected", "failure")):
                initial = [0.5, 0.0, 0.425]
                final = ([0.5, 0.0, 0.6] if task == Task.PICK_UP_OBJECT else [0.5, 0.15, 0.427]) if kind == "nominal" else initial
                diagnostic = {"task": task.value, "seed": 723, "initial_condition_id": index,
                    "controller_outcome": "success" if kind == "nominal" else "failed",
                    "failure_config": failure if kind == "injected" else None,
                    "telemetry_start_sim_s": 0.5,
                    "failure": {"activated": True, "activation_steps": 2, "first_activation_s": 0.54} if kind == "injected" else None,
                    "producer": {"recorded_at_generation": True, "mujoco_version": "fake_test_only", **producer_inputs},
                    "terminal_state": {"terminal_observed": True, "initial_position": initial,
                                       "final_position": final, "final_velocity": [0.0] * 3},
                    "physical_outcome": assess_physical_outcome(task, initial, final, [0.0] * 3,
                                                                source="simulator_terminal_state")}
                for repeat in range(2):
                    write_telemetry_csv(directory / f"{kind}_{repeat}.csv", rows)
                    exp._write(directory / f"{kind}_{repeat}.diagnostics.json", diagnostic)
                write_telemetry_csv(root / "input" / getattr(row, f"{side}_file"), rows)
            raw_tree = build_paired_bt(row, root / "input", parser_config)
            (root / f"raw-bt/pair_{pair_id}_bt.json").write_text(raw_tree.model_dump_json())
            tree = enrich_paired_bt(raw_tree, divergence)
            (root / f"bt/pair_{pair_id}_bt.json").write_text(tree.model_dump_json())
            attempts.append({"pair_id": pair_id, "task": task.value, "seed": 723, "initial_condition_id": index,
                "failure_config": failure, "accepted_for_paired_analysis": True,
                "pre_injection_agreement": {"status": "passed"}})
            references[pair_id] = {"split": fresh.SPLIT, "injection_timestamp": 0.54 - 0.5,
                                  "onset_source": "recorded_at_generation_canonical_origin_and_activation"}
            predictions.append({"pair_id": pair_id, "model_variant": "baseline_1", "backend": "rules",
                "predicted_failure_mode": None, "predicted_subtask": None,
                "explanation": "Explicit synthetic test fixture only", "evidence": [], "status": "abstain"})
    write_metadata_csv(root / "input/metadata.csv", metadata)
    exp._write(root / "raw-bt/parse_report.json", {
        "counts": {"input_pairs": len(metadata), "written_pairs": len(metadata), "rejected_pairs": 0},
        "compression_config": parser_config.model_dump(mode="json"),
        "provenance": {"config_file": str(parser_path), "config_sha256": exp._fingerprint(parser_path)["sha256"],
                       "metadata_sha256": exp._fingerprint(root / "input/metadata.csv")["sha256"]}})
    exp._write(root / "bt/divergence_report.json", {"status": "computed", "pair_count": len(metadata),
                                                   "divergence_config": divergence.model_dump(mode="json")})
    exp._write(root / "producer/attempts.json", attempts)
    exp._write(root / "fresh_manifest.json", {"dataset_kind": "fresh_mujoco_acceptance",
                                               "archive_certification": "not_asserted", "pairs": references})
    write_predictions(root / "predictions/baseline_1.jsonl", predictions)
    exp._write(root / "predictions/baseline_1.manifest.json", {
        "model_variant": "baseline_1", "backend": "rules", "n_inputs": len(metadata), "config": {}})
    receipt = {"sim_backend": "mujoco", "synthetic": False, "producer_acceptance": "passed",
        "status": "passed", "archive_certification": "not_asserted", "model_complete": False,
        "configuration_inputs": {str(parser_path): exp._fingerprint(parser_path)},
        "producer_inputs": producer_inputs, "checks": {"producer": {"accepted_pair_count": len(metadata)}}}
    refresh(root, receipt)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("fake tokenizer fixture only")
    config = load_llm_config(exp.ROOT / "configs/llm.yaml")
    lock = {"model_id": config.model_id, "revision": config.revision,
            "files": {"tokenizer.json": exp._fingerprint(tokenizer / "tokenizer.json")}}
    monkeypatch.setattr(exp, "load_cpu_tokenizer", lambda _: (object(), copy.deepcopy(lock)))
    monkeypatch.setattr(exp, "prompt_token_count", lambda *args: 100)
    return root, tokenizer


def refresh(root, receipt=None):
    receipt = receipt or exp._json(root / "integration_manifest.json")
    receipt["artifacts"] = exp._inventory(root, ("integration_manifest.json",))
    exp._write(root / "integration_manifest.json", receipt)


def prepare(native, tmp_path, **kwargs):
    return fresh.prepare_fresh_experiment(native_run=native[0], tokenizer_dir=native[1],
                                          output_dir=tmp_path / "bundle", **kwargs)


def test_full_bundle_preserves_all_cases_and_source_bytes(native, tmp_path):
    before = exp._inventory(native[0])
    report = prepare(native, tmp_path)
    assert report["request_count"] == 30
    assert report["pair_counts"] == {task: 5 for task in exp.TASKS}
    assert report["status"] == "ready_for_gpu"
    assert report["dataset_release_status"] == "fresh_physical_acceptance"
    assert report["archive_certification"] == "not_asserted"
    assert report["selection_policy"]["test_data_used"] is False
    assert report["native_model_complete"] is False
    assert exp.verify_experiment(tmp_path / "bundle") == report
    assert exp._inventory(native[0]) == before
    for task in exp.TASKS:
        task_dir = tmp_path / "bundle/tasks" / task
        assert len(exp._json(task_dir / "task_manifest.json")["pairs"]) == 5
        assert exp._json(task_dir / "baseline_1.manifest.json")["n_inputs"] == 5


def test_smoke_uses_context_eligibility_without_prediction_selection(native, tmp_path, monkeypatch):
    original = fresh.build_prompt
    def counted(tree, variant, **options):
        prompt = original(tree, variant, **options)
        object.__setattr__(prompt, "test_pair_id", tree.metadata.pair_id)
        return prompt
    monkeypatch.setattr(fresh, "build_prompt", counted)
    monkeypatch.setattr(exp, "prompt_token_count", lambda _, prompt: 100000 if prompt.test_pair_id.endswith("003") else 100)
    report = prepare(native, tmp_path, smoke=True)
    assert report["request_count"] == 24 and report["context_refused_requests"] == 0
    for task in exp.TASKS:
        assert f"{task}_003" in report["selection_policy"]["eligibility"][task]["candidate_pair_ids"]
        assert f"{task}_003" not in report["tasks"][task]["pair_ids"]
        assert f"{task}_004" in report["tasks"][task]["pair_ids"]


def test_full_bundle_keeps_context_refusals(native, tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "prompt_token_count", lambda _, prompt: 100000 if prompt.variant.value == "main" else 100)
    report = prepare(native, tmp_path)
    assert report["request_count"] == 30 and report["context_refused_requests"] == 10
    assert report["status"] == "ready_for_gpu_with_context_refusals"


def test_smoke_without_fitting_mode_is_refused_atomically(native, tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "prompt_token_count", lambda *args: 100000)
    with pytest.raises(ValueError, match="fully context-fitting"):
        prepare(native, tmp_path, smoke=True)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("mutation", ["changed", "added", "deleted", "symlink"])
def test_native_inventory_changes_are_refused_before_publication(native, tmp_path, mutation):
    source = native[0] / "input/metadata.csv"
    if mutation == "changed":
        source.write_text(source.read_text() + "\n")
    elif mutation == "added":
        (native[0] / "unexpected").write_text("extra")
    elif mutation == "deleted":
        source.unlink()
    else:
        (native[0] / "link").symlink_to(source)
    with pytest.raises(ValueError, match="inventory|Symlink"):
        prepare(native, tmp_path)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("mutation", ["outcome", "replay", "bt", "timing", "identity", "producer"])
def test_rehashed_but_inconsistent_native_evidence_is_refused(native, tmp_path, mutation):
    root = native[0]
    pair = "pick_up_object_000"
    if mutation in ("outcome", "identity", "producer"):
        for repeat in range(2):
            path = root / f"producer/{pair}/injected_{repeat}.diagnostics.json"
            data = exp._json(path)
            if mutation == "outcome":
                data["terminal_state"]["final_position"] = [0.5, 0.0, 0.8]
            elif mutation == "identity":
                data["seed"] = 999
            else:
                data["producer"]["recorded_at_generation"] = False
            exp._write(path, data)
    elif mutation == "replay":
        path = root / f"producer/{pair}/injected_1.csv"
        path.write_text(path.read_text().replace("15.0", "14.0", 1))
    elif mutation == "bt":
        path = root / f"bt/pair_{pair}_bt.json"
        data = exp._json(path)
        data["subtasks"][0]["failure_samples"][0]["gripper_force"] = 11.0
        exp._write(path, data)
    else:
        path = root / "fresh_manifest.json"
        data = exp._json(path)
        data["pairs"][pair]["injection_timestamp"] += 1.0
        exp._write(path, data)
    refresh(root)
    with pytest.raises(ValueError):
        prepare(native, tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_mutation_during_token_counting_never_publishes(native, tmp_path, monkeypatch):
    def mutate(*args):
        (native[0] / "fresh_manifest.json").write_text("changed source")
        return 100
    monkeypatch.setattr(exp, "prompt_token_count", mutate)
    with pytest.raises(ValueError, match="inventory|changed"):
        prepare(native, tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_additional_matplotlib_cache_is_bound_but_not_physical_evidence(native, tmp_path):
    path = native[0] / "runtime/matplotlib/fontlist-v3.11.0.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"test_cache_only":true}')
    report = prepare(native, tmp_path)
    assert report["upstream_inputs"][str(path.resolve())] == exp._fingerprint(path)
    assert path.relative_to(native[0]).as_posix() not in exp._json(native[0] / "integration_manifest.json")["artifacts"]


def test_output_inside_input_or_existing_output_is_refused(native, tmp_path):
    for output in (native[0] / "new", native[1] / "new", native[0]):
        with pytest.raises(ValueError, match="separate|absent"):
            fresh.prepare_fresh_experiment(native_run=native[0], tokenizer_dir=native[1], output_dir=output)


def test_import_and_public_cli_error_need_no_optional_runtimes(tmp_path):
    result = subprocess.run([sys.executable, "-c", '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco','torch','transformers','huggingface_hub'}:
            raise AssertionError('optional runtime imported: ' + fullname)
sys.meta_path.insert(0, Guard())
from scripts.prepare_fresh_experiment import main
assert main(['--native-run','/absent/native','--tokenizer-dir','/absent/tokenizer','--output-dir',sys.argv[1]]) == 2
''', str(tmp_path / "output")], capture_output=True, text=True,
        cwd=exp.ROOT, env={**os.environ, "PYTHONPATH": str(exp.ROOT / "src")})
    assert result.returncode == 0, result.stderr
    assert "error:" in result.stderr and "Traceback" not in result.stderr
    assert not (tmp_path / "output").exists()
