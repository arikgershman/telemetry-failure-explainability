"""Offline preference-formatting contracts; synthetic labels prove no quality."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from telemetry_reasoning.models.prompts import build_prompt
from telemetry_reasoning.models.training_data import main, prepare_training_data, verify_training_data
from telemetry_reasoning.schemas import (
    SUBTASK_METADATA, TASK_GOALS, TASK_SUBTASKS, DeterministicDivergence,
    MetadataRow, PairedBehaviorTree, RawTelemetryRow, SubtaskBT, Task,
    TelemetryFeature,
)
from telemetry_reasoning.sim.dataset_certification import inventory_sha256


ROOT = Path(__file__).resolve().parents[1]


def tree_for(task, pair_id="same-local-id"):
    nodes = []
    for i, subtask in enumerate(TASK_SUBTASKS[task]):
        row = RawTelemetryRow(
            timestamp=float(i * 2), subtask=subtask, joint_positions=[0.0] * 7,
            joint_velocities=[0.1] * 7, joint_torques=[0.2] * 7,
            gripper_state=1.0, gripper_force=10.0, contact_force=9.0,
            ee_pos=[0.1, 0.2, 0.3], ee_quat=[1.0, 0.0, 0.0, 0.0],
            obj_pos=[0.1, 0.2, 0.25], obj_quat=[1.0, 0.0, 0.0, 0.0],
            obj_vel=[0.0, 0.0, 0.0],
        )
        nodes.append(SubtaskBT(
            subtask=subtask, metadata=SUBTASK_METADATA[subtask], success_samples=[row],
            failure_samples=[row.model_copy(deep=True)], failure_observation_status="observed",
            failure_segment_end="trace_end" if i == len(TASK_SUBTASKS[task]) - 1 else "transitioned",
            divergence_status="computed", deterministic_divergence=DeterministicDivergence(
                max_abs_deltas={feature: 0.0 for feature in TelemetryFeature}),
        ))
    return PairedBehaviorTree(
        metadata=MetadataRow(pair_id=pair_id, task=task, success_file=f"success/{pair_id}.csv",
                             failure_file=f"failure/{pair_id}.csv", failed_subtask=TASK_SUBTASKS[task][0],
                             failure_mode="position_error", perturbation_param="hidden", perturbation_value=1,
                             seed=723), overall_goal=TASK_GOALS[task], subtasks=nodes,
    )


def response(explanation, *, evidence=None, mode=None, subtask=None):
    return json.dumps({"predicted_failure_mode": mode, "predicted_subtask": subtask,
                       "explanation": explanation, "evidence": evidence or [],
                       "status": "ok" if mode else "abstain"})


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "bt"
    records, pairs, files = [], [], {}
    trees = {}
    for task in Task:
        tree = tree_for(task)
        trees[task.value] = tree
        path = root / task.value / "enriched-bt" / f"pair_{tree.metadata.pair_id}_bt.json"
        path.parent.mkdir(parents=True)
        path.write_text(tree.model_dump_json(indent=2) + "\n")
        pair = {"task": task.value, "pair_id": tree.metadata.pair_id, "initial_condition_id": "7",
                "failure_mode": tree.metadata.failure_mode.value,
                "success_file": f"{task.value}/{tree.metadata.success_file}",
                "failure_file": f"{task.value}/{tree.metadata.failure_file}"}
        pairs.append(pair)
        for key in ("success_file", "failure_file"):
            files[pair[key]] = {"sha256": "0" * 64, "size_bytes": 0}
        for variant in ("baseline_2", "baseline_3", "main"):
            prompt = build_prompt(tree, variant)
            records.append({"task": task.value, "pair_id": tree.metadata.pair_id,
                            "model_variant": variant,
                            "chosen": response("Synthetic formatter choice; no supported diagnosis."),
                            "rejected": response("Synthetic alternative; uncertainty remains."),
                            "preference_provenance": {
                                "kind": "synthetic_demo", "generator_id": "test-suite",
                                "annotation_id": f"synthetic-{task.value}-{variant}",
                                "rationale": "Formatter test only, no annotation-quality claim.",
                                "bt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "prompt_sha256": prompt.prompt_sha256}})
    manifest = {"manifest_version": 2, "hash_algorithm": "sha256", "files": files,
                "source_inventory_sha256": inventory_sha256(files),
                "release_status": "parser_ready_certification_failed", "certification": {"status": "failed"},
                "splits": {"split_unit": "initial_condition_id", "splits": {"train": pairs, "dev": [], "test": []},
                           "condition_assignments": {task.value: {"train": ["7"], "dev": [], "test": []}
                                                     for task in Task}}}
    prefs = tmp_path / "preferences.jsonl"
    manifest_path = tmp_path / "dataset_manifest.json"
    output = tmp_path / "formatted"

    def write():
        prefs.write_text("".join(json.dumps(record) + "\n" for record in records))
        manifest_path.write_text(json.dumps(manifest))

    write()
    return {"root": root, "records": records, "manifest": manifest, "prefs": prefs,
            "manifest_path": manifest_path, "output": output, "write": write, "trees": trees}


def run(case, **kwargs):
    return prepare_training_data(case["root"], case["prefs"], case["manifest_path"],
                                 case["output"], synthetic_demo=True, **kwargs)


def readlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_all_variants_both_tasks_exact_inference_prompts_and_hashes(case):
    before = {p: p.read_bytes() for p in case["root"].rglob("*.json")}
    receipt = run(case)
    assert receipt["rows"] == {"train": 6, "dev": 0}
    assert receipt["training_acceptance"] == "synthetic_demo_only"
    assert receipt["dataset_release_status"] == "parser_ready_certification_failed"
    assert receipt["input_files_unchanged"]
    rows = readlines(case["output"] / "dpo/train.jsonl")
    sidecar = readlines(case["output"] / "records.jsonl")
    assert len({(r["task"], r["pair_id"], r["model_variant"]) for r in sidecar}) == 6
    for row, metadata in zip(rows, sidecar):
        prompt = build_prompt(case["trees"][metadata["task"]], metadata["model_variant"])
        assert row["prompt"] == prompt.messages
        assert metadata["prompt_sha256"] == prompt.prompt_sha256
        assert set(row) == {"prompt", "chosen", "rejected"}
        assert row["chosen"][0]["role"] == "assistant"
        assert "same-local-id" not in json.dumps(row)
        assert "hidden" not in json.dumps(row)
    sft = readlines(case["output"] / "sft/train.jsonl")
    assert sft[0]["messages"] == rows[0]["prompt"] + rows[0]["chosen"]
    for name, info in receipt["artifacts"].items():
        raw = (case["output"] / name).read_bytes()
        assert len(raw) == info["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == info["sha256"]
    assert before == {p: p.read_bytes() for p in case["root"].rglob("*.json")}


def test_reviewed_default_requires_named_timestamped_bound_annotations(case):
    with pytest.raises(ValueError, match="human_reviewed"):
        prepare_training_data(case["root"], case["prefs"], case["manifest_path"], case["output"])
    for record in case["records"]:
        provenance = record["preference_provenance"]
        provenance.update(kind="human_reviewed", reviewer_id="reviewer-example", reviewed_at="2026-11-18T10:00:00-05:00")
        del provenance["generator_id"]
    case["write"]()
    receipt = prepare_training_data(case["root"], case["prefs"], case["manifest_path"], case["output"])
    assert receipt["training_acceptance"] == "reviewed_annotations_supplied"
    assert "not independently certified" in receipt["review_verification"]


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c["records"][0].update(chosen="   "), "invalid preference"),
    (lambda c: c["records"][0].update(chosen=c["records"][0]["rejected"]), "differ"),
    (lambda c: c["records"][0].update(chosen="{}"), "validation"),
    (lambda c: c["records"][0].update(chosen="NaN"), "non-finite"),
    (lambda c: c["records"][0].update(pair_id="unknown"), "unknown task-local"),
    (lambda c: c["records"][0].update(model_variant="baseline_1"), "invalid preference"),
    (lambda c: c["records"].append(copy.deepcopy(c["records"][0])), "duplicate preference"),
    (lambda c: c["records"][0]["preference_provenance"].update(bt_sha256="f" * 64), "hashes"),
    (lambda c: c["records"][0]["preference_provenance"].update(prompt_sha256="f" * 64), "hashes"),
    (lambda c: c["records"][0]["preference_provenance"].update(reviewer_id="fake-human"), "forbids reviewer"),
    (lambda c: c["records"][1]["preference_provenance"].update(annotation_id=c["records"][0]["preference_provenance"]["annotation_id"]), "duplicate annotation"),
    (lambda c: c["manifest"].update(source_inventory_sha256="f" * 64), "inventory digest"),
])
def test_bad_annotations_fail_before_output_creation(case, mutation, match):
    mutation(case)
    case["write"]()
    with pytest.raises(ValueError, match=match):
        run(case)
    assert not case["output"].exists()


def test_numeric_response_reformatting_is_not_a_preference(case):
    value = json.loads(response("Same observation", mode="position_error", subtask="approach",
                                evidence=[{"subtask": "approach", "timestamp": 0.0,
                                           "feature": "contact_force", "value": 9.0}]))
    case["records"][0]["chosen"] = json.dumps(value)
    value["evidence"][0]["timestamp"] = 0
    value["evidence"][0]["value"] = 9
    case["records"][0]["rejected"] = json.dumps(value)
    case["write"]()
    with pytest.raises(ValueError, match="differ"):
        run(case)


def test_whitespace_reformatting_is_not_a_preference(case):
    case["records"][0]["rejected"] = json.dumps(json.loads(case["records"][0]["chosen"]), indent=4)
    case["write"]()
    with pytest.raises(ValueError, match="differ"):
        run(case)


@pytest.mark.parametrize("label", ["chosen", "rejected"])
def test_both_choices_require_failure_side_evidence(case, label):
    record = case["records"][0]
    record[label] = response("Unsupported fabricated observation", mode="position_error", subtask="approach",
                             evidence=[{"subtask": "approach", "timestamp": 0.0,
                                        "feature": "gripper_force", "value": 9999.0}])
    case["write"]()
    with pytest.raises(ValueError, match="evidence must match"):
        run(case)


def test_ok_grounded_response_accepted_without_using_ground_truth_as_a_label(case):
    case["records"][0]["chosen"] = response(
        "Observed contact force; synthetic causal claim is not human reviewed.", mode="early_gripper_release", subtask="approach",
        evidence=[{"subtask": "approach", "timestamp": 0.0, "feature": "contact_force", "value": 9.0}])
    case["write"]()
    run(case)


@pytest.mark.parametrize("field", ["pair_id", "model_variant", "backend"])
def test_completion_cannot_embed_foreign_identity(case, field):
    value = json.loads(case["records"][0]["chosen"])
    value[field] = "wrong"
    case["records"][0]["chosen"] = json.dumps(value)
    case["write"]()
    with pytest.raises(ValueError, match="Extra inputs"):
        run(case)


def test_test_split_is_prohibited_and_dev_stays_separate(case):
    grouping = case["manifest"]["splits"]
    pair = grouping["splits"]["train"].pop()
    grouping["splits"]["test"].append(pair)
    grouping["condition_assignments"][pair["task"]] = {"train": [], "dev": [], "test": ["7"]}
    case["write"]()
    with pytest.raises(ValueError, match="held-out test"):
        run(case)
    grouping["splits"]["dev"] = grouping["splits"].pop("test")
    grouping["splits"]["test"] = []
    grouping["condition_assignments"][pair["task"]] = {"train": [], "dev": ["7"], "test": []}
    case["write"]()
    assert run(case)["rows"] == {"train": 3, "dev": 3}


def test_condition_leakage_rejected_even_when_leaking_pair_has_no_preference(case):
    grouping = case["manifest"]["splits"]
    duplicate_condition = dict(grouping["splits"]["train"][0], pair_id="unannotated")
    grouping["splits"]["dev"].append(duplicate_condition)
    case["write"]()
    with pytest.raises(ValueError, match="crosses splits"):
        run(case)


def test_manifest_assignment_corruption_rejected(case):
    case["manifest"]["splits"]["condition_assignments"]["pick_up_object"]["train"] = ["8"]
    case["write"]()
    with pytest.raises(ValueError, match="assignments disagree"):
        run(case)


def test_main_rejects_uncomputed_divergence(case):
    task = Task.PICK_UP_OBJECT.value
    tree = case["trees"][task].model_copy(deep=True)
    tree.subtasks[0].divergence_status = "uncomputed"
    tree.subtasks[0].deterministic_divergence = None
    path = case["root"] / task / "enriched-bt" / "pair_same-local-id_bt.json"
    path.write_text(tree.model_dump_json())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for record in case["records"]:
        if record["task"] == task:
            record["preference_provenance"]["bt_sha256"] = digest
    case["write"]()
    with pytest.raises(ValueError, match="computed, nonnull"):
        run(case)


def test_source_mismatch_missing_or_aliased_bt_refused(case, tmp_path):
    path = case["root"] / "pick_up_object/enriched-bt/pair_same-local-id_bt.json"
    original = path.read_bytes()
    tree = json.loads(original)
    tree["metadata"]["success_file"] = "success/wrong.csv"
    path.write_text(json.dumps(tree))
    with pytest.raises(ValueError, match="source filenames"):
        run(case)
    path.unlink()
    with pytest.raises(ValueError, match="regular, unaliased"):
        run(case)
    target = tmp_path / "alias-target.json"
    target.write_bytes(original)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        run(case)


def test_existing_output_never_modified(case):
    case["output"].mkdir()
    marker = case["output"] / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="fresh directory"):
        run(case)
    assert marker.read_text() == "preserve"


def test_publication_verification_checks_bytes_inputs_and_extra_files(case):
    run(case)
    assert verify_training_data(case["output"])["rows"]["train"] == 6
    case["prefs"].write_text("changed")
    with pytest.raises(ValueError, match="source input changed"):
        verify_training_data(case["output"])
    assert verify_training_data(case["output"], check_inputs=False)["training_acceptance"] == "synthetic_demo_only"
    extra = case["output"] / "unexpected.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="differ from publication"):
        verify_training_data(case["output"], check_inputs=False)
    extra.unlink()
    (case["output"] / "dpo/train.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_training_data(case["output"], check_inputs=False)


def test_verifier_refuses_mode_forgery(case):
    run(case)
    receipt_path = case["output"] / "training_manifest.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["training_acceptance"] = "reviewed_annotations_supplied"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="inconsistent"):
        verify_training_data(case["output"])


def test_write_failure_cleans_stage_and_publishes_nothing(case, monkeypatch):
    original = Path.write_bytes

    def fail(path, data):
        if path.name == "records.jsonl":
            raise OSError("deliberate publication failure")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError, match="publication failure"):
        run(case)
    assert not case["output"].exists()
    assert not list(case["output"].parent.glob(".formatted-*"))


def test_concurrent_input_change_blocks_publication(case, monkeypatch):
    original = Path.write_bytes

    def mutate(path, data):
        result = original(path, data)
        if path.name == "records.jsonl":
            case["prefs"].write_text("changed\n")
        return result

    monkeypatch.setattr(Path, "write_bytes", mutate)
    with pytest.raises(ValueError, match="inputs changed"):
        run(case)
    assert not case["output"].exists()


def test_cli_clean_error(case, capsys):
    argv = ["--bt-dir", str(case["root"]), "--preferences", str(case["prefs"]),
            "--manifest", str(case["manifest_path"]), "--output-dir", str(case["output"])]
    assert main(argv) == 2
    output = capsys.readouterr()
    assert "training-data error:" in output.err and "Traceback" not in output.err
    assert not case["output"].exists()
    assert main([*argv, "--synthetic-demo"]) == 0


def test_module_import_and_execution_need_no_model_runtime_or_network(case):
    code = r'''
import importlib.abc, sys, socket
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','huggingface_hub','peft','trl','mujoco'}:
            raise AssertionError('forbidden optional runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
def blocked(*args, **kwargs):
    raise AssertionError('network forbidden')
socket.socket = blocked
from telemetry_reasoning.models.training_data import main
raise SystemExit(main(sys.argv[1:]))
'''
    result = subprocess.run([sys.executable, "-c", code, "--bt-dir", str(case["root"]),
                             "--preferences", str(case["prefs"]), "--manifest", str(case["manifest_path"]),
                             "--output-dir", str(case["output"]), "--synthetic-demo"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
