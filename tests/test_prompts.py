"""Track 5 prompt contracts; synthetic cases are not model-performance evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from telemetry_reasoning.models.prompts import (
    DEFAULT_TEMPLATE_DIR, ModelVariant, SYSTEM_PROMPT, TEMPLATE_FILENAMES,
    build_prompt,
)
from telemetry_reasoning.schemas import (
    SUBTASK_METADATA, TASK_GOALS, TASK_SUBTASKS, DeterministicDivergence,
    FailureMode, MetadataRow, PairedBehaviorTree, RawTelemetryRow, SubtaskBT,
    Task, TelemetryFeature,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/sample/pair_000_bt.json"
STORAGE_KEYS = {
    "pair_id", "failed_subtask", "failure_mode", "perturbation_param",
    "perturbation_value", "success_file", "failure_file", "seed",
}


@pytest.fixture
def sample():
    return PairedBehaviorTree.model_validate_json(SAMPLE.read_text())


def recursive_keys(value):
    if isinstance(value, dict):
        yield from value
        for child in value.values():
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def computed(tree):
    """Synthetic complete comparison objects on a copy, not computed metrics."""
    tree = tree.model_copy(deep=True)
    for node in tree.subtasks:
        if node.failure_samples:
            node.divergence_status = "computed"
            node.deterministic_divergence = DeterministicDivergence(
                first_divergence_timestamp=None,
                max_abs_deltas={feature: 0.0 for feature in TelemetryFeature},
            )
    return PairedBehaviorTree.model_validate_json(tree.model_dump_json())


def synthetic_tree(task, prefix_length):
    """Seven joints and uneven retention; failure ends inside its last stage."""
    stages = TASK_SUBTASKS[task]

    def row(stage, t):
        return RawTelemetryRow(
            timestamp=t, subtask=stage, joint_positions=[0.0] * 7,
            joint_velocities=[0.1] * 7, joint_torques=[0.2] * 7,
            gripper_state=1.0, gripper_force=10.0, contact_force=9.0,
            ee_pos=[0.1, 0.2, 0.3], ee_quat=[1.0, 0.0, 0.0, 0.0],
            obj_pos=[0.1, 0.2, 0.25], obj_quat=[1.0, 0.0, 0.0, 0.0],
            obj_vel=[0.0, 0.0, 0.0],
        )

    nodes = []
    for index, stage in enumerate(stages):
        observed = index < prefix_length
        nodes.append(SubtaskBT(
            subtask=stage, metadata=SUBTASK_METADATA[stage].model_copy(deep=True),
            success_samples=[row(stage, 2.0 * index), row(stage, 2.0 * index + 1.98)],
            failure_samples=[row(stage, 2.0 * index + 0.04)] if observed else [],
            failure_observation_status="observed" if observed else "not_observed",
            failure_segment_end=("trace_end" if index == prefix_length - 1 else "transitioned") if observed else None,
            divergence_status="uncomputed" if observed else "not_comparable",
        ))
    return PairedBehaviorTree(
        metadata=MetadataRow(
            pair_id="join-only-synthetic", task=task,
            success_file="hidden-success.csv", failure_file="hidden-failure.csv",
            failed_subtask=stages[0], failure_mode=FailureMode.POSITION_ERROR,
            perturbation_param="hidden-perturbation", perturbation_value=12345.0,
            seed=728391,
        ),
        overall_goal=TASK_GOALS[task], subtasks=nodes,
    )


@pytest.mark.parametrize("variant", list(ModelVariant))
def test_variants_use_the_canonical_view_and_exactly_the_allowed_information(sample, variant):
    tree = computed(sample)
    method = {
        ModelVariant.BASELINE_2: tree.to_failure_only,
        ModelVariant.BASELINE_3: tree.to_paired_without_divergence,
        ModelVariant.MAIN: tree.to_full_paired,
    }[variant]
    expected = method()
    expected.pop("pair_id")
    prompt = build_prompt(tree, variant)
    assert prompt.variant == variant.value
    assert prompt.payload == expected
    assert json.loads(prompt.user.split("\n\n", 1)[1]) == expected
    keys = set(recursive_keys(prompt.payload))
    assert not keys.intersection(STORAGE_KEYS)
    assert ("success_samples" in keys) == (variant is not ModelVariant.BASELINE_2)
    for key in ("divergence_status", "deterministic_divergence", "max_abs_deltas"):
        assert (key in keys) == (variant is ModelVariant.MAIN)
    assert "pair_id" not in prompt.system + prompt.user


def test_instruction_and_output_format_are_fixed_across_variants(sample):
    tree = computed(sample)
    prompts = [build_prompt(tree, variant) for variant in ModelVariant]
    assert {prompt.system for prompt in prompts} == {SYSTEM_PROMPT}
    assert len({prompt.user.split("\n\n", 1)[0] for prompt in prompts}) == 1
    assert len({prompt.template_sha256 for prompt in prompts}) == 1
    for prompt in prompts:
        for field in ("predicted_failure_mode", "predicted_subtask", "explanation", "evidence", "status"):
            assert field in prompt.system
        for mode in FailureMode:
            assert mode.value in prompt.system
        for feature in TelemetryFeature:
            assert feature.value in prompt.system
        assert "private chain-of-thought" in prompt.system
        assert "actual supplied failure_samples row" in prompt.system


def test_known_fixture_values_and_synthetic_grasp_divergence_are_preserved(sample):
    tree = computed(sample)
    grasp = tree.subtasks[2]
    grasp.deterministic_divergence.first_divergence_timestamp = 5.1
    grasp.deterministic_divergence.max_abs_deltas[TelemetryFeature.GRIPPER_FORCE] = 12.0
    prompt = build_prompt(tree, "main")
    node = prompt.payload["subtasks"][2]
    retained = {row["timestamp"]: row["gripper_force"] for row in node["failure_samples"]}
    assert retained[5.08] == 15.0 and retained[5.1] == 3.0
    assert node["deterministic_divergence"]["first_divergence_timestamp"] == 5.1
    assert node["deterministic_divergence"]["max_abs_deltas"]["gripper_force"] == 12.0
    assert prompt.payload["subtasks"][0]["deterministic_divergence"]["first_divergence_timestamp"] is None
    assert set(prompt.payload["subtasks"][0]["deterministic_divergence"]["max_abs_deltas"].values()) == {0.0}


@pytest.mark.parametrize("task,length", [
    (task, length) for task in Task for length in range(1, len(TASK_SUBTASKS[task]) + 1)
])
@pytest.mark.parametrize("variant", list(ModelVariant))
def test_every_task_and_failure_prefix_preserves_coverage_without_invented_rows(task, length, variant):
    tree = computed(synthetic_tree(task, length))
    before = tree.model_dump_json()
    prompt = build_prompt(tree, variant)
    assert prompt.payload["task"] == task.value
    assert len(prompt.payload["subtasks"]) == len(TASK_SUBTASKS[task])
    for index, node in enumerate(prompt.payload["subtasks"]):
        assert len(node["failure_samples"]) == (1 if index < length else 0)
        assert node["failure_observation_status"] == ("observed" if index < length else "not_observed")
        if index < length:
            assert len(node["failure_samples"][0]["joint_positions"]) == 7
        elif variant is ModelVariant.MAIN:
            assert node["deterministic_divergence"] is None
            assert node["divergence_status"] == "not_comparable"
    assert prompt.payload["subtasks"][length - 1]["failure_segment_end"] == "trace_end"
    assert tree.model_dump_json() == before


def test_main_rejects_uncomputed_sample_without_filling_missing_metrics(sample):
    before = sample.model_dump_json()
    with pytest.raises(ValueError, match="computed, nonnull"):
        build_prompt(sample, "main")
    assert sample.model_dump_json() == before
    assert build_prompt(sample, "baseline_2") and build_prompt(sample, "baseline_3")


@pytest.mark.parametrize("problem", ["missing_object", "uncomputed_status", "empty_deltas", "missing_feature", "invented_absent"])
def test_main_rejects_incomplete_or_inconsistent_comparisons(problem):
    tree = computed(synthetic_tree(Task.TRANSPORT_OBJECT, 1))
    node = tree.subtasks[0]
    if problem == "missing_object":
        node.deterministic_divergence = None
    elif problem == "uncomputed_status":
        node.divergence_status = "uncomputed"
    elif problem == "empty_deltas":
        node.deterministic_divergence.max_abs_deltas.clear()
    elif problem == "missing_feature":
        del node.deterministic_divergence.max_abs_deltas[TelemetryFeature.OBJ_VEL]
    else:
        tree.subtasks[-1].deterministic_divergence = copy.deepcopy(node.deterministic_divergence)
    with pytest.raises(ValueError, match="main requires"):
        build_prompt(tree, "main")


@pytest.mark.parametrize("variant", list(ModelVariant))
def test_all_storage_labels_identifiers_and_paths_leave_the_prompt_unchanged(sample, variant):
    original = computed(sample)
    other = original.model_copy(deep=True)
    other.metadata = original.metadata.model_copy(update={
        "pair_id": "do-not-send-opaque-join", "failed_subtask": "approach",
        "failure_mode": "orientation_error", "perturbation_param": "do-not-send-parameter",
        "perturbation_value": 9876543.0, "seed": 9876543,
        "success_file": "do-not-send-success.csv", "failure_file": "do-not-send-failure.csv",
    })
    assert build_prompt(original, variant) == build_prompt(other, variant)
    text = build_prompt(other, variant).user
    assert "do-not-send" not in text and "9876543" not in text


@pytest.mark.parametrize("key", sorted(STORAGE_KEYS | {"success_samples", "deterministic_divergence", "divergence_status"}))
def test_recursive_boundary_rejects_forbidden_future_view_fields(sample, monkeypatch, key):
    payload = sample.to_failure_only()
    payload["subtasks"][0]["failure_samples"][0][key] = "leak"
    monkeypatch.setattr(PairedBehaviorTree, "to_failure_only", lambda self: payload)
    with pytest.raises(ValueError, match="forbidden"):
        build_prompt(sample, "baseline_2")


@pytest.mark.parametrize("variant", list(ModelVariant))
def test_building_and_editing_returned_payload_never_mutate_storage(sample, variant):
    tree = computed(sample)
    before = tree.model_dump_json()
    source = SAMPLE.read_bytes()
    first = build_prompt(tree, variant)
    assert first == build_prompt(tree, variant)
    first.payload["subtasks"][0]["failure_samples"][0]["joint_positions"][0] = 999.0
    first.messages[0]["content"] = "edited container"
    assert tree.model_dump_json() == before
    assert SAMPLE.read_bytes() == source
    assert build_prompt(tree, variant).messages[0]["content"] == SYSTEM_PROMPT


def test_hashes_track_prompt_inputs_while_excluded_information_does_not(sample):
    tree = computed(sample)
    first = {variant: build_prompt(tree, variant) for variant in ModelVariant}
    for prompt in first.values():
        encoded = json.dumps(prompt.messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        assert prompt.prompt_sha256 == hashlib.sha256(encoded).hexdigest()
        assert len(prompt.prompt_sha256) == len(prompt.template_sha256) == 64
    tree.subtasks[0].success_samples[0].gripper_force = 55.0
    assert build_prompt(tree, "baseline_2").prompt_sha256 == first[ModelVariant.BASELINE_2].prompt_sha256
    assert build_prompt(tree, "baseline_3").prompt_sha256 != first[ModelVariant.BASELINE_3].prompt_sha256
    before_baseline3 = build_prompt(tree, "baseline_3").prompt_sha256
    before_main = build_prompt(tree, "main").prompt_sha256
    tree.subtasks[0].deterministic_divergence.max_abs_deltas[TelemetryFeature.GRIPPER_FORCE] = 55.0
    assert build_prompt(tree, "baseline_3").prompt_sha256 == before_baseline3
    assert build_prompt(tree, "main").prompt_sha256 != before_main


def write_templates(directory, content):
    for filename in TEMPLATE_FILENAMES.values():
        (directory / filename).write_text(content)


def test_custom_templates_are_versioned_and_keep_the_ablation_controlled(sample, tmp_path):
    content = "Common UTF-8 instruction: température.\n{{PAYLOAD}}\n"
    write_templates(tmp_path, content)
    prompt = build_prompt(sample, "baseline_2", template_dir=str(tmp_path))
    assert prompt.template_sha256 == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert "température" in prompt.user and "{{PAYLOAD}}" not in prompt.user
    assert prompt.prompt_sha256 != build_prompt(sample, "baseline_2").prompt_sha256
    (tmp_path / TEMPLATE_FILENAMES[ModelVariant.BASELINE_3]).write_text("different instructions {{PAYLOAD}}")
    with pytest.raises(ValueError, match="identical"):
        build_prompt(sample, "baseline_2", template_dir=tmp_path)


@pytest.mark.parametrize("content", ["missing marker", "{{PAYLOAD}} twice {{PAYLOAD}}"])
def test_template_requires_exactly_one_payload_marker(sample, tmp_path, content):
    write_templates(tmp_path, content)
    with pytest.raises(ValueError, match="exactly one"):
        build_prompt(sample, "baseline_2", template_dir=tmp_path)


def test_missing_template_fails_explicitly(sample, tmp_path):
    with pytest.raises(FileNotFoundError):
        build_prompt(sample, "baseline_2", template_dir=tmp_path)
    assert DEFAULT_TEMPLATE_DIR.is_dir()


@pytest.mark.parametrize("variant", ["baseline_1", "unknown", "", None, 7])
def test_unknown_variants_fail_explicitly(sample, variant):
    with pytest.raises(ValueError, match="variant must be"):
        build_prompt(sample, variant)


def test_full_storage_dict_is_not_accepted_as_a_prompt_input(sample):
    with pytest.raises(TypeError, match="PairedBehaviorTree"):
        build_prompt(sample.model_dump(mode="json"), "baseline_2")
