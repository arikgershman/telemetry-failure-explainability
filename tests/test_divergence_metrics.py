"""Metric and main-prompt integration checks; no model runtime or GPU required."""

from pathlib import Path

import pytest

from telemetry_reasoning.models.prompts import build_prompt
from telemetry_reasoning.parser.divergence_metrics import (
    DivergenceConfig, compute_subtask_divergence, enrich_paired_bt,
    load_divergence_config,
)
from telemetry_reasoning.schemas import PairedBehaviorTree, TelemetryFeature

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/sample/pair_000_bt.json"


def fixture_tree():
    return PairedBehaviorTree.model_validate_json(SAMPLE.read_text())


def only_force_threshold() -> DivergenceConfig:
    # Isolate the known grip event; all remaining features are effectively off.
    return DivergenceConfig(thresholds={
        feature: (2.0 if feature == TelemetryFeature.GRIPPER_FORCE else 1e9)
        for feature in TelemetryFeature
    })


def test_known_grasp_event():
    tree = fixture_tree()
    grasp = next(node for node in tree.subtasks if node.subtask.value == "grasp")
    result = compute_subtask_divergence(grasp, only_force_threshold())
    assert result is not None
    assert result.max_abs_deltas[TelemetryFeature.GRIPPER_FORCE] == pytest.approx(12.0)
    assert result.first_divergence_timestamp == pytest.approx(5.10)
    assert set(result.max_abs_deltas) == set(TelemetryFeature)
    assert grasp.deterministic_divergence is None  # no input mutation


def test_strict_threshold_and_no_crossing():
    tree = fixture_tree()
    grasp = next(node for node in tree.subtasks if node.subtask.value == "grasp")
    config = DivergenceConfig(thresholds={f: 1e9 for f in TelemetryFeature})
    result = compute_subtask_divergence(grasp, config)
    assert result.first_divergence_timestamp is None
    assert result.max_abs_deltas[TelemetryFeature.GRIPPER_FORCE] == pytest.approx(12.0)
    equal_threshold = DivergenceConfig(thresholds={
        f: (12.0 if f == TelemetryFeature.GRIPPER_FORCE else 1e9)
        for f in TelemetryFeature
    })
    assert compute_subtask_divergence(grasp, equal_threshold).first_divergence_timestamp is None


def test_identical_observations_return_zero():
    tree = fixture_tree()
    node = next(node for node in tree.subtasks if node.subtask.value == "grasp")
    node.failure_samples = [sample.model_copy(deep=True) for sample in node.success_samples]
    node.failure_segment_end = "transitioned"
    result = compute_subtask_divergence(node, only_force_threshold())
    assert result.first_divergence_timestamp is None
    assert all(value == pytest.approx(0.0) for value in result.max_abs_deltas.values())


def test_full_enrichment_works_for_main_and_keeps_baselines_blinded():
    original = fixture_tree()
    updated = enrich_paired_bt(original, only_force_threshold())
    assert updated is not original
    assert all(node.divergence_status == "computed" for node in updated.subtasks)
    assert all(node.divergence_status == "uncomputed" for node in original.subtasks)
    assert PairedBehaviorTree.model_validate_json(updated.model_dump_json()) == updated
    main = build_prompt(updated, "main")
    assert "deterministic_divergence" in main.user
    for variant in ("baseline_2", "baseline_3"):
        prompt = build_prompt(updated, variant)
        assert "deterministic_divergence" not in prompt.user
        assert "divergence_status" not in prompt.user
    with pytest.raises(ValueError, match="computed"):
        build_prompt(original, "main")


def test_unobserved_stages_remain_not_comparable():
    tree = fixture_tree()
    # Construct a valid partial failure prefix that stops during grasp.
    lift = tree.subtasks[-1]
    lift.failure_samples = []
    lift.failure_observation_status = "not_observed"
    lift.failure_segment_end = None
    lift.divergence_status = "not_comparable"
    tree.subtasks[-2].failure_segment_end = "trace_end"
    tree = PairedBehaviorTree.model_validate(tree.model_dump(mode="json"))
    enriched = enrich_paired_bt(tree, only_force_threshold())
    assert enriched.subtasks[-1].deterministic_divergence is None
    assert enriched.subtasks[-1].divergence_status == "not_comparable"
    build_prompt(enriched, "main")


def test_config_complete_and_loads():
    config = load_divergence_config(ROOT / "configs/divergence.yaml")
    assert set(config.thresholds) == set(TelemetryFeature)
    with pytest.raises(ValueError, match="missing"):
        DivergenceConfig(thresholds={TelemetryFeature.GRIPPER_FORCE: 2.0})
