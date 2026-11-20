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


@pytest.mark.parametrize("feature,offset", [
    (TelemetryFeature.GRIPPER_FORCE, 3.0),
    (TelemetryFeature.CONTACT_FORCE, -4.0),
    (TelemetryFeature.JOINT_POSITIONS, [0.0, -0.3, 0.2, 0.0, 0.1, 0.0]),
    (TelemetryFeature.EE_POS, [0.01, -0.04, 0.02]),
])
def test_metric_constant_scalar_and_vector_offsets(feature, offset):
    node = fixture_tree().subtasks[0]
    node.failure_samples = [row.model_copy(deep=True) for row in node.success_samples]
    node.failure_segment_end = "transitioned"
    for row in node.failure_samples:
        value = getattr(row, feature.value)
        setattr(row, feature.value, ([a + b for a, b in zip(value, offset)]
                                   if isinstance(offset, list) else value + offset))
    expected = max(map(abs, offset)) if isinstance(offset, list) else abs(offset)
    config = DivergenceConfig(thresholds={f: expected / 2 if f == feature else 1e9
                                        for f in TelemetryFeature})
    result = compute_subtask_divergence(node, config)
    assert result.max_abs_deltas[feature] == pytest.approx(expected)
    assert result.first_divergence_timestamp == node.failure_samples[0].timestamp
    assert all(value == pytest.approx(0.0) for key, value in result.max_abs_deltas.items() if key != feature)


@pytest.mark.parametrize("segment_end", ["transitioned", "trace_end"])
def test_metric_shifted_resampled_analytic_motion(segment_end):
    from test_divergence import node, observation

    success = [observation(2 + t, value=2 * t) for t in (0, 0.5, 2)]
    failure = [observation(10 + t, value=2 * t) for t in (0, 0.2, 0.8, 1.4, 2)]
    pair = node(success, failure, segment_end=segment_end)
    config = DivergenceConfig(thresholds={f: 1e-10 for f in TelemetryFeature})
    before = pair.model_dump_json()
    result = compute_subtask_divergence(pair, config)
    assert result.first_divergence_timestamp is None
    assert all(value == pytest.approx(0.0, abs=1e-12) for value in result.max_abs_deltas.values())
    assert pair.model_dump_json() == before


def test_metric_opposite_quaternion_signs_are_same_rotation():
    from test_divergence import node, observation
    import numpy as np

    success = [observation(t, quaternion=[float(np.cos(t / 4)), 0, 0, float(np.sin(t / 4))])
               for t in (0, 2)]
    failure = [observation(10 + t, quaternion=[float(-np.cos(t / 4)), 0, 0, float(-np.sin(t / 4))])
               for t in (0, 0.4, 0.8, 2)]
    config = DivergenceConfig(thresholds={f: 1e-10 for f in TelemetryFeature})
    result = compute_subtask_divergence(node(success, failure), config)
    assert result.first_divergence_timestamp is None
    assert result.max_abs_deltas[TelemetryFeature.EE_QUAT] == pytest.approx(0, abs=1e-14)
    assert result.max_abs_deltas[TelemetryFeature.OBJ_QUAT] == pytest.approx(0, abs=1e-14)
