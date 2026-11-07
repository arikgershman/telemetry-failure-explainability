"""Tests for Baseline 1 (rule-based, failure-only detector).

All synthetic cases are built here by a small view builder; nothing under
``data/sample/`` is modified. The builder mirrors the retained timestamps and
key values of ``pair_000_bt.json`` (grip 15 N -> 3 N at 5.10 s, object left
behind during ``lift``).

One mock pair only shows the pipeline runs. These tests are not an accuracy
estimate, and the thresholds in ``configs/rules.yaml`` are not calibrated.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from telemetry_reasoning.models.rule_based import (
    DEFAULT_CONFIG_PATH,
    RuleConfig,
    load_rule_config,
    predict,
)

PICKUP = "pick_up_object"
TRANSPORT = "transport_object"

PICKUP_STAGES = {
    "approach": [0.0, 0.32, 0.66, 1.0, 1.32, 1.64, 1.98],
    "align": [2.0, 2.32, 2.66, 3.0, 3.32, 3.64, 3.98],
    "grasp": [4.0, 4.32, 4.38, 4.4, 4.42, 4.66, 5.0, 5.08, 5.1, 5.12, 5.32, 5.64, 5.98],
    "lift": [6.0, 6.32, 6.66, 7.0, 7.32, 7.64, 7.98],
}
TRANSPORT_STAGES = {
    "maintain_grasp": [0.0, 0.5, 1.0, 1.5, 1.98],
    "move_along_path": [2.0, 3.0, 4.0, 5.0, 5.98],
    "reach_target": [6.0, 6.5, 7.0, 7.5, 7.98],
}
FOLLOW_STAGES = {"lift", "maintain_grasp", "move_along_path", "reach_target"}
BASE_JOINT_VEL = [0.1, -0.1, 0.05, 0.0, 0.1, -0.05]


# --------------------------------------------------------------------------- #
# View builder
# --------------------------------------------------------------------------- #
def default_ee(t: float, stage: str) -> list[float]:
    if stage == "approach":
        return [min(0.4, 0.1 + 0.3 * t / 1.98), 0.0, 0.2]
    if stage == "align":
        return [0.4, 0.0, 0.2 - 0.1 * (t - 2.0) / 1.98]
    if stage == "grasp":
        return [0.4, 0.0, 0.1]
    if stage == "lift":
        return [0.4, 0.0, 0.1 + 0.2 * (t - 6.0) / 1.98]
    return [0.4 + 0.4 * t / 7.98, 0.0, 0.2]  # transport stages


def default_obj(t: float, stage: str, ee: list[float]) -> list[float]:
    if stage in FOLLOW_STAGES:
        return [ee[0], ee[1], ee[2] - 0.05]  # held object rides with the gripper
    return [0.4, 0.0, 0.05]


def build_view(
    stages: dict[str, list[float]] | None = None,
    task: str = PICKUP,
    closed_from: float = 4.4,
    pair_id: str = "pair-XYZ",
    **fns,
) -> dict:
    """Healthy failure-only view; override any channel with fn(t, stage)."""
    stages = stages or PICKUP_STAGES
    nodes = []
    for stage, times in stages.items():
        rows = []
        for t in times:
            closed = t >= closed_from
            ee = fns["ee"](t, stage) if "ee" in fns else default_ee(t, stage)
            obj = fns["obj"](t, stage) if "obj" in fns else default_obj(t, stage, ee)
            rows.append(
                {
                    "timestamp": t,
                    "subtask": stage,
                    "joint_positions": [0.0] * 6,
                    "joint_velocities": (
                        fns["joint_vel"](t, stage) if "joint_vel" in fns else list(BASE_JOINT_VEL)
                    ),
                    "joint_torques": [0.0] * 6,
                    "gripper_state": fns["state"](t, stage) if "state" in fns else (1.0 if closed else 0.0),
                    "gripper_force": fns["force"](t, stage) if "force" in fns else (15.0 if closed else 0.2),
                    "ee_pos": ee,
                    "ee_quat": fns["ee_quat"](t, stage) if "ee_quat" in fns else [1.0, 0.0, 0.0, 0.0],
                    "obj_pos": obj,
                    "obj_quat": [1.0, 0.0, 0.0, 0.0],
                    "obj_vel": [0.0, 0.0, 0.0],
                    "contact_force": fns["contact"](t, stage) if "contact" in fns else (12.0 if closed else 0.0),
                }
            )
        nodes.append(
            {
                "subtask": stage,
                "metadata": {"subgoal_name": stage.title(), "action_description": "synthetic"},
                "failure_samples": rows,
            }
        )
    return {
        "schema_version": "1.0",
        "pair_id": pair_id,
        "task": task,
        "overall_goal": "synthetic goal",
        "subtasks": nodes,
    }


def sample_like_view() -> dict:
    """Mirrors pair_000: grip drops 15 -> 3 N at 5.10 s; object stays put in lift."""
    return build_view(
        force=lambda t, s: 0.2 if t < 4.4 else (15.0 if t < 5.1 else 3.0),
        contact=lambda t, s: 0.0 if t < 4.4 else (12.0 if t < 5.1 else 2.4),
        obj=lambda t, s: [0.4, 0.0, 0.05],
    )


def transport_view(**fns) -> dict:
    return build_view(TRANSPORT_STAGES, TRANSPORT, closed_from=0.0, **fns)


@pytest.fixture(scope="module")
def cfg() -> RuleConfig:
    return load_rule_config(DEFAULT_CONFIG_PATH)


# --------------------------------------------------------------------------- #
# Config loader
# --------------------------------------------------------------------------- #
def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(text)
    return path


def test_shipped_config_loads(cfg):
    assert cfg.closed_state > cfg.open_state
    assert cfg.min_grip_force_n > 0
    assert "grasp" in cfg.grip_stages


def test_config_rejects_unknown_rule_key(tmp_path):
    with pytest.raises(ValueError):
        load_rule_config(write_yaml(tmp_path, "rules:\n  not_a_threshold: 1.0\n"))


def test_config_rejects_unknown_top_level_key(tmp_path):
    with pytest.raises(ValueError):
        load_rule_config(write_yaml(tmp_path, "rules: {}\nextras: {}\n"))


def test_config_requires_rules_section(tmp_path):
    with pytest.raises(ValueError):
        load_rule_config(write_yaml(tmp_path, "min_grip_force_n: 8\n"))
    with pytest.raises(ValueError):
        load_rule_config(write_yaml(tmp_path, ""))


def test_config_rejects_inverted_hysteresis(tmp_path):
    with pytest.raises(ValueError):
        load_rule_config(write_yaml(tmp_path, "rules:\n  closed_state: 0.2\n  open_state: 0.5\n"))


def test_config_rejects_unknown_stage_name(tmp_path):
    with pytest.raises(ValueError):
        load_rule_config(write_yaml(tmp_path, "rules:\n  grip_stages: [grasp, graps]\n"))


def test_config_rejects_nonpositive_threshold(tmp_path):
    with pytest.raises(ValueError):
        load_rule_config(write_yaml(tmp_path, "rules:\n  min_grip_force_n: 0\n"))


# --------------------------------------------------------------------------- #
# The motivating example: pair_000-like inadequate grip
# --------------------------------------------------------------------------- #
def test_sample_like_view_detects_low_grip_in_grasp(cfg):
    rec = predict(sample_like_view(), cfg)
    assert rec["status"] == "ok"
    assert rec["predicted_failure_mode"] == "inadequate_grip_strength"
    assert rec["predicted_subtask"] == "grasp"  # injection stage, not the lift symptom
    assert rec["backend"] == "rules"

    ev = {(e["subtask"], e["timestamp"], e["feature"]): e["value"] for e in rec["evidence"]}
    assert ev[("grasp", 5.1, "gripper_force")] == 3.0
    assert ev[("grasp", 5.1, "contact_force")] == 2.4
    assert ev[("grasp", 5.08, "gripper_force")] == 15.0  # pre-drop reading, same trace


def test_sample_like_explanation_cites_values_and_symptom(cfg):
    text = predict(sample_like_view(), cfg)["explanation"]
    assert "15.0 N" in text and "3.0 N" in text and "t=5.10 s" in text
    assert "Downstream" in text and "did not follow" in text
    assert "inadequate grip strength" in text


def test_healthy_view_abstains(cfg):
    rec = predict(build_view(), cfg)
    assert rec["status"] == "abstain"
    assert rec["predicted_failure_mode"] is None and rec["predicted_subtask"] is None
    assert rec["evidence"] == []


def test_brief_force_dip_is_not_a_grip_failure(cfg):
    # 3 N for ~0.22 s (closure ramp / noise) then recovers: shorter than persist_s.
    view = build_view(force=lambda t, s: 0.2 if t < 4.4 else (3.0 if 5.1 <= t < 5.4 else 15.0))
    assert predict(view, cfg)["status"] == "abstain"


def test_open_gripper_low_force_is_ignored(cfg):
    # Never closed: 0.2 N is a normal open-gripper reading, not a failed grip.
    view = build_view(closed_from=1e9)
    assert predict(view, cfg)["status"] == "abstain"


# --------------------------------------------------------------------------- #
# Other modes and competing symptoms
# --------------------------------------------------------------------------- #
def test_early_release_in_lift(cfg):
    view = build_view(
        state=lambda t, s: 1.0 if 4.4 <= t < 6.5 else 0.0,
        force=lambda t, s: 15.0 if 4.4 <= t < 6.5 else 0.2,
        contact=lambda t, s: 12.0 if 4.4 <= t < 6.5 else 0.0,
    )
    rec = predict(view, cfg)
    assert (rec["predicted_failure_mode"], rec["predicted_subtask"]) == ("early_gripper_release", "lift")
    assert "t=6.66 s" in rec["explanation"]


def test_release_dominates_low_force_in_same_stage(cfg):
    # Force sags while still closed (0.66 s, persistent), then the gripper opens.
    view = build_view(
        state=lambda t, s: 1.0 if 4.4 <= t < 7.0 else 0.0,
        force=lambda t, s: 15.0 if 4.4 <= t < 6.0 else (3.0 if 6.0 <= t < 7.0 else 0.2),
        contact=lambda t, s: 12.0 if 4.4 <= t < 6.0 else (2.4 if 6.0 <= t < 7.0 else 0.0),
    )
    rec = predict(view, cfg)
    assert rec["predicted_failure_mode"] == "early_gripper_release"
    assert rec["predicted_subtask"] == "lift"


def test_position_error_found_at_end_of_approach(cfg):
    view = build_view(
        ee=lambda t, s: [default_ee(t, s)[0] + 0.05 * min(1.0, t / 1.98), 0.0, default_ee(t, s)[2]]
    )
    rec = predict(view, cfg)
    assert (rec["predicted_failure_mode"], rec["predicted_subtask"]) == ("position_error", "approach")


def test_position_error_found_in_align_when_approach_is_clean(cfg):
    def ee(t, s):
        base = default_ee(t, s)
        off = 0.05 * min(1.0, (t - 2.0) / 1.98) if t >= 2.0 else 0.0
        return [base[0] + off, base[1], base[2]]

    rec = predict(build_view(ee=ee), cfg)
    assert (rec["predicted_failure_mode"], rec["predicted_subtask"]) == ("position_error", "align")


def test_height_error_at_end_of_align(cfg):
    def ee(t, s):
        base = default_ee(t, s)
        return [base[0], base[1], base[2] + 0.05 * min(1.0, max(0.0, (t - 2.0) / 1.98))]

    rec = predict(build_view(ee=ee), cfg)
    assert (rec["predicted_failure_mode"], rec["predicted_subtask"]) == ("position_error", "align")
    assert "grasp height" in rec["explanation"]


@pytest.mark.parametrize(
    "angle, expected",
    [(0.5, "orientation_error"), (0.1, None)],
)
def test_orientation_threshold(cfg, angle, expected):
    q = [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
    view = build_view(ee_quat=lambda t, s: q if t >= 2.0 else [1.0, 0.0, 0.0, 0.0])
    rec = predict(view, cfg)
    assert rec["predicted_failure_mode"] == expected
    if expected:
        assert rec["predicted_subtask"] == "align"


def test_quaternion_sign_flip_is_the_same_rotation(cfg):
    view = build_view(ee_quat=lambda t, s: [-1.0, 0.0, 0.0, 0.0])
    assert predict(view, cfg)["status"] == "abstain"


# --------------------------------------------------------------------------- #
# Transport task
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "open_from, expected",
    [
        (2.0, ("early_gripper_release", "move_along_path")),  # dropped mid-transport
        (6.0, None),  # releasing at reach_target is the expected end of the task
    ],
)
def test_release_is_only_a_failure_before_reach_target(cfg, open_from, expected):
    view = transport_view(
        state=lambda t, s: 1.0 if t < open_from else 0.0,
        force=lambda t, s: 15.0 if t < open_from else 0.2,
        contact=lambda t, s: 12.0 if t < open_from else 0.0,
    )
    rec = predict(view, cfg)
    got = (rec["predicted_failure_mode"], rec["predicted_subtask"])
    assert got == (expected or (None, None))
    assert rec["status"] == ("ok" if expected else "abstain")


def test_healthy_transport_abstains(cfg):
    assert predict(transport_view(), cfg)["status"] == "abstain"


def test_low_grip_in_maintain_grasp(cfg):
    view = transport_view(force=lambda t, s: 3.0 if t < 1.5 else 15.0)
    rec = predict(view, cfg)
    assert (rec["predicted_failure_mode"], rec["predicted_subtask"]) == (
        "inadequate_grip_strength",
        "maintain_grasp",
    )
    assert "t=0.00 s" in rec["explanation"]  # no pre-drop reading exists


def test_transport_abstains_on_unobservable_target(cfg):
    # Position/orientation have no observable reference in transport: abstain,
    # while still reporting the object-not-following observation.
    view = transport_view(
        obj=lambda t, s: [5.0, 5.0, 5.0],
        ee_quat=lambda t, s: [math.cos(0.5), 0.0, 0.0, math.sin(0.5)],
    )
    rec = predict(view, cfg)
    assert rec["status"] == "abstain"
    assert rec["predicted_failure_mode"] is None
    assert "did not follow" in rec["explanation"]


# --------------------------------------------------------------------------- #
# Leakage guard, malformed input, record contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v.update(metadata={"failure_mode": "position_error"}),
        lambda v: v["subtasks"][0].update(success_samples=[]),
        lambda v: v["subtasks"][1].update(deterministic_divergence=None),
        lambda v: v["subtasks"][2]["failure_samples"][0].update(seed=723),
    ],
)
def test_label_or_success_fields_are_rejected(cfg, mutate):
    view = build_view()
    mutate(view)
    with pytest.raises(ValueError, match="failure-only"):
        predict(view, cfg)


def test_malformed_telemetry_reports_error_status(cfg):
    missing = build_view()
    del missing["subtasks"][1]["failure_samples"][2]["ee_pos"]
    assert predict(missing, cfg)["status"] == "error"

    backwards = build_view()
    backwards["subtasks"][0]["failure_samples"][3]["timestamp"] = 0.1
    assert predict(backwards, cfg)["status"] == "error"

    repeated = build_view()
    repeated["subtasks"].append(copy.deepcopy(repeated["subtasks"][0]))
    assert predict(repeated, cfg)["status"] == "error"

    zero_quat = build_view(ee_quat=lambda t, s: [0.0, 0.0, 0.0, 0.0])
    assert predict(zero_quat, cfg)["status"] == "error"


def test_error_record_still_has_full_shape(cfg):
    bad = build_view()
    bad["subtasks"] = []
    rec = predict(bad, cfg)
    assert rec["status"] == "error" and rec["predicted_failure_mode"] is None
    assert rec["pair_id"] == "pair-XYZ"


def test_non_mapping_input_is_a_type_error(cfg):
    with pytest.raises(TypeError):
        predict([1, 2, 3], cfg)


RECORD_KEYS = {
    "pair_id",
    "model_variant",
    "predicted_failure_mode",
    "predicted_subtask",
    "explanation",
    "evidence",
    "status",
    "backend",
}


def test_record_shape_and_pair_id_passthrough(cfg):
    rec = predict(sample_like_view(), cfg, model_variant="baseline_1")
    assert set(rec) == RECORD_KEYS
    assert rec["pair_id"] == "pair-XYZ" and rec["model_variant"] == "baseline_1"
    assert "pair-XYZ" not in rec["explanation"]  # join key only, never prose
    for e in rec["evidence"]:
        assert set(e) == {"subtask", "timestamp", "feature", "value"}


@pytest.mark.parametrize("make_view", [sample_like_view, build_view])
def test_evidence_exists_in_the_input(cfg, make_view):
    view = make_view()
    index = {(n["subtask"], r["timestamp"]): r for n in view["subtasks"] for r in n["failure_samples"]}
    rec = predict(view, cfg)
    if rec["status"] == "ok":
        assert rec["evidence"]
    for e in rec["evidence"]:
        row = index[(e["subtask"], e["timestamp"])]
        assert row[e["feature"]] == e["value"]


def test_predict_is_deterministic_and_does_not_mutate_input(cfg):
    view = sample_like_view()
    snapshot = copy.deepcopy(view)
    assert predict(view, cfg) == predict(view, cfg)
    assert view == snapshot


def test_pydantic_style_views_are_accepted(cfg):
    class FakeView:
        def __init__(self, data):
            self._data = data

        def model_dump(self, mode="python"):
            assert mode == "json"
            return copy.deepcopy(self._data)

    view = sample_like_view()
    assert predict(FakeView(view), cfg) == predict(view, cfg)


# --------------------------------------------------------------------------- #
# Real sample (skipped until data/sample/pair_000_bt.json is present)
# --------------------------------------------------------------------------- #
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "pair_000_bt.json"


@pytest.mark.skipif(not SAMPLE.exists(), reason="data/sample/pair_000_bt.json not present")
def test_real_sample_pair_000(cfg):
    from telemetry_reasoning.schemas import PairedBehaviorTree

    tree = PairedBehaviorTree.model_validate_json(SAMPLE.read_text())
    rec = predict(tree.to_failure_only(), cfg)
    assert rec["status"] == "ok"
    assert (rec["predicted_failure_mode"], rec["predicted_subtask"]) == (
        "inadequate_grip_strength",
        "grasp",
    )
    # The paired view contains success samples, which Baseline 1 must refuse.
    with pytest.raises(ValueError, match="failure-only"):
        predict(tree.to_paired_without_divergence(), cfg)
