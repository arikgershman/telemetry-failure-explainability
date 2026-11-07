"""Baseline 1: deterministic, failure-only rule detector.

Input is the failure-only view (``PairedBehaviorTree.to_failure_only()``); the
detector never sees success samples, divergence, ground-truth labels,
perturbations, filenames, or seeds. ``predict`` rejects any input that carries
those fields, so leakage fails loudly instead of silently.

Pipeline
    1. normalize the view into typed samples (malformed input -> ``error``)
    2. run one rule per failure mode; each returns a ``Finding`` or ``None``
    3. gather downstream evidence (object not following the end effector)
    4. arbitrate: earliest stage wins, an explicit dominance table settles
       same-stage conflicts, anything still ambiguous abstains
    5. render a template explanation and emit a prediction record

The record layout follows the Track 4 prediction contract described in the
TODO (``pair_id``, ``model_variant``, ``predicted_failure_mode``,
``predicted_subtask``, ``explanation``, ``evidence``, ``status``,
``backend``). Validate it against ``eval/contracts.py`` once that exists.

Thresholds come from ``configs/rules.yaml`` and are initial mock-data values,
not calibrated on real runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "BACKEND",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MODEL_VARIANT",
    "Evidence",
    "Finding",
    "RuleConfig",
    "RuleInputError",
    "analyze",
    "load_rule_config",
    "predict",
]

BACKEND = "rules"
DEFAULT_MODEL_VARIANT = "baseline_1"
# src/telemetry_reasoning/models/rule_based.py -> repository root is parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "rules.yaml"

PICKUP_TASK = "pick_up_object"

KNOWN_SUBTASKS = frozenset(
    {
        "approach",
        "align",
        "grasp",
        "lift",
        "maintain_grasp",
        "move_along_path",
        "reach_target",
    }
)

# Keys that must never appear anywhere in the detector's input.
FORBIDDEN_KEYS = frozenset(
    {
        "failure_mode",
        "failed_subtask",
        "perturbation_param",
        "perturbation_value",
        "success_file",
        "failure_file",
        "seed",
        "success_samples",
        "deterministic_divergence",
    }
)

# failure-mode labels (Data contract v1)
LOW_GRIP = "inadequate_grip_strength"
EARLY_RELEASE = "early_gripper_release"
POSITION = "position_error"
ORIENTATION = "orientation_error"

# (dominant, dominated): when both fire in the same stage, the dominated one is
# treated as a consequence. A released gripper explains low force, not vice versa.
_DOMINANCE = frozenset({(EARLY_RELEASE, LOW_GRIP)})


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class RuleConfig(BaseModel):
    """Validated thresholds. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    closed_state: float = Field(0.5, ge=0.0, le=1.0)
    open_state: float = Field(0.2, ge=0.0, le=1.0)

    min_grip_force_n: float = Field(8.0, gt=0.0)
    persist_s: float = Field(0.5, ge=0.0)
    grip_stages: tuple[str, ...] = (
        "grasp",
        "lift",
        "maintain_grasp",
        "move_along_path",
        "reach_target",
    )
    release_stages: tuple[str, ...] = ("grasp", "lift", "maintain_grasp", "move_along_path")

    position_stages: tuple[str, ...] = ("approach", "align")
    z_check_stages: tuple[str, ...] = ("align",)
    max_xy_offset_m: float = Field(0.02, gt=0.0)
    max_z_error_m: float = Field(0.02, gt=0.0)
    grasp_z_offset_m: float = 0.05

    orientation_stages: tuple[str, ...] = ("align",)
    max_orientation_rad: float = Field(0.2, gt=0.0)

    follow_stages: tuple[str, ...] = ("lift", "maintain_grasp", "move_along_path")
    min_ee_travel_m: float = Field(0.05, gt=0.0)
    min_follow_ratio: float = Field(0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "RuleConfig":
        if self.closed_state <= self.open_state:
            raise ValueError("closed_state must be greater than open_state")
        stage_fields = (
            "grip_stages",
            "release_stages",
            "position_stages",
            "z_check_stages",
            "orientation_stages",
            "follow_stages",
        )
        for name in stage_fields:
            unknown = set(getattr(self, name)) - KNOWN_SUBTASKS
            if unknown:
                raise ValueError(f"{name} has unknown subtasks: {sorted(unknown)}")
        return self


def load_rule_config(path: Union[str, Path, None] = None) -> RuleConfig:
    """Load ``configs/rules.yaml`` (or ``path``) and validate it strictly."""
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: expected a mapping with a 'rules' section")
    extra = set(raw) - {"rules"}
    if extra:
        raise ValueError(f"{source}: unknown top-level keys: {sorted(extra)}")
    section = raw.get("rules")
    if not isinstance(section, Mapping):
        raise ValueError(f"{source}: missing 'rules' mapping")
    return RuleConfig.model_validate(dict(section))


# --------------------------------------------------------------------------- #
# Typed view of the input
# --------------------------------------------------------------------------- #
class RuleInputError(ValueError):
    """The failure-only view is malformed; ``predict`` reports status=error."""


@dataclass(frozen=True)
class _Sample:
    stage: str
    t: float
    state: float  # gripper_state
    force: float  # gripper_force
    contact: float  # contact_force
    ee_pos: tuple[float, ...]
    ee_quat: tuple[float, ...]
    obj_pos: tuple[float, ...]
    obj_quat: tuple[float, ...]


# schema feature name -> _Sample attribute
_FEATURE_ATTR = {
    "gripper_state": "state",
    "gripper_force": "force",
    "contact_force": "contact",
    "ee_pos": "ee_pos",
    "ee_quat": "ee_quat",
    "obj_pos": "obj_pos",
    "obj_quat": "obj_quat",
}


@dataclass(frozen=True)
class _Trace:
    pair_id: str
    task: str
    stage_names: tuple[str, ...]  # task-template order, as supplied
    samples: tuple[_Sample, ...]  # chronological, all stages
    by_stage: Mapping[str, tuple[_Sample, ...]]

    @property
    def order(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.stage_names)}


@dataclass
class Evidence:
    """One cited telemetry value; must exist in the failure-only input."""

    subtask: str
    timestamp: float
    feature: str
    value: Union[float, tuple[float, ...]]

    def as_dict(self) -> dict[str, Any]:
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        return {
            "subtask": self.subtask,
            "timestamp": self.timestamp,
            "feature": self.feature,
            "value": value,
        }


@dataclass
class Finding:
    mode: str
    stage: str
    onset: float
    evidence: list[Evidence]
    facts: dict[str, float] = field(default_factory=dict)


@dataclass
class _Consequence:
    stage: str
    evidence: list[Evidence]
    facts: dict[str, float]


def _ev(sample: _Sample, feature: str) -> Evidence:
    return Evidence(sample.stage, sample.t, feature, getattr(sample, _FEATURE_ATTR[feature]))


def _num(row: Mapping[str, Any], key: str, where: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        raise RuleInputError(f"{where}: missing or non-numeric '{key}'") from None
    if not math.isfinite(value):
        raise RuleInputError(f"{where}: non-finite '{key}'")
    return value


def _vec(row: Mapping[str, Any], key: str, size: Optional[int], where: str) -> tuple[float, ...]:
    raw = row.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) == 0:
        raise RuleInputError(f"{where}: '{key}' must be a non-empty list")
    if size is not None and len(raw) != size:
        raise RuleInputError(f"{where}: '{key}' needs {size} components, got {len(raw)}")
    try:
        out = tuple(float(x) for x in raw)
    except (TypeError, ValueError):
        raise RuleInputError(f"{where}: '{key}' has non-numeric entries") from None
    if not all(math.isfinite(x) for x in out):
        raise RuleInputError(f"{where}: '{key}' has non-finite entries")
    return out


def _find_forbidden(node: Any, path: str = "") -> Iterator[str]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            here = f"{path}/{key}"
            if key in FORBIDDEN_KEYS:
                yield here
            yield from _find_forbidden(value, here)
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from _find_forbidden(value, f"{path}[{i}]")


def _to_mapping(view: Any) -> Mapping[str, Any]:
    if isinstance(view, Mapping):
        return view
    if hasattr(view, "model_dump"):
        return view.model_dump(mode="json")
    raise TypeError(f"expected a mapping or Pydantic view, got {type(view).__name__}")


def _normalize(data: Mapping[str, Any]) -> _Trace:
    task = data.get("task")
    if not isinstance(task, str):
        raise RuleInputError("missing 'task'")
    nodes = data.get("subtasks")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)) or not nodes:
        raise RuleInputError("'subtasks' must be a non-empty list")

    names: list[str] = []
    by_stage: dict[str, list[_Sample]] = {}
    samples: list[_Sample] = []
    last_t: Optional[float] = None

    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("subtask"), str):
            raise RuleInputError("each subtask node needs a string 'subtask'")
        name = node["subtask"]
        if name in by_stage:
            raise RuleInputError(f"repeated subtask '{name}'")
        names.append(name)
        by_stage[name] = []
        rows = node.get("failure_samples")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise RuleInputError(f"{name}: 'failure_samples' must be a list")
        for k, row in enumerate(rows):
            where = f"{name}[{k}]"
            if not isinstance(row, Mapping):
                raise RuleInputError(f"{where}: sample must be a mapping")
            if row.get("subtask") != name:
                raise RuleInputError(f"{where}: sample subtask does not match its node")
            sample = _Sample(
                stage=name,
                t=_num(row, "timestamp", where),
                state=_num(row, "gripper_state", where),
                force=_num(row, "gripper_force", where),
                contact=_num(row, "contact_force", where),
                ee_pos=_vec(row, "ee_pos", 3, where),
                ee_quat=_vec(row, "ee_quat", 4, where),
                obj_pos=_vec(row, "obj_pos", 3, where),
                obj_quat=_vec(row, "obj_quat", 4, where),
            )
            if last_t is not None and sample.t <= last_t:
                raise RuleInputError(f"{where}: timestamps must strictly increase")
            last_t = sample.t
            by_stage[name].append(sample)
            samples.append(sample)

    return _Trace(
        pair_id=str(data.get("pair_id", "")),
        task=task,
        stage_names=tuple(names),
        samples=tuple(samples),
        by_stage={k: tuple(v) for k, v in by_stage.items()},
    )


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _quat_angle(a: Sequence[float], b: Sequence[float]) -> float:
    """Smallest rotation angle (rad) between two w/x/y/z quaternions.

    Normalizes first (v1 does not validate normalization) and uses |dot| so
    that q and -q, which are the same rotation, give angle 0.
    """
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        raise RuleInputError("zero-norm quaternion")
    dot = abs(sum(x * y for x, y in zip(a, b))) / (na * nb)
    return 2.0 * math.acos(min(1.0, dot))


# --------------------------------------------------------------------------- #
# Rules: each returns a Finding or None
# --------------------------------------------------------------------------- #
def _rule_low_grip(trace: _Trace, cfg: RuleConfig) -> Optional[Finding]:
    """Closed gripper with force below the floor for at least persist_s.

    Only closed samples can be "low": the ~0.2 N reading of an open gripper is
    not a grip failure. A run breaks on any sample that is open or not low.
    """
    cand = [s for s in trace.samples if s.stage in cfg.grip_stages]

    def is_low(s: _Sample) -> bool:
        return s.state >= cfg.closed_state and s.force < cfg.min_grip_force_n

    i = 0
    while i < len(cand):
        if not is_low(cand[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(cand) and is_low(cand[j + 1]):
            j += 1
        if cand[j].t - cand[i].t >= cfg.persist_s:
            onset = cand[i]
            evidence: list[Evidence] = []
            facts = {"force": onset.force, "contact": onset.contact, "threshold": cfg.min_grip_force_n}
            prior = cand[i - 1] if i > 0 else None
            if prior is not None and prior.state >= cfg.closed_state and not is_low(prior):
                evidence += [_ev(prior, "gripper_force"), _ev(prior, "contact_force")]
                facts.update(prior_t=prior.t, prior_force=prior.force, prior_contact=prior.contact)
            evidence += [_ev(onset, "gripper_force"), _ev(onset, "contact_force")]
            return Finding(LOW_GRIP, onset.stage, onset.t, evidence, facts)
        i = j + 1
    return None


def _rule_early_release(trace: _Trace, cfg: RuleConfig) -> Optional[Finding]:
    """Gripper goes closed -> open inside a stage where holding is required."""
    last_closed: Optional[_Sample] = None
    for s in trace.samples:
        if s.state >= cfg.closed_state:
            last_closed = s
        elif s.state <= cfg.open_state and last_closed is not None and s.stage in cfg.release_stages:
            return Finding(
                EARLY_RELEASE,
                s.stage,
                s.t,
                [_ev(last_closed, "gripper_state"), _ev(s, "gripper_state")],
                {
                    "closed_t": last_closed.t,
                    "closed_state": last_closed.state,
                    "open_state": s.state,
                },
            )
    return None


def _rule_position(trace: _Trace, cfg: RuleConfig) -> Optional[Finding]:
    """EE vs object offset at the end of each checked stage (pickup only).

    Transport has no observable target in the failure-only view, so this rule
    does not apply there and the detector abstains rather than inventing one.
    """
    if trace.task != PICKUP_TASK:
        return None
    for name in trace.stage_names:
        rows = trace.by_stage[name]
        if name not in cfg.position_stages or not rows:
            continue
        s = rows[-1]
        dx, dy, dz = (e - o for e, o in zip(s.ee_pos, s.obj_pos))
        xy = math.hypot(dx, dy)
        z_err = abs(dz - cfg.grasp_z_offset_m) if name in cfg.z_check_stages else 0.0
        if xy > cfg.max_xy_offset_m or z_err > cfg.max_z_error_m:
            return Finding(
                POSITION,
                name,
                s.t,
                [_ev(s, "ee_pos"), _ev(s, "obj_pos")],
                {
                    "xy": xy,
                    "z_err": z_err,
                    "xy_limit": cfg.max_xy_offset_m,
                    "z_limit": cfg.max_z_error_m,
                    "xy_hit": float(xy > cfg.max_xy_offset_m),
                    "z_hit": float(z_err > cfg.max_z_error_m),
                },
            )
    return None


def _rule_orientation(trace: _Trace, cfg: RuleConfig) -> Optional[Finding]:
    """Angle between ee_quat and obj_quat at the end of the checked stages."""
    if trace.task != PICKUP_TASK:
        return None
    for name in trace.stage_names:
        rows = trace.by_stage[name]
        if name not in cfg.orientation_stages or not rows:
            continue
        s = rows[-1]
        angle = _quat_angle(s.ee_quat, s.obj_quat)
        if angle > cfg.max_orientation_rad:
            return Finding(
                ORIENTATION,
                name,
                s.t,
                [_ev(s, "ee_quat"), _ev(s, "obj_quat")],
                {"angle": angle, "limit": cfg.max_orientation_rad},
            )
    return None


_RULES: tuple[Callable[[_Trace, RuleConfig], Optional[Finding]], ...] = (
    _rule_low_grip,
    _rule_early_release,
    _rule_position,
    _rule_orientation,
)


def _object_not_following(trace: _Trace, cfg: RuleConfig) -> Optional[_Consequence]:
    """Downstream symptom: the EE travelled but the object did not follow.

    Several modes produce this, so it supports an explanation but never
    selects a failure mode by itself.
    """
    for name in trace.stage_names:
        rows = trace.by_stage[name]
        if name not in cfg.follow_stages or len(rows) < 2:
            continue
        a, b = rows[0], rows[-1]
        ee_travel = math.dist(a.ee_pos, b.ee_pos)
        obj_travel = math.dist(a.obj_pos, b.obj_pos)
        if ee_travel >= cfg.min_ee_travel_m and obj_travel < cfg.min_follow_ratio * ee_travel:
            return _Consequence(
                name,
                [_ev(a, "ee_pos"), _ev(b, "ee_pos"), _ev(a, "obj_pos"), _ev(b, "obj_pos")],
                {"ee_travel": ee_travel, "obj_travel": obj_travel},
            )
    return None


# --------------------------------------------------------------------------- #
# Arbitration
# --------------------------------------------------------------------------- #
def _arbitrate(findings: list[Finding], order: Mapping[str, int]) -> tuple[Optional[Finding], str]:
    """Earliest stage wins; dominance resolves same-stage ties; else abstain."""
    if not findings:
        return None, "no rule fired"
    first = min(order[f.stage] for f in findings)
    tied = [f for f in findings if order[f.stage] == first]
    modes = {f.mode for f in tied}
    survivors = [f for f in tied if not any((d, f.mode) in _DOMINANCE for d in modes)]
    if len({f.mode for f in survivors}) == 1:
        return survivors[0], ""
    names = ", ".join(sorted({f.mode for f in survivors}))
    return None, f"ambiguous: {names} all fire in {tied[0].stage}"


# --------------------------------------------------------------------------- #
# Explanation templates
# --------------------------------------------------------------------------- #
def _words(label: str) -> str:
    return label.replace("_", " ")


def _describe(f: Finding) -> str:
    k = f.facts
    if f.mode == LOW_GRIP:
        if "prior_force" in k:
            return (
                f"In {f.stage}, closed-gripper force fell from {k['prior_force']:.1f} N "
                f"(t={k['prior_t']:.2f} s) to {k['force']:.1f} N (t={f.onset:.2f} s), below the "
                f"{k['threshold']:.1f} N minimum, and contact force went from "
                f"{k['prior_contact']:.1f} N to {k['contact']:.1f} N."
            )
        return (
            f"In {f.stage}, closed-gripper force was {k['force']:.1f} N at t={f.onset:.2f} s, "
            f"below the {k['threshold']:.1f} N minimum, with contact force {k['contact']:.1f} N."
        )
    if f.mode == EARLY_RELEASE:
        return (
            f"In {f.stage}, the gripper opened (closure {k['closed_state']:.2f} at "
            f"t={k['closed_t']:.2f} s, then {k['open_state']:.2f} at t={f.onset:.2f} s) "
            "while the grasp was still needed."
        )
    if f.mode == POSITION:
        parts = []
        if k["xy_hit"]:
            parts.append(f"{k['xy']:.3f} m from the object in x/y (limit {k['xy_limit']:.3f} m)")
        if k["z_hit"]:
            parts.append(f"{k['z_err']:.3f} m off the expected grasp height (limit {k['z_limit']:.3f} m)")
        return f"At the end of {f.stage} (t={f.onset:.2f} s), the end effector was " + " and ".join(parts) + "."
    return (  # ORIENTATION
        f"At the end of {f.stage} (t={f.onset:.2f} s), the gripper orientation differed from "
        f"the object's by {k['angle']:.2f} rad, above the {k['limit']:.2f} rad limit."
    )


def _describe_consequence(c: _Consequence) -> str:
    return (
        f"In {c.stage}, the end effector moved {c.facts['ee_travel']:.3f} m while the object "
        f"moved {c.facts['obj_travel']:.3f} m, so the object did not follow the end effector."
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def analyze(
    trace: _Trace, cfg: RuleConfig
) -> tuple[Optional[Finding], Optional[_Consequence], str]:
    """Run all rules and arbitrate. Returns (finding, consequence, abstain_reason)."""
    findings = [f for rule in _RULES if (f := rule(trace, cfg)) is not None]
    chosen, reason = _arbitrate(findings, trace.order)
    return chosen, _object_not_following(trace, cfg), reason


def _record(
    pair_id: str,
    variant: str,
    status: str,
    mode: Optional[str],
    subtask: Optional[str],
    explanation: str,
    evidence: list[Evidence],
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "model_variant": variant,
        "predicted_failure_mode": mode,
        "predicted_subtask": subtask,
        "explanation": explanation,
        "evidence": [e.as_dict() for e in evidence],
        "status": status,
        "backend": BACKEND,
    }


def predict(
    view: Any,
    cfg: Optional[RuleConfig] = None,
    *,
    model_variant: str = DEFAULT_MODEL_VARIANT,
) -> dict[str, Any]:
    """Predict a failure mode from one failure-only view.

    Raises ``ValueError`` if the input carries label/success/divergence fields
    (a leakage bug in the caller). Malformed telemetry yields ``status="error"``;
    unsupported or ambiguous evidence yields ``status="abstain"``.
    """
    if cfg is None:
        cfg = load_rule_config()
    data = _to_mapping(view)

    leaks = list(_find_forbidden(data))
    if leaks:
        raise ValueError(
            "input is not a failure-only view; forbidden fields present: " + ", ".join(leaks[:5])
        )

    pair_id = str(data.get("pair_id", ""))
    try:
        trace = _normalize(data)
        finding, consequence, reason = analyze(trace, cfg)
    except RuleInputError as exc:
        return _record(pair_id, model_variant, "error", None, None, f"Input rejected: {exc}.", [])

    if finding is None:
        text = f"No rule identified a failure cause from the failed run alone ({reason})."
        evidence: list[Evidence] = []
        if consequence is not None:
            text += " Observed: " + _describe_consequence(consequence)
            evidence = list(consequence.evidence)
        return _record(trace.pair_id, model_variant, "abstain", None, None, text, evidence)

    text = _describe(finding)
    evidence = list(finding.evidence)
    if consequence is not None:
        sentence = _describe_consequence(consequence)
        text += " Downstream, " + sentence[0].lower() + sentence[1:]
        evidence += consequence.evidence
    text += f" Predicted failure: {_words(finding.mode)}, beginning in {finding.stage}."
    return _record(trace.pair_id, model_variant, "ok", finding.mode, finding.stage, text, evidence)
