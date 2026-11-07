"""Shared prediction contract: strict JSONL validation and ``(pair_id, model_variant)`` joins.

One JSON object per line, one line per ``(pair_id, model_variant)``::

    pair_id                 str        join key (strings preserve "000")
    model_variant           str        baseline_1 | baseline_2 | baseline_3 | main
    predicted_failure_mode  str|null   FailureMode value
    predicted_subtask       str|null   Subtask value
    explanation             str
    evidence                list       [{subtask, timestamp, feature, value}, ...]
    status                  str        ok | abstain | error
    backend                 str        rules | hf | stub

Nothing here is silently dropped. Lines that cannot be parsed are returned as
``InvalidLine`` objects; when their ``pair_id``/``model_variant`` can be read
they still occupy that key (and are scored as invalid), otherwise they are
reported as unattributable.

The label constants below mirror ``telemetry_reasoning.schemas`` (``FailureMode``,
``Subtask``, ``Task``, ``TelemetryFeature``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Iterable, Mapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Label vocabularies (mirror schemas.py; cross-checked in tests)
# --------------------------------------------------------------------------- #
TASKS: tuple[str, ...] = ("pick_up_object", "transport_object")
TASK_SUBTASKS: Mapping[str, tuple[str, ...]] = {
    "pick_up_object": ("approach", "align", "grasp", "lift"),
    "transport_object": ("maintain_grasp", "move_along_path", "reach_target"),
}
SUBTASKS: tuple[str, ...] = tuple(s for subs in TASK_SUBTASKS.values() for s in subs)
FAILURE_MODES: tuple[str, ...] = (
    "position_error",
    "orientation_error",
    "inadequate_grip_strength",
    "early_gripper_release",
)
MODEL_VARIANTS: tuple[str, ...] = ("baseline_1", "baseline_2", "baseline_3", "main")
STATUSES: tuple[str, ...] = ("ok", "abstain", "error")
BACKENDS: tuple[str, ...] = ("rules", "hf", "stub")

# Evidence feature -> required number of components (None = scalar, 0 = any nonzero length).
FEATURE_SHAPES: Mapping[str, Optional[int]] = {
    "joint_positions": 0,
    "joint_velocities": 0,
    "joint_torques": 0,
    "gripper_state": None,
    "gripper_force": None,
    "ee_pos": 3,
    "ee_quat": 4,
    "obj_pos": 3,
    "obj_quat": 4,
    "obj_vel": 3,
    "contact_force": None,
}

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class ContractError(ValueError):
    """Duplicate keys, unknown IDs, or other join violations."""


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subtask: str
    timestamp: FiniteFloat
    feature: str
    value: Union[FiniteFloat, list[FiniteFloat]]

    @field_validator("subtask")
    @classmethod
    def _known_subtask(cls, v: str) -> str:
        if v not in SUBTASKS:
            raise ValueError(f"unknown subtask {v!r}")
        return v

    @field_validator("timestamp")
    @classmethod
    def _nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be >= 0")
        return v

    @model_validator(mode="after")
    def _shape_matches_feature(self) -> "EvidenceItem":
        if self.feature not in FEATURE_SHAPES:
            raise ValueError(f"unknown feature {self.feature!r}")
        want = FEATURE_SHAPES[self.feature]
        is_vec = isinstance(self.value, list)
        if want is None:
            if is_vec:
                raise ValueError(f"{self.feature} is a scalar, got a vector")
        else:
            if not is_vec:
                raise ValueError(f"{self.feature} is a vector, got a scalar")
            if len(self.value) == 0 or (want and len(self.value) != want):
                raise ValueError(f"{self.feature} needs {want or 'a nonzero number of'} components")
        return self


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_id: Annotated[str, Field(strict=True, min_length=1)]
    model_variant: str
    predicted_failure_mode: Optional[str]
    predicted_subtask: Optional[str]
    explanation: Annotated[str, Field(strict=True)]
    evidence: list[EvidenceItem]
    status: str
    backend: str

    @field_validator("model_variant")
    @classmethod
    def _known_variant(cls, v: str) -> str:
        if v not in MODEL_VARIANTS:
            raise ValueError(f"unknown model_variant {v!r}; expected one of {MODEL_VARIANTS}")
        return v

    @field_validator("predicted_failure_mode")
    @classmethod
    def _known_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in FAILURE_MODES:
            raise ValueError(f"unknown failure mode {v!r}")
        return v

    @field_validator("predicted_subtask")
    @classmethod
    def _known_subtask(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in SUBTASKS:
            raise ValueError(f"unknown subtask {v!r}")
        return v

    @field_validator("status")
    @classmethod
    def _known_status(cls, v: str) -> str:
        if v not in STATUSES:
            raise ValueError(f"unknown status {v!r}; expected one of {STATUSES}")
        return v

    @field_validator("backend")
    @classmethod
    def _known_backend(cls, v: str) -> str:
        if v not in BACKENDS:
            raise ValueError(f"unknown backend {v!r}; expected one of {BACKENDS}")
        return v

    @model_validator(mode="after")
    def _status_consistency(self) -> "PredictionRecord":
        if self.status == "ok" and self.predicted_failure_mode is None:
            raise ValueError("status 'ok' requires a predicted_failure_mode")
        if self.status != "ok" and (
            self.predicted_failure_mode is not None or self.predicted_subtask is not None
        ):
            raise ValueError(f"status {self.status!r} must have null predicted mode and subtask")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (self.pair_id, self.model_variant)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InvalidLine:
    """A line that failed validation. ``pair_id``/``model_variant`` are set when readable."""

    source: str
    line_no: int
    message: str
    pair_id: Optional[str] = None
    model_variant: Optional[str] = None

    @property
    def attributable(self) -> bool:
        return self.pair_id is not None and self.model_variant in MODEL_VARIANTS

    @property
    def key(self) -> Optional[tuple[str, str]]:
        return (self.pair_id, self.model_variant) if self.attributable else None  # type: ignore[return-value]


@dataclass
class PredictionFile:
    records: list[PredictionRecord] = field(default_factory=list)
    invalid: list[InvalidLine] = field(default_factory=list)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def _short(exc: ValidationError, limit: int = 3) -> str:
    parts = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "<record>"
        parts.append(f"{loc}: {err['msg']}")
    more = len(exc.errors()) - limit
    return "; ".join(parts) + (f" (+{more} more)" if more > 0 else "")


def parse_prediction_lines(lines: Iterable[str], source: str = "<memory>") -> PredictionFile:
    """Validate JSONL text. Blank lines are ignored; everything else is kept or reported."""
    out = PredictionFile()
    for line_no, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text:
            continue
        try:
            obj = json.loads(text, parse_constant=_reject_constant)
        except ValueError as exc:
            out.invalid.append(InvalidLine(source, line_no, f"not valid JSON: {exc}"))
            continue
        if not isinstance(obj, dict):
            out.invalid.append(InvalidLine(source, line_no, "line is not a JSON object"))
            continue
        try:
            out.records.append(PredictionRecord.model_validate(obj))
        except ValidationError as exc:
            pid, variant = obj.get("pair_id"), obj.get("model_variant")
            out.invalid.append(
                InvalidLine(
                    source,
                    line_no,
                    _short(exc),
                    pair_id=pid if isinstance(pid, str) and pid else None,
                    model_variant=variant if isinstance(variant, str) else None,
                )
            )
    return out


def read_predictions(*paths: Union[str, Path]) -> PredictionFile:
    """Read and validate one or more JSONL files into a single ``PredictionFile``."""
    merged = PredictionFile()
    for path in paths:
        p = Path(path)
        with p.open("r", encoding="utf-8") as handle:
            part = parse_prediction_lines(handle, source=str(p))
        merged.records.extend(part.records)
        merged.invalid.extend(part.invalid)
    return merged


def write_predictions(path: Union[str, Path], records: Iterable[Mapping[str, Any]]) -> int:
    """Validate and write records as JSONL; returns the count. Invalid records raise."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as handle:
        for rec in records:
            validated = PredictionRecord.model_validate(dict(rec))
            handle.write(json.dumps(validated.model_dump(mode="json"), allow_nan=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Strict joins
# --------------------------------------------------------------------------- #
@dataclass
class IndexedPredictions:
    """Predictions keyed by ``(pair_id, model_variant)``.

    ``by_key`` holds a ``PredictionRecord`` for valid lines and an
    ``InvalidLine`` for attributable invalid ones. ``unattributable`` lines
    could not be tied to any key and are reported, never dropped.
    """

    by_key: dict[tuple[str, str], Union[PredictionRecord, InvalidLine]]
    unattributable: list[InvalidLine]

    @property
    def variants(self) -> list[str]:
        return sorted({variant for _, variant in self.by_key})


def index_predictions(file: PredictionFile, known_pair_ids: Iterable[str]) -> IndexedPredictions:
    """Join predictions on ``(pair_id, model_variant)``.

    Raises ``ContractError`` for duplicate keys (valid or invalid lines) and for
    pair IDs that are not in ``known_pair_ids``.
    """
    known = set(known_pair_ids)
    by_key: dict[tuple[str, str], Union[PredictionRecord, InvalidLine]] = {}
    unattributable: list[InvalidLine] = []
    duplicates: list[tuple[str, str]] = []
    unknown: set[str] = set()

    entries: list[tuple[tuple[str, str], Union[PredictionRecord, InvalidLine]]] = [
        (r.key, r) for r in file.records
    ]
    for bad in file.invalid:
        if bad.key is None:
            unattributable.append(bad)
        else:
            entries.append((bad.key, bad))

    for key, entry in entries:
        if key[0] not in known:
            unknown.add(key[0])
            continue
        if key in by_key:
            duplicates.append(key)
        else:
            by_key[key] = entry

    problems = []
    if duplicates:
        problems.append("duplicate (pair_id, model_variant): " + ", ".join(map(str, sorted(set(duplicates))[:5])))
    if unknown:
        problems.append("unknown pair_id: " + ", ".join(sorted(unknown)[:5]))
    if problems:
        raise ContractError("; ".join(problems))
    return IndexedPredictions(by_key, unattributable)
