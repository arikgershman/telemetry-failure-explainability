"""Deterministic scoring of failure-explanation predictions.

These metrics are proposed operational choices, not measures fixed by the proposal.

Definitions (per model variant, overall and per task)
    N                      ground-truth pairs in scope. Every one is scored.
    outcome                ok | abstain | error (model-reported), invalid
                           (record present but failed validation), missing
    coverage               ok / N;  abstain/error/invalid/missing rates likewise
    failure_mode_accuracy  correct predicted mode / N. Abstain, error, invalid
                           and missing count as incorrect.
    failed_subtask_accuracy  predicted_subtask == failed_subtask, / N
    joint_accuracy         both of the above
    per-class P/R/F1       F1 = 2TP / (2TP + FP + FN). FN includes abstentions
                           and missing predictions, so they are "missed true
                           classes". Abstentions are not predictions, so they
                           are never false positives.
    macro_f1               mean F1 over labels with support > 0
    macro_f1_fixed_labels  mean F1 over all fixed labels; a label with no
                           support and no predictions scores 0.0 (so absent
                           classes pull it down; prefer ``macro_f1`` on small sets)
    evidence               cited evidence is looked up in the failure-only
                           input and compared numerically within tolerance
    onset                  optional; needs a manifest injection timestamp AND a
                           defined prediction timestamp. Injection time,
                           observed divergence, and detection are different events.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from telemetry_reasoning.eval.contracts import (
    FAILURE_MODES,
    TASK_SUBTASKS,
    TASKS,
    ContractError,
    IndexedPredictions,
    InvalidLine,
    PredictionRecord,
)

DEFAULT_EVAL_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "eval.yaml"

NON_PREDICTION_COLUMNS = ("abstain", "error", "invalid", "missing")
OUTCOME_KINDS = ("ok", "abstain", "error", "invalid", "missing")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class EvidenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp_tol_s: float = Field(0.005, gt=0.0)
    value_atol: float = Field(1e-3, ge=0.0)
    value_rtol: float = Field(1e-2, ge=0.0)


class OnsetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    timestamp_rule: Literal["latest_evidence_in_predicted_subtask"] = "latest_evidence_in_predicted_subtask"


class RubricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n_pairs: int = Field(10, ge=0)
    seed: int = Field(723, ge=0)
    exclude_backends: tuple[str, ...] = ("stub",)


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dpi: int = Field(150, ge=50)


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_variants: tuple[str, ...] = ()
    min_n_for_reporting: int = Field(30, ge=1)
    evidence: EvidenceConfig = EvidenceConfig()
    onset: OnsetConfig = OnsetConfig()
    rubric: RubricConfig = RubricConfig()
    plots: PlotConfig = PlotConfig()


def load_eval_config(path: Union[str, Path, None] = None) -> EvalConfig:
    source = Path(path) if path is not None else DEFAULT_EVAL_CONFIG_PATH
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"{source}: expected a mapping with an 'eval' section")
    extra = set(raw) - {"eval"}
    if extra:
        raise ValueError(f"{source}: unknown top-level keys: {sorted(extra)}")
    section = raw.get("eval")
    if not isinstance(section, Mapping):
        raise ValueError(f"{source}: missing 'eval' mapping")
    return EvalConfig.model_validate(dict(section))


# --------------------------------------------------------------------------- #
# Ground truth and the Track 2 sidecar manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TruthRecord:
    pair_id: str
    task: str
    failed_subtask: str
    failure_mode: str


def _field(row: Any, name: str) -> Any:
    value = row[name] if isinstance(row, Mapping) else getattr(row, name)
    return getattr(value, "value", value)  # enums -> their string value


def truth_from_rows(rows: Iterable[Any]) -> list[TruthRecord]:
    """Convert ``MetadataRow`` objects (or mappings) into validated truth records."""
    out: list[TruthRecord] = []
    seen: set[str] = set()
    for row in rows:
        rec = TruthRecord(
            pair_id=str(_field(row, "pair_id")),
            task=str(_field(row, "task")),
            failed_subtask=str(_field(row, "failed_subtask")),
            failure_mode=str(_field(row, "failure_mode")),
        )
        if rec.task not in TASKS:
            raise ContractError(f"pair {rec.pair_id}: unknown task {rec.task!r}")
        if rec.failure_mode not in FAILURE_MODES:
            raise ContractError(f"pair {rec.pair_id}: unknown failure_mode {rec.failure_mode!r}")
        if rec.failed_subtask not in TASK_SUBTASKS[rec.task]:
            raise ContractError(
                f"pair {rec.pair_id}: failed_subtask {rec.failed_subtask!r} is not in the {rec.task} template"
            )
        if rec.pair_id in seen:
            raise ContractError(f"duplicate pair_id in metadata: {rec.pair_id}")
        seen.add(rec.pair_id)
        out.append(rec)
    return out


@dataclass(frozen=True)
class Manifest:
    splits: Mapping[str, str]
    injection_times: Mapping[str, float]


def load_manifest(path: Union[str, Path]) -> Manifest:
    """Read Track 2's sidecar ``manifest.json``.

    ASSUMED LAYOUT (adjust here if Track 2 differs)::

        {"pairs": {"000": {"split": "test", "injection_timestamp": 5.1, ...}}}

    ``pairs`` may also be a list of objects carrying ``pair_id``. Fields other
    than ``split`` and ``injection_timestamp`` are ignored.
    """
    data = json.loads(Path(path).read_text())
    pairs = data.get("pairs") if isinstance(data, Mapping) else None
    if isinstance(pairs, list):
        pairs = {str(item.get("pair_id")): item for item in pairs if isinstance(item, Mapping)}
    if not isinstance(pairs, Mapping):
        raise ContractError(f"{path}: expected a 'pairs' mapping (or list with pair_id)")
    splits: dict[str, str] = {}
    times: dict[str, float] = {}
    for pid, info in pairs.items():
        if not isinstance(info, Mapping):
            raise ContractError(f"{path}: pair {pid!r} must map to an object")
        if "split" in info:
            splits[str(pid)] = str(info["split"])
        if info.get("injection_timestamp") is not None:
            ts = info["injection_timestamp"]
            if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts) or ts < 0:
                raise ContractError(f"{path}: pair {pid!r} has an invalid injection_timestamp")
            times[str(pid)] = float(ts)
    return Manifest(splits, times)


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Outcome:
    kind: str  # one of OUTCOME_KINDS
    mode: Optional[str] = None
    subtask: Optional[str] = None
    record: Optional[PredictionRecord] = None


def _outcome(entry: Union[PredictionRecord, InvalidLine, None]) -> Outcome:
    if entry is None:
        return Outcome("missing")
    if isinstance(entry, InvalidLine):
        return Outcome("invalid")
    if entry.status == "ok":
        return Outcome("ok", entry.predicted_failure_mode, entry.predicted_subtask, entry)
    return Outcome(entry.status, record=entry)


def _rate(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def _slice_metrics(pairs: Sequence[tuple[TruthRecord, Outcome]]) -> dict[str, Any]:
    n = len(pairs)
    kinds = {k: sum(1 for _, o in pairs if o.kind == k) for k in OUTCOME_KINDS}
    mode_ok = sum(1 for t, o in pairs if o.kind == "ok" and o.mode == t.failure_mode)
    sub_ok = sum(1 for t, o in pairs if o.kind == "ok" and o.subtask == t.failed_subtask)
    both_ok = sum(
        1
        for t, o in pairs
        if o.kind == "ok" and o.mode == t.failure_mode and o.subtask == t.failed_subtask
    )

    per_class: dict[str, dict[str, Any]] = {}
    for label in FAILURE_MODES:
        support = sum(1 for t, _ in pairs if t.failure_mode == label)
        predicted = sum(1 for _, o in pairs if o.kind == "ok" and o.mode == label)
        tp = sum(1 for t, o in pairs if t.failure_mode == label and o.kind == "ok" and o.mode == label)
        fp, fn = predicted - tp, support - tp
        per_class[label] = {
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "precision": _rate(tp, predicted),
            "recall": _rate(tp, support),
            "f1": _rate(2 * tp, 2 * tp + fp + fn),
        }

    present = [lab for lab in FAILURE_MODES if per_class[lab]["support"] > 0]
    macro = statistics.fmean(per_class[lab]["f1"] for lab in present) if present else None
    macro_fixed = (
        statistics.fmean((per_class[lab]["f1"] or 0.0) for lab in FAILURE_MODES) if n else None
    )

    columns = list(FAILURE_MODES) + list(NON_PREDICTION_COLUMNS)
    matrix = [[0] * len(columns) for _ in FAILURE_MODES]
    for t, o in pairs:
        col = o.mode if o.kind == "ok" else o.kind
        matrix[FAILURE_MODES.index(t.failure_mode)][columns.index(col)] += 1

    return {
        "n": n,
        "n_ok": kinds["ok"],
        "n_abstain": kinds["abstain"],
        "n_error": kinds["error"],
        "n_invalid": kinds["invalid"],
        "n_missing": kinds["missing"],
        "coverage": _rate(kinds["ok"], n),
        "abstain_rate": _rate(kinds["abstain"], n),
        "error_rate": _rate(kinds["error"], n),
        "invalid_rate": _rate(kinds["invalid"], n),
        "missing_rate": _rate(kinds["missing"], n),
        "failure_mode_accuracy": _rate(mode_ok, n),
        "failed_subtask_accuracy": _rate(sub_ok, n),
        "joint_accuracy": _rate(both_ok, n),
        "macro_f1": macro,
        "macro_f1_fixed_labels": macro_fixed,
        "labels_absent": [lab for lab in FAILURE_MODES if per_class[lab]["support"] == 0],
        "per_class": per_class,
        "confusion": {"labels": list(FAILURE_MODES), "columns": columns, "matrix": matrix},
    }


# --------------------------------------------------------------------------- #
# Evidence verification
# --------------------------------------------------------------------------- #
def index_view(view: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    """Map subtask -> chronological retained failure samples of a failure-only view."""
    out: dict[str, list[Mapping[str, Any]]] = {}
    for node in view.get("subtasks", []):
        rows = [r for r in node.get("failure_samples", []) if isinstance(r, Mapping)]
        out[node.get("subtask")] = sorted(rows, key=lambda r: r["timestamp"])
    return out


def _values_close(cited: Any, observed: Any, cfg: EvidenceConfig) -> bool:
    a = list(cited) if isinstance(cited, (list, tuple)) else [cited]
    b = list(observed) if isinstance(observed, (list, tuple)) else [observed]
    return len(a) == len(b) and all(
        math.isclose(float(x), float(y), rel_tol=cfg.value_rtol, abs_tol=cfg.value_atol)
        for x, y in zip(a, b)
    )


def verify_evidence_item(
    item: Any, rows_by_subtask: Mapping[str, Sequence[Mapping[str, Any]]], cfg: EvidenceConfig
) -> Optional[str]:
    """Return ``None`` if supported, else the reason it is unsupported."""
    rows = rows_by_subtask.get(item.subtask)
    if not rows:
        return "subtask_not_in_input"
    nearest = min(rows, key=lambda r: abs(r["timestamp"] - item.timestamp))
    if abs(nearest["timestamp"] - item.timestamp) > cfg.timestamp_tol_s:
        return "no_sample_at_timestamp"
    if item.feature not in nearest:
        return "feature_not_in_sample"
    if not _values_close(item.value, nearest[item.feature], cfg):
        return "value_mismatch"
    return None


def _evidence_report(
    outcomes: Sequence[tuple[TruthRecord, Outcome]],
    views: Mapping[str, Mapping[str, Any]],
    cfg: EvidenceConfig,
    detail_limit: int = 100,
) -> dict[str, Any]:
    indexes: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    n_items = n_supported = n_unverifiable = 0
    by_reason: dict[str, int] = {}
    records_with_evidence = records_with_unsupported = ok_without_evidence = 0
    details: list[dict[str, Any]] = []
    for truth, outcome in outcomes:
        rec = outcome.record
        if rec is None:
            continue
        if outcome.kind == "ok" and not rec.evidence:
            ok_without_evidence += 1
        if not rec.evidence:
            continue
        records_with_evidence += 1
        view = views.get(truth.pair_id)
        if view is None:
            n_items += len(rec.evidence)
            n_unverifiable += len(rec.evidence)
            continue
        rows_by_subtask = indexes.setdefault(truth.pair_id, index_view(view))
        bad = 0
        for item in rec.evidence:
            n_items += 1
            reason = verify_evidence_item(item, rows_by_subtask, cfg)
            if reason is None:
                n_supported += 1
                continue
            bad += 1
            by_reason[reason] = by_reason.get(reason, 0) + 1
            if len(details) < detail_limit:
                details.append(
                    {
                        "pair_id": truth.pair_id,
                        "subtask": item.subtask,
                        "timestamp": item.timestamp,
                        "feature": item.feature,
                        "reason": reason,
                    }
                )
        records_with_unsupported += bool(bad)
    n_unsupported = sum(by_reason.values())
    return {
        "checked": True,
        "n_records_with_evidence": records_with_evidence,
        "n_ok_without_evidence": ok_without_evidence,
        "n_items": n_items,
        "n_supported": n_supported,
        "n_unsupported": n_unsupported,
        "n_unverifiable": n_unverifiable,
        "unsupported_by_reason": by_reason,
        "support_rate": _rate(n_supported, n_items - n_unverifiable),
        "n_records_with_unsupported": records_with_unsupported,
        "details": details,
    }


# --------------------------------------------------------------------------- #
# Onset (optional)
# --------------------------------------------------------------------------- #
def predicted_onset(record: PredictionRecord, rule: str = "latest_evidence_in_predicted_subtask") -> Optional[float]:
    """The defined prediction timestamp: latest evidence timestamp inside the predicted subtask.

    ``None`` unless the record is ``ok`` with a predicted subtask and at least
    one evidence item in it. This is a proxy chosen for comparability across
    variants, not a model-reported onset.
    """
    if rule != "latest_evidence_in_predicted_subtask":
        raise ValueError(f"unsupported onset rule {rule!r}")
    if record.status != "ok" or record.predicted_subtask is None:
        return None
    stamps = [e.timestamp for e in record.evidence if e.subtask == record.predicted_subtask]
    return max(stamps) if stamps else None


def _onset_report(
    outcomes: Sequence[tuple[TruthRecord, Outcome]],
    injection_times: Mapping[str, float],
    cfg: OnsetConfig,
) -> dict[str, Any]:
    with_ref = [(t, o) for t, o in outcomes if t.pair_id in injection_times]
    errors: list[float] = []
    for truth, outcome in with_ref:
        if outcome.kind != "ok" or outcome.record is None or outcome.subtask != truth.failed_subtask:
            continue  # onset is only meaningful for a correctly localized prediction
        pred = predicted_onset(outcome.record, cfg.timestamp_rule)
        if pred is not None:
            errors.append(pred - injection_times[truth.pair_id])
    return {
        "computed": True,
        "timestamp_rule": cfg.timestamp_rule,
        "n_with_reference": len(with_ref),
        "n_scored": len(errors),
        "mean_error_s": statistics.fmean(errors) if errors else None,
        "mean_abs_error_s": statistics.fmean(abs(e) for e in errors) if errors else None,
        "median_abs_error_s": statistics.median(abs(e) for e in errors) if errors else None,
        "note": "error = predicted timestamp - manifest injection time; a detection latency, not a divergence time",
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    truth: Sequence[TruthRecord],
    indexed: IndexedPredictions,
    cfg: EvalConfig,
    *,
    variants: Optional[Sequence[str]] = None,
    views: Optional[Mapping[str, Mapping[str, Any]]] = None,
    injection_times: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Score every variant against ``truth``. All truth pairs are scored for every variant."""
    in_scope = {t.pair_id for t in truth}
    chosen = list(variants) if variants else sorted(set(cfg.expected_variants) | set(indexed.variants))
    tasks_present = [task for task in TASKS if any(t.task == task for t in truth)]

    result: dict[str, Any] = {
        "n_truth_pairs": len(truth),
        "labels": list(FAILURE_MODES),
        "variants": {},
        "predictions_outside_scope": sum(1 for pid, _ in indexed.by_key if pid not in in_scope),
        "unattributable_invalid_lines": {
            "count": len(indexed.unattributable),
            "examples": [
                {"source": b.source, "line": b.line_no, "message": b.message} for b in indexed.unattributable[:20]
            ],
        },
        "warnings": [],
    }

    for variant in chosen:
        outcomes = [(t, _outcome(indexed.by_key.get((t.pair_id, variant)))) for t in truth]
        backends = sorted({o.record.backend for _, o in outcomes if o.record is not None})
        slices = {"all": _slice_metrics(outcomes)}
        for task in tasks_present:
            slices[task] = _slice_metrics([(t, o) for t, o in outcomes if t.task == task])

        entry: dict[str, Any] = {
            "backends": backends,
            "synthetic": "stub" in backends,
            "slices": slices,
            "evidence": _evidence_report(outcomes, views, cfg.evidence) if views is not None else {"checked": False},
            "onset": (
                _onset_report(outcomes, injection_times, cfg.onset)
                if (cfg.onset.enabled and injection_times)
                else {"computed": False}
            ),
        }
        result["variants"][variant] = entry
        for name, sl in slices.items():
            if sl["n"] < cfg.min_n_for_reporting:
                result["warnings"].append(
                    f"{variant}/{name}: n={sl['n']} is below {cfg.min_n_for_reporting}; "
                    "treat as a plumbing check, not an accuracy estimate"
                )
        if entry["synthetic"]:
            result["warnings"].append(f"{variant}: stub backend present; results are SYNTHETIC, not model output")
    if result["unattributable_invalid_lines"]["count"]:
        result["warnings"].append(
            f"{result['unattributable_invalid_lines']['count']} invalid prediction line(s) could not be tied to a pair"
        )
    return result


# --------------------------------------------------------------------------- #
# Flat reports
# --------------------------------------------------------------------------- #
_CSV_COLUMNS = (
    "n",
    "n_ok",
    "n_abstain",
    "n_error",
    "n_invalid",
    "n_missing",
    "coverage",
    "abstain_rate",
    "error_rate",
    "invalid_rate",
    "missing_rate",
    "failure_mode_accuracy",
    "failed_subtask_accuracy",
    "joint_accuracy",
    "macro_f1",
    "macro_f1_fixed_labels",
)


def metrics_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per (variant, slice) for ``metrics.csv``."""
    rows = []
    for variant, entry in result["variants"].items():
        for name, sl in entry["slices"].items():
            row = {"model_variant": variant, "task": name, "synthetic": entry["synthetic"]}
            row.update({col: sl[col] for col in _CSV_COLUMNS})
            ev = entry["evidence"]
            row["evidence_support_rate"] = ev.get("support_rate") if name == "all" and ev.get("checked") else None
            rows.append(row)
    return rows


def confusion_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Long-format confusion counts for ``confusion.csv``."""
    rows = []
    for variant, entry in result["variants"].items():
        for name, sl in entry["slices"].items():
            conf = sl["confusion"]
            for i, true_label in enumerate(conf["labels"]):
                for j, column in enumerate(conf["columns"]):
                    rows.append(
                        {
                            "model_variant": variant,
                            "task": name,
                            "true_label": true_label,
                            "predicted": column,
                            "count": conf["matrix"][i][j],
                        }
                    )
    return rows


def _pct(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def format_summary(result: Mapping[str, Any]) -> str:
    lines = [f"Evaluated {result['n_truth_pairs']} ground-truth pair(s)."]
    for variant, entry in result["variants"].items():
        banner = "  [SYNTHETIC: stub backend, not model output]" if entry["synthetic"] else ""
        lines.append(f"\n{variant}{banner}")
        lines.append(f"  {'slice':<18}{'n':>5}{'mode acc':>10}{'macroF1':>9}{'subtask':>9}{'coverage':>10}{'abstain':>9}")
        for name, sl in entry["slices"].items():
            lines.append(
                f"  {name:<18}{sl['n']:>5}{_pct(sl['failure_mode_accuracy']):>10}{_pct(sl['macro_f1']):>9}"
                f"{_pct(sl['failed_subtask_accuracy']):>9}{_pct(sl['coverage']):>10}{_pct(sl['abstain_rate']):>9}"
            )
        ev = entry["evidence"]
        if ev.get("checked"):
            lines.append(
                f"  evidence: {ev['n_supported']}/{ev['n_items'] - ev['n_unverifiable']} items supported"
                f" ({ev['n_unsupported']} unsupported, {ev['n_unverifiable']} unverifiable)"
            )
    for warning in result["warnings"]:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Blinded human rubric
# --------------------------------------------------------------------------- #
RUBRIC_COLUMNS = (
    "item_id",
    "task",
    "reference_failure_mode",
    "reference_failed_subtask",
    "explanation",
    "n_cited_evidence",
    "causal_correctness_1to5",
    "evidence_support_1to5",
    "clarity_1to5",
    "notes",
)


def build_blinded_rubric(
    truth: Sequence[TruthRecord],
    indexed: IndexedPredictions,
    variants: Sequence[str],
    cfg: RubricConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Sample pairs (seeded), include every eligible variant's explanation, hide variant/backend.

    Returns ``(rows, key)``. Only ``ok`` records from non-excluded backends are
    eligible, so stub output never reaches human raters. ``key`` maps each
    ``item_id`` back to ``(pair_id, model_variant, backend)`` and must be kept
    away from raters until scoring is complete.
    """

    def eligible(pair_id: str, variant: str) -> Optional[PredictionRecord]:
        entry = indexed.by_key.get((pair_id, variant))
        if isinstance(entry, PredictionRecord) and entry.status == "ok" and entry.backend not in cfg.exclude_backends:
            return entry
        return None

    by_id = {t.pair_id: t for t in truth}
    pool = sorted(pid for pid in by_id if any(eligible(pid, v) for v in variants))
    rng = random.Random(cfg.seed)
    chosen = rng.sample(pool, min(cfg.n_pairs, len(pool)))
    items = [(pid, v, eligible(pid, v)) for pid in sorted(chosen) for v in sorted(variants) if eligible(pid, v)]
    rng.shuffle(items)

    rows, key = [], []
    for i, (pid, variant, rec) in enumerate(items, start=1):
        item_id = f"R{i:03d}"
        t = by_id[pid]
        rows.append(
            {
                "item_id": item_id,
                "task": t.task,
                "reference_failure_mode": t.failure_mode,
                "reference_failed_subtask": t.failed_subtask,
                "explanation": rec.explanation,
                "n_cited_evidence": len(rec.evidence),
                "causal_correctness_1to5": "",
                "evidence_support_1to5": "",
                "clarity_1to5": "",
                "notes": "",
            }
        )
        key.append({"item_id": item_id, "pair_id": pid, "model_variant": variant, "backend": rec.backend})
    return rows, key
