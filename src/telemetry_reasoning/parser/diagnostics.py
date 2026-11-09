"""Raw-reference compression measurements, separate from paired divergence.

Events use adjacent raw observations within one stage, componentwise strict
``abs(delta) > threshold``. A retained event neighborhood contains every raw
row at clipped indices t-1/t/t+1, where t is the later observation. Reference
thresholds must be fixed across a tuning sweep; candidate compression settings
must not redefine the events against which that candidate is scored.
"""

from collections.abc import Mapping, Sequence
from itertools import groupby
import json
from math import fsum
from typing import Any

import numpy as np

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.io import validate_raw_run
from telemetry_reasoning.parser.divergence import QUATERNION_FEATURES, align_subtask_samples
from telemetry_reasoning.schemas import (
    TASK_SUBTASKS, PairedBehaviorTree, RawTelemetryRow, SubtaskBT, TelemetryFeature,
)


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _row_counts(raw: int, retained: int) -> dict[str, Any]:
    return {
        "raw_rows": raw,
        "retained_rows": retained,
        "retained_fraction": _fraction(retained, raw),
        "compression_ratio": _fraction(raw, retained),
    }


def _event_counts(
    raw_events: int, retained_event_rows: int, retained_event_neighborhoods: int,
    raw_neighbor_rows: int, retained_neighbor_rows: int,
) -> dict[str, Any]:
    return {
        "raw_events": raw_events,
        "retained_event_rows": retained_event_rows,
        "retained_event_neighborhoods": retained_event_neighborhoods,
        "event_row_retention_fraction": _fraction(retained_event_rows, raw_events),
        "event_retention_fraction": _fraction(retained_event_neighborhoods, raw_events),
        "raw_neighbor_rows": raw_neighbor_rows,
        "retained_neighbor_rows": retained_neighbor_rows,
        "neighbor_retention_fraction": _fraction(retained_neighbor_rows, raw_neighbor_rows),
    }


def _event_retention(
    raw: Sequence[RawTelemetryRow], retained_indices: set[int],
    reference_thresholds: Mapping[TelemetryFeature, float],
) -> dict[str, Any]:
    result = {}
    for feature, threshold in reference_thresholds.items():
        values = np.asarray([getattr(row, feature.value) for row in raw], dtype=float)
        with np.errstate(over="ignore"):
            jumps = np.abs(np.diff(values, axis=0)) > threshold
        if jumps.ndim > 1:
            jumps = jumps.any(axis=1)
        events = [int(t) for t in np.flatnonzero(jumps) + 1]
        neighborhoods = [set(range(max(0, t - 1), min(len(raw), t + 2))) for t in events]
        neighbors = set().union(*neighborhoods) if neighborhoods else set()
        result[feature.value] = _event_counts(
            len(events), sum(t in retained_indices for t in events),
            sum(neighbors <= retained_indices for neighbors in neighborhoods),
            len(neighbors), len(neighbors & retained_indices),
        )
    return result


def _reconstruction_error(
    node: SubtaskBT, raw: Sequence[RawTelemetryRow], retained: Sequence[RawTelemetryRow],
) -> dict[str, Any]:
    # Comparison uses the existing alignment semantics, with the retained rows
    # as the reconstructed trace and the raw rows as its reference. The checked
    # endpoints agree, so the union grid is exactly the raw observation grid.
    # Quaternion normalization is undefined at zero. The storage schema permits
    # zero quaternions, so report that feature unavailable without rejecting an
    # otherwise parser-compatible trace or hiding errors in other features.
    unavailable = {
        feature for feature in QUATERNION_FEATURES
        if any(not any(getattr(row, feature.value)) for row in raw)
    }
    if unavailable:
        replacements = {feature.value: [1.0, 0.0, 0.0, 0.0] for feature in unavailable}
        reference_rows = [row.model_copy(update=replacements) for row in raw]
        retained_rows = [row.model_copy(update=replacements) for row in retained]
    else:
        reference_rows, retained_rows = list(raw), list(retained)
    comparison = SubtaskBT(
        subtask=node.subtask, metadata=node.metadata,
        success_samples=retained_rows, failure_samples=reference_rows,
    )
    aligned = align_subtask_samples(comparison)
    result = {}
    for feature in TelemetryFeature:
        if feature in unavailable:
            result[feature.value] = {
                "max_abs_error": None, "mean_abs_error": None,
                "component_observations": 0, "unavailable_stages": 1,
                "unavailable_reason": "raw observations contain a zero quaternion",
            }
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            errors = np.abs(aligned.success_values[feature] - aligned.failure_values[feature])
        if not np.isfinite(errors).all():
            raise ValueError(f"non-finite reconstruction error for {feature.value}")
        result[feature.value] = {
            "max_abs_error": float(errors.max()),
            # Divide before summing to avoid overflow in the sum of finite errors.
            "mean_abs_error": float(np.sum(errors / errors.size)),
            "component_observations": int(errors.size), "unavailable_stages": 0,
        }
    return result


def _aggregate(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = _row_counts(
        sum(report["raw_rows"] for report in reports),
        sum(report["retained_rows"] for report in reports),
    )
    features = sorted({feature for report in reports for feature in report["event_retention"]})
    result["event_retention"] = {}
    count_keys = (
        "raw_events", "retained_event_rows", "retained_event_neighborhoods",
        "raw_neighbor_rows", "retained_neighbor_rows",
    )
    for feature in features:
        counts = [report["event_retention"][feature] for report in reports]
        result["event_retention"][feature] = _event_counts(
            *(sum(count[key] for count in counts) for key in count_keys)
        )
    result["reconstruction_error"] = {}
    for feature in TelemetryFeature:
        metrics = [report["reconstruction_error"][feature.value] for report in reports]
        count = sum(metric["component_observations"] for metric in metrics)
        unavailable = sum(metric["unavailable_stages"] for metric in metrics)
        complete = bool(metrics) and not unavailable
        entry = {
            "max_abs_error": max(metric["max_abs_error"] for metric in metrics) if complete else None,
            "mean_abs_error": fsum(
                metric["mean_abs_error"] * (metric["component_observations"] / count)
                for metric in metrics
            ) if complete else None,
            "component_observations": count, "unavailable_stages": unavailable,
        }
        if unavailable:
            entry["unavailable_reason"] = "one or more stages contain a zero quaternion"
        result["reconstruction_error"][feature.value] = entry
    return result


def _prompt_payload_sizes(tree: PairedBehaviorTree) -> dict[str, dict[str, int]]:
    result = {}
    for name, view in (
        ("failure_only", tree.to_failure_only()),
        ("paired_without_divergence", tree.to_paired_without_divergence()),
        ("full_paired", tree.to_full_paired()),
    ):
        # Opaque join IDs have no predictive role and do not belong in prose.
        view.pop("pair_id", None)
        payload = json.dumps(view, ensure_ascii=False, allow_nan=False,
                             sort_keys=True, separators=(",", ":"))
        result[name] = {"characters": len(payload), "utf8_bytes": len(payload.encode("utf-8"))}
    return result


def compression_diagnostics(
    success_rows: Sequence[RawTelemetryRow], failure_rows: Sequence[RawTelemetryRow],
    tree: PairedBehaviorTree, config: CompressionConfig,
    reference_thresholds: Mapping[TelemetryFeature | str, float] | None = None,
) -> dict[str, Any]:
    """Measure the actual stored samples against their original 50 Hz traces.

    The result is JSON serializable and never changes rows, tree, or settings.
    Both raw runs must satisfy the complete ordered task template and paired
    joint dimensions. Every retained row must equal an original observation;
    stage endpoints must remain present. This permits evaluating a lossy tree
    independently of how it was built, without manufacturing observations.

    Event features are exactly those in ``reference_thresholds`` (or
    ``config.major_event_thresholds`` if omitted). Pass the same explicit map
    for every candidate in a tuning sweep. Empty event denominators are null.
    Ratios are raw/retained; fractions are retained/raw. Reconstruction means
    are weighted over raw scalar components, with quaternion errors measured
    after normalization/sign alignment, not as radians. Prompt payload sizes
    measure compact sorted-key JSON with Unicode preserved and pair_id removed;
    they exclude prompt templates and are not tokenizer/model token counts.
    The full paired view includes any existing divergence objects/nulls, but
    diagnostics do not compute paired divergence or onset.
    """
    thresholds = CompressionConfig(major_event_thresholds=dict(
        config.major_event_thresholds if reference_thresholds is None else reference_thresholds
    )).major_event_thresholds
    validate_raw_run(success_rows)
    validate_raw_run(failure_rows)
    expected = TASK_SUBTASKS[tree.metadata.task]
    if tuple(node.subtask for node in tree.subtasks) != expected:
        raise ValueError("BT subtasks must match the complete ordered task template")
    if len(success_rows[0].joint_positions) != len(failure_rows[0].joint_positions):
        raise ValueError("paired runs must use the same joint dimensions")
    runs = {}
    for name, rows in (("success", success_rows), ("failure", failure_rows)):
        segments = [list(group) for _, group in groupby(rows, key=lambda row: row.subtask)]
        if tuple(segment[0].subtask for segment in segments) != expected:
            raise ValueError(f"{name} run must contain the complete ordered task template")
        stages = []
        for node, raw in zip(tree.subtasks, segments):
            retained = getattr(node, f"{name}_samples")
            if not retained or retained[0] != raw[0] or retained[-1] != raw[-1]:
                raise ValueError(f"{name} {node.subtask.value}: retained stage endpoints must match raw rows")
            by_time = {row.timestamp: (index, row) for index, row in enumerate(raw)}
            indices = []
            for sample in retained:
                original = by_time.get(sample.timestamp)
                if original is None or sample != original[1]:
                    raise ValueError(f"{name} {node.subtask.value}: retained samples must be original raw observations")
                indices.append(original[0])
            if indices != sorted(set(indices)):
                raise ValueError("retained timestamps must be strictly increasing with no duplicates")
            stage = {"subtask": node.subtask.value, **_row_counts(len(raw), len(retained))}
            stage["event_retention"] = _event_retention(raw, set(indices), thresholds)
            stage["reconstruction_error"] = _reconstruction_error(node, raw, retained)
            stages.append(stage)
        runs[name] = {**_aggregate(stages), "stages": stages}
    return {
        "compression_config": config.model_dump(mode="json"),
        "reference_thresholds": {feature.value: threshold for feature, threshold in thresholds.items()},
        "runs": runs, "totals": _aggregate(list(runs.values())),
        "prompt_payload_sizes": _prompt_payload_sizes(tree),
        "definitions": {
            "compression_ratio": "raw_rows / retained_rows",
            "event_retention_fraction": "all clipped t-1/t/t+1 raw rows retained / raw adjacent events",
            "neighbor_retention_fraction": "retained unique event-neighbor rows / unique raw event-neighbor rows",
            "empty_event_denominator": "null (no reference events)",
            "reconstruction": "within-stage alignment against raw observations; linear, previous-value closure, shortest-path SLERP",
            "quaternion_error": "normalized sign-aligned component differences, not radians",
            "prompt_payload_sizes": "compact sorted-key JSON of leakage-free views without pair_id; UTF-8 bytes and Unicode characters, not model tokens; templates excluded",
        },
    }


def combine_diagnostics(diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate pair reports with equal compression and reference settings.

    Row and payload counts are summed; event fractions are recomputed from the
    counts and reconstruction means are weighted by scalar component counts.
    Empty collections have zero counts and null ratios/errors. A quaternion
    metric is null if any contributing stage made that feature unavailable.
    """
    first = diagnostics[0] if diagnostics else None
    if first is not None and any(
        report[key] != first[key] for report in diagnostics
        for key in ("compression_config", "reference_thresholds")
    ):
        raise ValueError("cannot combine diagnostics with different compression/reference settings")
    return {
        "pair_count": len(diagnostics),
        "compression_config": first["compression_config"] if first else None,
        "reference_thresholds": dict(first["reference_thresholds"]) if first else {},
        "totals": _aggregate([report["totals"] for report in diagnostics]),
        "runs": {name: _aggregate([report["runs"][name] for report in diagnostics])
                 for name in ("success", "failure")},
        "prompt_payload_sizes": {
            view: {unit: sum(report["prompt_payload_sizes"][view][unit] for report in diagnostics)
                   for unit in ("characters", "utf8_bytes")}
            for view in ("failure_only", "paired_without_divergence", "full_paired")
        },
        "definitions": dict(first["definitions"]) if first else {},
    }
