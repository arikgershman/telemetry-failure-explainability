"""Compression error against raw paired-trace numerical divergence.

The raw reference uses the same alignment and thresholds as the compressed
comparison. Neither reference is an injection time or verified physical onset.
"""

from collections import Counter
from itertools import groupby
from math import fsum
from typing import Any, Mapping, Sequence

from telemetry_reasoning.io import validate_raw_run
from telemetry_reasoning.parser.compressor import build_bt_from_segments
from telemetry_reasoning.parser.divergence_metrics import (
    DivergenceConfig, compute_subtask_divergence,
)
from telemetry_reasoning.schemas import PairedBehaviorTree, RawTelemetryRow, TelemetryFeature


def raw_reference_divergence_diagnostics(
    success_rows: Sequence[RawTelemetryRow], failure_rows: Sequence[RawTelemetryRow],
    tree: PairedBehaviorTree, config: DivergenceConfig,
) -> dict[str, Any]:
    """Compare raw and compressed grid onsets without reading failure labels.

    Validate that compression preserves original observations and endpoints,
    complete success structure and the observed failure prefix. Missing and
    extra threshold crossings are counted separately from finite timing error;
    null onsets never become zero. Signed error is compressed minus raw time.
    """
    for rows in (success_rows, failure_rows):
        validate_raw_run(rows)
    raw_tree = build_bt_from_segments(
        tree.metadata,
        [list(group) for _, group in groupby(success_rows, key=lambda row: row.subtask)],
        [list(group) for _, group in groupby(failure_rows, key=lambda row: row.subtask)],
    )
    stages = []
    for original, compressed in zip(raw_tree.subtasks, tree.subtasks, strict=True):
        if original.subtask != compressed.subtask:
            raise ValueError("compressed and raw subtask templates disagree")
        if (original.failure_segment_end != compressed.failure_segment_end
                or original.failure_observation_status != compressed.failure_observation_status):
            raise ValueError("compressed stage coverage disagrees with raw trace")
        for side in ("success", "failure"):
            rows, samples = getattr(original, f"{side}_samples"), getattr(compressed, f"{side}_samples")
            if not rows:
                if samples:
                    raise ValueError("compressed unobserved stage contains samples")
                continue
            by_time = {row.timestamp: row for row in rows}
            if (not samples or samples[0] != rows[0] or samples[-1] != rows[-1]
                    or any(by_time.get(sample.timestamp) != sample for sample in samples)):
                raise ValueError("compressed samples must preserve raw observations and endpoints")
        raw = compute_subtask_divergence(original, config)
        estimated = compute_subtask_divergence(compressed, config)
        if raw is None:
            if estimated is not None:
                raise ValueError("compressed observation coverage disagrees with raw trace")
            stages.append({"subtask": original.subtask.value, "comparison": "not_comparable",
                           "raw_timestamp": None, "compressed_timestamp": None,
                           "signed_error_s": None, "absolute_error_s": None,
                           "max_delta_absolute_errors": None})
            continue
        if estimated is None:
            raise ValueError("compressed observed stage is missing")
        reference, estimate = raw.first_divergence_timestamp, estimated.first_divergence_timestamp
        if reference is None:
            comparison = "both_no_crossing" if estimate is None else "extra_compressed_crossing"
        else:
            comparison = "missed_compressed_crossing" if estimate is None else "both_crossing"
        error = estimate - reference if reference is not None and estimate is not None else None
        stages.append({
            "subtask": original.subtask.value, "comparison": comparison,
            "raw_timestamp": reference, "compressed_timestamp": estimate,
            "signed_error_s": error, "absolute_error_s": abs(error) if error is not None else None,
            "max_delta_absolute_errors": {
                feature.value: abs(estimated.max_abs_deltas[feature] - raw.max_abs_deltas[feature])
                for feature in TelemetryFeature
            },
        })
    return {"pair_id": tree.metadata.pair_id, "task": tree.metadata.task.value,
            "stages": stages, "summary": combine_onset_diagnostics([{"stages": stages}]),
            "reference": "raw_trace_aligned_grid_threshold_crossing",
            "note": "Numerical compression error; neither onset is verified physical or injection onset."}


def combine_onset_diagnostics(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine finite timing errors and keep missing/extra comparisons visible."""
    stages = [stage for report in reports for stage in report["stages"]]
    errors = [stage["signed_error_s"] for stage in stages if stage["signed_error_s"] is not None]
    counts = Counter(stage["comparison"] for stage in stages)
    return {
        "stage_count": len(stages),
        "comparison_counts": {key: counts[key] for key in (
            "both_crossing", "both_no_crossing", "missed_compressed_crossing",
            "extra_compressed_crossing", "not_comparable")},
        "timing_error_count": len(errors),
        "mean_signed_error_s": fsum(errors) / len(errors) if errors else None,
        "mean_absolute_error_s": fsum(abs(value) for value in errors) / len(errors) if errors else None,
        "max_absolute_error_s": max(map(abs, errors)) if errors else None,
        "max_delta_absolute_errors": {
            feature.value: max((stage["max_delta_absolute_errors"][feature.value]
                                for stage in stages if stage["max_delta_absolute_errors"] is not None),
                               default=None)
            for feature in TelemetryFeature
        },
        "timing_error_denominator": "only stages where both raw and compressed onsets exist",
    }
