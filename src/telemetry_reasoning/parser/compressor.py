"""Within-subtask compression from Section 4 / Figure 1(c) of the proposal."""

from itertools import groupby
from pathlib import Path
from typing import Sequence

import numpy as np

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.io import read_telemetry_csv, validate_raw_run
from telemetry_reasoning.schemas import (
    SUBTASK_METADATA, TASK_GOALS, TASK_SUBTASKS, MetadataRow, PairedBehaviorTree,
    RawTelemetryRow, SubtaskBT, validate_subtask_sequence,
)


def compress_subtask(
    rows: Sequence[RawTelemetryRow], config: CompressionConfig,
) -> list[RawTelemetryRow]:
    """Retain endpoints, k spaced interior rows, and t-1/t/t+1 at each jump.

    Here t is the later row of the adjacent pair whose change exceeds a
    threshold. For vectors, any component crossing its threshold is an event.
    Neighbors are clipped to this consecutive subtask segment.
    """
    validate_raw_run(rows)
    if any(row.subtask != rows[0].subtask for row in rows):
        raise ValueError("compress_subtask requires one consecutive subtask segment")
    n = len(rows)
    retained = {0, n - 1}
    k = min(config.k_uniform_samples, max(n - 2, 0))
    if k:
        # Partition the span into k+1 equal intervals and choose interior rows.
        retained.update(np.rint(np.linspace(0, n - 1, k + 2)[1:-1]).astype(int).tolist())
    for feature, threshold in config.major_event_thresholds.items():
        values = np.asarray([getattr(row, feature.value) for row in rows], dtype=float)
        changes = np.abs(np.diff(values, axis=0))
        jumps = changes > threshold
        if jumps.ndim > 1:
            jumps = jumps.any(axis=1)
        for t in np.flatnonzero(jumps) + 1:
            retained.update(range(max(0, int(t) - 1), min(n, int(t) + 2)))
    return [rows[index] for index in sorted(retained)]


def compress_rows(
    rows: Sequence[RawTelemetryRow], config: CompressionConfig,
) -> list[list[RawTelemetryRow]]:
    """Group adjacent labels, preserving separate visits to the same subtask."""
    validate_raw_run(rows)
    return [compress_subtask(list(group), config)
            for _, group in groupby(rows, key=lambda row: row.subtask)]


def compress_csv(path: str | Path, config: CompressionConfig) -> list[list[RawTelemetryRow]]:
    return compress_rows(read_telemetry_csv(path), config)


def build_paired_bt(
    metadata: MetadataRow, data_dir: str | Path, config: CompressionConfig,
) -> PairedBehaviorTree:
    """Read a pair relative to data_dir and populate the task's shared template.

    Successes require the complete template; failures may be a nonempty ordered
    prefix. Every pair has explicit coverage states and empty failure sample
    lists for any unobserved suffix. No temporal alignment or divergence is
    computed.
    """
    data_dir = Path(data_dir)
    success = compress_csv(data_dir / metadata.success_file, config)
    failure = compress_csv(data_dir / metadata.failure_file, config)
    return build_bt_from_segments(metadata, success, failure)


def build_bt_from_segments(
    metadata: MetadataRow, success: Sequence[Sequence[RawTelemetryRow]],
    failure: Sequence[Sequence[RawTelemetryRow]],
) -> PairedBehaviorTree:
    """Build the canonical task structure from consecutive compressed segments.

    Segment labels are checked before name-based matching so revisits cannot
    silently overwrite observations. States describe observed coverage only;
    terminal controller outcomes and recording integrity remain Track 2 checks.
    Ground-truth injection fields never determine coverage or stage endings.
    """
    expected = TASK_SUBTASKS[metadata.task]
    for name, segments in (("success", success), ("failure", failure)):
        if any(not segment for segment in segments):
            raise ValueError(f"{name} subtask segments must each contain at least one observation")
        validate_subtask_sequence(
            [segment[0].subtask for segment in segments], metadata.task,
            allow_partial=name == "failure", run_name=name,
        )
    success_by_subtask = {segment[0].subtask: list(segment) for segment in success}
    failure_by_subtask = {segment[0].subtask: list(segment) for segment in failure}
    nodes = []
    for subtask in expected:
        samples = failure_by_subtask.get(subtask, [])
        nodes.append(SubtaskBT(
            subtask=subtask,
            metadata=SUBTASK_METADATA[subtask].model_copy(deep=True),
            success_samples=success_by_subtask[subtask], failure_samples=samples,
            failure_observation_status="observed" if samples else "not_observed",
            failure_segment_end=(
                "trace_end" if subtask == failure[-1][0].subtask else "transitioned"
            ) if samples else None,
            divergence_status="uncomputed" if samples else "not_comparable",
        ))
    return PairedBehaviorTree(
        metadata=metadata,
        overall_goal=TASK_GOALS[metadata.task],
        subtasks=nodes,
    )
