"""Within-subtask compression from Section 4 / Figure 1(c) of the proposal."""

from itertools import groupby
from pathlib import Path
from typing import Sequence

import numpy as np

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.io import read_telemetry_csv, validate_raw_run
from telemetry_reasoning.schemas import (
    SUBTASK_METADATA, TASK_GOALS, TASK_SUBTASKS, MetadataRow, PairedBehaviorTree,
    RawTelemetryRow, SubtaskBT,
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

    V1 requires each named subtask once, in template order, in both runs.
    Alignment is structural; no temporal alignment or divergence is computed.
    """
    data_dir = Path(data_dir)
    success = compress_csv(data_dir / metadata.success_file, config)
    failure = compress_csv(data_dir / metadata.failure_file, config)
    expected = TASK_SUBTASKS[metadata.task]
    for name, segments in (("success", success), ("failure", failure)):
        if tuple(segment[0].subtask for segment in segments) != expected:
            raise ValueError(f"{name} run must contain the complete ordered task template")
    return PairedBehaviorTree(
        metadata=metadata,
        overall_goal=TASK_GOALS[metadata.task],
        subtasks=[
            SubtaskBT(
                subtask=subtask,
                metadata=SUBTASK_METADATA[subtask].model_copy(deep=True),
                success_samples=success_rows,
                failure_samples=failure_rows,
            )
            for subtask, success_rows, failure_rows in zip(expected, success, failure)
        ],
    )
