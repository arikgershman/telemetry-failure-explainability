"""Within-subtask alignment API for later deterministic divergence scoring.

Stages followed by a transition use the sorted union of normalized retained
timestamps for a duration-independent comparison; a transition establishes a
segment boundary, not goal achievement. The terminal observed stage uses
stage-relative elapsed seconds on the common observed interval. Its endpoint does not stand
for stage completion. No terminal alignment extrapolates either trajectory,
including singleton segments, which share only elapsed time zero. An absent
failure stage returns an explicit unavailable comparison with empty arrays.

Continuous features use linear interpolation, gripper closure uses the previous
value, and quaternions use normalized, sign-consistent shortest-path SLERP.
These arrays are comparison values, not additional telemetry observations or BT
samples. Queries within 1e-12 coordinate units of a retained sample snap to it
to avoid false gripper transitions from timestamp roundoff. The returned grid
itself remains the sorted union without rounding. In normalized progress mode a
singleton is held constant over the other trace's grid; its absolute timestamp
stays constant. No alignment crosses a subtask boundary. This module does not
yet calculate divergence metrics.
"""

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from telemetry_reasoning.schemas import (
    RawTelemetryRow, Subtask, SubtaskBT, TelemetryFeature,
    validate_observation_sequence,
)

QUATERNION_FEATURES = {TelemetryFeature.EE_QUAT, TelemetryFeature.OBJ_QUAT}
COORDINATE_ATOL = 1e-12


@dataclass(frozen=True)
class AlignedSubtask:
    """Detached arrays with one row per comparison coordinate.

    ``alignment_mode`` identifies the coordinate system: ``progress`` is a
    dimensionless (n,) array only in ``normalized_progress`` mode, whereas
    ``elapsed_seconds`` is a (n,) array only in ``common_elapsed_time`` mode.
    The inactive coordinate is None; elapsed seconds never replace progress
    under the same name. ``common_interval_seconds`` describes the inclusive
    elapsed interval [0, min(success duration, failure duration)] in time mode.
    Durations describe each observed segment, independently of the grid.

    Absolute ``success_timestamps`` and ``failure_timestamps`` have shape (n,).
    Each feature array has shape (n,) for scalars or (n, components) for vectors,
    with joint counts inherited from the input. Both mappings contain every
    TelemetryFeature. Interpolated timestamps need not be observations.
    Quaternion components are w/x/y/z; their differences are dimensionless,
    not angular distances.

    An unobserved failure stage has availability ``not_comparable`` and mode
    ``unavailable``. Both coordinates and the common interval are None, the
    failure duration is None, and all timestamp/feature arrays have zero rows.
    No comparison was computed, including a comparison of the success samples
    against themselves. This result does not change a node's divergence state.
    """

    subtask: Subtask
    progress: np.ndarray | None
    failure_timestamps: np.ndarray
    success_values: dict[TelemetryFeature, np.ndarray]
    failure_values: dict[TelemetryFeature, np.ndarray]
    alignment_mode: Literal["normalized_progress", "common_elapsed_time", "unavailable"] = "normalized_progress"
    availability: Literal["comparable", "not_comparable"] = "comparable"
    elapsed_seconds: np.ndarray | None = None
    common_interval_seconds: tuple[float, float] | None = None
    success_duration_seconds: float | None = None
    failure_duration_seconds: float | None = None
    success_timestamps: np.ndarray | None = None
    unavailability_reason: str | None = None


def _elapsed(rows: Sequence[RawTelemetryRow]) -> np.ndarray:
    times = np.asarray([row.timestamp for row in rows], dtype=float)
    return times - times[0]


def _progress(rows: Sequence[RawTelemetryRow]) -> np.ndarray:
    elapsed = _elapsed(rows)
    if len(elapsed) == 1:
        return np.zeros(1)
    return elapsed / elapsed[-1]


def _normalized_quaternions(values: np.ndarray) -> np.ndarray:
    # Scale first so normalization also works for large finite components.
    scales = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError("alignment requires nonzero quaternions")
    normalized = values / scales
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    # Canonical first sign makes q and -q representations deterministic, even
    # at an exactly orthogonal (180-degree) shortest-path tie.
    for index, quaternion in enumerate(normalized):
        first_nonzero = quaternion[np.flatnonzero(quaternion)[0]]
        if first_nonzero < 0:
            normalized[index] *= -1
        if index and np.dot(normalized[index - 1], normalized[index]) < 0:
            normalized[index] *= -1
    return normalized


def _interpolate_quaternions(
    progress: np.ndarray, values: np.ndarray, grid: np.ndarray,
) -> np.ndarray:
    values = _normalized_quaternions(values)
    if len(progress) == 1:
        return np.repeat(values, len(grid), axis=0)
    left = np.clip(np.searchsorted(progress, grid, side="right") - 1, 0, len(progress) - 2)
    weight = (grid - progress[left]) / (progress[left + 1] - progress[left])
    start, end = values[left], values[left + 1]
    dot = np.clip(np.sum(start * end, axis=1), 0.0, 1.0)
    # sinc(x) = sin(pi*x)/(pi*x) gives the exact SLERP weights with a stable
    # zero-angle limit. Switching to normalized linear interpolation for small
    # angles would make the result depend on the retained sample density.
    angle_over_pi = np.arccos(dot) / np.pi
    denominator = np.sinc(angle_over_pi)
    start_weight = (1.0 - weight) * np.sinc((1.0 - weight) * angle_over_pi) / denominator
    end_weight = weight * np.sinc(weight * angle_over_pi) / denominator
    result = start_weight[:, None] * start + end_weight[:, None] * end
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _interpolate_features(
    rows: Sequence[RawTelemetryRow], progress: np.ndarray, grid: np.ndarray,
) -> dict[TelemetryFeature, np.ndarray]:
    # Equivalent relative timestamps can differ by a few ulps after shifting
    # a trace's absolute start time. Treat these as exact sample hits, especially
    # for previous-value interpolation where roundoff would create a full jump.
    before = np.clip(np.searchsorted(progress, grid, side="right") - 1, 0, len(progress) - 1)
    after = np.minimum(before + 1, len(progress) - 1)
    nearest = np.where(np.abs(grid - progress[before]) <= np.abs(grid - progress[after]),
                       before, after)
    queries = np.where(np.abs(grid - progress[nearest]) <= COORDINATE_ATOL,
                       progress[nearest], grid)
    aligned = {}
    for feature in TelemetryFeature:
        values = np.asarray([getattr(row, feature.value) for row in rows], dtype=float)
        if feature in QUATERNION_FEATURES:
            aligned[feature] = _interpolate_quaternions(progress, values, queries)
        elif feature == TelemetryFeature.GRIPPER_STATE:
            previous = np.searchsorted(progress, queries, side="right") - 1
            aligned[feature] = values[previous].copy()
        elif values.ndim == 1:
            aligned[feature] = np.interp(queries, progress, values)
        else:
            aligned[feature] = np.column_stack([
                np.interp(queries, progress, values[:, column])
                for column in range(values.shape[1])
            ])
    return aligned


def align_subtask_samples(node: SubtaskBT) -> AlignedSubtask:
    """Align a single node without modifying it or populating its divergence.

    Compressed samples need only have increasing timestamps, not 50 Hz cadence.
    Unordered, mixed-subtask, mismatched-joint, or zero-quaternion comparisons
    raise ValueError. Empty success samples also raise; an unobserved failure
    stage returns an unavailable comparison. Feature interpolation is limited
    to this node's endpoints. The exact floating-point
    union is sorted and deduplicated without rounding. Terminal elapsed-time
    alignment assumes stage-relative controller time is a meaningful coordinate;
    callers must not interpret it as an independently measured physical phase.
    """
    observed = bool(node.failure_samples)
    if node.failure_observation_status != ("observed" if observed else "not_observed"):
        raise ValueError("failure observation status must match its samples")
    if node.failure_segment_end not in (("transitioned", "trace_end") if observed else (None,)):
        raise ValueError("failure segment end must match its observations")
    for field in ("success_samples", "failure_samples"):
        rows = getattr(node, field)
        if field == "failure_samples" and not rows:
            continue
        validate_observation_sequence(rows)
        if any(row.subtask != node.subtask for row in rows):
            raise ValueError("alignment requires samples from one matching subtask")
    success_duration = node.success_samples[-1].timestamp - node.success_samples[0].timestamp
    if not node.failure_samples:
        def empty_features() -> dict[TelemetryFeature, np.ndarray]:
            return {
                feature: np.empty((0,) + np.asarray(getattr(node.success_samples[0], feature.value)).shape)
                for feature in TelemetryFeature
            }

        return AlignedSubtask(
            subtask=node.subtask, progress=None, failure_timestamps=np.empty(0),
            success_values=empty_features(), failure_values=empty_features(),
            alignment_mode="unavailable", availability="not_comparable",
            success_duration_seconds=success_duration, success_timestamps=np.empty(0),
            unavailability_reason="failure stage has no observed samples",
        )
    if len(node.success_samples[0].joint_positions) != len(node.failure_samples[0].joint_positions):
        raise ValueError("paired runs must use the same joint dimensions")
    failure_duration = node.failure_samples[-1].timestamp - node.failure_samples[0].timestamp
    terminal = node.failure_segment_end == "trace_end"
    if terminal:
        success_coordinates = _elapsed(node.success_samples)
        failure_coordinates = _elapsed(node.failure_samples)
        common_end = min(success_duration, failure_duration)
        grid = np.union1d(success_coordinates[success_coordinates <= common_end],
                          failure_coordinates[failure_coordinates <= common_end])
    else:
        success_coordinates = _progress(node.success_samples)
        failure_coordinates = _progress(node.failure_samples)
        grid = np.union1d(success_coordinates, failure_coordinates)
    success = _interpolate_features(node.success_samples, success_coordinates, grid)
    failure = _interpolate_features(node.failure_samples, failure_coordinates, grid)
    for feature in QUATERNION_FEATURES:
        opposite = np.sum(success[feature] * failure[feature], axis=1) < 0
        failure[feature][opposite] *= -1
    return AlignedSubtask(
        subtask=node.subtask,
        progress=None if terminal else grid,
        failure_timestamps=node.failure_samples[0].timestamp + (grid if terminal else grid * failure_duration),
        success_values=success,
        failure_values=failure,
        alignment_mode="common_elapsed_time" if terminal else "normalized_progress",
        elapsed_seconds=grid if terminal else None,
        common_interval_seconds=(0.0, common_end) if terminal else None,
        success_duration_seconds=success_duration,
        failure_duration_seconds=failure_duration,
        success_timestamps=node.success_samples[0].timestamp + (grid if terminal else grid * success_duration),
    )
