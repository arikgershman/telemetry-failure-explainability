"""Within-subtask alignment API for later deterministic divergence scoring.

Alignment uses the sorted union of normalized retained timestamps. It removes
pure duration differences: duration is not a v1 divergence feature. Continuous
features use linear interpolation, gripper closure uses the previous value, and
quaternions use normalized, sign-consistent shortest-path SLERP. These arrays
are comparison values, not additional telemetry observations or BT samples.
Interpolation queries within 1e-12 progress of a retained sample snap to that
sample, avoiding false gripper transitions caused by timestamp roundoff. The
returned grid itself remains the sorted union without rounding.

For a singleton segment, its only sample has progress zero and is held constant
over the other trace's grid. Its absolute timestamp is constant on that grid.
Two singletons therefore compare once, at progress zero. No alignment crosses
a subtask boundary. This module does not yet calculate divergence metrics.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from telemetry_reasoning.schemas import (
    RawTelemetryRow, Subtask, SubtaskBT, TelemetryFeature,
    validate_observation_sequence,
)

QUATERNION_FEATURES = {TelemetryFeature.EE_QUAT, TelemetryFeature.OBJ_QUAT}
PROGRESS_ATOL = 1e-12


@dataclass(frozen=True)
class AlignedSubtask:
    """Detached comparison arrays with one row per shared progress value.

    ``progress`` and ``failure_timestamps`` have shape (n,). Each feature array
    has shape (n,) for scalars or (n, components) for vectors, with joint counts
    inherited from the input. Both feature mappings contain all TelemetryFeature
    values. Failure timestamps map the grid back to failure-run absolute seconds;
    interpolated times need not be observed timestamps. Quaternion components
    are w/x/y/z; their differences are dimensionless, not angular distances.
    """

    subtask: Subtask
    progress: np.ndarray
    failure_timestamps: np.ndarray
    success_values: dict[TelemetryFeature, np.ndarray]
    failure_values: dict[TelemetryFeature, np.ndarray]


def _progress(rows: Sequence[RawTelemetryRow]) -> np.ndarray:
    times = np.asarray([row.timestamp for row in rows], dtype=float)
    if len(times) == 1:
        return np.zeros(1)
    return (times - times[0]) / (times[-1] - times[0])


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
    # Equivalent normalized timestamps can differ by a few ulps after shifting
    # a trace's absolute start time. Treat these as exact sample hits, especially
    # for previous-value interpolation where roundoff would create a full jump.
    before = np.clip(np.searchsorted(progress, grid, side="right") - 1, 0, len(progress) - 1)
    after = np.minimum(before + 1, len(progress) - 1)
    nearest = np.where(np.abs(grid - progress[before]) <= np.abs(grid - progress[after]),
                       before, after)
    queries = np.where(np.abs(grid - progress[nearest]) <= PROGRESS_ATOL,
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
    Empty, unordered, mixed-subtask, mismatched-joint, or zero-quaternion inputs
    raise ValueError. Feature interpolation is limited to this node's endpoints.
    The exact floating-point union is sorted and deduplicated without rounding.
    """
    for rows in (node.success_samples, node.failure_samples):
        validate_observation_sequence(rows)
        if any(row.subtask != node.subtask for row in rows):
            raise ValueError("alignment requires samples from one matching subtask")
    if len(node.success_samples[0].joint_positions) != len(node.failure_samples[0].joint_positions):
        raise ValueError("paired runs must use the same joint dimensions")
    success_progress = _progress(node.success_samples)
    failure_progress = _progress(node.failure_samples)
    grid = np.union1d(success_progress, failure_progress)
    success = _interpolate_features(node.success_samples, success_progress, grid)
    failure = _interpolate_features(node.failure_samples, failure_progress, grid)
    for feature in QUATERNION_FEATURES:
        opposite = np.sum(success[feature] * failure[feature], axis=1) < 0
        failure[feature][opposite] *= -1
    start = node.failure_samples[0].timestamp
    duration = node.failure_samples[-1].timestamp - start
    return AlignedSubtask(
        subtask=node.subtask,
        progress=grid,
        failure_timestamps=start + grid * duration,
        success_values=success,
        failure_values=failure,
    )
