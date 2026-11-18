"""Deterministic divergence scoring for existing paired Behavior Trees.

Alignment is delegated to parser.divergence.align_subtask_samples. This module
never reads scoring labels, mutates the input tree, or invents missing telemetry.
Onset is the earliest *aligned comparison grid point* with a strict threshold
crossing, not necessarily the exact physical onset or an observed CSV row.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import yaml
from pydantic import Field, model_validator

from telemetry_reasoning.parser.divergence import align_subtask_samples
from telemetry_reasoning.schemas import (
    ContractModel, DeterministicDivergence, Nonnegative,
    PairedBehaviorTree, SubtaskBT, TelemetryFeature,
)


class DivergenceConfig(ContractModel):
    """Thresholds have the same physical units as each telemetry feature."""

    thresholds: dict[TelemetryFeature, Nonnegative]

    @model_validator(mode="after")
    def require_every_feature(self) -> "DivergenceConfig":
        missing = set(TelemetryFeature) - set(self.thresholds)
        if missing:
            raise ValueError(
                "divergence thresholds missing: "
                + ", ".join(sorted(feature.value for feature in missing))
            )
        return self


def load_divergence_config(path: str | Path) -> DivergenceConfig:
    """Load a separate YAML document: divergence: {thresholds: {...}}."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or set(raw) != {"divergence"}:
        raise ValueError("divergence config must contain exactly one 'divergence' mapping")
    return DivergenceConfig.model_validate(raw["divergence"])


def compute_subtask_divergence(
    node: SubtaskBT, config: DivergenceConfig,
) -> DeterministicDivergence | None:
    """Compute maximum absolute component deltas and first threshold crossing.

    All features contribute to both summaries. The max for a vector feature is
    the largest absolute component difference over all aligned grid points.
    For quaternions this is a sign-aligned component delta, not a rotation angle.
    Empty failure stages return None (not a numerical zero). A computed zero
    comparison has complete zero feature deltas and no onset timestamp.
    """
    aligned = align_subtask_samples(node)
    if aligned.availability == "not_comparable":
        return None
    if aligned.availability != "comparable" or len(aligned.failure_timestamps) == 0:
        raise ValueError("observed stage has no comparable alignment coordinates")

    first_crossing = np.zeros(len(aligned.failure_timestamps), dtype=bool)
    maximums: dict[TelemetryFeature, float] = {}
    for feature in TelemetryFeature:
        success = aligned.success_values[feature]
        failure = aligned.failure_values[feature]
        if success.shape != failure.shape or success.shape[0] != len(first_crossing):
            raise ValueError(f"invalid aligned shape for {feature.value}")
        difference = np.abs(failure - success)
        if difference.ndim == 1:
            per_point = difference
        else:
            per_point = np.max(difference, axis=1)
        if not np.all(np.isfinite(per_point)):
            raise ValueError(f"nonfinite divergence for {feature.value}")
        maximums[feature] = float(np.max(per_point))
        first_crossing |= per_point > config.thresholds[feature]

    onset = (float(aligned.failure_timestamps[np.flatnonzero(first_crossing)[0]])
             if np.any(first_crossing) else None)
    return DeterministicDivergence(
        first_divergence_timestamp=onset,
        max_abs_deltas=maximums,
    )


def enrich_paired_bt(
    tree: PairedBehaviorTree, config: DivergenceConfig,
) -> PairedBehaviorTree:
    """Return a fully validated deep copy with divergence on observed nodes."""
    enriched = tree.model_copy(deep=True)
    for node in enriched.subtasks:
        computed = compute_subtask_divergence(node, config)
        if computed is None:
            node.deterministic_divergence = None
            node.divergence_status = "not_comparable"
        else:
            node.deterministic_divergence = computed
            node.divergence_status = "computed"
    return PairedBehaviorTree.model_validate(enriched.model_dump(mode="json"))
