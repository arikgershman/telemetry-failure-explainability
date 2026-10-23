"""Batch construction API, independent of dataset storage and divergence scoring.

Call ``build_paired_bts(read_metadata_csv(path), data_dir, config.parser)``.
The result contains every valid tree and a separate error for each rejected
pair, in input order within each collection. No files are written. Dataset
serialization, compression reports, and explicit divergence enrichment belong
to the later pipeline milestone.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.parser.compressor import build_paired_bt
from telemetry_reasoning.schemas import MetadataRow, PairedBehaviorTree


@dataclass(frozen=True)
class PairBuildError:
    """A rejected pair; source filenames resolve relative to the data directory."""

    pair_id: str
    success_file: str
    failure_file: str
    error_type: str
    message: str


@dataclass(frozen=True)
class BatchBuildResult:
    """Exactly one tree or error per input pair; an empty batch yields both empty."""

    trees: tuple[PairedBehaviorTree, ...]
    errors: tuple[PairBuildError, ...]


def build_paired_bts(
    metadata: Sequence[MetadataRow], data_dir: str | Path, config: CompressionConfig,
) -> BatchBuildResult:
    """Build both task templates using the existing single-pair validation.

    Duplicate IDs reject the whole batch before reading any pair files. Expected
    file/validation failures are recorded per pair while later pairs continue;
    programming errors propagate. Metadata CSV errors are raised by the reader
    before this API is called. All successful trees retain null (uncomputed)
    divergence, matching ``build_paired_bt``. Inputs are never modified.
    """
    if len({row.pair_id for row in metadata}) != len(metadata):
        raise ValueError("metadata pair_id values must be unique")
    trees = []
    errors = []
    for row in metadata:
        try:
            trees.append(build_paired_bt(row, data_dir, config))
        except (OSError, ValueError, csv.Error) as exc:
            errors.append(PairBuildError(
                pair_id=row.pair_id,
                success_file=row.success_file,
                failure_file=row.failure_file,
                error_type=type(exc).__name__,
                message=str(exc),
            ))
    return BatchBuildResult(trees=tuple(trees), errors=tuple(errors))
