"""Batch construction and deterministic dataset serialization.

Call ``build_paired_bts(read_metadata_csv(path), data_dir, config.parser)``.
The result contains every valid tree and a separate error for each rejected
pair, in input order within each collection. No files are written. Dataset
serialization is available through ``parse_dataset``. Observed-node divergence
remains uncomputed until the explicit metric stage. Unobserved nodes are
explicitly not comparable, rather than computed-zero or pending comparisons.
"""

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Sequence

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.io import read_telemetry_csv
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


@dataclass(frozen=True)
class DatasetParseResult:
    """Committed output inventory and its separate parse/compression report."""

    files: tuple[Path, ...]
    report_path: Path
    report: dict


def build_paired_bts(
    metadata: Sequence[MetadataRow], data_dir: str | Path, config: CompressionConfig,
) -> BatchBuildResult:
    """Build both task templates using the existing single-pair validation.

    Duplicate IDs reject the whole batch before reading any pair files. Expected
    file/validation failures are recorded per pair while later pairs continue;
    programming errors propagate. Metadata CSV errors are raised by the reader
    before this API is called. Failure prefixes retain the full task structure;
    null divergence is uncomputed on observed stages and not comparable on
    unobserved stages. Inputs are never modified.
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


def _json_text(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_dataset(
    metadata: Sequence[MetadataRow], data_dir: str | Path, config: CompressionConfig,
    output_dir: str | Path, *, provenance: dict | None = None,
) -> DatasetParseResult:
    """Write valid ``pair_<id>_bt.json`` files and ``parse_report.json``.

    Invalid pair files, incomplete successes, and non-prefix failure sequences
    are reported and excluded; remaining pairs continue. Nothing is moved,
    repaired, or fabricated in the
    source dataset. Duplicate IDs and unsafe filename IDs reject the batch.
    Output must be absent or empty to prevent downstream glob readers consuming
    stale BTs. A staged directory is published only after all writes succeed;
    configuration/programming/output failures propagate without publishing it.

    Reports include source/output hashes, fixed-reference event-neighborhood
    retention, raw reconstruction errors, and canonical model-view payload
    sizes (not tokenizer counts). Results are deterministic for identical inputs,
    configuration, paths, and provenance; no wall-clock fields are inserted.
    This API does not select splits or claim the dataset is Track 2 validated.
    All pairs use the same explicit coverage and comparison states, including
    terminal-stage state on failures that reached every expected stage.
    """
    # Lazy import keeps the established in-memory API independent of reporting.
    from telemetry_reasoning.parser.diagnostics import (
        combine_diagnostics, compression_diagnostics,
    )

    if len({row.pair_id for row in metadata}) != len(metadata):
        raise ValueError("metadata pair_id values must be unique")
    for row in metadata:
        if any(char in row.pair_id for char in ("/", "\\", "\x00", "\r", "\n")):
            raise ValueError(f"pair_id cannot be used as a flat BT filename: {row.pair_id!r}")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.is_symlink() or (output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    )):
        raise ValueError("output_dir must be absent or empty; choose a fresh directory")

    config_value = config.model_dump(mode="json")
    report = {
        "data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "compression_config": config_value,
        "compression_config_sha256": hashlib.sha256(
            json.dumps(config_value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "provenance": provenance or {},
        "divergence": {
            "status": "not_computed",
            "note": ("Observed-node objects are uncomputed. Unobserved nodes explicitly "
                     "have divergence_status=not_comparable; null never means zero."),
            "node_counts": {"uncomputed": 0, "not_comparable": 0},
        },
        "failure_coverage": {"complete_template_pairs": 0, "partial_prefix_pairs": 0},
        "counts": {"input_pairs": len(metadata), "written_pairs": 0, "rejected_pairs": 0},
        "pairs": [],
        "errors": [],
    }
    diagnostics = []
    names = []
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".parse-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "bt"
        staged.mkdir()
        for row in metadata:
            try:
                sources = {
                    side: {"file": getattr(row, f"{side}_file"),
                           "sha256": _sha256(data_dir / getattr(row, f"{side}_file"))}
                    for side in ("success", "failure")
                }
                tree = build_paired_bt(row, data_dir, config)
                detail = compression_diagnostics(
                    read_telemetry_csv(data_dir / row.success_file),
                    read_telemetry_csv(data_dir / row.failure_file), tree, config,
                )
            except (OSError, ValueError, csv.Error) as exc:
                report["errors"].append(asdict(PairBuildError(
                    pair_id=row.pair_id, success_file=row.success_file,
                    failure_file=row.failure_file, error_type=type(exc).__name__,
                    message=str(exc),
                )))
                continue
            name = f"pair_{row.pair_id}_bt.json"
            payload = tree.model_dump_json(indent=2) + "\n"
            # Writes stay outside the pair-input exception handler: storage
            # errors must fail the run rather than masquerade as bad telemetry.
            (staged / name).write_text(payload, encoding="utf-8")
            names.append(name)
            diagnostics.append(detail)
            observed = [node.subtask.value for node in tree.subtasks if node.failure_samples]
            unobserved = [node.subtask.value for node in tree.subtasks if not node.failure_samples]
            coverage = "partial_prefix_pairs" if unobserved else "complete_template_pairs"
            report["failure_coverage"][coverage] += 1
            report["divergence"]["node_counts"]["uncomputed"] += len(observed)
            report["divergence"]["node_counts"]["not_comparable"] += len(unobserved)
            report["pairs"].append({
                "pair_id": row.pair_id, "task": row.task.value,
                "failure_coverage": {
                    "observed_subtasks": observed, "not_observed_subtasks": unobserved,
                    "last_observed_subtask": observed[-1],
                    "termination_verified": False,
                },
                "output_file": name,
                "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "sources": sources, "diagnostics": detail,
            })
        report["counts"].update(written_pairs=len(names), rejected_pairs=len(report["errors"]))
        report["summary"] = combine_diagnostics(diagnostics)
        (staged / "parse_report.json").write_text(_json_text(report), encoding="utf-8")
        if output_dir.exists():
            output_dir.rmdir()  # fails safely if another writer populated it
        staged.rename(output_dir)
    return DatasetParseResult(
        files=tuple(output_dir / name for name in names),
        report_path=output_dir / "parse_report.json", report=report,
    )
