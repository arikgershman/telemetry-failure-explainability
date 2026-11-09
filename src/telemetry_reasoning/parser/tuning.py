"""Deterministic compression sweeps on explicitly grouped train/dev data.

The reference events always use the supplied base configuration's thresholds.
Changing a candidate's thresholds therefore cannot redefine the events that it
must retain. No labels contribute to selection and held-out telemetry is never
opened. A recommendation is descriptive of the supplied split only; it does
not change parser defaults or establish held-out reconstruction performance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.io import read_telemetry_csv
from telemetry_reasoning.parser.compressor import build_paired_bt, compress_rows
from telemetry_reasoning.parser.diagnostics import combine_diagnostics, compression_diagnostics
from telemetry_reasoning.schemas import MetadataRow, PairedBehaviorTree, RawTelemetryRow, SubtaskBT

DEFAULT_K_GRID = (0, 2, 5)
DEFAULT_THRESHOLD_SCALES = (0.5, 1.0, 2.0)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _config_hash(config: CompressionConfig) -> str:
    return hashlib.sha256(_canonical_json(config.model_dump(mode="json"))).hexdigest()


def _identifier(value: Any, field: str) -> str:
    # Treat integer 7 and string "7" as the same initial condition. Booleans
    # must not alias the integer IDs 0 and 1.
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"manifest {field} must be a nonempty string or integer")
    return str(value)


def validate_tuning_manifest(
    metadata: Sequence[MetadataRow], manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Require complete pair coverage and disjoint (task, initial-condition) groups.

    Both evaluator-compatible ``pairs`` shapes are supported: a list carrying
    ``pair_id`` or a mapping keyed by pair ID. Unlike scoring, tuning requires
    an explicit valid split and initial-condition identity for every pair.
    Validation includes test entries before any telemetry is read.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be an object containing 'pairs'")
    rows = {row.pair_id: row for row in metadata}
    if len(rows) != len(metadata):
        raise ValueError("metadata pair_id values must be unique")
    pairs = manifest.get("pairs")
    if isinstance(pairs, list):
        entries = []
        for item in pairs:
            if not isinstance(item, Mapping) or "pair_id" not in item:
                raise ValueError("manifest pairs must contain objects with pair_id")
            entries.append((_identifier(item["pair_id"], "pair_id"), item))
    elif isinstance(pairs, Mapping):
        entries = [(_identifier(pair_id, "pair_id"), item) for pair_id, item in pairs.items()]
    else:
        raise ValueError("manifest requires a 'pairs' mapping or list")

    result: dict[str, dict[str, str]] = {}
    for pair_id, item in entries:
        if pair_id in result:
            raise ValueError(f"manifest duplicate pair_id: {pair_id}")
        if not isinstance(item, Mapping):
            raise ValueError(f"manifest pair {pair_id!r} must be an object")
        if "pair_id" in item and _identifier(item["pair_id"], "pair_id") != pair_id:
            raise ValueError(f"manifest pair_id disagrees with mapping key: {pair_id}")
        split = item.get("split")
        if split not in ("train", "dev", "test"):
            raise ValueError(f"manifest pair {pair_id!r} requires split train, dev, or test")
        if "initial_condition_id" not in item:
            raise ValueError(f"manifest pair {pair_id!r} requires initial_condition_id")
        result[pair_id] = {
            "split": split,
            "initial_condition_id": _identifier(item["initial_condition_id"], "initial_condition_id"),
        }
    missing, unknown = sorted(rows.keys() - result.keys()), sorted(result.keys() - rows.keys())
    if missing or unknown:
        raise ValueError(f"manifest must cover exactly the metadata IDs; missing={missing}, unknown={unknown}")

    sources = dict(entries)
    groups: dict[tuple[str, str], str] = {}
    for pair_id in sorted(result):
        item, row = result[pair_id], rows[pair_id]
        source = sources[pair_id]
        if "task" in source and source["task"] != row.task.value:
            raise ValueError(f"manifest task disagrees with metadata for pair {pair_id!r}")
        group = (row.task.value, item["initial_condition_id"])
        if group in groups and groups[group] != item["split"]:
            raise ValueError(f"initial-condition group {group!r} overlaps splits {groups[group]!r} and {item['split']!r}")
        groups[group] = item["split"]
    return result


def _grid(k_grid: Sequence[int], scales: Sequence[float]) -> tuple[list[int], list[float]]:
    if not k_grid or any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in k_grid):
        raise ValueError("k_grid must contain nonnegative integers")
    if not scales or any(isinstance(s, bool) or not isinstance(s, (int, float))
                         or not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("threshold_scales must contain finite positive numbers")
    return sorted(set(k_grid)), sorted({float(s) for s in scales})


def _candidate_tree(
    base: PairedBehaviorTree, success_rows: Sequence[RawTelemetryRow], failure_rows: Sequence[RawTelemetryRow],
    config: CompressionConfig,
) -> PairedBehaviorTree:
    # The public single-pair builder already validated the complete ordered
    # template. Recompress the cached raw rows rather than rereading files for
    # every grid point; retain semantics from that validated tree.
    success, failure = compress_rows(success_rows, config), compress_rows(failure_rows, config)
    return PairedBehaviorTree(
        metadata=base.metadata, overall_goal=base.overall_goal,
        subtasks=[SubtaskBT(subtask=node.subtask, metadata=node.metadata,
                            success_samples=s_rows, failure_samples=f_rows)
                  for node, s_rows, f_rows in zip(base.subtasks, success, failure)],
    )


def _selection_checks(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, bool]:
    events = candidate["totals"]["event_retention"].values()
    preserved = all(e["retained_event_neighborhoods"] == e["raw_events"] for e in events)
    bounds = base["totals"]["reconstruction_error"]
    reconstruction = candidate["totals"]["reconstruction_error"]
    available = bool(bounds) and bounds.keys() == reconstruction.keys() and all(
        bounds[f]["max_abs_error"] is not None and reconstruction[f]["max_abs_error"] is not None
        and not bounds[f].get("unavailable_stages", 0)
        and not reconstruction[f].get("unavailable_stages", 0) for f in bounds
    )
    # The tiny tolerance permits only floating-point interpolation roundoff;
    # it is recorded in the report rather than presented as an accuracy bound.
    within_bounds = available and all(
        reconstruction[f]["max_abs_error"] <= bounds[f]["max_abs_error"] + 1e-12 for f in bounds
    )
    return {"all_reference_event_neighborhoods_retained": preserved,
            "reconstruction_available": available,
            "within_base_max_reconstruction_bounds": within_bounds,
            "eligible": preserved and within_bounds}


def _compact_pair(pair_id: str, diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    return {"pair_id": pair_id, "totals": diagnostic["totals"],
            "prompt_payload_sizes": diagnostic["prompt_payload_sizes"]}


def sweep_compression(
    metadata: Sequence[MetadataRow], data_dir: str | Path, config: CompressionConfig,
    manifest: Mapping[str, Any], *, split: str = "dev",
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    threshold_scales: Sequence[float] = DEFAULT_THRESHOLD_SCALES,
) -> dict[str, Any]:
    """Report all grid points; recommend the smallest eligible paired payload.

    Reject the whole sweep on invalid selected telemetry instead of silently
    tuning on whichever pairs happened to parse. No test telemetry is opened,
    hashed, reconstructed, or used to choose configurations. Caller-supplied
    metadata objects are only used for task/group joins and building selected
    trees; failure labels are never selection inputs.
    """
    if split not in ("train", "dev"):
        raise ValueError("compression tuning permits only train or dev, never test")
    assignments = validate_tuning_manifest(metadata, manifest)
    ks, scales = _grid(k_grid, threshold_scales)
    selected = sorted((row for row in metadata if assignments[row.pair_id]["split"] == split),
                      key=lambda row: row.pair_id)
    if not selected:
        raise ValueError(f"manifest contains no metadata pairs in requested tuning split {split!r}")
    data_dir = Path(data_dir)
    cached = []
    base_reports, telemetry_hashes = [], []
    for row in selected:
        success_path, failure_path = data_dir / row.success_file, data_dir / row.failure_file
        success, failure = read_telemetry_csv(success_path), read_telemetry_csv(failure_path)
        base_tree = build_paired_bt(row, data_dir, config)
        diagnostic = compression_diagnostics(success, failure, base_tree, config,
                                             reference_thresholds=config.major_event_thresholds)
        cached.append((row.pair_id, success, failure, base_tree, diagnostic))
        base_reports.append(diagnostic)
        telemetry_hashes.append({"pair_id": row.pair_id,
                                 "success_sha256": hashlib.sha256(success_path.read_bytes()).hexdigest(),
                                 "failure_sha256": hashlib.sha256(failure_path.read_bytes()).hexdigest()})

    candidates = []
    for k in ks:
        for scale in scales:
            candidate_config = CompressionConfig(
                k_uniform_samples=k,
                major_event_thresholds={feature: threshold * scale
                                        for feature, threshold in config.major_event_thresholds.items()},
            )
            pair_reports, checks = [], []
            for pair_id, success, failure, base_tree, base_report in cached:
                tree = _candidate_tree(base_tree, success, failure, candidate_config)
                diagnostic = compression_diagnostics(success, failure, tree, candidate_config,
                                                     reference_thresholds=config.major_event_thresholds)
                check = _selection_checks(diagnostic, base_report)
                pair_reports.append(diagnostic)
                checks.append({"pair_id": pair_id, **check})
            summary = combine_diagnostics(pair_reports)
            candidates.append({
                "candidate_id": f"k={k};threshold_scale={scale:g}", "k": k, "threshold_scale": scale,
                "config": candidate_config.model_dump(mode="json"),
                "config_sha256": _config_hash(candidate_config), "summary": summary,
                "pairs": [_compact_pair(item[0], report) for item, report in zip(cached, pair_reports)],
                "selection_checks": checks, "eligible": all(c["eligible"] for c in checks),
            })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    recommended = min(eligible, key=lambda c: (
        c["summary"]["prompt_payload_sizes"]["paired_without_divergence"]["utf8_bytes"],
        c["k"], c["threshold_scale"],
    )) if eligible else None
    implementation_paths = (Path(__file__), Path(__file__).with_name("compressor.py"),
                            Path(__file__).with_name("diagnostics.py"),
                            Path(__file__).with_name("divergence.py"),
                            Path(__file__).parents[1] / "schemas.py",
                            Path(__file__).parents[1] / "io.py",
                            Path(__file__).parents[1] / "config.py")
    return {
        "report_version": "1.0", "split": split,
        "dataset_context": {
            "declared_kind": manifest.get("dataset_kind") if isinstance(manifest.get("dataset_kind"), str) else "unspecified",
            "validation_status": "not established by compression tuning",
        },
        "selected_pair_ids": [row.pair_id for row in selected],
        "manifest_split_counts": dict(sorted(Counter(item["split"] for item in assignments.values()).items())),
        "reference_thresholds": config.model_dump(mode="json")["major_event_thresholds"],
        "grid": {"k_uniform_samples": ks, "threshold_scales": scales},
        "base": {"config": config.model_dump(mode="json"), "config_sha256": _config_hash(config),
                 "summary": combine_diagnostics(base_reports),
                 "pairs": [_compact_pair(item[0], report) for item, report in zip(cached, base_reports)]},
        "source_hashes": {"telemetry": telemetry_hashes,
                          "implementation": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in implementation_paths}},
        "candidates": candidates,
        "recommendation": None if recommended is None else {
            "candidate_id": recommended["candidate_id"], "config": recommended["config"],
            "config_sha256": recommended["config_sha256"],
            "paired_payload_utf8_bytes": recommended["summary"]["prompt_payload_sizes"]["paired_without_divergence"]["utf8_bytes"],
        },
        "selection_policy": {
            "objective": "minimum summed paired_without_divergence payload UTF-8 bytes",
            "constraints": ["all fixed-reference event neighborhoods retained on every selected pair",
                            "all per-feature per-pair maximum reconstruction errors no greater than base"],
            "reconstruction_comparison_absolute_tolerance": 1e-12,
            "tie_break": ["smaller k_uniform_samples", "smaller threshold_scale"],
        },
        "notes": ["Split-local compression diagnostics; no held-out performance claim.",
                  "Payload sizes measure canonical JSON characters and UTF-8 bytes, not model tokens.",
                  "Recommendations never modify parser defaults or write a replacement configuration.",
                  "Dataset validation/finalization remains the responsibility of Track 2."],
    }
