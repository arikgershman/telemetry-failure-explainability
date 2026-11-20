"""Atomic full-dataset BT publication from a verified grouped handoff.

Compression selection is an earlier, dev-only operation. This gate applies an
explicit configuration to all splits and never uses labels to choose settings.
It preserves the handoff's certification limitations in the output receipt.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from telemetry_reasoning.config import load_config
from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.parser.divergence_metrics import enrich_paired_bt, load_divergence_config
from telemetry_reasoning.parser.onset_diagnostics import (
    combine_onset_diagnostics, raw_reference_divergence_diagnostics,
)
from telemetry_reasoning.parser.pipeline import parse_dataset
from telemetry_reasoning.parser.tuning import validate_tuning_manifest
from telemetry_reasoning.schemas import PairedBehaviorTree
from telemetry_reasoning.sim.freeze_robot_dataset import (
    task_manifests, verify_dataset, verify_handoff,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def produce_dataset(
    batch: Path, handoff: Path, config_path: Path, divergence_path: Path, output_dir: Path,
    *, selection_path: Path | None = None,
) -> dict[str, Any]:
    """Validate, parse, enrich and diagnose every pair before publishing.

    Require a complete, checksum-valid Track 2 handoff and matching task sidecars.
    Every source is verified before and after processing; any rejection or
    changed source prevents atomic publication. A parser-ready archive with
    pending/failed certification can be processed without relabeling its status.
    Output must be absent, outside source and handoff trees, and is never merged
    with old outputs. BT files are deterministic independent of staging paths.
    """
    batch, handoff, config_path, divergence_path, output_dir = map(
        Path, (batch, handoff, config_path, divergence_path, output_dir))
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("production output directory must be absent")
    for protected in (batch.resolve(), handoff.resolve()):
        destination = output_dir.resolve()
        if destination == protected or protected in destination.parents or destination in protected.parents:
            raise ValueError("production output must be separate from source and handoff trees")
    errors = verify_handoff(handoff)
    if errors:
        raise ValueError("invalid handoff: " + "; ".join(errors))
    manifest_path = handoff / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("verification_scope", {}).get("stored_trace_integrity_and_parser_compatibility") != "passed":
        raise ValueError("handoff must pass stored-trace integrity and parser compatibility")
    errors = verify_dataset(batch, manifest)
    if errors:
        raise ValueError("source verification failed: " + "; ".join(errors))
    project_config, divergence = load_config(config_path), load_divergence_config(divergence_path)
    input_paths = [config_path, divergence_path, *(path for path in handoff.rglob("*") if path.is_file())]
    selection = None
    if selection_path is not None:
        selection_path = Path(selection_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (selection.get("selection_version") != 1
                or selection.get("test_data_used_for_selection") is not False
                or selection.get("production_config_sha256") != _hash(config_path)
                or selection.get("dataset_manifest_sha256") != _hash(manifest_path)
                or set(selection.get("tasks", {})) != set(manifest["metadata_layout"])):
            raise ValueError("selection receipt must bind this config and frozen manifest without test tuning")
        config_hash = hashlib.sha256(json.dumps(project_config.parser.model_dump(mode="json"),
                         sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if any(item.get("split") != "dev" or item.get("config_sha256") != config_hash
               for item in selection["tasks"].values()):
            raise ValueError("selection receipt must select this compression configuration on dev only")
        input_paths.append(selection_path)
    initial_hashes = {str(path.resolve()): _hash(path) for path in input_paths}
    report: dict[str, Any] = {
        "production_version": 1, "status": "validated_parser_publication",
        "batch": str(batch.resolve()), "handoff": str(handoff.resolve()),
        "output_directory": str(output_dir.resolve()),
        "release_status": manifest["release_status"],
        "verification_scope": manifest["verification_scope"],
        "source_inventory_sha256": manifest["source_inventory_sha256"],
        "input_sha256": initial_hashes,
        "compression_config": project_config.parser.model_dump(mode="json"),
        "divergence_config": divergence.model_dump(mode="json"),
        "selection": "Explicit fixed configuration; this conversion performs no tuning.",
        "selection_receipt": selection,
        "onset_scope": "Raw-reference numerical grid diagnostics; no physical or injection onset claim.",
        "implementation_sha256": {
            path.name: _hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "tasks": {}, "outputs": {},
    }
    projections = task_manifests(manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".production-bt-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        for task, expected in sorted(projections.items()):
            task_manifest = json.loads((handoff / f"tasks/{task}.json").read_text(encoding="utf-8"))
            if task_manifest != expected:
                raise ValueError(f"task manifest disagrees with frozen release: {task}")
            metadata = read_metadata_csv(batch / manifest["metadata_layout"][task])
            if not metadata or any(row.task.value != task for row in metadata):
                raise ValueError(f"task metadata is empty or mixes tasks: {task}")
            assignments = validate_tuning_manifest(metadata, task_manifest)
            if any(task_manifest["pairs"][row.pair_id][f"{side}_file"] != f"{task}/{getattr(row, f'{side}_file')}"
                   for row in metadata for side in ("success", "failure")):
                raise ValueError(f"task metadata filenames disagree with frozen release: {task}")
            task_dir = staged / task
            raw_dir, enriched_dir = task_dir / "raw-bt", task_dir / "enriched-bt"
            parsed = parse_dataset(metadata, batch / task, project_config.parser, raw_dir,
                                   provenance={"dataset_manifest_sha256": _hash(manifest_path),
                                               "release_status": manifest["release_status"]})
            if parsed.report["counts"]["rejected_pairs"] or len(parsed.files) != len(metadata):
                raise ValueError(f"production conversion rejected source pairs: {task}")
            # Replace temporary publication paths in parse reports with their
            # stable final destinations. These are reports, never prompt inputs.
            parsed.report["output_dir"] = str((output_dir / task / "raw-bt").resolve())
            _json(parsed.report_path, parsed.report)
            enriched_dir.mkdir()
            onset_reports = []
            nodes = Counter()
            for path in parsed.files:
                tree = PairedBehaviorTree.model_validate_json(path.read_bytes())
                enriched = enrich_paired_bt(tree, divergence)
                (enriched_dir / path.name).write_text(enriched.model_dump_json(indent=2) + "\n", encoding="utf-8")
                for node in enriched.subtasks:
                    nodes[node.divergence_status] += 1
                    if (bool(node.failure_samples) != (node.divergence_status == "computed")
                            or (node.divergence_status == "computed") != (node.deterministic_divergence is not None)):
                        raise ValueError(f"invalid enriched comparison states: {task}/{path.name}")
                onset_reports.append({
                    **raw_reference_divergence_diagnostics(
                        read_telemetry_csv(batch / task / tree.metadata.success_file),
                        read_telemetry_csv(batch / task / tree.metadata.failure_file), enriched, divergence),
                    "split": assignments[tree.metadata.pair_id]["split"],
                })
            onset_summary = combine_onset_diagnostics(onset_reports)
            _json(task_dir / "onset_report.json", {"pairs": onset_reports, "summary": onset_summary,
                   "by_split": {split: combine_onset_diagnostics([row for row in onset_reports if row["split"] == split])
                                for split in ("train", "dev", "test")},
                   "note": report["onset_scope"]})
            report["tasks"][task] = {
                "counts": parsed.report["counts"],
                "split_counts": dict(sorted(Counter(row["split"] for row in assignments.values()).items())),
                "divergence_node_counts": dict(sorted(nodes.items())),
                "failure_coverage": parsed.report["failure_coverage"],
                "compression_summary": parsed.report["summary"], "onset_summary": onset_summary,
            }
        errors = verify_dataset(batch, manifest) + verify_handoff(handoff)
        if errors or any(_hash(path) != initial_hashes[str(path.resolve())] for path in input_paths):
            raise ValueError("source, handoff, or configuration changed during production conversion")
        if report["implementation_sha256"] != {path.name: _hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}:
            raise ValueError("parser implementation changed during production conversion")
        report["sources_unchanged"] = True
        report["pair_count"] = sum(task["counts"]["written_pairs"] for task in report["tasks"].values())
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                report["outputs"][path.relative_to(staged).as_posix()] = {
                    "sha256": _hash(path), "size_bytes": path.stat().st_size}
        _json(staged / "production_report.json", report)
        staged.rename(output_dir)
    return report
