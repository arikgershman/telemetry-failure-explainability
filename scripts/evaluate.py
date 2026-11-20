#!/usr/bin/env python
"""Score prediction JSONL files against ground truth.

    python scripts/evaluate.py \\
        --metadata data/sample/metadata.csv \\
        --predictions runs/smoke/baseline_1.jsonl \\
        --config configs/eval.yaml \\
        --output-dir runs/smoke/eval

Ground truth comes from ``metadata.csv`` only (``MetadataRow``). Optional inputs:

    --bt-dir DIR        verify cited evidence against each pair's failure-only input
    --manifest FILE     Track 2 sidecar: split assignments and injection timestamps
    --split NAME        score only pairs assigned to this split (needs --manifest)
    --variants V [V..]  score exactly these variants (default: those present, plus
                        ``expected_variants`` from the config)
    --run-manifest F..  sibling run-manifest JSONs; embedded in metrics.json

Writes ``metrics.json``, ``metrics.csv``, ``confusion.csv``, a blinded human
rubric (``rubric_blinded.csv`` + ``rubric_key.json``) and PNG plots. Duplicate
(pair_id, model_variant) keys and unknown pair IDs are rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from telemetry_reasoning.eval.contracts import (
    MODEL_VARIANTS,
    ContractError,
    index_predictions,
    read_predictions,
)
from telemetry_reasoning.eval.metrics import (
    DEFAULT_EVAL_CONFIG_PATH,
    RUBRIC_COLUMNS,
    TruthRecord,
    build_blinded_rubric,
    confusion_rows,
    evaluate,
    format_summary,
    load_eval_config,
    load_manifest,
    metrics_rows,
    truth_from_rows,
)
from telemetry_reasoning.eval.plots import plot_confusion_matrix, plot_variant_comparison, save_figure


def _read_metadata(path: Path) -> Any:
    # ASSUMPTION: io.read_metadata_csv(path) returns MetadataRow objects.
    from telemetry_reasoning.io import read_metadata_csv

    return read_metadata_csv(path)


def _load_failure_view(path: Path) -> dict[str, Any]:
    from telemetry_reasoning.schemas import PairedBehaviorTree

    # Consumer views are already JSON-compatible dictionaries. Validating the
    # storage schema first keeps labels/success traces out of evidence lookup.
    return PairedBehaviorTree.model_validate_json(path.read_text()).to_failure_only()


def load_views(bt_dir: Path, truth: Sequence[TruthRecord]) -> dict[str, dict[str, Any]]:
    """Failure-only views for pairs that have a BT file; the rest stay unverifiable."""
    if not bt_dir.is_dir():
        raise ContractError(f"BT directory does not exist or is not a directory: {bt_dir}")
    views: dict[str, dict[str, Any]] = {}
    for t in truth:
        path = bt_dir / f"pair_{t.pair_id}_bt.json"
        if path.is_file():
            try:
                view = _load_failure_view(path)
            except ValueError as exc:
                raise ContractError(f"{path}: invalid paired behavior tree: {exc}") from exc
            if view.get("pair_id") != t.pair_id or view.get("task") != t.task:
                raise ContractError(f"{path}: BT pair_id/task does not match metadata for pair {t.pair_id}")
            views[t.pair_id] = view
    return views


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    cols = list(columns or (rows[0].keys() if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _load_run_manifests(paths: Sequence[Path]) -> dict[str, Any]:
    """Attach both rules' single-variant and inference's multi-variant receipts.

    A multi-variant receipt is retained intact under each declared variant, so
    downstream consumers can always find that variant's complete provenance.
    Overlapping receipts are ambiguous and must never silently overwrite one
    another.
    """
    out: dict[str, Any] = {}
    for p in paths:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            raise ContractError(f"{p}: run manifest must be a JSON object")
        if "variants" in data:
            variants = data["variants"]
            if not isinstance(variants, list) or not variants or not all(isinstance(v, str) for v in variants):
                raise ContractError(f"{p}: run-manifest variants must be a nonempty list of strings")
            if "model_variant" in data and variants != [data["model_variant"]]:
                raise ContractError(f"{p}: conflicting model_variant and variants declarations")
        else:
            variants = [data.get("model_variant", p.stem)]
        if any(not isinstance(v, str) or v not in MODEL_VARIANTS for v in variants):
            raise ContractError(f"{p}: unknown run-manifest model variant(s): {variants!r}")
        if len(set(variants)) != len(variants):
            raise ContractError(f"{p}: duplicate run-manifest variants")
        for variant in variants:
            if variant in out:
                raise ContractError(f"duplicate run manifest for {variant}")
            out[variant] = data
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_EVAL_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bt-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--run-manifest", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)

    try:
        cfg = load_eval_config(args.config)
        truth_all = truth_from_rows(_read_metadata(args.metadata))
        manifest = load_manifest(args.manifest) if args.manifest else None

        truth = truth_all
        if args.split:
            if manifest is None:
                raise ContractError("--split requires --manifest")
            missing = [t.pair_id for t in truth_all if t.pair_id not in manifest.splits]
            if missing:
                raise ContractError(f"manifest has no split for pair(s): {', '.join(missing[:5])}")
            truth = [t for t in truth_all if manifest.splits[t.pair_id] == args.split]

        indexed = index_predictions(read_predictions(*args.predictions), {t.pair_id for t in truth_all})

        views = load_views(args.bt_dir, truth) if args.bt_dir else None
        result = evaluate(
            truth, indexed, cfg,
            variants=args.variants,
            views=views,
            injection_times=manifest.injection_times if manifest else None,
        )
        result["split"] = args.split
        result["run_manifests"] = _load_run_manifests(args.run_manifest)
    except (ContractError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    _write_csv(out / "metrics.csv", metrics_rows(result))
    _write_csv(out / "confusion.csv", confusion_rows(result), ["model_variant", "task", "true_label", "predicted", "count"])

    rows, key = build_blinded_rubric(truth, indexed, list(result["variants"]), cfg.rubric)
    if rows:
        _write_csv(out / "rubric_blinded.csv", rows, RUBRIC_COLUMNS)
        (out / "rubric_key.json").write_text(json.dumps(key, indent=2) + "\n")

    plots = out / "plots"
    for variant, entry in result["variants"].items():
        for name, sl in entry["slices"].items():
            fig = plot_confusion_matrix(sl, f"{variant} / {name}", synthetic=entry["synthetic"])
            save_figure(fig, plots / f"confusion_{variant}_{name}.png", cfg.plots.dpi)
    if result["variants"]:
        save_figure(plot_variant_comparison(result), plots / "variant_comparison.png", cfg.plots.dpi)

    print(format_summary(result))
    print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
