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

    return PairedBehaviorTree.model_validate_json(path.read_text()).to_failure_only().model_dump(mode="json")


def load_views(bt_dir: Path, truth: Sequence[TruthRecord]) -> dict[str, dict[str, Any]]:
    """Failure-only views for pairs that have a BT file; the rest stay unverifiable."""
    views: dict[str, dict[str, Any]] = {}
    for t in truth:
        path = bt_dir / f"pair_{t.pair_id}_bt.json"
        if path.is_file():
            views[t.pair_id] = _load_failure_view(path)
    return views


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    cols = list(columns or (rows[0].keys() if rows else []))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _load_run_manifests(paths: Sequence[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in paths:
        data = json.loads(p.read_text())
        out[str(data.get("model_variant", p.stem))] = data
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

    cfg = load_eval_config(args.config)
    try:
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
    except (ContractError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    views = load_views(args.bt_dir, truth) if args.bt_dir else None
    result = evaluate(
        truth, indexed, cfg,
        variants=args.variants,
        views=views,
        injection_times=manifest.injection_times if manifest else None,
    )
    result["split"] = args.split
    result["run_manifests"] = _load_run_manifests(args.run_manifest)

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
