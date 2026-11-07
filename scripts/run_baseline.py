#!/usr/bin/env python
"""Run Baseline 1 (rule-based, failure-only) over a directory of paired behavior trees.

    python scripts/run_baseline.py \\
        --bt-dir data/processed/smoke/bt \\
        --config configs/rules.yaml \\
        --output runs/smoke/baseline_1.jsonl

Reads ``pair_<id>_bt.json`` files, hands the detector ONLY
``tree.to_failure_only()`` (no success samples, divergence, labels,
perturbations, filenames or seeds), and writes one contract-validated JSONL
record per file plus a sibling ``<output stem>.manifest.json`` with runtime and
configuration provenance.

A file that cannot be loaded still produces an ``error`` record, so a
bad input is accounted for by the evaluator rather than silently missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from telemetry_reasoning.eval.contracts import MODEL_VARIANTS, write_predictions
from telemetry_reasoning.models.rule_based import (
    BACKEND,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_VARIANT,
    RuleConfig,
    load_rule_config,
    predict,
)

PAIR_FILE = re.compile(r"^pair_(?P<pair_id>.+)_bt\.json$")


def _load_failure_view(path: Path) -> Any:
    """Failure-only view of a stored PairedBehaviorTree (shared schema, imported lazily)."""
    from telemetry_reasoning.schemas import PairedBehaviorTree

    return PairedBehaviorTree.model_validate_json(path.read_text()).to_failure_only()


def _error_record(pair_id: str, variant: str, message: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "model_variant": variant,
        "predicted_failure_mode": None,
        "predicted_subtask": None,
        "explanation": f"Input rejected: {message}",
        "evidence": [],
        "status": "error",
        "backend": BACKEND,
    }


def run(
    bt_dir: Path,
    cfg: RuleConfig,
    *,
    variant: str = DEFAULT_MODEL_VARIANT,
    loader: Optional[Callable[[Path], Any]] = None,
) -> list[dict[str, Any]]:
    """Predict for every ``pair_*_bt.json`` in ``bt_dir`` (sorted, so output order is stable)."""
    load = loader or _load_failure_view
    files = sorted(p for p in bt_dir.glob("pair_*_bt.json") if PAIR_FILE.match(p.name))
    records: list[dict[str, Any]] = []
    for path in files:
        pair_id = PAIR_FILE.match(path.name).group("pair_id")  # type: ignore[union-attr]
        try:
            view = load(path)
        except Exception as exc:  # unreadable/invalid BT: report, do not drop
            records.append(_error_record(pair_id, variant, f"could not load {path.name} ({type(exc).__name__}: {str(exc)[:200]})"))
            continue
        record = predict(view, cfg, model_variant=variant)
        if record["pair_id"] != pair_id:
            record = _error_record(pair_id, variant, f"{path.name} contains pair_id {record['pair_id']!r}")
        records.append(record)
    return records


def write_run_manifest(
    output: Path, *, records: Sequence[dict[str, Any]], cfg: RuleConfig, config_path: Path,
    bt_dir: Path, variant: str, runtime_s: float,
) -> Path:
    path = output.with_name(output.stem + ".manifest.json")
    manifest = {
        "model_variant": variant,
        "backend": BACKEND,
        "predictions": str(output),
        "bt_dir": str(bt_dir),
        "n_inputs": len(records),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "runtime_s": runtime_s,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config": cfg.model_dump(mode="json"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bt-dir", type=Path, required=True, help="directory of pair_<id>_bt.json files")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="rules.yaml (default: configs/rules.yaml)")
    parser.add_argument("--output", type=Path, required=True, help="predictions JSONL to write")
    parser.add_argument("--model-variant", default=DEFAULT_MODEL_VARIANT, choices=MODEL_VARIANTS)
    args = parser.parse_args(argv)

    if not args.bt_dir.is_dir():
        print(f"error: --bt-dir {args.bt_dir} is not a directory", file=sys.stderr)
        return 2
    cfg = load_rule_config(args.config)

    start = time.perf_counter()
    records = run(args.bt_dir, cfg, variant=args.model_variant)
    runtime_s = time.perf_counter() - start
    if not records:
        print(f"error: no pair_<id>_bt.json files found in {args.bt_dir}", file=sys.stderr)
        return 2

    write_predictions(args.output, records)
    manifest = write_run_manifest(
        args.output, records=records, cfg=cfg, config_path=args.config,
        bt_dir=args.bt_dir, variant=args.model_variant, runtime_s=runtime_s,
    )
    counts = Counter(r["status"] for r in records)
    print(f"wrote {len(records)} prediction(s) to {args.output}  "
          f"(ok={counts['ok']}, abstain={counts['abstain']}, error={counts['error']})")
    print(f"run manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
