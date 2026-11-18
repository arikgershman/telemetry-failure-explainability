#!/usr/bin/env python
"""Enrich existing pair_*_bt.json files with deterministic divergence.

This is a separate, atomic post-parser stage. It never modifies input trees.
Run from the repository root after scripts/parse_dataset.py.

Example:
  python scripts/enrich_divergence.py --bt-dir data/processed/smoke/raw-bt \
      --config configs/divergence.yaml \
      --output-dir data/processed/smoke/enriched-bt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import yaml

from telemetry_reasoning.parser.divergence_metrics import (
    enrich_paired_bt, load_divergence_config,
)
from telemetry_reasoning.schemas import PairedBehaviorTree

BT_FILENAME = re.compile(r"pair_[^/\\\x00\r\n]+_bt\.json\Z")


def enrich_directory(bt_dir: Path, config_path: Path, output_dir: Path) -> dict:
    """Validate the entire batch before atomically publishing any outputs."""
    if not bt_dir.is_dir():
        raise ValueError(f"BT input directory does not exist: {bt_dir}")
    if output_dir.resolve() == bt_dir.resolve():
        raise ValueError("input and output directories must be different")
    if output_dir.is_symlink() or (output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    )):
        raise ValueError("output directory must be absent or empty")

    config = load_divergence_config(config_path)
    paths = sorted(bt_dir.glob("pair_*_bt.json"))
    if not paths:
        raise ValueError("no pair_*_bt.json inputs found")
    written = []
    seen_ids = set()
    for path in paths:
        if not path.is_file() or not BT_FILENAME.fullmatch(path.name):
            raise ValueError(f"invalid BT input path: {path}")
        original_bytes = path.read_bytes()
        tree = PairedBehaviorTree.model_validate_json(original_bytes)
        pair_id = tree.metadata.pair_id
        if pair_id in seen_ids or path.name != f"pair_{pair_id}_bt.json":
            raise ValueError(f"duplicate or mismatched pair ID: {path}")
        seen_ids.add(pair_id)
        enriched = enrich_paired_bt(tree, config)
        payload = enriched.model_dump_json(indent=2) + "\n"
        observed = sum(bool(node.failure_samples) for node in enriched.subtasks)
        written.append((path.name, payload, {
            "pair_id": pair_id,
            "task": tree.metadata.task.value,
            "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "computed_subtasks": observed,
            "not_comparable_subtasks": len(enriched.subtasks) - observed,
        }))

    cfg_bytes = config_path.read_bytes()
    report = {
        "status": "computed",
        "input_directory": str(bt_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
        "divergence_config": config.model_dump(mode="json"),
        "pair_count": len(written),
        "computed_subtasks": sum(row[2]["computed_subtasks"] for row in written),
        "not_comparable_subtasks": sum(row[2]["not_comparable_subtasks"] for row in written),
        "pairs": [row[2] for row in written],
        "onset_note": "Earliest aligned-grid threshold crossing; not verified physical onset",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".divergence-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "enriched"
        staging.mkdir()
        for filename, payload, _ in written:
            (staging / filename).write_text(payload, encoding="utf-8")
        (staging / "divergence_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.rmdir()
        staging.rename(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bt-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = enrich_directory(args.bt_dir, args.config, args.output_dir)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Enriched {report['pair_count']} paired BT(s); "
          f"computed {report['computed_subtasks']} observed subtask comparisons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
