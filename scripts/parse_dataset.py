#!/usr/bin/env python
"""Build schema-v1 paired BTs plus a raw-reference compression report.

Nov. 6 stage: divergence is explicitly uncomputed (null). Output must be a
fresh/empty directory; this prevents stale BTs entering downstream glob readers.
Invalid pairs are quarantined logically in parse_report.json, without changing
their source files. Exit codes: 0=all accepted, 1=pair rejections, 2=fatal input
or output error. Even with exit 1, valid pairs and the full report are published.

    python scripts/parse_dataset.py --metadata data/sample/metadata.csv \\
        --data-dir data/sample --config configs/parser.yaml \\
        --output-dir data/processed/smoke/bt

Parse task-specific metadata into separate directories until Track 2 provides
globally unique IDs; never silently rename pair IDs to merge task batches.
"""

import argparse
import csv
import hashlib
from pathlib import Path
import sys
from typing import Sequence

import yaml

from telemetry_reasoning.config import load_config
from telemetry_reasoning.io import read_metadata_csv
from telemetry_reasoning.parser.pipeline import parse_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/parser.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        rows = read_metadata_csv(args.metadata)
        provenance = {
            "metadata_file": str(args.metadata.resolve()),
            "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
            "config_file": str(args.config.resolve()),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "project_config": config.model_dump(mode="json"),
            "dataset_validation": "not_asserted_by_parser",
        }
        result = parse_dataset(rows, args.data_dir, config.parser, args.output_dir,
                               provenance=provenance)
    except (OSError, ValueError, csv.Error, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts = result.report["counts"]
    print(f"Wrote {counts['written_pairs']} BT(s); rejected {counts['rejected_pairs']} pair(s).")
    print(f"Parse/compression report: {result.report_path}")
    return 1 if counts["rejected_pairs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
