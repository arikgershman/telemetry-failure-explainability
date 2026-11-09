#!/usr/bin/env python
"""Sweep compression settings using a supplied grouped train/dev manifest.

Example (requires the actual Track 2 sidecar, not an invented split)::

    python scripts/tune_compression.py --metadata DATA/metadata.csv \\
        --data-dir DATA --config configs/parser.yaml --manifest DATA/manifest.json \\
        --output runs/compression-dev.json

The manifest must assign every metadata pair to train/dev/test and provide
initial_condition_id. Test telemetry is never opened. Tuning defaults to dev;
--split train is available for explicitly designated training diagnostics.
This writes a report and never changes parser configuration or stored BTs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Sequence

import yaml

from telemetry_reasoning.config import load_config
from telemetry_reasoning.io import read_metadata_csv
from telemetry_reasoning.parser.tuning import (
    DEFAULT_K_GRID, DEFAULT_THRESHOLD_SCALES, sweep_compression,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="shared ProjectConfig YAML")
    parser.add_argument("--manifest", type=Path, required=True, help="Track 2 grouped split sidecar")
    parser.add_argument("--output", type=Path, required=True, help="deterministic JSON diagnostics report")
    parser.add_argument("--k-grid", type=int, nargs="+", default=DEFAULT_K_GRID)
    parser.add_argument("--threshold-scales", type=float, nargs="+", default=DEFAULT_THRESHOLD_SCALES)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config).parser
        metadata = read_metadata_csv(args.metadata)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        input_paths = [args.metadata, args.config, args.manifest]
        input_paths.extend(args.data_dir / getattr(row, field) for row in metadata
                           for field in ("success_file", "failure_file"))
        if args.output.resolve() in {path.resolve() for path in input_paths}:
            raise ValueError("--output must not overwrite metadata, configuration, manifest, or telemetry")
        report = sweep_compression(metadata, args.data_dir, config, manifest,
                                   split=args.split, k_grid=args.k_grid,
                                   threshold_scales=args.threshold_scales)
        report["source_hashes"].update({
            "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "config_file_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                             prefix=f".{args.output.name}.", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                                        allow_nan=False) + "\n")
            temporary.replace(args.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    except (OSError, ValueError, csv.Error, yaml.YAMLError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {len(report['candidates'])} candidates on {len(report['selected_pair_ids'])} "
          f"{args.split} pair(s) to {args.output}")
    recommendation = report["recommendation"]
    if recommendation is None:
        print("No candidate satisfies all fixed-reference event and reconstruction constraints.")
    else:
        print(f"Recommended within this split: {recommendation['candidate_id']} "
              f"({recommendation['paired_payload_utf8_bytes']} paired payload UTF-8 bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
