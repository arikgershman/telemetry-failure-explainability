#!/usr/bin/env python
"""Publish all paired BTs and raw-reference diagnostics from a frozen handoff."""

import argparse
from pathlib import Path
import sys

import yaml

from telemetry_reasoning.parser.production import produce_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--divergence-config", type=Path, default=Path("configs/divergence.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="versioned dev-only selection receipt")
    args = parser.parse_args(argv)
    try:
        report = produce_dataset(args.batch, args.handoff, args.config, args.divergence_config, args.output_dir,
                                 selection_path=args.selection)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Published {report['pair_count']} enriched pairs to {args.output_dir}; "
          f"source release status: {report['release_status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
