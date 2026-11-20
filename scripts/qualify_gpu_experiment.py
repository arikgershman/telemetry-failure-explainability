#!/usr/bin/env python3
"""Run one measured longest fitting request before the full frozen GPU experiment."""
import argparse
import json
from pathlib import Path
import sys

from telemetry_reasoning.gpu_qualification import qualify_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--kernel-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = qualify_experiment(**vars(args))
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: receipt.get(key) for key in ("status", "selected_request", "failures")}, indent=2))
    print(f"Qualification receipt: {args.output_dir / 'qualification_receipt.json'}")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
