#!/usr/bin/env python
"""Replay nominal and injected runs independently; retain mismatches, never guess clocks.

Requires MuJoCo/model assets. Exit codes: 0=matched, 1=stored-trace mismatch,
2=setup error. A representative run certifies only explicitly listed coverage.
Original diagnostics and CSVs remain unchanged.
"""

import argparse
from pathlib import Path
import sys

from telemetry_reasoning.sim.replay import recover_dataset_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representative", action="store_true", help="One pair per task/released failure mode")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = recover_dataset_evidence(args.batch, args.output_dir,
                                          representative=args.representative, repeats=args.repeats)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Replay status: {report['status']}; receipt: {args.output_dir / 'evidence.json'}")
    return 0 if report["status"] == "matched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
