#!/usr/bin/env python
"""CPU: verify returned GPU artifacts, then score all four variants and plot."""

import argparse
from pathlib import Path
import sys

from telemetry_reasoning.experiment_results import evaluate_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "results", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--score-response-errors", action="store_true",
        help="Score independently reproduced response-validation errors while retaining failed acceptance.",
    )
    args = vars(parser.parse_args(argv))
    try:
        report = evaluate_results(**args)
        print(f"CPU evaluation: {report['status']}; real-model acceptance: {report['real_model_acceptance']}")
        return 1 if report["status"] == "failed" else 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
