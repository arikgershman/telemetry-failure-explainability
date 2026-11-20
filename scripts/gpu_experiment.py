#!/usr/bin/env python
"""Prepare on CPU, verify/transfer, then run comparable inference on a GPU."""

import argparse
import json
from pathlib import Path
import sys

import yaml

from telemetry_reasoning.experiments import prepare_experiment, run_experiment, verify_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="CPU: freeze inputs, run rules, count exact tokenizer budgets")
    for name in ("batch", "handoff", "bt-root", "tokenizer-dir", "output-dir"):
        prepare.add_argument(f"--{name}", required=True, type=Path)
    prepare.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    prepare.add_argument("--smoke", action="store_true", help="One pair per represented mode per task from the selected split")
    prepare.add_argument("--final-study", action="store_true", help="Explicitly authorize opening held-out test inputs")
    prepare.add_argument("--llm-config", type=Path)
    verify = commands.add_parser("verify", help="CPU: check portable bundle and implementation hashes")
    verify.add_argument("--bundle", type=Path, required=True)
    run = commands.add_parser("run", help="Run all HF variants, then evaluate alongside the prepared rule baseline")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--backend", choices=("hf", "stub"), default="hf")
    run.add_argument("--qualification", type=Path,
                     help="Passed longest-request qualification receipt required by the v3 GPU profile")
    run.add_argument("--inference-only", action="store_true",
                     help="Generate/validate responses only; evaluate the transferred results on the Mac")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            if args["llm_config"] is None:
                del args["llm_config"]
            result = prepare_experiment(**args)
            print(json.dumps({key: result[key] for key in ("status", "pair_counts", "request_count", "dataset_release_status")}, indent=2))
            return 0 if result["status"].startswith("ready_for_gpu") else 1
        if command == "verify":
            result = verify_experiment(**args)
            print(f"Experiment bundle verified: {result['request_count']} requests; {result['status']}")
            return 0
        return run_experiment(**args)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
