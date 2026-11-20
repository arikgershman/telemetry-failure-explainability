#!/usr/bin/env python
"""CPU: freeze a verified native-producer run for portable GPU inference."""

import argparse
import json
from pathlib import Path
import sys

import yaml

from telemetry_reasoning.fresh_experiments import prepare_fresh_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-run", "tokenizer-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="One fully context-fitting pair per released mode and task")
    parser.add_argument("--llm-config", type=Path)
    parser.add_argument("--eval-config", type=Path)
    args = {name: value for name, value in vars(parser.parse_args(argv)).items() if value is not None}
    try:
        result = prepare_fresh_experiment(**args)
        print(json.dumps({key: result[key] for key in ("status", "pair_counts", "request_count", "context_refused_requests")}, indent=2))
        return 0 if result["status"].startswith("ready_for_gpu") else 1
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
