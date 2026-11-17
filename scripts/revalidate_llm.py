#!/usr/bin/env python
"""Verify a retained Track 5 generation bundle and revalidate on CPU only.

Example::

    python scripts/revalidate_llm.py --bundle-dir runs/alienware-import-... \
        --response-policy json_or_fenced_v1 --output-dir runs/llm-revalidated

No model/tokenizer runtime is imported. Exit codes match run_llm.py: 0 for
validated responses/abstentions, 1 for response errors, and 2 for setup or
provenance errors. Original bundle contents are never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import yaml

from telemetry_reasoning.models.revalidation import POLICIES, revalidate_bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response-policy", choices=POLICIES, default="json_or_fenced_v1")
    parser.add_argument("--prompt-dir", type=Path, help="local copy of unchanged custom generation templates")
    args = parser.parse_args(argv)
    try:
        manifest = revalidate_bundle(
            args.bundle_dir, args.output_dir,
            response_policy=args.response_policy, prompt_dir=args.prompt_dir,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "status_counts": manifest["status_counts"],
        "execution": manifest["execution"],
        "acceptance": manifest["acceptance"],
        "original_acceptance": manifest["original_acceptance"],
    }, indent=2, allow_nan=False))
    return 1 if manifest["status_counts"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
