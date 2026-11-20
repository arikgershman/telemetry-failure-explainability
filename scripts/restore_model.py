#!/usr/bin/env python
"""Restore the release-pinned Panda assets without downloading other robot models."""

import argparse
import json
from pathlib import Path
import sys

from telemetry_reasoning.sim.model_assets import restore_panda

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/mujoco_menagerie/franka_emika_panda")
    parser.add_argument("--inventory", type=Path, help="Previously downloaded pinned Git tree JSON")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.receipt.exists() or args.receipt.is_symlink():
            raise ValueError("Receipt already exists; choose a fresh path")
        tree = json.loads(args.inventory.read_text()) if args.inventory else None
        receipt = restore_panda(args.output_dir, tree=tree)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Restored {len(receipt['files'])} pinned Panda files; receipt: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
