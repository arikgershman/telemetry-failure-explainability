#!/usr/bin/env python3
"""Restore pinned dataset assets offline, or explicitly download them first."""
import argparse
from pathlib import Path
import sys
from zipfile import BadZipFile

from telemetry_reasoning.sim.release import DEFAULT_LOCK, load_release_lock, restore_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=Path("data/raw/releases/dataset-v1"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/dataset"), help="Fresh directory to publish atomically; existing roots are never overwritten")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--download", action="store_true", help="Download missing pinned HTTPS assets; otherwise require an existing offline cache")
    args = parser.parse_args()
    try:
        batch = restore_release(load_release_lock(args.lock), args.asset_dir, args.dataset_root, download=args.download)
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        print(f"Dataset restoration failed: {exc}", file=sys.stderr)
        return 1
    print(f"Restored verified release assets: {batch}")
    print(f"Receipt: {batch / 'provenance/release.json'}")
    print("Producer provenance, published clocks, and physical outcomes require separate certification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
