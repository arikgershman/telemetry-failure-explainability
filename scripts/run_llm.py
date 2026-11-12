#!/usr/bin/env python
"""Run the Track 5 prompt adapter without mixing results from earlier runs.

CPU plumbing smoke (stub results are not LLM results)::

    python scripts/run_llm.py --bt-dir data/sample --backend stub \
        --output-dir runs/llm-smoke

Local access/compute inspection, without downloading or opening model weights::

    python scripts/run_llm.py --check-access

The output directory contains one ``<variant>.jsonl`` per requested variant
and ``run_manifest.json``. Exit codes: 0 for success/abstention, 1 if any
prediction has error status, and 2 for configuration, input, or setup errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import yaml

from telemetry_reasoning.models.llm import (
    HFBackend,
    StubBackend,
    capabilities,
    load_llm_config,
    run_inference,
)
from telemetry_reasoning.schemas import PairedBehaviorTree


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("baseline_2", "baseline_3", "main")
PAIR_FILE = re.compile(r"^pair_(?P<pair_id>[^/\\\x00\r\n]+)_bt\.json$")


def load_inputs(bt_dir: Path) -> tuple[tuple[PairedBehaviorTree, ...], list[dict[str, str]]]:
    """Validate every input before creating a backend or publishing outputs.

    Parser reports and other non-BT files can share the directory. Only exact
    ``pair_<id>_bt.json`` names are accepted as BT inputs; matching directories
    and malformed names resembling BT files fail explicitly.
    """
    if not bt_dir.is_dir():
        raise ValueError(f"--bt-dir {bt_dir} is not a directory")
    paths = []
    for path in sorted(bt_dir.iterdir(), key=lambda item: item.name):
        if not (path.name.startswith("pair_") and path.name.endswith("_bt.json")):
            continue
        if not PAIR_FILE.fullmatch(path.name):
            raise ValueError(f"invalid BT filename: {path.name!r}; expected pair_<id>_bt.json")
        if not path.is_file():
            raise ValueError(f"BT input is not a file: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"no pair_<id>_bt.json files found in {bt_dir}")

    trees, provenance = [], []
    seen: set[str] = set()
    for path in paths:
        try:
            raw = path.read_bytes()
            tree = PairedBehaviorTree.model_validate_json(raw)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid BT input {path}: {exc}") from exc
        pair_id = tree.metadata.pair_id
        if pair_id in seen:
            raise ValueError(f"duplicate embedded pair_id {pair_id!r} in {path}")
        seen.add(pair_id)
        trees.append(tree)
        provenance.append({
            "path": str(path.resolve()),
            "pair_id": pair_id,
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    for path, tree in zip(paths, trees):
        expected = f"pair_{tree.metadata.pair_id}_bt.json"
        if path.name != expected:
            raise ValueError(
                f"BT filename/pair_id mismatch: {path.name!r} contains pair_id "
                f"{tree.metadata.pair_id!r}; expected {expected!r}"
            )
    return tuple(trees), provenance


def _validate_output(output_dir: Path) -> None:
    if output_dir.is_symlink() or (output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    )):
        raise ValueError("--output-dir must be absent or empty; choose a fresh directory")


def publish_run(
    output_dir: Path, predictions: Sequence[dict[str, Any]],
    variants: Sequence[str], manifest: dict[str, Any],
) -> None:
    """Stage all files, then publish the run as one directory rename."""
    _validate_output(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".llm-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "run"
        staged.mkdir()
        for variant in variants:
            path = staged / f"{variant}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for record in predictions:
                    if record["model_variant"] == variant:
                        stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
        (staged / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # A second check protects against a writer populating the target while
        # inference ran. rmdir also fails if it becomes nonempty after this check.
        _validate_output(output_dir)
        if output_dir.exists():
            output_dir.rmdir()
        staged.rename(output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bt-dir", type=Path, help="directory containing pair_<id>_bt.json inputs")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/llm.yaml")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[:2]))
    parser.add_argument("--backend", choices=("stub", "hf"), default="stub")
    parser.add_argument("--output-dir", type=Path, help="fresh or empty destination directory")
    parser.add_argument(
        "--check-access", action="store_true",
        help="report local capabilities without loading weights or contacting Hugging Face",
    )
    args = parser.parse_args(argv)
    try:
        config_bytes = args.config.read_bytes()
        config = load_llm_config(args.config)
        if args.check_access:
            print(json.dumps(capabilities(config), indent=2, allow_nan=False))
            return 0
        if args.bt_dir is None or args.output_dir is None:
            raise ValueError("--bt-dir and --output-dir are required unless --check-access is used")
        if len(set(args.variants)) != len(args.variants):
            raise ValueError("--variants must be unique")
        _validate_output(args.output_dir)
        trees, input_files = load_inputs(args.bt_dir)
        backend = StubBackend(config) if args.backend == "stub" else HFBackend(config)
        result = run_inference(trees, args.variants, config, backend=backend)
        manifest = dict(result.manifest)
        manifest.update({
            "bt_dir": str(args.bt_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "input_files": input_files,
            "config_path": str(args.config.resolve()),
            "config_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "prediction_files": {variant: f"{variant}.jsonl" for variant in args.variants},
        })
        publish_run(args.output_dir, result.predictions, args.variants, manifest)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    counts = Counter(record["status"] for record in result.predictions)
    if args.backend == "stub":
        print("STUB BACKEND: plumbing-test outputs only; these are not LLM results.")
    print(
        f"wrote {len(result.predictions)} prediction(s) to {args.output_dir} "
        f"(ok={counts['ok']}, abstain={counts['abstain']}, error={counts['error']})"
    )
    print(f"run manifest: {args.output_dir / 'run_manifest.json'}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
