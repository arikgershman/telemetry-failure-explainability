#!/usr/bin/env python
"""Run the Track 5 prompt adapter without mixing results from earlier runs.

CPU plumbing smoke (stub results are not LLM results)::

    python scripts/run_llm.py --bt-dir data/sample --backend stub \
        --output-dir runs/llm-smoke

Local access/compute inspection, without downloading or opening model weights::

    python scripts/run_llm.py --check-access

The output directory contains one ``<variant>.jsonl`` per requested variant,
``completions.jsonl`` with exact generated text and parsing outcomes, and
``run_manifest.json`` with an explicit hashed artifact inventory. Exit codes:
0 for success/abstention, 1 if any
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
    parse_response,
    run_inference,
)
from telemetry_reasoning.models.prompts import DEFAULT_TEMPLATE_DIR, TEMPLATE_FILENAMES, build_prompt
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


def _file_provenance(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def implementation_provenance(prompt_dir: Path | None) -> dict[str, Any]:
    """Hash the adapter and the exact templates, including custom directories.

    Shared schema/evidence validation sources are read for reproducibility;
    ownership of those files remains with their respective tracks.
    """
    sources = (
        Path(__file__).resolve(),
        ROOT / "src/telemetry_reasoning/models/llm.py",
        ROOT / "src/telemetry_reasoning/models/prompts.py",
        ROOT / "src/telemetry_reasoning/schemas.py",
        ROOT / "src/telemetry_reasoning/eval/contracts.py",
    )
    directory = prompt_dir or DEFAULT_TEMPLATE_DIR
    return {
        "source_files": [_file_provenance(path) for path in sources],
        "prompt_template_files": [
            {"model_variant": variant.value, **_file_provenance(directory / filename)}
            for variant, filename in TEMPLATE_FILENAMES.items()
        ],
    }


def execution_summary(
    *, selected_backend: str, trees: Sequence[PairedBehaviorTree], variants: Sequence[str],
    predictions: Sequence[dict[str, Any]], completions: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    prompt_dir: Path | None = None,
) -> dict[str, Any]:
    """Distinguish selecting HF, generating text, and passing a smoke run.

    Rebuild prompts and independently parse saved text for every pair/variant.
    Claimed status or loaded weights alone cannot establish success. This
    demonstrates the inference contract, not diagnosis accuracy or human review.
    Stub predictions can never satisfy real-model acceptance.
    """
    expected = {(tree.metadata.pair_id, variant) for tree in trees for variant in variants}
    predicted = {(record["pair_id"], record["model_variant"]): record for record in predictions}
    generated = {(record["pair_id"], record["model_variant"]): record for record in completions}
    valid = set()
    for tree in trees:
        for variant in variants:
            key = (tree.metadata.pair_id, variant)
            try:
                record, completion = predicted[key], generated[key]
                if record["status"] not in ("ok", "abstain"):
                    continue
                if not (completion["status"] == record["status"] and
                        record["backend"] == selected_backend == completion["backend"]):
                    continue
                prompt = build_prompt(tree, variant, template_dir=prompt_dir)
                if (completion["messages"] != prompt.messages or
                        completion["prompt_sha256"] != prompt.prompt_sha256 or
                        not isinstance(completion["text"], str) or
                        completion["completion_sha256"] != hashlib.sha256(
                            completion["text"].encode("utf-8")).hexdigest()):
                    continue
                reparsed = parse_response(completion["text"], prompt, tree.metadata.pair_id, selected_backend)
                if reparsed == record:
                    valid.add(key)
            except (KeyError, TypeError, ValueError, OSError):
                # Preserve the original artifacts for review. Acceptance must
                # never repair a response or infer missing identity/provenance.
                continue
    coverage = (set(predicted) == expected and set(generated) == expected
                and len(predictions) == len(completions) == len(expected) and valid == expected)
    weights_loaded = manifest.get("backend_provenance", {}).get("weights_loaded") is True
    real_generation = (selected_backend == "hf" and weights_loaded and
                       any(record.get("backend") == "hf" for record in completions))
    passed = real_generation and coverage
    status = "synthetic" if selected_backend == "stub" else ("passed" if passed else "incomplete")
    return {
        "execution": {
            "selected_backend": selected_backend,
            "hf_weights_loaded": selected_backend == "hf" and weights_loaded,
            "hf_generation_completed": real_generation,
            "expected_requests": len(expected),
            "generated_requests": len(completions),
            "validated_requests": len(valid),
            "all_requests_validated": coverage,
        },
        "acceptance": {
            "scope": "inference_smoke",
            "status": status,
            "nov6_baselines_covered": passed and {"baseline_2", "baseline_3"}.issubset(variants),
            "note": ("Synthetic plumbing only; this run cannot establish real-model acceptance."
                     if selected_backend == "stub" else
                     "Smoke acceptance requires successful HF generation and validated outputs for every request; "
                     "it does not establish diagnosis accuracy or human review."),
        },
    }


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size, "n_records": len(records)}


def publish_run(
    output_dir: Path, predictions: Sequence[dict[str, Any]],
    variants: Sequence[str], manifest: dict[str, Any],
    *, completions: Sequence[dict[str, Any]] = (),
) -> None:
    """Stage predictions, raw completions, and their manifest before publication."""
    for record in completions:
        if (not isinstance(record.get("text"), str) or record.get("completion_sha256") !=
                hashlib.sha256(record["text"].encode("utf-8")).hexdigest()):
            raise ValueError("completion text/hash mismatch; refusing to publish unverifiable output")
    _validate_output(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".llm-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "run"
        staged.mkdir()
        artifacts = []
        for variant in variants:
            selected = [record for record in predictions if record["model_variant"] == variant]
            artifacts.append({
                "kind": "predictions", "model_variant": variant,
                **_write_jsonl(staged / f"{variant}.jsonl", selected),
            })
        artifacts.append({"kind": "completions", **_write_jsonl(staged / "completions.jsonl", completions)})
        published_manifest = {**manifest, "artifacts": artifacts, "manifest_file": "run_manifest.json"}
        (staged / "run_manifest.json").write_text(
            json.dumps(published_manifest, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
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
        implementation = implementation_provenance(config.prompt_dir)
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
            "completion_file": "completions.jsonl",
            **implementation,
        })
        manifest.update(execution_summary(
            selected_backend=args.backend, trees=trees, variants=args.variants,
            predictions=result.predictions, completions=result.completions, manifest=manifest,
            prompt_dir=config.prompt_dir,
        ))
        publish_run(args.output_dir, result.predictions, args.variants, manifest,
                    completions=result.completions)
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
    print(f"inference smoke: {manifest['acceptance']['status']}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
