"""CPU-only, source-bound preference formatting using the inference prompt API.

This module formats annotations; it neither supplies human preferences nor
trains a model. Reviewed inputs are assertions by their named reviewer, not an
independent certification of annotation quality. Synthetic demos are published
under a separate mode and are explicitly ineligible for training acceptance.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from telemetry_reasoning.models.llm import parse_response
from telemetry_reasoning.models.prompts import (
    DEFAULT_TEMPLATE_DIR, TEMPLATE_FILENAMES, ModelVariant, build_prompt,
)
from telemetry_reasoning.schemas import PairedBehaviorTree, Task
from telemetry_reasoning.sim.dataset_certification import inventory_sha256


Nonempty = Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]


class PreferenceProvenance(BaseModel):
    """Bind an annotation to the exact stored BT and inference messages."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["human_reviewed", "synthetic_demo"]
    annotation_id: Nonempty
    rationale: Nonempty
    bt_sha256: Digest
    prompt_sha256: Digest
    reviewer_id: Nonempty | None = None
    reviewed_at: Nonempty | None = None
    generator_id: Nonempty | None = None

    @model_validator(mode="after")
    def require_mode_details(self) -> "PreferenceProvenance":
        if self.kind == "human_reviewed":
            if not self.reviewer_id or not self.reviewed_at or self.generator_id is not None:
                raise ValueError("human_reviewed requires reviewer_id/reviewed_at and no generator_id")
            try:
                value = datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("reviewed_at must be an ISO-8601 timestamp") from exc
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("reviewed_at requires an explicit timezone")
        elif not self.generator_id or self.reviewer_id is not None or self.reviewed_at is not None:
            raise ValueError("synthetic_demo requires generator_id and forbids reviewer claims")
        return self


class PreferenceRecord(BaseModel):
    """One preference between strict inference-response JSON strings.

    Choice objects contain the five model-controlled response fields, exactly
    as returned by inference. Pair/variant identity belongs only to this outer
    record, so it cannot leak into the completion or prompt.
    """

    model_config = ConfigDict(extra="forbid")
    task: Task
    pair_id: Nonempty
    model_variant: ModelVariant
    chosen: Nonempty
    rejected: Nonempty
    preference_provenance: PreferenceProvenance


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _loads(raw: bytes | str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=reject)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"input must be a regular, unaliased file: {path}")
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": _hash(raw), "size_bytes": len(raw)}


def _split_index(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate grouping across every split, including excluded test records."""
    if (not isinstance(manifest, dict) or manifest.get("manifest_version") != 2
            or manifest.get("hash_algorithm") != "sha256"):
        raise ValueError("a frozen version-2 SHA-256 dataset manifest is required")
    files = manifest.get("files")
    if not isinstance(files, dict) or manifest.get("source_inventory_sha256") != inventory_sha256(files):
        raise ValueError("manifest source inventory digest is invalid")
    for name, details in files.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name
                or not isinstance(details, dict)
                or re.fullmatch(r"[0-9a-f]{64}", str(details.get("sha256"))) is None
                or type(details.get("size_bytes")) is not int or details["size_bytes"] < 0):
            raise ValueError("manifest source inventory contains an invalid path or fingerprint")
    grouping = manifest.get("splits", {})
    if not isinstance(grouping, dict) or grouping.get("split_unit") != "initial_condition_id":
        raise ValueError("manifest must group splits by initial_condition_id")
    splits = grouping.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "dev", "test"}:
        raise ValueError("manifest must contain train/dev/test splits")
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    conditions: dict[tuple[str, str], str] = {}
    references: set[str] = set()
    for split, records in splits.items():
        if not isinstance(records, list):
            raise ValueError("split entries must be lists")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("split records must be objects")
            task = Task(record["task"]).value
            pair_id, condition = record["pair_id"], record["initial_condition_id"]
            if not isinstance(pair_id, str) or not pair_id.strip() or not isinstance(condition, str) or not condition.strip():
                raise ValueError("pair and initial-condition IDs must be nonempty strings")
            if any(char in pair_id for char in ("/", "\\", "\x00", "\n", "\r")) or pair_id in (".", ".."):
                raise ValueError("unsafe pair ID")
            key, group = (task, pair_id), (task, condition)
            if key in pairs:
                raise ValueError(f"duplicate task-local pair identity: {key}")
            if group in conditions and conditions[group] != split:
                raise ValueError(f"initial condition crosses splits: {group}")
            conditions[group] = split
            for field in ("success_file", "failure_file"):
                name = record[field]
                path = PurePosixPath(name)
                if (not isinstance(name, str) or not name.startswith(task + "/")
                        or path.is_absolute() or ".." in path.parts or "\\" in name
                        or path.as_posix() != name or name not in files):
                    raise ValueError(f"missing or unsafe source reference: {name}")
                if name in references:
                    raise ValueError(f"reused source reference: {name}")
                references.add(name)
            pairs[key] = {**record, "split": split}
    if not pairs:
        raise ValueError("manifest contains no pairs")
    expected_conditions = {
        task: {split: sorted(condition for (candidate, condition), assigned in conditions.items()
                             if candidate == task and assigned == split)
               for split in splits}
        for task, _condition in conditions
    }
    actual_conditions = grouping.get("condition_assignments")
    if (not isinstance(actual_conditions, dict)
            or any(not isinstance(assigned, dict) or set(assigned) != set(splits)
                   or any(not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                          for values in assigned.values()) for assigned in actual_conditions.values())
            or {task: {split: sorted(values) for split, values in assigned.items()}
                for task, assigned in actual_conditions.items()} != expected_conditions):
        raise ValueError("manifest condition assignments disagree with pair splits")
    return pairs


def _bt_path(root: Path, task: str, pair_id: str, subdir: str) -> Path:
    path = root / task / subdir / f"pair_{pair_id}_bt.json"
    # No discovery/search can select a different task's identically named pair.
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"BT paths must not contain symlinks: {current}")
    return path


def prepare_training_data(
    bt_dir: Path | str, preferences: Path | str, manifest_path: Path | str,
    output_dir: Path | str, *, synthetic_demo: bool = False,
    template_dir: Path | str | None = None, bt_subdir: str = "enriched-bt",
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Atomically publish DPO and chosen-only SFT exports plus hashed sidecars.

    ``train`` and ``dev`` stay separate; preferences on held-out ``test`` pairs
    are refused. Every response is independently validated against its prompt,
    including exact failure-side evidence. This checks grounding and format,
    not whether the reviewer selected the correct causal explanation.
    """
    root, pref_path, manifest_path, output = map(Path, (bt_dir, preferences, manifest_path, output_dir))
    directory = Path(template_dir) if template_dir is not None else DEFAULT_TEMPLATE_DIR
    subpath = PurePosixPath(bt_subdir)
    if subpath.is_absolute() or ".." in subpath.parts or "\\" in bt_subdir or subpath.as_posix() != bt_subdir:
        raise ValueError("bt_subdir must be a safe relative POSIX path")
    if output.exists() or output.is_symlink():
        raise ValueError("output directory must be absent; choose a fresh directory")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("BT root must be a real directory")
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("outputs must be outside the immutable BT input tree")
    input_files = [_fingerprint(pref_path), _fingerprint(manifest_path)]
    source_paths = [Path(__file__), Path(__file__).with_name("llm.py"),
                    Path(__file__).with_name("prompts.py"), Path(__file__).parents[1] / "schemas.py",
                    Path(__file__).parents[1] / "eval/contracts.py",
                    Path(__file__).parents[1] / "sim/dataset_certification.py",
                    Path(__file__).parents[3] / "scripts/prepare_dpo_data.py"]
    source_files = [_fingerprint(path) for path in source_paths]
    templates = [{"model_variant": variant.value, **_fingerprint(directory / filename)}
                 for variant, filename in TEMPLATE_FILENAMES.items()]
    config = _fingerprint(Path(config_path)) if config_path is not None else None
    manifest = _loads(manifest_path.read_bytes())
    index = _split_index(manifest)
    records = []
    for number, line in enumerate(pref_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(PreferenceRecord.model_validate(_loads(line)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid preference line {number}: {exc}") from exc
    if not records:
        raise ValueError("preferences must contain at least one record")
    mode = "synthetic_demo" if synthetic_demo else "human_reviewed"
    generated: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "dev")}
    sft = {name: [] for name in generated}
    sidecar = []
    seen: set[tuple[str, str, str]] = set()
    annotations: set[str] = set()
    bt_cache: dict[tuple[str, str], tuple[PairedBehaviorTree, dict[str, Any]]] = {}
    for record in records:
        task, pair_id, variant = record.task.value, record.pair_id, record.model_variant.value
        key = (task, pair_id, variant)
        if key in seen:
            raise ValueError(f"duplicate preference identity: {key}")
        seen.add(key)
        provenance = record.preference_provenance
        if provenance.kind != mode:
            raise ValueError(f"preference mode must be {mode}; modes cannot be mixed")
        if provenance.annotation_id in annotations:
            raise ValueError("duplicate annotation_id")
        annotations.add(provenance.annotation_id)
        source = index.get((task, pair_id))
        if source is None:
            raise ValueError(f"unknown task-local pair: {(task, pair_id)}")
        split = source["split"]
        if split == "test":
            raise ValueError("held-out test preferences cannot enter training exports")
        if (task, pair_id) not in bt_cache:
            path = _bt_path(root, task, pair_id, bt_subdir)
            fingerprint = _fingerprint(path)
            tree = PairedBehaviorTree.model_validate_json(path.read_bytes())
            if tree.metadata.pair_id != pair_id or tree.metadata.task.value != task:
                raise ValueError("BT task/pair identity disagrees with preference and manifest")
            for kind in ("success", "failure"):
                if f"{task}/{getattr(tree.metadata, kind + '_file')}" != source[kind + "_file"]:
                    raise ValueError("BT source filenames disagree with frozen manifest")
            if tree.metadata.failure_mode.value != source["failure_mode"]:
                raise ValueError("BT storage metadata disagrees with frozen manifest")
            bt_cache[task, pair_id] = tree, fingerprint
            input_files.append(fingerprint)
        tree, fingerprint = bt_cache[task, pair_id]
        prompt = build_prompt(tree, variant, template_dir=directory)
        if provenance.bt_sha256 != fingerprint["sha256"] or provenance.prompt_sha256 != prompt.prompt_sha256:
            raise ValueError("annotation BT/prompt hashes do not match exact current inputs")
        choices = []
        for label in ("chosen", "rejected"):
            value = getattr(record, label)
            parsed = parse_response(value, prompt, pair_id, "stub")
            if parsed["status"] == "error":
                raise ValueError("training choices must be usable ok/abstain responses, not input errors")
            choices.append(_json({name: field for name, field in parsed.items()
                                  if name not in {"pair_id", "model_variant", "backend"}}))
        if choices[0] == choices[1]:
            raise ValueError("chosen and rejected must differ semantically after canonical JSON normalization")
        row_number = len(generated[split])
        generated[split].append({"prompt": prompt.messages,
                                 "chosen": [{"role": "assistant", "content": choices[0]}],
                                 "rejected": [{"role": "assistant", "content": choices[1]}]})
        sft[split].append({"messages": [*prompt.messages, {"role": "assistant", "content": choices[0]}]})
        sidecar.append({"task": task, "pair_id": pair_id, "model_variant": variant,
                        "split": split, "initial_condition_id": source["initial_condition_id"],
                        "row": row_number, "dpo_file": f"dpo/{split}.jsonl",
                        "sft_file": f"sft/{split}.jsonl", "bt": fingerprint,
                        "prompt_sha256": prompt.prompt_sha256, "template_sha256": prompt.template_sha256,
                        "chosen_sha256": _hash(choices[0].encode()),
                        "rejected_sha256": _hash(choices[1].encode()),
                        "preference_provenance": provenance.model_dump(mode="json", exclude_none=True)})
    receipt = {"schema_version": 1, "status": "formatted", "preference_mode": mode,
               "training_acceptance": "synthetic_demo_only" if synthetic_demo else "reviewed_annotations_supplied",
               "review_verification": "reviewer assertions retained; annotation quality is not independently certified",
               "rows": {split: len(rows) for split, rows in generated.items()},
               "tasks": sorted({record.task.value for record in records}),
               "variants": sorted({record.model_variant.value for record in records}),
               "excluded_splits": ["test"], "split_unit": "(task, initial_condition_id)",
               "dataset_release_status": manifest.get("release_status"),
               "dataset_certification": manifest.get("certification", {}).get("status"),
               "dataset_source_inventory_sha256": manifest["source_inventory_sha256"],
               "inputs": input_files, "source_files": source_files, "prompt_templates": templates,
               "config": config,
               "effective_configuration": {"bt_subdir": bt_subdir, "response_policy": "strict_json_v1",
                                           "synthetic_demo": synthetic_demo,
                                           "main_requires_computed_divergence": True},
               "artifacts": {}, "input_files_unchanged": False}
    # Validate everything before touching the destination. Staging files stay on
    # the same filesystem for the final directory rename.
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        artifacts = {f"dpo/{split}.jsonl": rows for split, rows in generated.items()}
        artifacts.update({f"sft/{split}.jsonl": rows for split, rows in sft.items()})
        artifacts["records.jsonl"] = sidecar
        for name, rows in artifacts.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = "".join(_json(row) + "\n" for row in rows).encode("utf-8")
            path.write_bytes(raw)
            receipt["artifacts"][name] = {"sha256": _hash(raw), "size_bytes": len(raw), "rows": len(rows)}
        for fingerprint in [*input_files, *source_files, *templates, *([config] if config else [])]:
            if _fingerprint(Path(fingerprint["path"]))["sha256"] != fingerprint["sha256"]:
                raise ValueError("inputs changed while formatting; no outputs published")
        receipt["input_files_unchanged"] = True
        (stage / "training_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise ValueError("output directory appeared during formatting; refusing overwrite")
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return receipt


def verify_training_data(output_dir: Path | str, *, check_inputs: bool = True) -> dict[str, Any]:
    """Verify published bytes and counts before consuming a training export.

    The default also verifies original local inputs/implementation. A transferred
    artifact can use ``check_inputs=False`` to check the portable publication;
    its annotation mode and dataset certification remain explicit in the result.
    """
    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("training output must be a real directory")
    receipt = _loads((root / "training_manifest.json").read_bytes())
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1 or receipt.get("status") != "formatted":
        raise ValueError("unsupported training manifest")
    mode = receipt.get("preference_mode")
    expected_acceptance = {"human_reviewed": "reviewed_annotations_supplied", "synthetic_demo": "synthetic_demo_only"}
    if mode not in expected_acceptance or receipt.get("training_acceptance") != expected_acceptance[mode]:
        raise ValueError("inconsistent training preference mode")
    expected = {"dpo/train.jsonl", "dpo/dev.jsonl", "sft/train.jsonl", "sft/dev.jsonl", "records.jsonl"}
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("training artifact inventory is incomplete or unexpected")
    found = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("training outputs must not contain symlinks")
        if path.is_file():
            found.add(path.relative_to(root).as_posix())
    if found != expected | {"training_manifest.json"}:
        raise ValueError("training output files differ from publication inventory")
    for name in expected:
        fingerprint = _fingerprint(root / name)
        details = artifacts[name]
        if (not isinstance(details, dict) or any(fingerprint[key] != details.get(key) for key in ("sha256", "size_bytes"))
                or len((root / name).read_text(encoding="utf-8").splitlines()) != details.get("rows")):
            raise ValueError(f"training artifact changed: {name}")
    if not receipt.get("input_files_unchanged"):
        raise ValueError("publication did not establish unchanged inputs")
    if check_inputs:
        fingerprints = [*receipt["inputs"], *receipt["source_files"], *receipt["prompt_templates"]]
        if receipt.get("config") is not None:
            fingerprints.append(receipt["config"])
        for fingerprint in fingerprints:
            actual = _fingerprint(Path(fingerprint["path"]))
            if any(actual[key] != fingerprint.get(key) for key in ("sha256", "size_bytes")):
                raise ValueError(f"training source input changed: {fingerprint['path']}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bt-dir", type=Path)
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bt-subdir", default="enriched-bt")
    parser.add_argument("--template-dir", type=Path)
    parser.add_argument("--config", type=Path, help="optional configuration/notes fingerprint")
    parser.add_argument("--synthetic-demo", action="store_true")
    parser.add_argument("--verify", type=Path, help="verify a published output and original local inputs")
    parser.add_argument("--artifacts-only", action="store_true", help="with --verify: omit unavailable original-source checks after transfer")
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            if any(value is not None for value in (args.bt_dir, args.preferences, args.manifest, args.output_dir,
                                                   args.template_dir, args.config)) or args.synthetic_demo:
                raise ValueError("--verify cannot be combined with formatting arguments")
            receipt = verify_training_data(args.verify, check_inputs=not args.artifacts_only)
            print(json.dumps({"verified": str(args.verify.resolve()), "rows": receipt["rows"],
                              "training_acceptance": receipt["training_acceptance"]}))
            return 0
        if args.artifacts_only:
            raise ValueError("--artifacts-only requires --verify")
        if any(value is None for value in (args.bt_dir, args.preferences, args.manifest, args.output_dir)):
            raise ValueError("--bt-dir, --preferences, --manifest and --output-dir are required for formatting")
        receipt = prepare_training_data(args.bt_dir, args.preferences, args.manifest, args.output_dir,
                                        synthetic_demo=args.synthetic_demo, template_dir=args.template_dir,
                                        bt_subdir=args.bt_subdir, config_path=args.config)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"training-data error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rows": receipt["rows"],
                      "training_acceptance": receipt["training_acceptance"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
