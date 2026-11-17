"""CPU-only replay of retained Track 5 completions under an explicit policy.

The original generation, configuration, raw text, and acceptance are immutable
evidence. A replay creates a separate run with new validation provenance; it
never loads weights, regenerates text, repairs JSON, or changes model claims.
The inference CLI's pure artifact helpers are loaded from this checkout so the
same publication and acceptance contract applies to generation and replay.
"""

from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
from typing import Any

import yaml

from telemetry_reasoning.models.llm import LLMConfig, parse_response, response_normalization
from telemetry_reasoning.models.prompts import DEFAULT_TEMPLATE_DIR, TEMPLATE_FILENAMES, build_prompt


ROOT = Path(__file__).resolve().parents[3]
POLICIES = ("strict_json_v1", "json_or_fenced_v1")
SOURCE_PATHS = (
    "scripts/run_llm.py",
    "src/telemetry_reasoning/models/llm.py",
    "src/telemetry_reasoning/models/prompts.py",
    "src/telemetry_reasoning/schemas.py",
    "src/telemetry_reasoning/eval/contracts.py",
)
VALIDATOR_UPGRADES = set(SOURCE_PATHS[:2])
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _file(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def _checked_hash(value: Any, description: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError(f"invalid {description} SHA-256")
    return value


def _basename(value: Any) -> str:
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value or any(ord(c) < 32 for c in value)):
        raise ValueError(f"artifact path must be a safe filename: {value!r}")
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL record in {path}:{number}")
        record = _json(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record must be an object in {path}:{number}")
        records.append(record)
    return records


def _index(records: Any, description: str) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"{description} must be a list")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"invalid {description} record")
        key = (record.get("pair_id"), record.get("model_variant"))
        if not all(isinstance(item, str) and item for item in key):
            raise ValueError(f"missing identity in {description}")
        if key in result:
            raise ValueError(f"duplicate pair/variant identity in {description}: {key}")
        result[key] = record
    return result


def _helpers() -> Any:
    path = ROOT / "scripts/run_llm.py"
    spec = importlib.util.spec_from_file_location("_track5_replay_artifacts", path)
    if spec is None or spec.loader is None:
        raise ValueError("offline revalidation requires the project checkout")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _UniqueSafeLoader(yaml.SafeLoader):
    """Disallow duplicate configuration keys rather than silently overwriting."""


def _yaml_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[str, Any]:
    return _unique_keys([(loader.construct_object(key), loader.construct_object(value))
                         for key, value in node.value])


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def _configuration(bundle: Path, manifest: dict[str, Any]) -> LLMConfig:
    config_path = bundle / _basename(PurePosixPath(manifest["config_path"]).name)
    provenance = _file(config_path)
    if provenance["sha256"] != _checked_hash(manifest["config_file_sha256"], "config file"):
        raise ValueError("configuration file hash mismatch")
    config = manifest["config"]
    if not isinstance(config, dict) or _canonical_hash(config) != manifest["config_sha256"]:
        raise ValueError("configuration object hash mismatch")
    recorded = LLMConfig.model_validate(config)
    loaded = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    if not isinstance(loaded, dict) or set(loaded) != {"llm"}:
        raise ValueError("configuration must contain exactly one llm mapping")
    config_value = LLMConfig.model_validate(loaded["llm"]).model_dump(mode="json")
    # Resolve relative paths in their original generation location, not on the
    # receiving machine. Replay never accesses either model-cache location.
    for key in ("prompt_dir", "cache_dir"):
        if config_value[key] is not None and not PurePosixPath(config_value[key]).is_absolute():
            config_value[key] = posixpath.normpath(str(
                PurePosixPath(manifest["config_path"]).parent / config_value[key],
            ))
    if "response_policy" not in config:
        config_value.pop("response_policy", None)
    if config_value != config:
        raise ValueError("configuration file/object mismatch")
    original_policy = manifest.get("response_policy", config.get("response_policy", "strict_json_v1"))
    if original_policy not in POLICIES or original_policy != recorded.response_policy:
        raise ValueError("original response policy/config mismatch")
    expected_model = {"id": recorded.model_id, "revision": recorded.revision, "type": "instruction_tuned"}
    if manifest.get("model") != expected_model or manifest.get("tokenizer") != {
        "id": recorded.model_id, "revision": recorded.revision,
    }:
        raise ValueError("model/tokenizer/config provenance mismatch")
    decoding = manifest.get("decoding")
    if decoding != {"do_sample": False, "num_beams": 1, "batch_size": recorded.batch_size,
                    "seed": recorded.seed, "max_new_tokens": recorded.max_new_tokens}:
        raise ValueError("decoding/config provenance mismatch")
    return recorded


def _inventory(directory: Path, manifest: dict[str, Any], variants: list[str]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]],
]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(variants) + 1:
        raise ValueError("invalid or missing artifact inventory")
    expected = {f"{variant}.jsonl": ("predictions", variant) for variant in variants}
    expected["completions.jsonl"] = ("completions", None)
    if manifest.get("prediction_files") != {v: f"{v}.jsonl" for v in variants} or (
        manifest.get("completion_file") != "completions.jsonl"
        or manifest.get("manifest_file") != "run_manifest.json"
    ):
        raise ValueError("artifact filename declarations do not match the inventory contract")
    if {path.name for path in directory.iterdir()} != set(expected) | {"run_manifest.json"}:
        raise ValueError("inference directory does not exactly match artifact inventory")
    predictions, completions, seen = [], [], set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("invalid artifact inventory entry")
        name = _basename(artifact.get("path"))
        if name in seen or name not in expected:
            raise ValueError("duplicate or unexpected artifact inventory path")
        seen.add(name)
        kind, variant = expected[name]
        if artifact.get("kind") != kind or artifact.get("model_variant") != variant:
            raise ValueError("artifact inventory kind/variant mismatch")
        provenance = _file(directory / name)
        if provenance["sha256"] != _checked_hash(artifact.get("sha256"), "artifact"):
            raise ValueError(f"artifact file hash mismatch: {name}")
        for field in ("size_bytes", "n_records"):
            if type(artifact.get(field)) is not int or artifact[field] < 0:
                raise ValueError(f"invalid artifact {field}: {name}")
        records = _records(directory / name)
        if artifact["size_bytes"] != provenance["size_bytes"] or artifact["n_records"] != len(records):
            raise ValueError(f"artifact size or record count mismatch: {name}")
        if kind == "predictions":
            if any(record.get("model_variant") != variant for record in records):
                raise ValueError("prediction record stored in the wrong variant artifact")
            predictions.extend(records)
        else:
            completions.extend(records)
    return predictions, completions


def _sources(bundle: Path, manifest: dict[str, Any], prompt_dir: Path | None) -> dict[str, Any]:
    revision_file = bundle / "git-revision.txt"
    revision = None
    if revision_file.exists():
        _file(revision_file)
        revision = revision_file.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("invalid recorded generation Git revision")
    records = manifest.get("source_files")
    if not isinstance(records, list) or len(records) != len(SOURCE_PATHS):
        raise ValueError("invalid generation source inventory")
    matched = set()
    source_checks = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid generation source entry")
        paths = [path for path in SOURCE_PATHS if record["path"].endswith("/" + path)]
        if len(paths) != 1 or paths[0] in matched:
            raise ValueError("unknown or duplicate generation source path")
        relative = paths[0]
        matched.add(relative)
        current = _file(ROOT / relative)
        expected_hash = _checked_hash(record.get("sha256"), "generation source")
        matches_current = expected_hash == current["sha256"] and record.get("size_bytes") == current["size_bytes"]
        if relative not in VALIDATOR_UPGRADES and not matches_current:
            raise ValueError(f"prompt/schema/evidence implementation changed since generation: {relative}")
        method = "current_checkout"
        if not matches_current:
            if revision is None:
                raise ValueError(f"historical generation source cannot be verified without git-revision.txt: {relative}")
            try:
                historic = subprocess.run(
                    ["git", "show", f"{revision}:{relative}"], cwd=ROOT,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=10,
                ).stdout
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError(f"historical generation source is unavailable: {relative}") from exc
            if hashlib.sha256(historic).hexdigest() != expected_hash or len(historic) != record.get("size_bytes"):
                raise ValueError(f"historical generation source hash mismatch: {relative}")
            method = "recorded_git_revision"
        source_checks.append({"repository_path": relative, "method": method, "verified": True})
    template_records = manifest.get("prompt_template_files")
    if not isinstance(template_records, list) or len(template_records) != len(TEMPLATE_FILENAMES):
        raise ValueError("invalid prompt template inventory")
    directory = prompt_dir or DEFAULT_TEMPLATE_DIR
    templates = set()
    for record in template_records:
        if not isinstance(record, dict):
            raise ValueError("invalid prompt template entry")
        variant = record.get("model_variant")
        filenames = {variant.value: filename for variant, filename in TEMPLATE_FILENAMES.items()}
        if variant in templates or variant not in filenames:
            raise ValueError("unknown or duplicate prompt template variant")
        if (not isinstance(record.get("path"), str)
                or PurePosixPath(record["path"]).name != filenames[variant]):
            raise ValueError("prompt template filename/variant mismatch")
        templates.add(variant)
        local = directory / filenames[variant]
        current = _file(local)
        if record.get("sha256") != current["sha256"] or record.get("size_bytes") != current["size_bytes"]:
            raise ValueError(f"prompt template hash mismatch: {variant}")
    return {"generation_git_revision": revision, "generation_source_checks": source_checks}


def _generation_metadata(
    request: dict[str, Any], completion: dict[str, Any], config: LLMConfig, backend: str,
) -> None:
    if "metadata_error_fields" in completion:
        raise ValueError("invalid original generation metadata cannot be revalidated")
    for field in ("input_tokens", "output_tokens"):
        value = completion.get(field)
        if (type(value) is not int or value <= 0
                or type(request.get(field)) is not int or request[field] != value):
            raise ValueError(f"invalid or inconsistent generation {field}")
    if completion["output_tokens"] > config.max_new_tokens or (
        completion["input_tokens"] + config.max_new_tokens > config.context_window_tokens
    ):
        raise ValueError("generation token counts violate recorded context/output budget")
    for field in ("token_count_method", "finish_reason", "completion_sha256", "prompt_sha256", "status"):
        if field not in request or request[field] != completion.get(field):
            raise ValueError(f"request/completion {field} mismatch")
    if not isinstance(completion.get("token_count_method"), str) or not completion["token_count_method"].strip():
        raise ValueError("invalid original token count method")
    if backend == "hf" and (completion["token_count_method"] != "tokenizer" or completion["finish_reason"] != "eos"):
        raise ValueError("HF replay requires tokenizer counts and normal EOS completion")
    if completion.get("finish_reason") not in (None, "eos"):
        raise ValueError("truncated or invalid completion cannot establish replay acceptance")
    for field in ("preparation_s", "batch_latency_s", "latency_s"):
        value = request.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid original generation timing: {field}")


def _batches(parent: dict[str, Any], config: LLMConfig) -> None:
    batches, requests = parent.get("batches"), parent["requests"]
    if not isinstance(batches, list) or not batches:
        raise ValueError("missing original generation batches")
    seen = set()
    for number, batch in enumerate(batches):
        if not isinstance(batch, dict) or type(batch.get("batch_id")) is not int or batch["batch_id"] != number:
            raise ValueError("invalid or duplicate original batch identity")
        indices = batch.get("request_indices")
        if (not isinstance(indices, list) or not indices or len(indices) > config.batch_size
                or any(type(index) is not int or not 0 <= index < len(requests) for index in indices)
                or len(set(indices)) != len(indices) or seen.intersection(indices)):
            raise ValueError("invalid, missing, or duplicate original batch request indices")
        seen.update(indices)
        latency = batch.get("latency_s")
        if (not isinstance(latency, (int, float)) or isinstance(latency, bool)
                or not math.isfinite(latency) or latency < 0
                or batch.get("latency_scope") != "shared_batch_generation_wall_time"
                or type(batch.get("batch_size")) is not int or batch["batch_size"] != len(indices)
                or batch.get("status_counts") != dict(Counter(requests[i]["status"] for i in indices))):
            raise ValueError("inconsistent original batch metadata")
        for index in indices:
            request = requests[index]
            if (type(request.get("batch_id")) is not int or request["batch_id"] != number
                    or type(request.get("batch_size")) is not int or request["batch_size"] != len(indices)
                    or request.get("batch_latency_s") != latency):
                raise ValueError("original request/batch metadata mismatch")
    if seen != set(range(len(requests))):
        raise ValueError("original generation batches do not cover every request")


def revalidate_bundle(
    bundle_dir: str | Path, output_dir: str | Path, *,
    response_policy: str = "json_or_fenced_v1", prompt_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a portable original bundle, then publish a separate CPU replay.

    ``bundle_dir`` contains the unchanged config, ``input/``, and original
    ``inference/`` artifacts. Setup/provenance failures raise before publication;
    genuine response errors are retained in a published incomplete replay.
    """
    if response_policy not in POLICIES:
        raise ValueError(f"unsupported response policy: {response_policy}")
    supplied = Path(bundle_dir)
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValueError("--bundle-dir must be a regular non-symlink directory")
    bundle, output = supplied.resolve(), Path(output_dir).resolve()
    if bundle == output or bundle in output.parents or output in bundle.parents:
        raise ValueError("derived output must not overlap the original bundle")
    if Path(output_dir).is_symlink():
        raise ValueError("derived output must not be a symlink")
    helper = _helpers()
    helper._validate_output(output)
    inference, inputs = bundle / "inference", bundle / "input"
    if any(path.is_symlink() or not path.is_dir() for path in (inference, inputs)):
        raise ValueError("original input/inference must be regular non-symlink directories")
    parent_file = inference / "run_manifest.json"
    parent_provenance = _file(parent_file)
    parent = _json(parent_file.read_text(encoding="utf-8"))
    if not isinstance(parent, dict) or parent.get("schema_version") != 2:
        raise ValueError("revalidation requires an original schema-version-2 run manifest")
    if "revalidation" in parent:
        raise ValueError("revalidate the original generation bundle, not a previous derived replay")
    backend = parent.get("backend")
    if backend not in ("hf", "stub") or parent.get("synthetic") is not (backend == "stub"):
        raise ValueError("backend/synthetic generation provenance mismatch")
    variants = parent.get("variants")
    if (not isinstance(variants, list) or not variants
            or any(not isinstance(v, str) or v not in helper.VARIANTS for v in variants)
            or len(set(variants)) != len(variants)):
        raise ValueError("invalid original variants")
    config = _configuration(bundle, parent)
    directory = Path(prompt_dir).resolve() if prompt_dir is not None else None
    if config.prompt_dir is not None and directory is None:
        raise ValueError("custom original templates require explicit --prompt-dir on this machine")
    source_checks = _sources(bundle, parent, directory)
    predictions, completions = _inventory(inference, parent, variants)
    for path in inputs.iterdir():
        if path.is_symlink():
            raise ValueError("original BT inputs must not contain symlinks")
    trees, input_provenance = helper.load_inputs(inputs)
    expected_inputs = parent.get("input_files")
    if not isinstance(expected_inputs, list) or len(expected_inputs) != len(input_provenance):
        raise ValueError("missing or inconsistent original input inventory")
    expected_input_map = {}
    for item in expected_inputs:
        if (not isinstance(item, dict) or not isinstance(item.get("pair_id"), str)
                or item["pair_id"] in expected_input_map):
            raise ValueError("invalid or duplicate original input identity")
        expected_input_map[item["pair_id"]] = item
    for item in input_provenance:
        original = expected_input_map.get(item["pair_id"])
        if (original is None or original.get("sha256") != item["sha256"]
                or PurePosixPath(original["path"]).name != Path(item["path"]).name):
            raise ValueError("original input file hash/identity mismatch")
    expected = {(tree.metadata.pair_id, variant) for tree in trees for variant in variants}
    generated = _index(completions, "completions")
    predicted = _index(predictions, "predictions")
    requested = _index(parent.get("requests"), "requests")
    if set(generated) != expected or set(predicted) != expected or set(requested) != expected:
        raise ValueError("missing or unexpected pair/variant requests, predictions, or completions")
    if any(type(parent.get(key)) is not int or parent[key] != value for key, value in (
        ("n_inputs", len(trees)), ("n_predictions", len(expected)), ("n_completions", len(expected)),
    )):
        raise ValueError("manifest input/prediction/completion count mismatch")
    if parent.get("status_counts") != dict(Counter(item["status"] for item in predictions)):
        raise ValueError("original prediction status counts mismatch")
    provenance = parent.get("backend_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("missing original backend provenance")
    if backend == "hf" and provenance.get("weights_loaded") is not True:
        raise ValueError("HF generation weights were not verifiably loaded")
    if backend == "hf" and provenance.get("tokenizer_loaded", True) is not True:
        raise ValueError("HF generation tokenizer was not verifiably loaded")
    if backend == "hf":
        for field in ("model", "tokenizer"):
            recorded = provenance.get(field)
            if recorded is not None and (
                not isinstance(recorded, dict) or recorded.get("id") != config.model_id
                or recorded.get("requested_revision") != config.revision
                or recorded.get("resolved_revision") not in (None, config.revision)
            ):
                raise ValueError(f"backend {field} checkpoint provenance mismatch")
        for field in ("device", "dtype", "batch_size", "quantization"):
            if field in provenance and provenance[field] != getattr(config, field):
                raise ValueError(f"backend {field}/config provenance mismatch")
    _batches(parent, config)
    original_policy = parent.get("response_policy", parent["config"].get("response_policy", "strict_json_v1"))
    replay_predictions, replay_completions, replay_requests = [], [], []
    by_pair = {tree.metadata.pair_id: tree for tree in trees}
    for original_request in parent["requests"]:
        key = (original_request["pair_id"], original_request["model_variant"])
        completion, original_prediction = generated[key], predicted[key]
        if any(item.get("backend") != backend for item in (completion, original_prediction)):
            raise ValueError("request backend identity mismatch")
        if original_prediction.get("status") != completion.get("status") or completion.get("status") not in ("ok", "abstain", "error"):
            raise ValueError("original prediction/completion status mismatch")
        _generation_metadata(original_request, completion, config, backend)
        text = completion.get("text")
        if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != completion["completion_sha256"]:
            raise ValueError("raw completion text hash mismatch")
        prompt = build_prompt(by_pair[key[0]], key[1], template_dir=directory)
        if (completion.get("messages") != prompt.messages
                or completion.get("prompt_sha256") != prompt.prompt_sha256
                or original_request.get("template_sha256") != prompt.template_sha256):
            raise ValueError("original prompt/messages/template provenance mismatch")
        if (("response_policy" in parent or "response_normalization" in completion)
                and completion.get("response_normalization") != response_normalization(text, original_policy)):
            raise ValueError("original response normalization provenance mismatch")
        # Every original outcome must reproduce before a new policy is applied.
        # This excludes unrelated generation failures from being reclassified
        # as recoverable formatting failures just because their text parses.
        try:
            original_parsed = parse_response(text, prompt, key[0], backend, response_policy=original_policy)
        except (ValueError, TypeError) as exc:
            expected_error = {"pair_id": key[0], "model_variant": key[1],
                              "predicted_failure_mode": None, "predicted_subtask": None,
                              "explanation": f"Inference rejected ({type(exc).__name__}): {str(exc)[:240]}",
                              "evidence": [], "status": "error", "backend": backend}
            if (original_prediction != expected_error or any(
                item.get("error_type") != type(exc).__name__ or item.get("error_message") != str(exc)[:1000]
                for item in (original_request, completion)
            )):
                raise ValueError("original parsing error does not reproduce from raw text") from exc
        else:
            if original_parsed != original_prediction or any(
                field in item for item in (original_request, completion)
                for field in ("error_type", "error_message")
            ):
                raise ValueError("original successful prediction does not reproduce from raw text")
        request, replay = copy.deepcopy(original_request), copy.deepcopy(completion)
        for record in (request, replay):
            record["original_validation"] = {field: record[field] for field in (
                "status", "error_type", "error_message",
            ) if field in record}
            record.pop("error_type", None)
            record.pop("error_message", None)
        normalization = response_normalization(text, response_policy)
        try:
            prediction = parse_response(text, prompt, key[0], backend, response_policy=response_policy)
        except (ValueError, TypeError) as exc:
            prediction = {"pair_id": key[0], "model_variant": key[1],
                          "predicted_failure_mode": None, "predicted_subtask": None,
                          "explanation": f"Inference rejected ({type(exc).__name__}): {str(exc)[:240]}",
                          "evidence": [], "status": "error", "backend": backend}
            for record in (request, replay):
                record.update(error_type=type(exc).__name__, error_message=str(exc)[:1000])
        for record in (request, replay):
            record.update(status=prediction["status"], response_normalization=normalization)
        replay_predictions.append(prediction)
        replay_completions.append(replay)
        replay_requests.append(request)
    original_summary = helper.execution_summary(
        selected_backend=backend, trees=trees, variants=variants,
        predictions=predictions, completions=completions, manifest=parent,
        prompt_dir=directory,
    )
    if (parent.get("execution") != original_summary["execution"]
            or not isinstance(parent.get("acceptance"), dict)
            or any(parent["acceptance"].get(key) != original_summary["acceptance"][key]
                   for key in ("scope", "status", "nov6_baselines_covered"))):
        raise ValueError("original execution/acceptance does not reproduce from retained artifacts")
    result = copy.deepcopy(parent)
    result.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        response_policy=response_policy,
        output_dir=str(output), requests=replay_requests,
        original_acceptance=copy.deepcopy(parent.get("acceptance")),
        original_execution=copy.deepcopy(parent.get("execution")),
        original_requests=copy.deepcopy(parent["requests"]),
        original_batches=copy.deepcopy(parent.get("batches")),
        original_response_processing=copy.deepcopy(parent.get("response_processing")),
        original_prediction_status_counts=copy.deepcopy(parent["status_counts"]),
        status_counts=dict(Counter(record["status"] for record in replay_predictions)),
        validation_source_files=[_file(Path(__file__)), _file(ROOT / "scripts/revalidate_llm.py")]
            + helper.implementation_provenance(directory)["source_files"],
        revalidation={
            "kind": "offline_response_revalidation",
            "parent_manifest": parent_provenance,
            "parent_artifacts": copy.deepcopy(parent["artifacts"]),
            "original_response_policy": original_policy,
            "response_policy": response_policy,
            "newly_generated_requests": 0, "model_loaded_this_process": False,
            "prior_generation_verified": True,
            "original_acceptance": copy.deepcopy(parent.get("acceptance")),
            "original_status_counts": copy.deepcopy(parent["status_counts"]),
            "local_input_files": input_provenance,
            "local_config_file": _file(bundle / PurePosixPath(parent["config_path"]).name),
            **source_checks,
            "limitations": [
                "Hashes and retained metadata establish internal provenance, not cryptographic proof of model execution.",
                "Response validation does not establish causal explanation correctness or diagnosis accuracy.",
            ],
        },
    )
    result["response_processing"] = {
        "policy": response_policy, "raw_text_preserved": True,
        "fenced_requests": sum(record["response_normalization"]["envelope"] == "json_code_fence"
                               for record in replay_completions),
    }
    for batch in result.get("batches", []):
        batch["status_counts"] = dict(Counter(replay_requests[i]["status"] for i in batch["request_indices"]))
    result.update(helper.execution_summary(
        selected_backend=backend, trees=trees, variants=variants,
        predictions=replay_predictions, completions=replay_completions,
        manifest=result, prompt_dir=directory,
    ))
    result["execution"].update(phase="offline_response_revalidation", newly_generated_requests=0,
                               model_loaded_this_process=False)
    result["acceptance"]["validation_phase"] = "offline_response_revalidation"
    result["evaluation_note"] = (
        "CPU-only response revalidation of retained generation; no inference was run. "
        "Original strict acceptance is retained separately. Explanation quality requires independent review."
    )
    helper.publish_run(output, replay_predictions, variants, result, completions=replay_completions)
    return _json((output / "run_manifest.json").read_text(encoding="utf-8"))
