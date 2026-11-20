"""Measure one worst-case frozen request before admitting a full CUDA run.

Kernel qualification alone does not include checkpoint or KV-cache memory.
This gate uses the longest fitting prepared prompt, unchanged inference
settings, retained raw output, independent parsing, and measured CUDA peaks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from telemetry_reasoning import experiments as exp
from telemetry_reasoning.models.llm import (
    HFBackend, MEMORY_POLICY, _hash, execution_policy, load_llm_config,
    parse_response, response_normalization, run_inference,
)
from telemetry_reasoning.models.prompts import build_prompt
from telemetry_reasoning.schemas import PairedBehaviorTree


RECEIPT = "qualification_receipt.json"


def _script(name):
    path = exp.ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location("telemetry_qualification_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _profile(config):
    if (config.device != "cuda" or config.dtype != "bfloat16" or config.batch_size != 1
            or config.quantization != "none" or config.prefill_chunk_size is None
            or config.attention_implementation != "sdpa" or not config.fused_attention_only
            or config.decoding_constraint != "observed_json_v1" or not config.local_files_only):
        raise ValueError("Qualification requires the cache-only BF16 CUDA v3 profile: bounded prefill, fused SDPA, observed_json_v1, batch_size=1")


def _selected(manifest, config):
    if manifest.get("status") not in ("ready_for_gpu", "ready_for_gpu_with_context_refusals"):
        raise ValueError("Experiment is not ready for GPU execution")
    rows = manifest.get("token_budget")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Experiment has no token-budget preflight")
    identities, fitting = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid token-budget row")
        key = (row.get("task"), row.get("pair_id"), row.get("model_variant"))
        if (key[0] not in exp.TASKS or not isinstance(key[1], str) or not key[1]
                or key[2] not in exp.VARIANTS or key in identities
                or type(row.get("input_tokens")) is not int or row["input_tokens"] <= 0
                or row.get("reserved_output_tokens") != config.max_new_tokens
                or row.get("context_window_tokens") != config.context_window_tokens
                or type(row.get("fits")) is not bool
                or row["fits"] != (row["input_tokens"] + config.max_new_tokens <= config.context_window_tokens)):
            raise ValueError("Invalid, duplicate, or inconsistent token-budget request")
        identities.add(key)
        if row["fits"]:
            fitting.append(row)
    if not fitting:
        raise ValueError("No fitting requests are available for qualification")
    return dict(sorted(fitting, key=lambda row: (-row["input_tokens"], row["task"], row["pair_id"], row["model_variant"]))[0])


def _prompt(bundle, selected, config):
    path = exp._safe(bundle, f"tasks/{selected['task']}/bt/pair_{selected['pair_id']}_bt.json")
    tree = PairedBehaviorTree.model_validate_json(path.read_bytes())
    if tree.metadata.pair_id != selected["pair_id"] or tree.metadata.task.value != selected["task"]:
        raise ValueError("Selected BT identity differs from its token preflight")
    prompt = build_prompt(tree, selected["model_variant"], template_dir=config.prompt_dir)
    if prompt.prompt_sha256 != selected.get("prompt_sha256"):
        raise ValueError("Selected prompt differs from its token preflight")
    return tree, prompt, path


def _positive_int(value):
    return type(value) is int and value > 0


def _validate_kernel(report, config):
    if (report.get("schema_version") != 1 or report.get("probe_kind") != "llama31_bfloat16_sdpa_kernel_v1"
            or report.get("status") != "passed" or report.get("model_weights_loaded") is not False
            or report.get("network_access_requested") is not False or report.get("math_backend_enabled") is not False
            or report.get("chunk_tokens") != config.prefill_chunk_size
            or type(report.get("prefix_tokens")) is not int
            or report["prefix_tokens"] < config.context_window_tokens):
        raise ValueError("A passed fused-only kernel probe covering the configured prefill and context is required")
    probe = _script("probe_cuda_inference.py")
    planned = {case.name: case.description() for case in probe.probe_cases(report["chunk_tokens"], report["prefix_tokens"])}
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != len(planned):
        raise ValueError("Kernel report has incomplete case coverage")
    observed = set()
    for row in cases:
        if not isinstance(row, dict) or row.get("name") not in planned or row["name"] in observed:
            raise ValueError("Kernel report has invalid or duplicate cases")
        observed.add(row["name"])
        operators = row.get("sdpa_operators", [])
        fused = ("scaled_dot_product_flash", "scaled_dot_product_efficient", "scaled_dot_product_cudnn", "cudnn_attention")
        if (any(row.get(key) != value for key, value in planned[row["name"]].items())
                or row.get("status") != "passed" or row.get("finite_output") is not True
                or not _positive_int(row.get("peak_allocated_bytes"))
                or not _positive_int(row.get("peak_reserved_bytes"))
                or row["peak_reserved_bytes"] < row["peak_allocated_bytes"]
                or not isinstance(operators, list) or not all(isinstance(value, str) for value in operators)
                or any("attention_math" in value for value in operators)
                or not any(any(name in value for name in fused) for value in operators)):
            raise ValueError("Kernel report case lacks measured successful fused execution")
    device, runtime = report.get("device", {}), report.get("runtime", {})
    if (not isinstance(device, dict) or type(device.get("index")) is not int or device["index"] < 0
            or not isinstance(device.get("name"), str) or not device["name"]
            or not isinstance(device.get("capability"), list) or len(device["capability"]) != 2
            or not all(type(value) is int and value >= 0 for value in device["capability"])
            or device.get("bfloat16_supported") is not True
            or not isinstance(runtime, dict) or not all(isinstance(runtime.get(key), str) and runtime[key]
                                                        for key in ("torch", "cuda", "python", "platform"))
            or report.get("transformers_package_version") != "5.23.0"):
        raise ValueError("Kernel report lacks pinned runtime/device identity")
    recorded_source = report.get("probe_script")
    if recorded_source is not None and recorded_source != exp._fingerprint(exp.ROOT / "scripts/probe_cuda_inference.py"):
        raise ValueError("Kernel report probe implementation differs from the current script")
    return "source_fingerprint_verified" if recorded_source is not None else "original_v1_report_has_no_source_fingerprint"


def _device_state(backend):
    torch = backend._torch
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info()
    return {
        "device": {"index": int(device), "name": torch.cuda.get_device_name(device),
                   "capability": list(torch.cuda.get_device_capability(device)),
                   "bfloat16_supported": bool(torch.cuda.is_bf16_supported())},
        "cuda_runtime": str(torch.version.cuda),
        "free_memory_bytes": int(free), "total_memory_bytes": int(total),
    }


def _binding(probe, state, provenance=None):
    if (state.get("device") != probe["device"] or state.get("cuda_runtime") != probe["runtime"]["cuda"]
            or not _positive_int(state.get("total_memory_bytes"))
            or type(state.get("free_memory_bytes")) is not int
            or not 0 <= state["free_memory_bytes"] <= state["total_memory_bytes"]
            or any(row.get("total_memory_bytes") != state["total_memory_bytes"] for row in probe["cases"])):
        raise ValueError("HF execution uses a different CUDA device/runtime from the passed kernel probe")
    if provenance is not None:
        runtime = provenance.get("runtime", {})
        if (runtime.get("torch") != probe["runtime"]["torch"]
                or runtime.get("transformers") != probe["transformers_package_version"]
                or runtime.get("platform") != probe["runtime"]["platform"]):
            raise ValueError("HF execution runtime differs from the passed kernel probe")


def _validate_inference(bundle, root, selected, config, probe, state):
    _, prompt, tree_path = _prompt(bundle, selected, config)
    directory = root / "inference"
    manifest = exp._json(directory / "run_manifest.json")
    def lines(path):
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(json.loads(line, object_pairs_hook=_unique, parse_constant=_nonfinite))
        return result
    completions = lines(directory / "completions.jsonl")
    predictions = lines(directory / f"{selected['model_variant']}.jsonl")
    requests = manifest.get("requests")
    if (manifest.get("backend") != "hf" or manifest.get("synthetic") is not False
            or manifest.get("n_inputs") != 1 or manifest.get("n_predictions") != 1
            or not isinstance(requests, list) or len(requests) != 1
            or len(completions) != 1 or len(predictions) != 1):
        raise ValueError("Qualification requires exactly one actual HF request and completion")
    request, completion, prediction = requests[0], completions[0], predictions[0]
    for record in (request, completion, prediction):
        if (record.get("pair_id"), record.get("model_variant")) != (selected["pair_id"], selected["model_variant"]):
            raise ValueError("Qualification request identity changed")
    if any(record.get(key) for record in (request, completion) for key in ("error_type", "error_message")):
        raise ValueError("Qualification retained an inference error")
    if (request.get("input_tokens") != selected["input_tokens"]
            or request.get("prompt_sha256") != prompt.prompt_sha256
            or request.get("template_sha256") != prompt.template_sha256
            or completion.get("input_tokens") != selected["input_tokens"]
            or completion.get("messages") != prompt.messages
            or completion.get("prompt_sha256") != prompt.prompt_sha256
            or completion.get("backend") != "hf" or completion.get("metadata_error_fields")
            or not isinstance(completion.get("text"), str)
            or completion.get("completion_sha256") != hashlib.sha256(completion["text"].encode()).hexdigest()
            or completion.get("response_normalization") != response_normalization(completion["text"], config.response_policy)
            or request.get("completion_sha256") != completion.get("completion_sha256")
            or any(record.get("finish_reason") != "eos" for record in (request, completion))
            or not _positive_int(completion.get("output_tokens"))
            or completion["output_tokens"] > config.max_new_tokens
            or request.get("output_tokens") != completion["output_tokens"]
            or any(record.get("token_count_method") != "tokenizer" for record in (request, completion))):
        raise ValueError("Qualification lacks matching token counts, exact prompt/raw text, or a complete EOS-terminated response")
    parsed = parse_response(completion["text"], prompt, selected["pair_id"], "hf", response_policy=config.response_policy)
    if (parsed != prediction or parsed["status"] not in ("ok", "abstain")
            or request.get("status") != parsed["status"] or completion.get("status") != parsed["status"]):
        raise ValueError("Qualification response does not independently reproduce schema/evidence acceptance")
    saved_config = dict(manifest.get("config", {}))
    if (manifest.get("config_sha256") != _hash(saved_config)
            or manifest.get("config_file_sha256") != exp._fingerprint(bundle / "configs/llm.yaml")["sha256"]):
        raise ValueError("Qualification saved configuration hashes changed")
    current_config = config.model_dump(mode="json")
    # Paths resolve on the machine that produced the receipt. Content and model
    # identities are verified separately so CPU return verification stays portable.
    for key in ("prompt_dir", "cache_dir"):
        if bool(saved_config.get(key)) != bool(current_config.get(key)):
            raise ValueError("Qualification configuration path policy changed")
        saved_config[key] = current_config[key]
    if saved_config != current_config:
        raise ValueError("Qualification inference configuration differs from the frozen bundle")
    inputs = manifest.get("input_files")
    if (not isinstance(inputs, list) or len(inputs) != 1
            or not _record_binds(inputs[0], tree_path)):
        raise ValueError("Qualification BT source fingerprint changed")
    implementation = _script("run_llm.py").implementation_provenance(config.prompt_dir)
    for field in ("source_files", "prompt_template_files"):
        records, current = manifest.get(field), implementation[field]
        if (not isinstance(records, list) or len(records) != len(current)
                or any(not _record_binds(record, Path(expected["path"]))
                       or record.get("model_variant") != expected.get("model_variant")
                       for record, expected in zip(records, current))):
            raise ValueError("Qualification inference/template source fingerprints changed")
    artifacts = manifest.get("artifacts")
    expected_artifacts = [("predictions", f"{selected['model_variant']}.jsonl"), ("completions", "completions.jsonl")]
    if (not isinstance(artifacts, list) or len(artifacts) != 2
            or any(not isinstance(record, dict) or record.get("kind") != kind
                   or record.get("path") != name or record.get("n_records") != 1
                   or not _record_binds(record, directory / name)
                   for record, (kind, name) in zip(artifacts, expected_artifacts))):
        raise ValueError("Qualification published inference artifact fingerprints changed")
    provenance = manifest.get("backend_provenance", {})
    resources = provenance.get("resources", {})
    if (provenance.get("implementation") != "transformers_batched_greedy_v3"
            or provenance.get("execution_policy") != execution_policy(config)
            or provenance.get("weights_loaded") is not True or provenance.get("tokenizer_loaded") is not True
            or provenance.get("device") != "cuda" or provenance.get("dtype") != config.dtype
            or provenance.get("batch_size") != 1 or provenance.get("quantization") != "none"
            or provenance.get("token_count_method") != "tokenizer"
            or not isinstance(resources, dict) or resources.get("policy") != MEMORY_POLICY
            or resources.get("host_admissible") is not True
            or resources.get("cuda_total_memory_bytes") != state.get("total_memory_bytes")
            or type(provenance.get("native_context_window_tokens")) is not int
            or provenance["native_context_window_tokens"] < config.context_window_tokens):
        raise ValueError("Qualification lacks actual v3 CUDA model execution provenance")
    for resource in ("model", "tokenizer"):
        identity = provenance.get(resource, {})
        if (identity.get("id") != config.model_id or identity.get("requested_revision") != config.revision
                or identity.get("resolved_revision") not in (None, config.revision)):
            raise ValueError("Qualification checkpoint/tokenizer identity differs from the frozen bundle")
    constraints, memory = provenance.get("constraints"), provenance.get("request_memory")
    expected_constraint = selected.get("output_constraint")
    if (not isinstance(constraints, list) or len(constraints) != 1
            or not isinstance(expected_constraint, dict)
            or constraints != [{"prompt_sha256": prompt.prompt_sha256, **expected_constraint}]
            or constraints[0].get("policy") != "observed_json_v1"
            or constraints[0].get("prompt_sha256") != prompt.prompt_sha256
            or constraints[0].get("repair") is not False
            or not _positive_int(constraints[0].get("worst_case_tokens_including_eos"))
            or constraints[0]["worst_case_tokens_including_eos"] > config.max_new_tokens
            or completion["output_tokens"] > constraints[0]["worst_case_tokens_including_eos"]):
        raise ValueError("Qualification lacks the bounded observed-JSON decoding receipt")
    if (not isinstance(memory, list) or len(memory) != 1 or memory[0].get("completed") is not True
            or memory[0].get("prompt_sha256") != [prompt.prompt_sha256]
            or memory[0].get("input_tokens") != [selected["input_tokens"]]
            or memory[0].get("prefill_chunk_size") != config.prefill_chunk_size
            or memory[0].get("measurement_error")
            or not _positive_int(memory[0].get("peak_allocated_bytes"))
            or not _positive_int(memory[0].get("peak_reserved_bytes"))
            or not memory[0]["peak_allocated_bytes"] <= memory[0]["peak_reserved_bytes"] <= state["total_memory_bytes"]):
        raise ValueError("Qualification requires a successful measured CUDA allocation peak for the selected request")
    _binding(probe, state, provenance)
    return provenance


def _record_binds(record, path):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return False
    expected = exp._fingerprint(path)
    return (Path(record["path"]).name == path.name
            and all(record.get(key) == value for key, value in expected.items()))


def _unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON: {value}")


def qualify_experiment(*, bundle, kernel_report, output_dir):
    bundle, kernel_report = Path(bundle).resolve(), Path(kernel_report).resolve()
    destination = exp._fresh(output_dir).resolve()
    for source in (bundle, kernel_report):
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError("Qualification output must be separate from its input trees")
    source_files = exp._sources()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gpu-qualification-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "qualification"
        staged.mkdir()
        receipt = {"qualification_version": 1, "status": "failed",
                   "created_utc": datetime.now(timezone.utc).isoformat(), "source_files": source_files,
                   "scope": "One longest fitting frozen request; no inference retries or prompt truncation",
                   "failures": []}
        try:
            manifest = exp.verify_experiment(bundle)
            receipt["bundle_manifest_sha256"] = exp._fingerprint(bundle / "bundle_manifest.json")["sha256"]
            config = load_llm_config(bundle / "configs/llm.yaml")
            _profile(config)
            selected = _selected(manifest, config)
            receipt["selected_request"] = selected
            tree, prompt, tree_path = _prompt(bundle, selected, config)
            probe_bytes = kernel_report.read_bytes()
            (staged / "kernel_report.json").write_bytes(probe_bytes)
            receipt["kernel_report_sha256"] = hashlib.sha256(probe_bytes).hexdigest()
            probe = exp._json(staged / "kernel_report.json")
            receipt["kernel_probe_source_binding"] = _validate_kernel(probe, config)
            for package, expected in (("torch", probe["runtime"]["torch"]),
                                      ("transformers", probe["transformers_package_version"])):
                if metadata.version(package) != expected:
                    raise ValueError(f"Installed {package} differs from the passed kernel probe")
            backend = HFBackend(config)
            count = backend.count_tokens(prompt)
            if type(count) is not int or count != selected["input_tokens"]:
                raise ValueError("Actual HF tokenizer count differs from the frozen CPU preflight; no weights opened")
            initial_device = _device_state(backend)
            _binding(probe, initial_device)
            result = run_inference([tree], [selected["model_variant"]], config, backend=backend)
            publisher = _script("run_llm.py")
            run_manifest = dict(result.manifest)
            run_manifest.update(input_files=[publisher._file_provenance(tree_path)],
                                config_file_sha256=exp._fingerprint(bundle / "configs/llm.yaml")["sha256"],
                                **publisher.implementation_provenance(config.prompt_dir))
            run_manifest.update(publisher.execution_summary(
                selected_backend="hf", trees=[tree], variants=[selected["model_variant"]],
                predictions=result.predictions, completions=result.completions,
                manifest=run_manifest, prompt_dir=config.prompt_dir))
            publisher.publish_run(staged / "inference", result.predictions, [selected["model_variant"]],
                                  run_manifest, completions=result.completions)
            receipt["device_state"] = _device_state(backend)
            receipt["backend_provenance"] = result.manifest.get("backend_provenance")
            _validate_inference(bundle, staged, selected, config, probe, receipt["device_state"])
            if exp.verify_experiment(bundle) != manifest or exp._sources() != source_files:
                raise ValueError("Qualification inputs or implementation changed during execution")
            if kernel_report.read_bytes() != probe_bytes:
                raise ValueError("Kernel report changed during qualification")
            receipt["status"] = "passed"
        except Exception as exc:
            receipt["failures"].append(f"{type(exc).__name__}: {str(exc)[:2000]}")
        receipt["files"] = exp._inventory(staged)
        exp._write(staged / RECEIPT, receipt)
        exp._fresh(destination)
        staged.rename(destination)
    return receipt


def verify_qualification(bundle, qualification_path):
    """Reproduce an immutable qualification on CPU; never open model weights."""
    bundle, path = Path(bundle).resolve(), Path(qualification_path)
    if path.is_symlink():
        raise ValueError("Symlink qualification path")
    if path.is_dir():
        path = path / RECEIPT
    if path.name != RECEIPT:
        raise ValueError(f"Qualification must refer to a directory or {RECEIPT}")
    root = path.parent
    manifest = exp.verify_experiment(bundle)
    receipt = exp._json(path)
    if (receipt.get("qualification_version") != 1 or receipt.get("status") != "passed"
            or receipt.get("failures") != [] or receipt.get("source_files") != exp._sources()
            or receipt.get("bundle_manifest_sha256") != exp._fingerprint(bundle / "bundle_manifest.json")["sha256"]):
        raise ValueError("Qualification did not pass for this unchanged bundle and implementation")
    expected = receipt.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Qualification has no immutable artifact inventory")
    for relative in expected:
        exp._safe(root, relative)
    if exp._inventory(root, (RECEIPT,)) != expected:
        raise ValueError("Qualification artifact inventory changed")
    config = load_llm_config(bundle / "configs/llm.yaml")
    _profile(config)
    selected = _selected(manifest, config)
    if receipt.get("selected_request") != selected:
        raise ValueError("Qualification did not execute the longest fitting request")
    if receipt.get("kernel_report_sha256") != exp._fingerprint(root / "kernel_report.json")["sha256"]:
        raise ValueError("Qualification kernel report hash changed")
    probe = exp._json(root / "kernel_report.json")
    if receipt.get("kernel_probe_source_binding") != _validate_kernel(probe, config):
        raise ValueError("Qualification probe source-binding declaration changed")
    provenance = _validate_inference(bundle, root, selected, config, probe, receipt.get("device_state", {}))
    if receipt.get("backend_provenance") != provenance:
        raise ValueError("Qualification receipt differs from measured HF provenance")
    return receipt
