"""Verify portable GPU completions and score them on CPU without model loading.

The GPU directory is immutable input. This module uses the same source-bound
bundle and strict response parser as inference; context refusals remain scored
requests. Synthetic runs are useful rehearsals and never establish HF acceptance.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

from telemetry_reasoning import experiments as exp
from telemetry_reasoning.eval.contracts import read_predictions
from telemetry_reasoning.models.llm import MEMORY_POLICY, _hash, execution_policy, load_llm_config, response_normalization
from telemetry_reasoning.models.prompts import TEMPLATE_FILENAMES


def _jsonl(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    rows, invalid = [], []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=unique, parse_constant=finite)
            if not isinstance(row, dict):
                raise ValueError("JSONL record must be an object")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            invalid.append({"line": line_number, "reason": str(exc)})
    return rows, invalid


def _coverage(rows, expected):
    if not isinstance(rows, list):
        raise ValueError("Inference records must be a list")
    counts = Counter()
    malformed = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            malformed.append({"record": number, "reason": "record must be an object"})
            continue
        key = row.get("pair_id"), row.get("model_variant")
        if not all(isinstance(value, str) for value in key):
            malformed.append({"record": number, "reason": "missing string pair_id/model_variant"})
        else:
            counts[key] += 1
    return {"record_count": len(rows), "expected_count": len(expected),
            "missing": [list(key) for key in sorted(expected - set(counts))],
            "unexpected": [list(key) for key in sorted(set(counts) - expected)],
            "duplicates": [{"key": list(key), "occurrences": count}
                           for key, count in sorted(counts.items()) if count > 1],
            "malformed": malformed}


def _task_accounting(task, target, manifest):
    expected = {(row["pair_id"], row["model_variant"]) for row in manifest["token_budget"]
                if row["task"] == task}
    fitting = {(row["pair_id"], row["model_variant"]) for row in manifest["token_budget"]
               if row["task"] == task and row["fits"]}
    parsed = read_predictions(*(target / "inference" / f"{variant}.jsonl" for variant in exp.VARIANTS))
    predictions = _coverage([row.model_dump(mode="json") for row in parsed.records], expected)
    predictions["invalid_lines"] = [
        {"source": Path(row.source).name, "line": row.line_no, "reason": row.message,
         "pair_id": row.pair_id, "model_variant": row.model_variant} for row in parsed.invalid]
    requests = _coverage(exp._json(target / "inference/run_manifest.json")["requests"], expected)
    completions, invalid = _jsonl(target / "inference/completions.jsonl")
    completion_coverage = _coverage(completions, fitting)
    completion_coverage["invalid_lines"] = invalid
    return {"predictions": predictions, "requests": requests, "completions": completion_coverage}


def _verify_inputs(bundle, results):
    try:
        manifest = exp.verify_experiment(bundle)
    except AttributeError as exc:
        raise ValueError("Malformed experiment bundle manifest") from exc
    receipt = exp._json(results / "experiment_receipt.json")
    if (not isinstance(receipt, dict) or receipt.get("run_version") != 1 or receipt.get("backend") not in ("hf", "stub")
            or receipt.get("bundle_manifest_sha256") != exp._fingerprint(bundle / "bundle_manifest.json")["sha256"]
            or receipt.get("split") != manifest["split"]
            or receipt.get("pair_counts") != manifest["pair_counts"]):
        raise ValueError("GPU receipt does not bind this backend, split, and exact experiment bundle")
    if results.is_symlink() or not isinstance(receipt.get("files"), dict):
        raise ValueError("GPU results require a regular source-bound file inventory")
    for name in receipt["files"]:
        exp._safe(results, name)
    if exp._inventory(results, ("experiment_receipt.json",)) != receipt["files"]:
        raise ValueError("GPU result inventory changed")
    config = load_llm_config(bundle / "configs/llm.yaml")
    if receipt["backend"] == "hf" and config.prefill_chunk_size is not None:
        if receipt.get("qualification_path") != "qualification/qualification_receipt.json":
            raise ValueError("GPU results lack the longest-request qualification evidence")
        qualification = exp._safe(results, receipt["qualification_path"])
        if exp._fingerprint(qualification)["sha256"] != receipt.get("qualification_receipt_sha256"):
            raise ValueError("GPU qualification receipt hash changed")
        from telemetry_reasoning.gpu_qualification import verify_qualification
        verify_qualification(bundle, qualification)
    stages = receipt.get("stages")
    if not isinstance(stages, list) or any(not isinstance(row, dict) for row in stages):
        raise ValueError("GPU receipt has no stage records")
    for task in exp.TASKS:
        stage = [row for row in stages if row.get("task") == task and row.get("stage") == "inference"]
        if len(stage) != 1 or stage[0].get("exit_code") not in (0, 1):
            raise ValueError(f"{task}: missing or failed GPU inference stage")
    return manifest, receipt


def _check_recorded_settings(bundle, target, manifest, backend, *, allow_response_errors=False):
    """Check configuration and accounting using stdlib, never a tokenizer import."""
    config = load_llm_config(bundle / "configs/llm.yaml")
    inference = exp._json(target / "inference/run_manifest.json")
    expected_config = config.model_dump(mode="json")
    recorded_config = inference.get("config", {})
    if not isinstance(recorded_config, dict):
        raise ValueError("Inference config must be an object")
    # Only the portable template directory changes across machines. Exact
    # template bytes and rendered prompts are checked separately by the parser.
    normalized = dict(recorded_config)
    prompt_dir = normalized.get("prompt_dir")
    if expected_config.get("prompt_dir") is not None:
        if not isinstance(prompt_dir, str) or PurePosixPath(prompt_dir).name != "prompts":
            raise ValueError("Inference template directory differs from the portable bundle")
        normalized["prompt_dir"] = expected_config["prompt_dir"]
    if (normalized != expected_config or inference.get("config_sha256") != _hash(recorded_config)
            or inference.get("backend") != backend or inference.get("synthetic") is not (backend == "stub")
            or inference.get("response_policy") != config.response_policy
            or inference.get("variants") != list(exp.VARIANTS)
            or inference.get("model") != {"id": config.model_id, "revision": config.revision,
                                          "type": "instruction_tuned"}
            or inference.get("tokenizer") != {"id": config.model_id, "revision": config.revision}):
        raise ValueError("Inference settings or model identity differ from the frozen bundle")
    expected_decoding = {"do_sample": False, "num_beams": 1,
                         "batch_size": config.batch_size if backend == "hf" else 1,
                         "seed": config.seed, "max_new_tokens": config.max_new_tokens}
    if inference.get("decoding") != expected_decoding:
        raise ValueError("Inference decoding settings differ from the frozen bundle")
    requests = {(row["pair_id"], row["model_variant"]): row for row in inference["requests"]}
    predictions = {(row.pair_id, row.model_variant): row for row in read_predictions(
        *(target / "inference" / f"{variant}.jsonl" for variant in exp.VARIANTS)).records}
    for key, request in requests.items():
        template = config.prompt_dir / TEMPLATE_FILENAMES[key[1]]
        if (request.get("template_sha256") != hashlib.sha256(template.read_bytes()).hexdigest()
                or type(request.get("input_tokens")) is not int or request["input_tokens"] <= 0
                or request.get("status") != predictions[key].status):
            raise ValueError("Request template hash, token count, or status differs from its prediction")
    completions, invalid = _jsonl(target / "inference/completions.jsonl")
    if invalid:
        raise ValueError("Invalid completion JSONL records")
    for completion in completions:
        request = requests[(completion["pair_id"], completion["model_variant"])]
        count = completion.get("output_tokens")
        status = completion.get("status")
        error_fields = ("error_type", "error_message")
        if status == "error":
            if (not allow_response_errors or any(
                    not isinstance(completion.get(field), str) or not completion[field].strip()
                    or completion[field] != request.get(field) for field in error_fields)):
                raise ValueError("Response error annotation differs from its independently verified request")
        elif any(field in completion or field in request for field in error_fields):
            raise ValueError("A successful completion cannot carry response error annotations")
        if (type(count) is not int or not 0 <= count <= config.max_new_tokens
                or completion.get("input_tokens") != request.get("input_tokens")
                or count != request.get("output_tokens")
                or completion.get("backend") != backend
                or completion.get("status") != request.get("status")
                or status not in (("ok", "abstain", "error") if allow_response_errors else ("ok", "abstain"))
                or "metadata_error_fields" in completion
                or completion.get("completion_sha256") != request.get("completion_sha256")
                or completion.get("finish_reason") != request.get("finish_reason")
                or not (completion.get("finish_reason") is None or isinstance(completion["finish_reason"], str))
                or not isinstance(completion.get("token_count_method"), str)
                or not completion["token_count_method"].strip()
                or completion["token_count_method"] != request.get("token_count_method")
                or (backend == "hf" and completion["token_count_method"] != "tokenizer")
                or completion.get("response_normalization") != response_normalization(
                    completion["text"], config.response_policy)):
            raise ValueError("Completion token accounting, backend, or normalization differs from its request")
    if backend == "hf" and completions:
        provenance = inference.get("backend_provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("GPU backend provenance must be an object")
        resources, runtime = provenance.get("resources"), provenance.get("runtime")
        if (provenance.get("weights_loaded") is not True or provenance.get("device") != "cuda"
                or provenance.get("tokenizer_loaded") is not True
                or provenance.get("implementation") != (
                    "transformers_batched_greedy_v3" if config.prefill_chunk_size is not None
                    or config.decoding_constraint != "none" else "transformers_batched_greedy_v2")
                or provenance.get("token_count_method") != "tokenizer"
                or provenance.get("dtype") != config.dtype
                or provenance.get("batch_size") != config.batch_size
                or provenance.get("quantization") != config.quantization
                or not isinstance(resources, dict) or not isinstance(runtime, dict)
                or resources.get("policy") != MEMORY_POLICY or resources.get("host_admissible") is not True
                or any(not isinstance(runtime.get(name), str) or not runtime[name].strip()
                       for name in ("torch", "transformers"))
                or type(provenance.get("native_context_window_tokens")) is not int
                or provenance["native_context_window_tokens"] < config.context_window_tokens):
            raise ValueError("Real GPU inference requires recorded CUDA weight loading")
        if config.prefill_chunk_size is not None or config.decoding_constraint != "none":
            if provenance.get("execution_policy") != execution_policy(config):
                raise ValueError("GPU execution policy differs from the frozen bundle")
        fitting = [row for row in manifest["token_budget"]
                   if row["task"] == target.name and row["fits"]]
        if config.decoding_constraint == "observed_json_v1":
            expected_constraints = [{"prompt_sha256": row["prompt_sha256"], **row["output_constraint"]}
                                    for row in fitting]
            actual_constraints = provenance.get("constraints")
            if (not isinstance(actual_constraints, list) or
                    Counter(_hash(row) for row in actual_constraints) !=
                    Counter(_hash(row) for row in expected_constraints)):
                raise ValueError("GPU output constraints differ from CPU-compiled observation choices")
            if any(row.get("finish_reason") != "eos" for row in completions):
                raise ValueError("Constrained generation did not finish its complete JSON object with EOS")
        if config.prefill_chunk_size is not None:
            memory = provenance.get("request_memory")
            if not isinstance(memory, list) or len(memory) != len(fitting):
                raise ValueError("GPU request memory measurements are incomplete")
            measured = Counter()
            for row in memory:
                if (not isinstance(row, dict) or row.get("completed") is not True or
                        row.get("prefill_chunk_size") != config.prefill_chunk_size or
                        any(type(row.get(name)) is not int or row[name] <= 0
                            for name in ("peak_allocated_bytes", "peak_reserved_bytes", "kv_cache_estimate_bytes")) or
                        row["peak_reserved_bytes"] < row["peak_allocated_bytes"] or
                        not isinstance(row.get("prompt_sha256"), list) or
                        not isinstance(row.get("input_tokens"), list) or
                        len(row["prompt_sha256"]) != 1 or len(row["input_tokens"]) != 1 or
                        "measurement_error" in row):
                    raise ValueError("GPU request memory measurement failed validation")
                measured[(row["prompt_sha256"][0], row["input_tokens"][0])] += 1
            if measured != Counter((row["prompt_sha256"], row["input_tokens"]) for row in fitting):
                raise ValueError("GPU request memory measurements differ from frozen prompts")
        for name in ("model", "tokenizer"):
            identity = provenance.get(name, {})
            if (identity.get("id") != config.model_id or identity.get("requested_revision") != config.revision
                    or identity.get("resolved_revision") not in (None, config.revision)):
                raise ValueError(f"GPU {name} provenance differs from the pinned model")


def verify_gpu_results(bundle, results, *, score_response_errors=False):
    """Return independently verified per-task coverage; never open model weights."""
    bundle, results = Path(bundle).resolve(), Path(results)
    manifest, receipt = _verify_inputs(bundle, results)
    coverage = {}
    for task in exp.TASKS:
        target = results / task
        coverage[task] = exp._check_task_run(
            bundle, task, target, manifest, receipt["backend"], allow_response_errors=score_response_errors,
        )
        _check_recorded_settings(bundle, target, manifest, receipt["backend"],
                                 allow_response_errors=score_response_errors)
    generated = sum(row["validated_fit_requests"] for row in coverage.values())
    response_errors = sum(row.get("response_error_requests", 0) for row in coverage.values())
    return {"backend": receipt["backend"], "synthetic": receipt["backend"] == "stub",
            "request_coverage": coverage,
            "response_error_requests": response_errors,
            "real_generation_acceptance": receipt["backend"] == "hf" and generated > 0 and not response_errors,
            "real_model_acceptance": receipt["backend"] == "hf" and generated > 0 and not response_errors and all(
                row["full_request_acceptance"] for row in coverage.values())}


def evaluate_results(*, bundle, results, output_dir, score_response_errors=False):
    """Run four-variant metrics, rubric, and plots locally; retain every request."""
    bundle, results, output_dir = Path(bundle).resolve(), Path(results), exp._fresh(output_dir).resolve()
    for source in (bundle, results.resolve()):
        if source == output_dir or source in output_dir.parents or output_dir in source.parents:
            raise ValueError("Evaluation output must be separate from bundle and GPU results")
    manifest, gpu_receipt = _verify_inputs(bundle, results)
    before = exp._inventory(results)
    source_before = exp._sources()
    output_dir.mkdir(parents=True)
    failures, stages, accounting, verified = [], [], {}, {}
    env = dict(os.environ, PYTHONPATH=str(exp.ROOT / "src"), MPLBACKEND="Agg",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    try:
        for task in exp.TASKS:
            source, target = bundle / "tasks" / task, results / task
            try:
                accounting[task] = _task_accounting(task, target, manifest)
                coverage = exp._check_task_run(
                    bundle, task, target, manifest, gpu_receipt["backend"],
                    allow_response_errors=score_response_errors,
                )
                _check_recorded_settings(bundle, target, manifest, gpu_receipt["backend"],
                                         allow_response_errors=score_response_errors)
                verified[task] = coverage
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                failures.append(f"{task}: {type(exc).__name__}: {exc}")
                continue
            destination = output_dir / task
            destination.mkdir()
            command = [sys.executable, "scripts/evaluate.py", "--metadata", str(source / "metadata.csv"),
                       "--predictions", str(source / "baseline_1.jsonl"),
                       *(str(target / "inference" / f"{variant}.jsonl") for variant in exp.VARIANTS),
                       "--bt-dir", str(source / "bt"), "--manifest", str(source / "task_manifest.json"),
                       "--split", manifest["split"], "--variants", "baseline_1", *exp.VARIANTS,
                       "--config", str(bundle / "configs/eval.yaml"), "--run-manifest",
                       str(source / "baseline_1.manifest.json"), str(target / "inference/run_manifest.json"),
                       "--output-dir", str(destination / "evaluation")]
            with (destination / "evaluation.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=exp.ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            stages.append({"task": task, "stage": "cpu_evaluation", "command": command,
                           "exit_code": completed.returncode})
            if completed.returncode:
                failures.append(f"{task}/cpu_evaluation: exit {completed.returncode}")
            if coverage.get("response_error_requests", 0):
                failures.append(
                    f"{task}: {coverage['response_error_requests']} independently reproduced response-validation "
                    "error(s) retained in scoring; real-model acceptance failed"
                )
        _verify_inputs(bundle, results)
        if before != exp._inventory(results) or source_before != exp._sources():
            failures.append("GPU results or evaluation sources changed during CPU evaluation")
    except BaseException as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        complete = not failures and set(verified) == set(exp.TASKS)
        synthetic = gpu_receipt["backend"] == "stub"
        refused = sum(row.get("context_refused_requests", 0) for row in verified.values())
        generated = sum(row.get("validated_fit_requests", 0) for row in verified.values())
        response_errors = sum(row.get("response_error_requests", 0) for row in verified.values())
        metrics_generated = {row["task"] for row in stages if row["exit_code"] == 0} == set(exp.TASKS)
        report = {"evaluation_version": 1,
                  "status": (("synthetic" if synthetic else "passed") +
                             ("_with_context_refusals" if refused else "")) if complete else "failed",
                  "backend": gpu_receipt["backend"], "synthetic": synthetic,
                  "real_generation_acceptance": complete and not synthetic and generated > 0,
                  "real_model_acceptance": complete and not synthetic and generated > 0 and not refused,
                  "score_response_errors": score_response_errors,
                  "response_error_requests": response_errors, "metrics_generated": metrics_generated,
                  "bundle_manifest_sha256": exp._fingerprint(bundle / "bundle_manifest.json")["sha256"],
                  "gpu_receipt_sha256": exp._fingerprint(results / "experiment_receipt.json")["sha256"],
                  "dataset_release_status": manifest["dataset_release_status"], "split": manifest["split"],
                  "pair_counts": manifest["pair_counts"], "request_coverage": verified,
                  "request_accounting": accounting, "stages": stages, "failures": failures,
                  "gpu_results_unchanged": before == exp._inventory(results),
                  "source_files": source_before, "files": exp._inventory(output_dir),
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "note": "CPU scoring only; no model, tokenizer runtime, or network access required. Stub runs never establish real-model acceptance."}
        exp._write(output_dir / "evaluation_receipt.json", report)
    return report
