"""CPU verification of chunked, constrained GPU execution receipts."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from telemetry_reasoning import experiment_results as er
from telemetry_reasoning import experiments as exp
from telemetry_reasoning import gpu_qualification as qualification_module
from telemetry_reasoning.models.constrained import ObservedResponseConstraint
from telemetry_reasoning.models.llm import LLMConfig, _hash, execution_policy, load_llm_config
from telemetry_reasoning.models.prompts import build_prompt
from telemetry_reasoning.schemas import PairedBehaviorTree

from test_constrained import CharacterTokenizer
from test_experiment_results import _fixture, _rebind


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _v3_fixture(tmp_path, monkeypatch, *, backend="hf"):
    """Upgrade source-bound synthetic receipts; never represent a real GPU run."""
    bundle, results = _fixture(tmp_path, monkeypatch, backend=backend)
    config = load_llm_config(bundle / "configs/llm.yaml").model_copy(update={
        "context_window_tokens": 8192, "max_new_tokens": 4096,
        "prefill_chunk_size": 256, "attention_implementation": "sdpa",
        "fused_attention_only": True, "decoding_constraint": "observed_json_v1",
    })
    exp._write(bundle / "configs/llm.yaml", {"llm": config.model_dump(mode="json")})
    manifest = exp._json(bundle / "bundle_manifest.json")
    tokenizer = CharacterTokenizer()
    for task in exp.TASKS:
        inference_dir = results / task / "inference"
        inference = exp._json(inference_dir / "run_manifest.json")
        inference["config"] = config.model_dump(mode="json")
        inference["config_sha256"] = _hash(inference["config"])
        inference["decoding"]["max_new_tokens"] = config.max_new_tokens
        tree = PairedBehaviorTree.model_validate_json((bundle / "tasks" / task / "bt/pair_000_bt.json").read_bytes())
        constraints, memory = [], []
        for row in manifest["token_budget"]:
            if row["task"] != task:
                continue
            row.update(reserved_output_tokens=config.max_new_tokens,
                       context_window_tokens=config.context_window_tokens)
            prompt = build_prompt(tree, row["model_variant"], template_dir=config.prompt_dir)
            row["output_constraint"] = ObservedResponseConstraint(
                prompt, tokenizer, tokenizer.eos_token_id, config.max_new_tokens,
            ).provenance()
            constraints.append({"prompt_sha256": row["prompt_sha256"], **row["output_constraint"]})
            memory.append({"prompt_sha256": [row["prompt_sha256"]], "input_tokens": [row["input_tokens"]],
                           "completed": True, "prefill_chunk_size": config.prefill_chunk_size,
                           "peak_allocated_bytes": 16 * 1024 ** 3, "peak_reserved_bytes": 17 * 1024 ** 3,
                           "kv_cache_estimate_bytes": 128 * 1024})
        provenance = inference["backend_provenance"]
        provenance.update(implementation="transformers_batched_greedy_v3", execution_policy=execution_policy(config),
                          constraints=constraints, request_memory=memory)
        for request in inference["requests"]:
            request["finish_reason"] = "eos"
        exp._write(inference_dir / "run_manifest.json", inference)
        completions = _rows(inference_dir / "completions.jsonl")
        for completion in completions:
            completion["finish_reason"] = "eos"
        (inference_dir / "completions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in completions))
    manifest["files"] = exp._inventory(bundle, ("bundle_manifest.json",))
    exp._write(bundle / "bundle_manifest.json", manifest)
    receipt = exp._json(results / "experiment_receipt.json")
    receipt["bundle_manifest_sha256"] = exp._fingerprint(bundle / "bundle_manifest.json")["sha256"]
    if backend == "hf":
        # This file exercises the qualification-binding boundary. The separate
        # test_gpu_qualification suite verifies actual qualification semantics.
        path = results / "qualification/qualification_receipt.json"
        path.parent.mkdir()
        exp._write(path, {"qualification_version": 1, "status": "passed", "synthetic_fixture_only": True})
        receipt.update(qualification_path="qualification/qualification_receipt.json",
                       qualification_receipt_sha256=exp._fingerprint(path)["sha256"])

        def fixture_qualification(actual_bundle, actual_path):
            actual_path = actual_path / "qualification_receipt.json" if actual_path.is_dir() else actual_path
            assert actual_bundle == bundle.resolve()
            return exp._json(actual_path)

        monkeypatch.setattr(qualification_module, "verify_qualification", fixture_qualification)
    exp._write(results / "experiment_receipt.json", receipt)
    _rebind(results)
    return bundle, results


def test_v3_cpu_verification_requires_complete_execution_constraints_memory_and_eos(tmp_path, monkeypatch):
    bundle, results = _v3_fixture(tmp_path, monkeypatch)
    verified = er.verify_gpu_results(bundle, results)
    assert verified["real_generation_acceptance"] and verified["real_model_acceptance"]
    assert all(row["validated_fit_requests"] == 3 for row in verified["request_coverage"].values())


@pytest.mark.parametrize("mutation", ["execution_missing", "execution_changed", "implementation",
                                      "constraint_missing", "constraint_duplicate", "constraint_changed",
                                      "memory_missing", "memory_duplicate", "memory_incomplete", "memory_changed",
                                      "memory_invalid_peak", "memory_measurement_error", "eos"])
def test_v3_semantic_receipt_tampering_is_refused_after_inventory_rebinding(tmp_path, monkeypatch, mutation):
    bundle, results = _v3_fixture(tmp_path, monkeypatch)
    directory = results / "pick_up_object/inference"
    inference = exp._json(directory / "run_manifest.json")
    provenance = inference["backend_provenance"]
    if mutation == "execution_missing":
        provenance.pop("execution_policy")
    elif mutation == "execution_changed":
        provenance["execution_policy"]["prefill_chunk_size"] = 2048
    elif mutation == "implementation":
        provenance["implementation"] = "transformers_batched_greedy_v2"
    elif mutation == "constraint_missing":
        provenance["constraints"].pop()
    elif mutation == "constraint_duplicate":
        provenance["constraints"][1] = copy.deepcopy(provenance["constraints"][0])
    elif mutation == "constraint_changed":
        provenance["constraints"][0]["prompt_sha256"] = "0" * 64
    elif mutation == "memory_missing":
        provenance.pop("request_memory")
    elif mutation == "memory_duplicate":
        provenance["request_memory"][1] = copy.deepcopy(provenance["request_memory"][0])
    elif mutation == "memory_incomplete":
        provenance["request_memory"][0]["completed"] = False
    elif mutation == "memory_changed":
        provenance["request_memory"][0]["input_tokens"][0] += 1
    elif mutation == "memory_invalid_peak":
        provenance["request_memory"][0]["peak_reserved_bytes"] = 1
    elif mutation == "memory_measurement_error":
        provenance["request_memory"][0]["measurement_error"] = "synthetic measurement failure"
    else:
        inference["requests"][0]["finish_reason"] = "length"
        completions = _rows(directory / "completions.jsonl")
        completions[0]["finish_reason"] = "length"
        (directory / "completions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in completions))
    exp._write(directory / "run_manifest.json", inference)
    _rebind(results)
    with pytest.raises(ValueError):
        er.verify_gpu_results(bundle, results)


def test_v3_stub_receipt_stays_synthetic_without_requiring_gpu_provenance(tmp_path, monkeypatch):
    bundle, results = _v3_fixture(tmp_path, monkeypatch, backend="stub")
    for task in exp.TASKS:
        path = results / task / "inference/run_manifest.json"
        record = exp._json(path)
        record["backend_provenance"] = {"fixture": "stub without GPU execution"}
        exp._write(path, record)
    _rebind(results)
    verified = er.verify_gpu_results(bundle, results)
    assert verified["synthetic"] and not verified["real_generation_acceptance"] and not verified["real_model_acceptance"]


@pytest.mark.parametrize("mutation", ["missing", "wrong_sha", "wrong_path", "verifier_failure"])
def test_v3_return_requires_bound_and_independently_verified_qualification(tmp_path, monkeypatch, mutation):
    bundle, results = _v3_fixture(tmp_path, monkeypatch)
    path = results / "experiment_receipt.json"
    receipt = exp._json(path)
    if mutation == "missing":
        receipt.pop("qualification_path")
    elif mutation == "wrong_sha":
        receipt["qualification_receipt_sha256"] = "0" * 64
    elif mutation == "wrong_path":
        receipt["qualification_path"] = "../qualification_receipt.json"
    else:
        def reject(*args, **kwargs):
            raise ValueError("Synthetic qualification verifier refusal")
        monkeypatch.setattr(qualification_module, "verify_qualification", reject)
    exp._write(path, receipt)
    _rebind(results)
    with pytest.raises(ValueError, match="qualification|Qualification"):
        er.verify_gpu_results(bundle, results)


def test_v3_run_copies_qualification_into_results_and_records_portable_binding(tmp_path, monkeypatch):
    bundle, fixture_results = _v3_fixture(tmp_path, monkeypatch)
    source = fixture_results / "qualification"
    expected_hash = exp._fingerprint(source / "qualification_receipt.json")["sha256"]
    destination = tmp_path / "new-run"
    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(exp, "_check_task_run", lambda *args, **kwargs: {
        "expected_requests": 3, "validated_fit_requests": 3, "context_refused_requests": 0,
        "full_request_acceptance": True,
    })
    assert exp.run_experiment(bundle, destination, backend="hf", inference_only=True, qualification=source) == 0
    receipt = exp._json(destination / "experiment_receipt.json")
    assert receipt["qualification_path"] == "qualification/qualification_receipt.json"
    assert receipt["qualification_receipt_sha256"] == expected_hash
    assert exp._fingerprint(destination / receipt["qualification_path"])["sha256"] == expected_hash
    assert receipt["qualification_path"] in receipt["files"]
    assert exp._inventory(destination, ("experiment_receipt.json",)) == receipt["files"]


class PromptCounter:
    eos_token_id = 0

    def __init__(self, count):
        self.count = count

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["truncation"] is False and kwargs["return_tensors"] is None
        return {"input_ids": [1] * self.count, "attention_mask": [1] * self.count}


@pytest.mark.parametrize("constrained,count,should_compile", [(True, 5, True), (True, 100, False),
                                                              (False, 5, False)])
def test_cpu_preflight_compiles_constraints_only_for_fitting_constrained_requests(
    tmp_path, monkeypatch, constrained, count, should_compile,
):
    from telemetry_reasoning.models import constrained as constrained_module
    tree = PairedBehaviorTree.model_validate_json((exp.ROOT / "data/sample/pair_000_bt.json").read_bytes())
    prompt = build_prompt(tree, "baseline_2")
    config = LLMConfig(context_window_tokens=128, max_new_tokens=32,
                       decoding_constraint="observed_json_v1" if constrained else "none")
    tokenizer, calls = PromptCounter(count), []

    class Compiler:
        def __init__(self, *args):
            calls.append(args)

        def provenance(self):
            return {"policy": "observed_json_v1", "fixture_cpu_compiler": True}

    monkeypatch.setattr(constrained_module, "ObservedResponseConstraint", Compiler)
    record = exp.request_preflight(config, tokenizer, prompt, "pick_up_object", "000", "baseline_2")
    assert record["input_tokens"] == count and record["fits"] is (count + 32 <= 128)
    assert bool(calls) is should_compile
    assert ("output_constraint" in record) is should_compile
    if should_compile:
        assert calls == [(prompt, tokenizer, tokenizer.eos_token_id, config.max_new_tokens)]
    else:
        assert set(record) == {"task", "pair_id", "model_variant", "prompt_sha256", "input_tokens",
                               "reserved_output_tokens", "context_window_tokens", "fits"}


@pytest.mark.parametrize("qualification", [None, "invalid-receipt"])
def test_v3_inference_requires_verified_qualification_before_creating_output_or_launching_gpu(
    tmp_path, monkeypatch, qualification,
):
    bundle, _ = _v3_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "new-inference"
    calls = []

    def reject_qualification(actual_bundle, receipt):
        calls.append((actual_bundle, receipt))
        raise ValueError("Qualification does not bind this exact bundle")

    monkeypatch.setitem(sys.modules, "telemetry_reasoning.gpu_qualification",
                        SimpleNamespace(verify_qualification=reject_qualification))
    monkeypatch.setattr(exp.subprocess, "run", lambda *args, **kwargs: pytest.fail("GPU process launched before qualification"))
    with pytest.raises(ValueError, match="qualification|Qualification"):
        exp.run_experiment(bundle, destination, backend="hf", inference_only=True, qualification=qualification)
    assert not destination.exists()
    assert calls == ([] if qualification is None else [(bundle.resolve(), qualification)])
