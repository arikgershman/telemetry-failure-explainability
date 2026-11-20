"""Qualification gates use synthetic backends; these are never GPU evidence."""
import builtins
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from telemetry_reasoning import experiments as exp
from telemetry_reasoning import gpu_qualification as qualify
from telemetry_reasoning.models.llm import Generation, LLMConfig, MEMORY_POLICY, execution_policy, load_llm_config
from telemetry_reasoning.models.prompts import DEFAULT_TEMPLATE_DIR, build_prompt
from telemetry_reasoning.schemas import PairedBehaviorTree
from telemetry_reasoning.parser.divergence_metrics import enrich_paired_bt, load_divergence_config


GIB = 1024**3


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "configs").mkdir(parents=True)
    shutil.copytree(DEFAULT_TEMPLATE_DIR, bundle / "prompts", ignore=shutil.ignore_patterns("__pycache__"))
    config = LLMConfig(device="cuda", dtype="bfloat16", prefill_chunk_size=8,
                       attention_implementation="sdpa", fused_attention_only=True,
                       decoding_constraint="observed_json_v1", prompt_dir=Path("../prompts"))
    (bundle / "configs/llm.yaml").write_text(yaml.safe_dump({"llm": config.model_dump(mode="json")}))
    config = load_llm_config(bundle / "configs/llm.yaml")
    tree = PairedBehaviorTree.model_validate_json((exp.ROOT / "data/sample/pair_000_bt.json").read_bytes())
    tree = enrich_paired_bt(tree, load_divergence_config(exp.ROOT / "configs/divergence.yaml"))
    task, pair_id = tree.metadata.task.value, tree.metadata.pair_id
    bt_dir = bundle / "tasks" / task / "bt"
    bt_dir.mkdir(parents=True)
    (bt_dir / f"pair_{pair_id}_bt.json").write_text(tree.model_dump_json())
    budgets = []
    for variant, count in (("baseline_2", 5), ("main", 10), ("baseline_3", config.context_window_tokens)):
        prompt = build_prompt(tree, variant, template_dir=config.prompt_dir)
        budgets.append({"task": task, "pair_id": pair_id, "model_variant": variant,
                        "prompt_sha256": prompt.prompt_sha256, "input_tokens": count,
                        "reserved_output_tokens": config.max_new_tokens,
                        "context_window_tokens": config.context_window_tokens,
                        "fits": count + config.max_new_tokens <= config.context_window_tokens,
                        "output_constraint": {"policy": "observed_json_v1", "repair": False,
                                              "worst_case_tokens_including_eos": 200}})
    sources = {"synthetic_fixture": {"sha256": "0" * 64, "size_bytes": 1}}
    monkeypatch.setattr(exp, "_sources", lambda: copy.deepcopy(sources))
    manifest = {"bundle_version": 1, "status": "ready_for_gpu_with_context_refusals",
                "source_files": sources, "files": exp._inventory(bundle), "token_budget": budgets}
    exp._write(bundle / "bundle_manifest.json", manifest)
    probe_script = qualify._script("probe_cuda_inference.py")
    cases = []
    for case in probe_script.probe_cases(8, 65536):
        cases.append({**case.description(), "status": "passed", "finite_output": True,
                      "peak_allocated_bytes": GIB, "peak_reserved_bytes": 2 * GIB,
                      "total_memory_bytes": 32 * GIB,
                      "sdpa_operators": ["aten::scaled_dot_product_attention", "aten::_scaled_dot_product_efficient_attention"]})
    probe = {"schema_version": 1, "probe_kind": "llama31_bfloat16_sdpa_kernel_v1", "status": "passed",
             "model_weights_loaded": False, "network_access_requested": False, "math_backend_enabled": False,
             "chunk_tokens": 8, "prefix_tokens": 65536, "cases": cases,
             "device": {"index": 0, "name": "synthetic GPU", "capability": [12, 0], "bfloat16_supported": True},
             "runtime": {"torch": "fixture", "cuda": "13.0", "python": "3.12.3", "platform": "synthetic platform"},
             "transformers_package_version": "5.23.0"}
    kernel = tmp_path / "kernel.json"
    exp._write(kernel, probe)
    monkeypatch.setattr(qualify.metadata, "version", lambda name: "fixture" if name == "torch" else "5.23.0")

    class FakeBackend:
        name = "hf"
        token_count_method = "tokenizer"
        generated = 0
        counted = 0

        def __init__(self, config):
            self.config = config
            self.prompt = None
            self._torch = SimpleNamespace(
                version=SimpleNamespace(cuda="13.0"),
                cuda=SimpleNamespace(current_device=lambda: 0,
                                     get_device_name=lambda index: "synthetic GPU",
                                     get_device_capability=lambda index: (12, 0),
                                     is_bf16_supported=lambda: True,
                                     mem_get_info=lambda: (12 * GIB, 32 * GIB)))

        def count_tokens(self, prompt):
            type(self).counted += 1
            return 10

        def generate(self, prompt):
            type(self).generated += 1
            self.prompt = prompt
            return Generation(json.dumps({"predicted_failure_mode": None, "predicted_subtask": None,
                                          "explanation": "Synthetic qualification fixture; no diagnosis.",
                                          "evidence": [], "status": "abstain"}),
                              10, 25, "tokenizer", "eos")

        def provenance(self):
            return {"implementation": "transformers_batched_greedy_v3",
                    "execution_policy": execution_policy(self.config),
                    "weights_loaded": self.prompt is not None, "tokenizer_loaded": True,
                    "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "quantization": "none",
                    "token_count_method": "tokenizer", "native_context_window_tokens": 131072,
                    "resources": {"policy": MEMORY_POLICY, "host_admissible": True, "cuda_total_memory_bytes": 32 * GIB},
                    "model": {"id": self.config.model_id, "requested_revision": self.config.revision,
                              "resolved_revision": self.config.revision},
                    "tokenizer": {"id": self.config.model_id, "requested_revision": self.config.revision,
                                  "resolved_revision": self.config.revision},
                    "runtime": {"torch": "fixture", "transformers": "5.23.0", "platform": "synthetic platform"},
                    "constraints": [{"policy": "observed_json_v1", "prompt_sha256": self.prompt.prompt_sha256,
                                     "repair": False, "worst_case_tokens_including_eos": 200}] if self.prompt else [],
                    "request_memory": [{"completed": True, "prompt_sha256": [self.prompt.prompt_sha256],
                                        "input_tokens": [10], "prefill_chunk_size": 8,
                                        "peak_allocated_bytes": 20 * GIB,
                                        "peak_reserved_bytes": 21 * GIB}] if self.prompt else []}

    monkeypatch.setattr(qualify, "HFBackend", FakeBackend)
    return SimpleNamespace(bundle=bundle, kernel=kernel, probe=probe, backend=FakeBackend, config=config,
                           output=tmp_path / "qualification", manifest=manifest)


def _run(prepared):
    return qualify.qualify_experiment(bundle=prepared.bundle, kernel_report=prepared.kernel,
                                      output_dir=prepared.output)


def _rewrite_receipt(output, update):
    path = output / qualify.RECEIPT
    receipt = exp._json(path)
    update(receipt)
    receipt["files"] = exp._inventory(output, (qualify.RECEIPT,))
    exp._write(path, receipt)


def test_one_longest_fitting_actual_request_is_measured_and_portably_verified(prepared):
    receipt = _run(prepared)
    assert receipt["status"] == "passed", receipt["failures"]
    assert receipt["selected_request"]["model_variant"] == "main"
    assert receipt["selected_request"]["input_tokens"] == 10
    assert prepared.backend.generated == 1
    assert prepared.backend.counted == 2
    assert receipt["kernel_probe_source_binding"] == "original_v1_report_has_no_source_fingerprint"
    assert (prepared.output / "kernel_report.json").read_bytes() == prepared.kernel.read_bytes()
    assert qualify.verify_qualification(prepared.bundle, prepared.output) == receipt
    assert qualify.verify_qualification(prepared.bundle, prepared.output / qualify.RECEIPT) == receipt


def test_token_recount_mismatch_refuses_before_any_generation_and_retains_receipt(prepared, monkeypatch):
    monkeypatch.setattr(prepared.backend, "count_tokens", lambda self, prompt: 11)
    receipt = _run(prepared)
    assert receipt["status"] == "failed"
    assert "tokenizer count differs" in receipt["failures"][0]
    assert prepared.backend.generated == 0
    assert not (prepared.output / "inference").exists()
    with pytest.raises(ValueError, match="did not pass"):
        qualify.verify_qualification(prepared.bundle, prepared.output)


@pytest.mark.parametrize("mutation", [
    lambda probe: probe.update(status="failed"),
    lambda probe: probe.update(math_backend_enabled=True),
    lambda probe: probe["cases"].pop(),
    lambda probe: probe["cases"][0].update(finite_output=False),
    lambda probe: probe["cases"][0].update(peak_allocated_bytes=0),
    lambda probe: probe["cases"][0].update(sdpa_operators=["aten::_scaled_dot_product_attention_math"]),
    lambda probe: probe["cases"][1].update(mask_rule="incorrect causal alignment"),
    lambda probe: probe.update(chunk_tokens=16),
    lambda probe: probe.update(probe_script={"sha256": "0" * 64, "size_bytes": 1}),
])
def test_bad_kernel_evidence_never_opens_weights(prepared, mutation):
    mutation(prepared.probe)
    exp._write(prepared.kernel, prepared.probe)
    receipt = _run(prepared)
    assert receipt["status"] == "failed"
    assert prepared.backend.counted == prepared.backend.generated == 0
    assert receipt["failures"]


def test_different_gpu_or_cuda_runtime_refuses_before_weights(prepared):
    prepared.probe["device"]["name"] = "another GPU"
    exp._write(prepared.kernel, prepared.probe)
    receipt = _run(prepared)
    assert receipt["status"] == "failed"
    assert "different CUDA device/runtime" in receipt["failures"][0]
    assert prepared.backend.generated == 0


def test_runtime_package_drift_refuses_before_backend_creation(prepared):
    prepared.probe["runtime"]["torch"] = "different Torch"
    exp._write(prepared.kernel, prepared.probe)
    receipt = _run(prepared)
    assert receipt["status"] == "failed"
    assert "Installed torch differs" in receipt["failures"][0]
    assert prepared.backend.counted == prepared.backend.generated == 0


def test_existing_output_is_never_overwritten(prepared):
    prepared.output.mkdir()
    (prepared.output / "existing.txt").write_text("preserve")
    with pytest.raises(ValueError, match="absent"):
        _run(prepared)
    assert (prepared.output / "existing.txt").read_text() == "preserve"
    assert prepared.backend.generated == 0


def test_output_cannot_be_published_inside_the_frozen_bundle(prepared):
    prepared.output = prepared.bundle / "qualification"
    with pytest.raises(ValueError, match="separate"):
        _run(prepared)
    assert not prepared.output.exists()


def test_saved_response_is_reparsed_even_if_artifact_inventory_is_updated(prepared):
    assert _run(prepared)["status"] == "passed"
    path = prepared.output / "inference/completions.jsonl"
    completion = json.loads(path.read_text())
    completion["text"] = '{"invalid":"response"}'
    import hashlib
    completion["completion_sha256"] = hashlib.sha256(completion["text"].encode()).hexdigest()
    path.write_text(json.dumps(completion) + "\n")
    _rewrite_receipt(prepared.output, lambda receipt: None)
    with pytest.raises(ValueError):
        qualify.verify_qualification(prepared.bundle, prepared.output)


@pytest.mark.parametrize("change", [
    lambda p: p.update(weights_loaded=False),
    lambda p: p.update(implementation="transformers_batched_greedy_v2"),
    lambda p: p["execution_policy"].update(fused_attention_only=False),
    lambda p: p["request_memory"][0].update(peak_allocated_bytes=0),
    lambda p: p["request_memory"][0].update(completed=False),
    lambda p: p["constraints"][0].update(repair=True),
    lambda p: p["runtime"].update(platform="another platform"),
])
def test_rehashed_claims_cannot_replace_measured_v3_evidence(prepared, change):
    assert _run(prepared)["status"] == "passed"
    path = prepared.output / "inference/run_manifest.json"
    manifest = exp._json(path)
    change(manifest["backend_provenance"])
    exp._write(path, manifest)
    _rewrite_receipt(prepared.output, lambda receipt: receipt.update(backend_provenance=manifest["backend_provenance"]))
    with pytest.raises(ValueError):
        qualify.verify_qualification(prepared.bundle, prepared.output)


def test_receipt_for_a_shorter_or_different_request_is_refused(prepared):
    assert _run(prepared)["status"] == "passed"
    _rewrite_receipt(prepared.output, lambda receipt: receipt.update(selected_request=prepared.manifest["token_budget"][0]))
    with pytest.raises(ValueError, match="longest fitting"):
        qualify.verify_qualification(prepared.bundle, prepared.output)


def test_readonly_return_verification_does_not_import_a_heavy_runtime(prepared, monkeypatch):
    assert _run(prepared)["status"] == "passed"
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "transformers"):
            raise AssertionError("pure verification imported a model runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert qualify.verify_qualification(prepared.bundle, prepared.output)["status"] == "passed"


def test_real_generation_failure_still_publishes_predictions_and_failure_receipt(prepared, monkeypatch):
    def fail(self, prompt):
        raise RuntimeError("synthetic CUDA failure")

    monkeypatch.setattr(prepared.backend, "generate", fail)
    receipt = _run(prepared)
    assert receipt["status"] == "failed"
    assert (prepared.output / "inference/run_manifest.json").is_file()
    assert (prepared.output / "inference/main.jsonl").is_file()
    assert (prepared.output / qualify.RECEIPT).is_file()
    assert json.loads((prepared.output / "inference/main.jsonl").read_text())["status"] == "error"


def test_cli_forwards_explicit_inputs_and_returns_failure_status(tmp_path, monkeypatch):
    cli = qualify._script("qualify_gpu_experiment.py")
    seen = []
    monkeypatch.setattr(cli, "qualify_experiment", lambda **kwargs: seen.append(kwargs) or {"status": "failed", "failures": ["fixture"]})
    assert cli.main(["--bundle", str(tmp_path / "bundle"), "--kernel-report", str(tmp_path / "kernel.json"),
                     "--output-dir", str(tmp_path / "output")]) == 1
    assert seen == [{"bundle": tmp_path / "bundle", "kernel_report": tmp_path / "kernel.json", "output_dir": tmp_path / "output"}]
