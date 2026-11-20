"""Track 5 adapter and CLI contracts, using local data and fake model resources.

These tests verify inference plumbing and response grounding, not diagnosis
quality. They must never download weights, contact Hugging Face, or need a token.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from scripts import run_llm
from telemetry_reasoning.models import llm
from telemetry_reasoning.models.llm import (
    Generation, LLMConfig, StubBackend, load_llm_config, parse_response,
    response_normalization, run_inference,
)
from telemetry_reasoning.models.prompts import build_prompt
from telemetry_reasoning.schemas import PairedBehaviorTree


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/sample/pair_000_bt.json"
CONFIG = ROOT / "configs/llm.yaml"
RECORD_KEYS = {
    "pair_id", "model_variant", "predicted_failure_mode", "predicted_subtask",
    "explanation", "evidence", "status", "backend",
}


@pytest.fixture
def tree():
    return PairedBehaviorTree.model_validate_json(SAMPLE.read_bytes())


def model_response(prompt, *, mode="position_error"):
    node = next(node for node in prompt.payload["subtasks"] if node["failure_samples"])
    sample = node["failure_samples"][0]
    return {
        "predicted_failure_mode": mode,
        "predicted_subtask": node["subtask"],
        "explanation": "Synthetic test diagnosis grounded in the supplied observation.",
        "evidence": [{
            "subtask": node["subtask"], "timestamp": sample["timestamp"],
            "feature": "gripper_force", "value": sample["gripper_force"],
        }],
        "status": "ok",
    }


class FakeBackend:
    """Exercise the same inference boundary as an HF backend without imports."""

    name = "hf"

    def __init__(self, *, text=None, tokens=64):
        self.text = text
        self.tokens = tokens
        self.generated = []

    def count_tokens(self, prompt):
        return self.tokens

    def generate(self, prompt):
        self.generated.append(prompt)
        text = self.text if self.text is not None else json.dumps(model_response(prompt))
        return Generation(text=text, input_tokens=self.tokens, output_tokens=32)

    def provenance(self):
        return {"backend": self.name, "test_double": True}


def write_tree(directory, tree, *, pair_id="000", filename=None):
    directory.mkdir(parents=True, exist_ok=True)
    payload = tree.model_dump(mode="json")
    payload["metadata"]["pair_id"] = pair_id
    path = directory / (filename or f"pair_{pair_id}_bt.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def cli_args(bt_dir, output_dir, *extra):
    return ["--bt-dir", str(bt_dir), "--output-dir", str(output_dir), *extra]


def test_exact_config_defaults_and_yaml_agree():
    config = load_llm_config(CONFIG)
    assert config.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert len(config.revision) == 40 and int(config.revision, 16) >= 0
    assert config.local_files_only is True
    assert config.quantization == "none"
    assert config.context_window_tokens > config.max_new_tokens > 0
    assert config == LLMConfig()


@pytest.mark.parametrize("update", [
    {"revision": "main"}, {"revision": "abc123"},
    {"model_id": "uncontrolled/model"}, {"quantization": "4bit"},
    {"device": "tpu"}, {"dtype": "int8"}, {"seed": -1},
    {"max_new_tokens": 0}, {"context_window_tokens": 0},
    {"max_new_tokens": 100, "context_window_tokens": 100},
    {"unexpected_setting": True},
])
def test_config_rejects_unpinned_or_inconsistent_settings(update):
    with pytest.raises((ValueError, ValidationError)):
        LLMConfig.model_validate(update)


@pytest.mark.parametrize("source", ["[]", "{}", "llm: []", "llm: {}\nother: 1", "llm:\n  unknown: 1"])
def test_config_yaml_boundary_is_strict(tmp_path, source):
    path = tmp_path / "invalid.yaml"
    path.write_text(source)
    with pytest.raises(ValueError):
        load_llm_config(path)


def test_structured_response_is_grounded_and_gets_join_fields(tree):
    prompt = build_prompt(tree, "baseline_2")
    response = model_response(prompt)
    record = parse_response(json.dumps(response), prompt, tree.metadata.pair_id, "hf")
    assert set(record) == RECORD_KEYS
    assert record == dict(response, pair_id="000", model_variant="baseline_2", backend="hf")
    assert tree.metadata.failure_mode.value not in record["explanation"]


@pytest.mark.parametrize("variant", ["baseline_2", "baseline_3"])
@pytest.mark.parametrize("newline,outer", [("\n", ""), ("\r\n", " \t\n")])
def test_explicit_fence_policy_preserves_payload_and_validates_both_variants(tree, variant, newline, outer):
    prompt = build_prompt(tree, variant)
    payload = json.dumps(model_response(prompt), ensure_ascii=False)
    raw = outer + "```json" + newline + payload + newline + "```" + outer
    expected = parse_response(payload, prompt, "000", "hf")
    with pytest.raises(json.JSONDecodeError):
        parse_response(raw, prompt, "000", "hf")
    assert parse_response(raw, prompt, "000", "hf", response_policy="json_or_fenced_v1") == expected
    assert response_normalization(raw, "json_or_fenced_v1") == {
        "policy": "json_or_fenced_v1", "envelope": "json_code_fence",
        "normalized_text_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    assert response_normalization(payload, "json_or_fenced_v1")["envelope"] == "none"


@pytest.mark.parametrize("wrap", [
    lambda body: "Here is the result:\n```json\n" + body + "\n```",
    lambda body: "```json\n" + body + "\n```\nExtra explanation.",
    lambda body: "```json\n" + body + "\n```\n```json\n" + body + "\n```",
    lambda body: "```json\n" + body,
    lambda body: body + "\n```",
    lambda body: "```\n" + body + "\n```",
    lambda body: "```JSON\n" + body + "\n```",
    lambda body: "```python\n" + body + "\n```",
    lambda body: "```json " + body + "```",
    lambda body: "~~~json\n" + body + "\n~~~",
    lambda body: "\u00a0```json\n" + body + "\n```",
])
def test_fence_policy_does_not_extract_json_from_other_output(tree, wrap):
    prompt = build_prompt(tree, "baseline_2")
    with pytest.raises(ValueError):
        parse_response(wrap(json.dumps(model_response(prompt))), prompt, "000", "hf",
                       response_policy="json_or_fenced_v1")


@pytest.mark.parametrize("defect", ["duplicate", "nonfinite", "extra_key", "unsupported_evidence", "malformed", "array"])
def test_fence_policy_retains_strict_json_schema_and_evidence_checks(tree, defect):
    prompt = build_prompt(tree, "baseline_2")
    response = model_response(prompt)
    if defect == "extra_key":
        response["pair_id"] = "injected_identity"
    if defect == "unsupported_evidence":
        response["evidence"][0]["value"] += 100
    payload = json.dumps(response)
    if defect == "duplicate":
        payload = payload[:-1] + ', "status": "ok"}'
    elif defect == "nonfinite":
        payload = payload.replace('"status": "ok"', '"status": NaN')
    elif defect == "malformed":
        payload = payload[:-1]
    elif defect == "array":
        payload = "[" + payload + "]"
    with pytest.raises(ValueError):
        parse_response("```json\n" + payload + "\n```", prompt, "000", "hf",
                       response_policy="json_or_fenced_v1")


def test_fence_handling_does_not_rewrite_explanation_or_non_ascii_content(tree):
    prompt = build_prompt(tree, "baseline_2")
    response = model_response(prompt)
    response["explanation"] = "Recorded text, including ``` and café, stays unchanged—even if causally wrong."
    raw = "```json\n" + json.dumps(response, ensure_ascii=False) + "\n```"
    parsed = parse_response(raw, prompt, "000", "hf", response_policy="json_or_fenced_v1")
    assert parsed["explanation"] == response["explanation"]


def test_response_policy_is_versioned_and_unknown_values_are_rejected(tree):
    assert LLMConfig().response_policy == "strict_json_v1"
    with pytest.raises(ValueError):
        LLMConfig(response_policy="auto_repair")
    prompt = build_prompt(tree, "baseline_2")
    with pytest.raises(ValueError, match="unsupported response policy"):
        parse_response(json.dumps(model_response(prompt)), prompt, "000", "hf", response_policy="unknown")


def test_inference_records_policy_and_raw_fenced_responses_without_content_repair(tree):
    prompt = build_prompt(tree, "baseline_2")
    raw = "```json\n" + json.dumps(model_response(prompt)) + "\n```"
    config = LLMConfig(response_policy="json_or_fenced_v1")
    result = run_inference([tree], ["baseline_2", "baseline_3"], config, backend=FakeBackend(text=raw))
    assert result.manifest["response_policy"] == config.response_policy
    assert result.manifest["response_processing"] == {
        "policy": config.response_policy, "raw_text_preserved": True, "fenced_requests": 2,
    }
    assert all(record["status"] == "ok" for record in result.predictions)
    for completion in result.completions:
        assert completion["text"] == raw
        assert completion["completion_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
        assert completion["response_normalization"] == response_normalization(raw, config.response_policy)


def test_every_shared_failure_mode_is_supported_including_excessive_speed(tree):
    prompt = build_prompt(tree, "baseline_2")
    response = model_response(prompt, mode="excessive_speed")
    assert parse_response(json.dumps(response), prompt, "000", "hf")["predicted_failure_mode"] == "excessive_speed"


@pytest.mark.parametrize("feature", ["ee_pos", "ee_quat", "joint_positions"])
def test_vector_evidence_uses_actual_sample_shape_and_values(tree, feature):
    prompt = build_prompt(tree, "baseline_2")
    response = model_response(prompt)
    sample = prompt.payload["subtasks"][0]["failure_samples"][0]
    response["evidence"][0].update(feature=feature, value=sample[feature])
    assert parse_response(json.dumps(response), prompt, "000", "hf")["evidence"][0]["value"] == sample[feature]
    response["evidence"][0]["value"] = sample[feature][:-1]
    with pytest.raises(ValueError):
        parse_response(json.dumps(response), prompt, "000", "hf")


def test_success_side_values_cannot_substitute_for_failure_evidence(tree):
    prompt = build_prompt(tree, "baseline_3")
    response = model_response(prompt)
    for node in prompt.payload["subtasks"]:
        failures = {row["timestamp"]: row for row in node["failure_samples"]}
        for row in node["success_samples"]:
            failure = failures.get(row["timestamp"])
            if failure is not None and row["gripper_force"] != failure["gripper_force"]:
                response["predicted_subtask"] = node["subtask"]
                response["evidence"] = [{
                    "subtask": node["subtask"], "timestamp": row["timestamp"],
                    "feature": "gripper_force", "value": row["gripper_force"],
                }]
                with pytest.raises(ValueError, match="failure sample"):
                    parse_response(json.dumps(response), prompt, "000", "hf")
                return
    pytest.fail("the canonical mock should contain a differing grasp force in paired samples")


@pytest.mark.parametrize("raw", [
    "not JSON", "[]", "null", "{}", "```json\n{}\n```", "{} extra prose",
    '{"status":"ok","status":"abstain"}',
])
def test_malformed_response_is_rejected(tree, raw):
    with pytest.raises(ValueError):
        parse_response(raw, build_prompt(tree, "baseline_2"), "000", "hf")


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(pair_id="injected"),
    lambda r: r.update(model_variant="baseline_3"),
    lambda r: r.update(backend="rules"),
    lambda r: r.update(predicted_failure_mode="unknown_cause"),
    lambda r: r.update(predicted_subtask="move_along_path"),
    lambda r: r.update(predicted_failure_mode=None),
    lambda r: r.update(evidence=[]),
    lambda r: r.update(explanation=""),
    lambda r: r.update(status="abstain"),
    lambda r: r["evidence"][0].update(timestamp=9999.0),
    lambda r: r["evidence"][0].update(value=9999.0),
    lambda r: r["evidence"][0].update(value=True),
    lambda r: r["evidence"][0].update(value=[1.0]),
    lambda r: r["evidence"][0].update(value=float("nan")),
    lambda r: r["evidence"][0].update(feature="unknown_sensor"),
    lambda r: r["evidence"][0].update(subtask="move_along_path"),
    lambda r: r["evidence"][0].update(extra="field"),
])
def test_invalid_or_ungrounded_response_is_rejected(tree, mutate):
    prompt = build_prompt(tree, "baseline_2")
    response = model_response(prompt)
    mutate(response)
    with pytest.raises(ValueError):
        parse_response(json.dumps(response), prompt, "000", "hf")


@pytest.mark.parametrize("status", ["abstain", "error"])
def test_non_ok_responses_preserve_uncertainty_without_labels(tree, status):
    response = {
        "predicted_failure_mode": None, "predicted_subtask": None,
        "explanation": "Insufficient test evidence.", "evidence": [], "status": status,
    }
    record = parse_response(json.dumps(response), build_prompt(tree, "baseline_2"), "000", "hf")
    assert record["status"] == status
    assert record["predicted_failure_mode"] is record["predicted_subtask"] is None


def test_stub_is_deterministic_and_explicitly_identifies_plumbing_results(tree):
    config = LLMConfig()
    backend = StubBackend(config)
    first = run_inference([tree], ["baseline_2", "baseline_3"], config, backend=backend)
    second = run_inference([tree], ["baseline_2", "baseline_3"], config, backend=StubBackend(config))
    assert first.predictions == second.predictions
    assert len(first.predictions) == 2
    assert all(set(record) == RECORD_KEYS for record in first.predictions)
    assert all(record["backend"] == "stub" and record["status"] == "abstain" for record in first.predictions)
    assert "stub" in json.dumps(first.manifest).lower()


def test_fake_model_roundtrip_records_all_requested_keys_without_mutating_tree(tree):
    config = LLMConfig()
    backend = FakeBackend()
    snapshot = tree.model_dump_json()
    result = run_inference([tree], ["baseline_2", "baseline_3"], config, backend=backend)
    assert len(backend.generated) == len(result.predictions) == 2
    assert [(r["pair_id"], r["model_variant"]) for r in result.predictions] == [
        ("000", "baseline_2"), ("000", "baseline_3"),
    ]
    assert all(r["status"] == "ok" and r["backend"] == "hf" for r in result.predictions)
    assert tree.model_dump_json() == snapshot
    manifest = json.dumps(result.manifest)
    assert "prompt_sha256" in manifest and "input_tokens" in manifest and "output_tokens" in manifest
    assert config.revision in manifest
    for request, prompt in zip(result.manifest["requests"], backend.generated):
        assert request["prompt_sha256"] == prompt.prompt_sha256
        assert request["template_sha256"] == prompt.template_sha256
        assert request["input_tokens"] == 64 and request["output_tokens"] == 32
        assert request["status"] == "ok" and request["latency_s"] >= 0
    assert result.manifest["decoding"] == {
        "do_sample": False, "num_beams": 1, "batch_size": 1,
        "seed": config.seed, "max_new_tokens": config.max_new_tokens,
    }


def test_invalid_generated_text_is_accounted_for_as_error_for_every_variant(tree):
    backend = FakeBackend(text="unstructured model response")
    result = run_inference([tree], ["baseline_2", "baseline_3"], LLMConfig(), backend=backend)
    assert len(result.predictions) == 2
    assert all(r["status"] == "error" and r["evidence"] == [] for r in result.predictions)
    assert all(r["predicted_failure_mode"] is r["predicted_subtask"] is None for r in result.predictions)


def test_context_overflow_is_explicit_and_never_generates(tree):
    config = LLMConfig()
    backend = FakeBackend(tokens=config.context_window_tokens)
    result = run_inference([tree], ["baseline_2", "baseline_3"], config, backend=backend)
    assert not backend.generated
    assert len(result.predictions) == 2
    assert all(r["status"] == "error" for r in result.predictions)
    assert all("context" in r["explanation"].lower() or "token" in r["explanation"].lower() for r in result.predictions)


@pytest.mark.parametrize("trees,variants", [([], ["baseline_2"]), (["tree"], []), (["tree"], ["baseline_2", "baseline_2"])])
def test_invalid_batch_request_is_rejected_without_generation(tree, trees, variants):
    backend = FakeBackend()
    inputs = [tree if item == "tree" else item for item in trees]
    with pytest.raises((ValueError, TypeError)):
        run_inference(inputs, variants, LLMConfig(), backend=backend)
    assert not backend.generated


def test_duplicate_pair_ids_rejected_before_generation(tree):
    backend = FakeBackend()
    with pytest.raises(ValueError, match="duplicate|unique"):
        run_inference([tree, tree.model_copy(deep=True)], ["baseline_2"], LLMConfig(), backend=backend)
    assert not backend.generated


@pytest.mark.parametrize("backend_type", [llm.StubBackend, llm.HFBackend])
def test_builtin_backend_cannot_hide_a_different_config_in_provenance(tree, backend_type):
    config = LLMConfig()
    backend = backend_type(LLMConfig(seed=config.seed + 1))
    with pytest.raises(ValueError, match="config"):
        run_inference([tree], ["baseline_2"], config, backend=backend)


def test_uncomputed_main_input_is_reported_without_model_calls(tree):
    class NoCalls(FakeBackend):
        def count_tokens(self, prompt):
            pytest.fail("uncomputed main input reached tokenization")

    backend = NoCalls()
    result = run_inference([tree], ["main"], LLMConfig(), backend=backend)
    record, = result.predictions
    assert record["status"] == "error" and "divergence" in record["explanation"]
    assert not backend.generated


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_token_count_never_generates(tree, count):
    backend = FakeBackend(tokens=count)
    result = run_inference([tree], ["baseline_2"], LLMConfig(), backend=backend)
    assert result.predictions[0]["status"] == "error"
    assert not backend.generated


@pytest.mark.parametrize("input_tokens,output_tokens", [(63, 32), (64, -1), (64, 513), (64, True)])
def test_inconsistent_generation_counts_are_errors_without_invented_labels(tree, input_tokens, output_tokens):
    class BadCounts(FakeBackend):
        def generate(self, prompt):
            return Generation(json.dumps(model_response(prompt)), input_tokens, output_tokens)

    result = run_inference([tree], ["baseline_2"], LLMConfig(), backend=BadCounts())
    record, = result.predictions
    assert record["status"] == "error" and record["predicted_failure_mode"] is None


def test_cli_default_stub_publishes_separate_predictions_and_source_provenance(tmp_path, tree, capsys):
    inputs = tmp_path / "bt"
    second = write_tree(inputs, tree, pair_id="z")
    first = write_tree(inputs, tree, pair_id="a")
    (inputs / "parse_report.json").write_text("{}")
    output = tmp_path / "run"
    before = {path.name: path.read_bytes() for path in inputs.iterdir()}
    assert run_llm.main(cli_args(inputs, output)) == 0
    assert "STUB" in capsys.readouterr().out
    assert {path.name for path in output.iterdir()} == {"baseline_2.jsonl", "baseline_3.jsonl", "run_manifest.json", "completions.jsonl"}
    for variant in ("baseline_2", "baseline_3"):
        predictions = [json.loads(line) for line in (output / f"{variant}.jsonl").read_text().splitlines()]
        assert [record["pair_id"] for record in predictions] == ["a", "z"]
        assert all(record["model_variant"] == variant and record["backend"] == "stub" for record in predictions)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["config_file_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert manifest["input_files"] == [
        {"path": str(path.resolve()), "pair_id": pair_id, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path, pair_id in ((first, "a"), (second, "z"))
    ]
    assert {path.name: path.read_bytes() for path in inputs.iterdir()} == before


@pytest.mark.parametrize("problem", ["malformed", "duplicate", "mismatch", "empty", "directory", "bad_filename"])
def test_cli_invalid_input_prevents_backend_creation_and_all_writes(tmp_path, tree, monkeypatch, capsys, problem):
    inputs = tmp_path / "bt"
    path = write_tree(inputs, tree)
    if problem == "malformed":
        path.write_text("{bad JSON")
    elif problem == "duplicate":
        write_tree(inputs, tree, filename="pair_other_bt.json")
    elif problem == "mismatch":
        path.rename(inputs / "pair_other_bt.json")
    elif problem == "empty":
        path.unlink()
    elif problem == "directory":
        path.unlink()
        path.mkdir()
    else:
        path.rename(inputs / "pair__bt.json")

    def unexpected_backend(*args, **kwargs):
        pytest.fail("invalid inputs must be rejected before constructing a backend")

    monkeypatch.setattr(run_llm, "StubBackend", unexpected_backend)
    output = tmp_path / "run"
    assert run_llm.main(cli_args(inputs, output)) == 2
    assert not output.exists()
    message = capsys.readouterr().err
    assert "error:" in message
    if problem not in {"empty", "bad_filename"}:
        assert "pair_" in message


def test_cli_rejects_existing_results_and_duplicate_variants_before_inference(tmp_path, tree, monkeypatch):
    inputs = tmp_path / "bt"
    write_tree(inputs, tree)
    output = tmp_path / "run"
    output.mkdir()
    old = output / "old.jsonl"
    old.write_text("keep these results")
    monkeypatch.setattr(run_llm, "StubBackend", lambda *a, **k: pytest.fail("unexpected backend"))
    assert run_llm.main(cli_args(inputs, output)) == 2
    assert old.read_text() == "keep these results"
    fresh = tmp_path / "fresh"
    assert run_llm.main(cli_args(inputs, fresh, "--variants", "baseline_2", "baseline_2")) == 2
    assert not fresh.exists()


def test_cli_prediction_errors_are_published_and_exit_one(tmp_path, tree, monkeypatch):
    inputs = tmp_path / "bt"
    write_tree(inputs, tree)
    monkeypatch.setattr(run_llm, "StubBackend", lambda config: FakeBackend(text="malformed response"))
    output = tmp_path / "run"
    assert run_llm.main(cli_args(inputs, output)) == 1
    assert (output / "run_manifest.json").exists()
    for variant in ("baseline_2", "baseline_3"):
        record = json.loads((output / f"{variant}.jsonl").read_text())
        assert record["status"] == "error"


def test_cli_publish_failure_cleans_staged_files_and_keeps_empty_destination(tmp_path, tree, monkeypatch):
    inputs = tmp_path / "bt"
    write_tree(inputs, tree)
    output = tmp_path / "run"
    output.mkdir()
    original = Path.write_text

    def fail_manifest(path, *args, **kwargs):
        if path.name == "run_manifest.json":
            raise OSError("synthetic storage failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_manifest)
    assert run_llm.main(cli_args(inputs, output)) == 2
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".llm-*"))


def test_cli_check_access_needs_no_inputs_or_outputs_and_never_constructs_backend(monkeypatch, capsys):
    monkeypatch.setattr(run_llm, "capabilities", lambda config: {"network_contacted": False, "weights_loaded": False})
    for name in ("StubBackend", "HFBackend"):
        monkeypatch.setattr(run_llm, name, lambda *a, **k: pytest.fail("access check loaded a backend"))
    assert run_llm.main(["--check-access"]) == 0
    assert json.loads(capsys.readouterr().out) == {"network_contacted": False, "weights_loaded": False}


def test_cli_module_runs_from_repository_root_without_llm_extras(tmp_path):
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_llm.py"), "--bt-dir", str(ROOT / "data/sample"),
         "--variants", "baseline_2", "--output-dir", str(tmp_path / "run")],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    assert "STUB" in process.stdout


@pytest.mark.parametrize("source", ["llm: [unterminated", "llm:\n  seed: invalid"])
def test_cli_invalid_config_has_clean_setup_exit_and_no_outputs(tmp_path, capsys, source):
    config = tmp_path / "invalid.yaml"
    config.write_text(source)
    output = tmp_path / "output"
    assert run_llm.main(["--config", str(config), "--bt-dir", str(ROOT / "data/sample"),
                         "--output-dir", str(output)]) == 2
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


def test_error_response_must_not_carry_evidence(tree):
    prompt = build_prompt(tree, "baseline_2")
    response = model_response(prompt)
    response.update(status="error", predicted_failure_mode=None, predicted_subtask=None)
    with pytest.raises(ValueError):
        parse_response(json.dumps(response), prompt, "000", "hf")


def test_custom_prompt_paths_resolve_relative_to_the_config(tmp_path):
    path = tmp_path / "llm.yaml"
    path.write_text("llm:\n  prompt_dir: templates\n")
    assert load_llm_config(path).prompt_dir == (tmp_path / "templates").resolve()


def fake_hf_modules(monkeypatch, *, native_context=131072, chat_template="test chat template",
                    continuations=None, mps_available=False, metal_available=True,
                    resolved_revision=llm.MODEL_REVISION, malformed_mask=False, kernel_source=None,
                    cuda_available=False, cuda_memory=None, mps_allocator_available=True):
    """Local tensor/runtime doubles; no weights, Hub calls, or torch imports."""
    calls = {
        "imports": [], "tokenizers": [], "models": [], "chat": [],
        "generate": [], "seeds": [], "devices": [], "decoded": [], "eval": 0,
        "metal_configs": [], "kernel_loads": 0, "mps_fractions": [], "cuda_memory_queries": 0,
    }

    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(64 * llm.GIB, "test_snapshot", 56 * llm.GIB))
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)

    class FakeTensor:
        def __init__(self, tokens):
            self.tokens = np.array(tokens, dtype=np.int64)
            self.shape = self.tokens.shape

        def to(self, device):
            calls["devices"].append(device)
            return self

        def __getitem__(self, index):
            return FakeTensor(self.tokens[index])

        def tolist(self):
            return self.tokens.tolist()

    class FakeTokenizer:
        eos_token_id = 128001
        eos_token = "<eos>"
        pad_token = None
        init_kwargs = {"_commit_hash": resolved_revision}

        def __init__(self):
            self.chat_template = chat_template
            self.padding_side = "right"

        @property
        def pad_token_id(self):
            return self.eos_token_id if self.pad_token else None

        def apply_chat_template(self, conversations, **kwargs):
            calls["chat"].append((conversations, kwargs))
            assert self.padding_side == "left"
            rows = [[11, 12, 13, 14, 15] if '"success_samples"' in messages[-1]["content"] else [11, 12, 13]
                    for messages in conversations]
            width = max(map(len, rows))
            masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
            if malformed_mask:
                masks[0] = [1, 0] + [1] * (width - 2)
            ids = [[self.pad_token_id] * (width - len(row)) + row for row in rows]
            return {"input_ids": FakeTensor(ids), "attention_mask": FakeTensor(masks)}

        def decode(self, tokens, **kwargs):
            calls["decoded"].append((tokens, kwargs))
            return '{"generated": "continuation only"}'

    class FakeModel:
        config = SimpleNamespace(max_position_embeddings=native_context, _commit_hash=resolved_revision)
        generation_config = SimpleNamespace(eos_token_id=[128001, 128009])

        def to(self, device):
            calls["devices"].append(device)
            return self

        def eval(self):
            calls["eval"] += 1

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            ids = kwargs["input_ids"].tolist()
            suffixes = continuations if continuations is not None else [[21, 22]] * len(ids)
            width = max(map(len, suffixes))
            return FakeTensor([prefix + suffix + [128001] * (width - len(suffix))
                               for prefix, suffix in zip(ids, suffixes)])

    def tokenizer_loader(model_id, **kwargs):
        calls["tokenizers"].append((model_id, kwargs))
        return FakeTokenizer()

    def model_loader(model_id, **kwargs):
        calls["models"].append((model_id, kwargs))
        model = FakeModel()
        if "quantization_config" in kwargs:
            model.hf_quantizer = SimpleNamespace(quantization_config=kwargs["quantization_config"])
            model.hf_device_map = {"": "mps"}
        return model

    def metal_config(**kwargs):
        calls["metal_configs"].append(kwargs)
        return SimpleNamespace(**kwargs)

    def metal_kernel():
        calls["kernel_loads"] += 1
        return SimpleNamespace(__file__=str(kernel_source or Path(__file__).resolve()), __name__="fake_metal_kernel")

    def cuda_mem_get_info():
        calls["cuda_memory_queries"] += 1
        return cuda_memory if cuda_memory is not None else (32 * llm.GIB, 48 * llm.GIB)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available, mem_get_info=cuda_mem_get_info, is_bf16_supported=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
        mps=SimpleNamespace(**({"set_per_process_memory_fraction": lambda fraction: calls["mps_fractions"].append(fraction)} if mps_allocator_available else {})),
        float32="fake-float32", float16="fake-float16", bfloat16="fake-bfloat16",
        manual_seed=lambda seed: calls["seeds"].append(seed), inference_mode=nullcontext,
    )
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    if metal_available:
        transformers.MetalConfig = metal_config
    original = llm.importlib.import_module

    def importer(name, *args, **kwargs):
        fake_modules = {
            "torch": torch, "transformers": transformers,
            "kernels": SimpleNamespace(__version__="fake-kernels"),
            "transformers.integrations.metal_quantization": SimpleNamespace(_get_metal_kernel=metal_kernel),
        }
        if name in fake_modules:
            calls["imports"].append(name)
            return fake_modules[name]
        return original(name, *args, **kwargs)

    monkeypatch.setattr(llm.importlib, "import_module", importer)
    return calls


def test_hf_adapter_is_lazy_pinned_and_decodes_only_the_generated_continuation(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch)
    config = LLMConfig(device="cpu", dtype="float32", max_new_tokens=8)
    backend = llm.HFBackend(config)
    assert calls["imports"] == calls["models"] == calls["tokenizers"] == []
    prompt = build_prompt(tree, "baseline_2")
    assert backend.count_tokens(prompt) == 3
    assert not calls["models"]
    result = backend.generate(prompt)
    assert result == Generation('{"generated": "continuation only"}', 3, 2, finish_reason="unknown")
    assert calls["tokenizers"] == [(config.model_id, {
        "revision": config.revision, "local_files_only": True, "trust_remote_code": False,
    })]
    assert calls["models"] == [(config.model_id, {
        "revision": config.revision, "local_files_only": True, "trust_remote_code": False,
        "torch_dtype": "fake-float32",
    })]
    assert all(messages == [prompt.messages] for messages, _ in calls["chat"])
    assert all(options == {
        "tokenize": True, "add_generation_prompt": True, "return_tensors": "pt", "return_dict": True,
        "padding": True, "truncation": False,
    } for _, options in calls["chat"])
    assert calls["decoded"] == [([21, 22], {"skip_special_tokens": True})]
    assert calls["seeds"] == [config.seed]
    assert calls["eval"] == 1
    generation_config = calls["generate"][0]["generation_config"]
    assert vars(generation_config) == {
        "do_sample": False, "num_beams": 1, "max_new_tokens": 8,
        "eos_token_id": [128001, 128009], "pad_token_id": 128001,
    }
    backend.generate(prompt)
    assert len(calls["models"]) == len(calls["tokenizers"]) == 1
    assert calls["seeds"] == [config.seed, config.seed]


def test_hf_context_overflow_does_not_open_weights_or_generate(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch)
    config = LLMConfig(device="cpu", dtype="float32", context_window_tokens=10, max_new_tokens=8)
    with pytest.raises(ValueError, match="context"):
        llm.HFBackend(config).generate(build_prompt(tree, "baseline_2"))
    assert len(calls["tokenizers"]) == 1
    assert not calls["models"] and not calls["generate"]


def test_hf_native_context_rejection_stays_rejected_for_later_requests(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, native_context=16)
    config = LLMConfig(device="cpu", dtype="float32", context_window_tokens=32, max_new_tokens=8)
    backend = llm.HFBackend(config)
    for variant in ("baseline_2", "baseline_3"):
        with pytest.raises(ValueError, match="context"):
            backend.generate(build_prompt(tree, variant))
    assert not calls["generate"]


def test_hf_requires_the_checkpoint_chat_template(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, chat_template=None)
    with pytest.raises(ValueError, match="chat template"):
        llm.HFBackend(LLMConfig(device="cpu", dtype="float32")).count_tokens(build_prompt(tree, "baseline_2"))
    assert not calls["models"] and not calls["generate"]


def test_hf_missing_optional_dependencies_has_actionable_error(tree, monkeypatch):
    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(64 * llm.GIB, "test_snapshot", 56 * llm.GIB))
    original = llm.importlib.import_module

    def importer(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            raise ImportError("synthetic missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(llm.importlib, "import_module", importer)
    with pytest.raises(RuntimeError, match=r"optional.*\[llm\]"):
        llm.HFBackend(LLMConfig()).count_tokens(build_prompt(tree, "baseline_2"))


def test_hf_unavailable_device_fails_without_fallback_or_weights(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        llm.HFBackend(LLMConfig(device="cuda")).count_tokens(build_prompt(tree, "baseline_2"))
    assert not calls["models"] and not calls["tokenizers"]


def test_capability_check_reads_only_local_runtime_and_never_credentials_or_weights(monkeypatch):
    calls = fake_hf_modules(monkeypatch)
    monkeypatch.setattr(llm, "_versions", lambda: {"torch": "fake", "transformers": "fake"})
    monkeypatch.setenv("HF_TOKEN", "secret-token-must-not-be-reported")
    result = llm.capabilities(LLMConfig())
    assert result["remote_access"] == "not_checked"
    assert result["weights"] == "not_loaded"
    assert "secret-token" not in json.dumps(result)
    assert calls["imports"] == []
    assert result["resources"]["host_admissible"] is True
    assert result["cuda_available"] is result["mps_available"] is None
    assert not calls["models"] and not calls["tokenizers"]


def test_hf_batch_uses_left_padding_real_masks_and_eos_lengths(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, continuations=[[21, 128009], [31, 32, 128001]])
    config = LLMConfig(device="cpu", dtype="float32", batch_size=2, max_new_tokens=8)
    backend = llm.HFBackend(config)
    prompts = [build_prompt(tree, "baseline_2"), build_prompt(tree, "baseline_3")]
    outputs = backend.generate_batch(prompts)
    assert [row.input_tokens for row in outputs] == [3, 5]
    assert [row.output_tokens for row in outputs] == [2, 3]
    assert [row.finish_reason for row in outputs] == ["eos", "eos"]
    assert len(calls["generate"]) == 1
    generated = calls["generate"][0]
    assert generated["input_ids"].tolist() == [[128001, 128001, 11, 12, 13], [11, 12, 13, 14, 15]]
    assert generated["attention_mask"].tolist() == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]
    assert calls["decoded"] == [([21, 128009], {"skip_special_tokens": True}),
                                ([31, 32, 128001], {"skip_special_tokens": True})]
    provenance = backend.provenance()
    assert provenance["tokenizer"]["padding_side"] == "left"
    assert provenance["model"]["resolved_revision"] == config.revision
    assert provenance["tokenizer"]["resolved_revision"] == config.revision
    assert provenance["native_context_window_tokens"] == 131072


def test_hf_marks_completion_limit_without_inventing_an_eos(tree, monkeypatch):
    fake_hf_modules(monkeypatch, continuations=[[21, 22]])
    backend = llm.HFBackend(LLMConfig(device="cpu", dtype="float32", max_new_tokens=2))
    output = backend.generate(build_prompt(tree, "baseline_2"))
    assert output.output_tokens == 2 and output.finish_reason == "length"


def test_hf_rejects_a_batch_containing_an_overlong_row_before_weights(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch)
    backend = llm.HFBackend(LLMConfig(device="cpu", dtype="float32", batch_size=2,
                                    context_window_tokens=10, max_new_tokens=6))
    with pytest.raises(ValueError, match="context"):
        backend.generate_batch([build_prompt(tree, "baseline_2"), build_prompt(tree, "baseline_3")])
    assert not calls["models"] and not calls["generate"]


def test_hf_rejects_right_padding_instead_of_using_wrong_attention(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, malformed_mask=True)
    with pytest.raises(ValueError, match="left padding"):
        llm.HFBackend(LLMConfig(device="cpu", dtype="float32")).generate(build_prompt(tree, "baseline_2"))
    assert not calls["models"]


def test_hf_batch_size_is_an_explicit_limit(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch)
    with pytest.raises(ValueError, match="batch_size"):
        llm.HFBackend(LLMConfig(device="cpu", dtype="float32")).generate_batch([
            build_prompt(tree, "baseline_2"), build_prompt(tree, "baseline_3"),
        ])
    assert not calls["tokenizers"] and not calls["models"]


def test_hf_metal_loading_uses_quantized_direct_mps_placement_and_kernel_provenance(tree, monkeypatch, tmp_path):
    calls = fake_hf_modules(monkeypatch, mps_available=True)
    config = LLMConfig(device="mps", dtype="float16", quantization="metal4",
                       local_files_only=False, cache_dir=tmp_path)
    backend = llm.HFBackend(config)
    backend.generate(build_prompt(tree, "baseline_2"))
    assert calls["metal_configs"] == [{"bits": 4, "group_size": 64, "dequantize": False}]
    assert calls["kernel_loads"] == 1
    kwargs = calls["models"][0][1]
    assert kwargs["device_map"] == "mps" and kwargs["local_files_only"] is False
    assert kwargs["cache_dir"] == str(tmp_path)
    assert calls["tokenizers"][0][1]["cache_dir"] == str(tmp_path)
    assert kwargs["torch_dtype"] == "fake-float16"
    # Only input_ids and attention_mask move; the model never gets a full .to().
    assert calls["devices"] == ["mps", "mps"]
    provenance = backend.provenance()
    assert provenance["kernel"]["loaded"] is True
    assert provenance["kernel"]["repository"] == llm.HFBackend.METAL_KERNEL_REPOSITORY
    assert provenance["kernel"]["module_file_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert provenance["effective_quantization"] == {"bits": 4, "group_size": 64, "dequantize": False}
    assert provenance["device_map"] == {"": "mps"}


def test_hf_metal_requires_explicit_runtime_support_before_weights(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, mps_available=True, metal_available=False)
    config = LLMConfig(device="mps", dtype="float16", quantization="metal4", local_files_only=False)
    with pytest.raises(RuntimeError, match="MetalConfig"):
        llm.HFBackend(config).generate(build_prompt(tree, "baseline_2"))
    assert not calls["models"] and not calls["generate"]


def test_metal_kernel_revision_comes_from_snapshot_path_before_symlink_resolution(tree, monkeypatch, tmp_path):
    revision = "6ed0fc4eff016df00830678eb96acb92a6a8f351"
    blob = tmp_path / "blobs" / "content-addressed-artifact"
    blob.parent.mkdir()
    blob.write_text("# synthetic kernel artifact\n")
    snapshot = tmp_path / "kernels--test" / "snapshots" / revision / "build" / "torch214-metal" / "__init__.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    fake_hf_modules(monkeypatch, mps_available=True, kernel_source=snapshot)
    config = LLMConfig(device="mps", dtype="float16", quantization="metal4", local_files_only=False,
                       metal_kernel_revision=revision)
    backend = llm.HFBackend(config)
    backend.generate(build_prompt(tree, "baseline_2"))
    kernel = backend.provenance()["kernel"]
    assert kernel["resolved_revision"] == kernel["expected_revision"] == revision
    assert kernel["module_file"] == str(snapshot)
    assert kernel["module_resolved_file"] == str(blob)
    assert kernel["module_file_sha256"] == hashlib.sha256(blob.read_bytes()).hexdigest()


@pytest.mark.parametrize("identifiable", [True, False])
def test_metal_kernel_revision_guard_refuses_drift_before_weights(tree, monkeypatch, tmp_path, identifiable):
    expected = "6ed0fc4eff016df00830678eb96acb92a6a8f351"
    actual = "1" * 40
    source = tmp_path / "snapshots" / actual / "kernel.py" if identifiable else tmp_path / "kernel.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# synthetic kernel\n")
    calls = fake_hf_modules(monkeypatch, mps_available=True, kernel_source=source)
    config = LLMConfig(device="mps", dtype="float16", quantization="metal4", local_files_only=False,
                       metal_kernel_revision=expected)
    backend = llm.HFBackend(config)
    with pytest.raises(ValueError, match="kernel revision"):
        backend.generate(build_prompt(tree, "baseline_2"))
    assert not calls["models"] and not calls["generate"]
    kernel = backend.provenance()["kernel"]
    assert kernel["expected_revision"] == expected
    assert kernel["resolved_revision"] == (actual if identifiable else None)


@pytest.mark.parametrize("update", [
    {"batch_size": 0}, {"batch_size": True}, {"batch_size": 129},
    {"quantization": "metal4", "device": "cpu", "dtype": "float16", "local_files_only": False},
    {"quantization": "metal4", "device": "mps", "dtype": "float32", "local_files_only": False},
    {"quantization": "metal4", "device": "mps", "dtype": "float16", "local_files_only": True},
    {"quantization_group_size": 128},
    {"metal_kernel_revision": "main"}, {"metal_kernel_revision": "z" * 40},
])
def test_batch_and_metal_config_reject_unsupported_settings(update):
    with pytest.raises(ValueError):
        LLMConfig.model_validate(update)


def test_local_profile_pins_the_same_checkpoint_and_resolves_repo_weight_cache():
    config = load_llm_config(ROOT / "configs/llm-local.yaml")
    assert config.model_id == LLMConfig().model_id and config.revision == LLMConfig().revision
    assert config.device == "mps" and config.dtype == "float16" and config.quantization == "metal4"
    assert config.context_window_tokens == 32768 and config.max_new_tokens == 1024
    assert config.cache_dir == ROOT / "weights/hub"
    assert config.local_files_only is False
    assert config.metal_kernel_revision == "6ed0fc4eff016df00830678eb96acb92a6a8f351"


def test_raw_completions_and_exact_prompts_are_retained_when_parsing_fails(tree):
    raw = "  invalid JSON\nwith exact whitespace\t"
    result = run_inference([tree], ["baseline_2"], LLMConfig(), backend=FakeBackend(text=raw))
    completion, = result.completions
    assert completion["text"] == raw and completion["status"] == "error"
    assert completion["messages"] == build_prompt(tree, "baseline_2").messages
    assert completion["completion_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert completion["error_type"] == "JSONDecodeError"
    assert result.manifest["requests"][0]["completion_sha256"] == completion["completion_sha256"]


def test_harness_forms_real_batches_and_keeps_individual_errors_and_order(tree):
    second = tree.model_copy(deep=True)
    second.metadata.pair_id = "second"

    class Batched(FakeBackend):
        def __init__(self):
            super().__init__()
            self.batches = []

        def generate(self, prompt):
            pytest.fail("batch-capable backend used single-request generation")

        def generate_batch(self, prompts):
            self.batches.append(prompts)
            if len(self.batches) == 1:
                return [Generation(json.dumps(model_response(prompts[0])), 64, 32), ValueError("one sample failed")]
            return [Generation(json.dumps(model_response(prompt)), 64, 32) for prompt in prompts]

    backend = Batched()
    result = run_inference([tree, second], ["baseline_2", "baseline_3"], LLMConfig(batch_size=2), backend=backend)
    assert [len(batch) for batch in backend.batches] == [2, 2]
    assert [row["pair_id"] for row in result.predictions] == ["000", "000", "second", "second"]
    assert [row["status"] for row in result.predictions] == ["ok", "error", "ok", "ok"]
    assert len(result.completions) == 3
    assert result.manifest["decoding"]["batch_size"] == 2
    assert len(result.manifest["batches"]) == 2
    first, second_request = result.manifest["requests"][:2]
    assert first["batch_latency_s"] == second_request["batch_latency_s"]
    assert result.manifest["batches"][0]["latency_scope"] == "shared_batch_generation_wall_time"


def test_harness_batch_exception_accounts_for_each_affected_row_and_continues(tree):
    second = tree.model_copy(deep=True)
    second.metadata.pair_id = "second"

    class FailingBatch(FakeBackend):
        calls = 0

        def generate_batch(self, prompts):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic batch error")
            return [Generation(json.dumps(model_response(prompt)), 64, 32) for prompt in prompts]

    backend = FailingBatch()
    result = run_inference([tree, second], ["baseline_2", "baseline_3"], LLMConfig(batch_size=2), backend=backend)
    assert [row["status"] for row in result.predictions] == ["error", "error", "ok", "ok"]
    assert len(result.completions) == 2 and backend.calls == 2


def test_harness_excludes_overlong_rows_from_batches_without_dropping_them(tree):
    class Budgeted(FakeBackend):
        batches = []

        def count_tokens(self, prompt):
            return 1000 if prompt.variant == "baseline_3" else 64

        def generate_batch(self, prompts):
            self.batches.append(prompts)
            return [Generation(json.dumps(model_response(prompt)), 64, 32) for prompt in prompts]

    backend = Budgeted()
    result = run_inference([tree], ["baseline_2", "baseline_3"],
                           LLMConfig(context_window_tokens=800, batch_size=2), backend=backend)
    assert [row["status"] for row in result.predictions] == ["ok", "error"]
    assert len(backend.batches) == 1 and len(backend.batches[0]) == 1
    assert result.manifest["requests"][1]["input_tokens"] == 1000
    assert len(result.completions) == 1


@pytest.mark.parametrize("field,value", [
    ("input_tokens", float("nan")), ("input_tokens", True),
    ("output_tokens", float("nan")), ("output_tokens", float("inf")),
    ("output_tokens", object()), ("token_count_method", object()),
    ("token_count_method", float("nan")), ("finish_reason", object()),
    ("finish_reason", float("inf")),
])
def test_invalid_generation_metadata_preserves_raw_text_in_json_safe_error_records(tree, field, value):
    raw = " exact raw response remains available\n"

    class InvalidMetadata(FakeBackend):
        def generate(self, prompt):
            fields = {"text": raw, "input_tokens": 64, "output_tokens": 32,
                      "token_count_method": "tokenizer", "finish_reason": None}
            fields[field] = value
            return Generation(**fields)

    result = run_inference([tree], ["baseline_2"], LLMConfig(), backend=InvalidMetadata())
    completion, = result.completions
    assert result.predictions[0]["status"] == completion["status"] == "error"
    assert completion["text"] == raw and completion["input_tokens"] == 64
    assert field in completion["metadata_error_fields"]
    assert "object at" not in json.dumps(completion, allow_nan=False)
    json.dumps(result.manifest, allow_nan=False)


@pytest.mark.parametrize("device,dtype,physical,available", [
    ("mps", "float16", 16, 12),
    ("cpu", "float32", 32, 28),
    ("cuda", "bfloat16", 16, 12),
    ("mps", "float16", None, 56),
    ("mps", "float16", 64, None),
    ("cpu", "float32", 64, 8),
])
def test_resource_guard_refuses_unsafe_or_unknown_ram_before_any_runtime_import(tree, monkeypatch, device, dtype, physical, available):
    snapshot = llm.ResourceSnapshot(None if physical is None else physical * llm.GIB, "synthetic_os_snapshot",
                                    None if available is None else available * llm.GIB)
    monkeypatch.setattr(llm, "_host_resources", lambda: snapshot)
    calls = []

    def forbidden_import(name, *args, **kwargs):
        calls.append(name)
        pytest.fail("resource rejection must precede optional runtime imports")

    monkeypatch.setattr(llm.importlib, "import_module", forbidden_import)
    config = LLMConfig(device=device, dtype=dtype)
    backend = llm.HFBackend(config)
    prompt = build_prompt(tree, "baseline_2")
    for invoke in (backend.count_tokens, backend.generate):
        with pytest.raises(llm.ResourceSafetyError):
            invoke(prompt)
    assert calls == []
    resources = backend.provenance()["resources"]
    assert resources["host_admissible"] is False
    assert resources["host_memory_bytes"] == snapshot.host_memory_bytes
    assert resources["available_host_memory_bytes"] == snapshot.available_host_memory_bytes
    assert backend.provenance()["weights_loaded"] is False


def test_metal4_does_not_bypass_16_gib_host_floor_or_load_its_kernel(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, mps_available=True)
    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(16 * llm.GIB, "synthetic", 12 * llm.GIB))
    config = LLMConfig(device="mps", dtype="float16", quantization="metal4", local_files_only=False)
    result = run_inference([tree], ["baseline_2", "baseline_3"], config, backend=llm.HFBackend(config))
    assert [row["status"] for row in result.predictions] == ["error", "error"]
    assert all("ResourceSafetyError" in row["explanation"] for row in result.predictions)
    assert result.completions == ()
    assert calls["imports"] == calls["tokenizers"] == calls["models"] == []
    assert calls["kernel_loads"] == 0 and calls["mps_fractions"] == []


def test_available_headroom_is_rechecked_immediately_before_kernel_and_weights(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, mps_available=True)
    snapshots = iter([
        llm.ResourceSnapshot(64 * llm.GIB, "before_tokenization", 56 * llm.GIB),
        llm.ResourceSnapshot(64 * llm.GIB, "before_weights", 8 * llm.GIB),
    ])
    monkeypatch.setattr(llm, "_host_resources", lambda: next(snapshots))
    config = LLMConfig(device="mps", dtype="float16", quantization="metal4", local_files_only=False)
    backend = llm.HFBackend(config)
    with pytest.raises(llm.ResourceSafetyError, match="headroom"):
        backend.generate(build_prompt(tree, "baseline_2"))
    assert len(calls["tokenizers"]) == 1
    assert calls["kernel_loads"] == 0 and calls["models"] == []
    assert backend.provenance()["resources"]["host_memory_source"] == "before_weights"


def test_cached_runtime_cannot_reuse_a_stale_host_admission(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch)
    backend = llm.HFBackend(LLMConfig(device="cpu", dtype="float32"))
    prompt = build_prompt(tree, "baseline_2")
    assert backend.count_tokens(prompt) == 3
    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(64 * llm.GIB, "memory_changed", 4 * llm.GIB))
    with pytest.raises(llm.ResourceSafetyError):
        backend.count_tokens(prompt)
    assert len(calls["chat"]) == 1 and not calls["models"]


@pytest.mark.parametrize("watermark", ["0", "0.0", "-1", "nan", "inf", "invalid"])
def test_disabled_or_invalid_mps_watermark_is_rejected_before_runtime(tree, monkeypatch, watermark):
    calls = fake_hf_modules(monkeypatch, mps_available=True)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", watermark)
    backend = llm.HFBackend(LLMConfig(device="mps", dtype="float16"))
    with pytest.raises(llm.ResourceSafetyError):
        backend.generate(build_prompt(tree, "baseline_2"))
    assert calls["imports"] == [] and not calls["models"]


@pytest.mark.parametrize("watermark,expected", [(None, 0.8), ("1.7", 0.8), ("0.5", 0.5)])
def test_safe_mps_admission_applies_a_bounded_allocator_preserving_stricter_user_limits(tree, monkeypatch, watermark, expected):
    calls = fake_hf_modules(monkeypatch, mps_available=True)
    if watermark is not None:
        monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", watermark)
    backend = llm.HFBackend(LLMConfig(device="mps", dtype="float16"))
    backend.generate(build_prompt(tree, "baseline_2"))
    assert calls["mps_fractions"] == [expected, expected, expected]
    resources = backend.provenance()["resources"]
    assert resources["mps_allocator_cap_applied"] is True
    assert resources["mps_allocator_fraction"] == expected


def test_missing_mps_allocator_cap_is_a_refusal_before_tokenizer_or_weights(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, mps_available=True, mps_allocator_available=False)
    backend = llm.HFBackend(LLMConfig(device="mps", dtype="float16"))
    with pytest.raises(llm.ResourceSafetyError, match="allocator cap"):
        backend.generate(build_prompt(tree, "baseline_2"))
    assert not calls["tokenizers"] and not calls["models"]


@pytest.mark.parametrize("dtype,free,total", [
    ("bfloat16", 12, 48), ("bfloat16", 48, 16), ("float32", 32, 48),
])
def test_cuda_requires_current_device_free_and_total_vram_before_tokenizer_or_weights(tree, monkeypatch, dtype, free, total):
    calls = fake_hf_modules(monkeypatch, cuda_available=True,
                            cuda_memory=(free * llm.GIB, total * llm.GIB))
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype=dtype))
    with pytest.raises(llm.ResourceSafetyError, match="VRAM"):
        backend.generate(build_prompt(tree, "baseline_2"))
    assert calls["cuda_memory_queries"] == 1
    assert not calls["tokenizers"] and not calls["models"]
    resources = backend.provenance()["resources"]
    assert resources["cuda_free_memory_bytes"] == free * llm.GIB


def test_cuda_free_vram_is_rechecked_after_tokenization_before_weights(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype="bfloat16"))
    prompt = build_prompt(tree, "baseline_2")
    backend.count_tokens(prompt)
    backend._torch.cuda.mem_get_info = lambda: (8 * llm.GIB, 48 * llm.GIB)
    with pytest.raises(llm.ResourceSafetyError, match="VRAM"):
        backend.generate(prompt)
    assert calls["models"] == []


def test_capabilities_never_imports_torch_and_reports_unsafe_host_admission(monkeypatch):
    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(16 * llm.GIB, "synthetic", 8 * llm.GIB))
    monkeypatch.setattr(llm.importlib, "import_module", lambda *args, **kwargs: pytest.fail("capability check imported a runtime"))
    result = llm.capabilities(LLMConfig(device="mps", dtype="float16"))
    assert result["resources"]["host_admissible"] is False
    assert result["cuda_available"] is result["mps_available"] is None
    assert result["weights"] == "not_loaded"


def test_posix_resource_probe_uses_os_metadata_without_optional_runtime_imports(monkeypatch):
    monkeypatch.setattr(llm.platform, "system", lambda: "Linux")
    values = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 64 * llm.GIB // 4096,
              "SC_AVPHYS_PAGES": 56 * llm.GIB // 4096}
    monkeypatch.setattr(llm.os, "sysconf", values.__getitem__)
    monkeypatch.setattr(llm, "_psutil_resources", lambda: llm.ResourceSnapshot(None, "psutil.unavailable"))
    assert llm._host_resources() == llm.ResourceSnapshot(64 * llm.GIB, "os.sysconf(physical_and_available_pages)", 56 * llm.GIB)


def test_unknown_os_ram_probe_fails_closed(monkeypatch):
    monkeypatch.setattr(llm.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llm, "_psutil_resources", lambda: llm.ResourceSnapshot(None, "psutil.unavailable"))

    def unavailable(name):
        raise OSError("synthetic unavailable sysconf")

    monkeypatch.setattr(llm.os, "sysconf", unavailable)
    snapshot = llm._host_resources()
    assert snapshot.host_memory_bytes is None
    assert llm._host_resource_assessment(LLMConfig(), snapshot)["host_admissible"] is False


@pytest.mark.parametrize("field", ["allow_unsafe_memory", "minimum_host_gib", "disable_resource_guard"])
def test_memory_policy_has_no_configuration_bypass(field):
    with pytest.raises(ValueError):
        LLMConfig.model_validate({field: True})


def test_windows_resource_probe_uses_psutil_total_and_available_without_sysconf(monkeypatch):
    monkeypatch.setattr(llm.platform, "system", lambda: "Windows")
    measured = llm.ResourceSnapshot(64 * llm.GIB, "psutil.virtual_memory", 48 * llm.GIB)
    monkeypatch.setattr(llm, "_psutil_resources", lambda: measured)

    def missing_sysconf(name):
        raise AttributeError("Windows has no sysconf")

    monkeypatch.setattr(llm.os, "sysconf", missing_sysconf)
    assert llm._host_resources() == measured
    assert llm._host_resource_assessment(LLMConfig(), measured)["host_admissible"] is True


def test_posix_prefers_reclaimable_available_memory_to_free_only_pages(monkeypatch):
    monkeypatch.setattr(llm.platform, "system", lambda: "Linux")
    measured = llm.ResourceSnapshot(64 * llm.GIB, "psutil.virtual_memory", 48 * llm.GIB)
    monkeypatch.setattr(llm, "_psutil_resources", lambda: measured)
    values = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 64 * llm.GIB // 4096,
              "SC_AVPHYS_PAGES": 1 * llm.GIB // 4096}
    monkeypatch.setattr(llm.os, "sysconf", values.__getitem__)
    snapshot = llm._host_resources()
    assert snapshot.available_host_memory_bytes == 48 * llm.GIB
    assert snapshot.source == "os.sysconf; available:psutil.virtual_memory"
    assert llm._host_resource_assessment(LLMConfig(), snapshot)["host_admissible"] is True


@pytest.mark.parametrize("total,available", [(True, 32), (64, True), (-1, 32), (64, float("inf"))])
def test_optional_psutil_measurements_require_positive_integer_bytes(monkeypatch, total, available):
    fake = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=total, available=available))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    snapshot = llm._psutil_resources()
    assert snapshot.host_memory_bytes == (total if type(total) is int and total > 0 else None)
    assert snapshot.available_host_memory_bytes == (available if type(available) is int and available > 0 else None)


def test_resident_cpu_model_does_not_reserve_checkpoint_loading_headroom_again(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch)
    backend = llm.HFBackend(LLMConfig(device="cpu", dtype="float32", max_new_tokens=8))
    backend.generate(build_prompt(tree, "baseline_2"))
    # The fake resident weights consumed 32 GiB. This is below initial 36 GiB
    # headroom but comfortably exceeds the next request's KV + workspace guard.
    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(64 * llm.GIB, "resident_cpu", 24 * llm.GIB))
    backend.generate(build_prompt(tree, "baseline_3"))
    assert len(calls["models"]) == 1 and len(calls["generate"]) == 2
    resources = backend.provenance()["resources"]
    assert resources["phase"] == "resident_execution"
    assert resources["minimum_available_host_memory_bytes_when_measurable"] < 24 * llm.GIB
    assert resources["kv_cache_estimate_bytes"] > 0


def test_resident_cuda_model_checks_request_kv_headroom_without_double_reserving_weights(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, cuda_available=True, cuda_memory=(24 * llm.GIB, 24 * llm.GIB))
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype="bfloat16", max_new_tokens=8))
    backend.generate(build_prompt(tree, "baseline_2"))
    backend._torch.cuda.mem_get_info = lambda: (8 * llm.GIB, 24 * llm.GIB)
    backend.generate(build_prompt(tree, "baseline_3"))
    assert len(calls["models"]) == 1 and len(calls["generate"]) == 2
    resources = backend.provenance()["resources"]
    assert resources["phase"] == "resident_execution"
    assert resources["minimum_cuda_memory_bytes"] < 8 * llm.GIB
    assert resources["minimum_cuda_memory_bytes"] == 4 * llm.GIB + resources["kv_cache_estimate_bytes"]


def test_resident_inference_still_refuses_insufficient_workspace(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch)
    backend = llm.HFBackend(LLMConfig(device="cpu", dtype="float32"))
    backend.generate(build_prompt(tree, "baseline_2"))
    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(64 * llm.GIB, "resident_exhausted", 3 * llm.GIB))
    with pytest.raises(llm.ResourceSafetyError, match="headroom"):
        backend.generate(build_prompt(tree, "baseline_3"))
    assert len(calls["generate"]) == 1


@pytest.mark.parametrize("cached_free_gib", [1, 4])
def test_resident_cuda_releases_unused_cache_only_when_existing_guard_would_refuse(
    tree, monkeypatch, cached_free_gib,
):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype="bfloat16", max_new_tokens=8))
    backend.generate(build_prompt(tree, "baseline_2"))
    memory = {"free": cached_free_gib * llm.GIB, "releases": 0}

    def empty_cache():
        memory["releases"] += 1
        memory["free"] = 8 * llm.GIB

    backend._torch.cuda.mem_get_info = lambda: (memory["free"], 32 * llm.GIB)
    backend._torch.cuda.empty_cache = empty_cache
    backend.generate(build_prompt(tree, "baseline_3"))
    assert len(calls["models"]) == 1 and len(calls["generate"]) == 2
    assert memory["releases"] == 1
    resources = backend.provenance()["resources"]
    assert resources["cuda_free_memory_bytes"] == 8 * llm.GIB
    assert resources["minimum_cuda_memory_bytes"] == 4 * llm.GIB + resources["kv_cache_estimate_bytes"]
    assert resources["cuda_cache_release"] == {
        "attempted": True,
        "reason": "resident_guard_insufficient_driver_free_memory",
        "free_before_bytes": cached_free_gib * llm.GIB,
        "free_after_bytes": 8 * llm.GIB,
        "minimum_cuda_memory_bytes": (4 * llm.GIB if cached_free_gib == 1
                                      else resources["minimum_cuda_memory_bytes"]),
    }
    assert resources["cuda_cache_releases"] == [resources["cuda_cache_release"]]


def test_resident_cuda_cache_release_does_not_relax_memory_threshold(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype="bfloat16", max_new_tokens=8))
    backend.generate(build_prompt(tree, "baseline_2"))
    memory = {"free": 1 * llm.GIB, "releases": 0}

    def empty_cache():
        memory["releases"] += 1
        memory["free"] = 2 * llm.GIB

    backend._torch.cuda.mem_get_info = lambda: (memory["free"], 32 * llm.GIB)
    backend._torch.cuda.empty_cache = empty_cache
    with pytest.raises(llm.ResourceSafetyError, match="VRAM"):
        backend.generate(build_prompt(tree, "baseline_3"))
    assert memory["releases"] == 1
    assert len(calls["models"]) == len(calls["generate"]) == 1
    resources = backend.provenance()["resources"]
    assert resources["cuda_free_memory_bytes"] == 2 * llm.GIB
    assert resources["minimum_cuda_memory_bytes"] == 4 * llm.GIB
    assert resources["cuda_cache_release"]["free_after_bytes"] == 2 * llm.GIB


def test_initial_cuda_admission_does_not_release_cache_to_bypass_checkpoint_guard(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype="bfloat16"))
    backend.count_tokens(build_prompt(tree, "baseline_2"))
    backend._torch.cuda.mem_get_info = lambda: (8 * llm.GIB, 32 * llm.GIB)
    backend._torch.cuda.empty_cache = lambda: pytest.fail("initial admission released cache")
    with pytest.raises(llm.ResourceSafetyError, match="VRAM"):
        backend.generate(build_prompt(tree, "baseline_2"))
    assert not calls["models"] and not calls["generate"]


def test_resident_cuda_does_not_release_cache_with_sufficient_free_memory(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype="bfloat16", max_new_tokens=8))
    backend.generate(build_prompt(tree, "baseline_2"))
    backend._torch.cuda.mem_get_info = lambda: (8 * llm.GIB, 32 * llm.GIB)
    backend._torch.cuda.empty_cache = lambda: pytest.fail("sufficient headroom released cache")
    backend.generate(build_prompt(tree, "baseline_3"))
    assert len(calls["models"]) == 1 and len(calls["generate"]) == 2
    assert "cuda_cache_release" not in backend.provenance()["resources"]


@pytest.mark.parametrize("free,total", [(True, 32 * llm.GIB), (1.5, 32 * llm.GIB), (32 * llm.GIB, None)])
def test_resident_cuda_rejects_invalid_memory_measurements_without_cache_release(
    tree, monkeypatch, free, total,
):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype="bfloat16", max_new_tokens=8))
    backend.generate(build_prompt(tree, "baseline_2"))
    backend._torch.cuda.mem_get_info = lambda: (free, total)
    backend._torch.cuda.empty_cache = lambda: pytest.fail("invalid measurements triggered cache release")
    with pytest.raises(llm.ResourceSafetyError, match="VRAM"):
        backend.generate(build_prompt(tree, "baseline_3"))
    assert len(calls["generate"]) == 1


def test_resident_cuda_without_cache_release_api_still_refuses_low_memory(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype="bfloat16", max_new_tokens=8))
    backend.generate(build_prompt(tree, "baseline_2"))
    backend._torch.cuda.mem_get_info = lambda: (1 * llm.GIB, 32 * llm.GIB)
    assert not hasattr(backend._torch.cuda, "empty_cache")
    with pytest.raises(llm.ResourceSafetyError, match="VRAM"):
        backend.generate(build_prompt(tree, "baseline_3"))
    assert len(calls["generate"]) == 1


@pytest.mark.parametrize("failure_stage", ["release", "remeasure"])
def test_resident_cuda_cache_recovery_failure_is_a_recorded_refusal(tree, monkeypatch, failure_stage):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    backend = llm.HFBackend(LLMConfig(device="cuda", dtype="bfloat16", max_new_tokens=8))
    backend.generate(build_prompt(tree, "baseline_2"))
    recovery = {"released": False, "attempts": 0}

    def empty_cache():
        recovery["attempts"] += 1
        if failure_stage == "release":
            raise RuntimeError("synthetic cache release failure")
        recovery["released"] = True

    def memory():
        if recovery["released"]:
            raise RuntimeError("synthetic memory remeasurement failure")
        return 1 * llm.GIB, 32 * llm.GIB

    backend._torch.cuda.mem_get_info = memory
    backend._torch.cuda.empty_cache = empty_cache
    with pytest.raises(llm.ResourceSafetyError, match="after releasing unused cache") as error:
        backend.generate(build_prompt(tree, "baseline_3"))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert recovery["attempts"] == 1 and len(calls["generate"]) == 1
    release = backend.provenance()["resources"]["cuda_cache_release"]
    assert release["attempted"] is True and release["error_type"] == "RuntimeError"


def test_bad_json_does_not_prevent_later_cuda_generation_after_cache_recovery(tree, monkeypatch):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    config = LLMConfig(device="cuda", dtype="bfloat16", max_new_tokens=8)
    backend = llm.HFBackend(config)
    second_prompt = build_prompt(tree, "baseline_3")
    backend.count_tokens(build_prompt(tree, "baseline_2"))
    memory = {"released": False, "attempts": 0}

    def current_memory():
        free = 1 * llm.GIB if calls["generate"] and not memory["released"] else 32 * llm.GIB
        return free, 32 * llm.GIB

    def empty_cache():
        memory["attempts"] += 1
        memory["released"] = True

    def decode(tokens, **kwargs):
        calls["decoded"].append((tokens, kwargs))
        return "not JSON" if len(calls["decoded"]) == 1 else json.dumps(model_response(second_prompt))

    backend._torch.cuda.mem_get_info = current_memory
    backend._torch.cuda.empty_cache = empty_cache
    backend._tokenizer.decode = decode
    result = run_inference([tree], ["baseline_2", "baseline_3"], config, backend=backend)
    assert [row["status"] for row in result.predictions] == ["error", "ok"]
    assert result.manifest["requests"][0]["error_type"] == "JSONDecodeError"
    assert "error_type" not in result.manifest["requests"][1]
    assert result.completions[0]["text"] == "not JSON"
    assert result.completions[0]["completion_sha256"] == hashlib.sha256(b"not JSON").hexdigest()
    assert len(result.completions) == 2
    assert len(calls["models"]) == 1 and len(calls["generate"]) == 2
    assert memory["attempts"] == 1
    resources = result.manifest["backend_provenance"]["resources"]
    assert resources["cuda_cache_releases"] == [resources["cuda_cache_release"]]
    assert resources["cuda_cache_release"]["free_before_bytes"] == 1 * llm.GIB
    assert resources["cuda_cache_release"]["free_after_bytes"] == 32 * llm.GIB
