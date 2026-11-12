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

import pytest
from pydantic import ValidationError

from scripts import run_llm
from telemetry_reasoning.models import llm
from telemetry_reasoning.models.llm import (
    Generation, LLMConfig, StubBackend, load_llm_config, parse_response, run_inference,
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
    assert {path.name for path in output.iterdir()} == {"baseline_2.jsonl", "baseline_3.jsonl", "run_manifest.json"}
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


def fake_hf_modules(monkeypatch, *, native_context=131072, chat_template="test chat template"):
    """Minimal torch/transformers stand-ins that expose all adapter operations."""
    calls = {
        "imports": [], "tokenizers": [], "models": [], "chat": [],
        "generate": [], "seeds": [], "devices": [], "decoded": [], "eval": 0,
    }

    class FakeTensor:
        def __init__(self, tokens, *, vector=False):
            self.tokens = tokens
            self.shape = (len(tokens),) if vector else (1, len(tokens))

        def to(self, device):
            calls["devices"].append(device)
            return self

        def __getitem__(self, index):
            assert isinstance(index, tuple) and index[0] == 0
            return FakeTensor(self.tokens[index[1]], vector=True)

    class FakeTokenizer:
        eos_token_id = 128001

        def __init__(self):
            self.chat_template = chat_template

        def apply_chat_template(self, messages, **kwargs):
            calls["chat"].append((messages, kwargs))
            return FakeTensor([11, 12, 13])

        def decode(self, tokens, **kwargs):
            calls["decoded"].append((tokens.tokens, kwargs))
            return '{"generated": "continuation only"}'

    class FakeModel:
        config = SimpleNamespace(max_position_embeddings=native_context)
        generation_config = SimpleNamespace(eos_token_id=[128001, 128009])

        def to(self, device):
            calls["devices"].append(device)
            return self

        def eval(self):
            calls["eval"] += 1

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return FakeTensor([11, 12, 13, 21, 22])

    def tokenizer_loader(model_id, **kwargs):
        calls["tokenizers"].append((model_id, kwargs))
        return FakeTokenizer()

    def model_loader(model_id, **kwargs):
        calls["models"].append((model_id, kwargs))
        return FakeModel()

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        float32="fake-float32", float16="fake-float16", bfloat16="fake-bfloat16",
        manual_seed=lambda seed: calls["seeds"].append(seed),
        inference_mode=nullcontext,
        ones_like=lambda tensor: FakeTensor([1] * tensor.shape[-1]),
    )
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader),
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    original = llm.importlib.import_module

    def importer(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            calls["imports"].append(name)
            return torch if name == "torch" else transformers
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
    assert result == Generation('{"generated": "continuation only"}', 3, 2)
    assert calls["tokenizers"] == [(config.model_id, {
        "revision": config.revision, "local_files_only": True, "trust_remote_code": False,
    })]
    assert calls["models"] == [(config.model_id, {
        "revision": config.revision, "local_files_only": True, "trust_remote_code": False,
        "torch_dtype": "fake-float32",
    })]
    assert all(messages == prompt.messages for messages, _ in calls["chat"])
    assert all(options == {
        "tokenize": True, "add_generation_prompt": True, "return_tensors": "pt", "truncation": False,
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
    assert calls["imports"] == ["torch"]
    assert not calls["models"] and not calls["tokenizers"]
