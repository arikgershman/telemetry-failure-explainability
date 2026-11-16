"""Persisted inference evidence and smoke acceptance, without model downloads."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_llm
from telemetry_reasoning.models import llm
from telemetry_reasoning.models.llm import Generation, LLMConfig, run_inference
from telemetry_reasoning.schemas import PairedBehaviorTree


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/sample"


class LocalHF:
    """An explicit adapter stand-in for successful/failed local HF execution."""

    name = "hf"

    def __init__(self, config, *, malformed=None, weights_loaded=True, fail_generation=False):
        self.config = config
        self.malformed = malformed
        self.weights_loaded = weights_loaded
        self.fail_generation = fail_generation

    def count_tokens(self, prompt):
        return 32

    def generate(self, prompt):
        if self.fail_generation:
            raise RuntimeError("synthetic generation failure")
        text = self.malformed
        if text is None:
            node = next(node for node in prompt.payload["subtasks"] if node["failure_samples"])
            row = node["failure_samples"][0]
            text = json.dumps({
                "predicted_failure_mode": "position_error", "predicted_subtask": node["subtask"],
                "explanation": "Synthetic adapter response for persistence tests.",
                "evidence": [{"subtask": node["subtask"], "timestamp": row["timestamp"],
                              "feature": "gripper_force", "value": row["gripper_force"]}],
                "status": "ok",
            })
        return Generation(text=text, input_tokens=32, output_tokens=32)

    def provenance(self):
        return {"weights_loaded": self.weights_loaded, "implementation": "local_test_double"}


def arguments(output, *extra):
    return ["--bt-dir", str(SAMPLE), "--output-dir", str(output), *extra]


def read_manifest(output):
    return json.loads((output / "run_manifest.json").read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def forbid_heavy_imports(monkeypatch):
    """Fail before importing any optional model, tokenizer, or device runtime."""
    imported = []
    original = llm.importlib.import_module

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "kernels", "huggingface_hub", "accelerate"}:
            imported.append(name)
            pytest.fail(f"resource refusal must precede optional runtime import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(llm.importlib, "import_module", guarded)
    return imported


@pytest.mark.parametrize("settings", [
    {"device": "mps", "dtype": "float16", "quantization": "metal4", "local_files_only": False},
    {"device": "cpu", "dtype": "float32"},
    {"device": "cuda", "dtype": "bfloat16"},
])
def test_16gib_host_records_resource_errors_before_any_heavy_import(tmp_path, monkeypatch, settings):
    imported = forbid_heavy_imports(monkeypatch)
    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(
        16 * llm.GIB, "fake_16gib_host", 12 * llm.GIB,
    ))
    config = tmp_path / "llm.yaml"
    config.write_text(json.dumps({"llm": settings}))
    output = tmp_path / "run"
    assert run_llm.main(arguments(output, "--backend", "hf", "--config", str(config))) == 1
    manifest = read_manifest(output)
    assert manifest["execution"]["hf_weights_loaded"] is False
    assert manifest["execution"]["hf_generation_completed"] is False
    assert manifest["execution"]["generated_requests"] == 0
    assert manifest["execution"]["validated_requests"] == 0
    assert manifest["acceptance"]["status"] == "incomplete"
    assert manifest["acceptance"]["nov6_baselines_covered"] is False
    assert (output / "completions.jsonl").read_bytes() == b""
    provenance = manifest["backend_provenance"]
    assert provenance["weights_loaded"] is provenance["tokenizer_loaded"] is False
    assert provenance["kernel"] is None
    assert provenance["resources"]["host_admissible"] is False
    assert provenance["resources"]["host_memory_bytes"] == 16 * llm.GIB
    assert len(manifest["requests"]) == 2
    for request in manifest["requests"]:
        assert request["status"] == "error"
        assert request["error_type"] == "ResourceSafetyError"
        assert request["input_tokens"] is request["output_tokens"] is None
    for variant in ("baseline_2", "baseline_3"):
        record, = read_jsonl(output / f"{variant}.jsonl")
        assert record["status"] == "error" and record["backend"] == "hf"
        assert "RAM" in record["explanation"]
        assert record["predicted_failure_mode"] is record["predicted_subtask"] is None
    assert imported == []


@pytest.mark.parametrize("total,available", [(16, 12), (64, None), (None, None)])
def test_check_access_reports_blocked_or_unknown_resources_without_torch(
    tmp_path, monkeypatch, capsys, total, available,
):
    imported = forbid_heavy_imports(monkeypatch)
    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(
        total * llm.GIB if total is not None else None, "fake_os_snapshot",
        available * llm.GIB if available is not None else None,
    ))
    config = tmp_path / "llm.yaml"
    config.write_text(json.dumps({"llm": {"device": "mps", "dtype": "float16"}}))
    assert run_llm.main(["--check-access", "--config", str(config)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["resources"]["host_admissible"] is False
    assert result["resources"]["reason"]
    assert result["weights"] == "not_loaded" and result["remote_access"] == "not_checked"
    assert result["cuda_available"] is result["mps_available"] is None
    assert imported == []


def test_artifact_inventory_exactly_matches_published_files_and_hashes(tmp_path):
    output = tmp_path / "run"
    assert run_llm.main(arguments(output)) == 0
    manifest = read_manifest(output)
    artifacts = manifest["artifacts"]
    assert [record["path"] for record in artifacts] == [
        "baseline_2.jsonl", "baseline_3.jsonl", "completions.jsonl",
    ]
    assert {record["path"] for record in artifacts} | {manifest["manifest_file"]} == {
        path.name for path in output.iterdir()
    }
    assert [item["model_variant"] for item in artifacts if item["kind"] == "predictions"] == [
        "baseline_2", "baseline_3",
    ]
    for artifact in artifacts:
        path = output / artifact["path"]
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert artifact["size_bytes"] == path.stat().st_size
        assert artifact["n_records"] == len(read_jsonl(path))


def test_source_and_all_prompt_template_files_have_exact_hashes(tmp_path):
    output = tmp_path / "run"
    assert run_llm.main(arguments(output, "--variants", "baseline_2")) == 0
    manifest = read_manifest(output)
    assert {Path(item["path"]).name for item in manifest["source_files"]} == {
        "run_llm.py", "llm.py", "prompts.py", "schemas.py", "contracts.py",
    }
    assert {item["model_variant"] for item in manifest["prompt_template_files"]} == {
        "baseline_2", "baseline_3", "main",
    }
    for item in manifest["source_files"] + manifest["prompt_template_files"]:
        raw = Path(item["path"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(raw).hexdigest()
        assert item["size_bytes"] == len(raw)


def test_custom_templates_are_recorded_from_the_configured_directory(tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    for filename in ("failure_only.txt", "paired_no_divergence.txt", "paired_with_divergence.txt"):
        (templates / filename).write_text("Custom identical instructions.\n{{PAYLOAD}}\n")
    config = tmp_path / "llm.yaml"
    config.write_text("llm:\n  prompt_dir: templates\n")
    output = tmp_path / "run"
    assert run_llm.main(arguments(output, "--config", str(config))) == 0
    manifest = read_manifest(output)
    assert manifest["execution"]["all_requests_validated"] is True
    for item in manifest["prompt_template_files"]:
        assert Path(item["path"]).parent == templates.resolve()


def test_custom_template_hf_acceptance_rebuilds_the_configured_prompt(tmp_path, monkeypatch):
    templates = tmp_path / "templates"
    templates.mkdir()
    for filename in ("failure_only.txt", "paired_no_divergence.txt", "paired_with_divergence.txt"):
        (templates / filename).write_text("Custom identical instructions.\n{{PAYLOAD}}\n")
    config = tmp_path / "llm.yaml"
    config.write_text("llm:\n  prompt_dir: templates\n")
    monkeypatch.setattr(run_llm, "HFBackend", LocalHF)
    output = tmp_path / "run"
    assert run_llm.main(arguments(output, "--config", str(config), "--backend", "hf")) == 0
    manifest = read_manifest(output)
    assert manifest["execution"]["validated_requests"] == 2
    assert manifest["acceptance"]["status"] == "passed"


def test_malformed_completions_preserve_exact_text_and_parse_errors(tmp_path, monkeypatch):
    raw = '\n```json\n{"status": "ok"}\n```\nUnparsed μ text\n'
    monkeypatch.setattr(run_llm, "HFBackend", lambda config: LocalHF(config, malformed=raw))
    output = tmp_path / "run"
    assert run_llm.main(arguments(output, "--backend", "hf")) == 1
    completions = read_jsonl(output / "completions.jsonl")
    assert len(completions) == 2
    for record in completions:
        assert record["text"] == raw
        assert record["completion_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert record["status"] == "error"
        assert record["error_type"] and record["error_message"]
        assert record["backend"] == "hf"
        assert record["input_tokens"] == record["output_tokens"] == 32
    manifest = read_manifest(output)
    assert manifest["execution"]["hf_generation_completed"] is True
    assert manifest["execution"]["validated_requests"] == 0
    assert manifest["acceptance"]["status"] == "incomplete"
    assert manifest["acceptance"]["nov6_baselines_covered"] is False


def test_successful_hf_requires_generated_and_validated_baseline_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "HFBackend", LocalHF)
    output = tmp_path / "run"
    assert run_llm.main(arguments(output, "--backend", "hf")) == 0
    manifest = read_manifest(output)
    assert manifest["execution"] == {
        "selected_backend": "hf", "hf_weights_loaded": True, "hf_generation_completed": True,
        "expected_requests": 2, "generated_requests": 2, "validated_requests": 2,
        "all_requests_validated": True,
    }
    assert manifest["acceptance"]["status"] == "passed"
    assert manifest["acceptance"]["nov6_baselines_covered"] is True
    completions = read_jsonl(output / "completions.jsonl")
    for record, request in zip(completions, manifest["requests"]):
        assert record["status"] == "ok"
        assert record["completion_sha256"] == request["completion_sha256"]
        assert record["prompt_sha256"] == request["prompt_sha256"]


def test_successful_single_variant_does_not_complete_the_nov6_baseline_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(run_llm, "HFBackend", LocalHF)
    output = tmp_path / "run"
    assert run_llm.main(arguments(output, "--backend", "hf", "--variants", "baseline_2")) == 0
    manifest = read_manifest(output)
    assert manifest["acceptance"]["status"] == "passed"
    assert manifest["acceptance"]["nov6_baselines_covered"] is False


@pytest.mark.parametrize("failure", ["generation", "weights"])
def test_hf_selection_or_weight_loading_alone_does_not_pass_acceptance(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(run_llm, "HFBackend", lambda config: LocalHF(
        config, fail_generation=failure == "generation", weights_loaded=failure != "weights",
    ))
    output = tmp_path / "run"
    code = run_llm.main(arguments(output, "--backend", "hf"))
    assert code == (1 if failure == "generation" else 0)
    manifest = read_manifest(output)
    assert manifest["acceptance"]["status"] == "incomplete"
    assert manifest["acceptance"]["nov6_baselines_covered"] is False
    if failure == "generation":
        assert manifest["execution"]["hf_weights_loaded"] is True
        assert manifest["execution"]["hf_generation_completed"] is False
        assert (output / "completions.jsonl").read_bytes() == b""


def test_stub_with_valid_predictions_remains_synthetic(tmp_path):
    output = tmp_path / "run"
    assert run_llm.main(arguments(output)) == 0
    manifest = read_manifest(output)
    assert manifest["execution"]["all_requests_validated"] is True
    assert manifest["execution"]["hf_generation_completed"] is False
    assert manifest["acceptance"]["status"] == "synthetic"
    assert manifest["acceptance"]["nov6_baselines_covered"] is False


@pytest.mark.parametrize("tamper", [
    "non_json", "messages", "prompt_hash", "completion_hash", "prediction",
    "completion_status", "completion_backend", "different_valid_text", "missing_messages",
])
def test_acceptance_independently_rejects_inconsistent_saved_artifacts(tamper):
    tree = PairedBehaviorTree.model_validate_json((SAMPLE / "pair_000_bt.json").read_bytes())
    config = LLMConfig()
    result = run_inference([tree], ["baseline_2"], config, backend=LocalHF(config))
    predictions, completions = copy.deepcopy(result.predictions), copy.deepcopy(result.completions)
    record, completion = predictions[0], completions[0]
    if tamper == "non_json":
        completion["text"] = "not JSON"
        completion["completion_sha256"] = hashlib.sha256(completion["text"].encode()).hexdigest()
    elif tamper == "messages":
        completion["messages"][1]["content"] += "\nModified model input."
    elif tamper == "prompt_hash":
        completion["prompt_sha256"] = "f" * 64
    elif tamper == "completion_hash":
        completion["completion_sha256"] = "f" * 64
    elif tamper == "prediction":
        record["explanation"] = "The published explanation was changed after parsing."
    elif tamper == "completion_status":
        completion["status"] = "abstain"
    elif tamper == "completion_backend":
        completion["backend"] = "stub"
    elif tamper == "different_valid_text":
        raw = json.loads(completion["text"])
        raw["explanation"] = "A different valid model response from the published prediction."
        completion["text"] = json.dumps(raw)
        completion["completion_sha256"] = hashlib.sha256(completion["text"].encode()).hexdigest()
    else:
        del completion["messages"]
    summary = run_llm.execution_summary(
        selected_backend="hf", trees=[tree], variants=["baseline_2"],
        predictions=predictions, completions=completions, manifest=result.manifest,
    )
    assert summary["execution"]["validated_requests"] == 0
    assert summary["execution"]["all_requests_validated"] is False
    assert summary["acceptance"]["status"] == "incomplete"
    # Independent verification leaves the provided artifacts available exactly
    # as supplied; it must not repair or overwrite suspect model results.
    if tamper == "non_json":
        assert completion["text"] == "not JSON" and completion["status"] == "ok"


@pytest.mark.parametrize("side", ["predictions", "completions"])
def test_duplicate_artifact_coverage_cannot_pass_acceptance(side):
    tree = PairedBehaviorTree.model_validate_json((SAMPLE / "pair_000_bt.json").read_bytes())
    config = LLMConfig()
    result = run_inference([tree], ["baseline_2"], config, backend=LocalHF(config))
    predictions, completions = list(result.predictions), list(result.completions)
    selected = predictions if side == "predictions" else completions
    selected.append(copy.deepcopy(selected[0]))
    summary = run_llm.execution_summary(
        selected_backend="hf", trees=[tree], variants=["baseline_2"], predictions=predictions,
        completions=completions, manifest=result.manifest,
    )
    assert summary["acceptance"]["status"] == "incomplete"


def test_completion_hash_mismatch_prevents_any_publication(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="text/hash mismatch"):
        run_llm.publish_run(output, [], [], {}, completions=[{
            "text": "exact unaltered text", "completion_sha256": "incorrect",
        }])
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_completion_storage_failure_never_publishes_partial_artifacts(tmp_path, monkeypatch):
    original = Path.open

    def fail_completion(path, *args, **kwargs):
        if path.name == "completions.jsonl" and args and args[0] == "w":
            raise OSError("synthetic completion storage failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_completion)
    output = tmp_path / "run"
    assert run_llm.main(arguments(output)) == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".llm-*"))
