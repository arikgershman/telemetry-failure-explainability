"""CPU/offline GPU-launch contracts; fake runtimes never load model weights."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from telemetry_reasoning.models import training
from telemetry_reasoning.models.training_data import prepare_training_data
from telemetry_reasoning.schemas import Task
from test_training_data import case  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training-gpu.yaml"


@pytest.fixture
def reviewed(case):
    # These reviewer assertions exist only inside a temporary unit fixture.
    # They are never published as actual reviewed project annotations.
    for record in case["records"]:
        provenance = record["preference_provenance"]
        provenance.pop("generator_id")
        provenance.update(kind="human_reviewed", reviewer_id="unit-fixture-reviewer",
                          reviewed_at="2026-11-18T12:00:00Z")
    splits = case["manifest"]["splits"]
    transport = [row for row in splits["splits"]["train"] if row["task"] == Task.TRANSPORT_OBJECT.value]
    splits["splits"]["dev"] = transport
    splits["splits"]["train"] = [row for row in splits["splits"]["train"] if row not in transport]
    splits["condition_assignments"][Task.TRANSPORT_OBJECT.value] = {"train": [], "dev": ["7"], "test": []}
    case["write"]()
    prepare_training_data(case["root"], case["prefs"], case["manifest_path"], case["output"])
    return case["output"]


def rewrite_artifact(root, name, mutation):
    path = root / name
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutation(rows)
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(raw)
    manifest = root / "training_manifest.json"
    receipt = json.loads(manifest.read_text())
    receipt["artifacts"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw), "rows": len(rows)}
    manifest.write_text(json.dumps(receipt))


def test_check_is_offline_without_heavy_imports(reviewed, tmp_path, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "trl", "peft", "datasets", "accelerate"}:
            raise AssertionError("heavy runtime imported by CPU check")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "adapter"
    report, rows = training.check_training(reviewed, CONFIG, "main", output)
    assert report["status"] == "inputs_validated_gpu_execution_pending"
    assert report["rows"] == {"train": 1, "dev": 1}
    assert all(len(items) == 1 for items in rows.values())
    assert report["model_id"] == training.MODEL_ID
    assert report["revision"] == training.MODEL_REVISION
    assert report["dataset_release_status"] == "parser_ready_certification_failed"
    assert not output.exists()


def test_synthetic_publication_never_trains(case, tmp_path, monkeypatch):
    prepare_training_data(case["root"], case["prefs"], case["manifest_path"], case["output"], synthetic_demo=True)
    monkeypatch.setattr(training, "_gpu_runtime", lambda config: pytest.fail("GPU preflight reached synthetic data"))
    with pytest.raises(ValueError, match="synthetic_demo_only"):
        training.run_training(case["output"], CONFIG, "main", tmp_path / "adapter")
    assert not (tmp_path / "adapter").exists()


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.pop(), "coverage"),
    (lambda rows: rows.append(rows[0]), "uniquely"),
    (lambda rows: rows[0].update(split="test"), "sidecar"),
    (lambda rows: rows[0].update(initial_condition_id=True), "condition"),
    (lambda rows: rows[0]["preference_provenance"].update(kind="synthetic_demo", generator_id="fake"), "synthetic_demo"),
    (lambda rows: rows[0].update(prompt_sha256="0" * 64), "hashes"),
])
def test_sidecar_corruption_is_refused_even_with_rehashed_inventory(reviewed, tmp_path, mutation, match):
    rewrite_artifact(reviewed, "records.jsonl", mutation)
    with pytest.raises(ValueError, match=match):
        training.check_training(reviewed, CONFIG, "main", tmp_path / "adapter")


def test_unreviewed_variant_and_existing_output_refuse(reviewed, tmp_path):
    manifest = reviewed / "training_manifest.json"
    receipt = json.loads(manifest.read_text())
    receipt["training_acceptance"] = "synthetic_demo_only"
    manifest.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="mode"):
        training.check_training(reviewed, CONFIG, "main", tmp_path / "adapter")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="absent"):
        training.check_training(reviewed, CONFIG, "main", output)


@pytest.mark.parametrize("field,value", [("device", "cpu"), ("local_files_only", False),
                                        ("revision", "main"), ("model_id", "other"),
                                        ("max_length", 100000), ("learning_rate", float("nan"))])
def test_config_rejects_unsafe_or_unpinned_settings(field, value):
    with pytest.raises(ValueError):
        training.TrainingConfig.model_validate({field: value})


def test_host_refusal_precedes_training_runtime_import(monkeypatch):
    monkeypatch.setattr(training, "_host_resources", lambda: SimpleNamespace(host_memory_bytes=16 * training.GIB,
                                                                            available_host_memory_bytes=8 * training.GIB))
    monkeypatch.setattr(training.importlib, "import_module", lambda name: pytest.fail("runtime imported"))
    with pytest.raises(ValueError, match="host RAM"):
        training._gpu_runtime(training.TrainingConfig())


def test_cuda_profile_checks_versions_and_available_memory(monkeypatch):
    monkeypatch.setattr(training, "_host_resources", lambda: SimpleNamespace(host_memory_bytes=128 * training.GIB,
                                                                            available_host_memory_bytes=96 * training.GIB))
    monkeypatch.setattr(training.importlib.metadata, "version", lambda name: training.RUNTIME_PINS[name])
    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                           current_device=lambda: 0, mem_get_info=lambda device: (64 * training.GIB, 80 * training.GIB),
                           get_device_name=lambda device: "unit-test fake CUDA")
    monkeypatch.setattr(training.importlib, "import_module", lambda name: SimpleNamespace(cuda=cuda))
    _, receipt = training._gpu_runtime(training.TrainingConfig())
    assert receipt["vram_free_bytes"] == 64 * training.GIB
    cuda.mem_get_info = lambda device: (16 * training.GIB, 80 * training.GIB)
    with pytest.raises(ValueError, match="VRAM"):
        training._gpu_runtime(training.TrainingConfig())
    cuda.is_available = lambda: False
    with pytest.raises(ValueError, match="CUDA"):
        training._gpu_runtime(training.TrainingConfig())


class FakeTokenizer:
    eos_token_id = 2
    eos_token = "EOS"
    chat_template = "unit fixture, not an actual tokenizer"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        text = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(range(max(1, len(text) // 32)))}

    def save_pretrained(self, destination):
        destination = Path(destination)
        destination.mkdir()
        (destination / "tokenizer.json").write_text("fixture")


def fake_modules(monkeypatch, *, wrong_tokens=False, failure=False):
    calls = []
    tokenizer = FakeTokenizer()
    class FakeTrainer:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.datasets = {}
            for split, field in (("train", "train_dataset"), ("dev", "eval_dataset")):
                self.datasets[split] = []
                for row in kwargs[field]:
                    ids = {f"{key}_input_ids": tokenizer(value, add_special_tokens=False)["input_ids"]
                           for key, value in row.items()}
                    ids["chosen_input_ids"].append(2)
                    ids["rejected_input_ids"].append(2)
                    if wrong_tokens:
                        ids["prompt_input_ids"] = ids["prompt_input_ids"][-1:]
                    self.datasets[split].append(ids)
            self.train_dataset, self.eval_dataset = self.datasets["train"], self.datasets["dev"]
        def train(self):
            if failure:
                raise RuntimeError("fixture optimizer failure")
            return SimpleNamespace(metrics={"train_loss": 0.1})
        def evaluate(self):
            return {"eval_loss": 0.2}
        def save_model(self, destination):
            path = Path(destination)
            path.mkdir()
            (path / "adapter_config.json").write_text("{}")
            (path / "adapter_model.safetensors").write_bytes(b"unit fixture")
        def save_state(self):
            pass
    def load_tokenizer(*args, **kwargs):
        assert kwargs == {"revision": training.MODEL_REVISION, "local_files_only": True, "trust_remote_code": False}
        return tokenizer
    def load_model(*args, **kwargs):
        assert kwargs["revision"] == training.MODEL_REVISION and kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False and kwargs["use_safetensors"] is True
        return SimpleNamespace(config=SimpleNamespace(use_cache=True))
    modules = {
        "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
                                         AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
                                         set_seed=lambda seed: None),
        "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
        "peft": SimpleNamespace(LoraConfig=lambda **kwargs: kwargs),
        "trl": SimpleNamespace(DPOConfig=lambda **kwargs: kwargs, DPOTrainer=FakeTrainer),
    }
    monkeypatch.setattr(training, "_gpu_runtime", lambda config: (SimpleNamespace(bfloat16="bf16"), {"cuda_device": 0}))
    monkeypatch.setattr(training.importlib, "import_module", lambda name: modules[name])
    return calls


def test_fake_gpu_execution_saves_adapter_and_complete_receipt(reviewed, tmp_path, monkeypatch):
    calls = fake_modules(monkeypatch)
    output = tmp_path / "adapter"
    report = training.run_training(reviewed, CONFIG, "main", output)
    assert report["status"] == "completed"
    assert report["optimizer_acceptance"] == "executed_on_reviewed_data"
    assert report["adapter_quality"] == "not_evaluated"
    assert report["inputs_unchanged"]
    assert calls[0]["ref_model"] is None
    assert calls[0]["args"]["max_prompt_length"] is None
    assert calls[0]["args"]["max_completion_length"] is None
    assert calls[0]["args"]["push_to_hub"] is False
    assert calls[0]["args"]["report_to"] == []
    assert set(report["artifacts"]) == {"adapter/adapter_config.json", "adapter/adapter_model.safetensors",
                                         "tokenizer/tokenizer.json", "training_config.yaml", "training_data_manifest.json"}
    assert json.loads((output / "training_run_manifest.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("mode", ["tokens", "optimizer"])
def test_fake_runtime_failure_never_claims_accepted_optimizer(reviewed, tmp_path, monkeypatch, mode):
    fake_modules(monkeypatch, wrong_tokens=mode == "tokens", failure=mode == "optimizer")
    output = tmp_path / "adapter"
    with pytest.raises((ValueError, RuntimeError)):
        training.run_training(reviewed, CONFIG, "main", output)
    receipt = json.loads((output / "training_run_manifest.json").read_text())
    assert receipt["status"] == "failed" and receipt["optimizer_acceptance"] == "not_accepted"
    assert not (output / "adapter").exists()


def test_exact_token_budget_refuses_before_output_or_weight_loading(reviewed, tmp_path, monkeypatch):
    fake_modules(monkeypatch)
    monkeypatch.setattr(FakeTokenizer, "__call__", lambda self, text, **kwargs: {"input_ids": list(range(9000))})
    with pytest.raises(ValueError, match="no truncation"):
        training.run_training(reviewed, CONFIG, "main", tmp_path / "adapter")
    assert not (tmp_path / "adapter").exists()


def test_optional_cpu_tokenizer_preflight_is_exact_and_loads_no_weights(reviewed, tmp_path, monkeypatch):
    from telemetry_reasoning import experiments

    monkeypatch.setattr(experiments, "load_cpu_tokenizer", lambda directory: (
        FakeTokenizer(), {"model_id": training.MODEL_ID, "revision": training.MODEL_REVISION}))
    report, _ = training.check_training(reviewed, CONFIG, "main", tmp_path / "adapter",
                                        tokenizer_dir=tmp_path / "tokenizer")
    assert report["tokenizer_preflight"]["weights_loaded"] is False
    assert report["tokenizer_preflight"]["device_initialized"] is False
    assert len(report["tokenizer_preflight"]["token_lengths"]) == 2
    assert "exact tokenizer context budgets" not in report["checks_pending"]
    monkeypatch.setattr(FakeTokenizer, "__call__", lambda self, text, **kwargs: {"input_ids": list(range(9000))})
    with pytest.raises(ValueError, match="no truncation"):
        training.check_training(reviewed, CONFIG, "main", tmp_path / "adapter", tokenizer_dir=tmp_path / "tokenizer")


def test_cli_check_is_default_and_refusal_is_clean(reviewed, tmp_path):
    command = [sys.executable, str(ROOT / "scripts/train_adapter.py"), "--data-dir", str(reviewed),
               "--variant", "main", "--output-dir", str(tmp_path / "adapter")]
    checked = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["status"] == "inputs_validated_gpu_execution_pending"
    assert not (tmp_path / "adapter").exists()
    (tmp_path / "adapter").mkdir()
    refused = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert refused.returncode == 2 and "Traceback" not in refused.stderr
