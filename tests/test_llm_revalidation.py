"""CPU-only replay of saved responses, never evidence of live model execution.

The HF-shaped fixtures explicitly use a fake adapter. They exercise provenance
and replay contracts; only independently collected model artifacts establish a
real-model result.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.abc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import revalidate_llm, run_llm
from telemetry_reasoning.models import llm
from telemetry_reasoning.models.revalidation import revalidate_bundle


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/sample"
VARIANTS = ("baseline_2", "baseline_3")
MODEL_RUNTIMES = {"torch", "transformers", "kernels", "huggingface_hub", "accelerate"}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_bytes(bundle):
    return {
        str(path.relative_to(bundle)): path.read_bytes()
        for path in bundle.rglob("*") if path.is_file()
    }


class FakeHF:
    """No tokenizers, kernels, credentials, network, or model weights are used."""

    name = "hf"

    def __init__(self, config, *, bad_evidence=False, weights_loaded=True):
        self.config = config
        self.bad_evidence = bad_evidence
        self.weights_loaded = weights_loaded

    def count_tokens(self, prompt):
        return 32

    def generate(self, prompt):
        node = next(node for node in prompt.payload["subtasks"] if node["failure_samples"])
        row = node["failure_samples"][0]
        response = {
            "predicted_failure_mode": "position_error",
            "predicted_subtask": node["subtask"],
            "explanation": "Fake response for offline replay contract tests only.",
            "evidence": [{
                "subtask": node["subtask"], "timestamp": row["timestamp"],
                "feature": "gripper_force",
                "value": row["gripper_force"] + (1000 if self.bad_evidence else 0),
            }],
            "status": "ok",
        }
        text = json.dumps(response, ensure_ascii=False)
        if prompt.variant.value == "baseline_2":
            text = " \r\n```json\r\n" + text + "\r\n```\r\n "
        return llm.Generation(text=text, input_tokens=32, output_tokens=32, finish_reason="eos")

    def provenance(self):
        return {"weights_loaded": self.weights_loaded, "implementation": "explicit_test_double"}


@pytest.fixture(autouse=True)
def forbid_model_runtimes(monkeypatch):
    """Replay must work even if optional runtimes are not installed/importable."""
    before = {name for name in sys.modules if name.split(".")[0] in MODEL_RUNTIMES}

    class BlockModelImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in MODEL_RUNTIMES:
                raise AssertionError(f"offline replay imported a model runtime: {fullname}")
            return None

    blocker = BlockModelImports()
    sys.meta_path.insert(0, blocker)

    def forbidden(*args, **kwargs):
        pytest.fail("offline replay tried to initialize or call an HF backend")

    for method in ("_runtime", "count_tokens", "generate", "generate_batch"):
        if hasattr(llm.HFBackend, method):
            monkeypatch.setattr(llm.HFBackend, method, forbidden)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
    after = {name for name in sys.modules if name.split(".")[0] in MODEL_RUNTIMES}
    assert after == before


@pytest.fixture
def make_bundle(tmp_path, monkeypatch):
    def make(*, backend="hf", bad_evidence=False, weights_loaded=True):
        bundle = tmp_path / f"bundle-{len(list(tmp_path.glob('bundle-*')))}"
        (bundle / "input").mkdir(parents=True)
        shutil.copyfile(SAMPLE / "pair_000_bt.json", bundle / "input/pair_000_bt.json")
        (bundle / "config.yaml").write_text("llm:\n  device: cpu\n  dtype: float32\n")
        monkeypatch.setattr(run_llm, "HFBackend", lambda config: FakeHF(
            config, bad_evidence=bad_evidence, weights_loaded=weights_loaded,
        ))
        code = run_llm.main([
            "--bt-dir", str(bundle / "input"),
            "--config", str(bundle / "config.yaml"),
            "--backend", backend,
            "--output-dir", str(bundle / "inference"),
        ])
        assert code == (0 if backend == "stub" else 1)
        return bundle

    return make


def refresh_artifact(bundle, filename, records):
    """Keep the outer file inventory valid when testing inner record checks."""
    path = bundle / "inference" / filename
    path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    manifest_path = bundle / "inference/run_manifest.json"
    manifest = read_json(manifest_path)
    artifact = next(item for item in manifest["artifacts"] if item["path"] == filename)
    artifact.update(sha256=digest(path.read_bytes()), size_bytes=path.stat().st_size,
                    n_records=len(records))
    write_json(manifest_path, manifest)


def test_replay_normalizes_only_declared_envelope_and_preserves_parent(make_bundle, tmp_path):
    bundle = make_bundle()
    original = source_bytes(bundle)
    parent = read_json(bundle / "inference/run_manifest.json")
    output = tmp_path / "derived"

    manifest = revalidate_bundle(bundle, output, response_policy="json_or_fenced_v1")

    assert source_bytes(bundle) == original
    assert manifest == read_json(output / "run_manifest.json")
    assert parent["acceptance"]["status"] == "incomplete"
    assert manifest["original_acceptance"] == parent["acceptance"]
    assert manifest["original_execution"] == parent["execution"]
    assert manifest["original_requests"] == parent["requests"]
    assert manifest["response_policy"] == "json_or_fenced_v1"
    assert manifest["revalidation"]["response_policy"] == "json_or_fenced_v1"
    assert manifest["revalidation"]["newly_generated_requests"] == 0
    assert manifest["revalidation"]["model_loaded_this_process"] is False
    assert manifest["revalidation"]["parent_manifest"]["sha256"] == digest(
        original["inference/run_manifest.json"]
    )
    assert manifest["acceptance"]["nov6_baselines_covered"] is True

    before = read_jsonl(bundle / "inference/completions.jsonl")
    after = read_jsonl(output / "completions.jsonl")
    assert len(before) == len(after) == 2
    for original_completion, completion in zip(before, after):
        for field in ("text", "completion_sha256", "messages", "prompt_sha256",
                      "input_tokens", "output_tokens", "finish_reason", "token_count_method"):
            assert completion[field] == original_completion[field]
        assert completion["completion_sha256"] == digest(completion["text"].encode())
        assert completion["original_validation"]["status"] == original_completion["status"]
        assert completion["response_normalization"]["policy"] == "json_or_fenced_v1"
        assert completion["status"] == "ok"
        assert "error_type" not in completion and "error_message" not in completion
    assert after[0]["original_validation"]["error_type"] == "JSONDecodeError"
    assert before[0]["error_message"] == after[0]["original_validation"]["error_message"]
    assert after[0]["response_normalization"]["envelope"] == "json_code_fence"
    assert after[1]["response_normalization"]["envelope"] == "none"
    assert after[0]["response_normalization"]["normalized_text_sha256"] == digest(
        before[0]["text"].split("\r\n")[2].encode()
    )
    assert after[1]["response_normalization"]["normalized_text_sha256"] == before[1]["completion_sha256"]
    for artifact in manifest["artifacts"]:
        path = output / artifact["path"]
        assert artifact["sha256"] == digest(path.read_bytes())
        assert artifact["size_bytes"] == path.stat().st_size
        assert artifact["n_records"] == len(read_jsonl(path))


def test_strict_policy_replay_does_not_accept_code_fence(make_bundle, tmp_path):
    bundle = make_bundle()
    original = source_bytes(bundle)
    output = tmp_path / "strict"
    manifest = revalidate_bundle(bundle, output, response_policy="strict_json_v1")
    assert manifest["acceptance"]["status"] == "incomplete"
    assert manifest["acceptance"]["nov6_baselines_covered"] is False
    assert [record["status"] for record in read_jsonl(output / "completions.jsonl")] == ["error", "ok"]
    assert source_bytes(bundle) == original


def test_legacy_run_without_policy_fields_still_records_original_strict_contract(make_bundle, tmp_path):
    bundle = make_bundle()
    manifest_path = bundle / "inference/run_manifest.json"
    manifest = read_json(manifest_path)
    manifest["config"].pop("response_policy", None)
    manifest.pop("response_policy", None)
    manifest.pop("response_processing", None)
    manifest["config_sha256"] = digest(json.dumps(
        manifest["config"], ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode())
    write_json(manifest_path, manifest)
    completions = read_jsonl(bundle / "inference/completions.jsonl")
    for completion in completions:
        completion.pop("response_normalization", None)
    refresh_artifact(bundle, "completions.jsonl", completions)
    original = source_bytes(bundle)

    derived = revalidate_bundle(bundle, tmp_path / "derived")

    assert derived["revalidation"]["original_response_policy"] == "strict_json_v1"
    assert derived["original_acceptance"]["status"] == "incomplete"
    assert derived["acceptance"]["nov6_baselines_covered"] is True
    assert source_bytes(bundle) == original


def test_stub_parent_remains_synthetic(make_bundle, tmp_path):
    bundle = make_bundle(backend="stub")
    manifest = revalidate_bundle(bundle, tmp_path / "derived")
    assert manifest["acceptance"]["nov6_baselines_covered"] is False
    assert manifest["acceptance"]["status"] == "synthetic"


def test_unloaded_hf_parent_is_refused_before_publication(make_bundle, tmp_path):
    bundle = make_bundle(weights_loaded=False)
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="weights"):
        revalidate_bundle(bundle, output)
    assert not output.exists()


def test_fence_handling_does_not_repair_invented_evidence(make_bundle, tmp_path):
    bundle = make_bundle(bad_evidence=True)
    original = source_bytes(bundle)
    output = tmp_path / "derived"
    manifest = revalidate_bundle(bundle, output)
    assert manifest["acceptance"]["status"] == "incomplete"
    assert manifest["acceptance"]["nov6_baselines_covered"] is False
    for record in read_jsonl(output / "completions.jsonl"):
        assert record["status"] == "error"
        assert "observed failure sample" in record["error_message"]
    for variant in VARIANTS:
        record, = read_jsonl(output / f"{variant}.jsonl")
        assert record["status"] == "error"
        assert record["predicted_failure_mode"] is record["predicted_subtask"] is None
        assert record["evidence"] == []
    assert source_bytes(bundle) == original


@pytest.mark.parametrize("damage", [
    "artifact_bytes", "artifact_size", "artifact_count", "config_bytes", "logical_config",
    "config_hash", "input_bytes", "prompt_source", "prompt_template", "raw_hash",
    "messages", "prompt_hash", "published_prediction", "unsafe_artifact_path", "extra_artifact",
])
def test_damaged_provenance_is_rejected_before_publication(make_bundle, tmp_path, damage):
    bundle = make_bundle()
    manifest_path = bundle / "inference/run_manifest.json"
    manifest = read_json(manifest_path)
    completions = read_jsonl(bundle / "inference/completions.jsonl")
    if damage == "artifact_bytes":
        with (bundle / "inference/baseline_2.jsonl").open("a") as stream:
            stream.write("\n")
    elif damage in ("artifact_size", "artifact_count"):
        field = "size_bytes" if damage == "artifact_size" else "n_records"
        manifest["artifacts"][0][field] += 1
        write_json(manifest_path, manifest)
    elif damage in ("config_bytes", "logical_config"):
        config = bundle / "config.yaml"
        config.write_text(config.read_text() + (
            "# harmless edit still changes provenance\n" if damage == "config_bytes" else "  seed: 123\n"
        ))
        if damage == "logical_config":
            manifest["config_file_sha256"] = digest(config.read_bytes())
            write_json(manifest_path, manifest)
    elif damage == "config_hash":
        manifest["config_sha256"] = "0" * 64
        write_json(manifest_path, manifest)
    elif damage == "input_bytes":
        path = bundle / "input/pair_000_bt.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage in ("prompt_source", "prompt_template"):
        if damage == "prompt_source":
            item = next(item for item in manifest["source_files"] if Path(item["path"]).name == "prompts.py")
        else:
            item = manifest["prompt_template_files"][0]
        item["sha256"] = "0" * 64
        write_json(manifest_path, manifest)
    elif damage in ("raw_hash", "messages", "prompt_hash"):
        if damage == "raw_hash":
            completions[0]["text"] += "\n"
        elif damage == "messages":
            completions[0]["messages"][1]["content"] += "\nChanged model input."
        else:
            completions[0]["prompt_sha256"] = "0" * 64
        refresh_artifact(bundle, "completions.jsonl", completions)
    elif damage == "published_prediction":
        records = read_jsonl(bundle / "inference/baseline_3.jsonl")
        records[0]["explanation"] = "Changed independently of the raw response."
        refresh_artifact(bundle, "baseline_3.jsonl", records)
    elif damage == "unsafe_artifact_path":
        manifest["artifacts"][0]["path"] = "../config.yaml"
        write_json(manifest_path, manifest)
    else:
        manifest["artifacts"].append(copy.deepcopy(manifest["artifacts"][0]))
        write_json(manifest_path, manifest)
    original = source_bytes(bundle)
    output = tmp_path / "refused"
    with pytest.raises((ValueError, OSError)):
        revalidate_bundle(bundle, output)
    assert not output.exists()
    assert source_bytes(bundle) == original


@pytest.mark.parametrize("collection", ["requests", "predictions", "completions"])
@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_pair_id"])
def test_missing_duplicate_or_unknown_request_identity_is_rejected(
    make_bundle, tmp_path, collection, damage,
):
    bundle = make_bundle()
    manifest_path = bundle / "inference/run_manifest.json"
    manifest = read_json(manifest_path)
    filename = "completions.jsonl" if collection == "completions" else "baseline_2.jsonl"
    records = manifest["requests"] if collection == "requests" else read_jsonl(bundle / "inference" / filename)
    if damage == "missing":
        records.pop(0)
    elif damage == "duplicate":
        records.append(copy.deepcopy(records[0]))
    else:
        records[0]["pair_id"] = "unknown_pair"
    if collection == "requests":
        write_json(manifest_path, manifest)
    else:
        refresh_artifact(bundle, filename, records)
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        revalidate_bundle(bundle, output)
    assert not output.exists()


def test_cli_reports_validation_errors_without_replacing_existing_results(make_bundle, tmp_path):
    bundle = make_bundle()
    output = tmp_path / "derived"
    args = ["--bundle-dir", str(bundle), "--output-dir", str(output)]
    assert revalidate_llm.main(args) == 0
    published = source_bytes(output)
    assert revalidate_llm.main(args) == 2
    assert source_bytes(output) == published
    strict = tmp_path / "strict"
    assert revalidate_llm.main([
        "--bundle-dir", str(bundle), "--output-dir", str(strict),
        "--response-policy", "strict_json_v1",
    ]) == 1
    assert read_json(strict / "run_manifest.json")["acceptance"]["status"] == "incomplete"


@pytest.mark.parametrize("relative", [".", "inference", "inference/derived", "derived"])
def test_output_cannot_overlap_original_bundle(make_bundle, relative):
    bundle = make_bundle()
    original = source_bytes(bundle)
    with pytest.raises(ValueError):
        revalidate_bundle(bundle, bundle / relative)
    assert source_bytes(bundle) == original


def test_invalid_policy_refuses_publication(make_bundle, tmp_path):
    bundle = make_bundle()
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        revalidate_bundle(bundle, output, response_policy="guess_and_repair")
    assert not output.exists()


def test_empty_output_directory_is_supported(make_bundle, tmp_path):
    bundle = make_bundle()
    output = tmp_path / "empty-output"
    output.mkdir()
    revalidate_bundle(bundle, output)
    assert (output / "run_manifest.json").is_file()


@pytest.mark.parametrize("damage", [
    "metadata_error_fields", "input_tokens", "output_tokens", "token_count_method",
    "finish_reason", "request_tokens", "request_finish", "boolean_tokens",
])
def test_invalid_generation_metadata_cannot_be_promoted(make_bundle, tmp_path, damage):
    bundle = make_bundle()
    manifest_path = bundle / "inference/run_manifest.json"
    manifest = read_json(manifest_path)
    completions = read_jsonl(bundle / "inference/completions.jsonl")
    request, completion = manifest["requests"][1], completions[1]
    if damage == "metadata_error_fields":
        completion["metadata_error_fields"] = ["output_tokens"]
    elif damage == "input_tokens":
        request["input_tokens"] = completion["input_tokens"] = 0
    elif damage == "output_tokens":
        request["output_tokens"] = completion["output_tokens"] = 0
    elif damage == "token_count_method":
        request["token_count_method"] = completion["token_count_method"] = "estimated"
    elif damage == "finish_reason":
        request["finish_reason"] = completion["finish_reason"] = "length"
    elif damage == "request_tokens":
        request["output_tokens"] += 1
    elif damage == "request_finish":
        request["finish_reason"] = "length"
    else:
        request["output_tokens"] = completion["output_tokens"] = True
    write_json(manifest_path, manifest)
    refresh_artifact(bundle, "completions.jsonl", completions)
    original = source_bytes(bundle)
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        revalidate_bundle(bundle, output)
    assert not output.exists()
    assert source_bytes(bundle) == original


@pytest.mark.parametrize("damage", ["missing", "index", "status", "timing", "request_timing"])
def test_inconsistent_generation_batch_provenance_is_rejected(make_bundle, tmp_path, damage):
    bundle = make_bundle()
    manifest_path = bundle / "inference/run_manifest.json"
    manifest = read_json(manifest_path)
    if damage == "missing":
        del manifest["batches"]
    elif damage == "index":
        manifest["batches"][0]["request_indices"] = [100]
    elif damage == "status":
        manifest["batches"][0]["status_counts"] = {"ok": 1}
    elif damage == "timing":
        manifest["batches"][0]["latency_s"] = -1
    else:
        manifest["requests"][0]["batch_latency_s"] += 1
    write_json(manifest_path, manifest)
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        revalidate_bundle(bundle, output)
    assert not output.exists()


def test_symlinked_original_artifact_is_rejected(make_bundle, tmp_path):
    bundle = make_bundle()
    artifact = bundle / "inference/baseline_2.jsonl"
    relocated = tmp_path / "relocated-prediction.jsonl"
    artifact.rename(relocated)
    artifact.symlink_to(relocated)
    original = source_bytes(bundle)
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        revalidate_bundle(bundle, output)
    assert artifact.is_symlink()
    assert source_bytes(bundle) == original
    assert not output.exists()


@pytest.mark.parametrize("field", ["error_type", "error_message"])
def test_original_error_must_reproduce_before_policy_change(make_bundle, tmp_path, field):
    bundle = make_bundle()
    manifest_path = bundle / "inference/run_manifest.json"
    manifest = read_json(manifest_path)
    completions = read_jsonl(bundle / "inference/completions.jsonl")
    incorrect = "ResourceSafetyError" if field == "error_type" else "An unrelated failure."
    manifest["requests"][0][field] = incorrect
    completions[0][field] = incorrect
    write_json(manifest_path, manifest)
    refresh_artifact(bundle, "completions.jsonl", completions)
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        revalidate_bundle(bundle, output)
    assert not output.exists()


def test_cli_from_unrelated_directory_does_not_import_model_runtimes(make_bundle, tmp_path):
    bundle = make_bundle()
    guard = tmp_path / "import-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import importlib.abc, sys\n"
        "class BlockModelImports(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        f"        if fullname.split('.')[0] in {MODEL_RUNTIMES!r}:\n"
        "            raise AssertionError('model runtime import: ' + fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, BlockModelImports())\n"
    )
    outside = tmp_path / "unrelated-working-directory"
    outside.mkdir()
    output = tmp_path / "derived"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(guard), str(ROOT / "src"))),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/revalidate_llm.py"),
         "--bundle-dir", str(bundle), "--output-dir", str(output)],
        cwd=outside, env=env, text=True, capture_output=True, timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["execution"]["newly_generated_requests"] == 0
    assert read_json(output / "run_manifest.json")["revalidation"]["model_loaded_this_process"] is False

