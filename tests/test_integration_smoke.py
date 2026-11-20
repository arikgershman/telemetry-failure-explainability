"""Public CLI acceptance, offline execution, and failure-reporting regressions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import integration_smoke
from telemetry_reasoning import integration


ROOT = Path(__file__).resolve().parents[1]


def read_receipt(output: Path) -> dict:
    return json.loads((output / integration.MANIFEST_NAME).read_text())


def test_public_cli_runs_every_handoff_offline_without_optional_runtimes(tmp_path):
    guard = tmp_path / "guard"
    guard.mkdir()
    guard_log = tmp_path / "guard.log"
    # sitecustomize also runs in every child CLI. An accidental real backend or
    # download therefore fails this end-to-end test rather than using local assets.
    (guard / "sitecustomize.py").write_text('''
import importlib.abc
import os
import socket
import sys
with open(os.environ["SMOKE_GUARD_LOG"], "a") as marker:
    marker.write(str(os.getpid()) + "\\n")
class NoOptionalRuntimes(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "transformers", "mujoco", "huggingface_hub", "accelerate", "peft", "trl", "bitsandbytes"}:
            raise RuntimeError("CPU smoke imported optional runtime: " + fullname)
sys.meta_path.insert(0, NoOptionalRuntimes())
def no_network(*args, **kwargs):
    raise RuntimeError("CPU smoke attempted network access")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network
''', encoding="utf-8")
    environment = dict(os.environ)
    environment.update({"SMOKE_GUARD_LOG": str(guard_log),
                        "PYTHONPATH": os.pathsep.join([str(guard), str(ROOT / "src")]),
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    # A similarly named unrelated prediction must never enter this run.
    unrelated = tmp_path / "baseline_1.jsonl"
    unrelated.write_text('{"unrelated": true}\n')
    output = tmp_path / "run"
    output.mkdir()  # An existing empty destination is supported.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/integration_smoke.py"), "--backend", "stub",
         "--seed", "19", "--output-dir", str(output)], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SYNTHETIC" in result.stdout
    receipt = read_receipt(output)
    assert receipt["status"] == "passed" and receipt["synthetic"] is True
    assert receipt["scope"] == "cpu_contract_smoke"
    assert receipt["seed"] == 19
    assert [(stage["name"], stage["status"], stage["returncode"]) for stage in receipt["stages"]] == [
        (name, "passed", 0) for name in (
            "generate_mock", "parse", "divergence", "rules", "llm_stub", "evaluate", "plot_telemetry")
    ]
    # Parent plus every public CLI demonstrably loaded the guard.
    assert len(set(guard_log.read_text().splitlines())) == len(receipt["stages"]) + 1
    checks = receipt["checks"]
    assert checks["raw_row_counts"] == {"success": 400, "failure": 400}
    assert checks["first_grasp_divergence_s"] == pytest.approx(5.10)
    assert checks["max_grip_delta_n"] == pytest.approx(12.0)
    assert checks["retained_event_timestamps_s"] == [5.08, 5.10, 5.12]
    assert set(checks["views"]) == {"baseline_2", "baseline_3", "main"}
    assert checks["prediction_count"] == 4
    assert checks["evidence_verified"] and checks["metrics_finite"]
    assert checks["source_files_unchanged"] and checks["versioned_samples_unchanged"]
    assert checks["generated_inputs_unchanged"] and checks["bt_inputs_unchanged"]
    assert len(checks["plot_files"]) == 10
    metrics = json.loads((output / "evaluation/metrics.json").read_text())
    raw_bt = json.loads((output / "raw-bt/pair_000_bt.json").read_text())
    assert raw_bt["metadata"]["seed"] == 19
    assert metrics["predictions_outside_scope"] == 0
    assert metrics["variants"]["baseline_1"]["evidence"]["n_supported"] > 0
    assert all(metrics["variants"][variant]["synthetic"] for variant in integration.VARIANTS[1:])
    rubric_key = json.loads((output / "evaluation/rubric_key.json").read_text())
    assert "stub" not in json.dumps(rubric_key)
    for relative, fingerprint in receipt["artifacts"].items():
        path = output / relative
        assert path.stat().st_size == fingerprint["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fingerprint["sha256"]
    for relative, fingerprint in receipt["source_files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == fingerprint["sha256"]
    assert integration.MANIFEST_NAME not in receipt["artifacts"]
    assert "runtime" not in {Path(path).parts[0] for path in receipt["artifacts"]}
    for stage in receipt["stages"]:
        assert (output / stage["stdout"]).is_file()
        assert (output / stage["stderr"]).is_file()
    assert unrelated.read_text() == '{"unrelated": true}\n'


@pytest.mark.parametrize("kind", ["nonempty", "file", "symlink"])
def test_refuses_existing_outputs_without_modifying_them(tmp_path, kind):
    output = tmp_path / "run"
    if kind == "file":
        output.write_text("preserve")
        preserved = output
    else:
        target = tmp_path / "target" if kind == "symlink" else output
        target.mkdir()
        preserved = target / "prior-result.json"
        preserved.write_text("preserve")
        if kind == "symlink":
            output.symlink_to(target, target_is_directory=True)
    assert integration_smoke.main(["--output-dir", str(output)]) == 2
    assert preserved.read_text() == "preserve"
    assert not (output / integration.MANIFEST_NAME).exists()


@pytest.mark.parametrize("arguments", [
    ["--seed", "-1"], ["--stage-timeout-s", "0"], ["--stage-timeout-s", "nan"],
    ["--stage-timeout-s", "inf"],
])
def test_invalid_parameters_create_no_run(tmp_path, arguments):
    output = tmp_path / "run"
    assert integration_smoke.main(["--output-dir", str(output), *arguments]) == 2
    assert not output.exists()


@pytest.mark.parametrize("arguments", [["--backend", "hf"], ["--sim-backend", "unknown"]])
def test_cpu_gate_cannot_accidentally_select_real_backends(tmp_path, arguments):
    output = tmp_path / "run"
    with pytest.raises(SystemExit) as exc:
        integration_smoke.main(["--output-dir", str(output), *arguments])
    assert exc.value.code == 2 and not output.exists()


def test_failed_stage_keeps_logs_and_failed_receipt(tmp_path, monkeypatch):
    def fail(command, **kwargs):
        kwargs["stderr"].write("injected stage failure\n")
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(integration.subprocess, "run", fail)
    output = tmp_path / "run"
    assert integration_smoke.main(["--output-dir", str(output)]) == 1
    receipt = read_receipt(output)
    assert receipt["status"] == "failed" and "exited 7" in receipt["error"]["message"]
    stage, = receipt["stages"]
    assert stage["status"] == "failed" and stage["returncode"] == 7
    assert (output / stage["stderr"]).read_text() == "injected stage failure\n"
    assert integration_smoke.main(["--output-dir", str(output)]) == 2
    assert read_receipt(output) == receipt


def test_stage_timeout_is_recorded_as_failure(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(integration.subprocess, "run", timeout)
    output = tmp_path / "run"
    assert integration_smoke.main(["--output-dir", str(output), "--stage-timeout-s", "0.01"]) == 1
    receipt = read_receipt(output)
    assert receipt["status"] == "failed" and receipt["error"]["type"] == "TimeoutExpired"
    assert receipt["stages"][0]["status"] == "failed"


def test_successful_exit_without_artifacts_cannot_pass_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(integration.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
    output = tmp_path / "run"
    with pytest.raises(integration.SmokeFailure, match="metadata.csv"):
        integration.run_cpu_smoke(output)
    receipt = read_receipt(output)
    assert receipt["status"] == "failed"
    assert receipt["stages"][0]["status"] == "passed"
    assert receipt["error"]["type"] == "FileNotFoundError"


def test_view_gate_rejects_future_label_leakage():
    tree = integration.PairedBehaviorTree.model_validate_json((ROOT / "data/sample/pair_000_bt.json").read_bytes())
    from telemetry_reasoning.parser.divergence_metrics import enrich_paired_bt, load_divergence_config

    tree = enrich_paired_bt(tree, load_divergence_config(ROOT / "configs/divergence.yaml"))
    original = tree.to_failure_only
    # Simulate an accidental future expansion of the shared view helper.
    from unittest.mock import patch

    with patch.object(type(tree), "to_failure_only", lambda self: {**original(), "failure_mode": "position_error"}):
        with pytest.raises(integration.SmokeFailure, match="leaks"):
            integration._validate_views(tree)


def test_output_destination_has_only_one_owner_under_concurrent_launch(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)
    output = tmp_path / "run"
    def reserve():
        barrier.wait()
        try:
            return integration._prepare_output(output)
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reserve(), range(2)))
    assert results.count(output.resolve()) == 1
    assert results.count(None) == 1
    assert (output / ".integration_claim").read_text().startswith("pid=")


@pytest.mark.parametrize("problem", ["stale_pair", "invented_telemetry"])
def test_tree_gate_rejects_valid_bts_that_do_not_come_from_fresh_csvs(tmp_path, problem):
    from scripts.generate_mock_pair import generate_mock_pair
    from telemetry_reasoning.config import load_config
    from telemetry_reasoning.parser.divergence_metrics import enrich_paired_bt, load_divergence_config

    generated = generate_mock_pair(tmp_path / "input", load_config(ROOT / "configs/default.yaml"), seed=19)
    if problem == "stale_pair":
        raw = integration.PairedBehaviorTree.model_validate_json((ROOT / "data/sample/pair_000_bt.json").read_bytes())
        message = "metadata"
    else:
        raw = generated.model_copy(deep=True)
        # Both BT stages agree with each other, but disagree with the real CSV.
        raw.subtasks[0].failure_samples[0].gripper_force += 0.25
        message = "freshly generated CSV"
    enriched = enrich_paired_bt(raw, load_divergence_config(ROOT / "configs/divergence.yaml"))
    for folder, tree in (("raw-bt", raw), ("bt", enriched)):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "pair_000_bt.json").write_text(tree.model_dump_json())
    with pytest.raises(integration.SmokeFailure, match=message):
        integration._validate_trees(tmp_path)
