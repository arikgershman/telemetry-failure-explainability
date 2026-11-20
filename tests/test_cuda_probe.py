"""The hardware diagnostic is bounded, import-safe, and preserves failed runs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cuda_probe_tests", ROOT / "scripts/probe_cuda_inference.py")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _torch(*, available=True, free=8 * probe.GIB, bf16=True):
    cuda = SimpleNamespace(
        is_available=lambda: available,
        current_device=lambda: 0,
        get_device_name=lambda device: "synthetic CUDA probe fixture",
        get_device_capability=lambda device: (12, 0),
        is_bf16_supported=lambda: bf16,
        mem_get_info=lambda: (free, 32 * probe.GIB),
        empty_cache=lambda: None,
    )
    return SimpleNamespace(cuda=cuda, __version__="fixture", version=SimpleNamespace(cuda="fixture"))


def test_plan_covers_hf_first_prefix_mask_forms_and_decode_under_two_gib():
    rows = [case.description() for case in probe.probe_cases()]
    assert [row["name"] for row in rows] == [
        "first_prefill_gqa", "prefix_prefill_bool", "prefix_prefill_bfloat16", "single_token_decode_gqa",
    ]
    assert all(row["planned_tensor_upper_bound_bytes"] < 2 * probe.GIB for row in rows)
    assert rows[0]["key_value_shape"] == [1, 8, 2048, 128]
    assert rows[0]["is_causal"] and rows[0]["enable_gqa"]
    assert rows[1]["key_value_shape"] == [1, 32, 65536, 128]
    assert rows[1]["mask_shape"] == [1, 1, 2048, 65536]
    assert rows[1]["mask_dtype"] == "bool" and rows[2]["mask_dtype"] == "bfloat16"
    assert rows[1]["enable_gqa"] is False and rows[1]["is_causal"] is False
    assert rows[3]["query_shape"] == [1, 32, 1, 128]
    assert rows[3]["enable_gqa"] and rows[3]["is_causal"] is False


@pytest.mark.parametrize("chunk,prefix", [(0, 65536), (2049, 65536), (True, 65536), (2048, 2047), (2048, 65537)])
def test_invalid_or_unbounded_shapes_are_rejected(chunk, prefix):
    with pytest.raises(ValueError):
        probe.probe_cases(chunk, prefix)


def test_no_cuda_refuses_without_allocation(monkeypatch):
    monkeypatch.setattr(probe, "_measure_case", lambda *args: pytest.fail("must not allocate"))
    report = probe.run_probe(_torch(available=False))
    assert report["status"] == "refused"
    assert report["error_type"] == "CUDAUnavailable"
    assert report["cases"] == []


def test_unsupported_bf16_refuses_without_allocation(monkeypatch):
    monkeypatch.setattr(probe, "_measure_case", lambda *args: pytest.fail("must not allocate"))
    report = probe.run_probe(_torch(bf16=False))
    assert report["status"] == "refused"
    assert report["error_type"] == "Bfloat16Unavailable"


def test_low_driver_free_memory_refuses_every_case_before_allocation(monkeypatch):
    monkeypatch.setattr(probe, "_measure_case", lambda *args: pytest.fail("must not allocate"))
    report = probe.run_probe(_torch(free=probe.MINIMUM_FREE_BYTES - 1))
    assert report["status"] == "failed"
    assert all(row["status"] == "refused" for row in report["cases"])
    assert all(row["error_type"] == "InsufficientFreeMemory" for row in report["cases"])


def test_failed_case_releases_tensor_frame_before_later_cases(monkeypatch):
    class Buffer:
        pass

    references = []
    calls = []

    def measured(torch, case):
        calls.append(case.name)
        if len(calls) == 1:
            tensor = Buffer()
            references.append(weakref.ref(tensor))
            raise RuntimeError("synthetic kernel error")
        assert all(reference() is None for reference in references)
        return {"status": "passed", "finite_output": True, "peak_allocated_bytes": 123}

    monkeypatch.setattr(probe, "_measure_case", measured)
    report = probe.run_probe(_torch())
    assert len(calls) == 4
    assert report["status"] == "failed"
    assert report["cases"][0]["error_type"] == "RuntimeError"
    assert all(row["status"] == "passed" for row in report["cases"][1:])
    json.dumps(report, allow_nan=False)


def test_atomic_report_preserves_existing_output_and_removes_temporary_files(tmp_path):
    path = tmp_path / "diagnostics" / "probe.json"
    probe.write_report(path, {"status": "failed", "error": "retained"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        probe.write_report(path, {"status": "passed"})
    assert path.read_bytes() == original
    assert [item.name for item in path.parent.iterdir()] == ["probe.json"]


def test_describe_does_not_import_torch_or_transformers():
    process = subprocess.run([sys.executable, "-c", """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers'}:
        raise AssertionError('heavy runtime imported by --describe')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['probe_cuda_inference.py', '--describe']
runpy.run_path('scripts/probe_cuda_inference.py', run_name='__main__')
"""], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert len(report["cases"]) == 4
    assert report["math_backend_enabled"] is False


def test_runtime_import_failure_still_writes_a_failed_report(tmp_path, monkeypatch):
    def missing(name):
        raise ImportError("synthetic Torch unavailable")

    monkeypatch.setattr(probe.importlib, "import_module", missing)
    path = tmp_path / "report.json"
    assert probe.main(["--output", str(path)]) == 1
    report = json.loads(path.read_text())
    assert report["error_type"] == "ImportError"
    assert report["status"] == "failed"
    assert report["model_weights_loaded"] is False


def test_existing_output_refuses_before_runtime_import(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text('{"original":true}\n')
    monkeypatch.setattr(probe.importlib, "import_module", lambda *args: pytest.fail("runtime imported"))
    with pytest.raises(SystemExit) as exit_info:
        probe.main(["--output", str(path)])
    assert exit_info.value.code == 2
    assert json.loads(path.read_text()) == {"original": True}
