#!/usr/bin/env python3
"""Qualify the pinned Llama SDPA shapes on CUDA without loading model weights.

The prefix cases match Transformers 5.23's masked SDPA path: it repeats eight
KV heads to 32 heads and passes an offset causal mask. The first prefill and
single-token decode use unmasked GQA. This probes kernels and workspace only;
it does not certify that the checkpoint plus its complete KV cache will fit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import traceback
from typing import Any


GIB = 1024**3
MINIMUM_FREE_BYTES = 4 * GIB
MAXIMUM_PLANNED_TENSOR_BYTES = 2 * GIB


@dataclass(frozen=True)
class ProbeCase:
    name: str
    query_tokens: int
    key_tokens: int
    key_value_heads: int
    mask_dtype: str | None
    is_causal: bool
    enable_gqa: bool

    def description(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            query_shape=[1, 32, self.query_tokens, 128],
            key_value_shape=[1, self.key_value_heads, self.key_tokens, 128],
            tensor_dtype="bfloat16",
            mask_shape=[1, 1, self.query_tokens, self.key_tokens] if self.mask_dtype else None,
            mask_rule="key_index <= key_tokens - query_tokens + query_index" if self.mask_dtype else None,
        )
        # Q, K, V, output, and a conservative allowance for the simultaneously
        # live boolean construction masks. Fused-kernel workspaces are measured
        # by execution, and are deliberately not claimed by this tensor bound.
        q_and_output = 2 * 1 * 32 * self.query_tokens * 128 * 2
        kv = 2 * 1 * self.key_value_heads * self.key_tokens * 128 * 2
        mask_elements = self.query_tokens * self.key_tokens if self.mask_dtype else 0
        mask_bytes = mask_elements * (1 if self.mask_dtype == "bool" else 2)
        construction_bytes = mask_elements * 2
        value["planned_tensor_upper_bound_bytes"] = q_and_output + kv + mask_bytes + construction_bytes
        return value


def probe_cases(chunk_tokens: int = 2048, prefix_tokens: int = 65536) -> tuple[ProbeCase, ...]:
    if type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 2048:
        raise ValueError("chunk_tokens must be an integer between 1 and 2048")
    if type(prefix_tokens) is not int or not chunk_tokens <= prefix_tokens <= 65536:
        raise ValueError("prefix_tokens must be between chunk_tokens and 65536")
    cases = (
        ProbeCase("first_prefill_gqa", chunk_tokens, chunk_tokens, 8, None, True, True),
        ProbeCase("prefix_prefill_bool", chunk_tokens, prefix_tokens, 32, "bool", False, False),
        ProbeCase("prefix_prefill_bfloat16", chunk_tokens, prefix_tokens, 32, "bfloat16", False, False),
        ProbeCase("single_token_decode_gqa", 1, prefix_tokens, 8, None, False, True),
    )
    if any(case.description()["planned_tensor_upper_bound_bytes"] > MAXIMUM_PLANNED_TENSOR_BYTES for case in cases):
        raise ValueError("probe plan exceeds the two-GiB tensor bound")
    return cases


def _inputs(torch: Any, case: ProbeCase) -> tuple[Any, Any, Any, Any]:
    q = torch.zeros((1, case.query_tokens, 32, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    if case.key_value_heads == 32:
        # repeat_kv(...).reshape(...) in Transformers makes this layout.
        k = torch.zeros((1, 32, case.key_tokens, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.zeros_like(k)
    else:
        k = torch.zeros((1, case.key_tokens, 8, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        v = torch.zeros_like(k)
    mask = None
    if case.mask_dtype:
        key_indices = torch.arange(case.key_tokens, device="cuda")
        query_indices = torch.arange(case.query_tokens, device="cuda")
        allowed = key_indices[None, :] <= case.key_tokens - case.query_tokens + query_indices[:, None]
        if case.mask_dtype == "bool":
            mask = allowed[None, None, :, :]
        else:
            mask = torch.zeros((1, 1, case.query_tokens, case.key_tokens), device="cuda", dtype=torch.bfloat16)
            mask.masked_fill_(~allowed[None, None, :, :], float("-inf"))
    return q, k, v, mask


def _measure_case(torch: Any, case: ProbeCase) -> dict[str, Any]:
    """Keep tensors inside this frame so its caller can clear a failed frame."""
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    q, k, v, mask = _inputs(torch, case)
    params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, case.is_causal, case.enable_gqa)
    capabilities = {
        name: bool(getattr(torch.backends.cuda, function)(params, debug=True))
        for name, function in (
            ("flash", "can_use_flash_attention"),
            ("efficient", "can_use_efficient_attention"),
            ("cudnn", "can_use_cudnn_attention"),
        )
    }
    backend = torch.nn.attention.SDPBackend
    fused = [backend.FLASH_ATTENTION, backend.EFFICIENT_ATTENTION, backend.CUDNN_ATTENTION]
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(backends=fused):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0,
                is_causal=case.is_causal, enable_gqa=case.enable_gqa,
            )
            torch.cuda.synchronize()
        finite = bool(torch.isfinite(output).all().item())
    operators = sorted({event.key for event in profiler.key_averages()
                        if "scaled_dot_product" in event.key or "cudnn_attention" in event.key})
    if not finite:
        raise RuntimeError("fused SDPA returned non-finite output")
    if list(output.shape) != [1, 32, case.query_tokens, 128]:
        raise RuntimeError("fused SDPA returned an unexpected output shape")
    if any("attention_math" in operator for operator in operators):
        raise RuntimeError("math SDPA was observed despite the fused-only execution policy")
    return {
        "status": "passed", "capabilities": capabilities, "finite_output": finite,
        "output_shape": list(output.shape), "sdpa_operators": operators,
        "baseline_allocated_bytes": int(baseline_allocated),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "query_stride": list(q.stride()), "key_stride": list(k.stride()),
        "mask_stride": list(mask.stride()) if mask is not None else None,
    }


def _run_case(torch: Any, case: ProbeCase) -> dict[str, Any]:
    row = case.description()
    free, total = torch.cuda.mem_get_info()
    row.update(free_before_bytes=int(free), total_memory_bytes=int(total))
    if min(free, total) < MINIMUM_FREE_BYTES:
        return {**row, "status": "refused", "error_type": "InsufficientFreeMemory",
                "error_message": "At least four GiB of driver-free CUDA memory is required before allocating probe tensors"}
    try:
        row.update(_measure_case(torch, case))
    except Exception as exc:
        row.update(status="failed", error_type=type(exc).__name__, error_message=str(exc)[:2000])
        # A failed model/kernel frame may own large CUDA tensors. Store strings
        # only, and release stopped frames before the next case is attempted.
        traceback.clear_frames(exc.__traceback__)
    gc.collect()
    torch.cuda.empty_cache()
    free_after, _ = torch.cuda.mem_get_info()
    row["free_after_cleanup_bytes"] = int(free_after)
    return row


def run_probe(torch: Any, *, chunk_tokens: int = 2048, prefix_tokens: int = 65536) -> dict[str, Any]:
    cases = probe_cases(chunk_tokens, prefix_tokens)
    report: dict[str, Any] = {
        "schema_version": 1, "probe_kind": "llama31_bfloat16_sdpa_kernel_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_weights_loaded": False, "network_access_requested": False,
        "math_backend_enabled": False, "minimum_free_cuda_bytes": MINIMUM_FREE_BYTES,
        "chunk_tokens": chunk_tokens, "prefix_tokens": prefix_tokens,
        "scope": "Kernel qualification only; checkpoint plus full KV-cache memory still requires real-generation validation",
        "source_basis": {
            "transformers_version": "5.23.0",
            "first_and_decode": "transformers.integrations.sdpa_attention.use_gqa_in_sdpa: unmasked GQA",
            "prefix": "transformers.integrations.sdpa_attention.sdpa_attention_forward: masked multi-token repeat_kv",
            "causal_mask": "Offset causal mask: key_index <= key_tokens - query_tokens + query_index",
        },
        "runtime": {"torch": str(getattr(torch, "__version__", "unknown")),
                    "cuda": getattr(getattr(torch, "version", None), "cuda", None),
                    "python": platform.python_version(), "platform": platform.platform()},
        "cases": [],
    }
    if not torch.cuda.is_available():
        return {**report, "status": "refused", "error_type": "CUDAUnavailable",
                "error_message": "CUDA is unavailable; no tensors were allocated"}
    device = torch.cuda.current_device()
    report["device"] = {"index": int(device), "name": torch.cuda.get_device_name(device),
                        "capability": list(torch.cuda.get_device_capability(device)),
                        "bfloat16_supported": bool(torch.cuda.is_bf16_supported())}
    if not report["device"]["bfloat16_supported"]:
        return {**report, "status": "refused", "error_type": "Bfloat16Unavailable",
                "error_message": "The CUDA device does not support BF16; no tensors were allocated"}
    report["cases"] = [_run_case(torch, case) for case in cases]
    report["status"] = "passed" if all(row["status"] == "passed" for row in report["cases"]) else "failed"
    return report


def write_report(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete JSON report atomically and never replace an earlier run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New JSON report path; existing reports are preserved")
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--prefix-tokens", type=int, default=65536)
    parser.add_argument("--describe", action="store_true", help="Print the allocation plan without importing Torch or allocating tensors")
    args = parser.parse_args(argv)
    try:
        cases = probe_cases(args.chunk_tokens, args.prefix_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    if args.describe:
        print(json.dumps({"cases": [case.description() for case in cases],
                          "math_backend_enabled": False, "model_weights_loaded": False}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required unless --describe is selected")
    if args.output.exists():
        parser.error("--output already exists; choose a new report path")
    try:
        torch = importlib.import_module("torch")
        report = run_probe(torch, chunk_tokens=args.chunk_tokens, prefix_tokens=args.prefix_tokens)
    except Exception as exc:
        report = {"schema_version": 1, "status": "failed", "probe_kind": "llama31_bfloat16_sdpa_kernel_v1",
                  "model_weights_loaded": False, "network_access_requested": False,
                  "error_type": type(exc).__name__, "error_message": str(exc)[:2000]}
        traceback.clear_frames(exc.__traceback__)
    try:
        report["transformers_package_version"] = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        report["transformers_package_version"] = None
    write_report(args.output, report)
    print(json.dumps({"status": report["status"], "report": str(args.output)}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
