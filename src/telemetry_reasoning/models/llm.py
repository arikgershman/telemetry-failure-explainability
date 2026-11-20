"""Track 5 inference boundary, with an offline stub and a lazy HF adapter.

Only :class:`Prompt` objects reach a backend. Storage labels and pair IDs stay
outside that boundary. Greedy decoding and the pinned checkpoint are shared by
all variants; optional batching and explicit Metal quantization change execution
settings, which are recorded with raw completions for reproducibility.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import time
import traceback
from typing import Annotated, Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from telemetry_reasoning.eval.contracts import EvidenceItem
from telemetry_reasoning.models.prompts import ModelVariant, Prompt, build_prompt
from telemetry_reasoning.schemas import FailureMode, PairedBehaviorTree, Subtask, TASK_SUBTASKS, Task


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
ResponsePolicy = Literal["strict_json_v1", "json_or_fenced_v1"]
RESPONSE_POLICIES = ("strict_json_v1", "json_or_fenced_v1")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/llm.yaml"
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
GIB = 1024 ** 3
# These are refusal thresholds for this exact 8B checkpoint, not a claim that
# every admitted workload fits. Quantizing during loading can still approach
# full-precision loading peaks, so Metal does not lower the host requirement.
MEMORY_POLICY = "llama31_8b_minimum_memory_v1"
MINIMUM_HOST_GIB = {"mps": 32, "cpu": 48, "cuda": 32}
MPS_ALLOCATOR_FRACTION = 0.8


class ResourceSafetyError(RuntimeError):
    """Real inference was refused before an unsafe or unverifiable allocation."""


@dataclass(frozen=True)
class ResourceSnapshot:
    host_memory_bytes: int | None
    source: str
    available_host_memory_bytes: int | None = None


def _psutil_resources() -> ResourceSnapshot:
    """Optional OS-only measurements, including Windows physical-memory data."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        total = memory.total if type(memory.total) is int and memory.total > 0 else None
        available = memory.available if type(memory.available) is int and memory.available > 0 else None
        return ResourceSnapshot(total, "psutil.virtual_memory", available)
    except Exception:
        return ResourceSnapshot(None, "psutil.unavailable")


def _host_resources() -> ResourceSnapshot:
    """Read RAM without importing torch or creating a device.

    Physical RAM uses Darwin sysctl or POSIX sysconf when supported, with an
    optional psutil fallback (including Windows). Prefer psutil's OS available
    estimate to free-only POSIX pages, which can omit reclaimable cache. Unknown
    physical or available measurements cannot admit real model loading.
    """
    optional = _psutil_resources()
    if platform.system() == "Darwin":
        try:
            import ctypes

            value = ctypes.c_uint64()
            size = ctypes.c_size_t(ctypes.sizeof(value))
            query = ctypes.CDLL(None, use_errno=True).sysctlbyname
            query.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                              ctypes.c_void_p, ctypes.c_size_t]
            query.restype = ctypes.c_int
            if query(b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0) == 0 and value.value > 0:
                return ResourceSnapshot(value.value, "sysctlbyname(hw.memsize); available:" + optional.source,
                                        optional.available_host_memory_bytes)
        except (AttributeError, OSError, ValueError, OverflowError):
            pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if type(page_size) is int and type(pages) is int and page_size > 0 and pages > 0:
            available = optional.available_host_memory_bytes
            source = "os.sysconf; available:" + optional.source
            if available is None:
                try:
                    available_pages = os.sysconf("SC_AVPHYS_PAGES")
                    if type(available_pages) is int and available_pages > 0:
                        available = available_pages * page_size
                        source = "os.sysconf(physical_and_available_pages)"
                except (AttributeError, OSError, ValueError):
                    pass
            return ResourceSnapshot(pages * page_size, source, available)
    except (AttributeError, OSError, ValueError, OverflowError):
        pass
    return optional


def _host_resource_assessment(
    config: "LLMConfig", snapshot: ResourceSnapshot, *, resident: bool = False,
    required_available_bytes: int | None = None,
) -> dict[str, Any]:
    required = MINIMUM_HOST_GIB[config.device] * GIB
    available_floor = (required_available_bytes if required_available_bytes is not None
                       else 4 * GIB if resident else required * 3 // 4)
    total = snapshot.host_memory_bytes
    available = snapshot.available_host_memory_bytes
    reason = None
    if type(total) is not int or total <= 0:
        reason = "physical host RAM could not be verified"
    elif total < required:
        reason = f"{config.device} requires at least {MINIMUM_HOST_GIB[config.device]} GiB physical host RAM for this checkpoint"
    elif type(available) is not int or available <= 0:
        reason = "currently available host RAM could not be verified"
    elif available < available_floor:
        reason = "currently available host RAM is below this checkpoint's conservative headroom policy"
    return {
        "policy": MEMORY_POLICY, "phase": "resident_execution" if resident else "weight_load",
        "host_memory_bytes": total,
        "available_host_memory_bytes": available, "host_memory_source": snapshot.source,
        "minimum_host_memory_bytes": required,
        "minimum_available_host_memory_bytes_when_measurable": available_floor,
        "host_admissible": reason is None, "reason": reason,
        "policy_note": "Minimum admission thresholds do not guarantee that a workload fits; context, batching, other applications and loading peaks still consume memory.",
    }


def _mps_allocator_fraction() -> float:
    value = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if value is None:
        return MPS_ALLOCATOR_FRACTION
    try:
        configured = float(value)
    except (ValueError, OverflowError) as exc:
        raise ResourceSafetyError("MPS allocator watermark override must be finite and positive") from exc
    if not math.isfinite(configured) or configured <= 0:
        raise ResourceSafetyError("disabled or invalid MPS allocator limits are not supported")
    return min(MPS_ALLOCATOR_FRACTION, configured)


class LLMConfig(BaseModel):
    """A separate, strict config; never extends the frozen project YAML.

    Model and tokenizer always use the same instruction-tuned checkpoint and
    immutable revision. Network downloads require an explicit config change.
    Metal 4-bit loading is an explicit Apple Silicon option. Its kernel loader
    can access the Hub independently of model downloads, so that option requires
    network-enabled loading. The unquantized default remains cache-only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model_id: Literal["meta-llama/Llama-3.1-8B-Instruct"] = MODEL_ID
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] = MODEL_REVISION
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    context_window_tokens: Annotated[int, Field(strict=True, gt=0, le=131072)] = 65536
    max_new_tokens: PositiveInt = 512
    batch_size: Annotated[int, Field(strict=True, gt=0, le=128)] = 1
    seed: Annotated[int, Field(strict=True, ge=0, lt=2**32)] = 723
    response_policy: ResponsePolicy = "strict_json_v1"
    decoding_constraint: Literal["none", "observed_json_v1"] = "none"
    prefill_chunk_size: Annotated[int, Field(strict=True, gt=0, le=2048)] | None = None
    attention_implementation: Literal["auto", "sdpa"] = "auto"
    fused_attention_only: Annotated[bool, Field(strict=True)] = False
    local_files_only: Annotated[bool, Field(strict=True)] = True
    quantization: Literal["none", "metal4"] = "none"
    quantization_group_size: Literal[64] = 64
    metal_kernel_revision: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{40}$")] | None = None
    prompt_dir: Path | None = None
    cache_dir: Path | None = None

    @model_validator(mode="after")
    def valid_budget(self) -> "LLMConfig":
        if self.max_new_tokens >= self.context_window_tokens:
            raise ValueError("max_new_tokens must leave room for the input prompt")
        if self.decoding_constraint != "none" and self.batch_size != 1:
            raise ValueError("observed_json_v1 requires batch_size=1")
        if self.fused_attention_only and (self.device != "cuda" or self.attention_implementation != "sdpa"):
            raise ValueError("fused_attention_only requires device=cuda and attention_implementation=sdpa")
        if self.prefill_chunk_size is not None:
            if self.attention_implementation != "sdpa" or self.device == "mps":
                raise ValueError("chunked prefill requires CPU/CUDA SDPA")
            if self.device == "cuda" and not self.fused_attention_only:
                raise ValueError("CUDA chunked prefill requires fused_attention_only; math fallback is unsafe")
        if self.quantization == "metal4":
            if self.device != "mps" or self.dtype != "float16":
                raise ValueError("metal4 requires device=mps and dtype=float16; no fallback is supported")
            if self.local_files_only:
                raise ValueError("metal4 requires local_files_only=false because its kernel loader may access the Hub")
        return self


def execution_policy(config: LLMConfig) -> dict[str, Any]:
    return {"prefill_chunk_size": config.prefill_chunk_size,
            "cache_implementation": "dynamic" if config.prefill_chunk_size is not None else "checkpoint_default",
            "disable_compile": config.prefill_chunk_size is not None,
            "attention_implementation": config.attention_implementation,
            "fused_attention_only": config.fused_attention_only,
            "decoding_constraint": config.decoding_constraint}


def load_llm_config(path: str | Path = DEFAULT_CONFIG_PATH) -> LLMConfig:
    """Load ``llm: {...}``; resolve custom templates relative to this file."""
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"llm"} or not isinstance(value["llm"], dict):
        raise ValueError("LLM config must contain exactly one 'llm' mapping")
    config = LLMConfig.model_validate(value["llm"])
    paths = {name: (path.parent / value).resolve()
             for name in ("prompt_dir", "cache_dir") if (value := getattr(config, name)) is not None}
    if paths:
        config = config.model_copy(update=paths)
    return config


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "transformers", "huggingface-hub", "kernels", "accelerate", "pydantic", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capabilities(config: LLMConfig) -> dict[str, Any]:
    """Inspect packages and host admission without importing torch or devices.

    This is a compute preflight, not proof of gated-repository access. No
    credentials, token values, or environment contents are returned.
    """
    result: dict[str, Any] = {
        "model_id": config.model_id, "revision": config.revision,
        "packages": _versions(), "configured_device": config.device,
        "configured_dtype": config.dtype, "local_files_only": config.local_files_only,
        "quantization": config.quantization, "quantization_group_size": config.quantization_group_size,
        "batch_size": config.batch_size,
        "remote_access": "not_checked", "weights": "not_loaded",
        "cuda_available": None, "mps_available": None,
        "device_probe": "not_performed; no torch import or device initialization",
        "resources": _host_resource_assessment(config, _host_resources()),
    }
    if config.device == "mps":
        try:
            result["resources"]["mps_allocator_fraction"] = _mps_allocator_fraction()
        except ResourceSafetyError as exc:
            result["resources"].update(host_admissible=False, reason=str(exc))
    return result


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    token_count_method: str = "tokenizer"
    finish_reason: str | None = None


class InferenceBackend(Protocol):
    name: str

    def count_tokens(self, prompt: Prompt) -> int: ...

    def generate(self, prompt: Prompt) -> Generation: ...

    def provenance(self) -> dict[str, Any]: ...


class StubBackend:
    """A deterministic plumbing stub that always abstains, never diagnoses.

    UTF-8 bytes plus a framing allowance are a conservative input token bound,
    not a Llama tokenizer measurement. It needs neither torch nor credentials.
    """

    name = "stub"
    token_count_method = "utf8_bytes_upper_bound"

    def __init__(self, config: LLMConfig):
        self.config = config

    def count_tokens(self, prompt: Prompt) -> int:
        return len((prompt.system + prompt.user).encode("utf-8")) + 32

    def generate(self, prompt: Prompt) -> Generation:
        text = _json({
            "predicted_failure_mode": None, "predicted_subtask": None,
            "explanation": "Synthetic stub for pipeline verification; no model diagnosis was performed.",
            "evidence": [], "status": "abstain",
        })
        return Generation(text, self.count_tokens(prompt), len(text.encode("utf-8")), self.token_count_method)

    def provenance(self) -> dict[str, Any]:
        return {"implementation": "fixed_abstention_v1", "weights_loaded": False,
                "token_count_method": self.token_count_method}


class HFBackend:
    """Lazy, pinned HF inference with left-padded batches and no truncation.

    ``generate`` remains the single-request API; ``generate_batch`` performs one
    real model call. Counts exclude prompt padding and completion padding after
    the first EOS. All rows share greedy decoding settings and a recorded seed.
    Unsupported quantization/device/runtime combinations fail without fallback.
    """

    name = "hf"
    token_count_method = "tokenizer"
    METAL_KERNEL_REPOSITORY = "kernels-community/mlx-quantization-metal-kernels"

    def __init__(self, config: LLMConfig):
        self.config = config
        self._torch = self._transformers = self._tokenizer = self._model = None
        self._kernel_diagnostics: dict[str, Any] | None = None
        self._resource_diagnostics: dict[str, Any] | None = None
        self._cuda_cache_releases: list[dict[str, Any]] = []
        self._constraint_diagnostics: list[dict[str, Any]] = []
        self._request_memory: list[dict[str, Any]] = []

    def _preflight_host(self, *, kv_cache_bytes: int | None = None) -> None:
        resident = self._model is not None
        available_floor = None
        if resident:
            available_floor = 4 * GIB + ((kv_cache_bytes or 0) if self.config.device in ("cpu", "mps") else 0)
        self._resource_diagnostics = _host_resource_assessment(
            self.config, _host_resources(), resident=resident, required_available_bytes=available_floor,
        )
        if self._cuda_cache_releases:
            self._resource_diagnostics["cuda_cache_releases"] = self._cuda_cache_releases
            self._resource_diagnostics["cuda_cache_release"] = self._cuda_cache_releases[-1]
        if kv_cache_bytes is not None:
            self._resource_diagnostics["kv_cache_estimate_bytes"] = kv_cache_bytes
        if not self._resource_diagnostics["host_admissible"]:
            raise ResourceSafetyError("HF loading refused: " + self._resource_diagnostics["reason"])
        if self.config.device == "mps":
            try:
                self._resource_diagnostics["mps_allocator_fraction"] = _mps_allocator_fraction()
            except ResourceSafetyError as exc:
                self._resource_diagnostics.update(host_admissible=False, reason=str(exc))
                raise

    def _preflight_device(self, torch: Any, *, kv_cache_bytes: int | None = None) -> None:
        if self.config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("configured CUDA device is unavailable")
            try:
                free, total = torch.cuda.mem_get_info()
            except Exception as exc:
                raise ResourceSafetyError("CUDA memory could not be verified before loading") from exc
            required = ((4 * GIB + (kv_cache_bytes or 0)) if self._model is not None
                        else (40 if self.config.dtype == "float32" else 24) * GIB)
            # Driver-free memory excludes PyTorch's inactive cached blocks.
            # A previous long request may leave little driver-free memory even
            # though its workspace/KV tensors have been released. Reclaim only
            # unused cache before refusing a resident request, then apply the
            # same guard to fresh measurements; never infer reusable capacity
            # from reserved-minus-allocated bytes or lower the safety floor.
            release_cache = getattr(torch.cuda, "empty_cache", None)
            if (self._model is not None and type(free) is int and type(total) is int
                    and min(free, total) < required and callable(release_cache)):
                recovery = {"attempted": True, "reason": "resident_guard_insufficient_driver_free_memory",
                            "minimum_cuda_memory_bytes": required,
                            "free_before_bytes": free, "free_after_bytes": None}
                self._cuda_cache_releases.append(recovery)
                self._resource_diagnostics["cuda_cache_release"] = recovery
                self._resource_diagnostics["cuda_cache_releases"] = self._cuda_cache_releases
                try:
                    release_cache()
                    free, total = torch.cuda.mem_get_info()
                except Exception as exc:
                    recovery["error_type"] = type(exc).__name__
                    raise ResourceSafetyError("CUDA memory could not be verified after releasing unused cache") from exc
                recovery["free_after_bytes"] = free
            self._resource_diagnostics.update(cuda_free_memory_bytes=free, cuda_total_memory_bytes=total,
                                              minimum_cuda_memory_bytes=required)
            if type(free) is not int or type(total) is not int or min(free, total) < required:
                raise ResourceSafetyError("CUDA loading refused: insufficient free/total VRAM for this checkpoint and dtype")
            if self.config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("configured CUDA device does not support bfloat16")
        if self.config.device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if not mps or not mps.is_available():
                raise RuntimeError("configured MPS device is unavailable")
            allocator = getattr(getattr(torch, "mps", None), "set_per_process_memory_fraction", None)
            if allocator is None:
                raise ResourceSafetyError("MPS allocator cap cannot be enforced by this runtime")
            fraction = self._resource_diagnostics["mps_allocator_fraction"]
            allocator(fraction)
            self._resource_diagnostics["mps_allocator_cap_applied"] = True

    def _runtime(self) -> None:
        # This must precede every optional runtime import, tokenizer/cache load,
        # or kernel access. A 16 GiB Mac is rejected using OS metadata alone.
        self._preflight_host()
        if self._torch is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError("HF backend requires the optional .[llm] dependencies") from exc
        self._preflight_device(torch)
        self._torch, self._transformers = torch, transformers

    def _load_arguments(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "revision": self.config.revision,
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": False,
        }
        if self.config.cache_dir is not None:
            kwargs["cache_dir"] = str(self.config.cache_dir)
        return kwargs

    def _load_tokenizer(self) -> None:
        self._runtime()
        if self._tokenizer is None:
            tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                self.config.model_id, **self._load_arguments(),
            )
            if not tokenizer.chat_template:
                raise ValueError("instruction checkpoint tokenizer must declare a chat template")
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
                    raise ValueError("tokenizer needs an EOS token to supply missing padding")
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                raise ValueError("tokenizer could not configure a padding token")
            resolved = self._resolved_revision(tokenizer)
            if resolved is not None and resolved != self.config.revision:
                raise ValueError("tokenizer resolved revision differs from the pinned revision")
            self._tokenizer = tokenizer

    def _encode(self, prompts: Sequence[Prompt]) -> tuple[Any, list[int]]:
        if not prompts:
            raise ValueError("cannot encode an empty prompt batch")
        self._load_tokenizer()
        encoded = self._tokenizer.apply_chat_template(
            [prompt.messages for prompt in prompts], tokenize=True,
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
            padding=True, truncation=False,
        )
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise ValueError("chat tokenization must return input_ids and an attention_mask")
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if len(ids.shape) != 2 or ids.shape != mask.shape or ids.shape[0] != len(prompts):
            raise ValueError("tokenizer returned inconsistent batch dimensions")
        masks = mask.tolist()
        counts = []
        for row in masks:
            # Enforce genuine left padding. Never infer attention from token IDs:
            # EOS is also the padding token in many Llama tokenizers.
            if any(value not in (0, 1) for value in row) or 1 not in row:
                raise ValueError("tokenizer returned an invalid attention mask")
            first = row.index(1)
            if any(value != 1 for value in row[first:]):
                raise ValueError("decoder-only batches require contiguous left padding")
            counts.append(sum(row))
        return encoded, counts

    def count_tokens(self, prompt: Prompt) -> int:
        return self._encode([prompt])[1][0]

    @staticmethod
    def _resolved_revision(resource: Any) -> str | None:
        revision = getattr(resource, "_commit_hash", None)
        if revision is None:
            kwargs = getattr(resource, "init_kwargs", {})
            revision = kwargs.get("_commit_hash") if isinstance(kwargs, dict) else None
        return revision if isinstance(revision, str) and revision else None

    def _prepare_metal(self) -> Any:
        if not hasattr(self._transformers, "MetalConfig"):
            raise RuntimeError("metal4 requires a Transformers runtime exposing MetalConfig")
        try:
            kernels = importlib.import_module("kernels")
            integration = importlib.import_module("transformers.integrations.metal_quantization")
            loader = getattr(integration, "_get_metal_kernel", None)
            if loader is None:
                raise RuntimeError("installed Transformers does not expose its Metal kernel loader")
            kernel = loader()
        except (ImportError, OSError, AttributeError) as exc:
            raise RuntimeError("metal4 requires compatible kernels and an Apple Silicon Metal kernel build") from exc
        source = getattr(kernel, "__file__", None)
        diagnostics: dict[str, Any] = {
            "repository": self.METAL_KERNEL_REPOSITORY,
            "loader": "transformers.integrations.metal_quantization._get_metal_kernel",
            "module": getattr(kernel, "__name__", None),
            "module_file": str(source) if source is not None else None,
            "kernels_version": getattr(kernels, "__version__", None),
            "expected_revision": self.config.metal_kernel_revision,
            "resolved_revision": None,
            "loaded": True,
        }
        integration_source = getattr(integration, "__file__", None)
        if integration_source is not None and Path(integration_source).is_file():
            diagnostics["transformers_integration_sha256"] = hashlib.sha256(Path(integration_source).read_bytes()).hexdigest()
        if source is not None and Path(source).is_file():
            # Hub snapshot files may be symlinks into blobs/. Resolve only for
            # the artifact path; the lexical snapshot path carries its revision.
            source_path = Path(source)
            diagnostics["module_file_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            diagnostics["module_resolved_file"] = str(source_path.resolve())
            parts = source_path.parts
            diagnostics["resolved_revision"] = (
                parts[parts.index("snapshots") + 1] if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts)
                else None
            )
        self._kernel_diagnostics = diagnostics
        if self.config.metal_kernel_revision is not None and diagnostics["resolved_revision"] != self.config.metal_kernel_revision:
            raise ValueError("loaded Metal kernel revision does not match metal_kernel_revision; refusing kernel drift")
        return self._transformers.MetalConfig(
            bits=4, group_size=self.config.quantization_group_size, dequantize=False,
        )

    def _load_model(self) -> None:
        # Recheck admission immediately before the memory-intensive operation;
        # context formatting and queue time may have changed available memory.
        self._preflight_host()
        self._preflight_device(self._torch)
        if self._model is None:
            kwargs = self._load_arguments()
            kwargs["torch_dtype"] = getattr(self._torch, self.config.dtype)
            if self.config.attention_implementation != "auto":
                kwargs["attn_implementation"] = self.config.attention_implementation
            if self.config.quantization == "metal4":
                kwargs.update(quantization_config=self._prepare_metal(), device_map="mps")
            model = self._transformers.AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
            if self.config.quantization == "none":
                model = model.to(self.config.device)
            model.eval()
            revision = self._resolved_revision(model.config)
            if revision is not None and revision != self.config.revision:
                raise ValueError("model resolved revision differs from the pinned revision")
            if self.config.quantization == "metal4":
                quantizer = getattr(model, "hf_quantizer", None)
                quant_config = getattr(quantizer, "quantization_config", None)
                if quant_config is None or getattr(quant_config, "dequantize", True):
                    raise ValueError("Metal quantization was not applied; refusing an unquantized fallback")
                devices = getattr(model, "hf_device_map", {})
                if any(not str(device).startswith("mps") for device in devices.values()):
                    raise ValueError("Metal quantized weights must remain on MPS without offloading")
            if (self.config.attention_implementation != "auto" and
                    getattr(model.config, "_attn_implementation", None) != self.config.attention_implementation):
                raise RuntimeError("loaded model attention differs from the requested implementation")
            self._model = model
        native_limit = getattr(self._model.config, "max_position_embeddings", None)
        if native_limit and self.config.context_window_tokens > native_limit:
            raise ValueError("configured context exceeds checkpoint context window")

    def generate(self, prompt: Prompt) -> Generation:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: Sequence[Prompt]) -> tuple[Generation, ...]:
        """Generate one left-padded batch; reject any overlong input explicitly.

        The harness filters invalid requests before forming batches. Direct
        callers receive an error for the whole batch if any input exceeds budget.
        No hidden retries alter batch composition, settings, or output limits.
        """
        if len(prompts) > self.config.batch_size:
            raise ValueError("prompt batch exceeds configured batch_size")
        encoded, counts = self._encode(prompts)
        if any(count + self.config.max_new_tokens > self.config.context_window_tokens for count in counts):
            raise ValueError("prompt plus reserved output exceeds context budget; no truncation performed")
        constraints = []
        if self.config.decoding_constraint == "observed_json_v1":
            from telemetry_reasoning.models.constrained import ObservedResponseConstraint
            constraints = [ObservedResponseConstraint(prompt, self._tokenizer, self._tokenizer.eos_token_id,
                                                      self.config.max_new_tokens) for prompt in prompts]
        if self.config.prefill_chunk_size is not None and not hasattr(
                self._transformers.GenerationConfig(), "prefill_chunk_size"):
            raise RuntimeError("installed Transformers does not support native chunked prefill")
        self._load_model()
        # Pinned Llama-3.1-8B KV shape: K/V x32 layers x8 KV heads x128
        # channels, at the configured dtype, across the padded batch. Reserve
        # this actual request plus 4 GiB workspace after weights are resident;
        # do not reserve the entire checkpoint a second time for later batches.
        element_bytes = 4 if self.config.dtype == "float32" else 2
        kv_cache_bytes = 2 * 32 * 8 * 128 * element_bytes * (max(counts) + self.config.max_new_tokens) * len(counts)
        self._preflight_host(kv_cache_bytes=kv_cache_bytes)
        self._preflight_device(self._torch, kv_cache_bytes=kv_cache_bytes)
        width = int(encoded["input_ids"].shape[-1])
        inputs = {name: encoded[name].to(self.config.device) for name in ("input_ids", "attention_mask")}
        self._torch.manual_seed(self.config.seed)
        eos_ids = getattr(self._model.generation_config, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = self._tokenizer.eos_token_id
        generation_options = dict(
            do_sample=False, num_beams=1, max_new_tokens=self.config.max_new_tokens,
            eos_token_id=eos_ids, pad_token_id=self._tokenizer.pad_token_id,
        )
        if self.config.prefill_chunk_size is not None:
            generation_options.update(prefill_chunk_size=self.config.prefill_chunk_size,
                                      cache_implementation="dynamic", disable_compile=True, use_cache=True)
        generation_config = self._transformers.GenerationConfig(**generation_options)
        callbacks = {}
        if constraints:
            declared_eos = set(eos_ids if isinstance(eos_ids, (tuple, list)) else [eos_ids])
            if not set(constraints[0].eos_ids) <= declared_eos:
                raise ValueError("constraint EOS is not an actual model generation terminator")
            def allowed(batch_id, sequence):
                return constraints[batch_id].allowed_tokens(sequence[width:].tolist())
            callbacks["prefix_allowed_tokens_fn"] = allowed
            self._constraint_diagnostics.extend({"prompt_sha256": prompt.prompt_sha256, **constraint.provenance()}
                                                for prompt, constraint in zip(prompts, constraints))
        attention_context = nullcontext()
        if self.config.fused_attention_only:
            attention = importlib.import_module("torch.nn.attention")
            attention_context = attention.sdpa_kernel([attention.SDPBackend.FLASH_ATTENTION,
                attention.SDPBackend.EFFICIENT_ATTENTION, attention.SDPBackend.CUDNN_ATTENTION])
        memory = None
        if self.config.device == "cuda" and self.config.prefill_chunk_size is not None:
            self._torch.cuda.reset_peak_memory_stats()
            memory = {"prompt_sha256": [prompt.prompt_sha256 for prompt in prompts],
                      "input_tokens": counts, "prefill_chunk_size": self.config.prefill_chunk_size,
                      "kv_cache_estimate_bytes": kv_cache_bytes, "completed": False}
        try:
            with self._torch.inference_mode(), attention_context:
                generated = self._model.generate(**inputs, generation_config=generation_config, **callbacks)
            if memory is not None:
                self._torch.cuda.synchronize()
                memory["completed"] = True
        finally:
            if memory is not None:
                try:
                    memory.update(peak_allocated_bytes=self._torch.cuda.max_memory_allocated(),
                                  peak_reserved_bytes=self._torch.cuda.max_memory_reserved())
                except Exception as exc:
                    memory["measurement_error"] = type(exc).__name__
                self._request_memory.append(memory)
        if len(generated.shape) != 2 or generated.shape[0] != len(prompts) or generated.shape[1] < width:
            raise ValueError("model returned inconsistent generated batch dimensions")
        eos = set(eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]) - {None}
        results = []
        for index, count in enumerate(counts):
            # Generation appends after the full padded width, not the row's
            # unpadded count. Stop at the first EOS to exclude later batch pads.
            token_ids = generated[index, width:].tolist()
            stop = next((position + 1 for position, token in enumerate(token_ids) if token in eos), None)
            if stop is not None:
                token_ids = token_ids[:stop]
            if len(token_ids) > self.config.max_new_tokens:
                raise ValueError("model returned more than max_new_tokens")
            if constraints:
                if not token_ids or token_ids[-1] not in constraints[index].eos_ids:
                    raise ValueError("constrained generation did not finish with its declared EOS")
                # Validate the actual emitted history as well as installing the
                # callback. An engine that bypasses it must fail explicitly.
                constraints[index].allowed_tokens(token_ids)
            results.append(Generation(
                self._tokenizer.decode(token_ids, skip_special_tokens=True,
                                       **({"clean_up_tokenization_spaces": False} if constraints else {})), count,
                len(token_ids), self.token_count_method,
                "eos" if stop is not None else "length" if len(token_ids) == self.config.max_new_tokens else "unknown",
            ))
        return tuple(results)

    def provenance(self) -> dict[str, Any]:
        tokenizer = self._tokenizer
        model_config = self._model.config if self._model is not None else None
        quantizer = getattr(self._model, "hf_quantizer", None)
        quant_config = getattr(quantizer, "quantization_config", None)
        return {
            "implementation": ("transformers_batched_greedy_v3" if self.config.prefill_chunk_size is not None
                               or self.config.decoding_constraint != "none" else "transformers_batched_greedy_v2"),
            "execution_policy": execution_policy(self.config),
            "constraints": self._constraint_diagnostics,
            "request_memory": self._request_memory,
            "weights_loaded": self._model is not None, "tokenizer_loaded": tokenizer is not None,
            "token_count_method": self.token_count_method,
            "device": self.config.device, "dtype": self.config.dtype,
            "batch_size": self.config.batch_size,
            "quantization": self.config.quantization,
            "quantization_group_size": self.config.quantization_group_size if self.config.quantization == "metal4" else None,
            "kernel": self._kernel_diagnostics,
            "resources": self._resource_diagnostics,
            "runtime": {"torch": getattr(self._torch, "__version__", None),
                        "transformers": getattr(self._transformers, "__version__", None),
                        "platform": platform.platform(), "machine": platform.machine()},
            "effective_quantization": {
                "bits": getattr(quant_config, "bits", None),
                "group_size": getattr(quant_config, "group_size", None),
                "dequantize": getattr(quant_config, "dequantize", None),
            } if quant_config is not None else None,
            "device_map": {name: str(device) for name, device in getattr(self._model, "hf_device_map", {}).items()},
            "model": {"id": self.config.model_id, "requested_revision": self.config.revision,
                      "resolved_revision": self._resolved_revision(model_config)},
            "tokenizer": {"id": self.config.model_id, "requested_revision": self.config.revision,
                          "resolved_revision": self._resolved_revision(tokenizer),
                          "padding_side": getattr(tokenizer, "padding_side", None),
                          "pad_token_id": getattr(tokenizer, "pad_token_id", None),
                          "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest() if tokenizer else None},
            "native_context_window_tokens": getattr(model_config, "max_position_embeddings", None),
        }


class _ModelResponse(BaseModel):
    """Model-controlled fields; join identity and backend are supplied locally."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    predicted_failure_mode: FailureMode | None
    predicted_subtask: Subtask | None
    explanation: Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]
    evidence: list[EvidenceItem]
    status: Literal["ok", "abstain", "error"]

    @model_validator(mode="after")
    def consistent_status(self) -> "_ModelResponse":
        if self.status == "ok":
            if self.predicted_failure_mode is None or self.predicted_subtask is None or not self.evidence:
                raise ValueError("ok requires a mode, subtask and observed evidence")
        elif self.predicted_failure_mode is not None or self.predicted_subtask is not None:
            raise ValueError("abstain/error requires null predicted labels")
        if self.status == "error" and self.evidence:
            raise ValueError("error requires empty evidence")
        return self


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _equal_measurement(actual: Any, claimed: Any) -> bool:
    if isinstance(actual, list) != isinstance(claimed, list):
        return False
    if isinstance(actual, list):
        return len(actual) == len(claimed) and all(_equal_measurement(a, b) for a, b in zip(actual, claimed))
    return math.isclose(actual, claimed, rel_tol=1e-9, abs_tol=1e-9)


def _response_payload(text: str, policy: ResponsePolicy) -> tuple[str, str]:
    """Recognize a complete transport envelope, never search or repair JSON.

    Legacy callers retain the bare-JSON contract. The opt-in envelope policy
    permits only lowercase ``json`` fences occupying the entire response, with
    LF or CRLF marker lines and optional surrounding JSON whitespace. The
    enclosed bytes still undergo every original JSON/schema/evidence check.
    """
    if policy not in RESPONSE_POLICIES:
        raise ValueError(f"unsupported response policy: {policy!r}")
    if not isinstance(text, str):
        raise TypeError("model completion must be a string")
    if policy == "json_or_fenced_v1":
        match = re.fullmatch(r"```json\r?\n(?P<body>[\s\S]*?)\r?\n```", text.strip(" \t\r\n"))
        if match is not None:
            return match.group("body"), "json_code_fence"
    return text, "none"


def response_normalization(text: str, policy: ResponsePolicy) -> dict[str, str]:
    """Describe envelope handling without changing or discarding raw text."""
    payload, envelope = _response_payload(text, policy)
    return {"policy": policy, "envelope": envelope,
            "normalized_text_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def parse_response(
    text: str, prompt: Prompt, pair_id: str, backend: str,
    *, response_policy: ResponsePolicy = "strict_json_v1",
) -> dict[str, Any]:
    """Require one strict JSON object and citations to available failure rows.

    A separately declared policy may unwrap one whole-response JSON fence.
    No repair, label guessing, prose extraction, or success-side evidence
    substitution occurs. Unsupported responses become errors in the harness.
    """
    if backend not in ("stub", "hf"):
        raise ValueError("unsupported inference backend")
    payload, _ = _response_payload(text, response_policy)
    response = _ModelResponse.model_validate(json.loads(
        payload, parse_constant=_reject_constant, object_pairs_hook=_unique_keys,
    ))
    allowed = TASK_SUBTASKS[Task(prompt.payload["task"])]
    if response.predicted_subtask is not None and response.predicted_subtask not in allowed:
        raise ValueError("predicted subtask does not belong to this task")
    rows = [row for node in prompt.payload["subtasks"] for row in node["failure_samples"]]
    if response.predicted_subtask is not None and not any(
        row["subtask"] == response.predicted_subtask.value for row in rows
    ):
        raise ValueError("predicted subtask was not observed in the failure run")
    for evidence in response.evidence:
        matches = [row for row in rows if row["subtask"] == evidence.subtask and
                   math.isclose(row["timestamp"], evidence.timestamp, rel_tol=0, abs_tol=1e-9)]
        if not matches or not _equal_measurement(matches[0][evidence.feature], evidence.value):
            raise ValueError("evidence must match an observed failure sample and value")
    return {"pair_id": pair_id, "model_variant": prompt.variant.value,
            **response.model_dump(mode="json"), "backend": backend}


@dataclass(frozen=True)
class InferenceRun:
    predictions: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    completions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _InferenceFailure:
    """CPU-only diagnostics; an exception object can retain device tensors."""

    error_type: str
    error_message: str
    causes: tuple[tuple[str, str, str], ...] = ()
    causes_truncated: bool = False


def _failure_message(error: BaseException) -> str:
    try:
        return str(error)[:1000]
    except Exception as formatting_error:
        return f"<{type(error).__name__}: message unavailable ({type(formatting_error).__name__})>"[:1000]


def _capture_inference_failure(error: Exception) -> _InferenceFailure:
    """Retain bounded text before dropping frame references outside ``except``."""
    causes = []
    current, seen = error, {id(error)}
    while len(causes) < 8:
        cause = current.__cause__
        relation = "cause"
        if cause is None and not current.__suppress_context__:
            cause, relation = current.__context__, "context"
        if cause is None:
            return _InferenceFailure(type(error).__name__, _failure_message(error), tuple(causes))
        if id(cause) in seen:
            break
        seen.add(id(cause))
        causes.append((relation, type(cause).__name__, _failure_message(cause)))
        current = cause
    remaining = current.__cause__
    if remaining is None and not current.__suppress_context__:
        remaining = current.__context__
    return _InferenceFailure(type(error).__name__, _failure_message(error), tuple(causes), remaining is not None)


def _release_failure_frames(error: Exception) -> None:
    """Clear finished traceback frames only after the handling block has exited.

    Failed model calls can leave their KV cache and temporary tensors in frame
    locals. Preserve their text diagnostics, then remove those references before
    admitting another batch. Live harness frames are left intact by clear_frames.
    No generation is retried and this does not release resident model weights.
    """
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)
        frames = current.__traceback__
        current.__traceback__ = None
        if frames is not None:
            traceback.clear_frames(frames)


def _capture_and_release_failure(error: Exception) -> _InferenceFailure:
    captured = _capture_inference_failure(error)
    _release_failure_frames(error)
    return captured


def _detach_batch_failures(results: tuple[Any, ...]) -> tuple[Any, ...]:
    # This function runs after generation's exception handler has exited. Its
    # output never keeps exception objects, even for per-row backend failures.
    return tuple(_capture_and_release_failure(result) if isinstance(result, Exception) else result
                 for result in results)


def run_inference(
    trees: Sequence[PairedBehaviorTree], variants: Sequence[str], config: LLMConfig, *,
    backend: InferenceBackend | None = None, template_dir: str | Path | None = None,
) -> InferenceRun:
    """Run validated results and retain actual completions, including bad JSON.

    Build/count/budget errors are isolated before generation. A backend may
    provide ``generate_batch(prompts)`` returning one Generation or Exception per
    prompt; a raised batch error accounts for every affected request and later
    batches continue. Single-request backends remain supported. There are no
    automatic retries, repairs, truncations, or invented completions.

    Request latency includes preparation and queue time. ``batch_latency_s`` is
    shared wall time for its batch, never a measured per-example generation cost.
    """
    variants = tuple(ModelVariant(variant) for variant in variants)
    if not trees or not variants or len(set(variants)) != len(variants):
        raise ValueError("nonempty inputs and unique nonempty variants are required")
    if any(not isinstance(tree, PairedBehaviorTree) for tree in trees):
        raise TypeError("inference inputs must be validated PairedBehaviorTree objects")
    pair_ids = [tree.metadata.pair_id for tree in trees]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("duplicate pair_id in inference inputs")
    backend = backend if backend is not None else StubBackend(config)
    if backend.name not in ("hf", "stub"):
        raise ValueError("backend must identify itself as hf or stub")
    if isinstance(backend, (HFBackend, StubBackend)) and backend.config != config:
        raise ValueError("backend configuration must match the recorded inference configuration")
    config_value = config.model_dump(mode="json")
    started = time.perf_counter()
    batch_method = getattr(backend, "generate_batch", None)
    effective_batch_size = config.batch_size if callable(batch_method) else 1
    manifest: dict[str, Any] = {
        "schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name, "synthetic": backend.name == "stub",
        "evaluation_note": ("Synthetic plumbing only; never report as LLM performance."
                            if backend.name == "stub" else "HF backend selected; verify successful generation and acceptance separately."),
        "model": {"id": config.model_id, "revision": config.revision, "type": "instruction_tuned"},
        "tokenizer": {"id": config.model_id, "revision": config.revision},
        "config": config_value, "config_sha256": _hash(config_value),
        "response_policy": config.response_policy,
        "decoding": {"do_sample": False, "num_beams": 1, "batch_size": effective_batch_size,
                     "seed": config.seed, "max_new_tokens": config.max_new_tokens},
        "python": platform.python_version(), "packages": _versions(),
        "variants": [variant.value for variant in variants], "requests": [], "batches": [],
    }
    predictions: list[dict[str, Any] | None] = []
    completions: dict[int, dict[str, Any]] = {}
    pending = []

    def fail(index: int, error: _InferenceFailure) -> None:
        request = manifest["requests"][index]
        request.update(error_type=error.error_type, error_message=error.error_message)
        if error.causes:
            request["error_causes"] = [
                {"relation": relation, "error_type": error_type, "error_message": message}
                for relation, error_type, message in error.causes
            ]
        if error.causes_truncated:
            request["error_causes_truncated"] = True
        predictions[index] = {
            "pair_id": request["pair_id"], "model_variant": request["model_variant"],
            "predicted_failure_mode": None, "predicted_subtask": None,
            "explanation": f"Inference rejected ({error.error_type}): {error.error_message[:240]}",
            "evidence": [], "status": "error", "backend": backend.name,
        }

    def finish(index: int, request_start: float) -> None:
        request = manifest["requests"][index]
        request.update(status=predictions[index]["status"], latency_s=time.perf_counter() - request_start)
        if index in completions:
            completions[index].update(status=request["status"])
            for key in ("error_type", "error_message", "error_causes", "error_causes_truncated"):
                if key in request:
                    completions[index][key] = request[key]

    for tree in trees:
        for variant in variants:
            index = len(predictions)
            predictions.append(None)
            request: dict[str, Any] = {"pair_id": tree.metadata.pair_id, "model_variant": variant.value,
                                       "input_tokens": None, "output_tokens": None,
                                       "token_count_method": (getattr(backend, "token_count_method", None)
                                                              if isinstance(getattr(backend, "token_count_method", None), str)
                                                              else None)}
            manifest["requests"].append(request)
            request_start = time.perf_counter()
            preparation_error = None
            try:
                prompt = build_prompt(tree, variant, template_dir=template_dir or config.prompt_dir)
                request.update(prompt_sha256=prompt.prompt_sha256, template_sha256=prompt.template_sha256)
                count = backend.count_tokens(prompt)
                if type(count) is not int or count <= 0:
                    raise ValueError("backend must return a positive integer input token count")
                request["input_tokens"] = count
                if count + config.max_new_tokens > config.context_window_tokens:
                    raise ValueError("prompt plus reserved output exceeds context budget; no truncation performed")
                pending.append((index, prompt, count, request_start))
            except Exception as exc:
                preparation_error = exc
            if preparation_error is not None:
                captured = _capture_and_release_failure(preparation_error)
                preparation_error = None
                fail(index, captured)
                finish(index, request_start)
            request["preparation_s"] = time.perf_counter() - request_start

    for offset in range(0, len(pending), effective_batch_size):
        batch = pending[offset:offset + effective_batch_size]
        batch_id = len(manifest["batches"])
        batch_start = time.perf_counter()
        batch_error = None
        try:
            if callable(batch_method):
                results = tuple(batch_method([item[1] for item in batch]))
            else:
                results = (backend.generate(batch[0][1]),)
            if len(results) != len(batch):
                raise ValueError("backend batch result count does not match its input count")
        except Exception as exc:
            batch_error = exc
        if batch_error is not None:
            captured = _capture_and_release_failure(batch_error)
            batch_error = None
            results = (captured,) * len(batch)
        else:
            results = _detach_batch_failures(results)
        batch_duration = time.perf_counter() - batch_start
        batch_record: dict[str, Any] = {
            "batch_id": batch_id, "request_indices": [item[0] for item in batch],
            "batch_size": len(batch), "latency_s": batch_duration,
            "latency_scope": "shared_batch_generation_wall_time",
        }
        manifest["batches"].append(batch_record)
        for (index, prompt, count, request_start), result in zip(batch, results):
            request = manifest["requests"][index]
            request.update(batch_id=batch_id, batch_size=len(batch), batch_latency_s=batch_duration)
            if isinstance(result, _InferenceFailure):
                fail(index, result)
                finish(index, request_start)
                continue
            response_error = None
            try:
                if not isinstance(result, Generation):
                    raise TypeError("backend must return a Generation or an Exception per prompt")
                if not isinstance(result.text, str):
                    raise TypeError("backend completion must be a string")
                # Preserve the exact text even when token accounting or parsing
                # rejects it. No raw response is rewritten into a diagnosis.
                digest = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
                metadata_errors = []
                if type(result.input_tokens) is not int or result.input_tokens != count:
                    metadata_errors.append("input_tokens")
                valid_output_count = type(result.output_tokens) is int and 0 <= result.output_tokens <= config.max_new_tokens
                if not valid_output_count:
                    metadata_errors.append("output_tokens")
                valid_method = isinstance(result.token_count_method, str) and bool(result.token_count_method.strip())
                if not valid_method:
                    metadata_errors.append("token_count_method")
                valid_finish_reason = result.finish_reason is None or isinstance(result.finish_reason, str)
                if not valid_finish_reason:
                    metadata_errors.append("finish_reason")
                completions[index] = {
                    "pair_id": request["pair_id"], "model_variant": request["model_variant"],
                    "backend": backend.name, "text": result.text, "messages": prompt.messages,
                    "completion_sha256": digest, "prompt_sha256": prompt.prompt_sha256,
                    "input_tokens": count, "output_tokens": result.output_tokens if valid_output_count else None,
                    "token_count_method": result.token_count_method if valid_method else None,
                    "finish_reason": result.finish_reason if valid_finish_reason else None,
                    "response_normalization": response_normalization(result.text, config.response_policy),
                }
                request.update(completion_sha256=digest,
                               finish_reason=result.finish_reason if valid_finish_reason else None)
                if metadata_errors:
                    # Never serialize NaN, arbitrary objects, or their reprs from
                    # an invalid backend. Exact response text and trusted input
                    # count remain reviewable alongside the failed field names.
                    completions[index]["metadata_error_fields"] = metadata_errors
                    raise ValueError("backend returned invalid generation metadata: " + ", ".join(metadata_errors))
                request.update(output_tokens=result.output_tokens, token_count_method=result.token_count_method)
                predictions[index] = parse_response(
                    result.text, prompt, request["pair_id"], backend.name,
                    response_policy=config.response_policy,
                )
            except Exception as exc:
                response_error = exc
            if response_error is not None:
                captured = _capture_and_release_failure(response_error)
                response_error = None
                fail(index, captured)
            finish(index, request_start)
        batch_record["status_counts"] = dict(Counter(predictions[item[0]]["status"] for item in batch))
        # A backend may return an invalid object as well as an Exception. Drop
        # all batch-result references before the next memory admission check.
        results, result = (), None
    manifest.update(backend_provenance=backend.provenance(), runtime_s=time.perf_counter() - started,
                    status_counts=dict(Counter(record["status"] for record in predictions)),
                    n_inputs=len(trees), n_predictions=len(predictions), n_completions=len(completions))
    manifest["response_processing"] = {
        "policy": config.response_policy, "raw_text_preserved": True,
        "fenced_requests": sum(record["response_normalization"]["envelope"] == "json_code_fence"
                               for record in completions.values()),
    }
    return InferenceRun(tuple(predictions), manifest, tuple(completions[index] for index in sorted(completions)))
