"""Track 5 inference boundary, with an offline stub and a lazy HF adapter.

Only :class:`Prompt` objects reach a backend. Storage labels and pair IDs stay
outside that boundary. The Oct. 30 adapter uses single-request, greedy decoding;
batched throughput, real-model acceptance and training are later milestones.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
from typing import Annotated, Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from telemetry_reasoning.eval.contracts import EvidenceItem
from telemetry_reasoning.models.prompts import ModelVariant, Prompt, build_prompt
from telemetry_reasoning.schemas import FailureMode, PairedBehaviorTree, Subtask, TASK_SUBTASKS, Task


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/llm.yaml"
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class LLMConfig(BaseModel):
    """A separate, strict config; never extends the frozen project YAML.

    Model and tokenizer always use the same instruction-tuned checkpoint and
    immutable revision. Network downloads require an explicit config change.
    Quantization is deliberately unsupported in this initial adapter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model_id: Literal["meta-llama/Llama-3.1-8B-Instruct"] = MODEL_ID
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] = MODEL_REVISION
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    context_window_tokens: Annotated[int, Field(strict=True, gt=0, le=131072)] = 65536
    max_new_tokens: PositiveInt = 512
    seed: Annotated[int, Field(strict=True, ge=0, lt=2**32)] = 723
    local_files_only: Annotated[bool, Field(strict=True)] = True
    quantization: Literal["none"] = "none"
    prompt_dir: Path | None = None

    @model_validator(mode="after")
    def valid_budget(self) -> "LLMConfig":
        if self.max_new_tokens >= self.context_window_tokens:
            raise ValueError("max_new_tokens must leave room for the input prompt")
        return self


def load_llm_config(path: str | Path = DEFAULT_CONFIG_PATH) -> LLMConfig:
    """Load ``llm: {...}``; resolve custom templates relative to this file."""
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"llm"} or not isinstance(value["llm"], dict):
        raise ValueError("LLM config must contain exactly one 'llm' mapping")
    config = LLMConfig.model_validate(value["llm"])
    if config.prompt_dir is not None:
        resolved = (path.parent / config.prompt_dir).resolve()
        config = config.model_copy(update={"prompt_dir": resolved})
    return config


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "transformers", "huggingface-hub", "pydantic", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capabilities(config: LLMConfig) -> dict[str, Any]:
    """Inspect local packages/devices, without opening caches or using the Hub.

    This is a compute preflight, not proof of gated-repository access. No
    credentials, token values, or environment contents are returned.
    """
    result: dict[str, Any] = {
        "model_id": config.model_id, "revision": config.revision,
        "packages": _versions(), "configured_device": config.device,
        "configured_dtype": config.dtype, "local_files_only": config.local_files_only,
        "remote_access": "not_checked", "weights": "not_loaded",
        "cuda_available": False, "mps_available": False,
    }
    if result["packages"]["torch"] is not None:
        try:
            torch = importlib.import_module("torch")
            result["cuda_available"] = bool(torch.cuda.is_available())
            mps = getattr(torch.backends, "mps", None)
            result["mps_available"] = bool(mps and mps.is_available())
        except (ImportError, OSError, RuntimeError) as exc:
            result["runtime_error"] = type(exc).__name__
    return result


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    token_count_method: str = "tokenizer"


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
    """Load the pinned model only on a real request, with no silent truncation.

    Counting a prompt loads only the tokenizer. Unsupported compute settings
    fail explicitly; there is no CPU fallback or automatic quantization.
    """

    name = "hf"
    token_count_method = "tokenizer"

    def __init__(self, config: LLMConfig):
        self.config = config
        self._torch = self._transformers = self._tokenizer = self._model = None

    def _runtime(self) -> None:
        if self._torch is not None:
            return
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError("HF backend requires the optional .[llm] dependencies") from exc
        if self.config.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("configured CUDA device is unavailable")
            if self.config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("configured CUDA device does not support bfloat16")
        if self.config.device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if not mps or not mps.is_available():
                raise RuntimeError("configured MPS device is unavailable")
        self._torch, self._transformers = torch, transformers

    def _encode(self, prompt: Prompt) -> Any:
        self._runtime()
        if self._tokenizer is None:
            self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(
                self.config.model_id, revision=self.config.revision,
                local_files_only=self.config.local_files_only, trust_remote_code=False,
            )
        if not self._tokenizer.chat_template:
            raise ValueError("instruction checkpoint tokenizer must declare a chat template")
        return self._tokenizer.apply_chat_template(
            prompt.messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", truncation=False,
        )

    def count_tokens(self, prompt: Prompt) -> int:
        return int(self._encode(prompt).shape[-1])

    def generate(self, prompt: Prompt) -> Generation:
        encoded = self._encode(prompt)
        count = int(encoded.shape[-1])
        if count + self.config.max_new_tokens > self.config.context_window_tokens:
            raise ValueError("prompt plus reserved output exceeds context budget; no truncation performed")
        if self._model is None:
            self._model = self._transformers.AutoModelForCausalLM.from_pretrained(
                self.config.model_id, revision=self.config.revision,
                local_files_only=self.config.local_files_only, trust_remote_code=False,
                torch_dtype=getattr(self._torch, self.config.dtype),
            ).to(self.config.device)
            self._model.eval()
        native_limit = getattr(self._model.config, "max_position_embeddings", None)
        if native_limit and self.config.context_window_tokens > native_limit:
            raise ValueError("configured context exceeds checkpoint context window")
        inputs = encoded.to(self.config.device)
        self._torch.manual_seed(self.config.seed)
        generation_config = self._transformers.GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=self.config.max_new_tokens,
            eos_token_id=self._model.generation_config.eos_token_id,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        with self._torch.inference_mode():
            generated = self._model.generate(
                input_ids=inputs, attention_mask=self._torch.ones_like(inputs),
                generation_config=generation_config,
            )
        continuation = generated[0, count:]
        text = self._tokenizer.decode(continuation, skip_special_tokens=True)
        return Generation(text, count, int(continuation.shape[-1]), self.token_count_method)

    def provenance(self) -> dict[str, Any]:
        return {"implementation": "transformers_single_greedy_v1",
                "weights_loaded": self._model is not None,
                "tokenizer_loaded": self._tokenizer is not None,
                "token_count_method": self.token_count_method,
                "device": self.config.device, "dtype": self.config.dtype}


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


def parse_response(text: str, prompt: Prompt, pair_id: str, backend: str) -> dict[str, Any]:
    """Require one strict JSON object and citations to available failure rows.

    No repair, label guessing, code-fence extraction, or success-side evidence
    substitution occurs. Unsupported responses become errors in the harness.
    """
    if backend not in ("stub", "hf"):
        raise ValueError("unsupported inference backend")
    response = _ModelResponse.model_validate(json.loads(
        text, parse_constant=_reject_constant, object_pairs_hook=_unique_keys,
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


def run_inference(
    trees: Sequence[PairedBehaviorTree], variants: Sequence[str], config: LLMConfig, *,
    backend: InferenceBackend | None = None, template_dir: str | Path | None = None,
) -> InferenceRun:
    """Run one validated result per pair/variant, preserving errors as records.

    Input/view/context errors never invoke generation. Per-request failures do
    not drop later inputs. Configuration/identity errors reject the whole run.
    """
    variants = tuple(ModelVariant(variant) for variant in variants)
    if not trees or not variants or len(set(variants)) != len(variants):
        raise ValueError("nonempty inputs and unique nonempty variants are required")
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
    manifest: dict[str, Any] = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name, "synthetic": backend.name == "stub",
        "evaluation_note": ("Synthetic plumbing only; never report as LLM performance."
                            if backend.name == "stub" else "HF backend selected; verify successful generation and acceptance separately."),
        "model": {"id": config.model_id, "revision": config.revision, "type": "instruction_tuned"},
        "tokenizer": {"id": config.model_id, "revision": config.revision},
        "config": config_value, "config_sha256": _hash(config_value),
        "decoding": {"do_sample": False, "num_beams": 1, "batch_size": 1,
                     "seed": config.seed, "max_new_tokens": config.max_new_tokens},
        "python": platform.python_version(), "packages": _versions(),
        "variants": [variant.value for variant in variants], "requests": [],
    }
    predictions = []
    for tree in trees:
        for variant in variants:
            request: dict[str, Any] = {"pair_id": tree.metadata.pair_id, "model_variant": variant.value,
                                       "input_tokens": None, "output_tokens": None,
                                       "token_count_method": getattr(backend, "token_count_method", None)}
            request_start = time.perf_counter()
            try:
                prompt = build_prompt(tree, variant, template_dir=template_dir or config.prompt_dir)
                request.update(prompt_sha256=prompt.prompt_sha256, template_sha256=prompt.template_sha256)
                count = backend.count_tokens(prompt)
                if type(count) is not int or count <= 0:
                    raise ValueError("backend must return a positive integer input token count")
                request["input_tokens"] = count
                if count + config.max_new_tokens > config.context_window_tokens:
                    raise ValueError("prompt plus reserved output exceeds context budget; no truncation performed")
                result = backend.generate(prompt)
                if result.input_tokens != count or type(result.output_tokens) is not int or not (
                    0 <= result.output_tokens <= config.max_new_tokens
                ):
                    raise ValueError("backend returned inconsistent input/output token counts")
                request.update(output_tokens=result.output_tokens, token_count_method=result.token_count_method,
                               completion_sha256=hashlib.sha256(result.text.encode("utf-8")).hexdigest())
                prediction = parse_response(result.text, prompt, tree.metadata.pair_id, backend.name)
            except Exception as exc:
                # Exceptions are kept per input. Interrupts/SystemExit still propagate.
                request["error_type"] = type(exc).__name__
                prediction = {"pair_id": tree.metadata.pair_id, "model_variant": variant.value,
                              "predicted_failure_mode": None, "predicted_subtask": None,
                              "explanation": f"Inference rejected ({type(exc).__name__}): {str(exc)[:240]}",
                              "evidence": [], "status": "error", "backend": backend.name}
            request.update(status=prediction["status"], latency_s=time.perf_counter() - request_start)
            manifest["requests"].append(request)
            predictions.append(prediction)
    manifest.update(backend_provenance=backend.provenance(), runtime_s=time.perf_counter() - started,
                    status_counts=dict(Counter(record["status"] for record in predictions)),
                    n_inputs=len(trees), n_predictions=len(predictions))
    return InferenceRun(tuple(predictions), manifest)
