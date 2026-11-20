"""New HF execution profile: real tiny CPU inference and isolated CUDA doubles.

No public checkpoint, network access, GPU or production memory override is used.
The host snapshot is patched only for these tiny/fake model unit fixtures.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from telemetry_reasoning.models import llm
from telemetry_reasoning.models.llm import HFBackend, LLMConfig, MODEL_REVISION, parse_response
from telemetry_reasoning.models.prompts import build_prompt
from telemetry_reasoning.schemas import PairedBehaviorTree
from test_constrained import CharacterTokenizer
from test_llm import fake_hf_modules

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prompt():
    tree = PairedBehaviorTree.model_validate_json((ROOT / "data/sample/pair_000_bt.json").read_bytes())
    # Fewer candidate values make the ASCII-only fixture quick. This is a test
    # view, never a transformation applied to real experiment input.
    for node in tree.subtasks:
        node.failure_samples = node.failure_samples[:1]
    return build_prompt(tree, "baseline_2")


@pytest.fixture
def tiny_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    class Tokenizer(CharacterTokenizer):
        pad_token_id = 0
        eos_token = "<EOS>"
        chat_template = "tiny-test-only-not-a-production-chat-template"
        _commit_hash = MODEL_REVISION
        padding_side = "left"

        def apply_chat_template(self, conversations, **kwargs):
            assert len(conversations) == 1
            assert kwargs["truncation"] is False
            assert kwargs["padding"] is True
            ids = torch.full((1, 17), ord('x') + 1, dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    tokenizer = Tokenizer()
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
                         num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
                         max_position_embeddings=8192, bos_token_id=1,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    config._commit_hash = MODEL_REVISION
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config)
    model.eval()
    # Zero logits create adversarial ties: the real generation engine must pick
    # the smallest *allowed* ID, including a nonnull label instead of abstaining.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    calls = {"model_load": [], "tokenizer_load": [], "generate": [], "mlp_widths": []}
    model.model.layers[0].mlp.register_forward_pre_hook(
        lambda _module, args: calls["mlp_widths"].append(int(args[0].shape[1])))
    generate = model.generate
    def recording_generate(*args, **kwargs):
        calls["generate"].append(kwargs)
        return generate(*args, **kwargs)
    monkeypatch.setattr(model, "generate", recording_generate)
    def load_model(model_id, **kwargs):
        calls["model_load"].append((model_id, kwargs))
        return model
    def load_tokenizer(model_id, **kwargs):
        calls["tokenizer_load"].append((model_id, kwargs))
        return tokenizer
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(llm, "_host_resources", lambda: llm.ResourceSnapshot(64 * llm.GIB, "tiny_fixture", 56 * llm.GIB))
    return torch, tokenizer, model, calls


def cpu_config(**changes):
    values = dict(device="cpu", dtype="float32", context_window_tokens=2048,
                  max_new_tokens=1024, batch_size=1, decoding_constraint="observed_json_v1",
                  prefill_chunk_size=8, attention_implementation="sdpa", response_policy="strict_json_v1")
    values.update(changes)
    return LLMConfig(**values)


def test_real_hf_backend_chunked_prefill_constrained_decode_and_model_reuse(prompt, tiny_runtime):
    torch, tokenizer, model, calls = tiny_runtime
    config = cpu_config()
    backend = HFBackend(config)
    for attempt in range(2):
        start = len(calls["mlp_widths"])
        result = backend.generate(prompt)
        record = parse_response(result.text, prompt, "test", "hf")
        assert record["status"] == "ok" and record["evidence"]
        assert result.finish_reason == "eos" and result.input_tokens == 17
        assert result.output_tokens < config.max_new_tokens
        widths = calls["mlp_widths"][start:]
        assert widths[:3] == [8, 8, 1]
        assert max(widths) <= 8
        assert len(calls["generate"]) == attempt + 1
        invocation = calls["generate"][-1]
        generation = invocation["generation_config"]
        assert callable(invocation["prefix_allowed_tokens_fn"])
        assert generation.prefill_chunk_size == 8
        assert generation.cache_implementation == "dynamic"
        assert generation.disable_compile is True and generation.use_cache is True
        assert generation.do_sample is False and generation.num_beams == 1
    assert len(calls["tokenizer_load"]) == len(calls["model_load"]) == 1
    model_id, kwargs = calls["model_load"][0]
    assert model_id == config.model_id
    assert kwargs["revision"] == MODEL_REVISION
    assert kwargs["torch_dtype"] == torch.float32
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
    provenance = backend.provenance()
    assert provenance["implementation"] == "transformers_batched_greedy_v3"
    assert provenance["execution_policy"] == llm.execution_policy(config)
    assert provenance["weights_loaded"] and provenance["tokenizer_loaded"]
    assert provenance["model"]["resolved_revision"] == MODEL_REVISION
    assert provenance["dtype"] == "float32"
    assert len(provenance["constraints"]) == 2
    assert all(item["policy"] == "observed_json_v1" for item in provenance["constraints"])
    assert provenance["request_memory"] == []  # CUDA measurements are never invented on CPU.


def test_compiler_budget_refusal_precedes_model_loading(prompt, tiny_runtime):
    _, _, _, calls = tiny_runtime
    backend = HFBackend(cpu_config(max_new_tokens=64))
    with pytest.raises(ValueError, match="including complete JSON and EOS"):
        backend.generate(prompt)
    assert len(calls["tokenizer_load"]) == 1
    assert calls["model_load"] == calls["generate"] == []


@pytest.mark.parametrize("changes", [
    {"decoding_constraint": "repair"},
    {"decoding_constraint": "observed_json_v1", "batch_size": 2},
    {"prefill_chunk_size": 0}, {"prefill_chunk_size": 2049}, {"prefill_chunk_size": True},
    {"prefill_chunk_size": 8, "attention_implementation": "auto"},
    {"prefill_chunk_size": 8, "attention_implementation": "sdpa", "device": "mps"},
    {"prefill_chunk_size": 8, "attention_implementation": "sdpa", "device": "cuda", "fused_attention_only": False},
    {"attention_implementation": "eager"},
    {"fused_attention_only": True, "device": "cpu", "attention_implementation": "sdpa"},
    {"fused_attention_only": True, "device": "cuda", "attention_implementation": "auto"},
    {"fused_attention_only": 1},
])
def test_unsafe_or_unknown_profile_configuration_rejected(changes):
    with pytest.raises(ValueError): LLMConfig(**changes)


def fake_cuda_profile(monkeypatch, prompt, *, attention_match=True, support_chunking=True):
    calls = fake_hf_modules(monkeypatch, cuda_available=True)
    config = LLMConfig(device="cuda", dtype="bfloat16", attention_implementation="sdpa",
                       prefill_chunk_size=8, fused_attention_only=True, max_new_tokens=8)
    backend = HFBackend(config)
    backend.count_tokens(prompt)
    runtime = backend._torch
    runtime.cuda.reset_peak_memory_stats = lambda: calls.setdefault("peak_reset", []).append(True)
    runtime.cuda.synchronize = lambda: calls.setdefault("synchronized", []).append(True)
    runtime.cuda.max_memory_allocated = lambda: 7 * llm.GIB
    runtime.cuda.max_memory_reserved = lambda: 8 * llm.GIB
    if support_chunking:
        backend._transformers.GenerationConfig = lambda **kwargs: SimpleNamespace(**{"prefill_chunk_size": None, **kwargs})
    loader = backend._transformers.AutoModelForCausalLM.from_pretrained
    state = {"active": False}
    def load(*args, **kwargs):
        model = loader(*args, **kwargs)
        model.config._attn_implementation = "sdpa" if attention_match else "eager"
        original_generate = model.generate
        def generate(**values):
            assert state["active"] is True, "model execution escaped fused-only SDPA context"
            return original_generate(**values)
        model.generate = generate
        return model
    backend._transformers.AutoModelForCausalLM.from_pretrained = load
    @contextmanager
    def kernel(backends):
        calls.setdefault("sdpa_backends", []).append(backends)
        state["active"] = True
        try: yield
        finally: state["active"] = False
    attention = SimpleNamespace(sdpa_kernel=kernel,
        SDPBackend=SimpleNamespace(FLASH_ATTENTION="flash", EFFICIENT_ATTENTION="efficient",
                                   CUDNN_ATTENTION="cudnn", MATH="forbidden_math"))
    original = llm.importlib.import_module
    monkeypatch.setattr(llm.importlib, "import_module", lambda name, *a, **kw:
                        attention if name == "torch.nn.attention" else original(name, *a, **kw))
    return backend, calls, state


def test_cuda_execution_explicitly_excludes_math_fallback_and_records_cache_peak_policy(prompt, monkeypatch):
    backend, calls, state = fake_cuda_profile(monkeypatch, prompt)
    backend.generate(prompt)
    assert calls["sdpa_backends"] == [["flash", "efficient", "cudnn"]]
    assert "forbidden_math" not in calls["sdpa_backends"][0]
    assert state["active"] is False
    generation = calls["generate"][0]["generation_config"]
    assert generation.cache_implementation == "dynamic"
    assert generation.prefill_chunk_size == 8
    assert generation.disable_compile is True and generation.use_cache is True
    assert calls["models"][0][1]["attn_implementation"] == "sdpa"
    assert calls["models"][0][1]["torch_dtype"] == "fake-bfloat16"
    assert calls["peak_reset"] == [True] and calls["synchronized"] == [True]
    provenance = backend.provenance()
    assert provenance["request_memory"] == [{
        "prompt_sha256": [prompt.prompt_sha256], "input_tokens": [3], "prefill_chunk_size": 8,
        "kv_cache_estimate_bytes": 2 * 32 * 8 * 128 * 2 * 11,
        "completed": True, "peak_allocated_bytes": 7 * llm.GIB, "peak_reserved_bytes": 8 * llm.GIB,
    }]
    assert provenance["execution_policy"]["fused_attention_only"] is True


def test_failed_cuda_call_retains_peak_evidence_and_exits_kernel_context(prompt, monkeypatch):
    backend, calls, state = fake_cuda_profile(monkeypatch, prompt)
    backend._load_model()
    def fail(**kwargs): raise RuntimeError("synthetic failed generation")
    backend._model.generate = fail
    with pytest.raises(RuntimeError, match="synthetic failed generation"):
        backend.generate(prompt)
    assert state["active"] is False
    memory = backend.provenance()["request_memory"][0]
    assert memory["completed"] is False and memory["peak_allocated_bytes"] == 7 * llm.GIB
    assert calls["peak_reset"] == [True]
    assert "synchronized" not in calls


def test_missing_native_chunking_support_refuses_before_weights(prompt, monkeypatch):
    backend, calls, _ = fake_cuda_profile(monkeypatch, prompt, support_chunking=False)
    with pytest.raises(RuntimeError, match="does not support native chunked prefill"):
        backend.generate(prompt)
    assert calls["models"] == calls["generate"] == []


def test_attention_mismatch_remains_refused_on_every_attempt(prompt, monkeypatch):
    backend, calls, _ = fake_cuda_profile(monkeypatch, prompt, attention_match=False)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="attention differs"):
            backend.generate(prompt)
    assert calls["generate"] == []


@pytest.mark.parametrize("suffix", ["", "{}", '{"predicted_failure_mode":null,"predicted_subtask":null,"explanation":"Input error.","evidence":[],"status":"error"}'])
def test_backend_independently_rejects_emitted_ids_that_bypass_constraint_callback(prompt, tiny_runtime, monkeypatch, suffix):
    torch, tokenizer, model, _ = tiny_runtime
    def invalid_generate(**kwargs):
        assert callable(kwargs["prefix_allowed_tokens_fn"])
        # Simulate an engine bug or wrong adapter that accepts the callback but
        # emits an early EOS, malformed object or an out-of-policy valid object.
        emitted = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
        return torch.cat([kwargs["input_ids"], torch.tensor([emitted])], dim=1)
    monkeypatch.setattr(model, "generate", invalid_generate)
    with pytest.raises(ValueError):
        HFBackend(cpu_config()).generate(prompt)


def test_constrained_complete_object_without_terminal_eos_is_rejected_not_repaired(prompt, tiny_runtime, monkeypatch):
    from telemetry_reasoning.models.constrained import ObservedResponseConstraint
    from test_constrained import compiled_path
    torch, tokenizer, model, _ = tiny_runtime
    c = ObservedResponseConstraint(prompt, tokenizer, tokenizer.eos_token_id, 1024)
    complete, text = compiled_path(c, mode=None)
    assert parse_response(text, prompt, "test", "hf")["status"] == "abstain"
    def no_eos_generate(**kwargs):
        return torch.cat([kwargs["input_ids"], torch.tensor([complete[:-1]])], dim=1)
    monkeypatch.setattr(model, "generate", no_eos_generate)
    with pytest.raises(ValueError, match="EOS|eos|terminat"):
        HFBackend(cpu_config()).generate(prompt)
