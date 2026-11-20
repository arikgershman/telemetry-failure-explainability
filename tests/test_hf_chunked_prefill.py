"""Opt-in native CPU checks use small random weights and never contact the Hub.

Run with HF_TINY_NATIVE=1. These certify the pinned runtime's chunk/mask/cache
semantics, not the 8B checkpoint's CUDA memory consumption or model accuracy.
"""
import os

import pytest


pytestmark = pytest.mark.skipif(os.environ.get("HF_TINY_NATIVE") != "1",
                                reason="opt-in tiny native Torch/Transformers CPU check")


@pytest.mark.parametrize("chunk_tokens,padding", [(1, False), (4, True), (16, True)])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_native_chunked_prefill_preserves_scores_tokens_and_cache(chunk_tokens, padding, dtype_name):
    import torch
    from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(723)
    config = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         bos_token_id=1, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).to(dtype=getattr(torch, dtype_name)).eval()
    ids = torch.randint(3, 97, (2, 53))
    mask = torch.ones_like(ids)
    if padding:
        mask[0, :7] = 0
        ids[0, :7] = 0
    mlp_widths = []
    hook = model.model.layers[0].mlp.register_forward_pre_hook(lambda module, args: mlp_widths.append(args[0].shape[1]))
    settings = dict(do_sample=False, num_beams=1, max_new_tokens=5, eos_token_id=None, pad_token_id=0,
                    cache_implementation="dynamic", disable_compile=True,
                    output_scores=True, return_dict_in_generate=True)
    try:
        with torch.inference_mode():
            full = model.generate(input_ids=ids, attention_mask=mask, generation_config=GenerationConfig(**settings))
            assert mlp_widths[0] == 53
            mlp_widths.clear()
            chunked = model.generate(input_ids=ids, attention_mask=mask,
                                     generation_config=GenerationConfig(**settings, prefill_chunk_size=chunk_tokens))
    finally:
        hook.remove()
    assert max(mlp_widths) <= chunk_tokens
    assert sum(mlp_widths[: (53 + chunk_tokens - 1) // chunk_tokens]) == 53
    assert torch.equal(full.sequences, chunked.sequences)
    torch.testing.assert_close(torch.stack(full.scores), torch.stack(chunked.scores), rtol=1e-5, atol=1e-6)
    assert full.past_key_values.get_seq_length() == chunked.past_key_values.get_seq_length() == 57


def test_probe_offset_mask_keeps_the_prefix_and_masks_future_tokens():
    import importlib.util
    from pathlib import Path
    import sys
    import torch

    path = Path(__file__).resolve().parents[1] / "scripts/probe_cuda_inference.py"
    spec = importlib.util.spec_from_file_location("native_cuda_probe_masks", path)
    probe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = probe
    spec.loader.exec_module(probe)

    # Exercise the real mask-construction helper on CPU, without making its
    # CUDA runtime available or changing any dtype/shape operation.
    class CpuTorch:
        bfloat16 = torch.bfloat16
        zeros_like = staticmethod(torch.zeros_like)

        @staticmethod
        def zeros(*args, **kwargs):
            kwargs["device"] = "cpu"
            return torch.zeros(*args, **kwargs)

        @staticmethod
        def arange(*args, **kwargs):
            kwargs["device"] = "cpu"
            return torch.arange(*args, **kwargs)

    expected = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    boolean_case = probe.ProbeCase("mask", 2, 5, 32, "bool", False, False)
    float_case = probe.ProbeCase("mask", 2, 5, 32, "bfloat16", False, False)
    _, _, _, boolean_mask = probe._inputs(CpuTorch, boolean_case)
    _, _, _, float_mask = probe._inputs(CpuTorch, float_case)
    assert torch.equal(boolean_mask[0, 0], expected)
    assert torch.equal(torch.isfinite(float_mask[0, 0]), expected)
    assert torch.equal(float_mask[0, 0][expected], torch.zeros(expected.sum(), dtype=torch.bfloat16))
    assert torch.isneginf(float_mask[0, 0][~expected]).all()
