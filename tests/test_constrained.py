"""Constrained generation protocol tests: no checkpoint download or GPU needed."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import sys

import pytest

from telemetry_reasoning.models.constrained import (
    MAX_EXPLANATION_CHARS, MAX_EXPLANATION_TOKENS, ObservedResponseConstraint,
    _safe_vocabulary,
)
from telemetry_reasoning.models.llm import parse_response
from telemetry_reasoning.models.prompts import build_prompt
from telemetry_reasoning.schemas import FailureMode, PairedBehaviorTree, TelemetryFeature

ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    """ASCII plus adversarial tokens and two distinct special EOS tokens."""
    all_special_ids = [0, 135]
    eos_token_id = 0
    chunks = {129: "x" * 512, 130: "\n", 131: 'bad"text', 132: "bad\\text",
              133: "\ufffd", 134: " " * 20, 135: "<EOS2>", 136: "a" * 513}

    def __len__(self): return 137

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) + 1 for char in text]

    def decode(self, ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join("" if skip_special_tokens and token in self.all_special_ids else
                       self.chunks.get(token, "<EOS>" if token == 0 else chr(token - 1))
                       for token in ids)


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


@pytest.fixture
def prompt():
    tree = PairedBehaviorTree.model_validate_json((ROOT / "data/sample/pair_000_bt.json").read_bytes())
    return build_prompt(tree, "baseline_2")


def constraint(prompt, tokenizer, budget=4096):
    return ObservedResponseConstraint(prompt, tokenizer, [0, 135], budget)


def append(constraint, prefix, ids):
    for token in ids:
        assert token in constraint.allowed_tokens(prefix)
        assert all(eos not in constraint.allowed_tokens(prefix) for eos in constraint.eos_ids)
        prefix.append(token)


def compiled_path(c, *, mode="position_error", subtask=None, explanation="Observed deviation.",
                  evidence=None, prose_ids=None, eos=None):
    prefix = []
    def emit(text): append(c, prefix, c._encode(text))
    emit('{"predicted_failure_mode":')
    emit(json.dumps(mode, separators=(",", ":")))
    emit(',"predicted_subtask":')
    emit(json.dumps(None if mode is None else subtask or c.observed_subtasks[0], separators=(",", ":")))
    emit(',"explanation":"')
    append(c, prefix, c._encode(explanation) if prose_ids is None else prose_ids)
    emit('"')
    emit(',"evidence":[')
    if mode is None:
        emit('],"status":"abstain"}')
    else:
        items = evidence or c.evidence_candidates[:1]
        for index, item in enumerate(items):
            if index: emit(',')
            emit(item)
        emit(']')
        emit(',"status":"ok"}')
    assert c.allowed_tokens(prefix) == list(c.eos_ids)
    prefix.append(c.eos_ids[0] if eos is None else eos)
    assert len(prefix) <= c.worst_case_tokens
    text = c.tokenizer.decode(prefix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return prefix, text


@pytest.mark.parametrize("mode", [None] + [mode.value for mode in FailureMode])
@pytest.mark.parametrize("items", [1, 2])
def test_every_mode_and_evidence_count_produces_strict_grounded_json(prompt, tokenizer, mode, items):
    c = constraint(prompt, tokenizer)
    for stage in c.observed_subtasks:
        for eos in c.eos_ids:
            _, text = compiled_path(c, mode=mode, subtask=stage, evidence=c.evidence_candidates[:items], eos=eos)
            record = parse_response(text, prompt, "test", "hf")
            assert set(json.loads(text)) == {"predicted_failure_mode", "predicted_subtask", "explanation", "evidence", "status"}
            assert record["predicted_failure_mode"] == mode
            assert record["status"] == ("ok" if mode is not None else "abstain")
            assert len(record["evidence"]) == (items if mode is not None else 0)


def test_all_failure_features_and_observations_are_available_and_no_reference_values_leak(prompt, tokenizer):
    c = constraint(prompt, tokenizer)
    expected = {
        json.dumps({"subtask": row["subtask"], "timestamp": row["timestamp"],
                    "feature": feature.value, "value": row[feature.value]}, separators=(",", ":"))
        for node in prompt.payload["subtasks"] for row in node["failure_samples"] for feature in TelemetryFeature
    }
    assert set(c.evidence_candidates) == expected
    for item in c.evidence_candidates:
        _, text = compiled_path(c, evidence=[item])
        assert parse_response(text, prompt, "test", "hf")["evidence"] == [json.loads(item)]
    provenance = c.provenance()
    assert provenance["policy"] == "observed_json_v1"
    assert provenance["repair"] is False
    assert provenance["diagnosis_quality"] == "not_guaranteed_by_constraints"


def test_constraints_identical_across_variants_and_insensitive_to_unavailable_metadata(prompt, tokenizer):
    tree = PairedBehaviorTree.model_validate_json((ROOT / "data/sample/pair_000_bt.json").read_bytes())
    from telemetry_reasoning.schemas import DeterministicDivergence
    # Synthetic complete comparisons only exercise the three input views.
    for node in tree.subtasks:
        if node.failure_samples:
            node.divergence_status = "computed"
            node.deterministic_divergence = DeterministicDivergence(
                first_divergence_timestamp=None, max_abs_deltas={f: 0.0 for f in TelemetryFeature})
    constraints = [constraint(build_prompt(tree, variant), tokenizer) for variant in ("baseline_2", "baseline_3", "main")]
    assert len({json.dumps(c.provenance(), sort_keys=True) for c in constraints}) == 1


def test_no_observations_permits_only_model_selected_abstention(prompt, tokenizer):
    payload = json.loads(json.dumps(prompt.payload))
    for node in payload["subtasks"]:
        node["failure_samples"] = []
    p = replace(prompt, payload=payload)
    c = constraint(p, tokenizer)
    assert c.observed_subtasks == () and c.evidence_candidates == ()
    _, text = compiled_path(c, mode=None)
    assert parse_response(text, p, "test", "hf")["status"] == "abstain"
    invalid = list(c._encode('{"predicted_failure_mode":')) + list(c._encode('"position_error"'))
    with pytest.raises(ValueError):
        c.allowed_tokens(invalid)


def test_budget_guard_requires_complete_json_and_eos_before_generation(prompt, tokenizer):
    c = constraint(prompt, tokenizer)
    with pytest.raises(ValueError, match="including complete JSON and EOS"):
        constraint(prompt, tokenizer, c.worst_case_tokens - 1)
    assert constraint(prompt, tokenizer, c.worst_case_tokens).worst_case_tokens == c.worst_case_tokens


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_output_budgets_refused(prompt, tokenizer, budget):
    with pytest.raises(ValueError): constraint(prompt, tokenizer, budget)


@pytest.mark.parametrize("eos", [[], [None], [1.5], [9999], [34], [True]])
def test_invalid_or_nonspecial_eos_refused(prompt, tokenizer, eos):
    with pytest.raises(ValueError): ObservedResponseConstraint(prompt, tokenizer, eos, 4096)


def test_maximum_prose_token_bound_forces_closure_without_truncation(prompt, tokenizer):
    c = constraint(prompt, tokenizer)
    ids, text = compiled_path(c, prose_ids=[ord("x") + 1] * MAX_EXPLANATION_TOKENS)
    assert json.loads(text)["explanation"] == "x" * MAX_EXPLANATION_TOKENS
    assert len(ids) <= c.worst_case_tokens


def test_maximum_prose_character_bound_forces_closure(prompt, tokenizer):
    c = constraint(prompt, tokenizer)
    _, text = compiled_path(c, prose_ids=[129])
    assert len(json.loads(text)["explanation"]) == MAX_EXPLANATION_CHARS


def test_prose_disallows_empty_whitespace_quotes_slashes_controls_unicode_and_specials(prompt, tokenizer):
    c = constraint(prompt, tokenizer)
    prefix = []
    for text in ['{"predicted_failure_mode":', 'null', ',"predicted_subtask":', 'null', ',"explanation":"']:
        append(c, prefix, c._encode(text))
    first = c.allowed_tokens(prefix)
    for invalid in (0, 135, ord('"') + 1, ord('\\') + 1, ord(' ') + 1, 130, 131, 132, 133, 134, 136):
        assert invalid not in first
    append(c, prefix, [ord('x') + 1])
    assert ord('"') + 1 in c.allowed_tokens(prefix)
    assert ord(' ') + 1 in c.allowed_tokens(prefix)
    assert 134 in c.allowed_tokens(prefix)


def test_invalid_history_does_not_corrupt_replay_state(prompt, tokenizer):
    c = constraint(prompt, tokenizer)
    expected = c.allowed_tokens([])
    with pytest.raises(ValueError): c.allowed_tokens([expected[0], 135])
    assert c.allowed_tokens([]) == expected
    ids, text = compiled_path(c, mode=None)
    assert c.allowed_tokens(ids) == list(c.eos_ids)
    assert c.allowed_tokens(ids[:-1]) == list(c.eos_ids)
    with pytest.raises(ValueError): c.allowed_tokens(ids + [ord('x') + 1])
    _, text_again = compiled_path(c, mode=None)
    assert text_again == text


def test_vocabulary_cache_is_bounded_and_offset_storage_is_linear(tokenizer):
    vocab = _safe_vocabulary(tokenizer)
    for remaining in range(513):
        for first in (True, False):
            ids = vocab.choices(remaining, first)
            assert all(len(vocab.text[token]) <= remaining for token in ids)
            if first: assert all(vocab.text[token].strip(' ') for token in ids)
    assert len(vocab._choices_cache) <= 8
    assert len(vocab.ids) <= len(tokenizer)
    assert len(vocab.offsets) == 513


def test_bad_literal_tokenizer_roundtrip_is_refused(prompt):
    class Broken(CharacterTokenizer):
        def encode(self, text, *, add_special_tokens=False):
            return [ord('x') + 1]
    with pytest.raises(ValueError, match="round-trip"):
        constraint(prompt, Broken())


def test_unknown_or_inconsistent_stage_and_nonfinite_values_refused(prompt, tokenizer):
    for mutation in ("stage", "nonfinite", "missing_feature"):
        payload = json.loads(json.dumps(prompt.payload))
        row = next(node["failure_samples"][0] for node in payload["subtasks"] if node["failure_samples"])
        if mutation == "stage": row["subtask"] = "not_a_stage"
        elif mutation == "nonfinite": row["gripper_force"] = float("nan")
        else: del row["joint_torques"]
        with pytest.raises((ValueError, KeyError)):
            constraint(replace(prompt, payload=payload), tokenizer)


@pytest.mark.parametrize("strategy", ["smallest", "largest", "random"])
def test_adversarial_greedy_selection_always_finishes_valid_json(prompt, tokenizer, strategy):
    c = constraint(prompt, tokenizer)
    rng = random.Random(723)
    for _ in range(10):
        prefix = []
        for _ in range(c.worst_case_tokens):
            allowed = c.allowed_tokens(prefix)
            token = min(allowed) if strategy == "smallest" else max(allowed) if strategy == "largest" else rng.choice(allowed)
            prefix.append(token)
            if token in c.eos_ids: break
        else: pytest.fail("constraint did not force complete JSON and EOS within its bound")
        text = tokenizer.decode(prefix, skip_special_tokens=True)
        record = parse_response(text, prompt, "test", "hf")
        assert record["status"] in ("ok", "abstain")


def test_real_pinned_tokenizer_on_all_eight_fresh_pairs():
    """Optional local artifact test; no weight loading or network access."""
    directory = ROOT / "data/raw/tokenizers/llama31-8b-0e9e39f"
    trees = sorted((ROOT / "runs/alienware-fresh-smoke-v2-20261119/tasks").glob("*/bt/*.json"))
    if not directory.exists() or len(trees) != 8:
        pytest.skip("local pinned tokenizer and eight-pair CPU handoff unavailable")
    from telemetry_reasoning.experiments import load_cpu_tokenizer
    tokenizer, _ = load_cpu_tokenizer(directory)
    for path in trees:
        tree = PairedBehaviorTree.model_validate_json(path.read_bytes())
        constraints = [ObservedResponseConstraint(build_prompt(tree, variant), tokenizer,
                                                  tokenizer.eos_token_id, 512)
                       for variant in ("baseline_2", "baseline_3", "main")]
        assert len({json.dumps(c.provenance(), sort_keys=True) for c in constraints}) == 1
        c = constraints[0]
        p = build_prompt(tree, "baseline_2")
        longest = max(c.evidence_candidates, key=lambda value: len(c._encode(value)))
        for mode in [None] + [mode.value for mode in FailureMode]:
            _, text = compiled_path(c, mode=mode, evidence=[longest, longest],
                                    prose_ids=[tokenizer.encode('x', add_special_tokens=False)[0]] * 128)
            parse_response(text, p, "test", "hf")
        assert c.worst_case_tokens <= 512


def test_real_transformers_tiny_random_cpu_generate_callback(prompt, tokenizer):
    """The real HF callback, using random tiny weights and a short fake input.

    This tests the actual public generation API only. It is not an 8B model,
    CUDA memory test, or diagnosis-quality test.
    """
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM
    import torch
    c = constraint(prompt, tokenizer)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=8,
                                        intermediate_size=16, num_hidden_layers=1,
                                        num_attention_heads=1, num_key_value_heads=1,
                                        max_position_embeddings=4096, bos_token_id=1,
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id))
    model.eval()
    source = torch.tensor([[ord('x') + 1]])
    callbacks = []
    def allowed(batch, row):
        assert batch == 0
        ids = row[1:].tolist()
        callbacks.append(len(ids))
        return c.allowed_tokens(ids)
    with torch.inference_mode():
        output = model.generate(source, attention_mask=torch.ones_like(source), do_sample=False,
                                max_new_tokens=c.worst_case_tokens, prefix_allowed_tokens_fn=allowed,
                                eos_token_id=list(c.eos_ids), pad_token_id=tokenizer.eos_token_id)
    ids = output[0, 1:].tolist()
    assert ids[-1] in c.eos_ids and callbacks == list(range(len(ids)))
    parse_response(tokenizer.decode(ids, skip_special_tokens=True), prompt, "test", "hf")


def evidence_prefix(c):
    prefix = []
    for text in ['{"predicted_failure_mode":', '"position_error"', ',"predicted_subtask":',
                 json.dumps(c.observed_subtasks[0]), ',"explanation":"', 'Observed', '"', ',"evidence":[']:
        append(c, prefix, c._encode(text))
    return prefix


@pytest.mark.parametrize("mutation", ["unknown_feature", "wrong_shape", "invented_value", "invented_timestamp", "unobserved_stage"])
def test_reported_invalid_citations_are_unreachable(prompt, tokenizer, mutation):
    c = constraint(prompt, tokenizer)
    item = json.loads(next(value for value in c.evidence_candidates
                           if json.loads(value)["feature"] == "joint_velocities"))
    if mutation == "unknown_feature": item["feature"] = "ee_vel"
    elif mutation == "wrong_shape": item["value"] = item["value"][0]
    elif mutation == "invented_value": item["value"] = [999999.0] * len(item["value"])
    elif mutation == "invented_timestamp": item["timestamp"] = 999999.0
    else: item["subtask"] = "move_along_path"
    prefix = evidence_prefix(c)
    invalid = c._encode(json.dumps(item, separators=(",", ":")))
    with pytest.raises(ValueError, match="violates observed JSON"):
        c.allowed_tokens(prefix + list(invalid))


def test_third_citation_and_extra_fields_are_unreachable(prompt, tokenizer):
    c = constraint(prompt, tokenizer)
    prefix = evidence_prefix(c)
    append(c, prefix, c._encode(c.evidence_candidates[0]))
    append(c, prefix, c._encode(','))
    append(c, prefix, c._encode(c.evidence_candidates[0]))
    with pytest.raises(ValueError): c.allowed_tokens(prefix + list(c._encode(',')))
    _, text = compiled_path(c)
    ids = []
    # Follow the actual segmented path rather than re-tokenizing the complete
    # object, which is intentionally a different canonical BPE tokenization.
    ids, _ = compiled_path(c)
    with pytest.raises(ValueError):
        c.allowed_tokens(ids[:-1] + list(c._encode(',"extra":true')))


def test_compiler_rejects_duplicate_or_prefix_ambiguous_token_paths(prompt, tokenizer):
    c = constraint(prompt, tokenizer)
    for choices in [[("a", 1), ("ab", 2)], [("ab", 1), ("a", 2)], [("a", 1), ("a", 2)]]:
        with pytest.raises(ValueError, match="prefix-free"):
            c._choice("bad", choices)
