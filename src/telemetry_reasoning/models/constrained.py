"""Versioned, model-selected JSON constrained decoding from observed inputs.

This module never reads labels or imports a model runtime. It constrains *new*
model generation; it does not repair a saved completion or choose a diagnosis.
The finite choices contain every available failure-side observation and no
success-side samples. Free explanation text remains a model claim that needs
scientific evaluation, even when the resulting JSON and citations are valid.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from typing import Any, Sequence

from telemetry_reasoning.eval.contracts import EvidenceItem
from telemetry_reasoning.models.prompts import Prompt
from telemetry_reasoning.schemas import FailureMode, TASK_SUBTASKS, Task, TelemetryFeature

POLICY = "observed_json_v1"
MAX_EXPLANATION_TOKENS = 128
MAX_EXPLANATION_CHARS = 512
MAX_EVIDENCE_ITEMS = 2


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass
class _Node:
    children: dict[int, "_Node"] = field(default_factory=dict)
    terminal: bool = False
    value: Any = None


@dataclass
class _SafeVocabulary:
    text: dict[int, str]
    ids: tuple[int, ...]
    first_ids: tuple[int, ...]
    offsets: tuple[int, ...]
    first_offsets: tuple[int, ...]
    sha256: str
    _choices_cache: OrderedDict = field(default_factory=OrderedDict)

    def choices(self, remaining: int, first: bool) -> tuple[int, ...]:
        key = (remaining, first)
        if key not in self._choices_cache:
            ids, offsets = (self.first_ids, self.first_offsets) if first else (self.ids, self.offsets)
            self._choices_cache[key] = ids[:offsets[remaining]]
            # Bound retained tuple references. Materializing every possible
            # character budget would multiply a 128K vocabulary by 1,024.
            if len(self._choices_cache) > 8:
                self._choices_cache.popitem(last=False)
        self._choices_cache.move_to_end(key)
        return self._choices_cache[key]


@lru_cache(maxsize=4)
def _safe_vocabulary(tokenizer: Any) -> _SafeVocabulary:
    """Only complete printable ASCII token strings can appear inside prose.

    Quotes/backslashes, Unicode byte fragments and special tokens cannot break
    out of the JSON string. The cache is tied to the exact tokenizer instance.
    """
    special = set(tokenizer.all_special_ids)
    text = {}
    for token in range(len(tokenizer)):
        if token in special:
            continue
        decoded = tokenizer.decode([token], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if (decoded and len(decoded) <= MAX_EXPLANATION_CHARS and
                all(32 <= ord(char) <= 126 and char not in '"\\' for char in decoded)):
            text[token] = decoded
    if not any(value.strip(" ") for value in text.values()):
        raise ValueError("tokenizer has no printable ASCII explanation vocabulary")
    ids = tuple(sorted(text, key=lambda token: (len(text[token]), token)))
    first_ids = tuple(token for token in ids if text[token].strip(" "))
    def offsets_for(values):
        result, index = [], 0
        for length in range(MAX_EXPLANATION_CHARS + 1):
            while index < len(values) and len(text[values[index]]) <= length:
                index += 1
            result.append(index)
        return tuple(result)
    return _SafeVocabulary(text, ids, first_ids, offsets_for(ids),
                           offsets_for(first_ids), _digest(list(text.items())))


class ObservedResponseConstraint:
    """A finite JSON state machine for ``prefix_allowed_tokens_fn``.

    ``allowed_tokens`` receives completion IDs only (exclude the prompt). All
    fields, including abstention, remain model-selected. At most two evidence
    citations and 128 explanation tokens are allowed in this explicit protocol.
    The constructor checks a conservative *complete output plus EOS* bound before
    generation, so reaching max_new_tokens cannot leave an unfinished object.
    A complete object still need not be a correct diagnosis or explanation.
    """

    def __init__(self, prompt: Prompt, tokenizer: Any, eos_ids: Sequence[int] | int,
                 max_new_tokens: int):
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        eos = [eos_ids] if type(eos_ids) is int else list(eos_ids)
        if not eos or any(type(value) is not int or not 0 <= value < len(tokenizer) for value in eos):
            raise ValueError("EOS IDs must be valid tokenizer token IDs")
        if any(value not in tokenizer.all_special_ids for value in eos):
            raise ValueError("EOS IDs must be declared tokenizer special tokens")
        self.eos_ids = tuple(sorted(set(eos)))
        self.tokenizer = tokenizer
        self._special = set(tokenizer.all_special_ids)
        self._safe = _safe_vocabulary(tokenizer)
        self._encodings: dict[str, tuple[int, ...]] = {}
        self._tries: dict[str, _Node] = {}
        self._max_lengths: dict[str, int] = {}

        allowed_stages = {stage.value for stage in TASK_SUBTASKS[Task(prompt.payload["task"])]}
        observed, candidates = [], []
        for node in prompt.payload["subtasks"]:
            for row in node["failure_samples"]:
                stage = row["subtask"]
                if stage != node["subtask"] or stage not in allowed_stages:
                    raise ValueError("failure sample has an inconsistent or unknown task stage")
                if stage not in observed:
                    observed.append(stage)
                for feature in TelemetryFeature:
                    item = EvidenceItem.model_validate({
                        "subtask": stage, "timestamp": row["timestamp"],
                        "feature": feature.value, "value": row[feature.value],
                    }).model_dump(mode="json")
                    candidates.append(_json(item))
        # Stable exact-byte deduplication does not remove any distinct observed
        # feature/value tuple. Reference-side and divergence values never enter.
        self.observed_subtasks = tuple(observed)
        self.evidence_candidates = tuple(dict.fromkeys(candidates))
        self._choice("prefix", [("{\"predicted_failure_mode\":", None)])
        modes = [( _json(mode.value), mode.value) for mode in FailureMode] if observed else []
        self._choice("mode", modes + [("null", None)])
        self._choice("subtask_prefix", [(',"predicted_subtask":', None)])
        self._choice("subtask", [(_json(stage), stage) for stage in observed]) if observed else None
        self._choice("null", [("null", None)])
        self._choice("explanation_prefix", [(',"explanation":"', None)])
        quote = self._encode('"')
        if len(quote) != 1:
            raise ValueError("observed_json_v1 requires one-token JSON quote encoding")
        self._quote = quote[0]
        self._choice("evidence_prefix", [(',"evidence":[', None)])
        if candidates:
            self._choice("evidence", [(item, None) for item in self.evidence_candidates])
        self._choice("separator", [(",", "second"), ("]", "close")])
        self._choice("close_evidence", [("]", None)])
        self._choice("ok_suffix", [(',"status":"ok"}', None)])
        self._choice("abstain_suffix", [('],"status":"abstain"}', None)])

        shared = sum(self._max_lengths[name] for name in
                     ("prefix", "mode", "subtask_prefix", "explanation_prefix", "evidence_prefix"))
        # 128 prose tokens, the mandatory closing quote and the terminal EOS.
        shared += MAX_EXPLANATION_TOKENS + 2
        abstain = shared + self._max_lengths["null"] + self._max_lengths["abstain_suffix"]
        nonnull = (shared + self._max_lengths.get("subtask", 0) +
                   MAX_EVIDENCE_ITEMS * self._max_lengths.get("evidence", 0) +
                   self._max_lengths["separator"] + self._max_lengths["close_evidence"] +
                   self._max_lengths["ok_suffix"])
        self.worst_case_tokens = max(abstain, nonnull if candidates else abstain)
        if self.worst_case_tokens > max_new_tokens:
            raise ValueError(f"observed_json_v1 needs at least {self.worst_case_tokens} output tokens "
                             f"including complete JSON and EOS; configured {max_new_tokens}")
        self._reset()

    def _encode(self, text: str) -> tuple[int, ...]:
        if text not in self._encodings:
            ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
            if (not ids or any(type(token) is not int or token in self._special or
                               not 0 <= token < len(self.tokenizer) for token in ids)):
                raise ValueError("constraint literal produced empty, special or invalid token IDs")
            decoded = self.tokenizer.decode(ids, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
            pieces = "".join(self.tokenizer.decode([token], skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False) for token in ids)
            if decoded != text or pieces != text:
                raise ValueError("constraint tokenization does not round-trip as concatenated ASCII")
            self._encodings[text] = ids
        return self._encodings[text]

    def _choice(self, name: str, choices: list[tuple[str, Any]]) -> None:
        root = _Node()
        max_length = 0
        for text, value in choices:
            ids = self._encode(text)
            max_length = max(max_length, len(ids))
            node = root
            for token in ids:
                if node.terminal:
                    raise ValueError("constraint alternatives must have prefix-free token encodings")
                node = node.children.setdefault(token, _Node())
            if node.children or node.terminal:
                raise ValueError("constraint alternatives must have distinct prefix-free encodings")
            node.terminal, node.value = True, value
        if not root.children:
            raise ValueError("constraint choice is empty")
        self._tries[name], self._max_lengths[name] = root, max_length

    def _reset(self) -> None:
        self._prefix: tuple[int, ...] = ()
        self._phase, self._node = "prefix", self._tries["prefix"]
        self._mode: str | None = None
        self._prose_tokens = self._prose_chars = 0
        self._evidence_count = 0

    def _enter(self, phase: str) -> None:
        self._phase = phase
        self._node = self._tries.get(phase)

    def _advance(self, token: int) -> None:
        if self._phase == "done":
            if token not in self.eos_ids:
                raise ValueError("completion contains non-EOS tokens after its JSON object")
            return
        if self._phase == "prose":
            if token == self._quote and self._prose_tokens:
                self._enter("evidence_prefix")
            elif (self._prose_tokens < MAX_EXPLANATION_TOKENS and token in self._safe.text and
                  len(self._safe.text[token]) <= MAX_EXPLANATION_CHARS - self._prose_chars and
                  (self._prose_tokens or self._safe.text[token].strip(" "))):
                self._prose_tokens += 1
                self._prose_chars += len(self._safe.text[token])
            else:
                raise ValueError("completion token violates bounded JSON explanation constraints")
            return
        if self._node is None or token not in self._node.children:
            raise ValueError("completion token violates observed JSON constraints")
        self._node = self._node.children[token]
        if not self._node.terminal:
            return
        phase, value = self._phase, self._node.value
        if phase == "prefix": self._enter("mode")
        elif phase == "mode":
            self._mode = value
            self._enter("subtask_prefix")
        elif phase == "subtask_prefix": self._enter("null" if self._mode is None else "subtask")
        elif phase in ("subtask", "null"): self._enter("explanation_prefix")
        elif phase == "explanation_prefix": self._enter("prose")
        elif phase == "evidence_prefix": self._enter("abstain_suffix" if self._mode is None else "evidence")
        elif phase == "evidence":
            self._evidence_count += 1
            self._enter("separator" if self._evidence_count < MAX_EVIDENCE_ITEMS else "close_evidence")
        elif phase == "separator": self._enter("evidence" if value == "second" else "ok_suffix")
        elif phase == "close_evidence": self._enter("ok_suffix")
        elif phase in ("ok_suffix", "abstain_suffix"): self._enter("done")
        else: raise AssertionError(f"unhandled constraint phase {phase}")

    def _prose_choices(self) -> tuple[int, ...]:
        if self._prose_tokens >= MAX_EXPLANATION_TOKENS:
            return ()
        remaining = MAX_EXPLANATION_CHARS - self._prose_chars
        return self._safe.choices(remaining, not self._prose_tokens)

    def allowed_tokens(self, generated_ids: Sequence[int]) -> list[int]:
        """Return valid next IDs for a completion prefix; reject invalid history.

        A cached previous prefix makes ordinary autoregressive calls linear.
        Rewinding/replacing a prefix is supported for tests or callback replays.
        """
        ids = tuple(generated_ids)
        if any(type(token) is not int for token in ids):
            raise ValueError("completion prefix must contain integer token IDs")
        if ids[:len(self._prefix)] != self._prefix:
            self._reset()
        try:
            for token in ids[len(self._prefix):]:
                self._advance(token)
        except ValueError:
            self._reset()
            raise
        self._prefix = ids
        if self._phase == "done":
            return list(self.eos_ids)
        if self._phase == "prose":
            return list(self._prose_choices()) + ([self._quote] if self._prose_tokens else [])
        assert self._node is not None and self._node.children
        return list(self._node.children)

    def provenance(self) -> dict[str, Any]:
        return {
            "policy": POLICY, "mode_selection": "all_canonical_modes_or_abstain",
            "observed_subtasks": list(self.observed_subtasks),
            "evidence_source": "all_exact_observed_failure_sample_feature_values",
            "evidence_candidates": len(self.evidence_candidates),
            "evidence_candidates_sha256": _digest(self.evidence_candidates),
            "maximum_evidence_items": MAX_EVIDENCE_ITEMS,
            "maximum_explanation_tokens": MAX_EXPLANATION_TOKENS,
            "maximum_explanation_characters": MAX_EXPLANATION_CHARS,
            "explanation_alphabet": "printable_ascii_without_quote_or_backslash",
            "safe_explanation_vocabulary_sha256": self._safe.sha256,
            "worst_case_tokens_including_eos": self.worst_case_tokens,
            "repair": False, "diagnosis_quality": "not_guaranteed_by_constraints",
        }
