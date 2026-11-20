"""Failure receipts never keep model-frame allocations alive across batches.

The buffers below are ordinary weak-referenceable Python objects. Disabling
cyclic GC makes the lifetime check deterministic without importing a GPU runtime.
"""

from contextlib import contextmanager
import gc
import json
from pathlib import Path
import weakref

from telemetry_reasoning.models import llm
from telemetry_reasoning.models.llm import Generation, LLMConfig, run_inference
from telemetry_reasoning.parser.divergence_metrics import enrich_paired_bt, load_divergence_config
from telemetry_reasoning.schemas import PairedBehaviorTree


ROOT = Path(__file__).resolve().parents[1]


class Allocation:
    """Stand-in for a device allocation owned by a failed model frame."""


def _tree():
    tree = PairedBehaviorTree.model_validate_json((ROOT / "data/sample/pair_000_bt.json").read_bytes())
    return enrich_paired_bt(tree, load_divergence_config(ROOT / "configs/divergence.yaml"))


def _completion():
    return Generation(json.dumps({
        "predicted_failure_mode": None, "predicted_subtask": None,
        "explanation": "Synthetic cleanup fixture; no diagnosis", "evidence": [], "status": "abstain",
    }), 5, 5, "fixture_counter")


@contextmanager
def _without_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class Backend:
    name = "stub"
    token_count_method = "fixture_counter"

    def count_tokens(self, prompt):
        return 5

    def provenance(self):
        return {"synthetic_cleanup_fixture": True}


def test_raised_batch_error_releases_frames_before_next_generation_without_retry():
    allocations, calls = [], []

    def inner_failure():
        buffer = Allocation()
        allocations.append(weakref.ref(buffer))
        raise ValueError("synthetic allocation cause")

    class FailingBackend(Backend):
        def generate_batch(self, prompts):
            calls.append([prompt.variant.value for prompt in prompts])
            if len(calls) == 1:
                buffer = Allocation()
                allocations.append(weakref.ref(buffer))
                try:
                    inner_failure()
                except ValueError as cause:
                    raise RuntimeError("synthetic CUDA out of memory") from cause
            assert all(reference() is None for reference in allocations), "previous failed batch still owns buffers"
            return tuple(_completion() for _ in prompts)

    with _without_cyclic_gc():
        run = run_inference([_tree()], ["baseline_2", "baseline_3", "main"], LLMConfig(),
                            backend=FailingBackend())
    assert calls == [["baseline_2"], ["baseline_3"], ["main"]]
    assert [row["status"] for row in run.predictions] == ["error", "abstain", "abstain"]
    assert run.manifest["requests"][0]["error_type"] == "RuntimeError"
    assert run.manifest["requests"][0]["error_message"] == "synthetic CUDA out of memory"
    assert run.manifest["requests"][0]["error_causes"] == [{
        "relation": "cause", "error_type": "ValueError", "error_message": "synthetic allocation cause",
    }]
    assert len(run.completions) == 2
    assert {row["model_variant"] for row in run.completions} == {"baseline_3", "main"}
    json.dumps(run.manifest)


def test_returned_per_row_exception_releases_its_frame_before_next_batch():
    allocations, calls = [], []

    def failed_result():
        buffer = Allocation()
        allocations.append(weakref.ref(buffer))
        try:
            raise RuntimeError("synthetic returned backend error")
        except RuntimeError as error:
            return error

    class PerRowBackend(Backend):
        def generate_batch(self, prompts):
            calls.append([prompt.variant.value for prompt in prompts])
            if len(calls) == 1:
                return failed_result(), _completion()
            assert all(reference() is None for reference in allocations), "returned exception retains allocation"
            return tuple(_completion() for _ in prompts)

    with _without_cyclic_gc():
        run = run_inference([_tree()], ["baseline_2", "baseline_3", "main"], LLMConfig(batch_size=2),
                            backend=PerRowBackend())
    assert calls == [["baseline_2", "baseline_3"], ["main"]]
    assert [row["status"] for row in run.predictions] == ["error", "abstain", "abstain"]
    assert run.manifest["requests"][0]["error_message"] == "synthetic returned backend error"
    assert len(run.completions) == 2


def test_preparation_failure_releases_its_frame_and_does_not_generate_rejected_prompt():
    allocations, counts, generated = [], [], []

    class PreparationBackend(Backend):
        def count_tokens(self, prompt):
            counts.append(prompt.variant.value)
            if len(counts) == 1:
                buffer = Allocation()
                allocations.append(weakref.ref(buffer))
                raise llm.ResourceSafetyError("synthetic unchanged admission refusal")
            assert all(reference() is None for reference in allocations)
            return 5

        def generate(self, prompt):
            generated.append(prompt.variant.value)
            return _completion()

    with _without_cyclic_gc():
        run = run_inference([_tree()], ["baseline_2", "baseline_3", "main"], LLMConfig(),
                            backend=PreparationBackend())
    assert counts == ["baseline_2", "baseline_3", "main"]
    assert generated == ["baseline_3", "main"]
    assert run.manifest["requests"][0]["error_type"] == "ResourceSafetyError"
    assert run.manifest["requests"][0]["error_message"] == "synthetic unchanged admission refusal"
    assert run.manifest["requests"][0]["input_tokens"] is None


def test_response_validation_failure_releases_frames_but_preserves_raw_completion(monkeypatch):
    allocations, parses = [], []
    original_parse = llm.parse_response

    def parser(text, prompt, *args, **kwargs):
        parses.append(prompt.variant.value)
        if len(parses) == 1:
            buffer = Allocation()
            allocations.append(weakref.ref(buffer))
            raise ValueError("synthetic strict response rejection")
        assert all(reference() is None for reference in allocations)
        return original_parse(text, prompt, *args, **kwargs)

    class SuccessfulBackend(Backend):
        def generate(self, prompt):
            return _completion()

    monkeypatch.setattr(llm, "parse_response", parser)
    with _without_cyclic_gc():
        run = run_inference([_tree()], ["baseline_2", "baseline_3"], LLMConfig(), backend=SuccessfulBackend())
    assert [row["status"] for row in run.predictions] == ["error", "abstain"]
    assert run.completions[0]["text"] == _completion().text
    assert run.completions[0]["error_type"] == "ValueError"
    assert run.completions[0]["error_message"] == "synthetic strict response rejection"
    assert len(run.completions) == 2


def test_error_cause_receipts_are_bounded_and_do_not_hold_exception_objects():
    class DeepErrorBackend(Backend):
        def generate(self, prompt):
            error = RuntimeError("top-level rejection")
            current = error
            for _ in range(20):
                current.__cause__ = ValueError("x" * 2000)
                current = current.__cause__
            raise error

    run = run_inference([_tree()], ["baseline_2"], LLMConfig(), backend=DeepErrorBackend())
    request = run.manifest["requests"][0]
    assert request["error_type"] == "RuntimeError" and request["error_message"] == "top-level rejection"
    assert len(request["error_causes"]) == 8
    assert request["error_causes_truncated"] is True
    assert all(len(row["error_message"]) == 1000 for row in request["error_causes"])
    assert not run.completions
    json.dumps(run.manifest)


def test_unprintable_backend_exception_remains_an_explicit_error():
    class UnprintableError(RuntimeError):
        def __str__(self):
            raise ValueError("synthetic broken exception formatter")

    class UnprintableBackend(Backend):
        def generate(self, prompt):
            raise UnprintableError()

    run = run_inference([_tree()], ["baseline_2"], LLMConfig(), backend=UnprintableBackend())
    request = run.manifest["requests"][0]
    assert request["error_type"] == "UnprintableError"
    assert request["error_message"] == "<UnprintableError: message unavailable (ValueError)>"
    assert run.predictions[0]["status"] == "error" and not run.completions
