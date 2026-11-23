"""Tests for the prediction contract, metrics, and the run_baseline/evaluate scripts.

All data here is synthetic and built in this module. A handful of pairs checks
that the arithmetic and plumbing are right; it says nothing about how well any
model performs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from telemetry_reasoning.eval.contracts import (
    BACKENDS,
    DATASET_FAILURE_MODES,
    FAILURE_MODES,
    FEATURE_SHAPES,
    MODEL_VARIANTS,
    SUBTASKS,
    TASK_SUBTASKS,
    TASKS,
    UNSUPPORTED_FAILURE_MODES,
    ContractError,
    EvidenceItem,
    PredictionRecord,
    index_predictions,
    parse_prediction_lines,
    read_predictions,
    write_predictions,
)
from telemetry_reasoning.eval.metrics import (
    DEFAULT_EVAL_CONFIG_PATH,
    RUBRIC_COLUMNS,
    EvalConfig,
    EvidenceConfig,
    RubricConfig,
    TruthRecord,
    build_blinded_rubric,
    confusion_rows,
    evaluate,
    format_summary,
    load_eval_config,
    load_manifest,
    metrics_rows,
    predicted_onset,
    run_manifest_provenance,
    summarize_run_manifests,
    truth_from_rows,
)

ROOT = Path(__file__).resolve().parents[1]
POS, GRIP, ORI, REL = "position_error", "inadequate_grip_strength", "orientation_error", "early_gripper_release"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def rec(pair, variant="baseline_1", mode=GRIP, subtask="grasp", status="ok", backend="rules",
        evidence=None, explanation="because"):
    if status != "ok":
        mode = subtask = None
    return {
        "pair_id": pair, "model_variant": variant, "predicted_failure_mode": mode,
        "predicted_subtask": subtask, "explanation": explanation, "evidence": evidence or [],
        "status": status, "backend": backend,
    }


def truth(pair, mode=GRIP, subtask="grasp", task="pick_up_object"):
    return TruthRecord(pair, task, subtask, mode)


def index(records, pair_ids):
    return index_predictions(parse_prediction_lines(json.dumps(r) for r in records), pair_ids)


def ev(subtask="grasp", timestamp=5.1, feature="gripper_force", value=3.0):
    return {"subtask": subtask, "timestamp": timestamp, "feature": feature, "value": value}


CFG = EvalConfig()


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_valid_record_roundtrips_and_preserves_string_ids():
    r = PredictionRecord.model_validate(rec("000", evidence=[ev(), ev(feature="ee_pos", value=[0.4, 0.0, 0.1])]))
    assert r.pair_id == "000" and r.key == ("000", "baseline_1")
    assert PredictionRecord.model_validate(r.model_dump(mode="json")) == r


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(pair_id=0),  # ids are strings; 0 would lose the "000"
        lambda d: d.update(pair_id=""),
        lambda d: d.update(extra_field=1),
        lambda d: d.update(model_variant="gpt"),
        lambda d: d.update(predicted_failure_mode="telekinesis"),
        lambda d: d.update(predicted_subtask="juggle"),
        lambda d: d.update(status="maybe"),
        lambda d: d.update(backend="oracle"),
        lambda d: d.update(explanation=None),
        lambda d: d.pop("evidence"),
        lambda d: d.update(predicted_failure_mode=None),  # ok without a mode
    ],
)
def test_invalid_records_are_rejected(mutate):
    d = rec("000")
    mutate(d)
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(d)


@pytest.mark.parametrize("status", ["abstain", "error"])
def test_non_ok_status_must_have_null_prediction(status):
    d = rec("000", status=status)
    d["predicted_failure_mode"] = GRIP
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(d)


@pytest.mark.parametrize(
    "item",
    [
        ev(feature="ee_pos", value=3.0),  # vector feature, scalar value
        ev(feature="gripper_force", value=[3.0]),  # scalar feature, vector value
        ev(feature="ee_pos", value=[0.1, 0.2]),  # wrong length
        ev(feature="ee_quat", value=[1.0, 0.0, 0.0]),
        ev(feature="joint_positions", value=[]),
        ev(feature="flux_capacitor"),
        ev(subtask="juggle"),
        ev(timestamp=-1.0),
        ev(value=True),  # bools are not numbers
        ev(timestamp=float("nan")),
        ev(value=float("inf")),
    ],
)
def test_invalid_evidence_is_rejected(item):
    with pytest.raises(ValidationError):
        EvidenceItem.model_validate(item)


def test_evidence_accepts_ints_and_variable_length_joint_vectors():
    EvidenceItem.model_validate(ev(timestamp=5, value=3))
    EvidenceItem.model_validate(ev(feature="joint_torques", value=[0.0] * 7))


def test_parser_keeps_every_problem_line_instead_of_dropping_it():
    lines = [
        "",
        json.dumps(rec("A")),
        "{not json",
        "[1, 2]",
        '{"pair_id": "B", "model_variant": "main", "status": "weird"}',  # attributable but invalid
        '{"pair_id": "C", "model_variant": "main", "predicted_failure_mode": NaN}',  # NaN is not JSON
    ]
    parsed = parse_prediction_lines(lines, source="t.jsonl")
    assert [r.pair_id for r in parsed.records] == ["A"]
    assert len(parsed.invalid) == 4
    assert [bad.line_no for bad in parsed.invalid] == [3, 4, 5, 6]
    attributable = [b for b in parsed.invalid if b.attributable]
    assert [b.key for b in attributable] == [("B", "main")]


def test_empty_prediction_input_is_fine():
    parsed = parse_prediction_lines([])
    assert parsed.records == [] and parsed.invalid == []
    idx = index_predictions(parsed, ["A"])
    assert idx.by_key == {} and idx.unattributable == [] and idx.variants == []


def test_duplicate_keys_are_rejected():
    with pytest.raises(ContractError, match="duplicate"):
        index([rec("A"), rec("A")], ["A"])
    # the same pair under different variants is fine
    assert len(index([rec("A"), rec("A", variant="main")], ["A"]).by_key) == 2


def test_duplicate_between_valid_and_invalid_line_is_rejected():
    bad = {"pair_id": "A", "model_variant": "baseline_1", "status": "weird"}
    file = parse_prediction_lines([json.dumps(rec("A")), json.dumps(bad)])
    with pytest.raises(ContractError, match="duplicate"):
        index_predictions(file, ["A"])


def test_unknown_pair_ids_are_rejected():
    with pytest.raises(ContractError, match="unknown pair_id: Z"):
        index([rec("A"), rec("Z")], ["A"])


def test_unattributable_lines_are_reported_not_dropped():
    idx = index_predictions(parse_prediction_lines(["garbage", json.dumps(rec("A"))]), ["A"])
    assert len(idx.unattributable) == 1 and len(idx.by_key) == 1


def test_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "out" / "p.jsonl"
    assert write_predictions(path, [rec("000"), rec("001", status="abstain")]) == 2
    assert [r.pair_id for r in read_predictions(path).records] == ["000", "001"]
    assert write_predictions(tmp_path / "empty.jsonl", []) == 0
    with pytest.raises(ValidationError):
        write_predictions(tmp_path / "bad.jsonl", [dict(rec("0"), status="maybe")])


def test_label_vocabularies_match_shared_schemas():
    schemas = pytest.importorskip("telemetry_reasoning.schemas")
    val = lambda x: getattr(x, "value", x)
    assert {m.value for m in schemas.FailureMode} == set(FAILURE_MODES)
    assert {s.value for s in schemas.Subtask} == set(SUBTASKS)
    assert {t.value for t in schemas.Task} == set(TASKS)
    for task, subs in schemas.TASK_SUBTASKS.items():
        assert tuple(val(s) for s in subs) == TASK_SUBTASKS[val(task)]
    assert {f.value for f in schemas.TelemetryFeature} <= set(FEATURE_SHAPES)
    assert set(BACKENDS) == {"rules", "hf", "stub"} and "main" in MODEL_VARIANTS


# --------------------------------------------------------------------------- #
# Ground truth, manifest, config
# --------------------------------------------------------------------------- #
def test_truth_accepts_mappings_and_enum_like_objects():
    enum_like = lambda v: SimpleNamespace(value=v)
    obj = SimpleNamespace(pair_id="001", task=enum_like("transport_object"),
                          failed_subtask=enum_like("reach_target"), failure_mode=enum_like(POS))
    rows = truth_from_rows([{"pair_id": "000", "task": "pick_up_object", "failed_subtask": "grasp",
                             "failure_mode": GRIP}, obj])
    assert [r.pair_id for r in rows] == ["000", "001"] and rows[1].failed_subtask == "reach_target"


@pytest.mark.parametrize(
    "bad",
    [
        {"task": "juggling"},
        {"failure_mode": "telekinesis"},
        {"failed_subtask": "reach_target"},  # not in the pickup template
    ],
)
def test_truth_rejects_inconsistent_metadata(bad):
    row = {"pair_id": "000", "task": "pick_up_object", "failed_subtask": "grasp", "failure_mode": GRIP, **bad}
    with pytest.raises(ContractError):
        truth_from_rows([row])


def test_truth_rejects_duplicate_pair_ids():
    row = {"pair_id": "000", "task": "pick_up_object", "failed_subtask": "grasp", "failure_mode": GRIP}
    with pytest.raises(ContractError, match="duplicate"):
        truth_from_rows([row, row])


def test_manifest_layouts_and_validation(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"pairs": {"000": {"split": "test", "injection_timestamp": 5.1}, "001": {"split": "train"}}}))
    m = load_manifest(p)
    assert m.splits == {"000": "test", "001": "train"} and m.injection_times == {"000": 5.1}
    p.write_text(json.dumps({"pairs": [{"pair_id": "000", "split": "dev", "injection_timestamp": 1}]}))
    assert load_manifest(p).injection_times == {"000": 1.0}
    for bad in ({"pairs": {"000": {"injection_timestamp": "soon"}}},
                {"pairs": {"000": {"injection_timestamp": -1}}},
                {"pairs": {"000": {"injection_timestamp": True}}},
                {"nothing": {}}):
        p.write_text(json.dumps(bad))
        with pytest.raises(ContractError):
            load_manifest(p)


def test_shipped_eval_config_loads_and_unknown_keys_fail(tmp_path):
    cfg = load_eval_config(DEFAULT_EVAL_CONFIG_PATH)
    assert cfg.evidence.timestamp_tol_s > 0 and cfg.rubric.exclude_backends == ("stub",)
    for text in ("eval:\n  nope: 1\n", "eval: {}\nother: 1\n", "x: 1\n", "eval:\n  evidence:\n    nope: 1\n",
                 "eval:\n  onset:\n    timestamp_rule: vibes\n"):
        p = tmp_path / "e.yaml"
        p.write_text(text)
        with pytest.raises(ValueError):
            load_eval_config(p)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_perfect_predictions_score_one():
    ts = [truth("A", POS, "approach"), truth("B", GRIP, "grasp")]
    idx = index([rec("A", mode=POS, subtask="approach"), rec("B")], ["A", "B"])
    sl = evaluate(ts, idx, CFG)["variants"]["baseline_1"]["slices"]["all"]
    assert sl["failure_mode_accuracy"] == sl["failed_subtask_accuracy"] == sl["joint_accuracy"] == 1.0
    assert sl["macro_f1"] == 1.0 and sl["coverage"] == 1.0 and sl["abstain_rate"] == 0.0


def test_abstentions_and_missing_count_as_incorrect_and_missed_recall():
    ts = [truth("1", POS, "approach"), truth("2", POS, "approach"), truth("3", GRIP), truth("4", GRIP)]
    idx = index(
        [rec("1", mode=POS, subtask="approach"),  # correct
         rec("2", status="abstain"),  # abstain: wrong, a missed POS, not a false positive
         # pair 3: no record at all -> missing
         rec("4", mode=POS, subtask="approach")],  # wrong class and wrong subtask
        ["1", "2", "3", "4"],
    )
    sl = evaluate(ts, idx, CFG)["variants"]["baseline_1"]["slices"]["all"]
    assert sl["n"] == 4 and sl["failure_mode_accuracy"] == 0.25
    assert sl["failed_subtask_accuracy"] == 0.25 and sl["joint_accuracy"] == 0.25
    assert (sl["n_ok"], sl["n_abstain"], sl["n_missing"]) == (2, 1, 1)
    assert sl["coverage"] == 0.5 and sl["abstain_rate"] == 0.25 and sl["missing_rate"] == 0.25

    a, b = sl["per_class"][POS], sl["per_class"][GRIP]
    assert (a["support"], a["predicted"], a["tp"]) == (2, 2, 1)
    assert a["precision"] == 0.5 and a["recall"] == 0.5 and a["f1"] == 0.5
    assert b["precision"] is None and b["recall"] == 0.0 and b["f1"] == 0.0  # never predicted, always missed

    assert sl["macro_f1"] == pytest.approx(0.25)  # mean over the two labels present

    conf = sl["confusion"]
    cell = lambda t, p: conf["matrix"][conf["labels"].index(t)][conf["columns"].index(p)]
    assert (cell(POS, POS), cell(POS, "abstain"), cell(GRIP, POS), cell(GRIP, "missing")) == (1, 1, 1, 1)
    assert sum(map(sum, conf["matrix"])) == 4


def test_classes_absent_from_a_small_set():
    ts = [truth("A", POS, "approach")]
    sl = evaluate(ts, index([rec("A", mode=POS, subtask="approach")], ["A"]), CFG)["variants"]["baseline_1"]["slices"]["all"]
    assert sl["labels_absent"] == [m for m in FAILURE_MODES if m != POS]
    absent = sl["per_class"]["orientation_error"]
    assert absent["support"] == 0 and absent["recall"] is None and absent["f1"] is None
    assert sl["macro_f1"] == 1.0  # only the present label counts


def test_predicting_an_absent_class_is_a_false_positive_not_ignored():
    ts = [truth("A", POS, "approach")]
    idx = index([rec("A", mode=ORI, subtask="approach")], ["A"])
    sl = evaluate(ts, idx, CFG)["variants"]["baseline_1"]["slices"]["all"]
    assert sl["per_class"][ORI]["precision"] == 0.0 and sl["per_class"][POS]["recall"] == 0.0
    assert sl["failure_mode_accuracy"] == 0.0


def test_right_mode_wrong_subtask_is_only_half_right():
    ts = [truth("A", GRIP, "grasp")]
    sl = evaluate(ts, index([rec("A", mode=GRIP, subtask="lift")], ["A"]), CFG)["variants"]["baseline_1"]["slices"]["all"]
    assert sl["failure_mode_accuracy"] == 1.0 and sl["failed_subtask_accuracy"] == 0.0 and sl["joint_accuracy"] == 0.0


def test_breakdown_by_task_and_variant():
    ts = [truth("P", GRIP, "grasp"), truth("T", POS, "reach_target", task="transport_object")]
    idx = index(
        [rec("P"), rec("T", mode=GRIP, subtask="maintain_grasp"),
         rec("P", variant="main", mode=POS, subtask="approach"),
         rec("T", variant="main", mode=POS, subtask="reach_target")],
        ["P", "T"],
    )
    res = evaluate(ts, idx, CFG)
    assert set(res["variants"]) == {"baseline_1", "main"}
    b1, main = res["variants"]["baseline_1"]["slices"], res["variants"]["main"]["slices"]
    assert set(b1) == {"all", "pick_up_object", "transport_object"}
    assert b1["pick_up_object"]["failure_mode_accuracy"] == 1.0 and b1["transport_object"]["failure_mode_accuracy"] == 0.0
    assert main["pick_up_object"]["failure_mode_accuracy"] == 0.0 and main["transport_object"]["failure_mode_accuracy"] == 1.0
    assert b1["all"]["failure_mode_accuracy"] == main["all"]["failure_mode_accuracy"] == 0.5


def test_empty_truth_and_empty_predictions_do_not_crash():
    empty = index([], [])
    res = evaluate([], empty, CFG)
    assert res["variants"] == {} and res["n_truth_pairs"] == 0

    res = evaluate([], empty, EvalConfig(expected_variants=("main",)))
    sl = res["variants"]["main"]["slices"]["all"]
    assert sl["n"] == 0 and sl["failure_mode_accuracy"] is None and sl["macro_f1"] is None and sl["coverage"] is None

    res = evaluate([truth("A")], empty, EvalConfig(expected_variants=("baseline_1", "main")))
    for v in ("baseline_1", "main"):  # everything missing, and counted as wrong
        sl = res["variants"][v]["slices"]["all"]
        assert sl["n_missing"] == 1 and sl["missing_rate"] == 1.0 and sl["failure_mode_accuracy"] == 0.0


def test_invalid_outputs_are_counted_and_unattributable_ones_reported():
    file = parse_prediction_lines(
        [json.dumps(rec("A")), '{"pair_id": "B", "model_variant": "baseline_1", "status": "weird"}', "garbage"],
        source="x.jsonl",
    )
    res = evaluate([truth("A"), truth("B")], index_predictions(file, ["A", "B"]), CFG)
    sl = res["variants"]["baseline_1"]["slices"]["all"]
    assert (sl["n_ok"], sl["n_invalid"]) == (1, 1) and sl["failure_mode_accuracy"] == 0.5
    assert res["unattributable_invalid_lines"]["count"] == 1
    assert res["unattributable_invalid_lines"]["examples"][0]["line"] == 3
    assert any("could not be tied" in w for w in res["warnings"])


def test_predictions_outside_the_evaluated_scope_are_counted():
    idx = index([rec("A"), rec("B")], ["A", "B"])
    res = evaluate([truth("A")], idx, CFG)  # e.g. B belongs to another split
    assert res["predictions_outside_scope"] == 1 and res["variants"]["baseline_1"]["slices"]["all"]["n"] == 1


def test_small_samples_are_flagged_and_stub_is_labelled_synthetic():
    idx = index([rec("A", backend="stub")], ["A"])
    res = evaluate([truth("A")], idx, CFG)
    assert res["variants"]["baseline_1"]["synthetic"] is True
    assert any("not an accuracy estimate" in w for w in res["warnings"])
    assert any("SYNTHETIC" in w for w in res["warnings"])
    assert "SYNTHETIC" in format_summary(res)
    real = evaluate([truth("A")], index([rec("A")], ["A"]), CFG)
    assert real["variants"]["baseline_1"]["synthetic"] is False and "SYNTHETIC" not in format_summary(real)


def test_flat_report_rows_cover_every_slice_and_conserve_counts():
    ts = [truth("P"), truth("T", POS, "reach_target", task="transport_object")]
    res = evaluate(ts, index([rec("P"), rec("T", status="abstain")], ["P", "T"]), CFG)
    rows = metrics_rows(res)
    assert [r["task"] for r in rows] == ["all", "pick_up_object", "transport_object"]
    assert all(r["model_variant"] == "baseline_1" and r["synthetic"] is False for r in rows)
    conf = confusion_rows(res)
    for task, n in (("all", 2), ("pick_up_object", 1), ("transport_object", 1)):
        assert sum(r["count"] for r in conf if r["task"] == task) == n
    json.dumps(res, allow_nan=False)  # fully serializable, no NaN


# --------------------------------------------------------------------------- #
# Evidence verification
# --------------------------------------------------------------------------- #
VIEW = {
    "pair_id": "A", "task": "pick_up_object",
    "subtasks": [{"subtask": "grasp", "failure_samples": [
        {"timestamp": 5.08, "subtask": "grasp", "gripper_force": 15.0, "contact_force": 12.0, "ee_pos": [0.4, 0.0, 0.1]},
        {"timestamp": 5.10, "subtask": "grasp", "gripper_force": 3.0, "contact_force": 2.4, "ee_pos": [0.4, 0.0, 0.1]},
    ]}],
}


def test_evidence_support_and_unsupported_reasons():
    items = [
        ev(),  # exact: supported
        ev(value=3.0004),  # inside atol
        ev(timestamp=5.102),  # inside the timestamp tolerance
        ev(feature="ee_pos", value=[0.4, 0.0, 0.1]),  # vector: supported
        ev(value=4.0),  # value_mismatch
        ev(feature="ee_pos", value=[0.4, 0.0, 0.3]),  # value_mismatch (vector)
        ev(timestamp=5.09),  # no_sample_at_timestamp
        ev(subtask="lift"),  # subtask_not_in_input
        ev(feature="obj_pos", value=[0.4, 0.0, 0.05]),  # feature_not_in_sample
    ]
    ts = [truth("A"), truth("B")]
    idx = index([rec("A", evidence=items), rec("B", evidence=[ev()])], ["A", "B"])
    e = evaluate(ts, idx, CFG, views={"A": VIEW})["variants"]["baseline_1"]["evidence"]
    assert e["checked"] and e["n_items"] == 10 and e["n_unverifiable"] == 1  # pair B has no input view
    assert e["n_supported"] == 4 and e["n_unsupported"] == 5
    assert e["unsupported_by_reason"] == {
        "value_mismatch": 2, "no_sample_at_timestamp": 1, "subtask_not_in_input": 1, "feature_not_in_sample": 1,
    }
    assert e["support_rate"] == pytest.approx(4 / 9)  # unverifiable items are not counted against the model
    assert e["n_records_with_unsupported"] == 1 and len(e["details"]) == 5


def test_evidence_tolerances_are_configurable():
    tight = EvalConfig(evidence=EvidenceConfig(timestamp_tol_s=0.001, value_atol=0.0, value_rtol=0.0))
    idx = index([rec("A", evidence=[ev(value=3.0004), ev(timestamp=5.102)])], ["A"])
    e = evaluate([truth("A")], idx, tight, views={"A": VIEW})["variants"]["baseline_1"]["evidence"]
    assert e["n_supported"] == 0 and e["n_unsupported"] == 2


def test_ok_without_evidence_is_reported_and_unchecked_runs_say_so():
    idx = index([rec("A", evidence=[])], ["A"])
    e = evaluate([truth("A")], idx, CFG, views={"A": VIEW})["variants"]["baseline_1"]["evidence"]
    assert e["n_ok_without_evidence"] == 1 and e["n_items"] == 0 and e["support_rate"] is None
    assert evaluate([truth("A")], idx, CFG)["variants"]["baseline_1"]["evidence"] == {"checked": False}


# --------------------------------------------------------------------------- #
# Onset
# --------------------------------------------------------------------------- #
def onset_run(records, injection):
    pair_ids = {r["pair_id"] for r in records}
    ts = [truth(p) for p in sorted(pair_ids)]
    return evaluate(ts, index(records, pair_ids), CFG, injection_times=injection)["variants"]["baseline_1"]["onset"]


def test_predicted_onset_is_latest_evidence_in_the_predicted_subtask():
    r = PredictionRecord.model_validate(
        rec("A", evidence=[ev(timestamp=5.08), ev(timestamp=5.1), ev(subtask="lift", timestamp=7.98, feature="gripper_force")])
    )
    assert predicted_onset(r) == 5.1
    assert predicted_onset(PredictionRecord.model_validate(rec("A", status="abstain"))) is None
    assert predicted_onset(PredictionRecord.model_validate(rec("A", evidence=[]))) is None
    with pytest.raises(ValueError):
        predicted_onset(r, rule="vibes")


def test_onset_error_is_scored_only_with_a_reference_and_a_correct_subtask():
    records = [
        rec("A", evidence=[ev(timestamp=5.12)]),  # +0.02 s late
        rec("B", evidence=[ev(timestamp=5.0)]),  # -0.1 s early
        rec("C", subtask="lift", evidence=[ev(subtask="lift", timestamp=6.0)]),  # wrong subtask: not scored
        rec("D", evidence=[ev(timestamp=5.1)]),  # no injection reference
        rec("E", status="abstain"),  # no prediction
    ]
    o = onset_run(records, {"A": 5.10, "B": 5.10, "C": 5.10, "E": 5.10})
    assert o["computed"] and o["n_with_reference"] == 4 and o["n_scored"] == 2
    assert o["mean_error_s"] == pytest.approx((0.02 - 0.1) / 2)
    assert o["mean_abs_error_s"] == pytest.approx(0.06) and o["median_abs_error_s"] == pytest.approx(0.06)


def test_onset_is_not_computed_without_a_manifest_reference():
    assert onset_run([rec("A")], None) == {"computed": False}
    assert onset_run([rec("A")], {}) == {"computed": False}


# --------------------------------------------------------------------------- #
# Blinded rubric
# --------------------------------------------------------------------------- #
def rubric_fixture(n=6):
    ids = [f"{i:03d}" for i in range(n)]
    ts = [truth(i) for i in ids]
    records = []
    for i in ids:
        records += [rec(i, "baseline_1", explanation=f"rules says {i}"),
                    rec(i, "main", backend="hf", explanation=f"llm says {i}"),
                    rec(i, "baseline_2", backend="stub", explanation=f"stub says {i}")]
    records.append(rec("000", "baseline_3", status="abstain"))
    return ts, index(records, ids)


def test_rubric_is_blinded_seeded_and_never_shows_stub_output():
    ts, idx = rubric_fixture()
    variants = ["baseline_1", "baseline_2", "baseline_3", "main"]
    rows, key = build_blinded_rubric(ts, idx, variants, RubricConfig(n_pairs=3, seed=7))
    assert len(rows) == 6 and len(key) == 6  # 3 pairs x {baseline_1, main}
    assert all(tuple(r) == RUBRIC_COLUMNS for r in rows)
    assert len({r["item_id"] for r in rows}) == 6
    assert {k["backend"] for k in key} == {"rules", "hf"} and "stub" not in {k["backend"] for k in key}
    for r in rows:  # nothing that identifies the variant, backend, or pair
        text = " ".join(str(v) for k, v in r.items() if k != "explanation").lower()
        assert not any(s in text for s in ("baseline", "main", "rules", "hf", "stub"))
        assert r["causal_correctness_1to5"] == r["evidence_support_1to5"] == r["clarity_1to5"] == ""
    assert not any("stub says" in r["explanation"] for r in rows)
    by_item = {k["item_id"]: k for k in key}
    for r in rows:  # key maps each item back to the right pair and explanation
        assert r["explanation"].endswith(by_item[r["item_id"]]["pair_id"])
    assert (rows, key) == build_blinded_rubric(ts, idx, variants, RubricConfig(n_pairs=3, seed=7))


def test_rubric_handles_small_pools_and_zero_requests():
    ts, idx = rubric_fixture(2)
    rows, _ = build_blinded_rubric(ts, idx, ["baseline_1", "main"], RubricConfig(n_pairs=50))
    assert len(rows) == 4
    assert build_blinded_rubric(ts, idx, ["baseline_1"], RubricConfig(n_pairs=0)) == ([], [])
    assert build_blinded_rubric([], index([], []), ["baseline_1"], RubricConfig()) == ([], [])
    only_stub = index([rec("000", backend="stub")], ["000"])
    assert build_blinded_rubric([truth("000")], only_stub, ["baseline_1"], RubricConfig()) == ([], [])


# --------------------------------------------------------------------------- #
# Scripts: run_baseline -> evaluate, with the shared readers stubbed out
# --------------------------------------------------------------------------- #
def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mini_view(pair_id, low_grip=True):
    """Small pickup failure-only view: grip drops 15 -> 3 N at 5.10 s, object left behind in lift."""
    stages = {
        "approach": [0.0, 1.0], "align": [2.0, 3.0],
        "grasp": [4.0, 4.4, 5.08, 5.1, 5.98], "lift": [6.0, 7.98],
    }
    ee_z = {"approach": [0.2, 0.2], "align": [0.2, 0.1], "grasp": [0.1] * 5, "lift": [0.1, 0.3]}
    nodes = []
    for name, times in stages.items():
        rows = []
        for t, z in zip(times, ee_z[name]):
            closed = t >= 4.4
            drop = low_grip and t >= 5.1
            rows.append({
                "timestamp": t, "subtask": name,
                "gripper_state": 1.0 if closed else 0.0,
                "gripper_force": (3.0 if drop else 15.0) if closed else 0.2,
                "contact_force": (2.4 if drop else 12.0) if closed else 0.0,
                "ee_pos": [0.4, 0.0, z], "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "obj_pos": [0.4, 0.0, 0.05 if low_grip or name != "lift" else z - 0.05],
                "obj_quat": [1.0, 0.0, 0.0, 0.0],
            })
        nodes.append({"subtask": name, "metadata": {"subgoal_name": name, "action_description": "x"}, "failure_samples": rows})
    return {"schema_version": "1.0", "pair_id": pair_id, "task": "pick_up_object", "overall_goal": "g", "subtasks": nodes}


@pytest.fixture()
def rb(monkeypatch):
    module = load_script("run_baseline")
    monkeypatch.setattr(module, "_load_failure_view", lambda p: json.loads(p.read_text()))
    return module


def write_bt(directory, pair_id, view=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"pair_{pair_id}_bt.json").write_text(json.dumps(view or mini_view(pair_id)))


def test_run_baseline_writes_valid_jsonl_and_a_run_manifest(rb, tmp_path, capsys):
    bt = tmp_path / "bt"
    write_bt(bt, "001")
    write_bt(bt, "000")
    write_bt(bt, "002", mini_view("002", low_grip=False))
    out = tmp_path / "runs" / "baseline_1.jsonl"
    assert rb.main(["--bt-dir", str(bt), "--output", str(out)]) == 0

    records = read_predictions(out)
    assert records.invalid == [] and [r.pair_id for r in records.records] == ["000", "001", "002"]  # sorted
    first = records.records[0]
    assert (first.model_variant, first.backend, first.status) == ("baseline_1", "rules", "ok")
    assert (first.predicted_failure_mode, first.predicted_subtask) == (GRIP, "grasp")
    assert records.records[2].status == "abstain"  # healthy run: no rule fires

    manifest = json.loads((tmp_path / "runs" / "baseline_1.manifest.json").read_text())
    assert manifest["n_inputs"] == 3 and manifest["status_counts"] == {"ok": 2, "abstain": 1}
    assert manifest["runtime_s"] >= 0 and len(manifest["config_sha256"]) == 64 and "min_grip_force_n" in manifest["config"]
    assert "wrote 3 prediction(s)" in capsys.readouterr().out


def test_run_baseline_reports_unloadable_and_mismatched_files_as_errors(rb, tmp_path):
    bt = tmp_path / "bt"
    write_bt(bt, "000")
    (bt / "pair_001_bt.json").write_text("{ not json")
    write_bt(bt, "002", mini_view("999"))  # file name says 002, content says 999
    (bt / "notes.txt").write_text("ignored")
    records = rb.run(bt, rb.load_rule_config())
    assert [r["pair_id"] for r in records] == ["000", "001", "002"]
    assert [r["status"] for r in records] == ["ok", "error", "error"]
    assert "could not load" in records[1]["explanation"] and "999" in records[2]["explanation"]


def test_run_baseline_never_hands_the_detector_a_paired_tree(rb, tmp_path):
    bt = tmp_path / "bt"
    leaky = mini_view("000")
    leaky["subtasks"][0]["success_samples"] = []
    write_bt(bt, "000", leaky)
    with pytest.raises(ValueError, match="failure-only"):
        rb.run(bt, rb.load_rule_config())


def test_run_baseline_input_errors(rb, tmp_path, capsys):
    out = tmp_path / "o.jsonl"
    assert rb.main(["--bt-dir", str(tmp_path / "missing"), "--output", str(out)]) == 2
    (tmp_path / "empty").mkdir()
    assert rb.main(["--bt-dir", str(tmp_path / "empty"), "--output", str(out)]) == 2
    assert not out.exists() and "error:" in capsys.readouterr().err


@pytest.fixture()
def ev_script(monkeypatch):
    module = load_script("evaluate")
    monkeypatch.setattr(module, "_load_failure_view", lambda p: json.loads(p.read_text()))
    return module


def meta_rows(*pairs):
    return [SimpleNamespace(pair_id=p, task="pick_up_object", failed_subtask="grasp", failure_mode=GRIP) for p in pairs]


def test_evaluate_end_to_end_from_run_baseline(rb, ev_script, monkeypatch, tmp_path, capsys):
    bt = tmp_path / "bt"
    write_bt(bt, "000")
    preds = tmp_path / "baseline_1.jsonl"
    assert rb.main(["--bt-dir", str(bt), "--output", str(preds)]) == 0
    monkeypatch.setattr(ev_script, "_read_metadata", lambda p: meta_rows("000"))

    out = tmp_path / "eval"
    code = ev_script.main(["--metadata", "m.csv", "--predictions", str(preds), "--output-dir", str(out),
                           "--bt-dir", str(bt), "--run-manifest", str(tmp_path / "baseline_1.manifest.json")])
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text())
    entry = metrics["variants"]["baseline_1"]
    assert entry["slices"]["all"]["failure_mode_accuracy"] == 1.0 and entry["synthetic"] is False
    assert entry["evidence"]["checked"] and entry["evidence"]["support_rate"] == 1.0  # baseline cites only real samples
    assert entry["evidence"]["n_unsupported"] == 0
    assert metrics["run_manifests"]["baseline_1"]["backend"] == "rules"
    assert any("not an accuracy estimate" in w for w in metrics["warnings"])  # n=1
    for name in ("metrics.csv", "confusion.csv", "rubric_blinded.csv", "rubric_key.json"):
        assert (out / name).is_file(), name
    assert (out / "plots" / "variant_comparison.png").is_file()
    assert (out / "plots" / "confusion_baseline_1_all.png").is_file()
    assert (out / "plots" / "confusion_baseline_1_pick_up_object.png").is_file()
    assert "baseline_1" in capsys.readouterr().out


def test_evaluate_splits_and_onset_use_the_manifest(ev_script, monkeypatch, tmp_path):
    monkeypatch.setattr(ev_script, "_read_metadata", lambda p: meta_rows("000", "001"))
    preds = tmp_path / "p.jsonl"
    write_predictions(preds, [rec("000", evidence=[ev(timestamp=5.12)]), rec("001")])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": {"000": {"split": "test", "injection_timestamp": 5.1}, "001": {"split": "train"}}}))

    out = tmp_path / "eval"
    assert ev_script.main(["--metadata", "m.csv", "--predictions", str(preds), "--output-dir", str(out),
                           "--manifest", str(manifest), "--split", "test"]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n_truth_pairs"] == 1 and metrics["split"] == "test"
    assert metrics["predictions_outside_scope"] == 1  # pair 001 is in the train split
    onset = metrics["variants"]["baseline_1"]["onset"]
    assert onset["n_scored"] == 1 and onset["mean_error_s"] == pytest.approx(0.02)


def test_evaluate_marks_stub_runs_synthetic_and_keeps_them_out_of_the_rubric(ev_script, monkeypatch, tmp_path):
    monkeypatch.setattr(ev_script, "_read_metadata", lambda p: meta_rows("000"))
    preds = tmp_path / "stub.jsonl"
    write_predictions(preds, [rec("000", variant="baseline_2", backend="stub")])
    out = tmp_path / "eval"
    assert ev_script.main(["--metadata", "m.csv", "--predictions", str(preds), "--output-dir", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["variants"]["baseline_2"]["synthetic"] is True
    assert "True" in (out / "metrics.csv").read_text().splitlines()[1]
    assert not (out / "rubric_blinded.csv").exists()


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--split", "test"], "--split requires --manifest"),
    ],
)
def test_evaluate_rejects_bad_invocations(ev_script, monkeypatch, tmp_path, capsys, extra, message):
    monkeypatch.setattr(ev_script, "_read_metadata", lambda p: meta_rows("000"))
    preds = tmp_path / "p.jsonl"
    write_predictions(preds, [rec("000")])
    assert ev_script.main(["--metadata", "m.csv", "--predictions", str(preds), "--output-dir", str(tmp_path / "o"), *extra]) == 2
    assert message in capsys.readouterr().err


def test_evaluate_rejects_duplicates_and_unknown_ids(ev_script, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ev_script, "_read_metadata", lambda p: meta_rows("000"))
    for name, records, message in (("dup", [rec("000"), rec("000")], "duplicate"), ("unk", [rec("999")], "unknown pair_id")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        assert ev_script.main(["--metadata", "m.csv", "--predictions", str(path), "--output-dir", str(tmp_path / name)]) == 2
        assert message in capsys.readouterr().err
        assert not (tmp_path / name / "metrics.json").exists()


def test_evaluate_with_an_empty_predictions_file_scores_everything_missing(ev_script, monkeypatch, tmp_path):
    monkeypatch.setattr(ev_script, "_read_metadata", lambda p: meta_rows("000", "001"))
    preds = tmp_path / "empty.jsonl"
    preds.write_text("")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text("eval:\n  expected_variants: [baseline_1]\n")
    out = tmp_path / "eval"
    assert ev_script.main(["--metadata", "m.csv", "--predictions", str(preds), "--config", str(cfg), "--output-dir", str(out)]) == 0
    sl = json.loads((out / "metrics.json").read_text())["variants"]["baseline_1"]["slices"]["all"]
    assert sl["n_missing"] == 2 and sl["failure_mode_accuracy"] == 0.0


# --------------------------------------------------------------------------- #
# excessive_speed: a valid label with no accepted dataset examples
# --------------------------------------------------------------------------- #
SPEED = "excessive_speed"


def test_excessive_speed_is_a_valid_prediction_and_vocabulary_matches_schemas():
    assert SPEED in FAILURE_MODES and SPEED in UNSUPPORTED_FAILURE_MODES
    assert SPEED not in DATASET_FAILURE_MODES and len(DATASET_FAILURE_MODES) == 4
    assert PredictionRecord.model_validate(rec("000", mode=SPEED)).predicted_failure_mode == SPEED
    assert not parse_prediction_lines([json.dumps(rec("000", mode=SPEED))]).invalid


def test_unsupported_class_is_reported_but_excluded_from_both_macro_averages():
    ts = [truth("A", POS, "approach"), truth("B", GRIP)]
    idx = index([rec("A", mode=POS, subtask="approach"), rec("B", mode=GRIP)], ["A", "B"])
    sl = evaluate(ts, idx, CFG)["variants"]["baseline_1"]["slices"]["all"]
    row = sl["per_class"][SPEED]
    assert row["unsupported"] is True and row["support"] == 0 and row["predicted"] == 0 and row["f1"] is None
    assert all(sl["per_class"][m]["unsupported"] is False for m in DATASET_FAILURE_MODES)
    assert sl["unsupported_labels"] == [SPEED]
    assert sl["macro_f1"] == 1.0  # mean over the two observed dataset classes only
    # two perfect classes out of four dataset labels; the unsupported class must not count as a fifth
    assert sl["macro_f1_fixed_labels"] == pytest.approx(0.5)


def test_predicting_the_unsupported_class_is_a_false_positive_and_incorrect():
    ts = [truth("A", GRIP)]
    res = evaluate(ts, index([rec("A", mode=SPEED)], ["A"]), CFG)
    sl = res["variants"]["baseline_1"]["slices"]["all"]
    assert sl["failure_mode_accuracy"] == 0.0 and sl["per_class"][SPEED]["predicted"] == 1
    assert sl["per_class"][GRIP]["recall"] == 0.0 and sl["macro_f1"] == 0.0
    conf = sl["confusion"]
    assert conf["matrix"][conf["labels"].index(GRIP)][conf["columns"].index(SPEED)] == 1
    assert res["unsupported_labels"] == [SPEED]


def test_truth_with_the_unsupported_mode_is_accepted_warned_and_not_in_macro():
    ts = truth_from_rows([SimpleNamespace(pair_id="S", task="pick_up_object", failed_subtask="approach", failure_mode=SPEED)])
    res = evaluate(ts + [truth("A", GRIP)], index([rec("A", mode=GRIP)], ["S", "A"]), CFG)
    sl = res["variants"]["baseline_1"]["slices"]["all"]
    assert sl["per_class"][SPEED]["support"] == 1 and sl["per_class"][SPEED]["recall"] == 0.0
    assert sl["macro_f1"] == 1.0  # only the supported GRIP class is averaged
    assert any("unsupported failure mode" in w for w in res["warnings"])
    json.dumps(res, allow_nan=False)


# --------------------------------------------------------------------------- #
# evaluate.py --bt-dir through the REAL loaders (regression: the script called
# .model_dump() on the dict returned by to_failure_only(); other tests stub the loader)
# --------------------------------------------------------------------------- #
SAMPLE = ROOT / "data" / "sample"


def test_evaluate_script_verifies_evidence_against_a_real_stored_bt(tmp_path, capsys):
    bt = tmp_path / "bt"
    bt.mkdir()
    (bt / "pair_000_bt.json").write_text((SAMPLE / "pair_000_bt.json").read_text())
    preds = tmp_path / "baseline_1.jsonl"
    assert load_script("run_baseline").main(["--bt-dir", str(bt), "--output", str(preds)]) == 0

    out = tmp_path / "eval"
    code = load_script("evaluate").main([
        "--metadata", str(SAMPLE / "metadata.csv"), "--predictions", str(preds),
        "--bt-dir", str(bt), "--output-dir", str(out),
    ])
    assert code == 0, capsys.readouterr().err
    entry = json.loads((out / "metrics.json").read_text())["variants"]["baseline_1"]
    assert entry["evidence"]["checked"] is True
    assert entry["evidence"]["n_items"] > 0 and entry["evidence"]["n_unsupported"] == 0
    assert entry["evidence"]["n_unverifiable"] == 0 and entry["evidence"]["support_rate"] == 1.0


def test_evaluate_script_flags_tampered_evidence_against_a_real_stored_bt(tmp_path):
    bt = tmp_path / "bt"
    bt.mkdir()
    (bt / "pair_000_bt.json").write_text((SAMPLE / "pair_000_bt.json").read_text())
    forged = rec("000", evidence=[ev(timestamp=5.1, value=14.0)])  # the stored sample says 3.0 N
    preds = tmp_path / "p.jsonl"
    write_predictions(preds, [forged])
    out = tmp_path / "eval"
    assert load_script("evaluate").main([
        "--metadata", str(SAMPLE / "metadata.csv"), "--predictions", str(preds),
        "--bt-dir", str(bt), "--output-dir", str(out),
    ]) == 0
    evidence = json.loads((out / "metrics.json").read_text())["variants"]["baseline_1"]["evidence"]
    assert evidence["n_unsupported"] == 1 and evidence["unsupported_by_reason"] == {"value_mismatch": 1}


# --------------------------------------------------------------------------- #
# Run manifests: provenance and runtime/token aggregation
# --------------------------------------------------------------------------- #
def request(pair, variant, tin=100, tout=10, latency=0.5, method="tokenizer", **extra):
    return {"pair_id": pair, "model_variant": variant, "input_tokens": tin, "output_tokens": tout,
            "token_count_method": method, "latency_s": latency, **extra}


def llm_manifest(requests, variants=("baseline_2", "main"), backend="hf", **extra):
    return {"backend": backend, "synthetic": backend == "stub", "variants": list(variants),
            "requests": requests, "runtime_s": 9.0, "model": {"id": "m", "revision": "r"},
            "decoding": {"seed": 723}, "config_sha256": "ab" * 32, **extra}


BASELINE_MANIFEST = {"model_variant": "baseline_1", "backend": "rules", "n_inputs": 3, "runtime_s": 0.25,
                     "status_counts": {"ok": 3}}


def test_multi_variant_manifest_is_summarized_per_variant_and_scoped_to_pairs():
    m = llm_manifest([
        request("A", "baseline_2", 100, 10, 0.5), request("B", "baseline_2", 300, 30, 1.5),
        request("A", "main", 700, 70, 2.0), request("Z", "main", 9999, 999, 99.0),  # Z is out of scope
    ])
    res = summarize_run_manifests([("run/run_manifest.json", m)], in_scope_pairs={"A", "B"})
    b2, main = res["baseline_2"], res["main"]
    assert (b2["input_tokens_total"], b2["output_tokens_total"]) == (400, 40)
    assert (b2["input_tokens_mean"], b2["latency_s_total"], b2["latency_s_mean"]) == (200.0, 2.0, 1.0)
    assert (b2["n_requests"], b2["n_with_tokens"], b2["scope"]) == (2, 2, "in_scope_requests")
    assert (main["input_tokens_total"], main["n_requests"], main["n_requests_in_run"]) == (700, 1, 2)
    assert main["latency_s_total"] == 2.0 and main["run_runtime_s"] == 9.0  # shared run time, not split
    assert b2["token_count_methods"] == ["tokenizer"] and b2["backend"] == "hf" and b2["synthetic"] is False
    assert b2["source_manifest"] == "run/run_manifest.json"


def test_without_a_pair_scope_every_request_counts():
    m = llm_manifest([request("A", "main"), request("Z", "main")], variants=("main",))
    res = summarize_run_manifests([("m", m)])["main"]
    assert res["n_requests"] == 2 and res["scope"] == "all_requests"


def test_single_variant_manifest_reports_wall_time_but_never_invented_tokens():
    res = summarize_run_manifests([("b1.manifest.json", BASELINE_MANIFEST)], in_scope_pairs={"A"})["baseline_1"]
    assert res["scope"] == "whole_run" and res["run_runtime_s"] == 0.25 and res["n_requests"] == 3
    assert res["input_tokens_total"] is None and res["output_tokens_total"] is None
    assert res["latency_s_total"] is None and res["token_count_methods"] == []
    assert res["backend"] == "rules" and res["synthetic"] is False


def test_missing_and_malformed_resource_fields_are_not_treated_as_zero():
    m = llm_manifest([
        request("A", "main"),
        request("B", "main", tin=None, tout=None, latency=None),      # not logged
        request("C", "main", tin=-5, tout=10, latency=float("nan")),  # malformed
        request("D", "main", tin=True, tout=10, latency="fast"),      # wrong types
    ], variants=("main",))
    res = summarize_run_manifests([("m", m)])["main"]
    assert res["n_requests"] == 4 and res["n_with_tokens"] == 1 and res["n_with_latency"] == 1
    assert res["input_tokens_total"] == 100 and res["latency_s_total"] == 0.5
    assert res["n_invalid_resource_fields"] == 4  # C: tokens + latency, D: tokens + latency

    empty = summarize_run_manifests([("m", llm_manifest([request("A", "main", None, None, None)], ("main",)))])["main"]
    assert empty["input_tokens_total"] is None and empty["latency_s_total"] is None and empty["n_requests"] == 1


def test_stub_token_counts_are_labelled_as_upper_bounds():
    m = llm_manifest([request("A", "main", method="utf8_bytes_upper_bound")], ("main",), backend="stub")
    res = summarize_run_manifests([("m", m)])["main"]
    assert res["synthetic"] is True and "upper bounds" in res["note"]
    assert res["token_count_methods"] == ["utf8_bytes_upper_bound"]


def test_two_manifests_for_one_variant_and_unrecognized_manifests_are_rejected():
    a = llm_manifest([request("A", "main")], ("main",))
    b = llm_manifest([request("A", "main")], ("main",), backend="stub")
    for fn in (summarize_run_manifests, run_manifest_provenance):
        with pytest.raises(ContractError, match="both describe variant 'main'"):
            fn([("real/run_manifest.json", a), ("stub/run_manifest.json", b)])
    for bad in ({"n_inputs": 3}, {"model_variant": "gpt"}, llm_manifest([request("A", "gpt")], ("gpt",))):
        with pytest.raises(ContractError):
            summarize_run_manifests([("bad.json", bad)])


def test_provenance_embeds_single_manifests_whole_and_compacts_shared_runs():
    shared = llm_manifest([request("A", "baseline_2"), request("A", "main")])
    prov = run_manifest_provenance([("b1.json", BASELINE_MANIFEST), ("run.json", shared)])
    assert prov["baseline_1"]["backend"] == "rules" and prov["baseline_1"]["source_manifest"] == "b1.json"
    for variant in ("baseline_2", "main"):
        entry = prov[variant]
        assert "requests" not in entry  # per-request records live in the resource summary
        assert entry["model"] == {"id": "m", "revision": "r"} and entry["source_manifest"] == "run.json"
        assert entry["shared_with_variants"] == ["baseline_2", "main"]


def test_evaluate_attaches_resources_without_changing_scores():
    ts = [truth("A")]
    idx = index([rec("A"), rec("A", variant="main", backend="hf")], ["A"])
    plain = evaluate(ts, idx, CFG)
    res = summarize_run_manifests([("m", llm_manifest([request("A", "main")], ("main",)))], in_scope_pairs={"A"})
    scored = evaluate(ts, idx, CFG, resources=res)
    assert scored["variants"]["main"]["resources"]["input_tokens_total"] == 100
    assert scored["variants"]["baseline_1"]["resources"] == {"reported": False}
    for v in ("baseline_1", "main"):
        assert scored["variants"][v]["slices"] == plain["variants"][v]["slices"]
    json.dumps(scored, allow_nan=False)


def test_evaluate_warns_on_backend_mismatch_and_on_manifests_for_unscored_variants():
    idx = index([rec("A", variant="main", backend="stub")], ["A"])
    res = summarize_run_manifests([
        ("real.json", llm_manifest([request("A", "main")], ("main",), backend="hf")),
        ("b1.json", BASELINE_MANIFEST),
    ])
    out = evaluate([truth("A")], idx, CFG, resources=res)
    assert any("main: run manifest backend 'hf' does not match" in w for w in out["warnings"])
    assert any("'baseline_1', which has no scored predictions" in w for w in out["warnings"])


def test_resources_reach_the_flat_csv_rows_and_the_text_summary():
    idx = index([rec("A", variant="main", backend="stub")], ["A"])
    res = summarize_run_manifests([("m", llm_manifest([request("A", "main", method="utf8_bytes_upper_bound")], ("main",), backend="stub"))])
    out = evaluate([truth("A", task="pick_up_object")], idx, CFG, resources=res)
    rows = metrics_rows(out)
    assert [r["input_tokens_total"] for r in rows] == [100, None]  # "all" slice only
    assert rows[0]["latency_s_total"] == 0.5 and rows[0]["run_runtime_s"] == 9.0
    assert len({tuple(r) for r in rows}) == 1  # identical columns on every row
    text = format_summary(out)
    assert "tokens in/out 100/10 (utf8_bytes_upper_bound)" in text and "request latency 0.500 s" in text
    unreported = format_summary(evaluate([truth("A")], index([rec("A")], ["A"]), CFG))
    assert "resources: not reported (pass --run-manifest)" in unreported


def test_evaluate_script_reports_resources_from_both_manifest_layouts(tmp_path, capsys):
    ev_script = load_script("evaluate")
    ev_script._read_metadata = lambda p: meta_rows("000", "001")
    preds = tmp_path / "p.jsonl"
    write_predictions(preds, [rec("000"), rec("001"),
                              rec("000", variant="main", backend="hf"), rec("001", variant="main", backend="hf")])
    b1 = tmp_path / "baseline_1.manifest.json"
    b1.write_text(json.dumps(BASELINE_MANIFEST))
    shared = tmp_path / "run_manifest.json"
    shared.write_text(json.dumps(llm_manifest(
        [request("000", "main", 100, 10, 1.0), request("001", "main", 200, 20, 2.0), request("999", "main", 5, 5, 5.0)],
        ("main",))))
    out = tmp_path / "eval"
    assert ev_script.main(["--metadata", "m.csv", "--predictions", str(preds), "--output-dir", str(out),
                           "--run-manifest", str(b1), str(shared)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    main = metrics["variants"]["main"]["resources"]
    assert (main["input_tokens_total"], main["output_tokens_total"], main["latency_s_total"]) == (300, 30, 3.0)
    assert main["n_requests"] == 2 and main["n_requests_in_run"] == 3  # pair 999 is not in the truth set
    assert metrics["variants"]["baseline_1"]["resources"]["run_runtime_s"] == 0.25
    assert metrics["run_manifests"]["baseline_1"]["backend"] == "rules"
    assert metrics["run_manifests"]["main"]["model"]["revision"] == "r" and "requests" not in metrics["run_manifests"]["main"]
    header = (out / "metrics.csv").read_text().splitlines()[0].split(",")
    assert {"input_tokens_total", "output_tokens_total", "latency_s_total"} <= set(header)
    assert "resources: tokens in/out 300/30" in capsys.readouterr().out


def test_evaluate_script_rejects_conflicting_or_unrecognized_run_manifests(tmp_path, capsys):
    ev_script = load_script("evaluate")
    ev_script._read_metadata = lambda p: meta_rows("000")
    preds = tmp_path / "p.jsonl"
    write_predictions(preds, [rec("000")])
    one, two, bad = (tmp_path / n for n in ("one.json", "two.json", "bad.json"))
    one.write_text(json.dumps(BASELINE_MANIFEST))
    two.write_text(json.dumps(BASELINE_MANIFEST))
    bad.write_text(json.dumps({"hello": "world"}))
    for manifests, message in (([one, two], "both describe variant 'baseline_1'"), ([bad], "unrecognized run manifest")):
        out = tmp_path / f"o{len(manifests)}{manifests[0].stem}"
        assert ev_script.main(["--metadata", "m.csv", "--predictions", str(preds), "--output-dir", str(out),
                               "--run-manifest", *map(str, manifests)]) == 2
        assert message in capsys.readouterr().err
        assert not (out / "metrics.json").exists()
