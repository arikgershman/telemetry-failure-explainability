"""Tests for telemetry and evaluation figures.

Figures are checked through the Matplotlib objects themselves (axes, line data,
labels, marker positions), not just "a file was written", and everything runs
headless on the Agg canvas.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from telemetry_reasoning.eval.contracts import write_predictions
from telemetry_reasoning.eval.metrics import EvalConfig, TruthRecord, evaluate
from telemetry_reasoning.eval.contracts import index_predictions, parse_prediction_lines
from telemetry_reasoning.eval.plots import (
    C_DIVERGENCE,
    C_FAILURE,
    C_INJECTION,
    C_ONSET,
    C_SUCCESS,
    DEFAULT_PANELS,
    PanelSpec,
    Series,
    TelemetryTrace,
    plot_confusion_matrix,
    plot_pair_telemetry,
    plot_variant_comparison,
    save_figure,
    trace_from_rows,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE_BOUNDS = [("approach", 0.0), ("align", 2.0), ("grasp", 4.0), ("lift", 6.0)]


# --------------------------------------------------------------------------- #
# Synthetic pair: 400 rows at 50 Hz, grip drops 15 -> 3 N at 5.10 s in the failure run
# --------------------------------------------------------------------------- #
def stage_of(t):
    return [name for name, start in STAGE_BOUNDS if t >= start - 1e-9][-1]


def make_row(i, failure):
    t = round(i * 0.02, 2)
    stage = stage_of(t)
    closed = t >= 4.4
    dropped = failure and t >= 5.1
    ee_z = {"approach": 0.2, "align": 0.2 - 0.1 * (t - 2.0) / 1.98, "grasp": 0.1, "lift": 0.1 + 0.2 * (t - 6.0) / 1.98}[stage]
    obj_z = 0.05 if (failure or stage != "lift") else ee_z - 0.05
    return {
        "timestamp": t, "subtask": stage,
        "gripper_state": 1.0 if closed else 0.0,
        "gripper_force": (3.0 if dropped else 15.0) if closed else 0.2,
        "contact_force": (2.4 if dropped else 12.0) if closed else 0.0,
        "ee_pos": [0.4, 0.0, ee_z], "obj_pos": [0.4, 0.0, obj_z],
    }


def make_rows(failure):
    return [make_row(i, failure) for i in range(400)]


RETAINED_IDX = [0, 50, 99, 100, 150, 199, 200, 215, 219, 220, 221, 254, 255, 256, 280, 299, 300, 350, 399]


def make_bt(divergence="computed"):
    """Compressed BT (as ``model_dump``) built from every retained index, grouped by subtask."""
    nodes = []
    for name, _ in STAGE_BOUNDS:
        pick = lambda failure: [make_row(i, failure) for i in RETAINED_IDX if stage_of(round(i * 0.02, 2)) == name]
        div = None
        if divergence == "computed":
            div = {
                "first_divergence_timestamp": 5.1 if name == "grasp" else None,
                "max_abs_deltas": {"gripper_force": 12.0 if name in ("grasp", "lift") else 0.0},
            }
        nodes.append({"subtask": name, "success_samples": pick(False), "failure_samples": pick(True),
                      "deterministic_divergence": div})
    return {"schema_version": "1.0", "pair_id": "000", "task": "pick_up_object", "subtasks": nodes}


@pytest.fixture(scope="module")
def traces():
    return trace_from_rows(make_rows(False)), trace_from_rows(make_rows(True))


def bars(ax):
    """Divergence bars only; the subtask band shading is also a patch, so use containers."""
    return [p for c in ax.containers for p in c.patches]


def lines_labelled(ax, label):
    return [ln for ln in ax.lines if ln.get_label() == label]


# --------------------------------------------------------------------------- #
# Traces
# --------------------------------------------------------------------------- #
def test_trace_from_mapping_rows_dataframe_and_json_cells():
    rows = make_rows(True)[:5]
    from_maps = trace_from_rows(rows)
    assert from_maps.t.shape == (5,) and from_maps.features["ee_pos"].shape == (5, 3)
    assert from_maps.series("ee_pos", 2)[0] == pytest.approx(0.2) and from_maps.series("gripper_force")[0] == 0.2

    frame = pd.DataFrame([{**r, "ee_pos": json.dumps(r["ee_pos"]), "obj_pos": json.dumps(r["obj_pos"])} for r in rows])
    from_frame = trace_from_rows(frame)  # vector cells stored as JSON strings, as in the CSVs
    np.testing.assert_allclose(from_frame.features["ee_pos"], from_maps.features["ee_pos"])

    class Row:  # validated-row style: attributes, enum-like subtask
        def __init__(self, d):
            self.__dict__.update(d, subtask=SimpleNamespace(value=d["subtask"]))

    from_objects = trace_from_rows([Row(r) for r in rows])
    assert from_objects.subtask == from_maps.subtask


def test_trace_spans_and_missing_feature_handling():
    t = trace_from_rows(make_rows(True))
    assert [(n, a) for n, a, _ in t.spans()] == [(n, pytest.approx(s)) for n, s in STAGE_BOUNDS]
    assert t.spans()[-1][2] == pytest.approx(7.98)
    partial = trace_from_rows([{k: v for k, v in r.items() if k != "contact_force"} for r in make_rows(True)[:3]])
    assert "contact_force" not in partial.features and "gripper_force" in partial.features
    with pytest.raises(ValueError):
        trace_from_rows([])
    with pytest.raises(ValueError, match="choose a component"):
        t.series("ee_pos")


# --------------------------------------------------------------------------- #
# Telemetry figure
# --------------------------------------------------------------------------- #
def test_figure_structure_axes_labels_and_units(traces):
    fig = plot_pair_telemetry(*traces, bt=make_bt(), injection_time=5.1, title="demo")
    assert isinstance(fig, Figure) and isinstance(fig.canvas, FigureCanvasAgg)  # headless; no pyplot
    assert len(fig.axes) == len(DEFAULT_PANELS) + 1
    assert fig.axes[-1].get_xlabel() == "Time (s)"
    assert all(ax.get_xlabel() == "" for ax in fig.axes[:-1])
    ylabels = [ax.get_ylabel() for ax in fig.axes]
    assert "(N)" in ylabels[0] and "(N)" in ylabels[1] and "(m)" in ylabels[3] and "max |Δ|" in ylabels[-1]
    assert fig._suptitle.get_text() == "demo"
    xlim = fig.axes[0].get_xlim()
    assert all(ax.get_xlim() == xlim for ax in fig.axes)  # one shared time axis


def test_raw_traces_are_plotted_exactly_with_distinct_run_colours(traces):
    success, failure = traces
    fig = plot_pair_telemetry(success, failure)
    ax = fig.axes[0]
    (s_line,) = lines_labelled(ax, "success: gripper force")
    (f_line,) = lines_labelled(ax, "failure: gripper force")
    np.testing.assert_array_equal(s_line.get_xdata(), success.t)
    np.testing.assert_array_equal(f_line.get_ydata(), failure.series("gripper_force"))
    assert f_line.get_ydata().min() == 0.2 and f_line.get_ydata()[-1] == 3.0 and s_line.get_ydata()[-1] == 15.0
    assert s_line.get_color() == C_SUCCESS and f_line.get_color() == C_FAILURE

    # two-series panel: height of end effector and object, line styles differ per series
    h = fig.axes[3]
    ee, obj = lines_labelled(h, "failure: end effector z")[0], lines_labelled(h, "failure: object z")[0]
    assert ee.get_ydata()[-1] == pytest.approx(0.3) and obj.get_ydata()[-1] == pytest.approx(0.05)  # object left behind
    assert ee.get_linestyle() != obj.get_linestyle()
    assert h.get_legend() is not None and len(h.get_legend().get_texts()) == 4


def test_retained_samples_are_drawn_as_markers_at_the_bt_timestamps(traces):
    bt = make_bt()
    fig = plot_pair_telemetry(*traces, bt=bt)
    ax = fig.axes[0]
    (fail_pts,) = lines_labelled(ax, "failure retained samples")
    (succ_pts,) = lines_labelled(ax, "success retained samples")
    expected = sorted(t for n in bt["subtasks"] for t in (r["timestamp"] for r in n["failure_samples"]))
    np.testing.assert_allclose(fail_pts.get_xdata(), expected)
    assert len(expected) == len(RETAINED_IDX) and fail_pts.get_linestyle() == "None" and fail_pts.get_marker() == "o"
    assert fail_pts.get_markerfacecolor() == C_FAILURE and succ_pts.get_markerfacecolor() == "white"  # filled vs hollow
    assert 5.08 in set(np.round(fail_pts.get_xdata(), 2)) and 5.1 in set(np.round(fail_pts.get_xdata(), 2))
    # without a BT there is nothing to mark
    bare = plot_pair_telemetry(*traces).axes[0]
    assert not lines_labelled(bare, "failure retained samples")


def test_subtask_boundaries_on_every_panel_and_names_on_top(traces):
    fig = plot_pair_telemetry(*traces)
    for ax in fig.axes:
        xs = sorted(round(float(ln.get_xdata()[0]), 6) for ln in ax.lines if ln.get_color() == "0.35")
        assert xs == [2.0, 4.0, 6.0]
    names = [t.get_text() for t in fig.axes[0].texts]
    assert names == ["approach", "align", "grasp", "lift"]
    # a success run with different boundaries is drawn separately (dotted, success colour)
    shifted = TelemetryTrace(traces[0].t, tuple(("align" if 1.9 <= t < 2.0 else s) for s, t in zip(traces[0].subtask, traces[0].t)),
                             traces[0].features)
    fig2 = plot_pair_telemetry(shifted, traces[1])
    dotted = [ln for ln in fig2.axes[0].lines if ln.get_label() == "subtask boundary (success run)"]
    assert len(dotted) == 1 and dotted[0].get_xdata()[0] == pytest.approx(1.9) and dotted[0].get_linestyle() == ":"


def test_injection_onset_and_divergence_markers_are_distinct_events(traces):
    fig = plot_pair_telemetry(*traces, bt=make_bt(), injection_time=5.10, predicted_onset=5.12)
    for ax in fig.axes:
        by_color = {ln.get_color(): ln for ln in ax.lines if ln.get_color() in (C_INJECTION, C_ONSET, C_DIVERGENCE)}
        assert set(by_color) == {C_INJECTION, C_ONSET, C_DIVERGENCE}
        assert by_color[C_INJECTION].get_xdata()[0] == 5.10
        assert by_color[C_ONSET].get_xdata()[0] == 5.12
        assert by_color[C_DIVERGENCE].get_xdata()[0] == 5.10  # computed first divergence from the BT
        styles = {ln.get_linestyle() for ln in by_color.values()}
        assert len(styles) == 3  # colour AND line style differ, so they survive greyscale printing
    labels = {ln.get_label() for ln in fig.axes[0].lines}
    assert {"injection time (manifest)", "predicted onset", "first divergence (computed)"} <= labels
    legend_texts = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert {"injection time (manifest)", "predicted onset", "first divergence (computed)",
            "failure retained samples", "success retained samples"} <= set(legend_texts)


def test_optional_markers_are_absent_when_not_supplied(traces):
    fig = plot_pair_telemetry(*traces, bt=make_bt())
    labels = {ln.get_label() for ln in fig.axes[0].lines}
    assert "injection time (manifest)" not in labels and "predicted onset" not in labels


def test_divergence_panel_shows_computed_values_per_subtask(traces):
    fig = plot_pair_telemetry(*traces, bt=make_bt("computed"), divergence_feature="gripper_force")
    ax = fig.axes[-1]
    heights = {round(p.get_x(), 6): p.get_height() for p in bars(ax)}
    assert heights == {0.0: 0.0, 2.0: 0.0, 4.0: 12.0, 6.0: 12.0}
    widths = [p.get_width() for p in bars(ax)]
    assert widths == pytest.approx([2.0, 2.0, 2.0, 1.98])  # each bar spans its subtask
    assert "gripper_force" in ax.get_ylabel() and "(N)" in ax.get_ylabel()
    assert not ax.texts


def test_null_divergence_is_labelled_not_computed_and_never_drawn_as_zero(traces):
    ax = plot_pair_telemetry(*traces, bt=make_bt("null")).axes[-1]
    assert bars(ax) == [] and any("not computed" in t.get_text() for t in ax.texts)
    ax = plot_pair_telemetry(*traces).axes[-1]
    assert bars(ax) == [] and any("no behavior tree" in t.get_text() for t in ax.texts)


def test_divergence_feature_absent_from_a_node_draws_no_bar_for_it(traces):
    bt = make_bt("computed")
    ax = plot_pair_telemetry(*traces, bt=bt, divergence_feature="obj_vel").axes[-1]
    assert bars(ax) == []  # no obj_vel deltas anywhere: nothing invented


def test_custom_panels_and_synthetic_note(traces):
    panels = (PanelSpec("Contact", "Force (N)", (Series("contact_force", None, "contact"),)),)
    fig = plot_pair_telemetry(*traces, panels=panels, note="SYNTHETIC: stub predictions")
    assert len(fig.axes) == 2
    assert any(t.get_text() == "SYNTHETIC: stub predictions" for t in fig.texts)


def test_figure_saves_as_a_real_png_headlessly(traces, tmp_path):
    fig = plot_pair_telemetry(*traces, bt=make_bt(), injection_time=5.1, predicted_onset=5.1)
    path = save_figure(fig, tmp_path / "nested" / "pair.png", dpi=60)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 5_000


# --------------------------------------------------------------------------- #
# Evaluation figures
# --------------------------------------------------------------------------- #
def eval_result(backend="rules"):
    mk = lambda mode, variant, status="ok", b=backend: {
        "pair_id": "A", "model_variant": variant, "predicted_failure_mode": mode if status == "ok" else None,
        "predicted_subtask": "grasp" if status == "ok" else None, "explanation": "x", "evidence": [],
        "status": status, "backend": b}
    idx = index_predictions(parse_prediction_lines(json.dumps(r) for r in
                            [mk("inadequate_grip_strength", "baseline_1"), mk(None, "main", "abstain")]), ["A"])
    truth = [TruthRecord("A", "pick_up_object", "grasp", "inadequate_grip_strength")]
    return evaluate(truth, idx, EvalConfig())


def test_confusion_figure_shows_the_counts_and_the_synthetic_banner():
    res = eval_result()
    sl = res["variants"]["baseline_1"]["slices"]["all"]
    fig = plot_confusion_matrix(sl, "baseline_1 / all")
    ax = fig.axes[0]
    image = ax.images[0].get_array()
    labels, columns = sl["confusion"]["labels"], sl["confusion"]["columns"]
    assert image.shape == (len(labels), len(columns))
    assert image[labels.index("inadequate_grip_strength"), columns.index("inadequate_grip_strength")] == 1
    assert image.sum() == 1 and "n=1" in ax.get_title()
    assert [t.get_text() for t in ax.get_yticklabels()] == labels
    assert not any(t.get_text() == "SYNTHETIC" for t in fig.texts)
    assert any(t.get_text() == "SYNTHETIC" for t in plot_confusion_matrix(sl, "x", synthetic=True).texts)


def test_variant_comparison_bars_values_and_synthetic_labels():
    res = eval_result()
    fig = plot_variant_comparison(res)
    ax = fig.axes[0]
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert ticks == ["baseline_1", "main"]
    heights = [round(p.get_height(), 6) for p in bars(ax)]
    assert heights[:2] == [1.0, 0.0]  # failure_mode_accuracy: baseline_1 right, main abstained
    assert not any(t.get_text() == "SYNTHETIC" for t in fig.texts)

    synthetic = plot_variant_comparison(eval_result(backend="stub"))
    assert all("SYNTHETIC" in t.get_text() for t in synthetic.axes[0].get_xticklabels())
    assert any(t.get_text() == "SYNTHETIC" for t in synthetic.texts)


def test_unavailable_metrics_are_labelled_na_not_plotted_as_perfect():
    res = evaluate([], index_predictions(parse_prediction_lines([]), []), EvalConfig(expected_variants=("main",)))
    ax = plot_variant_comparison(res).axes[0]
    assert [round(p.get_height(), 6) for p in bars(ax)] == [0.0] * 4
    assert {t.get_text() for t in ax.texts} == {"n/a"}


# --------------------------------------------------------------------------- #
# plot_telemetry script (shared readers stubbed out)
# --------------------------------------------------------------------------- #
def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script(monkeypatch, tmp_path):
    module = load_script("plot_telemetry")
    row = SimpleNamespace(pair_id="000", task=SimpleNamespace(value="pick_up_object"),
                          success_file="success_000.csv", failure_file="failure_000.csv",
                          failed_subtask=SimpleNamespace(value="grasp"), failure_mode=SimpleNamespace(value="inadequate_grip_strength"))
    monkeypatch.setattr(module, "_read_metadata", lambda p: [row])
    monkeypatch.setattr(module, "_read_telemetry", lambda p: make_rows("failure" in p.name))
    monkeypatch.setattr(module, "_load_bt", lambda p: make_bt())
    bt = tmp_path / "bt"
    bt.mkdir()
    (bt / "pair_000_bt.json").write_text("{}")  # existence only; _load_bt is stubbed
    return module, bt


def test_script_writes_one_png_per_pair(script, tmp_path, capsys):
    module, bt = script
    out = tmp_path / "plots"
    assert module.main(["--data-dir", str(tmp_path), "--bt-dir", str(bt), "--output-dir", str(out)]) == 0
    png = out / "telemetry_000.png"
    assert png.read_bytes()[:4] == b"\x89PNG" and str(png) in capsys.readouterr().out


def test_script_adds_injection_and_predicted_onset_markers(script, tmp_path, monkeypatch):
    module, bt = script
    seen = {}
    real = module.plot_pair_telemetry
    monkeypatch.setattr(module, "plot_pair_telemetry", lambda *a, **k: seen.update(k) or real(*a, **k))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": {"000": {"injection_timestamp": 5.1}}}))
    preds = tmp_path / "p.jsonl"
    write_predictions(preds, [{
        "pair_id": "000", "model_variant": "baseline_1", "predicted_failure_mode": "inadequate_grip_strength",
        "predicted_subtask": "grasp", "explanation": "x", "status": "ok", "backend": "stub",
        "evidence": [{"subtask": "grasp", "timestamp": 5.12, "feature": "gripper_force", "value": 3.0}]}])
    args = ["--data-dir", str(tmp_path), "--bt-dir", str(bt), "--output-dir", str(tmp_path / "o"),
            "--manifest", str(manifest), "--predictions", str(preds)]
    assert module.main(args) == 0
    assert seen["injection_time"] == 5.1 and seen["predicted_onset"] == 5.12
    assert "injected inadequate_grip_strength in grasp" in seen["title"]
    assert "SYNTHETIC" in seen["note"]  # stub predictions are labelled on the figure


def test_script_still_plots_raw_traces_when_the_bt_is_missing(script, tmp_path):
    module, bt = script
    (bt / "pair_000_bt.json").unlink()
    out = tmp_path / "o"
    assert module.main(["--data-dir", str(tmp_path), "--bt-dir", str(bt), "--output-dir", str(out)]) == 0
    assert (out / "telemetry_000.png").is_file()


def test_script_rejects_unknown_pair_ids(script, tmp_path, capsys):
    module, bt = script
    assert module.main(["--data-dir", str(tmp_path), "--bt-dir", str(bt), "--output-dir", str(tmp_path / "o"),
                        "--pair-id", "999"]) == 2
    assert "999" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()
