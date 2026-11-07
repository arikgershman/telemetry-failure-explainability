"""Matplotlib figures for telemetry validation and evaluation results.

Figures are built with the object-oriented API (``Figure`` + Agg canvas), never
through pyplot, so nothing here depends on a display or on global backend state.
Every function returns the ``Figure``; ``save_figure`` writes it.

Telemetry figure, top to bottom
    one panel per ``PanelSpec`` (raw success and failure traces, retained
    samples from the compressed BT drawn as markers), then a divergence panel.
    All panels share one time axis (s) and carry:
      * subtask boundaries (failure run dashed, success run dotted when different)
      * injection time (manifest), predicted onset, and first divergence (computed),
        each with its own colour *and* line style, since they are different events
Divergence ``null`` means *not computed* and is shown as such, never as zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from telemetry_reasoning.eval.contracts import FEATURE_SHAPES

FEATURE_UNITS: Mapping[str, str] = {
    "joint_positions": "rad",
    "joint_velocities": "rad/s",
    "joint_torques": "N m",
    "gripper_state": "closure, 0=open 1=closed",
    "gripper_force": "N",
    "ee_pos": "m",
    "ee_quat": "w/x/y/z, unitless",
    "obj_pos": "m",
    "obj_quat": "w/x/y/z, unitless",
    "obj_vel": "m/s",
    "contact_force": "N",
}

C_SUCCESS = "#1f77b4"
C_FAILURE = "#d62728"
C_INJECTION = "#2ca02c"
C_ONSET = "#9467bd"
C_DIVERGENCE = "#ff7f0e"
LINESTYLES = ("-", "--", ":", "-.")


# --------------------------------------------------------------------------- #
# Traces
# --------------------------------------------------------------------------- #
@dataclass
class TelemetryTrace:
    t: np.ndarray
    subtask: tuple[str, ...]
    features: dict[str, np.ndarray] = field(default_factory=dict)

    def series(self, feature: str, component: Optional[int] = None) -> np.ndarray:
        arr = self.features[feature]
        if component is None:
            if arr.ndim != 1:
                raise ValueError(f"{feature} is a vector; choose a component")
            return arr
        return arr[:, component]

    def spans(self) -> list[tuple[str, float, float]]:
        """Consecutive (subtask, first_t, last_t) runs, in time order."""
        out: list[tuple[str, float, float]] = []
        for name, t in zip(self.subtask, self.t):
            if out and out[-1][0] == name:
                out[-1] = (name, out[-1][1], float(t))
            else:
                out.append((name, float(t), float(t)))
        return out


def _get(row: Any, name: str) -> Any:
    value = row.get(name) if isinstance(row, Mapping) else getattr(row, name, None)
    return getattr(value, "value", value)  # enums -> strings


def trace_from_rows(rows: Any) -> TelemetryTrace:
    """Build a trace from validated rows, mappings, a DataFrame, or BT samples.

    Vector cells may be lists or JSON strings. A feature missing from any row
    is left out of the trace rather than guessed.
    """
    if hasattr(rows, "to_dict"):  # pandas DataFrame
        rows = rows.to_dict("records")
    rows = list(rows)
    if not rows:
        raise ValueError("cannot build a trace from zero rows")

    t = np.asarray([float(_get(r, "timestamp")) for r in rows])
    subtask = tuple(str(_get(r, "subtask")) for r in rows)
    features: dict[str, np.ndarray] = {}
    for name in FEATURE_SHAPES:
        cells = [_get(r, name) for r in rows]
        if any(c is None for c in cells):
            continue
        cells = [json.loads(c) if isinstance(c, str) else c for c in cells]
        features[name] = np.asarray(cells, dtype=float)
    return TelemetryTrace(t, subtask, features)


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Series:
    feature: str
    component: Optional[int]
    label: str


@dataclass(frozen=True)
class PanelSpec:
    title: str
    ylabel: str  # include units
    series: tuple[Series, ...]


DEFAULT_PANELS: tuple[PanelSpec, ...] = (
    PanelSpec("Gripper force", "Force (N)", (Series("gripper_force", None, "gripper force"),)),
    PanelSpec("Contact force", "Force (N)", (Series("contact_force", None, "contact force"),)),
    PanelSpec(
        "Gripper closure",
        "Closure (0 = open, 1 = closed)",
        (Series("gripper_state", None, "closure"),),
    ),
    PanelSpec(
        "Height: end effector vs object",
        "z (m)",
        (Series("ee_pos", 2, "end effector z"), Series("obj_pos", 2, "object z")),
    ),
)


def _divergence_nodes(bt: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-subtask (start, end, divergence) from a ``model_dump`` of a PairedBehaviorTree."""
    if not bt:
        return []
    nodes = []
    subtasks = bt.get("subtasks", [])
    starts = [min(float(r["timestamp"]) for r in n["failure_samples"]) for n in subtasks]
    ends = [max(float(r["timestamp"]) for r in n["failure_samples"]) for n in subtasks]
    for i, node in enumerate(subtasks):
        end = starts[i + 1] if i + 1 < len(subtasks) else ends[i]
        nodes.append(
            {
                "subtask": str(getattr(node["subtask"], "value", node["subtask"])),
                "start": starts[i],
                "end": end,
                "divergence": node.get("deterministic_divergence"),
            }
        )
    return nodes


def _delta_for(divergence: Mapping[str, Any], feature: str) -> Optional[float]:
    value = (divergence.get("max_abs_deltas") or {}).get(feature)
    if value is None:
        return None
    return float(max(value)) if isinstance(value, (list, tuple)) else float(value)


# --------------------------------------------------------------------------- #
# Telemetry figure
# --------------------------------------------------------------------------- #
def plot_pair_telemetry(
    success: TelemetryTrace,
    failure: TelemetryTrace,
    *,
    bt: Optional[Mapping[str, Any]] = None,
    injection_time: Optional[float] = None,
    predicted_onset: Optional[float] = None,
    panels: Sequence[PanelSpec] = DEFAULT_PANELS,
    divergence_feature: str = "gripper_force",
    title: Optional[str] = None,
    note: Optional[str] = None,
) -> Figure:
    """Raw success/failure telemetry with retained samples, boundaries, events, divergence.

    ``bt`` is ``PairedBehaviorTree.model_dump(mode="json")``; it supplies the
    retained samples and the computed divergence. Without it only raw traces
    are drawn.
    """
    success_ret = failure_ret = None
    if bt:
        success_ret = trace_from_rows([r for n in bt["subtasks"] for r in n["success_samples"]])
        failure_ret = trace_from_rows([r for n in bt["subtasks"] for r in n["failure_samples"]])
    nodes = _divergence_nodes(bt)

    fig = Figure(figsize=(11, 2.3 * (len(panels) + 1) + 1.4), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(panels) + 1, 1, sharex=True)
    top = axes[0]

    f_spans = failure.spans()
    f_bounds = [s[1] for s in f_spans[1:]]
    s_bounds = [s[1] for s in success.spans()[1:]]
    different = [b for b in s_bounds if all(abs(b - f) > 1e-9 for f in f_bounds)]
    first_div = [n["divergence"]["first_divergence_timestamp"] for n in nodes
                 if n["divergence"] and n["divergence"].get("first_divergence_timestamp") is not None]

    def decorate(ax: Any, first: bool) -> None:
        """Bands, boundaries, and event markers shared by every panel."""
        for i, (_, a, b) in enumerate(f_spans):
            if i % 2 == 0:
                ax.axvspan(a, b, color="0.5", alpha=0.07, lw=0)
        for k, b in enumerate(f_bounds):
            ax.axvline(b, color="0.35", ls="--", lw=0.8, label="subtask boundary (failure run)" if first and k == 0 else "_nolegend_")
        for k, b in enumerate(different):
            ax.axvline(b, color=C_SUCCESS, ls=":", lw=0.9, label="subtask boundary (success run)" if first and k == 0 else "_nolegend_")
        for k, ts in enumerate(first_div):
            ax.axvline(ts, color=C_DIVERGENCE, ls="--", lw=1.6, label="first divergence (computed)" if first and k == 0 else "_nolegend_")
        if injection_time is not None:
            ax.axvline(injection_time, color=C_INJECTION, ls="-.", lw=1.8, label="injection time (manifest)" if first else "_nolegend_")
        if predicted_onset is not None:
            ax.axvline(predicted_onset, color=C_ONSET, ls=":", lw=2.0, label="predicted onset" if first else "_nolegend_")

    for i, (ax, panel) in enumerate(zip(axes[:-1], panels)):
        series_handles = []
        for k, s in enumerate(panel.series):
            ls = LINESTYLES[k % len(LINESTYLES)]
            for run, trace, color in (("success", success, C_SUCCESS), ("failure", failure, C_FAILURE)):
                (line,) = ax.plot(trace.t, trace.series(s.feature, s.component), color=color, ls=ls, lw=1.3,
                                  label=f"{run}: {s.label}")
                series_handles.append(line)
        for run, ret, color, marker, fill in (
            ("success", success_ret, C_SUCCESS, "o", "white"),
            ("failure", failure_ret, C_FAILURE, "o", C_FAILURE),
        ):
            if ret is None:
                continue
            for s in panel.series:
                ax.plot(ret.t, ret.series(s.feature, s.component), ls="none", marker=marker, ms=4.5,
                        mfc=fill, mec=color, color=color, label=f"{run} retained samples")
        decorate(ax, first=(i == 0))
        ax.set_ylabel(panel.ylabel)
        ax.set_title(panel.title, loc="left", fontsize=9)
        ax.legend(handles=series_handles, fontsize=7, loc="upper left", ncols=2, framealpha=0.85)
        ax.grid(alpha=0.25)

    # subtask names above the top panel
    for name, a, b in f_spans:
        top.text((a + b) / 2, 1.02, name, transform=top.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)

    div_ax = axes[-1]
    decorate(div_ax, first=False)
    unit = FEATURE_UNITS.get(divergence_feature, "")
    div_ax.set_ylabel(f"max |Δ| {divergence_feature}\n({unit})")
    div_ax.set_title("Deterministic divergence per subtask (success vs failure)", loc="left", fontsize=9)
    computed = [n for n in nodes if n["divergence"] is not None]
    if not nodes:
        div_ax.text(0.5, 0.5, "no behavior tree supplied", transform=div_ax.transAxes, ha="center", va="center", color="0.4")
    elif not computed:
        div_ax.text(0.5, 0.5, "divergence not computed (null, not zero)", transform=div_ax.transAxes,
                    ha="center", va="center", color="0.4")
    for n in computed:
        height = _delta_for(n["divergence"], divergence_feature)
        if height is not None:
            div_ax.bar(n["start"], height, width=max(n["end"] - n["start"], 1e-6), align="edge",
                       color=C_DIVERGENCE, alpha=0.6, edgecolor="0.3", label=f"max |Δ| ({n['subtask']})")
    div_ax.set_xlabel("Time (s)")
    div_ax.grid(alpha=0.25)

    # one figure-level legend for everything that is not a data series
    extras: list[tuple[str, Any]] = []
    if success_ret is not None:
        extras.append(("success retained samples",
                       Line2D([], [], ls="none", marker="o", mfc="white", mec=C_SUCCESS, color=C_SUCCESS)))
    if failure_ret is not None:
        extras.append(("failure retained samples",
                       Line2D([], [], ls="none", marker="o", mfc=C_FAILURE, mec=C_FAILURE, color=C_FAILURE)))
    shared = ("subtask boundary", "first divergence", "injection", "predicted onset")
    listed: set[str] = set()
    for line in top.lines:
        label = line.get_label()
        if label.startswith(shared) and label not in listed:
            listed.add(label)
            extras.append((label, line))
    if extras:
        labels, handles = [e[0] for e in extras], [e[1] for e in extras]
        try:
            fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8)
        except ValueError:  # matplotlib < 3.7 has no "outside" placement
            fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=8)

    fig.suptitle(title or "Telemetry: success vs failure", fontsize=12)
    if note:
        fig.text(0.99, 0.995, note, ha="right", va="top", color="firebrick", fontsize=9, weight="bold")
    return fig


# --------------------------------------------------------------------------- #
# Evaluation figures
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(slice_metrics: Mapping[str, Any], title: str, *, synthetic: bool = False) -> Figure:
    """Counts of true label (rows) vs predicted label / non-prediction outcome (columns)."""
    conf = slice_metrics["confusion"]
    matrix = np.asarray(conf["matrix"], dtype=float)
    fig = Figure(figsize=(8.5, 5.0), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1.0, matrix.max()))
    ax.set_xticks(range(len(conf["columns"])), conf["columns"], rotation=40, ha="right")
    ax.set_yticks(range(len(conf["labels"])), conf["labels"])
    ax.set_xlabel("predicted (or non-prediction outcome)")
    ax.set_ylabel("true failure mode")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = int(matrix[i, j])
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > matrix.max() / 2 and v else "black", fontsize=9)
    ax.axvline(len(conf["labels"]) - 0.5, color="0.3", lw=1.2)  # predictions | non-predictions
    fig.colorbar(im, ax=ax, label="pairs")
    ax.set_title(f"{title}  (n={slice_metrics['n']})", fontsize=10)
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC", ha="center", va="center", fontsize=48, color="firebrick", alpha=0.18, weight="bold")
    return fig


def plot_variant_comparison(
    result: Mapping[str, Any],
    metrics: Sequence[str] = ("failure_mode_accuracy", "macro_f1", "failed_subtask_accuracy", "coverage"),
    slice_name: str = "all",
) -> Figure:
    """Grouped bars: one group per model variant. Unavailable values (``None``) are labelled n/a."""
    variants = list(result["variants"])
    fig = Figure(figsize=(1.6 * max(len(variants), 3) + 3, 4.6), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    width = 0.8 / max(len(metrics), 1)
    for k, metric in enumerate(metrics):
        xs, ys, labels = [], [], []
        for i, v in enumerate(variants):
            val = result["variants"][v]["slices"][slice_name][metric]
            xs.append(i + (k - (len(metrics) - 1) / 2) * width)
            ys.append(0.0 if val is None else val)
            labels.append("n/a" if val is None else f"{val:.2f}")
        bars = ax.bar(xs, ys, width=width, label=metric.replace("_", " "))
        ax.bar_label(bars, labels=labels, fontsize=7, padding=2)
    ticks = [f"{v}\n(SYNTHETIC)" if result["variants"][v]["synthetic"] else v for v in variants]
    ax.set_xticks(range(len(variants)), ticks)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of ground-truth pairs")
    ax.set_title(f"Variant comparison ({slice_name}, n={result['n_truth_pairs']})", fontsize=10)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))  # outside: never hides a bar
    ax.grid(axis="y", alpha=0.25)
    if any(result["variants"][v]["synthetic"] for v in variants):
        fig.text(0.5, 0.5, "SYNTHETIC", ha="center", va="center", fontsize=48, color="firebrick", alpha=0.15, weight="bold")
    return fig


def save_figure(fig: Figure, path: Union[str, Path], dpi: int = 150) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    return out
