#!/usr/bin/env python
r"""Integration smokes: a CPU contract smoke, plus optional real-hardware smokes.

CPU contract smoke (default; stub backend, mock data, offline)::

    python scripts/integration_smoke.py --backend stub --output-dir runs/integration-cpu

Runs every stage through its public CLI, in a fresh output directory, with an explicit file list at
each hand-off so earlier runs or manifests cannot pollute the evaluation:

    generate_mock_pair -> parse_dataset -> enrich_divergence -> run_baseline (Baseline 1)
    -> run_llm --backend stub (Baselines 2/3/main) -> evaluate -> plot_telemetry

and asserts the contracts between them: valid CSVs, the known grasp event (15 N -> 3 N at 5.10 s, a
12 N drop) at the raw, retained and computed-divergence levels, label/view exclusions for all three
prompt views, one prediction per (pair, variant), finite metrics, and nonempty plots. Baselines 2/3/main
run on the stub backend, so their scores are SYNTHETIC plumbing output.

Real-hardware smokes (optional; need the released dataset batch)::

    python scripts/integration_smoke.py --sim-backend mujoco --backend hf --tasks all \
        --cover-dataset-failure-modes --dataset-dir outputs/dataset/batch_20261030_152125_e5ca72de \
        --output-dir runs/integration-real

``--sim-backend mujoco`` is the real producer smoke and ``--backend hf`` the real model smoke; either
can be run alone. The CPU contract smoke always runs first, in ``<output-dir>/cpu-contract/``, so real
and stub results never share a directory. Each real stage is reported as passed / failed / blocked /
not_run on its own in ``smoke_report.json``; a stage that did not run is never reported as passed.

* Eligible stages are the (mode, subtask) pairs the RELEASED dataset actually used, read from the
  batch's ``<task>/metadata.csv`` and ``<task>/diagnostics.jsonl`` (the per-task manifests carry
  neither) and cross-checked against ``handoff/<task>.json``. One released pair per stage is replayed
  with its own condition, seed and failure config. Pairs come from dev, then train; the held-out test
  split is never used.
* Producer smoke: for each replayed pair, require nominal success, injection activation, a failed
  injected run, an observed failure trace, schema-valid traces, and replay agreement (atol 1e-8). The
  excessive_speed injection is probed for activation only. Accepted pairs go through parse ->
  divergence -> Baseline 1 -> evaluate -> plots.
* Model smoke: Baselines 2/3/main on the Hugging Face backend, plus Baseline 1, on the same parsed
  pairs under the ``json_or_fenced_v1`` response policy (applied by a config override). The run
  manifest must show real weights loaded and generation completed; a stub run fails these checks.
* Onset: the task manifests carry only ``injection_timestamp_s`` (absolute simulator time), not the
  evaluator's CSV-clock ``injection_timestamp``, so onset is reported as unscored rather than guessed.

Writes ``<output-dir>/smoke_report.json`` plus per-stage logs. Exit codes: 0 = every requested check
passed, 1 = a contract check failed, 2 = setup error, blocked hardware, or a stage CLI failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIGS = ROOT / "configs"
# Tests substitute a double for one stage CLI (for example the Hugging Face ``run_llm.py``); empty in real use.
SCRIPT_OVERRIDES: dict[str, Path] = {}

VARIANTS = ("baseline_1", "baseline_2", "baseline_3", "main")
LLM_VARIANTS = ("baseline_2", "baseline_3", "main")
EXPECTED_BACKEND = {"baseline_1": "rules", "baseline_2": "stub", "baseline_3": "stub", "main": "stub"}

# The mock fixture's documented injected event: grip drops 15 N -> 3 N at 5.10 s.
GRASP_EVENT_S = 5.10
GRASP_FORCE_DROP_N = 12.0
EVENT_TOL = 1e-6

# Storage-only fields that must never appear in any model-facing view.
FORBIDDEN_VIEW_KEYS = frozenset(
    {"failed_subtask", "failure_mode", "perturbation_param", "perturbation_value",
     "success_file", "failure_file", "seed"}
)
DIVERGENCE_KEYS = ("deterministic_divergence", "divergence_status")

BANNER = (
    "SYNTHETIC CPU CONTRACT SMOKE: Baselines 2/3/main ran on the stub backend, so their "
    "scores are plumbing output, not model results. MuJoCo and Hugging Face were not exercised."
)
REAL_CHECKS_NOTE = (
    "Not run by the CPU contract smoke. Run them with --sim-backend mujoco / --backend hf (see the module docstring). "
    "Never mark the real-MuJoCo producer smoke or the real-HF model smoke passed because this stub smoke passed."
)


class SmokeError(RuntimeError):
    """A stage CLI failed or the setup was unusable (exit code 2)."""


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested directly)
# --------------------------------------------------------------------------- #
def find_view_leaks(view: Any, forbidden_values: Sequence[str] = ()) -> list[str]:
    """Paths of forbidden storage keys, or strings containing a forbidden value, inside a view.

    Keys are checked at any depth; ``forbidden_values`` (the true failure mode, source
    filenames, ...) are matched as substrings of every string value.
    """
    leaks: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                here = f"{path}.{key}" if path else str(key)
                if key in FORBIDDEN_VIEW_KEYS:
                    leaks.append(f"key {here}")
                walk(value, here)
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, str):
            for bad in forbidden_values:
                if bad and bad in node:
                    leaks.append(f"value {bad!r} at {path}")

    walk(view, "")
    return leaks


def check_views(tree: Any, forbidden_values: Sequence[str]) -> list[tuple[str, bool, str]]:
    """Structural and label exclusions for the three model-facing views of one tree."""
    views = {
        "failure_only": tree.to_failure_only(),
        "paired_without_divergence": tree.to_paired_without_divergence(),
        "full_paired": tree.to_full_paired(),
    }
    wants_success = {"failure_only": False, "paired_without_divergence": True, "full_paired": True}
    wants_divergence = {"failure_only": False, "paired_without_divergence": False, "full_paired": True}
    results: list[tuple[str, bool, str]] = []
    for name, view in views.items():
        nodes = view["subtasks"]
        has_success = all(bool(node.get("success_samples")) for node in nodes)
        no_success = all("success_samples" not in node for node in nodes)
        results.append((f"view.{name}.success_samples",
                        has_success if wants_success[name] else no_success,
                        "present" if wants_success[name] else "excluded"))
        has_div = all(key in node for node in nodes for key in DIVERGENCE_KEYS)
        no_div = all(key not in node for node in nodes for key in DIVERGENCE_KEYS)
        results.append((f"view.{name}.divergence",
                        has_div if wants_divergence[name] else no_div,
                        "present" if wants_divergence[name] else "excluded"))
        leaks = find_view_leaks(view, forbidden_values)
        results.append((f"view.{name}.no_label_leakage", not leaks, "; ".join(leaks[:5]) or "clean"))
    return results


def measure_grasp_event(success: Sequence[Any], failure: Sequence[Any]) -> tuple[float, float]:
    """First time the failure run's gripper force departs from the success run, and the largest drop.

    Both runs share initial conditions, so any early difference is the injected event.
    Raises ``ValueError`` if the runs are not time-aligned or never differ.
    """
    if len(success) != len(failure) or any(
        abs(a.timestamp - b.timestamp) > 1e-9 for a, b in zip(success, failure)
    ):
        raise ValueError("success and failure runs are not on the same time grid")
    first: Optional[float] = None
    drop = 0.0
    for a, b in zip(success, failure):
        diff = a.gripper_force - b.gripper_force
        if abs(diff) > 1e-6 and first is None:
            first = float(a.timestamp)
        drop = max(drop, diff)
    if first is None:
        raise ValueError("gripper force never differs between the success and failure runs")
    return first, float(drop)


def all_finite(node: Any) -> bool:
    """True when every number inside a JSON-like structure is finite."""
    if isinstance(node, bool):
        return True
    if isinstance(node, (int, float)):
        return math.isfinite(node)
    if isinstance(node, Mapping):
        return all(all_finite(v) for v in node.values())
    if isinstance(node, (list, tuple)):
        return all(all_finite(v) for v in node)
    return True


def is_nonempty_png(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1000 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def require_fresh_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SmokeError(
            f"output directory {path} is not empty; use a fresh directory so earlier runs and "
            "manifests cannot pollute this smoke"
        )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class Recorder:
    """Collects stage results and contract checks (a plain class: this script is also loaded by path in tests)."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.steps: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "passed": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if not c["passed"]]

    def run(self, name: str, script: str, args: Sequence[Any], ok_codes: Sequence[int] = (0,)) -> subprocess.CompletedProcess:
        """Run ``scripts/<script>`` as a subprocess; an exit code outside ``ok_codes`` is a ``SmokeError``."""
        argv = [sys.executable, str(SCRIPT_OVERRIDES.get(script, SCRIPTS / script)), *map(str, args)]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH", "")]))
        env.update(MPLBACKEND="Agg", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")  # CPU/offline by construction
        start = time.perf_counter()
        proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
        duration = time.perf_counter() - start
        log = self.out / "logs" / f"{len(self.steps) + 1:02d}_{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"$ {' '.join(argv)}\nexit code: {proc.returncode}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
        self.steps.append({"name": name, "script": script, "returncode": proc.returncode,
                           "duration_s": round(duration, 3), "log": str(log.relative_to(self.out))})
        if proc.returncode not in ok_codes:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            raise SmokeError(f"stage '{name}' ({script}) exited {proc.returncode}; see {log}\n  " + "\n  ".join(tail))
        return proc


def _load_json(path: Path) -> Any:
    def reject(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant {token} in {path}")

    return json.loads(path.read_text(), parse_constant=reject)


def run_smoke(output_dir: Path, *, seed: int = 723) -> dict[str, Any]:
    """Run the CPU contract smoke and return the report (also written to ``smoke_report.json``)."""
    from telemetry_reasoning.eval.contracts import index_predictions, read_predictions
    from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv, validate_raw_run
    from telemetry_reasoning.schemas import PairedBehaviorTree

    require_fresh_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(output_dir)
    mock, raw_bt, bt = output_dir / "mock", output_dir / "bt_raw", output_dir / "bt"
    runs, llm = output_dir / "runs", output_dir / "runs" / "llm"
    evaluation, plots = output_dir / "eval", output_dir / "plots"

    # 1. fresh mock pair, validated through the public CSV readers -------------------------
    rec.run("generate_mock_pair", "generate_mock_pair.py", ["--output-dir", mock, "--seed", seed])
    metadata = read_metadata_csv(mock / "metadata.csv")
    pair_ids = [row.pair_id for row in metadata]
    rec.check("input.metadata_valid", bool(metadata), f"{len(metadata)} pair(s): {pair_ids}")
    grasp_events: dict[str, float] = {}
    for row in metadata:
        success = read_telemetry_csv(mock / row.success_file)
        failure = read_telemetry_csv(mock / row.failure_file)
        validate_raw_run(success)
        validate_raw_run(failure)
        rec.check(f"input.{row.pair_id}.csv_valid_50hz", True, f"{len(success)} + {len(failure)} rows at 0.02 s")
        rec.check(f"input.{row.pair_id}.paired_joint_dims",
                  len(success[0].joint_positions) == len(failure[0].joint_positions), "")
        when, drop = measure_grasp_event(success, failure)
        grasp_events[row.pair_id] = when
        rec.check(f"input.{row.pair_id}.grasp_event_time", abs(when - GRASP_EVENT_S) <= EVENT_TOL,
                  f"first force departure at {when:.2f} s (expected {GRASP_EVENT_S:.2f})")
        rec.check(f"input.{row.pair_id}.grasp_force_drop", abs(drop - GRASP_FORCE_DROP_N) <= EVENT_TOL,
                  f"{drop:.2f} N drop (expected {GRASP_FORCE_DROP_N:.1f})")

    # 2. parse, then divergence (separate public stages; each needs a fresh directory) ---------
    rec.run("parse_dataset", "parse_dataset.py", [
        "--metadata", mock / "metadata.csv", "--data-dir", mock,
        "--config", CONFIGS / "parser.yaml", "--output-dir", raw_bt])
    counts = _load_json(raw_bt / "parse_report.json")["counts"]
    rec.check("parse.all_pairs_written", counts["written_pairs"] == len(pair_ids) and counts["rejected_pairs"] == 0, str(counts))
    rec.run("enrich_divergence", "enrich_divergence.py", [
        "--bt-dir", raw_bt, "--config", CONFIGS / "divergence.yaml", "--output-dir", bt])
    div_report = _load_json(bt / "divergence_report.json")
    rec.check("divergence.all_pairs_enriched", div_report["pair_count"] == len(pair_ids), f"pair_count={div_report['pair_count']}")

    # 3. BT-level contracts: event retention, computed divergence, the three views -------------
    forbidden_base = ("failure_mode",)
    for row in metadata:
        tree = PairedBehaviorTree.model_validate_json((bt / f"pair_{row.pair_id}_bt.json").read_text())
        grasp = next(n for n in tree.subtasks if n.subtask.value == "grasp")
        kept = [(r.timestamp, r.gripper_force) for r in grasp.failure_samples if abs(r.timestamp - GRASP_EVENT_S) <= EVENT_TOL]
        rec.check(f"bt.{row.pair_id}.event_sample_retained", bool(kept) and abs(kept[0][1] - 3.0) <= EVENT_TOL,
                  f"failure sample at {GRASP_EVENT_S:.2f} s: {kept}")
        div = grasp.deterministic_divergence
        first = div.first_divergence_timestamp if div else None
        rec.check(f"bt.{row.pair_id}.computed_divergence_at_event",
                  grasp.divergence_status == "computed" and first is not None and abs(first - GRASP_EVENT_S) <= EVENT_TOL,
                  f"grasp first_divergence_timestamp={first}")
        force_delta = (div.max_abs_deltas.get("gripper_force") if div else None)
        rec.check(f"bt.{row.pair_id}.divergence_force_delta",
                  force_delta is not None and abs(force_delta - GRASP_FORCE_DROP_N) <= 1e-3, f"max |dF|={force_delta}")
        rec.check(f"bt.{row.pair_id}.every_observed_stage_computed",
                  all(n.divergence_status == "computed" for n in tree.subtasks if n.failure_samples), "")
        forbidden = (str(row.failure_mode.value), row.success_file, row.failure_file, *forbidden_base)
        for name, ok, detail in check_views(tree, forbidden):
            rec.check(f"bt.{row.pair_id}.{name}", ok, detail)

    # 4. Baseline 1 (rules) and stubbed Baselines 2/3/main, from the enriched BTs ----------------
    rec.run("run_baseline", "run_baseline.py", [
        "--bt-dir", bt, "--config", CONFIGS / "rules.yaml", "--output", runs / "baseline_1.jsonl"])
    rec.run("run_llm_stub", "run_llm.py", [
        "--bt-dir", bt, "--backend", "stub", "--variants", *LLM_VARIANTS, "--output-dir", llm])
    prediction_files = {"baseline_1": runs / "baseline_1.jsonl", **{v: llm / f"{v}.jsonl" for v in LLM_VARIANTS}}
    for variant, path in prediction_files.items():
        parsed = read_predictions(path)
        rec.check(f"predictions.{variant}.valid", not parsed.invalid, f"{len(parsed.records)} valid, {len(parsed.invalid)} invalid")
        index = index_predictions(parsed, pair_ids)  # rejects duplicates and unknown pair IDs
        rec.check(f"predictions.{variant}.one_per_pair",
                  sorted(k[0] for k in index.by_key) == sorted(pair_ids) and all(k[1] == variant for k in index.by_key),
                  f"{len(index.by_key)} record(s) for {len(pair_ids)} pair(s)")
        rec.check(f"predictions.{variant}.backend_label",
                  {r.backend for r in parsed.records} == {EXPECTED_BACKEND[variant]}, f"expected {EXPECTED_BACKEND[variant]}")

    # 5. evaluation: explicit file lists; the manifest sidecar carries the event time as the
    #    injection reference so the split and onset paths run too -----------------------------------
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "provenance": "synthetic mock fixture: injection time = first gripper_force departure between the raw CSVs",
        "pairs": {pid: {"split": "test", "injection_timestamp": t} for pid, t in grasp_events.items()},
    }, indent=2) + "\n")
    proc = rec.run("evaluate", "evaluate.py", [
        "--metadata", mock / "metadata.csv",
        "--predictions", *[prediction_files[v] for v in VARIANTS],
        "--variants", *VARIANTS,
        "--bt-dir", bt, "--manifest", manifest, "--split", "test",
        "--run-manifest", runs / "baseline_1.manifest.json", llm / "run_manifest.json",
        "--config", CONFIGS / "eval.yaml", "--output-dir", evaluation])
    metrics = _load_json(evaluation / "metrics.json")
    rec.check("metrics.all_values_finite", all_finite(metrics), "")
    rec.check("metrics.every_variant_scored", list(metrics["variants"]) == list(VARIANTS), str(list(metrics["variants"])))
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        entry = metrics["variants"][variant]
        sl = entry["slices"]["all"]
        accounted = sum(sl[k] for k in ("n_ok", "n_abstain", "n_error", "n_invalid", "n_missing"))
        rec.check(f"metrics.{variant}.one_result_per_pair", sl["n"] == len(pair_ids) == accounted, f"n={sl['n']}, accounted={accounted}")
        rec.check(f"metrics.{variant}.nothing_missing_or_invalid", sl["n_missing"] == 0 and sl["n_invalid"] == 0, "")
        rec.check(f"metrics.{variant}.synthetic_flag", entry["synthetic"] is (variant != "baseline_1"), f"synthetic={entry['synthetic']}")
        res = entry["resources"]
        rec.check(f"metrics.{variant}.resources_reported", res.get("reported") is True, f"source={res.get('source_manifest')}")
        summary[variant] = {"synthetic": entry["synthetic"], "backends": entry["backends"], "n": sl["n"],
                            "n_ok": sl["n_ok"], "n_abstain": sl["n_abstain"], "n_error": sl["n_error"],
                            "failure_mode_accuracy": sl["failure_mode_accuracy"]}
    for variant in LLM_VARIANTS:
        rec.check(f"metrics.{variant}.token_counts_logged",
                  metrics["variants"][variant]["resources"].get("input_tokens_total") is not None, "from run_llm run_manifest.json")
    b1 = metrics["variants"]["baseline_1"]
    rec.check("baseline_1.finds_known_low_grip",
              b1["slices"]["all"]["failure_mode_accuracy"] == 1.0 and b1["slices"]["all"]["failed_subtask_accuracy"] == 1.0,
              "inadequate_grip_strength in grasp")
    rec.check("baseline_1.evidence_verified_and_supported",
              b1["evidence"]["checked"] and b1["evidence"]["n_items"] > 0 and b1["evidence"]["n_unsupported"] == 0
              and b1["evidence"]["n_unverifiable"] == 0, str({k: b1["evidence"][k] for k in ("n_items", "n_unsupported", "n_unverifiable")}))
    rec.check("baseline_1.onset_scored", b1["onset"].get("computed") and b1["onset"]["n_scored"] == len(pair_ids),
              str({k: b1["onset"].get(k) for k in ("n_with_reference", "n_scored", "mean_error_s")}))
    rec.check("evaluate.prints_synthetic_banner", "SYNTHETIC" in proc.stdout, "")
    key = _load_json(evaluation / "rubric_key.json") if (evaluation / "rubric_key.json").is_file() else []
    rec.check("rubric.excludes_stub_output", bool(key) and {k["backend"] for k in key} == {"rules"}, f"{len(key)} rubric item(s)")

    # 6. plots --------------------------------------------------------------------------------------
    rec.run("plot_telemetry", "plot_telemetry.py", [
        "--data-dir", mock, "--bt-dir", bt, "--manifest", manifest,
        "--predictions", runs / "baseline_1.jsonl", "--output-dir", plots])
    for pid in pair_ids:
        png = plots / f"telemetry_{pid}.png"
        rec.check(f"plots.telemetry_{pid}", is_nonempty_png(png), f"{png.stat().st_size if png.is_file() else 0} bytes")
    for name in ["variant_comparison.png"] + [f"confusion_{v}_all.png" for v in VARIANTS]:
        rec.check(f"plots.eval.{name}", is_nonempty_png(evaluation / "plots" / name), "")

    report = {
        "smoke": "cpu_contract", "backend": "stub", "sim_backend": "mock", "synthetic": True, "banner": BANNER,
        "seed": seed, "output_dir": str(output_dir), "n_pairs": len(pair_ids), "pair_ids": pair_ids,
        "passed": not rec.failed, "n_checks": len(rec.checks), "n_failed": len(rec.failed),
        "steps": rec.steps, "checks": rec.checks, "variants": summary,
        "real_checks": {"real_producer_smoke": "not_run", "real_model_smoke": "not_run", "note": REAL_CHECKS_NOTE},
    }
    (output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


# --------------------------------------------------------------------------- #
# Real-hardware smokes: pure helpers (no MuJoCo, no model, no network)
# --------------------------------------------------------------------------- #
DATASET_TASKS = ("pick_up_object", "transport_object")
DATASET_MODES = ("position_error", "orientation_error", "inadequate_grip_strength", "early_gripper_release")
SPLITS = ("train", "dev", "test")
# Pairs are drawn from dev, then train. The held-out test split is never touched.
SMOKE_SPLITS = ("dev", "train")
RESPONSE_POLICY = "json_or_fenced_v1"
SIM_SPREAD = 0.015  # the released dataset's producer command used --spread 0.015
REPLAY_ATOL = 1e-8
# excessive_speed has no accepted dataset examples, so there is no released config to replay. These
# uncalibrated probes mirror failure_viewer's defaults and only test that the injection activates.
EXCESSIVE_SPEED_PROBES = {"pick_up_object": ("approach", 4.0), "transport_object": ("move_along_path", 4.0)}
TOKEN_BOUND_METHOD = "utf8_bytes_upper_bound"
_PAIR_KEYS = ("task", "pair_id", "initial_condition_id", "failure_mode", "success_file", "failure_file", "split")


def _problems(path: Path, what: str, problems: Sequence[str]) -> SmokeError:
    shown = "; ".join(problems[:5]) + (f"; ... and {len(problems) - 5} more" if len(problems) > 5 else "")
    return SmokeError(f"{path}: invalid {what}: {shown}")


def load_task_manifest(path: Path, task: str) -> dict[str, Any]:
    """Read and validate Track 2's per-task handoff manifest (``handoff/<task>.json``)."""
    path = Path(path)
    try:
        data = _load_json(path)
    except (OSError, ValueError) as exc:
        raise SmokeError(f"cannot read task manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SmokeError(f"{path}: a task manifest must be a JSON object")
    problems: list[str] = []
    if data.get("dataset_kind") != "mujoco":
        problems.append(f"dataset_kind is {data.get('dataset_kind')!r}, expected 'mujoco'")
    if data.get("task") != task:
        problems.append(f"task is {data.get('task')!r}, expected {task!r}")
    if data.get("metadata") != f"{task}/metadata.csv":
        problems.append(f"metadata is {data.get('metadata')!r}, expected '{task}/metadata.csv'")
    pairs = data.get("pairs")
    if not isinstance(pairs, dict) or not pairs:
        problems.append("missing 'pairs' mapping (is this the batch-level manifest.json instead of a task manifest?)")
    else:
        for key, pair in pairs.items():
            if not isinstance(pair, dict):
                problems.append(f"pair {key!r} is not an object")
                continue
            missing = [k for k in _PAIR_KEYS if k not in pair]
            if missing:
                problems.append(f"pair {key!r} lacks {missing}")
                continue
            if pair["pair_id"] != key or pair["task"] != task:
                problems.append(f"pair {key!r} has pair_id={pair['pair_id']!r}, task={pair['task']!r}")
            if pair["split"] not in SPLITS:
                problems.append(f"pair {key!r} has unknown split {pair['split']!r}")
            if pair["failure_mode"] not in DATASET_MODES:
                problems.append(f"pair {key!r} has non-dataset failure_mode {pair['failure_mode']!r}")
            for field in ("injection_timestamp_s", "injection_timestamp"):
                value = pair.get(field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value) or value < 0):
                    problems.append(f"pair {key!r} has an invalid {field}")
    if problems:
        raise _problems(path, "task manifest", problems)
    return data


def has_evaluator_onset_clock(manifest: Mapping[str, Any]) -> bool:
    """True when pairs carry the evaluator's CSV-clock ``injection_timestamp``.

    ``injection_timestamp_s`` is on the absolute simulator clock and is NOT a substitute: it is
    offset from the failure CSV's time axis, so it must never be used as an onset reference.
    """
    return any(isinstance(p, Mapping) and p.get("injection_timestamp") is not None for p in manifest["pairs"].values())


def read_diagnostics(path: Path) -> dict[str, dict[str, Any]]:
    """The released batch's ``diagnostics.jsonl`` keyed by pair_id (one record per accepted pair)."""
    out: dict[str, dict[str, Any]] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SmokeError(f"cannot read {path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise SmokeError(f"{path}:{number}: not valid JSON ({exc})") from exc
        pair_id, config = record.get("pair_id") if isinstance(record, dict) else None, record.get("failure_config") if isinstance(record, dict) else None
        if not isinstance(pair_id, str) or not isinstance(config, dict):
            raise SmokeError(f"{path}:{number}: expected an object with 'pair_id' and 'failure_config'")
        if pair_id in out:
            raise SmokeError(f"{path}:{number}: duplicate diagnostics for pair {pair_id}")
        out[pair_id] = record
    return out


def load_released_pairs(batch_dir: Path, task: str, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join the released batch's metadata and diagnostics to the task manifest, one dict per pair.

    The failed stage and the injection config exist only in the batch's own ``metadata.csv`` and
    ``diagnostics.jsonl`` (the manifests carry neither), so the "eligible stages" are read from here.
    Any disagreement between the three sources is an error, not a silent pick. (A stage that does not
    belong to the task is already rejected by ``MetadataRow`` when the metadata is read.)
    """
    from telemetry_reasoning.io import read_metadata_csv

    batch_dir = Path(batch_dir)
    try:
        metadata = {row.pair_id: row for row in read_metadata_csv(batch_dir / manifest["metadata"])}
    except (OSError, ValueError) as exc:
        raise SmokeError(f"cannot read the released metadata for {task} under {batch_dir}: {exc}") from exc
    diagnostics = read_diagnostics(batch_dir / task / "diagnostics.jsonl")
    pairs = manifest["pairs"]
    problems: list[str] = []
    for label, ids in (("metadata.csv", metadata), ("diagnostics.jsonl", diagnostics)):
        if set(ids) != set(pairs):
            problems.append(f"{label} pair IDs differ from the manifest (missing {len(set(pairs) - set(ids))}, extra {len(set(ids) - set(pairs))})")
    out: list[dict[str, Any]] = []
    for pair_id in sorted(set(pairs) & set(metadata) & set(diagnostics)):
        info, row, config = pairs[pair_id], metadata[pair_id], diagnostics[pair_id]["failure_config"]
        if row.failure_mode.value != info["failure_mode"] or config.get("mode") != info["failure_mode"]:
            problems.append(f"{pair_id}: failure mode differs (manifest {info['failure_mode']}, metadata {row.failure_mode.value}, diagnostics {config.get('mode')})")
        if config.get("subtask") != row.failed_subtask.value:
            problems.append(f"{pair_id}: failed stage differs (metadata {row.failed_subtask.value}, diagnostics {config.get('subtask')})")
        if str(diagnostics[pair_id].get("initial_condition_id")) != str(info["initial_condition_id"]):
            problems.append(f"{pair_id}: initial condition differs between diagnostics and the manifest")
        out.append({"task": task, "pair_id": pair_id, "split": info["split"], "mode": info["failure_mode"],
                    "subtask": row.failed_subtask.value, "condition": int(info["initial_condition_id"]),
                    "seed": row.seed, "config": dict(config), "metadata": row})
    if problems:
        raise _problems(batch_dir / task, "released dataset (disagrees with the task manifest)", problems)
    return out


def eligible_stages(pairs: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    """Every ``(task, mode, subtask)`` the released dataset actually used, sorted."""
    return sorted({(p["task"], p["mode"], p["subtask"]) for p in pairs})


def select_stage_pairs(
    pairs: Sequence[Mapping[str, Any]], per_stage: int = 1
) -> tuple[list[dict[str, Any]], list[tuple[str, str, str]]]:
    """Up to ``per_stage`` released pairs per eligible stage, dev first, then train, never test.

    Returns ``(selected, uncovered_stages)``; a stage is uncovered only if every one of its pairs
    is in the held-out test split. Output order is deterministic: stage, then pair_id.
    """
    if per_stage < 1:
        raise SmokeError("--per-stage must be at least 1")
    selected: list[dict[str, Any]] = []
    uncovered: list[tuple[str, str, str]] = []
    for stage in eligible_stages(pairs):
        candidates = [p for p in pairs if (p["task"], p["mode"], p["subtask"]) == stage]
        ordered = [p for split in SMOKE_SPLITS for p in sorted((q for q in candidates if q["split"] == split), key=lambda q: q["pair_id"])]
        chosen = ordered[:per_stage]
        selected.extend(dict(p) for p in chosen)
        if not chosen:
            uncovered.append(stage)
    return selected, uncovered


def rows_agree(first: Sequence[Any], second: Sequence[Any], atol: float = REPLAY_ATOL) -> tuple[bool, str]:
    """Absolute-tolerance comparison of two telemetry traces (the replay-agreement check)."""
    import numpy as np

    if len(first) != len(second):
        return False, f"{len(first)} vs {len(second)} rows"
    worst = 0.0
    for index, (a, b) in enumerate(zip(first, second)):
        for field in type(a).model_fields:
            x, y = getattr(a, field), getattr(b, field)
            if field == "subtask":
                if x != y:
                    return False, f"subtask differs at row {index}"
                continue
            xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            if xa.shape != ya.shape:
                return False, f"{field} shape differs at row {index}"
            diff = float(np.max(np.abs(xa - ya))) if xa.size else 0.0
            if not diff <= atol:  # also catches NaN
                return False, f"{field} differs by {diff:.3g} at row {index} (atol {atol:g})"
            worst = max(worst, diff)
    return True, f"{len(first)} rows agree (max abs diff {worst:.2g}, atol {atol:g})"


def _trace_valid(rows: Sequence[Any]) -> tuple[bool, str]:
    from telemetry_reasoning.io import validate_raw_run

    try:
        validate_raw_run(list(rows))
    except Exception as exc:  # any validation error means the trace is not schema-valid
        return False, f"{type(exc).__name__}: {exc}"
    return True, f"{len(rows)} rows schema-valid"


def check_episode_pair(nominal: Any, injected: Any, replay: Any) -> list[tuple[str, bool, str]]:
    """Acceptance and determinism checks for one released pair replayed through the simulator.

    ``replay`` is a second run of the same injected episode. Mirrors the generator's acceptance gate
    (nominal success, injection activated, injected run failed, failure trace observed) and adds
    schema validity and replay agreement.
    """
    failure = (injected.diagnostics.get("failure") or {})
    ok_nominal = bool(nominal.task_succeeded) and bool(nominal.diagnostics.get("complete_subtasks"))
    rows_ok, rows_detail = rows_agree(injected.rows, replay.rows)
    checks = [
        ("nominal_succeeded_complete", ok_nominal,
         f"task_succeeded={nominal.task_succeeded}, complete_subtasks={nominal.diagnostics.get('complete_subtasks')}"),
        ("injection_activated", bool(failure.get("activated")), f"first_activation_s={failure.get('first_activation_s')}"),
        ("injected_run_failed", not injected.task_succeeded, f"task_succeeded={injected.task_succeeded}"),
        ("failure_trace_observed", bool(injected.diagnostics.get("observed_subtasks")),
         f"observed_subtasks={injected.diagnostics.get('observed_subtasks')}"),
    ]
    for label, run in (("nominal", nominal), ("injected", injected)):
        ok, detail = _trace_valid(run.rows)
        checks.append((f"{label}_trace_valid", ok, detail))
    checks.append(("replay_rows_agree", rows_ok, rows_detail))
    checks.append(("replay_diagnostics_agree", injected.diagnostics == replay.diagnostics, "identical diagnostics on replay"))
    return checks


def check_excessive_speed_probe(result: Any) -> list[tuple[str, bool, str]]:
    """Activation only: an excessive_speed run is *not* required to fail (it has no accepted examples)."""
    failure = (result.diagnostics.get("failure") or {})
    ok, detail = _trace_valid(result.rows)
    return [
        ("activated", bool(failure.get("activated")),
         f"first_activation_s={failure.get('first_activation_s')}; task_succeeded={result.task_succeeded} (not required to fail)"),
        ("trace_valid", ok, detail),
    ]


def write_policy_config(source: Path, dest: Path, policy: str = RESPONSE_POLICY) -> dict[str, Any]:
    """Copy the LLM config with ``response_policy`` set, keeping relative paths pointing where they did."""
    import yaml

    value = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("llm"), dict):
        raise SmokeError(f"{source}: LLM config must contain an 'llm' mapping")
    llm = value["llm"]
    llm["response_policy"] = policy
    for key in ("prompt_dir", "cache_dir"):  # resolved relative to the config file, which is moving
        if llm.get(key) is not None:
            llm[key] = str((Path(source).parent / llm[key]).resolve())
    Path(dest).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return llm


def check_real_llm_run(llm_dir: Path, pair_ids: Sequence[str], variants: Sequence[str] = LLM_VARIANTS) -> list[tuple[str, bool, str]]:
    """Hard checks that a run directory is a REAL Hugging Face run under ``json_or_fenced_v1``.

    A stub run fails these on purpose: wrong backend label, ``synthetic`` set, no weights loaded,
    or token counts that are only UTF-8 upper bounds.
    """
    from telemetry_reasoning.eval.contracts import index_predictions, read_predictions

    checks: list[tuple[str, bool, str]] = []
    manifest = _load_json(Path(llm_dir) / "run_manifest.json")
    execution = manifest.get("execution") or {}
    expected = len(pair_ids) * len(variants)
    checks += [
        ("manifest.backend_is_hf", manifest.get("backend") == "hf", f"backend={manifest.get('backend')!r}"),
        ("manifest.not_synthetic", manifest.get("synthetic") is False, f"synthetic={manifest.get('synthetic')!r}"),
        ("manifest.weights_loaded", execution.get("hf_weights_loaded") is True, f"hf_weights_loaded={execution.get('hf_weights_loaded')!r}"),
        ("manifest.generation_completed", execution.get("hf_generation_completed") is True,
         f"hf_generation_completed={execution.get('hf_generation_completed')!r}"),
        ("manifest.all_requests_validated", execution.get("all_requests_validated") is True,
         f"all_requests_validated={execution.get('all_requests_validated')!r}"),
        ("manifest.request_counts", execution.get("expected_requests") == execution.get("generated_requests")
         == execution.get("validated_requests") == expected,
         f"expected={execution.get('expected_requests')} generated={execution.get('generated_requests')} "
         f"validated={execution.get('validated_requests')} (want {expected})"),
        ("manifest.response_policy", manifest.get("response_policy") == RESPONSE_POLICY, f"response_policy={manifest.get('response_policy')!r}"),
        ("manifest.no_error_status", not (manifest.get("status_counts") or {}).get("error"), f"status_counts={manifest.get('status_counts')}"),
    ]
    methods = {r.get("token_count_method") for r in manifest.get("requests", [])}
    checks.append(("manifest.real_token_counts", bool(methods) and TOKEN_BOUND_METHOD not in methods, f"token_count_methods={sorted(map(str, methods))}"))
    for variant in variants:
        parsed = read_predictions(Path(llm_dir) / f"{variant}.jsonl")
        checks.append((f"predictions.{variant}.valid", not parsed.invalid, f"{len(parsed.records)} valid, {len(parsed.invalid)} invalid"))
        index = index_predictions(parsed, list(pair_ids))
        checks.append((f"predictions.{variant}.one_per_pair",
                       sorted(k[0] for k in index.by_key) == sorted(pair_ids) and all(k[1] == variant for k in index.by_key),
                       f"{len(index.by_key)} record(s) for {len(pair_ids)} pair(s)"))
        backends = {r.backend for r in parsed.records}
        checks.append((f"predictions.{variant}.backend_is_hf", backends == {"hf"}, f"backends={sorted(backends)}"))
        statuses = sorted({r.status for r in parsed.records})
        checks.append((f"predictions.{variant}.no_error_status", "error" not in statuses, f"statuses={statuses}"))
    return checks


# --------------------------------------------------------------------------- #
# Real-hardware smokes: stages
# --------------------------------------------------------------------------- #
REAL_BANNER = (
    "REAL-HARDWARE SMOKE: the CPU contract smoke ran first (stub, synthetic) in cpu-contract/; the stages below ran "
    "only where requested, and each reports passed / failed / blocked / not_run on its own. A stage that did not run "
    "is never reported as passed."
)


def _environment() -> dict[str, Any]:
    import importlib.metadata as metadata
    import platform

    def version(name: str) -> Optional[str]:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {"python": platform.python_version(), "platform": platform.platform(),
            "mujoco": version("mujoco"), "torch": version("torch"), "transformers": version("transformers")}


def preflight_mujoco() -> list[str]:
    """Reasons the real simulator cannot run here (empty = ready). Imports mujoco; nothing else heavy."""
    reasons: list[str] = []
    try:
        from telemetry_reasoning.sim.runner import MODEL_PATH
    except Exception as exc:  # ImportError, or a native-module block (OSError) as on locked-down Windows
        return [f"the MuJoCo runner cannot be imported ({type(exc).__name__}: {exc})"]
    if not Path(MODEL_PATH).is_file():
        reasons.append(f"scene asset missing: {MODEL_PATH}")
    menagerie = ROOT / "assets" / "mujoco_menagerie"
    if not menagerie.is_dir() or not any(menagerie.iterdir()):
        reasons.append("MuJoCo Menagerie submodule is not initialized (git submodule update --init --depth 1 assets/mujoco_menagerie)")
    return reasons


def preflight_hf(rec: "Recorder", config_path: Path) -> tuple[list[str], dict[str, Any]]:
    """Reasons the real model cannot run here, from ``run_llm.py --check-access`` (loads no weights)."""
    proc = rec.run("hf_check_access", "run_llm.py", ["--check-access", "--config", config_path])
    try:
        info = json.loads(proc.stdout)
    except ValueError as exc:
        raise SmokeError(f"run_llm.py --check-access did not print JSON: {exc}") from exc
    reasons = [f"{package} is not installed" for package in ("torch", "transformers") if not (info.get("packages") or {}).get(package)]
    resources = info.get("resources") or {}
    if resources.get("host_admissible") is not True:
        reasons.append(str(resources.get("reason") or "host memory admission check failed"))
    return reasons, info


def prepare_released_subset(batch_dir: Path, task: str, selected: Sequence[Mapping[str, Any]], dest: Path) -> None:
    """Copy the selected released pairs into a small task directory (released layout) for the model smoke."""
    from telemetry_reasoning.io import write_metadata_csv

    dest.mkdir(parents=True, exist_ok=True)
    for pair in selected:
        for rel in (pair["metadata"].success_file, pair["metadata"].failure_file):
            source = Path(batch_dir) / task / rel
            if not source.is_file():
                raise SmokeError(f"released CSV missing: {source}")
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest / rel)
    write_metadata_csv(dest / "metadata.csv", [pair["metadata"] for pair in selected])


def write_producer_pairs(accepted: Sequence[tuple[Mapping[str, Any], Any, Any]], dest: Path) -> None:
    """Write regenerated pairs in the released layout, under the released pair IDs and metadata."""
    from telemetry_reasoning.io import write_metadata_csv, write_telemetry_csv

    dest.mkdir(parents=True, exist_ok=True)
    for pair, nominal, injected in accepted:
        row = pair["metadata"]
        for rel, run in ((row.success_file, nominal), (row.failure_file, injected)):
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            write_telemetry_csv(dest / rel, run.rows)
    write_metadata_csv(dest / "metadata.csv", [pair["metadata"] for pair, _, _ in accepted])


def run_downstream(rec: "Recorder", tag: str, task: str, data_dir: Path, work: Path) -> dict[str, Any]:
    """parse -> divergence -> Baseline 1 on one task's directory, with output validation."""
    from telemetry_reasoning.eval.contracts import index_predictions, read_predictions
    from telemetry_reasoning.io import read_metadata_csv

    pair_ids = [row.pair_id for row in read_metadata_csv(data_dir / "metadata.csv")]
    bt_raw, bt = work / "bt_raw", work / "bt"
    rec.run(f"{task}_parse", "parse_dataset.py", [
        "--metadata", data_dir / "metadata.csv", "--data-dir", data_dir, "--config", CONFIGS / "parser.yaml", "--output-dir", bt_raw])
    counts = _load_json(bt_raw / "parse_report.json")["counts"]
    rec.check(f"{tag}.parse.all_pairs_written", counts["written_pairs"] == len(pair_ids) and counts["rejected_pairs"] == 0, str(counts))
    rec.run(f"{task}_enrich", "enrich_divergence.py", ["--bt-dir", bt_raw, "--config", CONFIGS / "divergence.yaml", "--output-dir", bt])
    baseline = work / "baseline_1.jsonl"
    rec.run(f"{task}_baseline", "run_baseline.py", ["--bt-dir", bt, "--config", CONFIGS / "rules.yaml", "--output", baseline])
    parsed = read_predictions(baseline)
    index = index_predictions(parsed, pair_ids)
    rec.check(f"{tag}.baseline_1.valid_one_per_pair", not parsed.invalid and sorted(k[0] for k in index.by_key) == sorted(pair_ids),
              f"{len(parsed.records)} valid, {len(parsed.invalid)} invalid, for {len(pair_ids)} pair(s)")
    rec.check(f"{tag}.baseline_1.backend_is_rules", {r.backend for r in parsed.records} == {"rules"}, "")
    return {"task": task, "data": data_dir, "bt": bt, "baseline": baseline, "work": work,
            "baseline_manifest": work / "baseline_1.manifest.json", "pair_ids": pair_ids}


def run_evaluate(rec: "Recorder", tag: str, prepared: Mapping[str, Any], variants: Sequence[str],
                 predictions: Mapping[str, Path], run_manifests: Sequence[Path], manifest_path: Optional[Path],
                 out: Path) -> dict[str, Any]:
    """Score the variants on one task and assert the output contracts. Never expects synthetic results."""
    args: list[Any] = ["--metadata", prepared["data"] / "metadata.csv", "--predictions", *[predictions[v] for v in variants],
                       "--variants", *variants, "--bt-dir", prepared["bt"], "--config", CONFIGS / "eval.yaml", "--output-dir", out]
    if manifest_path is not None:
        args += ["--manifest", manifest_path]
    if run_manifests:
        args += ["--run-manifest", *run_manifests]
    rec.run(f"{prepared['task']}_evaluate_{len(variants)}v", "evaluate.py", args)
    metrics = _load_json(out / "metrics.json")
    n = len(prepared["pair_ids"])
    rec.check(f"{tag}.metrics.all_values_finite", all_finite(metrics), "")
    rec.check(f"{tag}.metrics.variants_scored", list(metrics["variants"]) == list(variants), str(list(metrics["variants"])))
    for variant in variants:
        entry = metrics["variants"][variant]
        sl = entry["slices"]["all"]
        accounted = sum(sl[k] for k in ("n_ok", "n_abstain", "n_error", "n_invalid", "n_missing"))
        rec.check(f"{tag}.metrics.{variant}.one_result_per_pair", sl["n"] == n == accounted, f"n={sl['n']}, accounted={accounted}, pairs={n}")
        rec.check(f"{tag}.metrics.{variant}.nothing_missing_invalid_or_error",
                  sl["n_missing"] == 0 and sl["n_invalid"] == 0 and sl["n_error"] == 0,
                  f"missing={sl['n_missing']}, invalid={sl['n_invalid']}, error={sl['n_error']}")
        rec.check(f"{tag}.metrics.{variant}.not_synthetic", entry["synthetic"] is False, f"synthetic={entry['synthetic']}")
        rec.check(f"{tag}.metrics.{variant}.resources_reported", entry["resources"].get("reported") is True, "")
    b1 = metrics["variants"]["baseline_1"]
    rec.check(f"{tag}.baseline_1.evidence_verified_and_supported",
              b1["evidence"]["checked"] and b1["evidence"]["n_items"] > 0 and b1["evidence"]["n_unsupported"] == 0
              and b1["evidence"]["n_unverifiable"] == 0, str({k: b1["evidence"].get(k) for k in ("checked", "n_items", "n_unsupported", "n_unverifiable")}))
    # Onset must follow the manifest's clock: with only absolute-simulator-time timestamps there is no reference.
    clock = manifest_path is not None and has_evaluator_onset_clock(_load_json(manifest_path))
    references = b1["onset"].get("n_with_reference") or 0
    rec.check(f"{tag}.onset_reference_matches_manifest_clock", references == (n if clock else 0),
              f"{references} reference(s); evaluator-clock injection_timestamp present: {clock}")
    return metrics


def run_plots(rec: "Recorder", tag: str, prepared: Mapping[str, Any], out: Path) -> None:
    rec.run(f"{prepared['task']}_plot_telemetry", "plot_telemetry.py", [
        "--data-dir", prepared["data"], "--bt-dir", prepared["bt"], "--predictions", prepared["baseline"], "--output-dir", out])
    for pair_id in prepared["pair_ids"]:
        png = out / f"telemetry_{pair_id}.png"
        rec.check(f"{tag}.plots.telemetry_{pair_id}", is_nonempty_png(png), f"{png.stat().st_size if png.is_file() else 0} bytes")


def run_producer_stage(rec: "Recorder", out: Path, selections: Mapping[str, Sequence[Mapping[str, Any]]],
                       manifest_paths: Mapping[str, Path], episode_runner: Callable[..., Any]) -> dict[str, dict[str, Any]]:
    """Replay the selected released pairs through the simulator, then run the accepted pairs downstream."""
    from telemetry_reasoning.schemas import FailureMode, Subtask
    from telemetry_reasoning.sim.failures import FailureConfig

    prepared: dict[str, dict[str, Any]] = {}
    for task, selected in selections.items():
        tag = f"real.producer.{task}"
        nominal_cache: dict[tuple[int, int], Any] = {}
        accepted: list[tuple[Mapping[str, Any], Any, Any]] = []
        for pair in selected:
            cfg = pair["config"]
            failure = FailureConfig(mode=FailureMode(cfg["mode"]), subtask=Subtask(cfg["subtask"]), magnitude=float(cfg["magnitude"]),
                                    delay_s=float(cfg.get("delay_s", 0.0)), axis=int(cfg.get("axis", 0)))
            kwargs = {"task": task, "seed": pair["seed"], "initial_condition_id": pair["condition"], "spread": SIM_SPREAD}
            key = (pair["seed"], pair["condition"])
            if key not in nominal_cache:
                nominal_cache[key] = episode_runner(**kwargs)
            nominal = nominal_cache[key]
            injected = episode_runner(**kwargs, failure=failure)
            replay = episode_runner(**kwargs, failure=failure)
            results = check_episode_pair(nominal, injected, replay)
            for name, ok, detail in results:
                rec.check(f"{tag}.{pair['pair_id']}.{name}", ok, f"{pair['mode']}/{pair['subtask']}: {detail}")
            if all(ok for _, ok, _ in results):
                accepted.append((pair, nominal, injected))
        rec.check(f"{tag}.all_selected_pairs_accepted", len(accepted) == len(selected), f"{len(accepted)}/{len(selected)} replayed pairs passed every check")

        subtask, magnitude = EXCESSIVE_SPEED_PROBES[task]
        anchor = selected[0]
        probe = FailureConfig(mode=FailureMode.EXCESSIVE_SPEED, subtask=Subtask(subtask), magnitude=magnitude)
        result = episode_runner(task=task, seed=anchor["seed"], initial_condition_id=anchor["condition"], spread=SIM_SPREAD, failure=probe)
        for name, ok, detail in check_excessive_speed_probe(result):
            rec.check(f"{tag}.excessive_speed.{name}", ok, f"{subtask} x{magnitude:g}: {detail}")

        if accepted:
            data_dir = out / "real" / "producer" / task / "data"
            write_producer_pairs(accepted, data_dir)
            prepared[task] = run_downstream(rec, tag, task, data_dir, out / "real" / "producer" / task)
            run_evaluate(rec, tag, prepared[task], ("baseline_1",), {"baseline_1": prepared[task]["baseline"]},
                         [prepared[task]["baseline_manifest"]], manifest_paths[task], out / "real" / "producer" / task / "eval")
            run_plots(rec, tag, prepared[task], out / "real" / "producer" / task / "plots")
    return prepared


def run_model_stage(rec: "Recorder", out: Path, prepared: Mapping[str, Mapping[str, Any]], manifest_paths: Mapping[str, Path],
                    policy_config: Path, llm_backend: str = "hf") -> None:
    """Baselines 2/3/main on a real model, plus Baseline 1, on the same parsed pairs, per task."""
    for task, prep in prepared.items():
        tag = f"real.model.{task}"
        work = out / "real" / "model" / task
        llm_dir = work / "llm"
        # exit 1 means some request recorded status 'error' (outputs are still published and are checked below)
        rec.run(f"{task}_run_llm_{llm_backend}", "run_llm.py", [
            "--bt-dir", prep["bt"], "--config", policy_config, "--variants", *LLM_VARIANTS,
            "--backend", llm_backend, "--output-dir", llm_dir], ok_codes=(0, 1))
        for name, ok, detail in check_real_llm_run(llm_dir, prep["pair_ids"]):
            rec.check(f"{tag}.{name}", ok, detail)
        predictions = {"baseline_1": prep["baseline"], **{v: llm_dir / f"{v}.jsonl" for v in LLM_VARIANTS}}
        run_evaluate(rec, tag, prep, VARIANTS, predictions, [prep["baseline_manifest"], llm_dir / "run_manifest.json"],
                     manifest_paths[task], work / "eval")
        run_plots(rec, tag, prep, work / "plots")


def run_real_smoke(
    output_dir: Path, *, dataset_dir: Path, task_manifest_dir: Path, tasks: Sequence[str] = DATASET_TASKS,
    sim_backend: str = "mujoco", backend: str = "stub", per_stage: int = 1, llm_config: Path = CONFIGS / "llm.yaml",
    episode_runner: Optional[Callable[..., Any]] = None,
) -> tuple[dict[str, Any], int]:
    """Run the requested real-hardware smokes. Returns ``(report, exit_code)``.

    ``sim_backend='mujoco'`` runs the producer smoke; ``backend='hf'`` runs the model smoke (on the
    producer's regenerated pairs if both are requested, otherwise on the released pairs). The CPU
    contract smoke always runs first, in ``cpu-contract/``, so real and stub results stay apart.
    ``episode_runner`` replaces the MuJoCo runner (tests only).
    """
    want_producer, want_model = sim_backend == "mujoco", backend == "hf"
    require_fresh_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(output_dir)
    status = {"real_producer_smoke": "not_run", "real_model_smoke": "not_run"}
    errors: list[dict[str, str]] = []
    report: dict[str, Any] = {
        "smoke": "real", "sim_backend": sim_backend, "backend": backend, "response_policy": RESPONSE_POLICY,
        "tasks": list(tasks), "per_stage": per_stage, "banner": REAL_BANNER, "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir), "task_manifest_dir": str(task_manifest_dir),
        "split_policy": {"drawn_from": list(SMOKE_SPLITS), "test_split_touched": False}, "environment": _environment(), "real": {},
    }

    def blocked(stage_keys: Sequence[str], stage: str, reason: str) -> None:
        errors.append({"stage": stage, "error": reason})
        for key in stage_keys:
            status[key] = "blocked"

    ran: set[str] = set()
    prefixes = {"real_producer_smoke": ("real.dataset.", "real.producer."), "real_model_smoke": ("real.dataset.", "real.model.")}

    def finish() -> tuple[dict[str, Any], int]:
        requested = [k for k, wanted in (("real_producer_smoke", want_producer), ("real_model_smoke", want_model)) if wanted]
        for key in requested:  # a stage is "passed" only if it ran and none of its checks failed
            if key in ran and status[key] == "not_run":
                status[key] = "failed" if any(c["name"].startswith(prefixes[key]) for c in rec.failed) else "passed"
        report.update(steps=rec.steps, checks=rec.checks, errors=errors, n_checks=len(rec.checks), n_failed=len(rec.failed),
                      real_checks={**status, "note": "Each stage is reported on its own; the CPU contract smoke (stub) never counts toward either."},
                      passed=bool(requested) and all(status[k] == "passed" for k in requested))
        (output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        code = 2 if errors or "blocked" in status.values() else (0 if report["passed"] else 1)
        return report, code

    # 0. the CPU contract smoke must pass first (stub results live in cpu-contract/, never mixed with real ones)
    cpu = run_smoke(output_dir / "cpu-contract")
    report["cpu_contract"] = {"passed": cpu["passed"], "n_checks": cpu["n_checks"], "n_failed": cpu["n_failed"], "synthetic": True,
                              "report": "cpu-contract/smoke_report.json"}
    rec.check("real.cpu_contract_smoke_passed", cpu["passed"], f"{cpu['n_checks'] - cpu['n_failed']}/{cpu['n_checks']} checks")
    if not cpu["passed"]:
        return finish()

    # 1. the released dataset: manifests, stages, selection, coverage (needs no hardware)
    selections: dict[str, list[dict[str, Any]]] = {}
    manifest_paths: dict[str, Path] = {}
    keys = [k for k, wanted in (("real_producer_smoke", want_producer), ("real_model_smoke", want_model)) if wanted]
    try:
        for task in tasks:
            manifest_paths[task] = Path(task_manifest_dir) / f"{task}.json"
            manifest = load_task_manifest(manifest_paths[task], task)
            pairs = load_released_pairs(dataset_dir, task, manifest)
            selected, uncovered = select_stage_pairs(pairs, per_stage)
            selections[task] = selected
            stages = eligible_stages(pairs)
            tag = f"real.dataset.{task}"
            rec.check(f"{tag}.all_four_modes_covered", {p["mode"] for p in selected} == set(DATASET_MODES),
                      f"covered {sorted({p['mode'] for p in selected})}")
            rec.check(f"{tag}.every_eligible_stage_covered", not uncovered, f"uncovered stages: {uncovered}" if uncovered else f"{len(stages)} stage(s) covered")
            rec.check(f"{tag}.test_split_untouched", all(p["split"] in SMOKE_SPLITS for p in selected), "selected splits: " + ", ".join(sorted({p["split"] for p in selected})))
            report["real"][task] = {
                "released_pairs": len(pairs), "eligible_stages": [{"mode": m, "subtask": s, "released_pairs": sum(1 for p in pairs if (p["mode"], p["subtask"]) == (m, s)),
                                                                   "selected": [p["pair_id"] for p in selected if (p["mode"], p["subtask"]) == (m, s)]} for _, m, s in stages],
                "selected_pairs": [p["pair_id"] for p in selected], "evaluator_onset_clock_in_manifest": has_evaluator_onset_clock(manifest),
                "onset_note": None if has_evaluator_onset_clock(manifest) else "manifest has only absolute-simulator-time injection_timestamp_s; onset is not scored",
            }
    except SmokeError as exc:
        blocked(keys, "dataset", str(exc))
        return finish()

    # 2. hardware preflight for everything requested, before spending any time on either
    policy_config = output_dir / "real" / "llm_policy.yaml"
    if want_model:
        policy_config.parent.mkdir(parents=True, exist_ok=True)
        write_policy_config(llm_config, policy_config)
    if want_producer and episode_runner is None:
        reasons = preflight_mujoco()
        if reasons:
            blocked(keys, "preflight_mujoco", "; ".join(reasons))
    if want_model and not errors:
        reasons, info = preflight_hf(rec, policy_config)
        report["environment"]["hf_check_access"] = {k: info.get(k) for k in ("model_id", "revision", "configured_device", "configured_dtype", "packages", "resources")}
        if reasons:
            blocked(keys, "preflight_hf", "; ".join(reasons))
    if errors:
        return finish()

    # 3. the stages
    prepared: dict[str, dict[str, Any]] = {}
    try:
        if want_producer:
            ran.add("real_producer_smoke")
            runner = episode_runner
            if runner is None:
                from telemetry_reasoning.sim.runner import run_episode as runner  # type: ignore[assignment]
            prepared = run_producer_stage(rec, output_dir, selections, manifest_paths, runner)
        if want_model:
            ran.add("real_model_smoke")
            if not want_producer:
                for task in tasks:
                    data_dir = output_dir / "real" / "model" / task / "data"
                    prepare_released_subset(dataset_dir, task, selections[task], data_dir)
                    prepared[task] = run_downstream(rec, f"real.model.{task}", task, data_dir, output_dir / "real" / "model" / task)
            missing = sorted(set(tasks) - set(prepared))
            if missing:
                raise SmokeError(f"no accepted pairs to run the model on for {missing}")
            run_model_stage(rec, output_dir, prepared, manifest_paths, policy_config)
    except SmokeError as exc:
        errors.append({"stage": "real", "error": str(exc)})
        for key in ran:
            status[key] = "failed"
    return finish()


def _print_real_summary(report: Mapping[str, Any], output_dir: Path) -> None:
    print(REAL_BANNER)
    for step in report["steps"]:
        print(f"  [{step['returncode']}] {step['name']:<28} {step['duration_s']:>7.2f}s  {step['log']}")
    for error in report["errors"]:
        print(f"  ERROR ({error['stage']}): {error['error']}")
    for check in report["checks"]:
        if not check["passed"]:
            print(f"  FAILED {check['name']}: {check['detail']}")
    cpu = report.get("cpu_contract", {})
    print(f"\nCPU contract smoke (synthetic): {cpu.get('n_checks', 0) - cpu.get('n_failed', 0)}/{cpu.get('n_checks', 0)} checks")
    for key in ("real_producer_smoke", "real_model_smoke"):
        print(f"{key}: {report['real_checks'][key]}")
    print(f"report: {output_dir / 'smoke_report.json'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh (absent or empty) directory")
    parser.add_argument("--backend", choices=("stub", "hf"), default="stub", help="LLM backend; 'hf' runs the real model smoke")
    parser.add_argument("--sim-backend", choices=("mock", "mujoco"), default="mock", help="data source; 'mujoco' runs the real producer smoke")
    parser.add_argument("--seed", type=int, default=723, help="mock-fixture seed (CPU contract smoke)")
    parser.add_argument("--dataset-dir", type=Path, help="released batch directory (required for the real smokes)")
    parser.add_argument("--task-manifest-dir", type=Path, help="directory holding <task>.json (default: <dataset-dir>/handoff)")
    parser.add_argument("--tasks", choices=("all", *DATASET_TASKS), default="all")
    parser.add_argument("--per-stage", type=int, default=1, help="released pairs replayed per eligible stage")
    parser.add_argument("--llm-config", type=Path, default=CONFIGS / "llm.yaml")
    parser.add_argument("--cover-dataset-failure-modes", action="store_true",
                        help="accepted for the planned command; covering all four dataset modes is always required")
    args = parser.parse_args(argv)

    if (args.backend, args.sim_backend) != ("stub", "mock"):
        if args.dataset_dir is None:
            print("error: the real smokes (--sim-backend mujoco / --backend hf) need --dataset-dir, the released batch "
                  "directory (the stage each failure used is read from its metadata.csv and diagnostics.jsonl).", file=sys.stderr)
            return 2
        tasks = DATASET_TASKS if args.tasks == "all" else (args.tasks,)
        try:
            report, code = run_real_smoke(
                args.output_dir, dataset_dir=args.dataset_dir, task_manifest_dir=args.task_manifest_dir or args.dataset_dir / "handoff",
                tasks=tasks, sim_backend=args.sim_backend, backend=args.backend, per_stage=args.per_stage, llm_config=args.llm_config)
        except (SmokeError, OSError, ValueError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _print_real_summary(report, args.output_dir)
        return code

    try:
        report = run_smoke(args.output_dir, seed=args.seed)
    except (SmokeError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(BANNER)
    for step in report["steps"]:
        print(f"  [{step['returncode']}] {step['name']:<20} {step['duration_s']:>6.2f}s  {step['log']}")
    for check in report["checks"]:
        if not check["passed"]:
            print(f"  FAILED {check['name']}: {check['detail']}")
    verdict = "PASSED" if report["passed"] else "FAILED"
    print(f"\ncontract checks: {report['n_checks'] - report['n_failed']}/{report['n_checks']} passed -> {verdict} (synthetic)")
    print("real MuJoCo producer smoke: not run | real Hugging Face model smoke: not run")
    print(f"report: {args.output_dir / 'smoke_report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
