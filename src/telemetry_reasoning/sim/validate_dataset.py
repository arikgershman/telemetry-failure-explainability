"""Strict, read-only Track 2 acceptance gate; no simulator imports required."""

from itertools import groupby
from collections import Counter
import json
from pathlib import Path

import numpy as np

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.parser.compressor import build_paired_bt
from telemetry_reasoning.schemas import TASK_SUBTASKS


def validate_dataset(batch: Path) -> dict:
    """Validate every pair and quarantine failures logically, preserving sources.

    State comparisons use absolute tolerance 1e-8 and no relative tolerance.
    Activation timestamps in diagnostics are absolute simulator time; CSV time
    starts at the canonical task's first sample. No conversion is guessed.
    """
    batch = Path(batch).resolve()
    records = []
    fatal = []
    for task in ("pick_up_object", "transport_object"):
        folder = batch / task
        try:
            metadata = read_metadata_csv(folder / "metadata.csv")
            diagnostics = {}
            for line in (folder / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                diagnostic = json.loads(line)
                key = str(diagnostic["pair_id"])
                if key in diagnostics:
                    raise ValueError(f"duplicate diagnostic ID {key}")
                diagnostics[key] = diagnostic
            if set(diagnostics) != {m.pair_id for m in metadata}:
                raise ValueError("metadata/diagnostics IDs differ")
        except (OSError, ValueError, KeyError) as exc:
            fatal.append(f"{task}: {exc}")
            continue
        modes = Counter(m.failure_mode.value for m in metadata)
        expected_modes = {'position_error', 'orientation_error', 'inadequate_grip_strength', 'early_gripper_release'}
        if modes != Counter({mode: 125 for mode in expected_modes}):
            fatal.append(f'{task}: expected 500 pairs, 125 per calibrated mode')
        conditions = Counter(str(d.get('initial_condition_id')) for d in diagnostics.values())
        if conditions != Counter({str(i): 10 for i in range(50)}):
            fatal.append(f'{task}: expected 50 initial conditions, 10 pairs each')
        references = [getattr(m, f'{kind}_file') for m in metadata for kind in ('success', 'failure')]
        resolved_references = [(folder / reference).resolve() for reference in references]
        if len(set(resolved_references)) != len(resolved_references):
            fatal.append(f'{task}: reused telemetry file references')
        for meta in metadata:
            issues = []
            diagnostic = diagnostics[meta.pair_id]
            traces = {}
            for kind in ("success", "failure"):
                path = (folder / getattr(meta, f"{kind}_file")).resolve()
                try:
                    if not path.is_relative_to(folder.resolve()):
                        raise ValueError("path escapes task directory")
                    rows = read_telemetry_csv(path)
                    traces[kind] = rows
                    stages = tuple(label for label, _ in groupby(r.subtask for r in rows))
                    if stages != TASK_SUBTASKS[meta.task]:
                        issues.append(f"{kind}: incomplete or unordered subtasks")
                    for field in ("ee_quat", "obj_quat"):
                        values = np.asarray([getattr(r, field) for r in rows])
                        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6, rtol=0):
                            issues.append(f"{kind}: {field} not normalized")
                        if np.any(np.sum(values[:-1] * values[1:], axis=1) < 0):
                            issues.append(f"{kind}: {field} sign discontinuity")
                except (OSError, ValueError) as exc:
                    issues.append(f"{kind}: {exc}")
            nominal = diagnostic.get("nominal", {})
            injected = diagnostic.get("injected", {})
            injection = injected.get("failure") or {}
            for field in ("seed", "initial_condition_id", "initial_offset_xy", "task"):
                if field not in nominal or nominal.get(field) != injected.get(field):
                    issues.append(f"paired initial identity mismatch: {field}")
            if meta.task.value != task or nominal.get("task") != task:
                issues.append("task identity mismatch")
            if nominal.get("seed") != meta.seed:
                issues.append("metadata seed mismatch")
            if nominal.get("initial_condition_id") != diagnostic.get("initial_condition_id"):
                issues.append("condition identity mismatch")
            config = diagnostic.get("failure_config", {})
            if config.get("mode") != meta.failure_mode.value or config.get("subtask") != meta.failed_subtask.value:
                issues.append("injection configuration mismatch")
            if nominal.get("controller_outcome") != "success" or injected.get("controller_outcome") != "failed":
                issues.append("controller outcomes not success/failure")
            onset = injection.get("first_activation_s")
            if (injection.get("activated") is not True or injection.get("activation_steps", 0) <= 0
                    or not isinstance(onset, (int, float)) or not np.isfinite(onset)):
                issues.append("missing valid injection activation")
            if len(traces) == 2:
                a, b = traces["success"][0], traces["failure"][0]
                if len(a.joint_positions) != len(b.joint_positions):
                    issues.append("paired joint dimensions differ")
                # An immediate injection may affect the first observation; use
                # simulator initial-state identity above for those cases.
                if config.get("delay_s", 0) > 0:
                    for field in type(a).model_fields:
                        if field == "subtask":
                            continue
                        if not np.allclose(getattr(a, field), getattr(b, field), atol=1e-8, rtol=0):
                            issues.append(f"first pre-injection state mismatch: {field}")
                if not issues:
                    try:
                        build_paired_bt(meta, folder, CompressionConfig())
                    except (OSError, ValueError) as exc:
                        issues.append(f"BT incompatibility: {exc}")
            records.append({"task": task, "pair_id": meta.pair_id,
                            "initial_condition_id": diagnostic.get("initial_condition_id"),
                            "injection_timestamp_s": onset,
                            "injection_time_basis": "absolute_simulator_time",
                            "issues": issues, "accepted": not issues})
    return {"validation_version": 1, "batch": str(batch),
            "accepted_pairs": sum(r["accepted"] for r in records),
            "quarantined_pairs": sum(not r["accepted"] for r in records),
            "fatal_errors": fatal, "pairs": records,
            "limitations": ["Stored controller outcomes are not independent physical outcome verification.",
                            "Replay and pre-injection trajectory verification require simulator runs and a task time origin."]}
