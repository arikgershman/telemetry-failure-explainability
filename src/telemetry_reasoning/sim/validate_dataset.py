"""Strict, read-only Track 2 acceptance gate; no simulator imports required."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.parser.compressor import build_paired_bt
from telemetry_reasoning.sim.dataset_certification import (
    compare_pre_injection, load_evidence, replay_has_explicit_failure, replay_verified, safe_source_path,
    producer_certification, state, summarize_certification, verified_origin,
)


def _physical_certification(task, diagnostic, run, replay_ok, rows):
    """Recompute an objective from terminal measurements, never status strings."""
    from telemetry_reasoning.sim.outcomes import assess_physical_outcome
    terminal = diagnostic.get("terminal_state")
    source = "recorded_terminal_state"
    if terminal is None and replay_ok:
        terminal = run.get("terminal_state")
        source = "verified_exact_replay_terminal_state"
    if not isinstance(terminal, dict) or terminal.get("terminal_observed") is not True:
        return {"status": "unknown", "source": source,
                "reason": "No recorded or exact-replay terminal measurements."}
    try:
        initial = np.asarray(terminal['initial_position'], dtype=float)
        if (not rows or initial.shape != (3,)
                or not np.allclose(initial, rows[0].obj_pos, atol=1e-8, rtol=0)):
            return {"status": "unknown", "source": source,
                    "reason": "Terminal objective origin does not match the first canonical object observation."}
        return assess_physical_outcome(
            task, terminal["initial_position"], terminal["final_position"],
            terminal.get("final_velocity"), terminal_observed=True, source=source,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "unknown", "source": source,
                "reason": f"Invalid terminal measurements: {exc}"}


def _pair_certification(task, traces, diagnostic, pair_evidence, evidence, paths, onset, issues):
    cert = {"stored_trace_integrity_and_parser_compatibility":
            state("failed" if issues else "passed", "Strict source/schema/parser checks.")}
    origins, replays, physical, replay_failures = {}, {}, {}, {}
    for kind, key in (("success", "nominal"), ("failure", "injected")):
        run = pair_evidence.get(key) or {}
        rows = traces.get(kind, [])
        replay_ok = replay_verified(run, paths[kind], rows, evidence.get("source_files", {}))
        replays[key] = replay_ok
        replay_failures[key] = replay_has_explicit_failure(run)
        origin_diagnostic = dict(diagnostic.get(key, {}))
        if diagnostic.get('clock_alignment_source') is not None:
            origin_diagnostic['clock_alignment_source'] = diagnostic['clock_alignment_source']
        origins[key] = verified_origin(origin_diagnostic, run, replay_ok)
        physical[key] = _physical_certification(task, diagnostic.get(key, {}), run, replay_ok, rows)
    cert["deterministic_replay"] = state(
        "passed" if all(replays.values()) else "failed" if any(replay_failures.values()) else "pending",
        "Both source-bound episodes require two fresh repeats matching all rows, fields, identities, duration, and injection diagnostics.",
        runs=replays, explicit_mismatches=replay_failures)
    expected = {"nominal": "succeeded", "injected": "failed"}
    result = ("passed" if all(physical[k]["status"] == v for k, v in expected.items()) else
              "failed" if any(physical[k]["status"] not in (v, "unknown") for k, v in expected.items())
              else "pending")
    cert["independent_physical_outcomes"] = state(
        result, "Independent geometric/kinematic task objectives, using terminal state measurements.",
        runs=physical)
    producers = {key: producer_certification({'producer_provenance': diagnostic.get(key, {}).get('producer', {})})
                 for key in ('nominal', 'injected')}
    cert['producer_provenance'] = state(
        'passed' if all(value['status'] == 'passed' for value in producers.values()) else 'pending',
        'Exact producer source/model state must be recorded for both runs.', runs=producers)
    cert["complete_pre_injection_agreement"] = (
        compare_pre_injection(traces["success"], traces["failure"],
                              origins["nominal"], origins["injected"], onset)
        if len(traces) == 2 else state("pending", "Both valid source traces are required."))
    timing = {"injection_timestamp_s": onset, "injection_time_basis": "absolute_simulator_time",
              "clock_origins": origins}
    origin = origins["injected"]
    if origin.get("verified") and isinstance(onset, (float, int)) and np.isfinite(onset):
        relative = onset - origin["timestamp_s"]
        rows = traces.get("failure", [])
        if relative >= -1e-8 and rows and relative <= rows[-1].timestamp + 0.022:
            timing.update(injection_timestamp=max(0.0, relative),
                          injection_timestamp_basis="failure_csv_elapsed_seconds",
                          injection_conversion_verified=True)
    return cert, timing


def validate_dataset(batch: Path, *, evidence: Path | dict | None = None) -> dict:
    """Validate every pair and quarantine failures logically, preserving sources.

    State comparisons use absolute tolerance 1e-8 and no relative tolerance.
    Activation timestamps in diagnostics are absolute simulator time; CSV time
    starts at the canonical task's first sample. No conversion is guessed.
    """
    batch = Path(batch).resolve()
    records = []
    fatal = []
    seen_inodes = {}
    document, evidence_pairs, evidence_errors = load_evidence(batch, evidence)
    fatal.extend(f"certification evidence: {error}" for error in evidence_errors)
    for task in ("pick_up_object", "transport_object"):
        folder = batch / task
        try:
            if folder.is_symlink():
                raise ValueError("task directory must not be a symlink")
            metadata = read_metadata_csv(safe_source_path(batch, f"{task}/metadata.csv"))
            diagnostics = {}
            for line in safe_source_path(batch, f"{task}/diagnostics.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                diagnostic = json.loads(line)
                if not isinstance(diagnostic, dict):
                    raise ValueError("diagnostic must be an object")
                if any(not isinstance(diagnostic.get(key), dict)
                       for key in ('nominal', 'injected', 'failure_config')):
                    raise ValueError("diagnostic nominal/injected/failure_config must be objects")
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
        existing = [p for p in resolved_references if p.is_file()]
        if len({(p.stat().st_dev, p.stat().st_ino) for p in existing}) != len(existing):
            fatal.append(f'{task}: reused telemetry file inodes')
        for path in existing:
            inode = (path.stat().st_dev, path.stat().st_ino)
            if inode in seen_inodes and seen_inodes[inode] != task:
                fatal.append(f'{task}: telemetry file inode reused across tasks')
                break
            seen_inodes[inode] = task
        for meta in metadata:
            issues = []
            diagnostic = diagnostics[meta.pair_id]
            traces = {}
            paths = {kind: f"{task}/{getattr(meta, f'{kind}_file')}" for kind in ("success", "failure")}
            for kind in ("success", "failure"):
                try:
                    path = safe_source_path(batch, paths[kind])
                    rows = read_telemetry_csv(path)
                    traces[kind] = rows
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
            if not isinstance(injection, dict):
                issues.append("injection diagnostics must be an object")
                injection = {}
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
            steps = injection.get("activation_steps", 0)
            if (injection.get("activated") is not True or not isinstance(steps, int)
                    or isinstance(steps, bool) or steps <= 0
                    or isinstance(onset, bool) or not isinstance(onset, (int, float)) or not np.isfinite(onset)):
                issues.append("missing valid injection activation")
            if len(traces) == 2:
                a, b = traces["success"][0], traces["failure"][0]
                if len(a.joint_positions) != len(b.joint_positions):
                    issues.append("paired joint dimensions differ")
                # An immediate injection may affect the first observation; use
                # simulator initial-state identity above for those cases.
                delay = config.get("delay_s", 0)
                valid_delay = (isinstance(delay, (int, float)) and not isinstance(delay, bool)
                               and np.isfinite(delay) and delay >= 0)
                if not valid_delay:
                    issues.append("invalid injection delay")
                if valid_delay and delay > 0:
                    for field in type(a).model_fields:
                        if field == "subtask":
                            continue
                        if not np.allclose(getattr(a, field), getattr(b, field), atol=1e-8, rtol=0):
                            issues.append(f"first pre-injection state mismatch: {field}")
                if not issues:
                    try:
                        # The parser owns stage coverage rules for both runs.
                        build_paired_bt(meta, folder, CompressionConfig())
                    except (OSError, ValueError) as exc:
                        issues.append(f"BT incompatibility: {exc}")
            certification, timing = _pair_certification(
                task, traces, diagnostic, evidence_pairs.get((task, meta.pair_id), {}),
                document, paths, onset, issues)
            records.append({"task": task, "pair_id": meta.pair_id,
                            "initial_condition_id": diagnostic.get("initial_condition_id"),
                            **timing, "certification": certification,
                            "issues": issues, "accepted": not issues})
    unknown_evidence = set(evidence_pairs) - {(r["task"], r["pair_id"]) for r in records}
    if unknown_evidence:
        fatal.append(f"certification evidence references unknown pairs: {sorted(unknown_evidence)}")
    certification = summarize_certification(records, fatal, document)
    return {"validation_version": 2, "batch": str(batch),
            "accepted_pairs": sum(r["accepted"] for r in records),
            "quarantined_pairs": sum(not r["accepted"] for r in records),
            "fatal_errors": fatal, "pairs": records,
            "certification": certification,
            "limitations": ["Stored controller outcomes alone are not independent physical outcome verification.",
                            "Pre-injection agreement covers stored 50 Hz observations, not every physics step.",
                            "A nominal replay does not establish an injected run's logging origin.",
                            "Current validator hashes do not establish historical producer provenance."]}
