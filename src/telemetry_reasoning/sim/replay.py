"""Source-bound replay of both archived executions, without clock assumptions.

Every observation, identity, duration, and repeated execution must match before
an independently recovered clock can certify an archived run. Original sources
are never rewritten. Reproducing nominal alone cannot certify injected clocks.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np

from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.schemas import FailureMode, Subtask
from telemetry_reasoning.sim.dataset_certification import inventory_sha256, inventory_sources
from telemetry_reasoning.sim.failures import FailureConfig

TASKS = ("pick_up_object", "transport_object")
ATOL = 1e-8


def compare_telemetry(original, replayed):
    """Compare full ordered observations with absolute 1e-8 tolerance only."""
    result = {"matched": len(original) == len(replayed) and bool(original),
              "all_rows": True, "all_fields": True, "row_count": len(original),
              "replayed_row_count": len(replayed), "atol": ATOL, "rtol": 0,
              "max_abs_deltas": {}, "mismatched_fields": []}
    if not result["matched"]:
        result["all_rows"] = result["all_fields"] = False
        result["mismatched_fields"].append("row_count")
        return result
    for field in type(original[0]).model_fields:
        a, b = [getattr(row, field) for row in original], [getattr(row, field) for row in replayed]
        if field == "subtask":
            matched = a == b
        else:
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            matched = a.shape == b.shape and np.allclose(a, b, atol=ATOL, rtol=0)
            if a.shape == b.shape:
                result["max_abs_deltas"][field] = float(np.max(np.abs(a - b)))
        if not matched:
            result["mismatched_fields"].append(field)
            result["matched"] = False
    return result


def compare_injection_diagnostics(expected, actual):
    """Compare explicitly versioned counts without rewriting archived records.

    Historical unversioned ``activation_steps`` counted active context calls.
    The current engine records that legacy quantity separately from unique
    active simulator timestamps. Unknown bases cannot establish a replay match.
    """
    if expected is None and actual is None:
        return {"matched": True, "activation_count_comparison_basis": "not_applicable"}
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return {"matched": False, "activation_count_comparison_basis": "incompatible_diagnostics"}
    unique = "unique_active_simulator_timestamps"
    expected_basis, actual_basis = expected.get("activation_steps_basis"), actual.get("activation_steps_basis")
    if expected_basis is None and actual_basis in (None, unique):
        actual_count = actual.get("activation_context_calls" if actual_basis == unique else "activation_steps")
        basis = "archived_unversioned_context_calls"
    elif expected_basis == actual_basis == unique:
        actual_count = actual.get("activation_steps")
        basis = unique
    else:
        return {"matched": False, "activation_count_comparison_basis": "unsupported_or_incompatible_basis"}
    expected_time, actual_time = expected.get("first_activation_s"), actual.get("first_activation_s")
    matched = (expected.get("activated") is True and actual.get("activated") is True
               and type(expected.get("activation_steps")) is int and expected["activation_steps"] > 0
               and type(actual_count) is int and actual_count == expected["activation_steps"]
               and type(expected_time) in (int, float) and math.isfinite(expected_time)
               and type(actual_time) in (int, float) and math.isfinite(actual_time)
               and math.isclose(expected_time, actual_time, abs_tol=ATOL, rel_tol=0))
    return {"matched": bool(matched), "activation_count_comparison_basis": basis}


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def recover_dataset_evidence(batch, output_dir, *, representative=False, repeats=2, episode_runner=None):
    """Retain full or explicitly representative replay evidence in a fresh run.

    Identical source bytes and episode inputs reuse comparison results, retaining
    per-file hashes. Repeats are never replaced by nominal-derived assumptions.
    The optional runner supports offline tests; the CLI always uses real physics.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("At least two repeat executions are required")
    batch, output_dir = Path(batch).resolve(), Path(output_dir)
    if output_dir.is_symlink() or (output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    )):
        raise ValueError("Replay output directory must be absent or empty")
    if output_dir.resolve().is_relative_to(batch):
        raise ValueError("Replay evidence must live outside the immutable dataset batch")
    files, selected, total_pairs = inventory_sources(batch), [], 0
    for task in TASKS:
        metadata = read_metadata_csv(batch / task / "metadata.csv")
        records = [json.loads(line) for line in (batch / task / "diagnostics.jsonl").read_text().splitlines() if line.strip()]
        diagnostics = {str(record["pair_id"]): record for record in records}
        if len(diagnostics) != len(records) or set(diagnostics) != {row.pair_id for row in metadata}:
            raise ValueError(f"{task}: diagnostic and metadata identities differ")
        seen_modes = set()
        for meta in metadata:
            total_pairs += 1
            if representative and meta.failure_mode in seen_modes:
                continue
            seen_modes.add(meta.failure_mode)
            selected.append((meta, diagnostics[meta.pair_id]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    (output_dir / "logs").mkdir()
    receipt = output_dir / "evidence.json"
    document = {"schema_version": 1, "status": "running", "source_files": files,
                "source_inventory_sha256": inventory_sha256(files), "pairs": [],
                "selection": {"scope": "one_pair_per_task_and_released_mode" if representative else "all_pairs",
                              "available_pairs": total_pairs, "selected_pairs": len(selected)},
                "producer_provenance": {"status": "pending", "recorded_at_generation": False,
                                        "note": "A release tag or replay cannot prove original generator/model/calibration inputs."},
                "limitations": ["Current-runtime replay is not original-environment attestation.",
                                "Unmatched executions supply candidate clocks, never certified terminal states.",
                                "Published nominal-derived clock files are never promoted by filename."]}
    cache, executions, started = {}, 0, time.perf_counter()
    _write(receipt, document)
    try:
        if episode_runner is None:
            from telemetry_reasoning.sim.runner import run_episode
            episode_runner = run_episode
        for number, (meta, diagnostic) in enumerate(selected, 1):
            pair = {"task": meta.task.value, "pair_id": meta.pair_id}
            for kind, file_field in (("nominal", "success_file"), ("injected", "failure_file")):
                relative = f"{meta.task.value}/{getattr(meta, file_field)}"
                fingerprint = files.get(relative)
                if fingerprint is None:
                    raise ValueError(f"Trace is outside inventoried sources: {relative}")
                archived = diagnostic[kind]
                if (archived.get("task") != meta.task.value or archived.get("seed") != meta.seed
                        or archived.get("initial_condition_id") != diagnostic["initial_condition_id"]):
                    raise ValueError(f"Archived identity mismatch: {relative}")
                raw_config = diagnostic["failure_config"] if kind == "injected" else None
                if raw_config is not None and (raw_config.get("mode") != meta.failure_mode.value
                                                or raw_config.get("subtask") != meta.failed_subtask.value):
                    raise ValueError(f"Injection configuration disagrees with metadata: {relative}")
                config = None if raw_config is None else FailureConfig(
                    mode=FailureMode(raw_config["mode"]), subtask=Subtask(raw_config["subtask"]),
                    magnitude=raw_config["magnitude"], delay_s=raw_config.get("delay_s", 0), axis=raw_config.get("axis", 0))
                spread = archived.get("spread_m", 0.015)
                if isinstance(spread, bool) or not isinstance(spread, (int, float)) or not math.isfinite(spread) or spread < 0:
                    raise ValueError(f"Invalid recorded initial-condition spread: {relative}")
                # A cached match is specific to these diagnostics, not only CSV
                # bytes: a later inconsistent outcome/duration must be rechecked.
                key = (meta.task.value, meta.seed, diagnostic["initial_condition_id"], spread,
                       json.dumps(raw_config, sort_keys=True), fingerprint["sha256"], json.dumps(archived, sort_keys=True))
                if key not in cache:
                    rows = read_telemetry_csv(batch / relative)
                    comparisons, prior, repeatable, result = [], None, True, None
                    for repeat in range(repeats):
                        log = output_dir / "logs" / f"episode_{len(cache):04d}_{repeat}.log"
                        with log.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
                            result = episode_runner(task=meta.task, seed=meta.seed,
                                                    initial_condition_id=diagnostic["initial_condition_id"], failure=config, spread=spread)
                        executions += 1
                        comparison = compare_telemetry(rows, result.rows)
                        identity = all(result.diagnostics.get(field) == archived.get(field)
                                       for field in ("task", "seed", "initial_condition_id", "initial_offset_xy", "controller_outcome"))
                        duration = result.diagnostics.get("duration_s")
                        duration_ok = (isinstance(duration, (int, float)) and math.isfinite(duration)
                                       and math.isclose(duration, archived.get("duration_s", -1), abs_tol=ATOL, rel_tol=0))
                        expected, actual = archived.get("failure"), result.diagnostics.get("failure")
                        injection_comparison = compare_injection_diagnostics(expected, actual)
                        injection_ok = (injection_comparison["matched"]
                                        and (expected is None and actual is None if kind == "nominal"
                                             else isinstance(expected, dict) and isinstance(actual, dict)))
                        comparison.update(identity_matched=identity, duration_matched=duration_ok,
                                          injection_diagnostics_matched=bool(injection_ok),
                                          activation_count_comparison_basis=injection_comparison["activation_count_comparison_basis"],
                                          log=log.relative_to(output_dir).as_posix())
                        comparison["matched"] = bool(comparison["matched"] and identity and duration_ok and injection_ok)
                        comparisons.append(comparison)
                        if prior is not None:
                            repeatable &= compare_telemetry(prior.rows, result.rows)["matched"]
                            repeatable &= prior.diagnostics == result.diagnostics
                        prior = result
                    matched = all(row["matched"] for row in comparisons) and repeatable
                    origin = result.diagnostics.get("telemetry_start_sim_s")
                    valid_origin = isinstance(origin, (int, float)) and not isinstance(origin, bool) and math.isfinite(origin) and origin >= 0
                    cache[key] = {
                        "origin": {"timestamp_s": origin if valid_origin else None,
                                   "basis": "verified_exact_replay" if matched and valid_origin else "unverified_replay_candidate",
                                   "verified": bool(matched and valid_origin)},
                        "replay": {**comparisons[0], "matched": bool(matched), "repeats": repeats,
                                   "fresh_repeatability_matched": bool(repeatable), "repeat_comparisons": comparisons},
                        "terminal_state": result.diagnostics.get("terminal_state"),
                        "terminal_state_verification": "archived_replay_matched" if matched else "current_replay_candidate_only",
                        "physical_outcome": result.diagnostics.get("physical_outcome"),
                        "runtime_provenance": result.diagnostics.get("producer")}
                pair[kind] = {**deepcopy(cache[key]), "file": relative, "source_sha256": fingerprint["sha256"]}
            document["pairs"].append(pair)
            if number % 10 == 0 or number == len(selected):
                _write(receipt, document)
                print(f"Replayed {number}/{len(selected)} selected pairs ({executions} fresh executions).", flush=True)
        if inventory_sources(batch) != files:
            raise ValueError("Archived sources changed during replay")
        matched = sum(pair[kind]["replay"]["matched"] for pair in document["pairs"] for kind in ("nominal", "injected"))
        document["status"] = "matched" if matched == 2 * len(selected) else "mismatched"
        document["summary"] = {"pair_records": len(selected), "matched_run_records": matched,
                               "mismatched_run_records": 2 * len(selected) - matched,
                               "unique_episode_inputs": len(cache), "fresh_executions": executions,
                               "full_dataset_coverage": len(selected) == total_pairs, "source_files_unchanged": True}
    except BaseException as exc:
        document["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        document["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        document["runtime_s"] = time.perf_counter() - started
        _write(receipt, document)
    return document
