"""Validate Track 1's recovered clocks against the archived runs."""

import json
import math

import numpy as np

from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Expected finite nonnegative clock seconds, got {value!r}")
    return value


def _index(path):
    records = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            record = json.loads(line)
            pair_id = record["pair_id"]
            if pair_id in records:
                raise ValueError(f"{path}: duplicate pair ID {pair_id}")
            records[pair_id] = record
    return records


def _compare_prefix(nominal, injected, onset):
    prefix = [row for row in injected if row.timestamp < onset - 1e-6]
    reference = [row for row in nominal if row.timestamp < onset - 1e-6]
    for a, b in zip(reference, prefix):
        if a.subtask != b.subtask:
            return len(prefix), False
        for field in type(a).model_fields:
            if field != "subtask" and not np.allclose(
                getattr(a, field), getattr(b, field), atol=1e-8, rtol=0,
            ):
                return len(prefix), False
    return len(prefix), len(reference) == len(prefix)


def load_aligned_clocks(task_dir, sidecar):
    original = _index(task_dir / "diagnostics.jsonl")
    aligned = _index(sidecar)
    metadata = read_metadata_csv(task_dir / "metadata.csv")
    if set(aligned) != set(original) or set(aligned) != {row.pair_id for row in metadata}:
        raise ValueError(f"{sidecar}: pair IDs must exactly match archived metadata and diagnostics")

    clocks = {}
    for meta in metadata:
        pair_id = meta.pair_id
        try:
            record = aligned[pair_id]
            stored = original[pair_id]
            if any(record[key] != stored[key] for key in ("initial_condition_id", "failure_config")):
                raise ValueError("pair configuration differs from archived diagnostics")
            for run in ("nominal", "injected"):
                if any(record[run][key] != stored[run][key] for key in ("task", "seed", "initial_condition_id")):
                    raise ValueError(f"{run} identity differs from archived diagnostics")
            nominal_start = _seconds(record["nominal"]["telemetry_start_sim_s"])
            injected_start = _seconds(record["injected"]["telemetry_start_sim_s"])
            failure = record["injected"]["failure"]
            activation = _seconds(failure["first_activation_s"])
            onset = _seconds(failure["first_activation_csv_s"])
            if activation != stored["injected"]["failure"]["first_activation_s"]:
                raise ValueError("absolute injection time differs from archived diagnostics")
            if not math.isclose(activation - injected_start, onset, rel_tol=0, abs_tol=1e-6):
                raise ValueError("CSV injection time disagrees with simulator clock")
            if not math.isclose(nominal_start, injected_start, rel_tol=0, abs_tol=1e-6):
                raise ValueError("paired logging origins differ")
            nominal = read_telemetry_csv(task_dir / meta.success_file)
            injected = read_telemetry_csv(task_dir / meta.failure_file)
            if onset > injected[-1].timestamp + 0.022:
                raise ValueError("injection occurs beyond the last sample and logging tolerance")
            count, agrees = _compare_prefix(nominal, injected, onset)
            clocks[pair_id] = {
                "injection_timestamp": onset,
                "nominal_telemetry_start_sim_s": nominal_start,
                "injected_telemetry_start_sim_s": injected_start,
                "pre_injection_samples": count,
                "pre_injection_agrees": agrees if count or not agrees else None,
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{sidecar}: pair {pair_id}: {exc}") from exc
    return clocks
