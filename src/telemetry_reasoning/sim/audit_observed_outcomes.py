"""Check object goals from the first and last archived 50 Hz observations."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.schemas import Task
from telemetry_reasoning.sim.split_robot_dataset import TASKS


def observed_goal(task, first, last):
    """Apply the v1 object-goal thresholds to sampled positions in meters."""
    if task == Task.PICK_UP_OBJECT:
        rise = last[2] - first[2]
        return rise > 0.12, {"object_rise_m": rise}
    target_xy = (first[0], first[1] + 0.15)
    xy_error = math.dist(last[:2], target_xy)
    z_error = abs(last[2] - 0.427)
    return xy_error < 0.04 and z_error < 0.025, {
        "placement_xy_error_m": xy_error, "placement_z_error_m": z_error,
    }


def audit_pair(folder, meta):
    record = {"task": meta.task.value, "pair_id": meta.pair_id,
              "failure_mode": meta.failure_mode.value}
    for kind in ("success", "failure"):
        rows = read_telemetry_csv(folder / getattr(meta, f"{kind}_file"))
        first, last = rows[0], rows[-1]
        met, measurements = observed_goal(meta.task, first.obj_pos, last.obj_pos)
        record[kind] = {
            "last_timestamp_s": last.timestamp,
            "last_subtask": last.subtask.value,
            "object_goal_met": met,
            **measurements,
        }
    return record


def audit_observed_outcomes(batch):
    pairs = []
    summary = {}
    for task in TASKS:
        folder = batch / task
        metadata = read_metadata_csv(folder / "metadata.csv")
        records = [audit_pair(folder, meta) for meta in metadata]
        pairs.extend(records)
        summary[task] = {
            kind: dict(Counter(
                "goal_met" if record[kind]["object_goal_met"] else "goal_unmet"
                for record in records
            ))
            for kind in ("success", "failure")
        }
    return {
        "summary": summary,
        "scope": "Object goals at the last archived observation. Controller termination may occur "
                 "between samples. Pickup also checks end-effector error in the controller.",
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_observed_outcomes(args.batch)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
