
import argparse
import json
from pathlib import Path

import numpy as np

from telemetry_reasoning.io import read_telemetry_csv
from telemetry_reasoning.sim.runner import run_episode


def compare_telemetry(original, replayed):
    """Require identical stages and nearly identical numerical data."""
    if len(original) != len(replayed):
        return False

    for a, b in zip(original, replayed):
        a = a.model_dump(mode="json")
        b = b.model_dump(mode="json")

        for key in a:
            if key == "subtask":
                if a[key] != b[key]:
                    return False
            elif not np.allclose(
                a[key], b[key], rtol=1e-7, atol=1e-6
            ):
                return False

    return True


def recover_task(folder):
    source = folder / "diagnostics.jsonl"
    records = [
        json.loads(line)
        for line in source.read_text().splitlines()
        if line.strip()
    ]

    # Replay once per unique initial condition.
    representatives = {}
    for record in records:
        cid = record["initial_condition_id"]
        representatives.setdefault(cid, record)

    offsets = {}

    for cid, record in sorted(representatives.items()):
        original_csv = (
            folder / "success" / f"{record['pair_id']}.csv"
        )

        replay = run_episode(
            task=record["nominal"]["task"],
            seed=record["nominal"]["seed"],
            initial_condition_id=cid,
        )

        if not replay.task_succeeded:
            raise RuntimeError(
                f"{folder.name}, condition {cid}: replay failed"
            )

        original = read_telemetry_csv(original_csv)

        if not compare_telemetry(original, replay.rows):
            raise RuntimeError(
                f"{folder.name}, condition {cid}: "
                "telemetry mismatch; recovery aborted"
            )

        origin = replay.diagnostics["telemetry_start_sim_s"]
        if origin is None:
            raise RuntimeError("Missing replay logging origin")

        offsets[cid] = float(origin)
        print(f"PASS {folder.name} condition {cid}: {origin:.6f}s")

    # Both runs have the same pre-injection trajectory.
    # Thus their logging origins should coincide.
    for record in records:
        cid = record["initial_condition_id"]
        origin = offsets[cid]

        record["nominal"]["telemetry_start_sim_s"] = origin
        record["injected"]["telemetry_start_sim_s"] = origin

        failure = record["injected"]["failure"]
        if failure and failure["first_activation_s"] is not None:
            injection_csv_s = (
                failure["first_activation_s"] - origin
            )

            if injection_csv_s < -1e-6:
                raise RuntimeError(
                    f"Injection precedes logging: {record['pair_id']}"
                )

            last_csv_s = read_telemetry_csv(
                folder / "failure" / f"{record['pair_id']}.csv"
            )[-1].timestamp

            if injection_csv_s > last_csv_s + 0.022:
                raise RuntimeError(
                    f"Injection outside recorded trace: "
                    f"{record['pair_id']}"
                )

            failure["first_activation_csv_s"] = max(
                0.0, injection_csv_s
            )

        record["clock_alignment_source"] = (
            "verified_nominal_replay"
        )

    # Never overwrite the original diagnostics.
    output = folder / "diagnostics.clock_aligned.jsonl"
    output.write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )

    print(
        f"\nSaved {len(records)} aligned records to {output}\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()

    for name in ("pick_up_object", "transport_object"):
        recover_task(args.batch / name)


if __name__ == "__main__":
    main()
