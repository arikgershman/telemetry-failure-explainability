
"""Validate nominal MuJoCo executions and canonical telemetry."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from telemetry_reasoning.io import read_telemetry_csv
from telemetry_reasoning.schemas import TASK_SUBTASKS, Task

ROOT = Path(__file__).resolve().parents[3]

TASKS = {
    "pickup": (
        "telemetry_reasoning.sim.batch",
        Task.PICK_UP_OBJECT,
        ROOT / "outputs/pickup",
    ),
    "transport": (
        "telemetry_reasoning.sim.transport_batch",
        Task.TRANSPORT_OBJECT,
        ROOT / "outputs/transport",
    ),
}


def validate_trace(path, task):
    try:
        rows = read_telemetry_csv(path)
    except Exception as exc:
        return [f"Schema/cadence validation: {exc}"]

    expected = list(TASK_SUBTASKS[task])
    observed = []

    for row in rows:
        if not observed or row.subtask != observed[-1]:
            observed.append(row.subtask)

    errors = []

    if observed != expected:
        errors.append(
            f"Subtask sequence: expected "
            f"{[s.value for s in expected]}, "
            f"got {[s.value for s in observed]}"
        )

    if abs(rows[0].timestamp) > 1e-8:
        errors.append(
            f"First timestamp is {rows[0].timestamp}, expected 0.00"
        )

    return errors


def run_task(name, runs, seed, spread):
    module, task, parent = TASKS[name]

    print(f"\n{'=' * 55}")
    print(f"VALIDATING {name.upper()}")
    print(f"{'=' * 55}")

    parent.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in parent.glob("batch_*") if p.is_dir()}

    result = subprocess.run(
        [
            sys.executable, "-m", module,
            "--runs", str(runs),
            "--seed", str(seed),
            "--spread", str(spread),
        ],
        cwd=ROOT,
    )

    if result.returncode != 0:
        print("FAIL: Simulation runner crashed")
        return False

    new_batches = sorted(
        [
            p for p in parent.glob("batch_*")
            if p.is_dir() and p.name not in before
        ],
        key=lambda p: p.stat().st_mtime,
    )

    if not new_batches:
        print("FAIL: No new batch found")
        return False

    batch = new_batches[-1]
    print("Validating batch:", batch)

    with (batch / "summary.csv").open(newline="") as f:
        outcomes = list(csv.DictReader(f))

    successes = sum(
        row["result"] == "success" for row in outcomes
    )

    print(f"\nSuccess rate: {successes}/{len(outcomes)}")
    print(f"Batch: {batch}")

    valid = True

    if len(outcomes) != runs or successes != runs:
        valid = False
        print("FAIL: Nominal run count or success rate")
        for row in outcomes:
            if row["result"] != "success":
                print(
                    f"  Run {row['run']}: "
                    f"{row['result']} ({row['reason']})"
                )

    traces = sorted(batch.glob(f"{name}_*.csv"))

    if len(traces) != runs:
        print(
            f"FAIL: Expected {runs} traces, found {len(traces)}"
        )
        valid = False

    for path in traces:
        errors = validate_trace(path, task)
        if errors:
            valid = False
            print(f"\n{path.name}:")
            for error in errors:
                print(f"  - {error}")

    if valid:
        print("PASS: Nominal simulation, schema, cadence, and subtasks")
    else:
        print("FAIL: See issues above")

    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=723)
    parser.add_argument("--spread", type=float, default=0.015)
    parser.add_argument(
        "--task",
        choices=["pickup", "transport", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.spread < 0:
        parser.error("--spread must be nonnegative")

    names = (
        list(TASKS) if args.task == "all" else [args.task]
    )

    results = [
        run_task(name, args.runs, args.seed, args.spread)
        for name in names
    ]

    if all(results):
        print("\nALL VALIDATIONS PASSED")
    else:
        print("\nVALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
