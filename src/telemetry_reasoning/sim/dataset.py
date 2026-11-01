
"""Generate balanced, paired success/failure telemetry datasets.

For 500 pairs per task:
- Exactly 50 initial conditions
- Exactly 10 accepted pairs per initial condition
- Exactly 125 accepted pairs per failure mode
- Nominal and failed runs share identical initial conditions
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from telemetry_reasoning.io import (
    write_metadata_csv,
    write_telemetry_csv,
)
from telemetry_reasoning.schemas import (
    FailureMode,
    MetadataRow,
    Subtask,
    Task,
)
from telemetry_reasoning.sim.failures import FailureConfig
from telemetry_reasoning.sim.runner import ROOT, run_episode


OUTPUT_ROOT = ROOT / "outputs" / "dataset"
NUM_INITIAL_CONDITIONS = 50

DATASET_MODES = (
    FailureMode.POSITION_ERROR,
    FailureMode.ORIENTATION_ERROR,
    FailureMode.INADEQUATE_GRIP_STRENGTH,
    FailureMode.EARLY_GRIPPER_RELEASE,
)


def load_calibrated_configs(paths, threshold=0.9):
    """Load reliable configurations from calibration CSV files."""
    by_task = defaultdict(lambda: defaultdict(list))

    for path in paths:
        with Path(path).open(
            newline="", encoding="utf-8"
        ) as f:
            for row in csv.DictReader(f):
                mode = FailureMode(row["mode"])

                if mode not in DATASET_MODES:
                    continue

                if (
                    float(row["failure_rate"]) < threshold
                    or float(row["activation_rate"]) < threshold
                ):
                    continue

                task = Task(row["task"])

                config = FailureConfig(
                    mode=mode,
                    subtask=Subtask(row["subtask"]),
                    magnitude=float(row["magnitude"]),
                    delay_s=float(row["delay_s"]),
                    axis=int(row["axis"]),
                )

                if config not in by_task[task][mode]:
                    by_task[task][mode].append(config)

    return by_task


def build_schedule(pairs, seed):
    """Assign a fixed condition and failure mode to every dataset slot.

    For 500 pairs:
        50 conditions x 10 repetitions
        4 modes x 125 pairs

    A rejected injection never changes the slot's condition or mode.
    """
    if pairs < 1:
        raise ValueError("pairs must be positive")

    condition_count = min(NUM_INITIAL_CONDITIONS, pairs)

    if pairs % condition_count != 0:
        raise ValueError(
            f"pairs={pairs} cannot be evenly distributed across "
            f"{condition_count} initial conditions"
        )

    if pairs % len(DATASET_MODES) != 0:
        raise ValueError(
            f"pairs must be divisible by {len(DATASET_MODES)} "
            "for exact failure-mode balance"
        )

    repetitions = pairs // condition_count

    # Every condition occurs exactly 'repetitions' times.
    conditions = [
        condition
        for condition in range(condition_count)
        for _ in range(repetitions)
    ]

    # Every failure mode occurs exactly pairs/4 times.
    modes = [
        DATASET_MODES[i % len(DATASET_MODES)]
        for i in range(pairs)
    ]

    # Randomize slot ordering without changing quotas.
    schedule = list(zip(conditions, modes))
    random.Random(seed).shuffle(schedule)

    return schedule


def perturbation_parameter(mode):
    return {
        FailureMode.POSITION_ERROR: "position_offset_m",
        FailureMode.ORIENTATION_ERROR: "orientation_offset_rad",
        FailureMode.INADEQUATE_GRIP_STRENGTH: "gripper_ctrl_min",
        FailureMode.EARLY_GRIPPER_RELEASE: "release_command",
    }[mode]


def generate(
    task,
    pairs,
    seed,
    max_attempts,
    output,
    spread,
    calibrated,
):
    task = Task(task)

    available = calibrated.get(task, {})

    for mode in DATASET_MODES:
        if not available.get(mode):
            raise ValueError(
                f"{task.value}: no reliable configurations "
                f"for {mode.value}"
            )

    schedule = build_schedule(
        pairs,
        seed + (0 if task == Task.PICK_UP_OBJECT else 1),
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "success").mkdir(exist_ok=True)
    (output / "failure").mkdir(exist_ok=True)

    rng = random.Random(
        seed + (0 if task == Task.PICK_UP_OBJECT else 1)
    )

    metadata = []
    rejects = Counter()
    accepted_by_condition = Counter()
    accepted_by_mode = Counter()

    attempts = 0

    diagnostics_path = output / "diagnostics.jsonl"

    for slot, (condition_id, mode) in enumerate(schedule):
        accepted = False

        # The condition and failure mode are fixed for this slot.
        # Only the calibrated configuration can change on retry.
        while attempts < max_attempts:
            attempts += 1

            config = rng.choice(available[mode])

            nominal = run_episode(
                task=task,
                seed=seed,
                initial_condition_id=condition_id,
                spread=spread,
            )

            if (
                not nominal.task_succeeded
                or not nominal.diagnostics["complete_subtasks"]
            ):
                rejects["nominal_failed"] += 1

                # Retrying an unchanged nominal run is not useful.
                raise RuntimeError(
                    f"{task.value}: nominal episode failed "
                    f"for condition {condition_id}. "
                    "Cannot guarantee the requested condition coverage."
                )

            injected = run_episode(
                task=task,
                seed=seed,
                initial_condition_id=condition_id,
                failure=config,
                spread=spread,
            )

            if not injected.diagnostics["failure"]["activated"]:
                rejects["not_activated"] += 1
                continue

            if injected.task_succeeded:
                rejects["injection_did_not_fail"] += 1
                continue

            if not injected.diagnostics["observed_subtasks"]:
                rejects["no_canonical_telemetry"] += 1
                continue

            pair_id = f"{slot:06d}"

            success_rel = f"success/{pair_id}.csv"
            failure_rel = f"failure/{pair_id}.csv"

            write_telemetry_csv(
                output / success_rel,
                nominal.rows,
            )

            write_telemetry_csv(
                output / failure_rel,
                injected.rows,
            )

            metadata.append(
                MetadataRow(
                    pair_id=pair_id,
                    task=task,
                    success_file=success_rel,
                    failure_file=failure_rel,
                    failed_subtask=config.subtask,
                    failure_mode=config.mode,
                    perturbation_param=perturbation_parameter(
                        config.mode
                    ),
                    perturbation_value=float(config.magnitude),
                    seed=seed,
                )
            )

            diagnostic = {
                "pair_id": pair_id,
                "initial_condition_id": condition_id,
                "failure_config": {
                    **asdict(config),
                    "mode": config.mode.value,
                    "subtask": config.subtask.value,
                },
                "nominal": nominal.diagnostics,
                "injected": injected.diagnostics,
                "failure_trace_is_partial": (
                    not injected.diagnostics["complete_subtasks"]
                ),
            }

            with diagnostics_path.open(
                "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(diagnostic) + "\n")

            accepted_by_condition[condition_id] += 1
            accepted_by_mode[mode.value] += 1

            accepted = True

            print(
                f"[{task.value}] accepted "
                f"{len(metadata)}/{pairs}, "
                f"attempts={attempts}, "
                f"condition={condition_id}, "
                f"mode={mode.value}, "
                f"stage={config.subtask.value}"
            )

            break

        if not accepted:
            break

    if metadata:
        write_metadata_csv(
            output / "metadata.csv",
            metadata,
        )

    expected_condition_count = min(
        NUM_INITIAL_CONDITIONS, pairs
    )

    expected_per_condition = (
        pairs // expected_condition_count
    )

    expected_per_mode = (
        pairs // len(DATASET_MODES)
    )

    condition_coverage_valid = (
        len(accepted_by_condition) == expected_condition_count
        and all(
            accepted_by_condition[i] == expected_per_condition
            for i in range(expected_condition_count)
        )
    )

    mode_balance_valid = all(
        accepted_by_mode[mode.value] == expected_per_mode
        for mode in DATASET_MODES
    )

    summary = {
        "task": task.value,
        "requested_pairs": pairs,
        "accepted_pairs": len(metadata),
        "attempts": attempts,
        "rejected": dict(rejects),
        "initial_conditions_used": len(accepted_by_condition),
        "expected_initial_conditions": expected_condition_count,
        "expected_pairs_per_condition": expected_per_condition,
        "accepted_by_condition": {
            str(i): accepted_by_condition[i]
            for i in range(expected_condition_count)
        },
        "accepted_by_mode": dict(accepted_by_mode),
        "condition_coverage_valid": condition_coverage_valid,
        "mode_balance_valid": mode_balance_valid,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    if len(metadata) != pairs:
        raise RuntimeError(
            f"{task.value}: generated only "
            f"{len(metadata)}/{pairs} pairs. "
            "Increase --max-attempts or inspect rejected attempts."
        )

    if not condition_coverage_valid:
        raise RuntimeError(
            f"{task.value}: initial-condition coverage failed"
        )

    if not mode_balance_valid:
        raise RuntimeError(
            f"{task.value}: failure-mode balance failed"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["pickup", "transport", "all"],
        default="all",
    )

    parser.add_argument(
        "--pairs-per-task",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=723,
    )

    parser.add_argument(
        "--spread",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT,
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to calibrated recommended_configs.csv files",
    )

    parser.add_argument(
        "--min-failure-rate",
        type=float,
        default=0.9,
    )

    args = parser.parse_args()

    if (
        args.pairs_per_task < 1
        or args.max_attempts < 1
        or args.seed < 0
        or args.spread < 0
    ):
        parser.error(
            "Invalid pairs, attempts, seed, or spread"
        )

    if not 0 <= args.min_failure_rate <= 1:
        parser.error(
            "--min-failure-rate must be between 0 and 1"
        )

    calibrated = load_calibrated_configs(
        args.calibration,
        args.min_failure_rate,
    )

    batch_id = (
        f"batch_{datetime.now():%Y%m%d_%H%M%S}"
        f"_{uuid4().hex[:8]}"
    )

    folder = args.output / batch_id
    folder.mkdir(parents=True, exist_ok=False)

    print(f"Dataset output: {folder}")

    tasks = (
        [Task.PICK_UP_OBJECT, Task.TRANSPORT_OBJECT]
        if args.task == "all"
        else [
            Task.PICK_UP_OBJECT
            if args.task == "pickup"
            else Task.TRANSPORT_OBJECT
        ]
    )

    for task in tasks:
        generate(
            task=task,
            pairs=args.pairs_per_task,
            seed=args.seed,
            max_attempts=args.max_attempts,
            output=folder / task.value,
            spread=args.spread,
            calibrated=calibrated,
        )

    print(f"\nDataset generation complete: {folder}")


if __name__ == "__main__":
    main()
