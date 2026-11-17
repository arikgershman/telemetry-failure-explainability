
"""Create reproducible train/dev/test splits by initial condition."""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

TASKS = ("pick_up_object", "transport_object")
SPLIT_SIZES = {"train": 35, "dev": 5, "test": 10}


def create_splits(batch: Path, seed: int = 723) -> dict:
    """Assign whole initial conditions to splits for each task."""
    rng = random.Random(seed)
    manifest = {
        "seed": seed,
        "split_unit": "initial_condition_id",
        "splits": {name: [] for name in SPLIT_SIZES},
        "condition_assignments": {},
    }

    for task in TASKS:
        task_dir = batch / task

        with (task_dir / "metadata.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            pairs = list(csv.DictReader(stream))

        with (task_dir / "diagnostics.jsonl").open(
            encoding="utf-8"
        ) as stream:
            diagnostics = [
                json.loads(line)
                for line in stream
                if line.strip()
            ]

        if len({str(record['pair_id']) for record in diagnostics}) != len(diagnostics):
            raise ValueError(f"{task}: duplicate diagnostic IDs")
        condition_by_pair = {
            str(record["pair_id"]): str(record["initial_condition_id"])
            for record in diagnostics
        }

        pair_ids = [str(pair["pair_id"]) for pair in pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError(f"{task}: duplicate pair IDs")
        if set(pair_ids) != set(condition_by_pair):
            raise ValueError(f"{task}: metadata/diagnostics pair IDs differ")

        conditions = sorted(set(condition_by_pair.values()), key=int)
        counts = Counter(condition_by_pair.values())

        if len(conditions) != 50 or any(count != 10 for count in counts.values()):
            raise ValueError(f"{task}: unexpected initial-condition distribution")

        rng.shuffle(conditions)

        assignments = {}
        start = 0
        for split, size in SPLIT_SIZES.items():
            selected = conditions[start:start + size]
            assignments[split] = sorted(selected, key=int)
            start += size

        manifest["condition_assignments"][task] = assignments

        split_by_condition = {
            condition: split
            for split, selected in assignments.items()
            for condition in selected
        }

        for pair in pairs:
            pair_id = str(pair["pair_id"])
            condition = condition_by_pair[pair_id]
            split = split_by_condition[condition]

            manifest["splits"][split].append({
                "task": task,
                "pair_id": pair_id,
                "initial_condition_id": condition,
                "failure_mode": pair["failure_mode"],
                "success_file": (Path(task) / pair["success_file"]).as_posix(),
                "failure_file": (Path(task) / pair["failure_file"]).as_posix(),
            })

    for split, pairs in manifest["splits"].items():
        for task in TASKS:
            modes = {pair['failure_mode'] for pair in pairs if pair['task'] == task}
            if modes != {'position_error', 'orientation_error', 'inadequate_grip_strength', 'early_gripper_release'}:
                raise ValueError(f"{task}/{split}: incomplete failure-mode coverage")

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--seed", type=int, default=723)
    parser.add_argument(
        "--output", type=Path, default=Path("dataset_splits.json")
    )
    args = parser.parse_args()

    manifest = create_splits(args.batch, args.seed)
    args.output.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    for split, pairs in manifest["splits"].items():
        print(f"{split}: {len(pairs)} pairs")

    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
