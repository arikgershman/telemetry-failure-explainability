
"""Integration tests for the released MuJoCo telemetry dataset."""

import csv
from collections import Counter
from pathlib import Path

import pytest

from telemetry_reasoning.io import read_telemetry_csv


DATASET_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "dataset"

TASKS = ("pick_up_object", "transport_object")
FAILURE_MODES = {
    "position_error",
    "orientation_error",
    "inadequate_grip_strength",
    "early_gripper_release",
}


@pytest.fixture(scope="module")
def dataset():
    """Find the downloaded dataset, or skip when it isn't installed."""
    if not DATASET_ROOT.exists():
        pytest.skip("Released dataset not downloaded")

    batches = [
        path for path in DATASET_ROOT.iterdir()
        if path.is_dir()
    ]
    if len(batches) != 1:
        pytest.fail(f"Expected one dataset batch, found {len(batches)}")

    return batches[0]


@pytest.mark.parametrize("task", TASKS)
def test_dataset_structure(dataset, task):
    """Each task should have 500 paired success/failure runs."""
    task_dir = dataset / task

    assert (task_dir / "metadata.csv").is_file()
    assert (task_dir / "diagnostics.jsonl").is_file()
    assert (task_dir / "summary.json").is_file()

    for outcome in ("success", "failure"):
        files = list((task_dir / outcome).glob("*.csv"))
        assert len(files) == 500, (
            f"{task}/{outcome}: expected 500 traces, found {len(files)}"
        )


@pytest.mark.parametrize("task", TASKS)
def test_failure_mode_distribution(dataset, task):
    """Each task should have 125 pairs per failure mode."""
    with (dataset / task / "metadata.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 500

    modes = Counter(row["failure_mode"] for row in rows)
    assert modes == Counter({mode: 125 for mode in FAILURE_MODES})


@pytest.mark.parametrize("task", TASKS)
def test_telemetry_parser_compatibility(dataset, task):
    """All released traces must pass the canonical telemetry parser."""
    for outcome in ("success", "failure"):
        files = sorted((dataset / task / outcome).glob("*.csv"))

        for path in files:
            try:
                rows = read_telemetry_csv(path)
            except Exception as exc:
                pytest.fail(f"Cannot parse {path}: {exc}")

            assert rows, f"Empty telemetry trace: {path}"

@pytest.mark.parametrize("task", TASKS)
def test_initial_condition_distribution(dataset, task):
    """Each task must contain 50 initial conditions, with 10 pairs each."""
    import json

    diagnostics_path = dataset / task / "diagnostics.jsonl"

    with diagnostics_path.open(encoding="utf-8") as stream:
        diagnostics = [
            json.loads(line)
            for line in stream
            if line.strip()
        ]

    assert len(diagnostics) == 500

    conditions = Counter(
        str(record["initial_condition_id"])
        for record in diagnostics
    )

    assert conditions == Counter({
        str(condition_id): 10
        for condition_id in range(50)
    })
