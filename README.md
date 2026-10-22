# Telemetry-Driven Abductive Reasoning for Natural Language Explanations of Robotics Failures

Foundational scaffolding for natural language explanations of robotics failures.
This repository currently provides data contracts, subtask compression, and a
synthetic paired run. Simulation, divergence computation, LLM inference/DPO, and
model evaluation are future work.

## Quick start

Use Python 3.10 or newer, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/generate_mock_pair.py
python -m pytest
```

The generator writes `success_000.csv`, `failure_000.csv`, `metadata.csv`, and
`pair_000_bt.json` under `data/sample/`. Each CSV trace has 400 observations at
50 Hz (0.00 through 7.98 seconds). Both runs share initial conditions and seed
723. Grip force drops from 15 N to 3 N during the failed `grasp`; the object
subsequently remains on its support during `lift`. These are illustrative
signals, not physics outputs. Re-running replaces these four sample files.

```bash
python scripts/generate_mock_pair.py --output-dir /tmp/mock-pair --seed 723
python scripts/generate_mock_pair.py --config configs/default.yaml
```

Core installation includes NumPy, pandas, Pydantic v2, PyYAML, Matplotlib, and
pytest. Heavy libraries are optional and unused by the smoke tests:

```bash
python -m pip install -e '.[sim]'  # MuJoCo
python -m pip install -e '.[llm]'  # torch, transformers, trl, peft, bitsandbytes
```

## Shared interfaces

| Location | Responsibility |
| --- | --- |
| `src/telemetry_reasoning/schemas.py` | Task/failure enums, raw rows, metadata, paired BT, divergence placeholder, baseline views |
| `src/telemetry_reasoning/io.py` | Validated telemetry and metadata CSV readers/writers |
| `src/telemetry_reasoning/parser/` | Consecutive segmentation, compression, and BT construction |
| `src/telemetry_reasoning/sim/` | Placeholder for simulation/logging |
| `src/telemetry_reasoning/models/` | Placeholder for baselines, inference, and training |
| `src/telemetry_reasoning/eval/` | Placeholder for evaluation |
| `configs/default.yaml` | Timing and compression configuration |
| `scripts/`, `tests/`, `data/sample/` | Synthetic fixture generator, contract tests, and versioned examples |

`data/raw/`, `data/processed/`, virtual environments, and model weights are
ignored; `data/sample/` is explicitly allowed in Git.

## Data contract v1

Task templates are `pick_up_object` → `approach`, `align`, `grasp`, `lift`, and
`transport_object` → `maintain_grasp`, `move_along_path`, `reach_target`.
Failure modes are `position_error`, `orientation_error`,
`inadequate_grip_strength`, `early_gripper_release`, and `excessive_speed`.

- CSV headers match the Pydantic field names exactly. Vectors occupy JSON array
  cells (CSV-quoted as needed), rather than separate component columns.
- Timestamps are seconds, strictly increasing at 0.02 s intervals in raw runs.
  Compressed samples retain their original timestamps and have variable gaps.
- Joint arrays have the same nonzero length within a row and throughout paired
  runs. Joint positions use radians, velocities rad/s, and torques N m.
- `ee_pos`, `obj_pos`, and `obj_vel` have three components in x/y/z order;
  positions use meters and velocity m/s. Quaternions have four components in
  **w/x/y/z** order. Producers should emit normalized quaternions with consistent
  signs; v1 validates shape and finiteness, not quaternion normalization.
- `gripper_state` is normalized closure in [0, 1], with 0=open and 1=closed.
  `gripper_force` and `contact_force` are nonnegative scalar magnitudes in N.
- Metadata has `pair_id`, `task`, `success_file`, `failure_file`, `failed_subtask`,
  `failure_mode`, `perturbation_param`, `perturbation_value`, and `seed`.
  Pair IDs are strings (preserving `"000"`), file paths resolve relative to the
  supplied data directory, perturbation values are finite numbers, and seeds
  are nonnegative integers. A perturbation value uses its parameter's units.
- Unknown fields, invalid enums, malformed vector shapes, non-finite values,
  duplicate timestamps, and incorrect logging cadence are rejected.

The paired BT nests `metadata`, `overall_goal`, and ordered `subtasks`. Each node
contains its `subtask`, semantic `metadata` (`subgoal_name`, `action_description`),
`success_samples`, `failure_samples`, and `deterministic_divergence: null`.
Null means **not computed**. The reserved divergence schema provides
`first_divergence_timestamp` and per-feature `max_abs_deltas`; an eventual
implementation must specify alignment and metric semantics before filling it.

```python
from pathlib import Path
from telemetry_reasoning.schemas import PairedBehaviorTree

tree = PairedBehaviorTree.model_validate_json(
    Path("data/sample/pair_000_bt.json").read_text()
)
baseline_1_and_2 = tree.to_failure_only()
baseline_3 = tree.to_paired_without_divergence()
main_model = tree.to_full_paired()
json_schema = PairedBehaviorTree.model_json_schema()
```

These views retain task semantics but omit ground-truth failure labels,
perturbations, filenames, and seeds. Use `model_dump()` / `model_dump_json()`
for dataset storage, and the view helpers for model inputs.

## Compression

For each consecutive subtask segment, retain endpoints and up to `k=5` uniformly
spaced interior observations. Interior indices are the rounded interior points
of `linspace(0, n-1, k+2)` after capping k at `n-2`. Whenever a feature's adjacent
absolute change **exceeds** its threshold, retain `t-1`, `t`, and `t+1`, where t
is the later row. Vector comparisons use the largest absolute component change,
including quaternion components (not an angular distance). Clip neighbors to the
segment, deduplicate indices, and return them chronologically.

Thresholds in `configs/default.yaml` are initial mock-data values, not values
specified by the proposal or calibrated on real runs. Omitted features do not
trigger events; `k=0` disables uniform interior sampling. Events may retain more
than seven samples, or even the whole segment. Short segments keep all available
rows when k exceeds their interior count.

The compressor keeps separate visits to repeated subtask labels. BT construction
currently requires each run to contain every task-template subtask once, in
order, and rejects missing/repeated/reordered segments instead of silently
aligning them. Success and failure segment durations and sample counts may differ.
No divergence calculation or time alignment is performed.
