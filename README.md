# Telemetry-Driven Abductive Reasoning for Natural Language Explanations of Robotics Failures

An implementation of paired robotics telemetry collection, Behavior Tree (BT)
compression, deterministic divergence, failure diagnosis, and evaluation.
The repository includes a MuJoCo Panda simulator, dataset validation and grouped
splits, a failure-only rule baseline, and a shared Hugging Face harness for three
prompt variants. A CPU integration smoke exercises these interfaces using fresh
synthetic telemetry and explicitly labeled stub predictions.

Production compression is selected and all 1,000 archived pairs have enriched
BTs. CPU workflows cover real physics, the training formatter, and exact-tokenized
GPU preparation. Real main-model inference, reviewed preference annotations,
GPU optimization, the held-out study, and the archived dataset's final
physical-outcome/provenance certification remain open. See [TODO.md](TODO.md)
and the [GPU handoff](docs/gpu-handoff.md) for evidence and runnable next steps.

## Quick start

Use Python 3.10 or newer, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/integration_smoke.py --backend stub --output-dir runs/integration-cpu
python -m pytest
```

Use a fresh output directory for each integration run. The smoke generates new
mock CSVs, validates and parses them, computes divergence, runs all four variants
(rules plus three stubs), verifies prediction evidence, evaluates explicit result
files, and renders telemetry plots. Stub outputs are plumbing checks and are
labeled synthetic; they do not establish HF inference or model quality.

Inspect `runs/integration-cpu/integration_manifest.json` for the final status,
acceptance checks, stage commands, and artifact hashes. Stage logs are retained
for failures as well as successful runs. The CLI returns 0 on success, 1 on a
failed stage/check, and 2 on invalid setup. `--sim-backend mujoco` additionally
exercises fresh physical simulation on the CPU; inference remains explicitly
stubbed in that producer gate. Real HF acceptance uses the prepared GPU workflow.

The versioned `data/sample/` fixture contains `success_000.csv`, `failure_000.csv`,
`metadata.csv`, and `pair_000_bt.json`. Each CSV trace has 400 observations at
50 Hz (0.00 through 7.98 seconds), sharing initial conditions and seed 723.
Grip force drops from 15 N to 3 N at 5.10 s during failed `grasp`; the object
subsequently remains on its support during `lift`. These are illustrative
signals. The stored BT intentionally has uncomputed divergence.

Generate a separate copy to inspect the fixture without replacing versioned
files:

```bash
python scripts/generate_mock_pair.py --output-dir /tmp/telemetry-mock --seed 723
```

Core installation includes NumPy, pandas, Pydantic v2, PyYAML, Matplotlib, and
pytest. The default suite is CPU/offline compatible; dataset-dependent and
opt-in real-physics checks report skips when their inputs are unavailable.
Heavy libraries are optional and unused by the default mock integration smoke:

```bash
python -m pip install -e '.[sim]'  # MuJoCo
python -m pip install -e '.[llm]'  # torch, transformers, trl, peft, bitsandbytes
```

## MuJoCo Simulation

MuJoCo is used for physics-based robotics simulation, with robot models provided by the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). Our initial simulation uses the Franka Emika Panda robotic arm.

### Setup

Install the simulation dependencies from the repository root:

```bash
python -m pip install -e '.[sim]'
```

Initialize the MuJoCo Menagerie Git submodule:

```bash
git submodule update --init --recursive
```

Alternatively, when cloning the repository for the first time:

```bash
git clone --recurse-submodules <repository-url>
```

The Panda model is located at:

```text
assets/mujoco_menagerie/franka_emika_panda/scene.xml
```

### Running the Simulation

On macOS, launch the Panda viewer from the repository root:

```bash
mjpython -m telemetry_reasoning.sim.viewer
```

The viewer uses `mujoco.viewer.launch_passive()` with an explicit simulation loop that advances physics using `mujoco.mj_step()` and synchronizes visualization using `viewer.sync()`.

## Shared interfaces

| Location | Responsibility |
| --- | --- |
| `src/telemetry_reasoning/schemas.py` | Task/failure enums, raw rows, metadata, paired BT/divergence contracts, leakage-free model views |
| `src/telemetry_reasoning/io.py` | Validated telemetry and metadata CSV readers/writers |
| `src/telemetry_reasoning/parser/` | Segmentation, compression, batch BT construction, alignment, divergence, diagnostics, grouped tuning |
| `src/telemetry_reasoning/sim/` | MuJoCo controllers, command perturbations, telemetry, paired generation, validation, splits, checksums |
| `src/telemetry_reasoning/models/` | Rule baseline, leakage-free prompts, HF/stub inference, offline response revalidation |
| `src/telemetry_reasoning/eval/` | Strict prediction joins, evidence verification, scoring, blinded rubric, plots |
| `configs/` | Separate parser, divergence, rules, evaluation, and model/prompt configurations |
| `scripts/`, `tests/`, `data/sample/` | Public CLIs, CPU and opt-in acceptance checks, versioned synthetic examples |

`data/raw/`, `data/processed/`, virtual environments, and model weights are
ignored; `data/sample/` is explicitly allowed in Git.

## Data contract

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
  signs; storage validates shape and finiteness, not quaternion normalization.
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
`success_samples`, `failure_samples`, observation/end states, and divergence
availability. Successes cover the complete template; failures cover a nonempty
ordered prefix. Every expected node remains present, with `failure_samples: []`
for an unobserved suffix. `failure_observation_status` is `observed` or
`not_observed`; `failure_segment_end` is `transitioned`, `trace_end`, or null
for an unobserved stage. These states describe observation, not physical success.
`divergence_status` distinguishes `uncomputed`, `computed`, and `not_comparable`.
A null divergence object never means zero. Computed divergence provides
`first_divergence_timestamp` and per-feature `max_abs_deltas`. Alignment and
metric semantics are documented in the [parser handoff](src/telemetry_reasoning/parser/README.md).

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

Baseline 1 applies failure-only rules for all five modes and abstains when
observable evidence is ambiguous or unavailable. Its excessive-speed rule uses
end-effector displacement divided by actual elapsed time between retained
observations (m/s), with persistence and direction checks within a stage.
The configured speed limit is provisional and requires producer calibration.
These interval averages cannot recover unseen peaks or a command multiplier,
and a detected limit violation does not establish physical causality.

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
requires complete successes and accepts failure prefixes; skipped interior,
repeated, and reordered stages are rejected. Samples are matched by subtask name
under the complete task structure. Numerical alignment uses normalized progress
for stages followed by a transition and common elapsed time for the final
observed stage, without extrapolation. Unobserved stages are not comparable.
Divergence is an explicit second stage; parsing leaves observed stages
uncomputed. Enrichment computes maximum absolute component differences and
the first strict per-feature threshold crossing on the aligned comparison
grid. Its timestamp is expressed in the failure trace's time coordinate and
does not establish exact physical onset. Unobserved stages remain unavailable.

## Parse, enrich, predict, and evaluate

Run from the repository root after installation, with fresh output directories:

```bash
python scripts/generate_mock_pair.py --output-dir runs/manual-smoke/raw --seed 723
python scripts/parse_dataset.py --metadata runs/manual-smoke/raw/metadata.csv --data-dir runs/manual-smoke/raw --config configs/parser.yaml --output-dir runs/manual-smoke/bt
python scripts/enrich_divergence.py --bt-dir runs/manual-smoke/bt --config configs/divergence.yaml --output-dir runs/manual-smoke/enriched-bt
python scripts/run_baseline.py --bt-dir runs/manual-smoke/enriched-bt --config configs/rules.yaml --output runs/manual-smoke/baseline_1.jsonl
python scripts/run_llm.py --bt-dir runs/manual-smoke/enriched-bt --config configs/llm.yaml --variants baseline_2 baseline_3 main --backend stub --output-dir runs/manual-smoke/llm
python scripts/evaluate.py --metadata runs/manual-smoke/raw/metadata.csv --predictions runs/manual-smoke/baseline_1.jsonl runs/manual-smoke/llm/baseline_2.jsonl runs/manual-smoke/llm/baseline_3.jsonl runs/manual-smoke/llm/main.jsonl --bt-dir runs/manual-smoke/enriched-bt --run-manifest runs/manual-smoke/baseline_1.manifest.json runs/manual-smoke/llm/run_manifest.json --config configs/eval.yaml --output-dir runs/manual-smoke/eval
python scripts/plot_telemetry.py --data-dir runs/manual-smoke/raw --bt-dir runs/manual-smoke/enriched-bt --output-dir runs/manual-smoke/plots
```

The integration smoke runs this sequence with acceptance assertions. Keep task
outputs separate for the archived release because task-local pair IDs overlap.
Never pass storage `model_dump()` output to a predictor: use the model-view
helpers so labels, perturbations, source paths, and seeds stay out of inputs.

The [release workflow](docs/dataset-release.md) describes pinned online/offline
restoration, preserved source bytes, recovered configuration provenance, and
certification evidence requirements. The Nov. 18 restore verified all 2,006
original task files. Current-runtime fresh replay passed, while all 16 archived
run records in the representative replay mismatched; archived timing, terminal
outcomes, and exact producer provenance remain uncertified. The canonical
`runs/dataset-v1-20261118/handoff` contains 1,000 accepted pairs and
700/100/200 grouped train/dev/test assignments, with the explicit status
`parser_ready_certification_failed` for the known archival replay failure.

The [parser handoff](src/telemetry_reasoning/parser/README.md) covers alignment,
reports, tuning, and divergence. The [dataset handoff](docs/track2-dataset-handoff.md)
records archived validation and grouped split inputs; reproducing those checks
requires a separate copy of the ignored dataset and producer assets. The
[model handoff](src/telemetry_reasoning/models/README.md) covers resource guards
and HF execution. Real HF baseline receipts and their limits are recorded in
[the Nov. 6 acceptance record](docs/track5-nov06-acceptance.md).

## Production outputs and GPU continuation

The [production parser](docs/production-parser.md) publishes 1,000 enriched BTs
under `data/processed/production-v1-20261118/<task>/enriched-bt`, using dev-selected
`configs/parser-production-v1.yaml`. All 101,357 fixed-reference event
neighborhoods are retained; numerical onset diagnostics keep physical onset
and unverified archived injection clocks separate.

The [GPU handoff](docs/gpu-handoff.md) and [Ubuntu 24.04 runbook](docs/alienware-ubuntu24.md)
document portable source-bound inputs and exact setup/transfer commands.
`runs/alienware-fresh-smoke-v2-20261119` contains eight verified physical pairs/24
fitting requests; `runs/alienware-fresh-all-v2-20261119` retains all 18 fresh pairs,
including 10 context refusals. The optional `runs/alienware-v1-dev-v2-20261119`
contains all 100 archived dev pairs/300 requests, including 39 refusals.
Use `--inference-only` on the GPU; `scripts/evaluate_gpu_results.py` independently
validates returned artifacts and generates four-variant metrics/review sheets/
plots on the Mac without model loading. CPU rehearsals establish no new HF results.

Fresh real-producer acceptance is recorded in
`runs/integration-mujoco-cpu-20261118/integration_manifest.json`: 18 physical
pairs, 76 executions, all four represented modes per task, measured outcomes,
strict replay, and explicit pre-injection/context coverage. The rule baseline's
joint accuracy on those cases is only 1/18; contract correctness does not imply
diagnostic quality.

The [training formatter](docs/training-data.md) and [GPU training launcher](docs/gpu-training.md)
are implemented and CPU-tested. The 24-record formatter demonstration is
synthetic and ineligible for optimization. Reviewed train/dev annotations are
required before GPU LoRA/DPO training; test labels remain held out.

The [Nov. 19 inference fix](docs/alienware-inference-fix-20261119.md) corrects a
resident-memory admission false refusal without lowering safety thresholds.
Apply its incremental transfer after the original full handoff. Production v2
accepts whole-response JSON fences; independently reproduced invalid citations
remain errors and can be scored on the Mac with `--score-response-errors`.

## Current Alienware recovery: v3

The later v2 retry exposed genuine long-prompt CUDA OOMs and invalid/truncated
model outputs. Follow the [v3 recovery runbook](docs/alienware-v3-recovery-20261119.md),
which supersedes the earlier v2 retry instructions. It reuses the existing
checkpoint, environment and passed fused-kernel report. Native 2,048-token
prefill, fused-only SDPA and a versioned observed-JSON decoder address the
execution and output-contract failures. Exact citations, all five failure modes,
and model-selected diagnoses/abstentions remain; outputs are not repaired.

The new `runs/alienware-fresh-smoke-v3-20261119` requires a measured longest-request
qualification before its 24-request GPU run. Complete model acceptance is still
pending. Compare all three prompted variants within v3, return only results and
qualification evidence, and compute metrics/plots/review sheets on the Mac.
