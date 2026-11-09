# Project TODO: five parallel workstreams

Each member claims **one track** by filling in its owner field. Each track owns implementation, configuration, tests, and a runnable demonstration. The five tracks are intended to have comparable scope; GPU training and the final study are not additional requirements for any one owner during this period.

## Milestones and current starting point

| Target | Proposal milestone | Track deliverables |
| --- | --- | --- |
| **Oct. 30** | Generate, run, and store MuJoCo data | Track 1: MuJoCo environment, nominal controllers, failure injection, and 500 accepted success/failure pairs per task. Track 2: validation and finalization begin after dataset generation. Tracks 3-5: work immediately against the existing mock contracts. |
| **Nov. 6** | Implement simulation parser and all three baselines | Track 3: batch parsing and compression tuning. Track 4: Baseline 1 and evaluator. Track 5: Baselines 2 and 3. Tracks 1-2: replay, coverage, and data-quality validation. |
| **Nov. 13** | Convert simulations to BTs and implement deterministic divergence | Track 3: full dataset conversion and divergence. Track 4: integrated metrics/plots. Track 5: main paired-BT-plus-divergence model. All tracks: end-to-end smoke test. |

Already implemented: `src/telemetry_reasoning/schemas.py`, validated CSV I/O, configuration loading, consecutive subtask segmentation, endpoint/uniform/major-event compression, and `build_paired_bt()` in `src/telemetry_reasoning/parser/compressor.py`. Extend these APIs rather than rebuilding them.

### Shared contracts: use these immediately

All paths in this file are relative to the repository root. Paths marked **create** and their CLI flags are planned deliverables, not commands already implemented. The following files and APIs already exist:

| Contract/source | Required use |
| --- | --- |
| `src/telemetry_reasoning/schemas.py`: `Task`, `Subtask`, `FailureMode`, `TASK_SUBTASKS` | Use the exact enum values. Pickup order: `approach`, `align`, `grasp`, `lift`. Transport order: `maintain_grasp`, `move_along_path`, `reach_target`. |
| `RawTelemetryRow` | Exact CSV columns: `timestamp`, `subtask`, `joint_positions`, `joint_velocities`, `joint_torques`, `gripper_state`, `gripper_force`, `ee_pos`, `ee_quat`, `obj_pos`, `obj_quat`, `obj_vel`, `contact_force`. Vector cells contain JSON arrays. |
| `MetadataRow` | Exact columns: `pair_id`, `task`, `success_file`, `failure_file`, `failed_subtask`, `failure_mode`, `perturbation_param`, `perturbation_value`, `seed`. IDs are strings; filenames resolve relative to the supplied data directory. |
| `PairedBehaviorTree`, `SubtaskBT`, `SubtaskMetadata` | `schema_version="1.0"`; complete ordered task template; semantic `subgoal_name`/`action_description`; nonempty success and failure samples for every subtask. Use `TASK_GOALS` and `SUBTASK_METADATA` for semantics. |
| `DeterministicDivergence`, `TelemetryFeature` | Per-subtask `first_divergence_timestamp` and `max_abs_deltas` keyed by existing feature names. A null divergence object means **not computed**, never zero. |
| `src/telemetry_reasoning/io.py` | Reuse `read_telemetry_csv()`, `write_telemetry_csv()`, `read_metadata_csv()`, `write_metadata_csv()`, and `validate_raw_run()`. |
| `src/telemetry_reasoning/config.py`, `configs/default.yaml` | `ProjectConfig` accepts only `simulation` and `parser`; nested configuration rejects unknown keys. Add track-specific config loaders in the owning modules instead of adding arbitrary sections to the default YAML. |
| `data/sample/success_000.csv`, `data/sample/failure_000.csv`, `data/sample/metadata.csv` | One synthetic pickup pair, ID `"000"`, seed `723`; 400 rows per trace, timestamps 0.00-7.98 s at 50 Hz. The failed grasp drops from 15 N to 3 N at **5.10 s**, then the object does not lift. |
| `data/sample/pair_000_bt.json` | Valid paired BT, already compressed; every divergence is null. Start model formatting/evaluation tests here. It is neither physics output nor evidence of performance across all tasks/modes. |

Keep units consistent: seconds; meters; rad and rad/s for joint positions/velocities; N m for joint torques; N for scalar force magnitudes; m/s for object velocity. Quaternions use **w/x/y/z**, and producers must normalize them and keep signs consistent. `gripper_state` is closure in [0, 1], with 0=open. Joint-vector lengths must remain fixed within and across paired runs, but must not be hard-coded to the mock's six joints.

Raw observations must be 0.02 s apart. Compressed samples keep their original timestamps and may have unequal gaps/counts. V1 requires each task's subtasks exactly once, in order, in both traces. Do not fabricate observations for an aborted or missing stage; quarantine incompatible runs and report them. A future partial-trace schema would need a separate, coordinated change.

Run the existing starting-point checks without overwriting versioned samples:

```bash
python -m pip install -e .
python scripts/generate_mock_pair.py --output-dir /tmp/telemetry-mock --seed 723
python -m pytest tests/test_parser.py
```

Tracks 3-5 can load the existing fixture now:

```python
from pathlib import Path
from telemetry_reasoning.schemas import PairedBehaviorTree

tree = PairedBehaviorTree.model_validate_json(
    Path("data/sample/pair_000_bt.json").read_text()
)
failure_input = tree.to_failure_only()                # Baselines 1 and 2
paired_input = tree.to_paired_without_divergence()     # Baseline 3
full_input = tree.to_full_paired()                    # Main; divergence still null here
```

**Prevent label leakage:** storage metadata is for validation/scoring only. Never pass `model_dump()` of the complete stored tree into a detector or prompt. The view helpers remove `failed_subtask`, `failure_mode`, perturbation details, source filenames, and seed. `pair_id` is an opaque join key, not a predictive feature; omit it from prompt prose. Build additional synthetic task/mode cases within each track's own tests, without editing `data/sample/`.

### Freeze these handoffs to allow parallel development

These are planned interfaces agreed by this task breakdown; each owner implements its side using local fakes until the producer is merged.

1. **Track 1 -> Track 2:** Track 1 provides the completed MuJoCo simulation, nominal controllers, failure injection engine, paired dataset generator, and generated telemetry dataset. `sim/runner.py` exposes `run_episode(*, task, seed, initial_condition_id, control_hook=None) -> (rows, task_succeeded, diagnostics)`. `rows` is `list[RawTelemetryRow]`; `task` is `Task`; `diagnostics` is a JSON-serializable dict with initial-state identity, outcome checks, and timing. The optional hook receives `(timestamp, subtask, nominal_command)` at each 500 Hz control step and returns a modified command. The command dict has `ee_pos` (m), `ee_quat` (w/x/y/z), `gripper_state` (desired closure in [0, 1]), `gripper_force` (N), and `speed_scale` (unitless). These are control targets; measured telemetry is logged separately. Track 2 verifies deterministic replay, identical paired initial states, pre-injection trajectory agreement, perturbation isolation, physical outcomes, and telemetry integrity. The generated dataset is stored under `outputs/dataset/batch_20261030_152125_e5ca72de/`, with separate `pick_up_object/` and `transport_object/` directories containing paired CSVs, `metadata.csv`, `diagnostics.jsonl`, and `summary.json`. The dataset contains 500 accepted pairs per task across 50 initial conditions, balanced across four calibrated failure modes. All five failure mechanisms are implemented, but `excessive_speed` is excluded from the accepted dataset due to insufficient failure reliability.
2. **Track 2 -> Tracks 3/4:** Track 2 provides a validated, parser-ready dataset by consolidating or indexing the existing task-specific `metadata.csv` files and their referenced success/failure telemetry CSVs, plus `manifest.json` for information absent from `MetadataRow`: initial-condition IDs, train/dev/test split assignments, model/config revisions, outcome checks, and injection timestamps. The final handoff must provide a documented dataset root and metadata layout compatible with the existing CSV I/O and parser interfaces, without modifying the shared schemas. Track 2 verifies unique pair IDs, complete ordered subtask sequences, deterministic replay, and split isolation before freezing the dataset. Keep sidecar fields out of the strict CSV schemas. Track 3 uses the validated CSVs for Behavior Tree construction; Track 4 uses the manifest for grouped evaluation splits and optional onset scoring. The dataset contains four calibrated failure modes, with 125 accepted pairs per mode per task; `excessive_speed` remains a valid `FailureMode` but has no accepted dataset examples.
3. **Track 3 -> Tracks 4/5:** `pair_<pair_id>_bt.json` files validated as `PairedBehaviorTree`. After divergence runs, every node has a computed `DeterministicDivergence`; unchanged nodes have zero deltas and a null first-divergence timestamp. A null node-level object remains explicitly uncomputed. Track 5 may populate a deep copy of the sample with hand-checked divergence values for tests while Track 3 works.
4. **Tracks 4/5 -> evaluator:** one JSONL prediction per `(pair_id, model_variant)`, with fields `pair_id`, `model_variant` (`baseline_1`, `baseline_2`, `baseline_3`, `main`), `predicted_failure_mode` (`FailureMode` value or null), `predicted_subtask` (`Subtask` value or null), `explanation` (string), `evidence` (list of `{subtask, timestamp, feature, value}` records using `TelemetryFeature` values), `status` (`ok`, `abstain`, `error`), and `backend` (`rules`, `hf`, `stub`). Evidence refers to observed failure samples, with `value` a finite scalar or vector matching that feature's `RawTelemetryRow` shape; paired/divergence comparisons may also be described in the explanation. Track 4 implements validation in `eval/contracts.py`; Track 5 can emit this documented format before that module lands. Additional runtime/token/config provenance belongs in a sibling run-manifest JSON. Stub outputs are for plumbing tests and must never be reported as LLM results.

## Track 1: MuJoCo Simulation, Failure Injection & Paired Dataset Generation

**Owner / name:** Gili Gordiyenko

**Scope:** implement a physics-based robot simulation with successful waypoint controllers, configurable command perturbations, and a balanced paired success/failure telemetry dataset. Track 2 owns final reproducibility testing, dataset validation, and integration readiness.

### Target milestone dates

- **Oct. 30:** implement the MuJoCo Panda environment, both robotic tasks, and failure injection mechanisms; calibrate perturbations and collect 500 accepted pairs per task.
- **Nov. 1:** complete the initial dataset audit and verify task, failure-mode, and initial-condition coverage.

### Files created/modified

- **Simulation:** `src/telemetry_reasoning/sim/` — MuJoCo environment, robotic controllers, episode runner, telemetry logging, and viewer.
- **Failure injection:** `src/telemetry_reasoning/sim/failures.py`.
- **Dataset generation:** `src/telemetry_reasoning/sim/dataset.py`.
- **Dataset auditing:** `src/telemetry_reasoning/sim/audit_robot_dataset.py`.
- **Calibration:** `data/calibration/pickup_recommended.csv`, `data/calibration/transport_recommended.csv`.
- **Read/reuse:** `src/telemetry_reasoning/schemas.py`, `src/telemetry_reasoning/io.py`, and shared task/subtask definitions.

### Input & output contracts

- Input: `Task`, `TASK_SUBTASKS`, simulation seed, initial-condition ID, and optional failure injection configuration. The runner supports `run_episode(*, task, seed, initial_condition_id, control_hook=None)` for nominal and perturbed execution.
- Output: `RawTelemetryRow` observations with timestamps, subtask labels, joint positions/velocities/torques, gripper state/force, end-effector pose, object pose/velocity, and contact force. Simulation steps at 500 Hz and telemetry is recorded at 50 Hz.
- Dataset: task-specific `success/` and `failure/` directories containing paired telemetry CSVs, with `metadata.csv`, `diagnostics.jsonl`, and `summary.json` for each task.
- Coverage: 50 initial conditions × 10 accepted pairs = 500 pairs per task; 1,000 pairs and 2,000 telemetry CSVs total. Four selected failure modes are balanced at 125 pairs per mode per task.

### Definition of Done

- [x] Set up the Franka Emika Panda robot using MuJoCo Menagerie and implement the simulation environment, episode runner, and control interface.
- [x] Implement bounded waypoint-based controllers for `pick_up_object` (`approach`, `align`, `grasp`, `lift`) and `transport_object` (`maintain_grasp`, `move_along_path`, `reach_target`).
- [x] Step physics at **500 Hz / 0.002 s** and collect telemetry at **50 Hz / 0.02 s**, using the shared `RawTelemetryRow` schema and measured robot/object state.
- [x] Implement 50 seeded initial-condition IDs per task and generate successful nominal executions for paired dataset collection.
- [x] Implement all five `FailureMode` mechanisms: `position_error`, `orientation_error`, `inadequate_grip_strength`, `early_gripper_release`, and `excessive_speed`. Apply perturbations through the control hook rather than modifying recorded telemetry.
- [x] Calibrate failure configurations and evaluate activation and task failure rates. Select four reliable modes for dataset generation; exclude `excessive_speed` after calibration showed insufficient failure reliability.
- [x] Implement paired nominal/perturbed execution generation using matching seeds and initial-condition IDs, with acceptance checks, bounded retries, rejection diagnostics, and metadata recording.
- [x] Generate **500 accepted success/failure pairs per task**, covering 50 initial conditions with 10 accepted pairs per condition.
- [x] Balance the four selected failure modes at 125 accepted pairs per mode per task, producing **1,000 pairs and 2,000 telemetry CSVs**.
- [x] Save telemetry, metadata, episode diagnostics, and generation summaries; complete an initial dataset audit confirming expected file counts, condition coverage, failure-mode balance, and zero reported issues.

Implemented generation command (run from `src/`):

```bash
python -m telemetry_reasoning.sim.dataset \
  --task all \
  --pairs-per-task 500 \
  --seed 723 \
  --calibration ../data/calibration/pickup_recommended.csv ../data/calibration/transport_recommended.csv
```

Generated dataset:

```text
outputs/dataset/batch_20261030_152125_e5ca72de/
├── pick_up_object/
│   ├── success/
│   ├── failure/
│   ├── metadata.csv
│   ├── diagnostics.jsonl
│   └── summary.json
└── transport_object/
    ├── success/
    ├── failure/
    ├── metadata.csv
    ├── diagnostics.jsonl
    └── summary.json
```

## Track 2: Simulation Validation & Dataset Finalization

**Owner / name:** Suhani Chaudhary

**Scope:** validate reproducibility, physical outcomes, telemetry integrity, and parser compatibility of the completed simulation dataset. Establish leakage-free dataset splits, automated checks, and a frozen dataset manifest for downstream Behavior Tree conversion and evaluation.

### Target milestone dates

- **Nov. 6:** validate deterministic replay, physical outcomes, telemetry quality, and parser compatibility; identify and resolve invalid pairs.
- **Nov. 13:** finalize dataset splits, automated validation, provenance, and checksum inventory for downstream integration.

### Files to create/modify

- **Extend:** `src/telemetry_reasoning/sim/audit_robot_dataset.py` for final dataset validation.
- **Create:** reproducibility and failure-injection tests under `tests/`, using the existing repository test conventions.
- **Create:** dataset `manifest.json` and checksum inventory.
- **Read/reuse:** `src/telemetry_reasoning/sim/runner.py`, `src/telemetry_reasoning/sim/failures.py`, `src/telemetry_reasoning/sim/dataset.py`, `src/telemetry_reasoning/parser/compressor.py`, and shared schemas/I/O.

### Input & output contracts

- Input: the completed dataset under `outputs/dataset/batch_20261030_152125_e5ca72de/`, including paired telemetry CSVs, task-specific metadata, diagnostics, calibration configurations, and initial-condition IDs.
- Output: validated paired telemetry, deterministic train/dev/test assignments, reproducibility and compatibility results, `manifest.json`, and file checksums.
- Keep the existing `RawTelemetryRow` and `MetadataRow` schemas unchanged. Store split assignments, simulation provenance, and additional validation information in the manifest. Accepted traces must satisfy the parser's complete ordered subtask requirements without fabricating missing observations.

### Definition of Done

- [ ] Verify deterministic replay, identical paired initial states, pre-injection trajectory agreement, and perturbation isolation within documented numerical tolerances.
- [ ] Validate physical task outcomes and telemetry integrity, including nominal success across all 50 initial conditions per task, logging cadence, units, quaternions, and measured force sources.
- [ ] Check all accepted pairs for complete ordered subtask sequences and compatibility with `build_paired_bt()`; identify, quarantine, or replace incompatible traces.
- [ ] Create deterministic train/dev/test splits grouped by initial condition (35/5/10 per task), keeping paired and repeated executions together and validating task/failure-mode coverage.
- [ ] Add automated tests for simulation resets, control hooks, failure injection boundaries, deterministic replay, retries, and dataset validation; document simulation/configuration provenance and the exclusion of `excessive_speed`.
- [ ] Generate the final dataset manifest and checksum inventory, verify metadata integrity and split isolation, and freeze the validated dataset for downstream processing.

### Final deliverables

- Reproducibility, physical outcome, and telemetry validation results.
- Parser-compatible paired telemetry dataset.
- Leakage-free train/dev/test assignments.
- Automated simulation and dataset validation tests.
- Dataset manifest, simulation provenance, and checksum inventory.
- Frozen dataset ready for telemetry compression, Behavior Tree construction, and model evaluation.

## Track 3: Behavior Tree Builder & Deterministic Divergence Function

**Owner / name:** Arik Gershman

**Scope:** extend the existing parser into a batch BT pipeline, tune compression, and calculate reproducible numerical divergence. The proposal does not prescribe a formula; the alignment and metric definitions below are explicit implementation decisions to document and test.

### Target milestone dates

- [x] **Oct. 30:** run the existing parser on the sample, define batch/alignment APIs, and add synthetic transport and timing edge cases. Verification and public interfaces are recorded below.
- [ ] **Nov. 6 — implementation complete; real-data tuning pending:** batch parsing, diagnostics, tuning tools, public API preservation, and sample reproducibility are verified. Calibrating `k` and major-event thresholds still requires Track 2's validated grouped train/dev manifest.
- [ ] **Nov. 13:** convert the full accepted dataset to BT JSONs with computed divergence and publish compression/event-retention diagnostics.

### Oct. 30 completion evidence and API handoff

- [x] Rebuilt the checked-in pickup pair with the existing parser and reproduced all four sample files byte for byte in `data/processed/track3-oct30/mock/`. Each raw trace has 400 rows; the BT retains 31 success and 34 failure samples, including the 5.08/5.10/5.12 s grip-drop neighborhood. Shared schemas, default configuration, and versioned samples are unchanged.
- [x] Added `build_paired_bts(metadata, data_dir, config) -> BatchBuildResult` in `parser/pipeline.py`. It accepts `Sequence[MetadataRow]` and `CompressionConfig`, reuses `build_paired_bt()`, and returns ordered `trees` and `errors` tuples. Each input pair produces exactly one tree or a `PairBuildError` containing its ID, source filenames, exception type, and message. Duplicate IDs reject the batch; an empty sequence returns empty tuples. File/validation failures do not prevent later pairs from being processed. Metadata CSV errors still raise from `read_metadata_csv()` before batch construction.
- [x] Added `align_subtask_samples(node: SubtaskBT) -> AlignedSubtask` in `parser/divergence.py`. It returns the shared progress grid, failure-run absolute timestamps, and success/failure arrays for every `TelemetryFeature`; scalar shapes are `(n,)`, vector shapes `(n, components)`. Interpolation is linear for continuous features, previous-value for gripper closure, and normalized/sign-aligned shortest-path SLERP for quaternions. A singleton stays constant over its partner's grid; two singletons compare once at progress zero. Queries within `1e-12` progress of an observed sample snap to that sample to avoid roundoff-induced transitions; the grid remains the unrounded sorted union. Inputs are not mutated, and interpolated values are comparisons rather than new observations.
- [x] Added synthetic tests for both task templates, transport semantics, seven-joint arrays, unequal stage lengths/sample counts, shifted timestamps, singleton stages, gripper transitions, quaternion sign/shortest-path behavior, malformed or missing files, and missing/repeated/reordered stages on either side. The additional completion audit on Oct. 23 passes all **97 tests with warnings treated as errors**, including the original parser/reproducibility tests, 100 seeded analytic-motion cases, raw-validation failures, and batch error ordering/preflight checks.
- [x] Completed an independent mixed-task stress check: 100 synthetic pairs (50 per task), one through nine joints, unequal stage lengths, and `k=0/1/5/100`, for 400 successful pair builds and 1,400 aligned subtasks. All trees round-trip through the schema, all aligned values are finite, and input files remain unchanged. Fresh generation reproduces all four sample files byte for byte, and the documented example passes verbatim. Fixed a small-angle quaternion approximation exposed by this audit using stable SLERP weights; differently sampled constant-angular-speed motion now agrees within numerical tolerance. Evidence is in `runs/track3-oct30/verification.json`; these synthetic parameter checks are validation, not train/dev tuning.

Both APIs and their result types are exported from `telemetry_reasoning.parser`. At the Oct. 30 handoff, batch construction was in-memory; the Nov. 6 work below adds serialization, reporting, and tuning tools. Divergence remains null pending metric implementation. Alignment removes pure duration differences; duration is not a v1 divergence feature. Its mapped failure timestamps can be interpolated times and must not be presented as exact observed onset.

Reproduce the Oct. 30 verification from the repository root (after installing the project into `.venv`):

```bash
PYTHONPATH=src .venv/bin/python -W error -m pytest
PYTHONPATH=src .venv/bin/python scripts/generate_mock_pair.py --output-dir data/processed/track3-oct30/mock --seed 723
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from telemetry_reasoning.config import load_config
from telemetry_reasoning.io import read_metadata_csv
from telemetry_reasoning.parser import align_subtask_samples, build_paired_bts
from telemetry_reasoning.schemas import PairedBehaviorTree

sample = Path("data/sample")
config = load_config("configs/default.yaml")
result = build_paired_bts(read_metadata_csv(sample / "metadata.csv"), sample, config.parser)
assert not result.errors
tree, = result.trees
assert tree == PairedBehaviorTree.model_validate_json((sample / "pair_000_bt.json").read_text())
for name in ("success_000.csv", "failure_000.csv", "metadata.csv", "pair_000_bt.json"):
    assert (sample / name).read_bytes() == (Path("data/processed/track3-oct30/mock") / name).read_bytes()
for node in tree.subtasks:
    aligned = align_subtask_samples(node)
    assert node.deterministic_divergence is None
    print(node.subtask.value, len(node.success_samples), len(node.failure_samples), len(aligned.progress))
PY
```

Expected columns are subtask, retained success count, retained failure count, and alignment grid count: `approach 7 7 7`, `align 7 7 7`, `grasp 10 13 13`, `lift 7 7 7`.

### Nov. 6 completion evidence and remaining dependencies

- [x] Implemented exported `parse_dataset()` and `scripts/parse_dataset.py`, producing schema-valid `pair_<pair_id>_bt.json` files and a separate `parse_report.json`. Reports record per-pair rejections, source/output/configuration hashes, and compression diagnostics. Staged publication and fresh-output-directory checks prevent partial or stale batches; rejected source files are left untouched.
- [x] Added complete `configs/parser.yaml`, preserving the existing defaults and byte-identical sample output. All divergence objects remain explicitly uncomputed. Settings are provisional, not calibrated on the real dataset.
- [x] Added raw-reference compression diagnostics: per-stage counts, compression ratios, fixed-reference event-center/neighborhood retention, reconstruction errors for every feature, and leakage-free JSON payload sizes in characters/UTF-8 bytes (not model tokens).
- [x] Implemented `scripts/tune_compression.py` and grouped-split validation. The sweep checks all manifest groups before opening telemetry, reads only the requested train/dev split, keeps reference-event thresholds fixed across candidates, and never changes configuration automatically. A reproducible synthetic demonstration covers both tasks, three seeds, `k=[0,2,5]`, and threshold scales `[0.5,1,2]`; this verifies the tools, not real-data calibration.
- [x] Ran the parser against all 1,000 pairs in Gili's supplied `dataset_v1.zip`: pickup **384 written / 116 rejected**, transport **226 written / 274 rejected**. All 390 rejections are incomplete failure-stage sequences. The 610 compatible pairs retain 110,283 of 448,309 raw observations and all 71,708 fixed-reference feature-event neighborhoods. All output BTs validate; archive/source hashes are unchanged. Task outputs are separate because task-local pair IDs collide.
- [x] Verified existing Track 4 baseline consumption and all **152 Track 3 tests** with warnings treated as errors. The full suite has **279 passed / 1 pre-existing Track 4 failure**: `test_label_vocabularies_match_shared_schemas`, because the evaluator vocabulary omits the shared `excessive_speed` mode. Other tracks' code is unchanged.
- [ ] Calibrate and select production compression settings using Track 2's validated train/dev splits. The archive has no split manifest; the compatible subset is not balanced and contains no transport `inadequate_grip_strength` examples. Track 2 must resolve partial traces and freeze grouped splits before real tuning/final conversion.

Handoff and commands: `src/telemetry_reasoning/parser/README.md`. Local generated evidence: `runs/track3-nov06/verification.json`, `runs/track3-nov06/archive-full-compatibility/verification.json`, and `runs/track3-nov06/synthetic-tuning/SYNTHETIC-evidence.json` (ignored outputs, not versioned fixtures).

### Files to create/modify

- **Modify (exclusive ownership):** `src/telemetry_reasoning/parser/compressor.py`, `src/telemetry_reasoning/parser/__init__.py`, `tests/test_parser.py`.
- **Implemented; extend for later milestones:** `src/telemetry_reasoning/parser/divergence.py`, `src/telemetry_reasoning/parser/pipeline.py`.
- **Implemented:** `configs/parser.yaml`, `scripts/parse_dataset.py`, `scripts/tune_compression.py`, `src/telemetry_reasoning/parser/diagnostics.py`, `src/telemetry_reasoning/parser/tuning.py`, `tests/test_compression_diagnostics.py`, `tests/test_compression_tuning.py`, and the parser handoff README.
- **Still to create for Nov. 13:** `configs/divergence.yaml`.
- **Implemented; extend for later milestones:** `tests/test_bt_pipeline.py`, `tests/test_divergence.py` (currently alignment tests; metric tests remain pending).
- Keep `configs/default.yaml` and the checked-in sample unchanged. `configs/parser.yaml` is a complete `ProjectConfig` (`simulation` plus `parser`); load the separate divergence settings in `parser/divergence.py`.

### Input & output contracts

- Start with `read_metadata_csv("data/sample/metadata.csv")`, the paired sample CSVs, `CompressionConfig`, and existing `build_paired_bt(metadata, data_dir, config)`.
- Output: one schema-valid `PairedBehaviorTree` per input pair under `data/processed/<dataset>/bt/`, plus a separate parse/compression report. Preserve the existing nested goal/subgoal/action semantics; do not introduce an incompatible tree shape.
- Compute divergence from each node's `success_samples` and `failure_samples`; do not zip arrays by index. Fill only the existing `DeterministicDivergence` fields. Keep alignment settings and diagnostics in the report, not extra schema fields.
- Raw 50 Hz traces are the reference for evaluating information lost through compression. The LLM inputs remain the compressed BTs and their computed comparisons.

### Definition of Done

- [x] Extend in-memory batch construction for both task templates with clear per-pair errors. Preserve original timestamps, semantic metadata, structural alignment, and all existing CSV/schema validation. Keep the existing `build_paired_bt()` default behavior (uncomputed divergence). Test unequal durations/counts, missing/repeated/reordered subtasks, and invalid pair files.
- [x] Finish dataset serialization, parse/compression reporting, parser configuration, and the CLI. Preserve the existing API and sample reproducibility; explicitly report uncomputed divergence.
- [ ] Add divergence explicitly in the new pipeline once metric computation and the separate divergence configuration are implemented (Nov. 13).
- [x] Retain first/last rows, up to `k` interior samples, and major-event neighbors `t-1/t/t+1`; compare adjacent rows only within a consecutive subtask. Preserve strict `>` threshold semantics, sorted deduplication, clipping, and `k=0` behavior. Existing quaternion compression uses component deltas; any replacement needs explicit regression evidence. Verified by the existing parser regression suite on Oct. 23.
- [ ] Sweep a documented small `k` grid and per-feature event thresholds using validated real train/dev data only. Measure compression ratio, event retention against raw traces, and resulting prompt size. Preserve the mock's 5.08/5.10/5.12 s grip-drop neighborhood; do not tune on test labels or overwrite the mock defaults. The guarded sweep implementation and synthetic development-data demonstration are complete; real calibration awaits Track 2's manifest.
- [x] Implement deterministic **within-subtask progress alignment**: map each trace's timestamps to `u=(t-t_start)/(t_end-t_start)` and compare on the sorted union of retained progress values. Linearly interpolate continuous scalar/vector features, use previous-value interpolation for gripper closure, and normalize/sign-align quaternions with shortest-path interpolation. Define a deterministic singleton-segment rule and test it. Never interpolate across subtask boundaries. Report that this alignment removes pure duration differences, so duration itself is not a v1 divergence feature. See the API handoff above for singleton and numerical-tolerance rules.
- [ ] Define `delta_f(u)=max_j(abs(failure_f,j(u)-success_f,j(u)))` (one component for scalars), then `max_abs_deltas[f]=max_u(delta_f(u))` for all `TelemetryFeature` values. Quaternion deltas are dimensionless component differences after sign alignment, not radians. First divergence is the earliest grid point exceeding a configured per-feature threshold, mapped back to the **failure run's absolute timestamp**. Separate these paired-trace thresholds from compression's adjacent-row thresholds. Store a computed object with null first timestamp and zero deltas for identical traces.
- [ ] Test known scalar/vector offsets, threshold equality, identical traces, shifted/differently sampled traces, quaternion `q` versus `-q`, and deterministic serialization. For the sample and a 2 N grip threshold, verify the first grip deviation at **5.10 s** and maximum grip delta **12 N**. Compare compressed-onset estimates against raw references and report their error; do not claim unobserved exact physical onset.

Current Nov. 6 verification commands (use a fresh/empty output directory):

```bash
PYTHONPATH=src .venv/bin/python -W error -m pytest tests/test_parser.py tests/test_bt_pipeline.py tests/test_divergence.py tests/test_compression_diagnostics.py tests/test_compression_tuning.py
PYTHONPATH=src .venv/bin/python scripts/parse_dataset.py --metadata data/sample/metadata.csv --data-dir data/sample --config configs/parser.yaml --output-dir data/processed/track3-nov06/sample/bt
```

Later Nov. 13 acceptance commands (computed divergence, its configuration/CLI flag, and the final validated dataset are still pending):

```bash
python -m pytest tests/test_parser.py tests/test_bt_pipeline.py tests/test_divergence.py
python scripts/parse_dataset.py --metadata data/sample/metadata.csv --data-dir data/sample --config configs/parser.yaml --divergence-config configs/divergence.yaml --output-dir data/processed/smoke/bt
python scripts/parse_dataset.py --metadata data/raw/dataset/metadata.csv --data-dir data/raw/dataset --config configs/parser.yaml --divergence-config configs/divergence.yaml --output-dir data/processed/dataset/bt
```

The full-dataset command uses a provisional `data/raw/dataset/` path. Update `--metadata` and `--data-dir` to the validated dataset location established by Track 2 before running it. The current generated dataset is under `outputs/dataset/batch_20261030_152125_e5ca72de/`, with separate task-specific metadata files.


## Track 4: Rule-Based Baseline (Baseline 1) & Evaluation Pipeline

**Owner / name:** David Simberg

**Scope:** a failure-only threshold/template baseline, shared result validation, evaluation, and Matplotlib plots. This owner assembles the final smoke test using the other tracks' public CLIs; model owners remain responsible for their own tests.

### Target milestone dates

- **Oct. 30:** implement the sample-driven detector skeleton, prediction contract, synthetic evaluation cases, and one telemetry plot.
- **Nov. 6:** complete Baseline 1, automated metrics, and deterministic failure-mode/subtask scoring for all variants.
- **Nov. 13:** integrate real BT/divergence outputs, all four model variants, visualization, and end-to-end smoke checks. The final held-out study remains Dec. 1.

### Files to create/modify

- **Create:** `src/telemetry_reasoning/models/rule_based.py`.
- **Create:** `src/telemetry_reasoning/eval/contracts.py`, `src/telemetry_reasoning/eval/metrics.py`, `src/telemetry_reasoning/eval/plots.py`.
- **Create:** `configs/rules.yaml`, `configs/eval.yaml`.
- **Create:** `scripts/run_baseline.py`, `scripts/evaluate.py`, `scripts/plot_telemetry.py`, `scripts/integration_smoke.py`.
- **Create:** `tests/test_rule_based.py`, `tests/test_evaluation.py`, `tests/test_plots.py`, `tests/test_integration_smoke.py`.
- Keep shared `models/__init__.py` and `eval/__init__.py` unchanged; use direct module imports.

### Input & output contracts

- Detector input: **only** `tree.to_failure_only()`; the CLI may load storage to obtain that view, but its predictor receives neither success samples nor divergence nor ground-truth metadata. Use compressed failure telemetry consistently with Baseline 2.
- Initial fixture: `data/sample/pair_000_bt.json`. The rule should find low grip during `grasp` and supporting failed lift evidence without reading its answer label. Add local synthetic coverage for both tasks/all modes.
- Output: `baseline_1.jsonl` using the shared prediction contract, metrics CSV/JSON and PNG plots under `runs/<experiment>/`. Evaluator ground truth comes separately from `MetadataRow`; split/onset provenance comes from Track 2's manifest.
- These metrics are proposed operational choices, not measures or thresholds already fixed by the proposal. One mock pair verifies plumbing only and is never a scientific accuracy estimate.

### Definition of Done

- [x] Build subtask-aware rules for position offset, orientation mismatch, low grip, premature opening, and excessive speed, using only observable failure telemetry and declared task constants. Use object/EE relative geometry where possible; identify absent target/desired-pose information explicitly and abstain when evidence cannot distinguish a mode. Do not substitute label-derived target values. The released real dataset covers four calibrated failure modes; excessive_speed remains supported by the rule interface but has no accepted dataset examples.
- [x] Define precedence for competing rules and `abstain` for ambiguous/unsupported cases. Generate a concise causal template identifying predicted mode/stage and measured evidence with timestamps/units. Test all five modes, competing symptoms, benign approach/open-gripper behavior, and no-evidence input.
- [x] Implement the shared JSONL validator and strict `(pair_id, model_variant)` joins. Reject duplicates/unknown IDs; account for missing/invalid outputs instead of silently dropping them. Test empty inputs and classes absent from a small evaluation set.
- [x] Report failure-mode accuracy and macro-F1 across the four represented dataset classes, with excessive_speed reported as an unsupported class rather than included in the observed-class macro average.  Count abstentions/missing predictions as incorrect for accuracy and as missed true classes for recall. Compute optional onset error only where a defined prediction timestamp and independent reference exist; injection time and observed divergence are different events.
- [ ] Verify cited evidence exists in the allowed input and agrees numerically within tolerance; report unsupported evidence separately. Log runtime/token counts when available. Prepare a small blinded human rubric for causal correctness, evidence support, and clarity; automated label accuracy alone does not measure explanation quality. Stub-backend metrics must be prominently labeled synthetic.
- [ ] Plot raw success/failure telemetry, retained samples, and computed divergence with subtask boundaries, units, legends, and distinct injection/onset markers. Test headless Matplotlib output and plotted data/axes. Assemble a CPU-only smoke test now with mock predictions, then wire real public CLIs after merge; keep a separate optional real-MuJoCo/HF check.

Implement and pass (the first command is independent; the remaining commands consume Track 3's output):

```bash
python -m pytest tests/test_rule_based.py tests/test_evaluation.py tests/test_plots.py
python scripts/run_baseline.py --bt-dir data/processed/smoke/bt --config configs/rules.yaml --output runs/smoke/baseline_1.jsonl
python scripts/evaluate.py --metadata data/sample/metadata.csv --predictions runs/smoke/baseline_1.jsonl --config configs/eval.yaml --output-dir runs/smoke/eval
python scripts/plot_telemetry.py --data-dir data/sample --bt-dir data/processed/smoke/bt --output-dir runs/smoke/plots
```

## Track 5: LLM Reasoning Harness & Prompt Baselines (Baselines 2, 3, and Main Model)

**Owner / name:** ____________________

**Scope:** one reproducible Hugging Face Llama-3.1-8B inference harness, three controlled prompt variants, and future training-data formatting. DPO/LoRA training is deferred to the later proposal milestone.

### Target milestone dates

- **Oct. 30:** create the model adapter, mock inference backend, prompt views, and a compute/access smoke plan.
- **Nov. 6:** run Baseline 2 (failure-only BT) and Baseline 3 (paired BT without divergence) with the same model/settings; finish output parsing and leakage tests.
- **Nov. 13:** run the main paired-BT-plus-divergence variant, deliver comparable four-variant evaluation inputs with Track 4, and validate the DPO/LoRA data formatter.

### Files to create/modify

- **Create:** `src/telemetry_reasoning/models/llm.py`, `src/telemetry_reasoning/models/prompts.py`, `src/telemetry_reasoning/models/training_data.py`.
- **Create:** `configs/llm.yaml`, `configs/prompts/failure_only.txt`, `configs/prompts/paired_no_divergence.txt`, `configs/prompts/paired_with_divergence.txt`.
- **Create:** `scripts/run_llm.py`, `scripts/prepare_dpo_data.py`.
- **Create:** `tests/test_llm.py`, `tests/test_prompts.py`, `tests/test_training_data.py`.
- Read/reuse the shared view methods and prediction contract. Do not modify `models/rule_based.py`, evaluator files, or shared schema/default configuration.

### Input & output contracts

- Start from `PairedBehaviorTree.model_validate_json()` on `data/sample/pair_000_bt.json`. Baseline 2 uses `to_failure_only()`; Baseline 3 uses `to_paired_without_divergence()`; main uses `to_full_paired()` after checking that divergence has been computed.
- For independent main-model tests, deep-copy the fixture and attach explicitly synthetic `DeterministicDivergence` objects, including the 5.10 s / 12 N grasp example and computed-zero unchanged stages. Leave the source sample untouched. Production main-model inference must reject null divergence objects rather than treating them as zero.
- Output: `baseline_2.jsonl`, `baseline_3.jsonl`, and `main.jsonl` matching Track 4's prediction contract, plus a run manifest with model/tokenizer revision, prompt/config hashes, decoding settings, seed, latency, and token counts.
- DPO input: curated JSONL records containing `pair_id`, `model_variant`, `chosen`, `rejected`, and preference provenance. Formatter joins each record to its leakage-free BT view and emits `prompt`, `chosen`, `rejected`; keep split/provenance in a sidecar. Synthetic preferences may test the formatter but must not be presented as reviewed training labels.

### Definition of Done

- [ ] Configure an exact Hugging Face Llama-3.1-8B checkpoint and revision, tokenizer, device/dtype, context budget, and optional supported quantization. Record whether the base or instruction-tuned checkpoint is used and use the same choice for all variants. Check access/compute early; provide a stub backend so CPU unit tests never download weights or require credentials.
- [ ] Implement consistent tokenization, chat formatting where applicable, batched inference, seeded decoding, output limits, and validated structured predictions. Handle overlong prompts explicitly and consistently; do not silently truncate task stages or divergence evidence. Capture invalid outputs as errors without inventing labels.
- [ ] Design the requested Chain-of-Thought-style prompts around examining task expectations, observable deviations, and plausible causes; request a final diagnosis with a concise evidence-grounded rationale and the shared JSON fields. Keep the instruction/output format and decoding settings fixed across variants so the experimental difference is available input information.
- [ ] Test prompt payloads recursively: Baseline 2 contains no success/divergence; Baseline 3 contains success but no divergence; main contains both. No variant sees ground-truth metadata or identifies a cause from filenames/IDs. Verify null-divergence rejection, known fixture formatting, malformed model responses, and no mutation of storage objects.
- [ ] Demonstrate real HF inference for both prompt baselines by Nov. 6 and all three variants by Nov. 13; save reproducible run manifests and validated JSONL. Unit tests may use stubs, but real-inference acceptance cannot be satisfied by a stub. Describe causal evidence rather than attributing every later symptom to a separate injected fault.
- [ ] Implement the training formatter with nonempty/different chosen/rejected validation, matching variant/pair IDs, reviewed-preference provenance, and initial-condition split preservation. Use identical input views for inference and future training. Emit a LoRA/DPO-ready dataset and configuration notes; leave optimizer/training runs and adapter tuning for Nov. 27.

Implement and pass (stub execution can use a local synthetic enriched fixture until Track 3 merges):

```bash
python -m pytest tests/test_llm.py tests/test_prompts.py tests/test_training_data.py
python scripts/run_llm.py --bt-dir data/processed/smoke/bt --config configs/llm.yaml --variants baseline_2 baseline_3 main --backend stub --output-dir runs/smoke
python -m pip install -e '.[llm]'
python scripts/run_llm.py --bt-dir data/processed/smoke/bt --config configs/llm.yaml --variants baseline_2 baseline_3 main --backend hf --output-dir runs/hf-smoke
python scripts/prepare_dpo_data.py --bt-dir data/processed/dataset/bt --preferences data/processed/preferences/reviewed.jsonl --manifest data/raw/dataset/manifest.json --output-dir data/processed/dpo
```
The `--manifest data/raw/dataset/manifest.json` path is provisional. Update it to the final manifest produced by Track 2 before preparing training data.


The final command requires a separately prepared reviewed-preference file. Until those annotations exist, test formatting with temporary synthetic records and mark the real preference dataset as pending.

## Git workflow and integration checklist

### Branches and ownership

| Track | Branch |
| --- | --- |
| 1 | `codex/track-1-mujoco-controllers` |
| 2 | `codex/track-2-failure-dataset` |
| 3 | `codex/track-3-bt-divergence` |
| 4 | `codex/track-4-baseline-evaluation` |
| 5 | `codex/track-5-llm-prompts` |

- [ ] Claim owner lines together in one small planning commit before branching; then change only the paths assigned to your track. Keep progress in your PR to avoid five simultaneous edits to this file.
- [ ] Treat `schemas.py`, `io.py`, `config.py`, `configs/default.yaml`, `scripts/generate_mock_pair.py`, and `data/sample/` as frozen shared contracts. Do not edit shared package initializers except Track 3's explicitly assigned parser initializer. Keep extra fixtures in the owning test modules or their temporary directories, not a shared `tests/conftest.py`.
- [ ] If a shared contract/dependency/README change is essential, coordinate a small separate PR with affected consumers and merge it once; do not make competing copies on five branches. Existing `.[sim]` and `.[llm]` extras already provide the starting optional dependencies.
- [ ] Keep generated data in ignored `data/raw/` and `data/processed/`, reports in ignored `runs/`, and weights outside tracked source. Include lightweight test fixtures, config/provenance, and exact verification commands in each PR.
- [ ] Rebase before merging and run the existing `tests/test_parser.py` plus your track's tests. File ownership permits implementation PRs in any order; execute cross-track tests only after their producers are available. Each PR needs another member's review of its contract and acceptance evidence.

### End-to-end integration after Tracks 1-5 merge

Track 4 owns `scripts/integration_smoke.py` and `tests/test_integration_smoke.py`; each owner fixes defects within their own modules.

- [ ] **CPU contract smoke:** generate a fresh mock into the requested output directory, validate its CSVs, build BTs, compute divergence, run Baseline 1 plus stubbed Baselines 2/3/main, validate predictions, score them, and render plots. Use explicit file lists so earlier runs/manifests cannot pollute evaluation. Assert the known grasp event (5.10 s, 12 N), all three view exclusions, one result per variant/pair, finite metric outputs, and nonempty plot artifacts.
- [ ] **Real producer smoke:** run both tasks through MuJoCo with a small deterministic seed set and injections covering the four calibrated dataset failure modes across eligible stages; require verified success/failure outcomes, complete schema-valid traces, and replay agreement. Separately verify excessive_speed injection activation without requiring it to produce a failed execution. Feed the newly generated accepted CSVs through the same parser and baseline stages.
- [ ] **Real model smoke:** on a machine with model access/compute, run the three HF variants on the same parsed real pairs and the rule baseline on their failure-only views. Validate allowed inputs and JSONL outputs before evaluating. Keep real and stub results in separate run directories.
- [ ] **Full dataset gate:** validate 500 pairs per task / 50 conditions per task, with 125 accepted pairs per mode per task across four calibrated failure modes. Then produce 1,000 validated paired BTs with computed divergence. Freeze train/dev/test assignments, prompt/config versions, and checksums; reserve held-out test results for later final evaluation.
- [ ] **Integration evidence:** attach the smoke commands, dataset counts, parser/compression diagnostics, four-variant result validation, plot examples, and any remaining hardware-dependent checks to the integration PR. Never mark real-MuJoCo or HF checks passed because a stub test passed.

Planned integration commands:

```bash
python -m pytest
python scripts/integration_smoke.py --backend stub --output-dir runs/integration-cpu
python -m pytest tests/test_integration_smoke.py
python scripts/integration_smoke.py --sim-backend mujoco --backend hf --tasks all --cover-dataset-failure-modes --output-dir runs/integration-real
python scripts/evaluate.py --metadata data/sample/metadata.csv --predictions runs/smoke/baseline_1.jsonl runs/smoke/baseline_2.jsonl runs/smoke/baseline_3.jsonl runs/smoke/main.jsonl --config configs/eval.yaml --output-dir runs/smoke/comparison
```

Default `python -m pytest` must remain CPU/offline compatible: skip explicitly marked hardware/model tests when their resources are absent and report those skips. Passing that command is necessary, but the separate real producer/model smoke evidence is also required for the corresponding track to be complete.
