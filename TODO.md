# Project TODO: five parallel workstreams

Each member claims **one track** by filling in its owner field. Each track owns implementation, configuration, tests, and a runnable demonstration. The five tracks are intended to have comparable scope; GPU training and the final study are not additional requirements for any one owner during this period.

## Milestones and current starting point

| Target | Proposal milestone | Track deliverables |
| --- | --- | --- |
| **Oct. 30** | Generate, run, and store MuJoCo data | Track 1: nominal controllers and 50 initial conditions per task. Track 2: 500 valid success/failure pairs per task. Tracks 3-5: work immediately against the existing mock contracts. |
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

1. **Track 1 -> Track 2:** `sim/runner.py` exposes `run_episode(*, task, seed, initial_condition_id, control_hook=None) -> (rows, task_succeeded, diagnostics)`. `rows` is `list[RawTelemetryRow]`; `task` is `Task`; `diagnostics` is a JSON-serializable dict with initial-state identity, outcome checks, and timing. The optional hook receives `(timestamp, subtask, nominal_command)` at each 500 Hz control step and returns a modified command. The command dict has `ee_pos` (m), `ee_quat` (w/x/y/z), `gripper_state` (desired closure in [0, 1]), `gripper_force` (N), and `speed_scale` (unitless). These are control targets; log measured telemetry separately. Pass a copy, apply the returned command before stepping physics, and keep the no-hook path identical to nominal execution. Track 2 tests a fake runner with this signature; Track 1 tests identity and simple test hooks without importing Track 2.
2. **Track 2 -> Tracks 3/4:** a data directory containing `metadata.csv` and its referenced CSVs, plus `manifest.json` for information absent from `MetadataRow`: initial-condition IDs, split assignments, model/config revisions, outcome checks, and injection timestamps. Keep sidecar fields out of the strict CSV schemas. Track 3 needs only the existing CSV contract; Track 4 uses the sidecar for grouped splits and optional onset scoring.
3. **Track 3 -> Tracks 4/5:** `pair_<pair_id>_bt.json` files validated as `PairedBehaviorTree`. After divergence runs, every node has a computed `DeterministicDivergence`; unchanged nodes have zero deltas and a null first-divergence timestamp. A null node-level object remains explicitly uncomputed. Track 5 may populate a deep copy of the sample with hand-checked divergence values for tests while Track 3 works.
4. **Tracks 4/5 -> evaluator:** one JSONL prediction per `(pair_id, model_variant)`, with fields `pair_id`, `model_variant` (`baseline_1`, `baseline_2`, `baseline_3`, `main`), `predicted_failure_mode` (`FailureMode` value or null), `predicted_subtask` (`Subtask` value or null), `explanation` (string), `evidence` (list of `{subtask, timestamp, feature, value}` records using `TelemetryFeature` values), `status` (`ok`, `abstain`, `error`), and `backend` (`rules`, `hf`, `stub`). Evidence refers to observed failure samples, with `value` a finite scalar or vector matching that feature's `RawTelemetryRow` shape; paired/divergence comparisons may also be described in the explanation. Track 4 implements validation in `eval/contracts.py`; Track 5 can emit this documented format before that module lands. Additional runtime/token/config provenance belongs in a sibling run-manifest JSON. Stub outputs are for plumbing tests and must never be reported as LLM results.

## Track 1: MuJoCo Environment & Nominal Task Controllers

**Owner / name:** ____________________

**Scope:** a reproducible robot environment, trustworthy telemetry, and successful waypoint state machines. Track 2 owns actual perturbations and paired batch collection.

### Target milestone dates

- **Oct. 30:** load a pinned MuJoCo Menagerie arm/gripper, implement both tasks, and demonstrate success over 50 initial conditions per task; deliver the runner/hook interface for collection.
- **Nov. 6:** validate replay, physical outcome checks, and logger calibration using the collected data; resolve simulator-side defects found by other tracks.
- **Nov. 13:** supply a stable headless smoke run and final simulation/config provenance for integration.

### Files to create/modify

- **Create:** `src/telemetry_reasoning/sim/environment.py`, `src/telemetry_reasoning/sim/controllers.py`, `src/telemetry_reasoning/sim/logger.py`, `src/telemetry_reasoning/sim/runner.py`.
- **Create:** `configs/sim.yaml`, `scripts/run_nominal.py`.
- **Create:** `tests/test_sim_environment.py`, `tests/test_nominal_controllers.py`, `tests/test_sim_logger.py`.
- **Read/reuse:** `schemas.py`, `io.py`, `config.py`, and `configs/default.yaml`. Keep shared `sim/__init__.py` unchanged; consumers import concrete modules.

### Input & output contracts

- Input: `Task`, `TASK_SUBTASKS`, seed, initial-condition ID, and the optional control hook above. `configs/sim.yaml` has its own validated loader in `sim/environment.py`; reuse `SimulationConfig` for timing and add robot/scene/controller settings there.
- Output: validated `RawTelemetryRow` lists and CSVs, truthful task-success booleans, and diagnostic JSON under `data/raw/nominal/`. Use the sample CSVs to check encoding and units; do not copy their fabricated dynamics.
- `transport_object` starts in a verified held-object state so it can run independently of the pickup dataset. Define 50 reproducible, physically feasible initial conditions per task and document pose/object/target variation.

### Definition of Done

- [ ] Pin the Menagerie model revision and MuJoCo/runtime settings; document model retrieval in the CLI help and config. Resolve assets through a configured local path and record their revision rather than copying a large asset repository into Git.
- [ ] Step at **500 Hz / 0.002 s**; log every tenth step at **50 Hz / 0.02 s** using integer step counts. Run within the proposal's 6-10 s horizon and document an end-exclusive logging convention (300-500 rows). Test exact cadence, no duplicate timestamps, units, quaternion normalization, joint-array consistency, and measured force sources.
- [ ] Implement bounded waypoint state machines in the exact enum order. Verify pickup by object height and sustained grasp; verify transport by target tolerance and retained grasp. Thresholds belong in `configs/sim.yaml`, not in ground-truth labels or parser logic.
- [ ] Log the subtask being commanded, even if a preceding physical action failed. Attempt subsequent bounded stages when physically valid; never equate a state-machine transition with task success. Flag unusable aborted traces for Track 2 instead of padding them.
- [ ] Test reset reproducibility and the control hook with a fake physics backend. Test that a hook cannot mutate cached nominal commands or leak state into the next episode. A headless real-MuJoCo run verifies the same interface.
- [ ] Produce a report with **50/50 successful initial conditions for each task**, using declared tolerances, and record failures honestly until corrected. Test both tasks and require the CLI to exit nonzero when any requested condition fails.

Implement and pass:

```bash
python -m pip install -e '.[sim]'
python -m pytest tests/test_sim_environment.py tests/test_nominal_controllers.py tests/test_sim_logger.py
python scripts/run_nominal.py --config configs/sim.yaml --task all --initial-conditions 50 --seed 723 --output-dir data/raw/nominal
```

## Track 2: Failure Injection Engine & Batch Dataset Generation

**Owner / name:** ____________________

**Scope:** command perturbations, reproducible paired replay, collection, and dataset auditing. Develop against a fake runner immediately; switching to Track 1's runner is an adapter change.

### Target milestone dates

- **Oct. 30:** implement all five modes and collect **500 valid pairs per task**, across 50 initial conditions per task, with canonical CSVs and metadata.
- **Nov. 6:** audit task/mode coverage, replayability, split isolation, and parser compatibility; replace invalid pairs with tracked retries.
- **Nov. 13:** freeze the validated dataset manifest and checksum inventory for BT conversion and evaluation.

### Files to create/modify

- **Create:** `src/telemetry_reasoning/sim/failures.py`, `src/telemetry_reasoning/sim/dataset.py`.
- **Create:** `configs/failures.yaml`, `scripts/generate_dataset.py`, `scripts/validate_dataset.py`.
- **Create:** `tests/test_failure_injection.py`, `tests/test_dataset_generation.py`.
- **Read/reuse:** `MetadataRow`, `FailureMode`, CSV I/O, and Track 1's documented runner interface. Do not edit Track 1 modules.

### Input & output contracts

- Input: nominal control command, current timestamp/subtask, a deterministic perturbation spec, seed, and initial-condition ID. Keep mode configuration/validation in `sim/failures.py`.
- Output: `data/raw/dataset/metadata.csv`, referenced `success_<pair_id>.csv` / `failure_<pair_id>.csv`, and `manifest.json`. Use task-prefixed IDs such as `pick_up_object_000` to remain unique in one combined metadata file.
- `MetadataRow.failed_subtask` identifies the targeted injection stage, not a later observed symptom. `perturbation_param` names the changed quantity and `perturbation_value` is one finite scalar with documented units; axes, onset, duration, retries, and other details go in the sidecar.
- Target size: **50 conditions x 10 accepted pairs = 500 pairs per task; 1,000 metadata rows and 2,000 trace CSVs total**. A pair contains one independently verified successful nominal run and one verified failed perturbed replay, not merely a run with an injection flag.

### Definition of Done

- [ ] Implement the exact five `FailureMode` values with a task/subtask eligibility map: `position_error` offsets a Cartesian target; `orientation_error` rotates a target by a documented axis/angle; `inadequate_grip_strength` lowers grip force; `early_gripper_release` opens/releases before the intended completion; `excessive_speed` raises the motion speed scale. Document scalar parameter meanings, safe simulation bounds, onset/duration, and which command is changed.
- [ ] Test every mode's targeted command, eligible stages, activation boundary, and no-op behavior outside its window. Ensure mode state resets between episodes and the nominal trace never receives the perturbation.
- [ ] Replay identical initial state, controller, physics settings, and seed; change only the selected control perturbation. Use a separate deterministic random stream for choosing injections so selection cannot alter nominal environment randomness. Test equality before activation and reproducible reruns within a documented numeric tolerance.
- [ ] Accept pairs only after checking nominal success, perturbed failure, finite 50 Hz telemetry, and the full ordered subtask template. Bound retries, preserve rejection reasons, and fail collection clearly if the requested count cannot be reached. Resume without duplicate IDs or partially written metadata rows.
- [ ] Sample eligible subtask/mode combinations with recorded coverage. Create deterministic train/dev/test groups at the **initial-condition** level (proposed starting split: 35/5/10 conditions per task). Keep all ten repetitions, both traces, and any shared physical initial-state group in the same split; store assignments in `manifest.json` without expanding `MetadataRow`.
- [ ] Validate exact counts, unique IDs, file existence, all five modes across the dataset, task/subtask validity, replay provenance, split separation, and absence of quarantined runs in accepted metadata. Fake-runner tests cover retries/resume without MuJoCo; real replay tests establish physics reproducibility.

Implement and pass:

```bash
python -m pytest tests/test_failure_injection.py tests/test_dataset_generation.py
python scripts/generate_dataset.py --sim-config configs/sim.yaml --failure-config configs/failures.yaml --task all --initial-conditions 50 --pairs-per-task 500 --seed 723 --output-dir data/raw/dataset
python scripts/validate_dataset.py --data-dir data/raw/dataset --expected-pairs-per-task 500 --expected-conditions-per-task 50
```

## Track 3: Behavior Tree Builder & Deterministic Divergence Function

**Owner / name:** Arik Gershman

**Scope:** extend the existing parser into a batch BT pipeline, tune compression, and calculate reproducible numerical divergence. The proposal does not prescribe a formula; the alignment and metric definitions below are explicit implementation decisions to document and test.

### Target milestone dates

- [x] **Oct. 30 — completed Oct. 23:** run the existing parser on the sample, define batch/alignment APIs, and add synthetic transport and timing edge cases. Verification and public interfaces are recorded below.
- **Nov. 6:** finish batch parsing, tune `k` and major-event thresholds on development data, and preserve the existing public API and sample reproducibility.
- **Nov. 13:** convert the full accepted dataset to BT JSONs with computed divergence and publish compression/event-retention diagnostics.

### Oct. 30 completion evidence and API handoff

- [x] Rebuilt the checked-in pickup pair with the existing parser and reproduced all four sample files byte for byte in `data/processed/track3-oct30/mock/`. Each raw trace has 400 rows; the BT retains 31 success and 34 failure samples, including the 5.08/5.10/5.12 s grip-drop neighborhood. Shared schemas, default configuration, and versioned samples are unchanged.
- [x] Added `build_paired_bts(metadata, data_dir, config) -> BatchBuildResult` in `parser/pipeline.py`. It accepts `Sequence[MetadataRow]` and `CompressionConfig`, reuses `build_paired_bt()`, and returns ordered `trees` and `errors` tuples. Each input pair produces exactly one tree or a `PairBuildError` containing its ID, source filenames, exception type, and message. Duplicate IDs reject the batch; an empty sequence returns empty tuples. File/validation failures do not prevent later pairs from being processed. Metadata CSV errors still raise from `read_metadata_csv()` before batch construction.
- [x] Added `align_subtask_samples(node: SubtaskBT) -> AlignedSubtask` in `parser/divergence.py`. It returns the shared progress grid, failure-run absolute timestamps, and success/failure arrays for every `TelemetryFeature`; scalar shapes are `(n,)`, vector shapes `(n, components)`. Interpolation is linear for continuous features, previous-value for gripper closure, and normalized/sign-aligned shortest-path SLERP for quaternions. A singleton stays constant over its partner's grid; two singletons compare once at progress zero. Queries within `1e-12` progress of an observed sample snap to that sample to avoid roundoff-induced transitions; the grid remains the unrounded sorted union. Inputs are not mutated, and interpolated values are comparisons rather than new observations.
- [x] Added synthetic tests for both task templates, transport semantics, seven-joint arrays, unequal stage lengths/sample counts, shifted timestamps, singleton stages, gripper transitions, quaternion sign/shortest-path behavior, malformed or missing files, and missing/repeated/reordered stages on either side. The additional completion audit on Oct. 23 passes all **97 tests with warnings treated as errors**, including the original parser/reproducibility tests, 100 seeded analytic-motion cases, raw-validation failures, and batch error ordering/preflight checks.
- [x] Completed an independent mixed-task stress check: 100 synthetic pairs (50 per task), one through nine joints, unequal stage lengths, and `k=0/1/5/100`, for 400 successful pair builds and 1,400 aligned subtasks. All trees round-trip through the schema, all aligned values are finite, and input files remain unchanged. Fresh generation reproduces all four sample files byte for byte, and the documented example passes verbatim. Fixed a small-angle quaternion approximation exposed by this audit using stable SLERP weights; differently sampled constant-angular-speed motion now agrees within numerical tolerance. Evidence is in `runs/track3-oct30/verification.json`; these synthetic parameter checks are validation, not train/dev tuning.

Both APIs and their result types are exported from `telemetry_reasoning.parser`. The batch API is currently in-memory and leaves divergence null. Dataset serialization, the CLI/configs, compression tuning/reporting, and divergence metric calculation remain on the Nov. 6/13 milestones. Alignment removes pure duration differences; duration is not a v1 divergence feature. Its mapped failure timestamps can be interpolated times and must not be presented as exact observed onset.

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

### Files to create/modify

- **Modify (exclusive ownership):** `src/telemetry_reasoning/parser/compressor.py`, `src/telemetry_reasoning/parser/__init__.py`, `tests/test_parser.py`.
- **Implemented; extend for later milestones:** `src/telemetry_reasoning/parser/divergence.py`, `src/telemetry_reasoning/parser/pipeline.py`.
- **Create:** `configs/parser.yaml`, `configs/divergence.yaml`, `scripts/parse_dataset.py`.
- **Implemented; extend for later milestones:** `tests/test_bt_pipeline.py`, `tests/test_divergence.py` (currently alignment tests; metric tests remain pending).
- Keep `configs/default.yaml` and the checked-in sample unchanged. `configs/parser.yaml` is a complete `ProjectConfig` (`simulation` plus `parser`); load the separate divergence settings in `parser/divergence.py`.

### Input & output contracts

- Start with `read_metadata_csv("data/sample/metadata.csv")`, the paired sample CSVs, `CompressionConfig`, and existing `build_paired_bt(metadata, data_dir, config)`.
- Output: one schema-valid `PairedBehaviorTree` per input pair under `data/processed/<dataset>/bt/`, plus a separate parse/compression report. Preserve the existing nested goal/subgoal/action semantics; do not introduce an incompatible tree shape.
- Compute divergence from each node's `success_samples` and `failure_samples`; do not zip arrays by index. Fill only the existing `DeterministicDivergence` fields. Keep alignment settings and diagnostics in the report, not extra schema fields.
- Raw 50 Hz traces are the reference for evaluating information lost through compression. The LLM inputs remain the compressed BTs and their computed comparisons.

### Definition of Done

- [x] Extend in-memory batch construction for both task templates with clear per-pair errors. Preserve original timestamps, semantic metadata, structural alignment, and all existing CSV/schema validation. Keep the existing `build_paired_bt()` default behavior (uncomputed divergence). Test unequal durations/counts, missing/repeated/reordered subtasks, and invalid pair files.
- [ ] Finish dataset serialization, parse/compression reporting, configuration, and the CLI; add divergence explicitly in the new pipeline once metric computation is implemented.
- [x] Retain first/last rows, up to `k` interior samples, and major-event neighbors `t-1/t/t+1`; compare adjacent rows only within a consecutive subtask. Preserve strict `>` threshold semantics, sorted deduplication, clipping, and `k=0` behavior. Existing quaternion compression uses component deltas; any replacement needs explicit regression evidence. Verified by the existing parser regression suite on Oct. 23.
- [ ] Sweep a documented small `k` grid and per-feature event thresholds using train/dev data only. Measure compression ratio, event retention against raw traces, and resulting prompt size. Preserve the mock's 5.08/5.10/5.12 s grip-drop neighborhood; do not tune on test labels or overwrite the mock defaults.
- [x] Implement deterministic **within-subtask progress alignment**: map each trace's timestamps to `u=(t-t_start)/(t_end-t_start)` and compare on the sorted union of retained progress values. Linearly interpolate continuous scalar/vector features, use previous-value interpolation for gripper closure, and normalize/sign-align quaternions with shortest-path interpolation. Define a deterministic singleton-segment rule and test it. Never interpolate across subtask boundaries. Report that this alignment removes pure duration differences, so duration itself is not a v1 divergence feature. See the API handoff above for singleton and numerical-tolerance rules.
- [ ] Define `delta_f(u)=max_j(abs(failure_f,j(u)-success_f,j(u)))` (one component for scalars), then `max_abs_deltas[f]=max_u(delta_f(u))` for all `TelemetryFeature` values. Quaternion deltas are dimensionless component differences after sign alignment, not radians. First divergence is the earliest grid point exceeding a configured per-feature threshold, mapped back to the **failure run's absolute timestamp**. Separate these paired-trace thresholds from compression's adjacent-row thresholds. Store a computed object with null first timestamp and zero deltas for identical traces.
- [ ] Test known scalar/vector offsets, threshold equality, identical traces, shifted/differently sampled traces, quaternion `q` versus `-q`, and deterministic serialization. For the sample and a 2 N grip threshold, verify the first grip deviation at **5.10 s** and maximum grip delta **12 N**. Compare compressed-onset estimates against raw references and report their error; do not claim unobserved exact physical onset.

Later milestone acceptance commands (the tests exist now; the CLI/configs and full dataset are still pending):

```bash
python -m pytest tests/test_parser.py tests/test_bt_pipeline.py tests/test_divergence.py
python scripts/parse_dataset.py --metadata data/sample/metadata.csv --data-dir data/sample --config configs/parser.yaml --divergence-config configs/divergence.yaml --output-dir data/processed/smoke/bt
python scripts/parse_dataset.py --metadata data/raw/dataset/metadata.csv --data-dir data/raw/dataset --config configs/parser.yaml --divergence-config configs/divergence.yaml --output-dir data/processed/dataset/bt
```

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

- [ ] Build subtask-aware rules for position offset, orientation mismatch, low grip, premature opening, and excessive speed, using only observable failure telemetry and declared task constants. Use object/EE relative geometry where possible; identify absent target/desired-pose information explicitly and abstain when evidence cannot distinguish a mode. Do not substitute label-derived target values.
- [ ] Define precedence for competing rules and `abstain` for ambiguous/unsupported cases. Generate a concise causal template identifying predicted mode/stage and measured evidence with timestamps/units. Test all five modes, competing symptoms, benign approach/open-gripper behavior, and no-evidence input.
- [ ] Implement the shared JSONL validator and strict `(pair_id, model_variant)` joins. Reject duplicates/unknown IDs; account for missing/invalid outputs instead of silently dropping them. Test empty inputs and classes absent from a small evaluation set.
- [ ] Report failure-mode accuracy and macro-F1 across the fixed five labels, failed-subtask accuracy, abstention/error/coverage rates, and confusion matrices, broken down by task and model variant. Count abstentions/missing predictions as incorrect for accuracy and as missed true classes for recall. Compute optional onset error only where a defined prediction timestamp and independent reference exist; injection time and observed divergence are different events.
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
- [ ] **Real producer smoke:** run both tasks through MuJoCo with a small deterministic seed set and injections covering all five modes across eligible stages; require verified success/failure outcomes, complete schema-valid traces, and replay agreement. Feed those newly generated CSVs through the same parser and baseline stages. The mock smoke alone does not verify Tracks 1-2.
- [ ] **Real model smoke:** on a machine with model access/compute, run the three HF variants on the same parsed real pairs and the rule baseline on their failure-only views. Validate allowed inputs and JSONL outputs before evaluating. Keep real and stub results in separate run directories.
- [ ] **Full dataset gate:** `validate_dataset.py` confirms 500 pairs per task / 50 conditions per task, then `parse_dataset.py` produces 1,000 validated paired BTs with computed divergence. Freeze train/dev/test assignment, prompt/config versions, and checksums; reserve the held-out test results for the later final evaluation.
- [ ] **Integration evidence:** attach the smoke commands, dataset counts, parser/compression diagnostics, four-variant result validation, plot examples, and any remaining hardware-dependent checks to the integration PR. Never mark real-MuJoCo or HF checks passed because a stub test passed.

Planned integration commands:

```bash
python -m pytest
python scripts/integration_smoke.py --backend stub --output-dir runs/integration-cpu
python -m pytest tests/test_integration_smoke.py
python scripts/integration_smoke.py --sim-backend mujoco --backend hf --tasks all --cover-all-failure-modes --output-dir runs/integration-real
python scripts/evaluate.py --metadata data/sample/metadata.csv --predictions runs/smoke/baseline_1.jsonl runs/smoke/baseline_2.jsonl runs/smoke/baseline_3.jsonl runs/smoke/main.jsonl --config configs/eval.yaml --output-dir runs/smoke/comparison
```

Default `python -m pytest` must remain CPU/offline compatible: skip explicitly marked hardware/model tests when their resources are absent and report those skips. Passing that command is necessary, but the separate real producer/model smoke evidence is also required for the corresponding track to be complete.
