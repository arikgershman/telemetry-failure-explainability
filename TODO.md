# Project TODO: five parallel workstreams

Each member claims **one track** by filling in its owner field. Each track owns implementation, configuration, tests, and a runnable demonstration. The five tracks are intended to have comparable scope; GPU training and the final study are not additional requirements for any one owner during this period.

## Milestones and current starting point

| Target | Proposal milestone | Track deliverables |
| --- | --- | --- |
| **Oct. 30** | Generate, run, and store MuJoCo data | Track 1: MuJoCo environment, nominal controllers, failure injection, and 500 accepted success/failure pairs per task. Track 2: validation and finalization begin after dataset generation. Tracks 3-5: work immediately against the existing mock contracts. |
| **Nov. 6** | Implement simulation parser and all three baselines | Track 3: batch parsing and compression tuning. Track 4: Baseline 1 and evaluator. Track 5: Baselines 2 and 3. Tracks 1-2: replay, coverage, and data-quality validation. |
| **Nov. 13** | Convert simulations to BTs and implement deterministic divergence | Track 3: full dataset conversion and divergence. Track 4: integrated metrics/plots. Track 5: main paired-BT-plus-divergence model. All tracks: end-to-end smoke test. |

Already implemented: `src/telemetry_reasoning/schemas.py`, validated CSV I/O, configuration loading, consecutive subtask segmentation, endpoint/uniform/major-event compression, and `build_paired_bt()` in `src/telemetry_reasoning/parser/compressor.py`. Extend these APIs rather than rebuilding them.

### Status conventions and current scope (Nov. 18)

Checked implementation items describe available code and verified interfaces;
dated generation/audit receipts describe historical runs. The Nov. 18 restoration
verified all three pinned GitHub release assets and restored the ignored
1,000-pair archive without changing its 2,006 source files. Reproduce restoration
with `scripts/restore_dataset.py` and `configs/dataset-v1.json`; see the
[release workflow](docs/dataset-release.md). Original calibration CSVs and exact
producer provenance are not supplied by the release. The recovered nine accepted
configurations per task are explicitly labeled as a catalogue, with no invented
calibration rates. Track 2's unchecked provenance, physical-outcome, timing, and
certification gates remain open until their required evidence is verified.

The current pipeline is **generate/validate -> parse -> enrich divergence ->
predict -> evaluate/plot**. CPU integration covers synthetic and fresh real
physics traces, the rules baseline, and three explicitly labeled stubs.
Real-model generation, reviewed training labels, GPU optimization, archival
certification, and the held-out study remain separate gates. Historical test
totals below are dated receipts, not current suite totals.

The CPU foundation, pinned restoration, runner hooks, grouped handoff,
production compression/full enriched-BT publication, preference formatter, and
GPU preparation are implemented. The handoff retains its failed archival-replay
gate; historical certification is deferred. Next run comparable real-model
inference on the GPU, obtain reviewed train/dev preferences before GPU adapter
optimization, and preserve held-out test data for the final study. Recover the
original producer inputs and publish a newly certified dataset handoff separately.

### Shared contracts: use these immediately

All paths in this file are relative to the repository root. Paths marked **create** and their CLI flags are planned deliverables, not commands already implemented. The following files and APIs already exist:

| Contract/source | Required use |
| --- | --- |
| `src/telemetry_reasoning/schemas.py`: `Task`, `Subtask`, `FailureMode`, `TASK_SUBTASKS` | Use the exact enum values. Pickup order: `approach`, `align`, `grasp`, `lift`. Transport order: `maintain_grasp`, `move_along_path`, `reach_target`. |
| `RawTelemetryRow` | Exact CSV columns: `timestamp`, `subtask`, `joint_positions`, `joint_velocities`, `joint_torques`, `gripper_state`, `gripper_force`, `ee_pos`, `ee_quat`, `obj_pos`, `obj_quat`, `obj_vel`, `contact_force`. Vector cells contain JSON arrays. |
| `MetadataRow` | Exact columns: `pair_id`, `task`, `success_file`, `failure_file`, `failed_subtask`, `failure_mode`, `perturbation_param`, `perturbation_value`, `seed`. IDs are strings; filenames resolve relative to the supplied data directory. |
| `PairedBehaviorTree`, `SubtaskBT`, `SubtaskMetadata` | One complete ordered task structure; nonempty success samples for every node; observed failure samples form a nonempty prefix, with empty lists for its unobserved suffix. Required observation/end/divergence states distinguish coverage from completion. Use `TASK_GOALS` and `SUBTASK_METADATA` for semantics. |
| `DeterministicDivergence`, `TelemetryFeature` | Per-subtask `first_divergence_timestamp` and `max_abs_deltas` keyed by existing feature names. `divergence_status` distinguishes uncomputed observed stages from not-comparable unobserved stages; null never means zero. |
| `src/telemetry_reasoning/io.py` | Reuse `read_telemetry_csv()`, `write_telemetry_csv()`, `read_metadata_csv()`, `write_metadata_csv()`, and `validate_raw_run()`. |
| `src/telemetry_reasoning/config.py`, `configs/default.yaml` | `ProjectConfig` accepts only `simulation` and `parser`; nested configuration rejects unknown keys. Add track-specific config loaders in the owning modules instead of adding arbitrary sections to the default YAML. |
| `data/sample/success_000.csv`, `data/sample/failure_000.csv`, `data/sample/metadata.csv` | One synthetic pickup pair, ID `"000"`, seed `723`; 400 rows per trace, timestamps 0.00-7.98 s at 50 Hz. The failed grasp drops from 15 N to 3 N at **5.10 s**, then the object does not lift. |
| `data/sample/pair_000_bt.json` | Valid paired BT, already compressed; every divergence is null. Start model formatting/evaluation tests here. It is neither physics output nor evidence of performance across all tasks/modes. |

Keep units consistent: seconds; meters; rad and rad/s for joint positions/velocities; N m for joint torques; N for scalar force magnitudes; m/s for object velocity. Quaternions use **w/x/y/z**, and producers must normalize them and keep signs consistent. `gripper_state` is closure in [0, 1], with 0=open. Joint-vector lengths must remain fixed within and across paired runs, but must not be hard-coded to the mock's six joints.

Raw observations must be 0.02 s apart. Compressed samples keep their original timestamps and may have unequal gaps/counts. Successful traces require every task subtask once, in order; failures may contain a nonempty ordered prefix. The BT keeps every expected node, with explicit coverage states and empty failure samples for unobserved stages. Do not fabricate observations; quarantine malformed or non-prefix traces and report them. Track 2 independently verifies intentional termination and physical outcomes.

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

These are the current interfaces and explicit unresolved handoff decisions.
Use local fakes for unavailable producers without claiming real acceptance.

1. **Track 1 -> Track 2:** Track 1 provides the MuJoCo simulation, nominal controllers, failure injection engine, and paired dataset generator. The implemented runner is `run_episode(*, task, seed, initial_condition_id, failure=None, output=None, spread=0.015, control_hook=None) -> EpisodeResult`, with `rows`, `task_succeeded`, and JSON-serializable `diagnostics` attributes. `failure` is an optional `FailureConfig`; `FailureEngine` transforms control targets separately from measured telemetry. New diagnostics include `telemetry_start_sim_s`, machine-readable terminal state, independent physical outcomes, and source/model/runtime fingerprints. The optional `control_hook(timestamp, subtask, nominal_command)` executes at the canonical task's 500 Hz command boundary with trace-relative time; `EpisodeResult` also supports tuple unpacking as `(rows, task_succeeded, diagnostics)`. Hook command semantics and validation are defined in `sim/controls.py`; hooks never receive stored telemetry as a mutable control object. Track 2 verifies deterministic replay, paired initial states, pre-injection agreement, perturbation isolation, physical outcomes, and telemetry integrity. The historically generated dataset is under `outputs/dataset/batch_20261030_152125_e5ca72de/`, with task-specific paired CSVs, `metadata.csv`, `diagnostics.jsonl`, and `summary.json`. Its audit reports 500 accepted pairs per task across 50 initial conditions and four calibrated modes. All five mechanisms are implemented; `excessive_speed` is excluded from the accepted dataset because of insufficient failure reliability. The archived data and exact calibration inputs require separate availability/provenance checks.
2. **Track 2 -> Tracks 3/4:** the [dataset handoff](docs/track2-dataset-handoff.md) keeps task-specific `metadata.csv` files and referenced telemetry intact, provides an atomic handoff directory with `dataset_manifest.json`, `dataset_splits.json`, and `tasks/<task>.json`, containing grouped train/dev/test assignments, source checksums, and explicit per-gate certification status. Existing handoffs are never overwritten. Producer revisions, archived time origins, and independent terminal outcomes still require confirmation before final certification. Keep sidecar fields out of strict CSV schemas. Track 3 consumes the task CSVs and grouped manifest for tuning; Track 4 consumes the corresponding manifest for split-based evaluation. Archived injection timestamps are absolute simulator times and must not be used as trace-relative onset references without verified origins. The dataset reports four modes at 125 accepted pairs per mode per task; `excessive_speed` is a valid `FailureMode` with no accepted release examples.
3. **Track 3 -> Tracks 4/5:** `pair_<pair_id>_bt.json` files validated as `PairedBehaviorTree`, using one canonical contract for complete and partial failure traces. After divergence runs, observed nodes have `divergence_status="computed"` with a `DeterministicDivergence`; unchanged comparisons have zero deltas and a null first-divergence timestamp. Unobserved nodes retain empty failure samples, `divergence_status="not_comparable"`, and a null object. Observed nodes with null objects remain explicitly uncomputed. Track 5 may populate a deep copy of the sample with hand-checked divergence values and matching statuses for tests while Track 3 works.
4. **Tracks 4/5 -> evaluator:** one JSONL prediction per `(pair_id, model_variant)`, with fields `pair_id`, `model_variant` (`baseline_1`, `baseline_2`, `baseline_3`, `main`), `predicted_failure_mode` (any of the five `FailureMode` values or null), `predicted_subtask` (`Subtask` value or null), `explanation` (string), `evidence` (list of `{subtask, timestamp, feature, value}` records using `TelemetryFeature` values), `status` (`ok`, `abstain`, `error`), and `backend` (`rules`, `hf`, `stub`). Evidence refers to observed failure samples, with `value` a finite scalar or vector matching that feature's `RawTelemetryRow` shape; paired/divergence comparisons may also be described in the explanation. Validation is implemented in `eval/contracts.py`. Release reporting distinguishes the four represented classes from the valid but unrepresented `excessive_speed` class. Additional runtime/token/config provenance belongs in a sibling run-manifest JSON. Stub outputs are for plumbing tests and must never be reported as LLM results.

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
- **Historical calibration inputs (not supplied by the release):** `data/calibration/pickup_recommended.csv`, `data/calibration/transport_recommended.csv`; obtain exact producer copies for reproducibility. Restoration produces a separately labeled accepted-configuration catalogue, with no invented calibration rates.
- **Read/reuse:** `src/telemetry_reasoning/schemas.py`, `src/telemetry_reasoning/io.py`, and shared task/subtask definitions.

### Input & output contracts

- Input: `Task`, `TASK_SUBTASKS`, simulation seed, initial-condition ID, optional `FailureConfig`, and optional `control_hook`. The runner returns `EpisodeResult`, supporting attributes and `(rows, task_succeeded, diagnostics)` unpacking. Hook commands are detached, validated dictionaries; timestamps are relative to canonical task start. The hook operates on command targets and force limits, while telemetry records measured state.
- Output: `RawTelemetryRow` observations with timestamps, subtask labels, joint positions/velocities/torques, gripper state/force, end-effector pose, object pose/velocity, and contact force. Simulation steps at 500 Hz and telemetry is recorded at 50 Hz.
- Dataset: task-specific `success/` and `failure/` directories containing paired telemetry CSVs, with `metadata.csv`, `diagnostics.jsonl`, and `summary.json` for each task.
- Coverage: 50 initial conditions × 10 accepted pairs = 500 pairs per task; 1,000 pairs and 2,000 telemetry CSVs total. Four selected failure modes are balanced at 125 pairs per mode per task.

### Definition of Done

- [x] Set up the Franka Emika Panda robot using MuJoCo Menagerie and implement the simulation environment, episode runner, and `FailureConfig`/`FailureEngine` control path.
- [x] Implement the optional generic `control_hook` alongside `FailureConfig`, retain `EpisodeResult` attributes and add tuple unpacking; document strict command units, immutable nominal inputs, relative hook time, and verified identity/perturbation behavior.
- [x] Implement bounded waypoint-based controllers for `pick_up_object` (`approach`, `align`, `grasp`, `lift`) and `transport_object` (`maintain_grasp`, `move_along_path`, `reach_target`).
- [x] Step physics at **500 Hz / 0.002 s** and collect telemetry at **50 Hz / 0.02 s**, using the shared `RawTelemetryRow` schema and measured robot/object state.
- [x] Implement 50 seeded initial-condition IDs per task and generate successful nominal executions for paired dataset collection.
- [x] Implement all five `FailureMode` mechanisms: `position_error`, `orientation_error`, `inadequate_grip_strength`, `early_gripper_release`, and `excessive_speed`. Apply perturbations through `FailureEngine` command transformations rather than modifying recorded telemetry.
- [x] Calibrate failure configurations and evaluate activation and task failure rates. Select four reliable modes for dataset generation; exclude `excessive_speed` after calibration showed insufficient failure reliability.
- [x] Implement paired nominal/perturbed execution generation using matching seeds and initial-condition IDs, with acceptance checks, bounded retries, rejection diagnostics, and metadata recording.
- [x] Generate **500 accepted success/failure pairs per task**, covering 50 initial conditions with 10 accepted pairs per condition.
- [x] Balance the four selected failure modes at 125 accepted pairs per mode per task, producing **1,000 pairs and 2,000 telemetry CSVs**.
- [x] Save telemetry, metadata, episode diagnostics, and generation summaries; complete an initial dataset audit confirming expected file counts, condition coverage, failure-mode balance, and zero reported issues (historical generation/audit evidence).
- [ ] Supply the exact archived calibration files and confirm producer source/model/config revisions so these historical runs can be independently reproduced and certified.

Implemented generation command (run from `src/` after obtaining the exact calibration inputs; paths below describe the archived generation):

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

**Status (Nov. 17 handoff):** all 1,000 archived pairs passed validation and batch parsing. The handoff records a verified checksum inventory and task-specific grouped split manifests. See [the dataset handoff](docs/track2-dataset-handoff.md). These ignored artifacts are not bundled with the source checkout. Producer provenance, independent failure-outcome verification, and final release certification remain pending.

**Scope:** validate reproducibility, physical outcomes, telemetry integrity, and parser compatibility of the completed simulation dataset. Establish leakage-free dataset splits, automated checks, and a frozen dataset manifest for downstream Behavior Tree conversion and evaluation.

### Target milestone dates

- **Nov. 6:** validate deterministic replay, physical outcomes, telemetry quality, and parser compatibility; identify and resolve invalid pairs.
- **Nov. 13:** finalize dataset splits, automated validation, provenance, and checksum inventory for downstream integration.

### Files to create/modify

- **Implemented:** `src/telemetry_reasoning/sim/release.py`, `scripts/restore_dataset.py`, and `configs/dataset-v1.json` for pinned online/offline restoration; `audit_robot_dataset.py`, `validate_dataset.py`, `dataset_certification.py`, `split_robot_dataset.py`, and `freeze_robot_dataset.py` for strict validation, source-bound certification, grouped splitting, and atomic handoff generation/verification.
- **Implemented tests:** `tests/test_dataset_validation.py`, `test_dataset_acceptance.py`, `test_dataset_splits.py`, `test_dataset_manifest.py`, `test_dataset_retries.py`, `test_failure_validation.py`, and opt-in `test_sim_reproducibility.py`.
- **Generated:** `<batch>/provenance/release.json`, copied published clock assets, and recovered accepted-configuration catalogue; an explicit fresh `<handoff>/dataset_manifest.json`, `dataset_splits.json`, and `tasks/<task>.json` for tuning and evaluation. Earlier `<batch>/manifest.json` and `<batch>/handoff/<task>.json` layouts remain historical, not the preferred publication API.
- **Read/reuse:** `src/telemetry_reasoning/sim/runner.py`, `src/telemetry_reasoning/sim/failures.py`, `src/telemetry_reasoning/sim/dataset.py`, `src/telemetry_reasoning/parser/compressor.py`, and shared schemas/I/O.

### Input & output contracts

- Input: the restored dataset under `outputs/dataset/batch_20261030_152125_e5ca72de/`, including paired telemetry CSVs, task-specific metadata, original diagnostics/accepted configurations, and initial-condition IDs. Exact original calibration inputs and independent producer evidence remain separate required certification inputs.
- Output: preserved validated telemetry, deterministic train/dev/test assignments, per-gate certification/compatibility results, and an atomic handoff containing `dataset_manifest.json`, `dataset_splits.json`, task manifests, and source checksums.
- Keep the existing `RawTelemetryRow` and `MetadataRow` schemas unchanged. Store split assignments, simulation provenance, and additional validation information in the manifest. Success traces require the full ordered task sequence. Failure traces may end during any stage as long as the observed stages form a nonempty ordered prefix. Unobserved stages have no failure samples.

### Definition of Done

- [x] Restore all three GitHub release assets using pinned IDs/sizes/SHA-256 digests, preserve archived source bytes, retain published clocks separately as unverified inputs, and recover an explicitly labeled accepted-configuration catalogue. Cover offline/download behavior, archive safety/resource limits, corruption, immutable destinations, and failed-publication cleanup with CPU tests.
- [x] Verify fresh nominal deterministic replay/reset for both tasks at seed 723, initial condition 0, and injected replay for one released configuration per calibrated failure mode per task, each repeated twice. Numeric telemetry agrees within absolute tolerance `1e-8`, with no relative tolerance. This is representative replay evidence, not replay certification of every stored CSV.
- [x] Validate stored paired initial-state identities, metadata/diagnostic consistency, and activation diagnostics; test command isolation and activation boundaries for all five mechanisms, including `excessive_speed`.
- [x] Record canonical telemetry/task onset in absolute simulator time as `telemetry_start_sim_s` for newly generated episodes.
- [ ] Recover and verify archived per-run task origins, use the new timing diagnostics for fresh runs, and verify full pre-injection trajectory agreement. Recording a new diagnostic does not certify old CSVs.
- [x] Verify real nominal success across all 50 initial conditions per task at seed 723. Independently check final pickup object rise exceeds 0.12 m and transport XY placement is within 0.04 m of the expected 0.15 m displacement.
- [x] Check all released traces using canonical CSV validation, including logging cadence, schema, ordered stages, paired joint dimensions, quaternion normalization/sign continuity, and metadata file-reference reuse. Inspect units and measured force sources in the logger; this does not independently calibrate force accuracy.
- [ ] Independently verify physical outcomes for all stored failure runs using machine-readable outcome measurements from the producer; controller outcome strings alone are insufficient.
- [x] Use `build_paired_bt()` as the acceptance gate for stage coverage. Test that ordered partial failures pass without changing source CSVs and that skipped stages and incomplete success traces are rejected.
- [x] Pass full-release coverage and parser acceptance for all 1,000 pairs. Each task has 500 pairs across 50 initial conditions and 125 pairs per calibrated mode.
- [x] Implement deterministic train/dev/test splits grouped by initial condition (35/5/10 per task and 700/100/200 total pairs), keeping paired and repeated executions together and checking task/failure-mode coverage, duplicate diagnostics, and split isolation.
- [x] Add automated reset/replay, failure-injection boundary/isolation, dataset validation, split, and checksum tests. Record the exclusion of `excessive_speed` from the accepted dataset due to insufficient failure reliability.
- [x] Add 12 CPU-only generator retry tests using fake episode results: inactive injection, injected success, and empty telemetry rejection; preserved condition/mode and paired inputs; accepted-only CSV/diagnostic output; attempt exhaustion and partial-run reporting; and immediate abort for failed/incomplete nominal episodes. These exercise the real generator and CSV I/O without MuJoCo or model assets.
- [x] Implement and verify the runner's optional `control_hook`: CPU command validation/isolation plus native identity and position/orientation/gripper/speed acceptance for both tasks. A `None` result retains the nominal command; motion commands compose with `FailureConfig` and actuator failure overrides apply afterward.
- [x] Implement manifest/checksum generation and verification, including changed/missing/unexpected files, same-size corruption, unsafe inventory paths, source fingerprints, split indices, validation records, and explicit verification limitations. Refuse to freeze an incompatible release.
- [x] Regenerate the release splits at seed 723, publish task-specific manifest files locally, and verify their layout against the Track 3 tuner and Track 4 evaluator.
- [x] Generate the archived dataset manifest and verify all 2,006 inventoried files against their SHA-256 checksums.
- [x] Verify that the `dataset-v1` release tag points to `73eeb975c2593e7fe8ad7a8a31a6253693303784` and pins MuJoCo Menagerie to `f3b6013629c7b94f8ee93fd977b161fa8350f9bc`.
- [ ] Confirm the exact generator working tree, model and calibration inputs used for the ZIP, and obtain per-run task time origins before final certification.

### Historical validation results and remaining certification blockers

The release audit exited **0**. The detailed report is in `runs/track2-partial-audit.json`.

| Task | Released pairs | Parser-compatible pairs | Quarantined pairs |
| --- | ---: | ---: | ---: |
| `pick_up_object` | 500 | 500 | 0 |
| `transport_object` | 500 | 500 | 0 |
| **Total** | **1,000** | **1,000** | **0** |

- All 131 selected Track 2 and parser tests passed, including a regression test for the task-specific manifest layout.
- The batch parser wrote 1,000 BTs and rejected zero pairs. Both task-specific compression tuning runs evaluated nine candidates on 50 dev pairs and recommended `k=5;threshold_scale=1`.
- At that handoff, the default full suite reported 718 passed, 23 failed, and 6 skipped. Failures were in Track 3 (one parser output hash), Track 4 (one failure-mode vocabulary mismatch), and Track 5 (21 LLM, CLI, replay, and prompt cases). This is a historical receipt, not the current suite result. The six opt-in Track 2 MuJoCo cases were skipped.
- Earlier real MuJoCo runs passed nominal replay/reset for both tasks, nominal success and object-motion checks across all 100 task/condition combinations, and injected replay/failure/activation for four representative modes per task. Menagerie was pinned to `f3b6013629c7b94f8ee93fd977b161fa8350f9bc`. On that historical opt-in rerun, Windows Application Control blocked a native MuJoCo module during import, so all six cases stopped before reaching simulation assertions.
- **Producer provenance/time follow-up:** new episodes record task onset as `telemetry_start_sim_s`; archived origins still need recovery and verification. Confirm exact generator/model/config revisions and machine-readable physical outcome measurements. Current source hashes cannot establish the revision that generated old CSVs.
- **Runner interface acceptance (Nov. 18):** `run_episode()` accepts `FailureConfig` and the optional 500 Hz command hook, returns attribute-compatible `EpisodeResult`, and supports tuple unpacking. Current native acceptance passed 57 tests, including fresh reproducibility, exact hook identity traces for both tasks, and physical command/outcome checks. Fresh diagnostics record physical terminal measurements and source/model fingerprints. The Panda home-keyframe free-object initialization defect was fixed by restoring the cube's scene default before applying its seeded offset; current-runtime behavior does not establish the archived producer state.

### Nov. 18 restoration and representative archival replay

- Restored all three digest-locked GitHub release assets and preserved 2,006 original task files. Recovered nine accepted configurations per task, explicitly distinct from original calibration inputs. Restored the local release tag at the verified source reference.
- Restored 80 pinned Panda Git blobs; receipt: `data/raw/model-assets/panda-f3b60136.json`. With MuJoCo 3.3.7, the current native acceptance suite passed 57 tests. These establish the current runner, not the historical producer runtime.
- The hardened representative recovery independently repeated both nominal and injected executions for eight archived pairs: 32 actual executions and 16 archived run records. All 16 fresh repeatability checks passed; **all 16 archived records mismatched** strict `atol=1e-8, rtol=0` comparison. `runs/dataset-v1-replay-20261118-final/evidence.json` preserves mismatch evidence and unverified candidate origins; published clock sidecars do not become verified from this attempt.
- Focused source-bound certification checks passed 75 tests; pinned-restoration regressions passed 38 tests. These tests verify evidence handling and refusal behavior rather than certifying unknown historical facts.
- Published and verified canonical handoff: `runs/dataset-v1-20261118/handoff`, with 1,000 accepted pairs, 2,006 source files, and 700/100/200 train/dev/test pairs. Its actual `release_status` is `parser_ready_certification_failed`: strict archival replay is a known failed gate, while timing, physical-outcome, and producer-provenance gates remain pending. Archived per-run origins, terminal physical outcomes, exact producer/model/runtime state, and original calibration inputs remain unresolved. `--require-certified` refuses this handoff. The optional current-runtime profile in `configs/sim-validation.requirements.txt` pins MuJoCo 3.3.7 and NumPy 2.5.4; it does not identify the original producer environment.
- Final current default suite: **989 passed, 12 skipped** (native checks are opt-in). The separate native acceptance run passed **57 tests**. `runs/dataset-v1-20261118/verification.json` records source/handoff verification, split counts, excluded unverified relative clocks, and clean strict-certification refusal without publication.

### Downstream handoff details

Use `<batch>/<task>/metadata.csv` with `<batch>/<task>` as the parser's `--data-dir`, and separate BT output directories per task. Pair IDs overlap across tasks: use `(task, pair_id)` for joins, and coordinate globally unique IDs before combining tasks in one evaluation run. Do not silently merge metadata or globally glob those BT directories.

Split entries record task, pair ID, initial condition, failure mode, and batch-relative POSIX file paths. Stored `injection_timestamp_s` is explicitly labeled **absolute simulator time**; CSV timestamps are relative to canonical task onset. The task manifest exposes evaluator-compatible `injection_timestamp` only when the failure run's origin has been independently verified. Published clock filenames and nominal-only replay do not establish that verification.

The logger uses generalized actuator forces for joint torques/finger force, measured finger closure for gripper state, and MuJoCo contact normal forces involving the cube for `contact_force`. Units follow the shared SI schema, with w/x/y/z quaternions; quaternion norm checks use absolute tolerance `1e-6`.

Prefer `freeze_robot_dataset --handoff-dir <fresh-path>` to publish a complete immutable handoff, and consume `<handoff>/tasks/<task>.json`. The verifier checks source hashes and grouped assignments; `--require-certified` refuses any pending gate. Historical `<batch>/manifest.json`, `<batch>/handoff/<task>.json`, and root-level manifest/audit files are preserved as older receipts. Final release certification still requires the producer checks above.

The handoff supports Track 3 compression and divergence work and Track 4 split-based evaluation. Track 5 can use the grouped splits and gets computed divergence from Track 3. Optional onset scoring requires the simulator time corresponding to the start of each failure trace.

### Verification commands (PowerShell, repository root)

```powershell
$python = '.\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Path runs -Force | Out-Null
& $python -m pytest tests/test_dataset_acceptance.py tests/test_dataset_validation.py tests/test_dataset_retries.py tests/test_failure_validation.py tests/test_dataset_manifest.py tests/test_dataset_splits.py tests/test_parser.py tests/test_partial_traces.py -q -p no:cacheprovider --basetemp runs/track2-partial-test-temp
& $python -m telemetry_reasoning.sim.audit_robot_dataset outputs/dataset/batch_20261030_152125_e5ca72de --report runs/track2-partial-audit.json
git submodule update --init --depth 1 assets/mujoco_menagerie
$env:TRACK2_REAL_SIM = '1'
& $python -m pytest tests/test_sim_reproducibility.py -q -p no:cacheprovider --basetemp runs/track2-real-temp
Remove-Item Env:TRACK2_REAL_SIM
& $python -m telemetry_reasoning.sim.split_robot_dataset outputs/dataset/batch_20261030_152125_e5ca72de --output runs/track2-splits.json
```

Default tests skip real physics unless explicitly enabled; dataset-dependent tests skip if the release is absent. Use a fresh temporary output path when needed. A generated split file alone does not establish dataset acceptance.

To regenerate the archived dataset inventory and the downstream task manifests, then verify the inventory:

```powershell
$batch = 'outputs/dataset/batch_20261030_152125_e5ca72de'
$handoff = 'runs/dataset-v1-20261118/handoff' # Must be a fresh destination.
& $python -m telemetry_reasoning.sim.freeze_robot_dataset $batch --evidence runs/dataset-v1-replay-20261118-final/evidence.json --handoff-dir $handoff
& $python -m telemetry_reasoning.sim.freeze_robot_dataset $batch --verify "$handoff/dataset_manifest.json"
# Omit --evidence when no separate replay receipt is available. A mismatch receipt never verifies clocks.
# Add --require-certified when every certification gate must pass.
```

### Final deliverables

- Reproducibility, physical outcome, and telemetry validation results.
- Parser-compatible paired telemetry dataset.
- Leakage-free train/dev/test assignments.
- Automated simulation and dataset validation tests.
- Dataset manifest, simulation provenance, and checksum inventory.
- Frozen dataset ready for telemetry compression, Behavior Tree construction, and model evaluation.

## Track 3: Behavior Tree Builder & Deterministic Divergence Function

**Owner / name:** Arik Gershman & Gili Gordiyenko

**Scope:** extend the existing parser into a batch BT pipeline, tune compression, and calculate reproducible numerical divergence. The proposal does not prescribe a formula; the alignment and metric definitions below are explicit implementation decisions to document and test.

### Target milestone dates

- [x] **Oct. 30:** run the existing parser on the sample, define batch/alignment APIs, and add synthetic transport and timing edge cases. Verification and public interfaces are recorded below.
- [x] **Nov. 6 deliverable, completed Nov. 18:** rerun nine-candidate grouped dev sweeps on 50 pairs/task and version the selected `k=5;threshold_scale=1` configuration and selection receipt. No test telemetry was used for selection.
- [x] **Nov. 13 deliverable, completed Nov. 18:** publish all 1,000 accepted pairs with computed observed-stage divergence, unavailable suffixes, compression/event-retention and raw-reference numerical-onset diagnostics. See [production evidence](docs/production-parser.md).

### Oct. 30 completion evidence and API handoff

- [x] Rebuilt the checked-in pickup pair with the existing parser and reproduced all four sample files byte for byte in `data/processed/track3-oct30/mock/`. Each raw trace has 400 rows; the BT retains 31 success and 34 failure samples, including the 5.08/5.10/5.12 s grip-drop neighborhood. Shared schemas, default configuration, and versioned samples are unchanged.
- [x] Added `build_paired_bts(metadata, data_dir, config) -> BatchBuildResult` in `parser/pipeline.py`. It accepts `Sequence[MetadataRow]` and `CompressionConfig`, reuses `build_paired_bt()`, and returns ordered `trees` and `errors` tuples. Each input pair produces exactly one tree or a `PairBuildError` containing its ID, source filenames, exception type, and message. Duplicate IDs reject the batch; an empty sequence returns empty tuples. File/validation failures do not prevent later pairs from being processed. Metadata CSV errors still raise from `read_metadata_csv()` before batch construction.
- [x] Added `align_subtask_samples(node: SubtaskBT) -> AlignedSubtask` in `parser/divergence.py`. It returns the shared progress grid, failure-run absolute timestamps, and success/failure arrays for every `TelemetryFeature`; scalar shapes are `(n,)`, vector shapes `(n, components)`. Interpolation is linear for continuous features, previous-value for gripper closure, and normalized/sign-aligned shortest-path SLERP for quaternions. A singleton stays constant over its partner's grid; two singletons compare once at progress zero. Queries within `1e-12` progress of an observed sample snap to that sample to avoid roundoff-induced transitions; the grid remains the unrounded sorted union. Inputs are not mutated, and interpolated values are comparisons rather than new observations.
- [x] Added synthetic tests for both task templates, transport semantics, seven-joint arrays, unequal stage lengths/sample counts, shifted timestamps, singleton stages, gripper transitions, quaternion sign/shortest-path behavior, malformed or missing files, and missing/repeated/reordered stages on either side. The additional completion audit on Oct. 23 passes all **97 tests with warnings treated as errors**, including the original parser/reproducibility tests, 100 seeded analytic-motion cases, raw-validation failures, and batch error ordering/preflight checks.
- [x] Completed an independent mixed-task stress check: 100 synthetic pairs (50 per task), one through nine joints, unequal stage lengths, and `k=0/1/5/100`, for 400 successful pair builds and 1,400 aligned subtasks. All trees round-trip through the schema, all aligned values are finite, and input files remain unchanged. Fresh generation reproduces all four sample files byte for byte, and the documented example passes verbatim. Fixed a small-angle quaternion approximation exposed by this audit using stable SLERP weights; differently sampled constant-angular-speed motion now agrees within numerical tolerance. Evidence is in `runs/track3-oct30/verification.json`; these synthetic parameter checks are validation, not train/dev tuning.

Both APIs and their result types are exported from `telemetry_reasoning.parser`. At the Oct. 30 handoff, batch construction was in-memory; the Nov. 6 work below adds serialization, reporting, and tuning tools. Parsing deliberately leaves observed-node divergence uncomputed; the implemented enrichment stage fills it separately. Transitioned stages use progress alignment, while the final observed stage uses common elapsed time and unobserved stages are not comparable. Observed durations are reported separately. Mapped timestamps can be interpolated times and must not be presented as exact observed onset.

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
    print(node.subtask.value, len(node.success_samples), len(node.failure_samples), len(aligned.failure_timestamps))
PY
```

Expected columns are subtask, retained success count, retained failure count, and alignment grid count: `approach 7 7 7`, `align 7 7 7`, `grasp 10 13 13`, `lift 7 7 7`.

### Historical Nov. 6 completion evidence and remaining dependencies

- [x] Implemented exported `parse_dataset()` and `scripts/parse_dataset.py`, producing schema-valid `pair_<pair_id>_bt.json` files and a separate `parse_report.json`. Reports record per-pair rejections, source/output/configuration hashes, and compression diagnostics. Staged publication and fresh-output-directory checks prevent partial or stale batches; rejected source files are left untouched.
- [x] Added complete `configs/parser.yaml`, preserving the existing defaults and byte-identical sample output. All divergence objects remain explicitly uncomputed. Settings are provisional, not calibrated on the real dataset.
- [x] Added raw-reference compression diagnostics: per-stage counts, compression ratios, fixed-reference event-center/neighborhood retention, reconstruction errors for every feature, and leakage-free JSON payload sizes in characters/UTF-8 bytes (not model tokens).
- [x] Implemented `scripts/tune_compression.py` and grouped-split validation. The sweep checks all manifest groups before opening telemetry, reads only the requested train/dev split, keeps reference-event thresholds fixed across candidates, and never changes configuration automatically. A reproducible synthetic demonstration covers both tasks, three seeds, `k=[0,2,5]`, and threshold scales `[0.5,1,2]`; this verifies the tools, not real-data calibration.
- [x] Before partial-trace support, ran the parser against all 1,000 pairs in Gili's supplied `dataset_v1.zip`: pickup **384 written / 116 rejected**, transport **226 written / 274 rejected**. All 390 rejections are incomplete failure-stage sequences. The 610 compatible pairs retain 110,283 of 448,309 raw observations and all 71,708 fixed-reference feature-event neighborhoods. All output BTs validate; archive/source hashes are unchanged. Task outputs are separate because task-local pair IDs collide.
- [x] Verified existing Track 4 baseline consumption and all **152 Track 3 tests** with warnings treated as errors at that handoff. Its full suite had **279 passed / 1 Track 4 failure**: `test_label_vocabularies_match_shared_schemas`, because the evaluator vocabulary then omitted `excessive_speed`. This is historical evidence, not current test status.
- [x] Obtain grouped task-manifest inputs and run the nine-candidate real dev sweeps (reported by the Nov. 17 Track 2 handoff); both tasks recommend `k=5;threshold_scale=1`.
- [x] Verify restored inputs, rerun the dev sweeps, and version production compression in `configs/parser-production-v1.yaml` with `configs/parser-production-v1.selection.json`. Published full conversion at `data/processed/production-v1-20261118/`; test labels remain held out from tuning and model evaluation.

Handoff and commands: `src/telemetry_reasoning/parser/README.md`. Local generated evidence: `runs/track3-nov06/verification.json`, `runs/track3-nov06/archive-full-compatibility/verification.json`, and `runs/track3-nov06/synthetic-tuning/SYNTHETIC-evidence.json` (ignored outputs, not versioned fixtures).

### Files to create/modify

- **Modify (exclusive ownership):** `src/telemetry_reasoning/parser/compressor.py`, `src/telemetry_reasoning/parser/__init__.py`, `tests/test_parser.py`.
- **Implemented; extend for later milestones:** `src/telemetry_reasoning/parser/divergence.py`, `src/telemetry_reasoning/parser/pipeline.py`.
- **Implemented:** `configs/parser.yaml`, `scripts/parse_dataset.py`, `scripts/tune_compression.py`, `src/telemetry_reasoning/parser/diagnostics.py`, `src/telemetry_reasoning/parser/tuning.py`, `tests/test_compression_diagnostics.py`, `tests/test_compression_tuning.py`, and the parser handoff README.
- **Implemented:** `configs/divergence.yaml`, `src/telemetry_reasoning/parser/divergence_metrics.py`, `scripts/enrich_divergence.py`, and `tests/test_divergence_metrics.py`.
- **Implemented:** `parser/production.py`, `parser/onset_diagnostics.py`, `scripts/produce_bts.py`, `tests/test_production_parser.py`, and `tests/test_onset_diagnostics.py`; expanded metric tests cover offsets, resampling, quaternion signs, and deterministic publication. Production selection and raw-reference numerical onset diagnostics are complete.
- Keep `configs/default.yaml` and the checked-in sample unchanged. `configs/parser.yaml` is a complete `ProjectConfig` (`simulation` plus `parser`); `parser/divergence_metrics.py` loads the separate divergence configuration.

### Input & output contracts

- Start with `read_metadata_csv("data/sample/metadata.csv")`, the paired sample CSVs, `CompressionConfig`, and existing `build_paired_bt(metadata, data_dir, config)`.
- Output: one schema-valid `PairedBehaviorTree` per input pair under `data/processed/<dataset>/bt/`, plus a separate parse/compression report. Preserve the existing nested goal/subgoal/action semantics; do not introduce an incompatible tree shape.
- Compute divergence from each node's `success_samples` and `failure_samples`; do not zip arrays by index. Fill only the existing `DeterministicDivergence` fields. Keep alignment settings and diagnostics in the report, not extra schema fields.
- Raw 50 Hz traces are the reference for evaluating information lost through compression. The LLM inputs remain the compressed BTs and their computed comparisons.

### Definition of Done

- [x] Extend in-memory batch construction for both task templates with clear per-pair errors. Preserve original timestamps, semantic metadata, structural alignment, and all existing CSV/schema validation. Keep the existing `build_paired_bt()` default behavior (uncomputed divergence). Test unequal durations/counts, missing/repeated/reordered subtasks, and invalid pair files.
- [x] Finish dataset serialization, parse/compression reporting, parser configuration, and the CLI. Preserve the existing API and sample reproducibility; explicitly report uncomputed divergence.
- [x] Add divergence as an explicit second pipeline stage using `scripts/enrich_divergence.py`, `parser/divergence_metrics.py`, and `configs/divergence.yaml`. Validate every input before staged publication into a separate fresh directory; retain source/output/config hashes and computed/unavailable counts in `divergence_report.json`.
- [x] Retain first/last rows, up to `k` interior samples, and major-event neighbors `t-1/t/t+1`; compare adjacent rows only within a consecutive subtask. Preserve strict `>` threshold semantics, sorted deduplication, clipping, and `k=0` behavior. Existing quaternion compression uses component deltas; any replacement needs explicit regression evidence. Verified by the existing parser regression suite on Oct. 23.
- [x] Implement a documented `k`/event-threshold sweep with raw-reference compression/event-retention and prompt-size measurements, preserving the mock's 5.08/5.10/5.12 s grip-drop neighborhood and grouped split isolation. The tool and synthetic demonstrations are verified; Track 2 reports completed real dev sweeps.
- [x] Publish the final production compression choice with fresh dev-sweep evidence and convert all 1,000 accepted pairs with that versioned configuration. Preserved mock defaults and held-out tuning isolation; all 2,004 generated artifacts verified against the production receipt.
- [x] Implement deterministic **within-subtask alignment**: stages followed by a transition use `u=(t-t_start)/(t_end-t_start)` and the union of retained progress values; the final observed stage uses common stage-relative elapsed time without extrapolation, including when failure reached the final expected stage. Unobserved stages are explicitly not comparable. Interpolate continuous features linearly, use previous-value closure and normalized/sign-aligned shortest-path quaternion interpolation. Never cross stage boundaries. Keep observed durations separate from sensor deltas. See the parser handoff for coordinate, singleton, and numerical-tolerance rules.
- [x] Define and implement `delta_f(u)=max_j(abs(failure_f,j(u)-success_f,j(u)))` (one component for scalars), then `max_abs_deltas[f]=max_u(delta_f(u))` for all `TelemetryFeature` values. Quaternion deltas are dimensionless component differences after sign alignment, not radians. First divergence is the earliest grid point strictly exceeding a configured per-feature threshold, mapped back to the **failure run's absolute timestamp**. Keep paired-trace thresholds separate from compression's adjacent-row thresholds; identical traces produce zero deltas and null onset.
- [x] Test the sample's **5.10 s / 12 N** grip event, threshold equality, identical traces, schema round-trip, complete threshold configuration, non-mutation, observed/unobserved enrichment states, and main/baseline prompt compatibility.
- [x] Extend metric-level tests for scalar/vector offsets, shifted/differently sampled traces, quaternion `q` versus `-q`, and deterministic publication. Raw-reference diagnostics report 1,309 matching numerical crossings, zero missed/extra crossings and maximum onset error 0 s; feature maximum-delta errors remain nonzero. Grid onset is not physical fault onset. All 259 parser tests passed with warnings treated as errors.

Current local verification commands (use fresh/empty output directories):

```bash
python -W error -m pytest tests/test_parser.py tests/test_bt_pipeline.py tests/test_partial_traces.py tests/test_divergence.py tests/test_divergence_metrics.py tests/test_compression_diagnostics.py tests/test_compression_tuning.py
python scripts/parse_dataset.py --metadata data/sample/metadata.csv --data-dir data/sample --config configs/parser.yaml --output-dir data/processed/track3-local/bt
python scripts/enrich_divergence.py --bt-dir data/processed/track3-local/bt --config configs/divergence.yaml --output-dir data/processed/track3-local/enriched-bt
```

Full-dataset conversion commands after obtaining the archived batch and selecting
the production compression configuration (run once per task; do not combine IDs):

```bash
batch=outputs/dataset/batch_20261030_152125_e5ca72de
for task in pick_up_object transport_object; do
  python scripts/parse_dataset.py --metadata "$batch/$task/metadata.csv" --data-dir "$batch/$task" --config configs/parser-production-v1.yaml --output-dir "data/processed/final/$task/bt"
  python scripts/enrich_divergence.py --bt-dir "data/processed/final/$task/bt" --config configs/divergence.yaml --output-dir "data/processed/final/$task/enriched-bt"
done
```

`parse_dataset.py` does not accept `--divergence-config`; enrichment is the
separate command above. Replace `configs/parser.yaml` with the selected
production configuration if the acceptance decision changes it. These commands
do not certify physical outcomes or supply missing release inputs.


## Track 4: Rule-Based Baseline (Baseline 1) & Evaluation Pipeline

**Owner / name:** David Simberg

**Scope:** a failure-only threshold/template baseline, shared result validation, evaluation, and Matplotlib plots. This owner assembles the final smoke test using the other tracks' public CLIs; model owners remain responsible for their own tests.

### Target milestone dates

- **Oct. 30:** implement the sample-driven detector skeleton, prediction contract, synthetic evaluation cases, and one telemetry plot.
- **Nov. 6:** complete Baseline 1, automated metrics, and deterministic failure-mode/subtask scoring for all variants.
- **Nov. 13:** integrate real BT/divergence outputs, all four model variants, visualization, and end-to-end smoke checks. The final held-out study remains Dec. 1.

### Implemented files to maintain

- `src/telemetry_reasoning/models/rule_based.py`.
- `src/telemetry_reasoning/eval/contracts.py`, `src/telemetry_reasoning/eval/metrics.py`, `src/telemetry_reasoning/eval/plots.py`.
- `configs/rules.yaml`, `configs/eval.yaml`.
- `scripts/run_baseline.py`, `scripts/evaluate.py`, `scripts/plot_telemetry.py`, `scripts/integration_smoke.py`, and `src/telemetry_reasoning/integration.py`.
- `tests/test_rule_based.py`, `tests/test_evaluation.py`, `tests/test_plots.py`, `tests/test_integration_smoke.py`.
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
- [x] Report failure-mode accuracy and macro-F1 across truth-supported classes. Retain all five schema labels in per-class/confusion output and the fixed-label macro average; the four-class release gives `excessive_speed` zero truth support, so it is excluded from the observed-class macro average. Count abstentions/missing predictions as incorrect for accuracy and missed true classes for recall. Compute optional onset error only where a defined prediction timestamp and independent reference exist; injection time and observed divergence are different events.
- [x] Verify cited evidence against actual allowed failure observations with numeric tolerances; report unsupported evidence separately. The evaluator's `--bt-dir` adapter consumes the dictionary returned by `to_failure_only()` directly.
- [x] Prepare a seeded, blinded human rubric for causal correctness, evidence support, and clarity. Prominently label stub-backend metrics synthetic; automated label accuracy does not measure explanation quality.
- [x] Preserve available runtime/token/config provenance in inference run manifests rather than adding it to the strict prediction schema.
- [x] Plot raw success/failure telemetry, retained samples, and computed divergence with subtask boundaries, units, legends, and distinct injection/onset markers. Distinguish uncomputed comparisons from unavailable suffix stages; test headless artifacts and plotted data/axes.
- [x] Verify the reusable CPU smoke against all seven public CLIs with fresh mock telemetry, rules plus three stubs, evidence-enabled evaluation, provenance, and plots. The integration suite includes a nondefault-seed run with network and optional physics/model imports blocked; see the integration gate below.
- [x] Complete the CPU real-MuJoCo producer and stored full-dataset integration gates, with explicit context-refusal and archival-certification limitations.
- [ ] Execute comparable real-HF generation and the later held-out study. Synthetic smoke success cannot establish these results.

Implement and pass (the first command is independent; the remaining commands consume Track 3's output):

```bash
python -m pytest tests/test_rule_based.py tests/test_evaluation.py tests/test_plots.py
python scripts/run_baseline.py --bt-dir data/processed/smoke/bt --config configs/rules.yaml --output runs/smoke/baseline_1.jsonl
python scripts/evaluate.py --metadata data/sample/metadata.csv --predictions runs/smoke/baseline_1.jsonl --bt-dir data/processed/smoke/bt --config configs/eval.yaml --output-dir runs/smoke/eval
python scripts/plot_telemetry.py --data-dir data/sample --bt-dir data/processed/smoke/bt --output-dir runs/smoke/plots
```

## Track 5: LLM Reasoning Harness & Prompt Baselines (Baselines 2, 3, and Main Model)

**Owner / name:** Arik Gershman

**Scope:** one reproducible Hugging Face Llama-3.1-8B inference harness, three controlled prompt variants, reviewed-preference formatting, and an optional GPU training launcher. Actual DPO/LoRA optimization and adapter evaluation require reviewed annotations and GPU execution.

### Target milestone dates

- [x] **Oct. 30:** create the model adapter, mock inference backend, prompt views, and a compute/access smoke plan. Implemented in `src/telemetry_reasoning/models/llm.py`, `src/telemetry_reasoning/models/prompts.py`, `configs/llm.yaml`, the three prompt templates, and `scripts/run_llm.py`; see the [Track 5 handoff and smoke plan](src/telemetry_reasoning/models/README.md).
- [x] **Nov. 6 — complete, verified Nov. 16:** real HF Baselines 2 and 3 ran on the Alienware with identical model/settings. CPU-only revalidation under the explicit `json_or_fenced_v1` policy accepts both exact saved responses with validated evidence; the original strict rejection, raw text, and generation provenance remain unchanged. Parser, leakage, resource, and replay checks pass. See the [acceptance receipt and limitations](docs/track5-nov06-acceptance.md). No further GPU work is needed for this milestone.
- [ ] **Nov. 13:** run the main paired-BT-plus-divergence variant, deliver comparable four-variant evaluation inputs with Track 4, and validate the DPO/LoRA data formatter.

Oct. 30 verification (CPU/offline; fresh output directory):

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_parser.py tests/test_prompts.py tests/test_llm.py
PYTHONPATH=src .venv/bin/python scripts/run_llm.py --bt-dir data/sample --config configs/llm.yaml --variants baseline_2 baseline_3 --backend stub --output-dir runs/track5-oct30/baselines
PYTHONPATH=src .venv/bin/python scripts/run_llm.py --config configs/llm.yaml --check-access
```

The handoff documents a separate synthetic enriched fixture for testing all three variants without altering `data/sample/`. Stub predictions deliberately abstain and are explicitly marked synthetic in their run manifest; they are not HF results. The Nov. 6 adapter adds batched greedy decoding, exact raw-response retention, independently verified output acceptance, explicit context-budget rejection, and memory admission checks. Nov. 6 real-baseline acceptance is complete under the documented response policy. Formatting and CPU preparation are complete; new main-model acceptance requires GPU generation.

Nov. 6 safe local verification: **284 parser/Track 5 tests passed**, with network access disabled and imports of actual model runtimes blocked. A fresh two-baseline stub smoke produced two explicit abstentions and no errors in `runs/track5-nov06/safe-stub/`. `runs/track5-nov06/safe-local-preflight.json` records the actual 16 GiB host refusal without importing Torch or initializing a device. These are CPU contract checks; they do not satisfy real-HF acceptance. That historical full-suite receipt had one Track 4 failure concerning `excessive_speed`; it is not the current suite result.

Nov. 16 completion verification: **367 parser/Track 5 tests passed**, including 54 CPU-only offline replay tests with model/GPU runtime imports blocked. `runs/track5-nov06/accepted-baselines/run_manifest.json` reports two validated real-HF responses, `nov6_baselines_covered: true`, and zero newly generated requests during replay. Original strict outputs remain in `runs/alienware-import-20261116T215259Z/`. Both explanations' causal limitations are recorded separately; one mock pair is not a dataset-wide accuracy or explanation-quality claim.

### Files to create/modify

- **Implemented:** `src/telemetry_reasoning/models/llm.py`, `src/telemetry_reasoning/models/prompts.py`, and `scripts/run_llm.py`.
- **Implemented:** `configs/llm.yaml`, `configs/prompts/failure_only.txt`, `configs/prompts/paired_no_divergence.txt`, `configs/prompts/paired_with_divergence.txt`.
- **Implemented:** `configs/llm-local.yaml` and `configs/llm-local.requirements.txt` record the experimental Mac configuration and dependency versions; the profile is refused on the 16 GB laptop.
- **Implemented:** `tests/test_llm.py`, `tests/test_prompts.py`, and `tests/test_llm_cli.py` for atomic publication, raw-response provenance, real/stub acceptance, and resource-refusal checks.
- **Implemented:** `src/telemetry_reasoning/models/training_data.py`, `scripts/prepare_dpo_data.py`, and `tests/test_training_data.py`; source-bound conversational DPO/SFT exports, reviewed/synthetic provenance, split protection, and verification. See [training data](docs/training-data.md).
- **Implemented:** `models/training.py`, `scripts/train_adapter.py`, dedicated GPU config/requirements, CPU checks and CUDA LoRA/DPO launch guards. [GPU training](docs/gpu-training.md) remains contingent on reviewed annotations and actual optimizer execution.
- **Implemented:** `src/telemetry_reasoning/models/revalidation.py`, `scripts/revalidate_llm.py`, and `tests/test_llm_revalidation.py` for CPU-only, provenance-checked revalidation into separate derived artifacts.
- **Completion evidence:** `docs/track5-nov06-acceptance.md` records the real generation, response policies, artifact hashes, verification, and explanation limitations.
- Read/reuse the shared view methods and prediction contract. Do not modify `models/rule_based.py`, evaluator files, or shared schema/default configuration.

### Input & output contracts

- Start from `PairedBehaviorTree.model_validate_json()` on `data/sample/pair_000_bt.json`. Baseline 2 uses `to_failure_only()`; Baseline 3 uses `to_paired_without_divergence()`; main uses `to_full_paired()` after checking that divergence has been computed.
- For independent main-model tests, deep-copy the fixture and attach explicitly synthetic `DeterministicDivergence` objects, including the 5.10 s / 12 N grasp example and computed-zero unchanged stages. Leave the source sample untouched. Production main-model inference must reject null divergence objects rather than treating them as zero.
- Output: `baseline_2.jsonl`, `baseline_3.jsonl`, and `main.jsonl` matching Track 4's prediction contract, plus a run manifest with model/tokenizer revision, prompt/config hashes, decoding settings, seed, latency, and token counts.
- DPO input: curated JSONL records containing `pair_id`, `model_variant`, `chosen`, `rejected`, and preference provenance. Formatter joins each record to its leakage-free BT view and emits `prompt`, `chosen`, `rejected`; keep split/provenance in a sidecar. Synthetic preferences may test the formatter but must not be presented as reviewed training labels.

### Definition of Done

- [x] Configure an exact Hugging Face Llama-3.1-8B checkpoint and revision, tokenizer, device/dtype, context budget, and optional supported quantization. Record whether the base or instruction-tuned checkpoint is used and use the same choice for all variants. Check access/compute early; provide a stub backend so CPU unit tests never download weights or require credentials. Configuration, access, and real CUDA/BF16 baseline execution are verified; the resource guard still rejects full-model loading on the 16 GiB Mac.
- [x] Implement consistent tokenization, chat formatting where applicable, batched inference, seeded decoding, output limits, and validated structured predictions. Handle overlong prompts explicitly and consistently; do not silently truncate task stages or divergence evidence. Capture invalid outputs as errors without inventing labels.
- [x] Design the requested Chain-of-Thought-style prompts around examining task expectations, observable deviations, and plausible causes; request a final diagnosis with a concise evidence-grounded rationale and the shared JSON fields. Keep the instruction/output format and decoding settings fixed across variants so the experimental difference is available input information.
- [x] Test prompt payloads recursively: Baseline 2 contains no success/divergence; Baseline 3 contains success but no divergence; main contains both. No variant sees ground-truth metadata or identifies a cause from filenames/IDs. Verify null-divergence rejection, known fixture formatting, malformed model responses, and no mutation of storage objects.
- [x] **Nov. 6:** demonstrate real HF inference for both prompt baselines and retain reproducible manifests, exact raw text, and validated JSONL. The accepted derived run explicitly records its new response policy and the preserved original strict failure; no stub establishes HF acceptance. Record causal explanation defects separately from syntax, labels, and structured evidence.
- [ ] **Nov. 13:** demonstrate real HF inference for the main paired-BT-plus-divergence variant and deliver comparable three-variant outputs. Describe causal evidence rather than attributing every later symptom to a separate injected fault.
- [x] Implement the training formatter with nonempty/different chosen/rejected validation, matching variant/pair IDs, reviewed-preference provenance, evidence checks, and initial-condition split preservation. Identical inference messages feed DPO and chosen-only SFT exports. The verified 24-record CPU demonstration uses real production BTs but explicitly synthetic preferences; it cannot establish reviewed-data or optimizer acceptance. Formatter tests: 34 passed.
- [x] Prepare and test the reviewed-data-only CUDA LoRA/DPO launcher, strict runtime/configuration profile, optional CPU tokenizer check, no-truncation guards, and success/failure provenance. Actual CPU CLI check refuses the synthetic export without importing model runtimes or creating output. Training/formatter tests: 57 passed.
- [ ] Obtain reviewed train/dev preferences, execute GPU optimization and evaluate adapters. Synthetic preferences must remain ineligible; no held-out test data may enter training or validation.

Current CPU verification and inference commands (use an enriched fixture for main):

```bash
python -m pytest tests/test_llm.py tests/test_prompts.py tests/test_llm_cli.py tests/test_llm_revalidation.py
python scripts/run_llm.py --bt-dir data/processed/smoke/bt --config configs/llm.yaml --variants baseline_2 baseline_3 main --backend stub --output-dir runs/smoke
```

Real-model acceptance requires model access and suitable compute:

```bash
python -m pip install -e '.[llm]'
python scripts/run_llm.py --bt-dir data/processed/smoke/bt --config configs/llm.yaml --variants baseline_2 baseline_3 main --backend hf --output-dir runs/hf-smoke
```

Implemented formatter acceptance commands (use reviewed preferences and a fresh output):

```bash
python -m pytest tests/test_training_data.py
python scripts/prepare_dpo_data.py --bt-dir data/processed/production-v1-20261118 --preferences data/processed/preferences/reviewed.jsonl --manifest runs/dataset-v1-20261118/handoff/dataset_manifest.json --config configs/training-data.yaml --output-dir data/processed/dpo
```
Use task-scoped enriched BT directories under the production root. The frozen full manifest supplies task/initial-condition groups; no test preferences are accepted.


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

- [x] Assign owners for all five tracks (owner fields above are filled).

The following are ongoing collaboration practices, not unfinished feature
deliverables. Source inspection does not certify historical planning commits,
rebases, or reviewer approval:

- Coordinate ownership before branching and keep progress in the PR. Cross-track integration fixes require affected consumers' contract review.
- Treat `schemas.py`, `io.py`, `config.py`, `configs/default.yaml`, `scripts/generate_mock_pair.py`, and `data/sample/` as shared contracts. Coordinate changes and preserve regression evidence; keep additional fixtures in owning tests or temporary directories.
- Coordinate essential shared contract/dependency/README changes rather than making competing copies. Existing `.[sim]` and `.[llm]` extras provide optional dependencies.
- Keep generated data in ignored `data/raw/` and `data/processed/`, reports in ignored `runs/`, and weights outside tracked source. Include lightweight fixtures, config/provenance, and verification commands in each PR.
- Rebase before merging and run the parser, affected track, and cross-track acceptance tests. Each PR needs another member's review of contracts and evidence.

### End-to-end integration after Tracks 1-5 merge

Track 4 owns the reusable `src/telemetry_reasoning/integration.py`, its thin
`scripts/integration_smoke.py` CLI, and `tests/test_integration_smoke.py`; each
owner fixes defects within their own modules.

- [x] **CPU contract smoke:** generate a fresh mock into the requested output directory, validate its CSVs, build BTs, compute divergence, run Baseline 1 plus stubbed Baselines 2/3/main, validate predictions, score them, and render plots. Use explicit file lists so earlier runs/manifests cannot pollute evaluation. Assert the known grasp event (5.10 s, 12 N), all three view exclusions, one result per variant/pair, finite metric outputs, and nonempty plot artifacts. Verify that retained observations match newly generated CSVs and prompt hashes match the executed inference manifest; retain regression cases for rejected/corrupted artifacts and failure receipts.
- [x] **Real producer CPU smoke:** run 18 fresh accepted configurations across both tasks and four released modes, with 76 repeated executions, independent physical goals, strict telemetry replay, source/model fingerprints, and two excessive-speed activation probes. Thirteen cases have complete observed pre-injection equality; five have no prior stored sample and stay explicitly unavailable. Public parser/baseline/evaluator/plot stages verified 72 outputs and 31 PNGs. Exact-token context negatives are explicit: 44 stub abstentions and 10 refusals across 54 requests. Receipt: `runs/integration-mujoco-cpu-20261118/integration_manifest.json`, status `passed_with_context_refusals`; producer acceptance passed, model completion false.
- [ ] **Real model smoke:** on a machine with model access/compute, run the three HF variants on the same parsed real pairs and the rule baseline on their failure-only views. Validate allowed inputs and JSONL outputs before evaluating. Keep real and stub results in separate run directories.
- [x] **Stored full dataset gate:** validate 500 pairs/task, 50 conditions/task, and 125 pairs/mode/task across four represented modes; publish 1,000 enriched BTs with 700/100/200 grouped splits and verified checksums/configuration. All 101,357 fixed-reference event neighborhoods retained. Archived physical/provenance certification remains explicitly failed/pending; no held-out model results were generated.
- [x] **CPU integration evidence:** retain stage logs and `integration_manifest.json` with status, known-event/view/evidence/metric/plot assertions, and artifact hashes from a fresh CPU run. The Nov. 18 receipt is at `runs/integration-cpu-20261118-foundation/integration_manifest.json` (ignored generated output).
- [ ] **Release integration evidence:** attach real dataset counts, production parser/compression diagnostics, real four-variant validation, plot examples, and remaining hardware-dependent checks to the integration PR. Never mark real-MuJoCo or HF checks passed because a stub test passed.
- [x] **CPU GPU-workflow rehearsal:** prepare immutable, exact-tokenized bundles and exercise all four variants using synthetic LLM responses. Current fresh physical smoke: eight pairs/24 fitting requests; complete fresh batch: 18 pairs/44 fitting requests and 10 context refusals. Optional archived dev: 100 pairs/261 fitting requests and 39 refusals. All requests remain accounted for. GPU `--inference-only` defers metrics, review sheets, and plots to the independently validating Mac evaluator. See [GPU commands and limits](docs/gpu-handoff.md) and [exact Ubuntu 24.04 setup/transfer commands](docs/alienware-ubuntu24.md).

Implemented CPU integration commands (fresh absent/empty output directory):

```bash
python -m pytest
python scripts/integration_smoke.py --backend stub --sim-backend mock --output-dir runs/integration-cpu
python -m pytest tests/test_integration_smoke.py
```

### Earlier Nov. 18 CPU foundation acceptance receipt

The current full suite passed **826 tests with 21 skips** in 33.54 s. Fifteen
skips require the absent archived release, and six are opt-in real-physics
checks. Verification command (use a fresh temporary base directory on rerun):

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp /tmp/telemetry-foundation-full-suite
```

The persisted run at `runs/integration-cpu-20261118-foundation/` passed all
seven public CLI stages at seed 723. Its receipt records four validated
predictions, 10 nonempty PNG plots, supported rule evidence, executed-request
prompt/template hashes matching the verified model views, and unchanged
generated inputs, source/configuration files, and versioned samples. All three
stub variants abstain explicitly and remain labeled synthetic. The focused
integration suite passed 16 tests, including a nondefault-seed public-CLI run
with network and optional physics/model imports blocked, plus invalid-output
and failure-receipt regressions. Partial-failure plotting and excessive-speed
decimal-roundoff regressions are included in the passing full suite. The
telemetry PNG from the persisted run was also visually reviewed.

To repeat the documented gate, choose a new absent/empty output directory:

```bash
python scripts/integration_smoke.py --backend stub --sim-backend mock --seed 723 --output-dir runs/integration-cpu-repeat
```

The smoke supports `--sim-backend mock --backend stub` and the separate
`--sim-backend mujoco --backend stub` physical-producer gate. Its optional
`--parser-config` and `--stub-tokenizer-dir` flags select the versioned parser
and exact pinned CPU tokenizer. Real HF orchestration is implemented separately
by `scripts/gpu_experiment.py`; see [the GPU handoff](docs/gpu-handoff.md).

Default `python -m pytest` must remain CPU/offline compatible: skip explicitly marked hardware/model tests when their resources are absent and report those skips. Passing that command is necessary, but the separate real producer/model smoke evidence is also required for the corresponding track to be complete.

### Nov. 18 remaining CPU work acceptance

- Current full suite: **1,169 passed, 12 skipped** in 62.18 s. The skips are opt-in native cases; the separate native run passed **57 tests** in 39.03 s. New portable fresh-input verification, inference-only orchestration and CPU returned-result validation are included in the full suite.
- All 1,000 enriched BTs and 2,004 production artifacts verified; eight production telemetry plot examples were rendered and pickup/transport figures visually inspected.
- The real-producer gate passed 18 fresh physical pairs/76 executions, with explicit observed-prehistory coverage and 10 exact context refusals. Its rule baseline achieved only **1/18 joint accuracy**; pipeline acceptance does not establish diagnostic quality.
- Current source-bound bundles are `runs/alienware-fresh-smoke-20261118`, `runs/alienware-fresh-all-20261118`, and `runs/alienware-v1-dev-20261118`. Their separate GPU-generation/Mac-evaluation CPU rehearsals preserve all requests and exact token counts. No new HF completions or optimizer steps were generated locally. Earlier source-bound bundles remain historical and must not be executed with changed source.
- Training formatting and the optional GPU launcher passed CPU tests. The actual synthetic-export training check refused cleanly, with no model runtime imports/output creation. Reviewed train/dev annotations, GPU optimization and adapter evaluation remain open.
- Earlier consolidated evidence: `runs/cpu-completion-20261118/verification.json`. Current portable acceptance and transfer inventory: `runs/alienware-handoff-20261118/verification.json` and `transfer_receipt.json`. The source/input archive includes uncommitted implementation files, since cloning alone cannot restore them. GPU execution and context limits: [GPU handoff](docs/gpu-handoff.md). Original v1 certification is deferred as requested, without changing archived bytes or its failed/pending gates.

### Nov. 19 Alienware smoke recovery

- The first real fresh smoke loaded weights and generated responses, but failed on both strict JSON responses and later CUDA resource refusals. Real-model acceptance remains open; no CUDA OOM was shown in the supplied logs.
- Fixed a reproducible resident CUDA guard false refusal caused by inactive allocator cache. On insufficient driver-free memory, release unused cache once, remeasure, and apply the unchanged KV/workspace threshold. Preserve cleanup history in provenance; no model setting or automatic generation retry changes. The separately versioned production v2 config accepts a whole-response JSON fence while retaining strict schema/evidence checks.
- Current full CPU suite: **1,200 passed, 12 skipped** in 62.29 s, including cache/recovery and independently verified response-error scoring regressions. Current-source eight-pair inference-only and Mac-evaluation stub rehearsal passed; it remains synthetic.
- Refrozen all three source-bound bundles as `runs/alienware-fresh-smoke-v2-20261119`, `runs/alienware-fresh-all-v2-20261119`, and `runs/alienware-v1-dev-v2-20261119`, preserving selected pairs and token/refusal counts. Earlier bundles and full transfer archive remain historical.
- [Incremental patch and exact retry commands](docs/alienware-inference-fix-20261119.md). The pasted raw text confirms JSON fences. One of three transcribed payloads passes evidence checks; two remain rejected. Opt-in Mac scoring retains reproducible response errors without granting acceptance. Alienware confirmation of the memory fix and accepted real-model results remain pending.

### Nov. 19 v3 execution and output recovery

- The later v2 retry showed genuine CUDA OOMs on approximately 54–56K-token prompts, alongside invalid citations and responses truncated at 512 tokens. The earlier no-OOM observation above applies only to the first smoke.
- Implemented a separately versioned v3 execution profile: native prefill chunks of at most 2,048 tokens, dynamic KV cache, disabled compilation, explicit fused-only SDPA, and bounded CPU-only failure diagnostics with finished traceback-frame release. No safety threshold was lowered and no automatic inference retries were added.
- Implemented `observed_json_v1`: all five canonical modes or model-selected abstention, observed failure stages, one or two model-selected exact failure-side citations, bounded printable ASCII explanation, complete bare JSON and EOS. Schema/evidence checks stay strict. This changes the inference protocol; compare all three prompted variants under v3 and retain v1/v2 outputs as historical evidence.
- The user's RTX 5090 passed all four fused-kernel probe cases with math attention disabled. Those checks did not load model weights. A new source/input/device-bound longest-request qualification must pass before each full v3 bundle runs; actual checkpoint qualification and accepted four-variant real-model evidence remain open.
- New bundles are `runs/alienware-fresh-smoke-v3-20261119`, `runs/alienware-fresh-all-v3-20261119`, and `runs/alienware-v1-dev-v3-20261119`. Existing context refusals stay accounted for. Run only the eight-pair/24-request smoke first, then transfer results for CPU evaluation on the Mac. [Exact v3 commands](docs/alienware-v3-recovery-20261119.md).
- Real-model smoke, final integration review, reviewed training annotations, original v1 certification and the held-out study remain unchecked. Formatting/citation guarantees do not establish diagnostic accuracy or truthful prose.
