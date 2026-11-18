# Track 2 dataset handoff

Use the archived MuJoCo batch at `outputs/dataset/batch_20261030_152125_e5ca72de`.
It contains 500 pairs per task, with 125 pairs per task for each of the four
calibrated modes. The release excludes `excessive_speed` because calibration
showed insufficient failure reliability. Keep the original task directories
and CSVs intact.

| Task | Metadata relative to batch | Task manifest relative to batch |
| --- | --- | --- |
| Pickup | `pick_up_object/metadata.csv` | `handoff/pick_up_object.json` |
| Transport | `transport_object/metadata.csv` | `handoff/transport_object.json` |

The batch-level `manifest.json` is the SHA-256 inventory and validation
record. Use the corresponding task manifest for tuning or evaluation. Each
task manifest covers every ID in that task's metadata. Use `(task, pair_id)`
when joining across tasks and keep parser output, predictions, and evaluation
runs in separate task directories. IDs overlap across tasks.

Each task has 350 train, 50 dev, and 100 test pairs, grouped into 35, 5, and 10
initial conditions respectively. Every split contains all four released failure
modes for its task. Use dev for compression tuning and keep test held out.

## Generate and verify the handoff

From the repository root in PowerShell, with the full batch and the project
environment available:

```powershell
$python = '.\.venv\Scripts\python.exe'
$batch = 'outputs/dataset/batch_20261030_152125_e5ca72de'
& $python -m telemetry_reasoning.sim.freeze_robot_dataset $batch --output "$batch/manifest.json" --task-manifests-dir "$batch/handoff"
& $python -m telemetry_reasoning.sim.freeze_robot_dataset $batch --verify "$batch/manifest.json"
```

For onset scoring, download `pickup_diagnostics.clock_aligned.jsonl` and
`transport_diagnostics.clock_aligned.jsonl` from the
[`dataset-v1` release](https://github.com/arikgershman/telemetry-failure-explainability/releases/tag/dataset-v1)
into `runs/`. Add `--clock-sidecars runs/pickup_diagnostics.clock_aligned.jsonl runs/transport_diagnostics.clock_aligned.jsonl`
to the manifest generation command. The importer joins records by task and
pair ID, checks the recovered clock arithmetic against the archived activation
times, and records the sidecar hashes in the batch manifest.

The generator rechecks all 1,000 pairs before writing the inventory and task
manifests. The inventory covers the files under the original task directories.
The generated files live alongside those directories. The root-level
`dataset_manifest.json` and `dataset_splits.json` are older snapshots.
The batch and generated BT directories are Git-ignored. Teammates need a
copy of the archived batch to regenerate or verify the handoff.

## Track 3 parser and compression inputs

The parser takes a task's metadata CSV and its parent directory. The tuner
also takes the matching task manifest. Run the commands with fresh BT output
directories. The parser output contains uncomputed divergence on observed
nodes and unavailable comparisons on unobserved nodes.

```powershell
foreach ($task in 'pick_up_object', 'transport_object') {
    $source = "$batch/$task"
    $output = "data/processed/track2-handoff/$task"
    & $python scripts/parse_dataset.py --metadata "$source/metadata.csv" --data-dir $source --config configs/parser.yaml --output-dir "$output/bt"
    & $python scripts/tune_compression.py --metadata "$source/metadata.csv" --data-dir $source --config configs/parser.yaml --manifest "$batch/handoff/$task.json" --split dev --output "$output/compression-dev.json"
}
```

Track 4 can pass the same task manifest to `scripts/evaluate.py` with
`--split dev` or `--split test`. For each evaluation, use the metadata and
predictions for that manifest's task. Track 5 should keep pair identity and
split assignments tied to the same task when preparing its downstream inputs.

## Verification scope

GitHub's MuJoCo Telemetry Dataset v1 release lists `dataset_v1.zip` and
generation seed 723. The local `dataset-v1` tag resolves to
`73eeb975c2593e7fe8ad7a8a31a6253693303784`. That commit pins MuJoCo Menagerie
to `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` and includes the generator and
scene. The Track 2 owner confirmed that the ZIP was generated from this
commit with its pinned model and no local changes. The generator is
`src/telemetry_reasoning/sim/dataset.py`. The producer supplied this command:

```bash
python -m telemetry_reasoning.sim.dataset --task all --pairs-per-task 500 --seed 723 --spread 0.015 --calibration outputs/calibration/batch_20261030_144101_38130aff/pick_up_object/recommended_configs.csv outputs/calibration/batch_20261030_144101_38130aff/transport_object/recommended_configs.csv
```

The command produced `outputs/dataset/batch_20261030_152125_e5ca72de/` with
500 pairs per task. The command records both calibration paths. The input
files are unavailable in the validation workspace, so their contents and
checksums could not be verified here.

All 1,000 pairs passed the acceptance gate and batch parser, with zero
rejections. The checksum verification passed for 2,006 files. The tuner
ran nine candidates on 50 dev pairs per task and recommended
`k=5;threshold_scale=1` for both tasks. The archived
activation timestamp is stored as `injection_timestamp_s`, with
`injection_time_basis` set to `absolute_simulator_time`. With the sidecars,
the task manifests also contain the evaluator's `injection_timestamp`,
copied from `injected.failure.first_activation_csv_s` on the failure CSV
timeline. Track 1 recovered the origins from verified nominal replay and
applied them to the paired failure runs. Track 2 compares the archived
samples strictly before activation, with a 1 microsecond cutoff tolerance
and an absolute numeric comparison tolerance of `1e-8`.
All 91,976 comparable samples agreed across 777 pairs. The other 223
pairs activated at CSV time zero and had no earlier samples to compare.
The evaluator loaded all 500 onset timestamps from each task manifest.

The independent sampled-outcome audit found that all 1,000 nominal traces
met their object goal at the last observation and all 1,000 injected traces
missed it. Pickup uses object rise from the first sample. Transport uses
the first sample's XY position plus the 0.15 m commanded displacement and
the controller's target height. Run the audit from the repository root:

```powershell
New-Item -ItemType Directory -Path runs -Force | Out-Null
& $python -m telemetry_reasoning.sim.audit_observed_outcomes $batch --output runs/track2-observed-outcomes.json
```

The report records the observed endpoint measurements per pair. The controller's
terminal decision can occur between samples, and pickup also checks end-effector
error.

The sampled-outcome audit closes the v1 failure-outcome check. Exact
termination-step verification is unavailable for v1.

The manifest fingerprints the archived bytes and records the validator's
provenance. Windows Application Control blocks the native
MuJoCo import in the current environment. The agreed runner interface
accepts an optional `FailureConfig` and returns `EpisodeResult`.
