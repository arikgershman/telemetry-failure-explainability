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
scene. The release information establishes a source reference. The exact
working tree and calibration inputs used to generate the ZIP still need
producer confirmation.

All 1,000 pairs passed the acceptance gate and batch parser, with zero
rejections. The checksum verification passed for 2,006 files. The tuner
ran nine candidates on 50 dev pairs per task and recommended
`k=5;threshold_scale=1` for both tasks. The archived
activation timestamp is stored as `injection_timestamp_s`, with
`injection_time_basis` set to `absolute_simulator_time`. The evaluator's
`injection_timestamp` field requires a time relative to the failure trace, so
these task manifests omit that field and leave onset scoring unavailable.

The archived batch still needs the producer's exact generator, model, and
configuration revisions, per-run task time origins, and machine-readable
terminal outcome measurements. Those are required for the full pre-injection
trajectory and independent physical outcome checks and final release
certification. The manifest fingerprints the archived bytes and records the
validator's current provenance. Windows Application Control blocks the native
MuJoCo import in the current environment. The runner hook contract remains
an open interface decision with Track 1.
