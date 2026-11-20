# Track 2 dataset handoff

The archived MuJoCo batch is
`outputs/dataset/batch_20261030_152125_e5ca72de`. Restore the original assets
with the pinned [release workflow](dataset-release.md); keep published clock
assets and recovered inputs separate from the original task directories. V1
contains 500 pairs per task, with 125 pairs per task for each of four calibrated
modes. `excessive_speed` is a valid schema/prediction mode with no accepted v1
examples, following the producer's reported reliability exclusion.

The original task directories and CSVs are immutable inputs. Use `(task,
pair_id)` when joining across tasks: pair IDs overlap. Keep parser output,
predictions, and evaluations in separate task directories.

| Task | Original metadata relative to batch | New task manifest relative to handoff |
| --- | --- | --- |
| Pickup | `pick_up_object/metadata.csv` | `tasks/pick_up_object.json` |
| Transport | `transport_object/metadata.csv` | `tasks/transport_object.json` |

## Validate and publish a handoff

After installing the project, restore v1 into a fresh dataset root:

```bash
python scripts/restore_dataset.py --download \
  --asset-dir data/raw/releases/dataset-v1 --dataset-root outputs/dataset
```

Omit `--download` when the three pinned assets are already cached. Asset size
and SHA-256 checks run before extraction. A restore receipt and copied clock
assets live under `<batch>/provenance/`; original task directories retain their
2,006-file inventory. The recovered catalogue has nine accepted configurations
per task and does not substitute for original calibration records.

Create a fresh, complete handoff directory, then verify it:

```bash
batch=outputs/dataset/batch_20261030_152125_e5ca72de
handoff=runs/dataset-v1-20261118/handoff
python -m telemetry_reasoning.sim.freeze_robot_dataset "$batch" \
  --evidence runs/dataset-v1-replay-20261118-final/evidence.json \
  --handoff-dir "$handoff"
python -m telemetry_reasoning.sim.freeze_robot_dataset "$batch" \
  --verify "$handoff/dataset_manifest.json"
```

The freezer validates all pairs before publication, checks the source inventory
again afterward, generates deterministic initial-condition-grouped splits, and
atomically publishes `dataset_manifest.json`, `dataset_splits.json`, and both
`tasks/<task>.json` files. Existing handoff destinations are refused. Use a fresh
path for each revision. Omit `--evidence` when no separately generated replay
receipt is available; the remaining gates then stay explicitly pending. The manifest records source bytes, validator fingerprints,
working-tree status, per-pair checks, and certification gate coverage.

Parser acceptance and complete release certification have distinct statuses.
An accepted release with missing evidence is
`release_status="parser_ready_pending_certification"`. A known failed gate uses
`release_status="parser_ready_certification_failed"`; this is the actual status
of the Nov. 18 canonical handoff because archived replay failed. Both states
preserve accepted parser inputs and explicit certification limits. Add
`--require-certified` to freeze or verification when all certification gates
must pass; incomplete evidence is refused. Source-bound timing/replay/producer
evidence may be provided with `--evidence <receipt.json>` during freezing.
The root-level v1 `dataset_manifest.json`, `dataset_splits.json`, and
`dataset_audit_report.json` remain historical inputs. The verifier understands
legacy Windows inventory separators; newly published paths use POSIX separators.

Each task has 350 train, 50 dev, and 100 test pairs, grouped into 35, 5, and 10
initial conditions. Every split contains all four released failure modes.
Checksums and grouped assignments are rechecked during v2 manifest verification.
Use dev for compression tuning and preserve test for the final study.

## Parser and evaluation inputs

Run commands with fresh output directories. Parsing creates uncomputed
divergence on observed nodes and unavailable comparisons on unobserved nodes;
run the enrichment stage separately to compute deterministic divergence.

```bash
for task in pick_up_object transport_object; do
  source="$batch/$task"
  output="data/processed/track2-handoff/$task"
  python scripts/parse_dataset.py --metadata "$source/metadata.csv" \
    --data-dir "$source" --config configs/parser.yaml --output-dir "$output/bt"
  python scripts/tune_compression.py --metadata "$source/metadata.csv" \
    --data-dir "$source" --config configs/parser.yaml \
    --manifest "$handoff/tasks/$task.json" --split dev \
    --output "$output/compression-dev.json"
done
```

Track 4 passes the matching task manifest to `scripts/evaluate.py` with
`--split dev` or `--split test`. Predictions and metadata must belong to the
same task. Track 5 must preserve task identity and initial-condition groups in
training inputs. Never merge task-local IDs or use ground-truth sidecars in
prediction views/prompts.

## Verification scope and remaining producer evidence

Restoration on Nov. 18 verified the ZIP and both published clock assets against
pinned GitHub release digests, preserved all 2,006 original task files, and
recovered nine accepted configurations per task. The release tag points to
`73eeb975c2593e7fe8ad7a8a31a6253693303784`, referencing Menagerie
`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`. This is a source reference;
it does not establish the producer's dirty working tree, native runtime/model
bytes, generation configuration, or original calibration inputs.

Historical acceptance and parser receipts report all 1,000 pairs passing, zero
rejections, and 2,006 matching file hashes. Historical compression sweeps tested
nine candidates on 50 dev pairs per task and recommended
`k=5;threshold_scale=1`. The fresh Nov. 18 dev-only reruns selected and versioned
those settings, then published all 1,000 enriched BTs. See the
[production handoff](production-parser.md) for exact source/configuration/output
hashes and numerical diagnostics; that publication preserves archival limits.

Archived `first_activation_s`/`injection_timestamp_s` values use absolute
simulator time; CSV time starts at canonical task onset. Published clock
sidecars are retained as unverified release inputs. A nominal-only replay cannot
establish the origin of an injected episode. The validator admits a trace-relative
`injection_timestamp` into a task manifest only after verification of that
failure run's origin. Without it, onset scoring stays unavailable.

The validator distinguishes trace integrity/parser compatibility, independent
terminal physical outcomes, complete observed pre-injection agreement,
deterministic per-run replay, and exact producer provenance. Controller status
strings alone do not certify physical outcomes. The last 50 Hz CSV observation
does not establish exact terminal state. Complete pre-injection agreement covers
every stored pre-activation row at `atol=1e-8, rtol=0`; it does not claim coverage
of every 500 Hz physics step. A missing or empty pre-activation prefix remains
pending rather than passing vacuously.

The runner now accepts optional `control_hook` alongside `FailureConfig`,
supports attributes and tuple unpacking of `EpisodeResult`, and records task
origins, terminal measurements, independent objectives, and actual runtime/source/
model fingerprints for new runs. Current native acceptance passed
57 tests, including fresh reproducibility, exact identity traces for
both tasks, and physical command/outcome checks, with MuJoCo 3.3.7 and restored
Panda assets. The scene-default cube free-joint initialization fix applies to
new runs; current behavior does not prove historical generation provenance.

The pinned Panda restoration verified 80 Git blobs and recorded them in
`data/raw/model-assets/panda-f3b60136.json`. This restores the referenced model;
it does not prove which model bytes the original producer loaded. To repeat:

```bash
python scripts/restore_model.py --receipt data/raw/model-assets/panda-f3b60136.json
python scripts/recover_clocks.py --batch "$batch" \
  --output-dir runs/dataset-v1-replay-20261118-final --representative
```

Use fresh receipt/output paths on repeat runs. The hardened recovery compares
both archived nominal and injected traces independently, with repeated fresh
runs at `atol=1e-8, rtol=0`, and retains source-bound evidence without modifying
original diagnostics. The Nov. 18 representative attempt covered eight pairs,
32 actual fresh executions, and 16 archived run records. All 16 fresh
repeatability checks passed, while every archived record mismatched the current runtime's replay. Its receipt at
`runs/dataset-v1-replay-20261118-final/evidence.json` records the mismatch; candidate
origins remain unverified. Fresh deterministic replay success does not imply
matching an archived trajectory. The published clocks therefore remain
unverified, and this attempt cannot supply archived terminal-state certification
or enable archived onset scoring.

The canonical handoff at `runs/dataset-v1-20261118/handoff` was published and
verified: 1,000 accepted pairs, 2,006 source files, and 700/100/200 grouped
train/dev/test assignments. Its actual status is
`parser_ready_certification_failed`: archival replay is a known failed gate;
archived origins, terminal physical outcomes, and exact producer provenance
remain pending. `--require-certified` refuses this handoff. The optional current
validation profile `configs/sim-validation.requirements.txt` pins MuJoCo 3.3.7
and NumPy 2.5.4 and does not identify the original producer runtime.

Original producer calibration records and exact source/model/runtime evidence
remain necessary to close archival certification. Use the manifest's explicit
gate statuses to determine what is available; freezing usable inputs does not
silently close those gates. See the [release evidence matrix](dataset-release.md)
for required evidence and limitations.
