# Production parser v1

The Nov. 18 CPU production gate applies a versioned compression configuration
to every stored v1 pair and enriches all observed stages with deterministic
divergence. It preserves unavailable failure suffixes and the archive's
`parser_ready_certification_failed` status. Parser publication does not certify
historical physics, producer revisions, or injection clocks.

## Configuration selection

`configs/parser-production-v1.yaml` was selected by fresh nine-candidate sweeps
on the frozen grouped **dev split only: 50 pairs per task, 100 total**. Neither
training nor held-out test telemetry was opened by the selection runs. The
candidate grid was `k=[0,2,5]` and threshold scale `[0.5,1,2]`.

Both tasks had exactly one eligible candidate: **k=5, threshold scale=1**.
Eligibility required every fixed-reference event neighborhood on every dev
pair and per-feature maximum reconstruction errors no larger than the base
configuration (absolute numerical tolerance `1e-12`). Among eligible candidates,
the policy minimizes summed paired-view JSON bytes. The selected paired payload
sizes were 7,305,421 bytes for pickup and 8,187,322 bytes for transport across
their respective 50 dev pairs. These are JSON byte counts, not tokenizer counts.

`configs/parser-production-v1.selection.json` retains the decision, constraints,
grid, dev report hashes, configuration hashes, and exact frozen manifest hash.
The full reports are generated at:

- `runs/production-parser-20261118/pick_up_object-dev-sweep.json`
- `runs/production-parser-20261118/transport_object-dev-sweep.json`

This configuration matches the provisional base values; it does not change
`configs/default.yaml`, `configs/parser.yaml`, or the versioned sample. The
reconstruction constraint is relative to the base compressor, not an absolute
physical-accuracy guarantee. Paired divergence thresholds remain a separate,
fixed numerical policy in `configs/divergence.yaml`; this work does not calibrate
them against injected fault onset.

Reproduce a selection report using a fresh output filename, changing only the
task-specific paths for transport:

```bash
PYTHONPATH=src .venv/bin/python scripts/tune_compression.py \
  --metadata outputs/dataset/batch_20261030_152125_e5ca72de/pick_up_object/metadata.csv \
  --data-dir outputs/dataset/batch_20261030_152125_e5ca72de/pick_up_object \
  --config configs/parser.yaml \
  --manifest runs/dataset-v1-20261118/handoff/tasks/pick_up_object.json \
  --output runs/pickup-dev-repeat.json
```

## Atomic production gate

`parser.production.produce_dataset()` and `scripts/produce_bts.py` validate the
frozen handoff's artifact receipt, all source checksums, exact task sidecars,
grouped split coverage, and metadata identities before conversion. Optional
`--selection` binds the explicit configuration to a dev-only decision receipt
and this exact manifest. A rejected pair, changed source/configuration, or
invalid comparison state prevents publication of the entire output directory.
Existing output directories are refused. Sources, the handoff, and all parser
implementation files are rechecked before the final atomic rename.

```bash
PYTHONPATH=src .venv/bin/python scripts/produce_bts.py \
  --batch outputs/dataset/batch_20261030_152125_e5ca72de \
  --handoff runs/dataset-v1-20261118/handoff \
  --config configs/parser-production-v1.yaml \
  --selection configs/parser-production-v1.selection.json \
  --divergence-config configs/divergence.yaml \
  --output-dir data/processed/production-v1-repeat
```

For each task, output contains `raw-bt/` with the parser's explicit uncomputed
comparisons, `enriched-bt/` with computed divergence, and `onset_report.json`.
Each source pair has one BT in each directory. Task-local IDs remain in separate
directories. The root `production_report.json` records all generated artifact
hashes/sizes, source/configuration/implementation hashes, split counts,
compression/event retention, coverage, divergence states, and certification
scope. BT bytes are deterministic; reports also include the requested final
paths. Use only `pair_*_bt.json` when consuming a BT directory.

The persisted `data/processed/production-v1-20261118/` run published **500 pairs
per task, 1,000 total, with zero rejections**. It retained 156,159 of 708,708 raw
observations and every one of 101,357 fixed-reference feature-event
neighborhoods. Pickup has 1,695 computed failure-stage comparisons and 305
unavailable suffix stages; transport has 1,117 computed and 383 unavailable.
Each task preserves 350/50/100 train/dev/test pairs. All 2,004 generated
artifacts were independently checked against the production receipt, with a
summary at `runs/production-parser-20261118/verification.json`.

## Numerical onset diagnostics

`raw_reference_divergence_diagnostics()` applies the same within-stage alignment
and divergence thresholds to original 50 Hz paired traces and compressed paired
traces. Its signed error is `compressed_timestamp - raw_timestamp` in seconds;
absolute error and maximum-delta errors are also reported. A raw reference
crossing and a compressed crossing must both exist for finite timing error.
Missed crossings, extra compressed crossings, no-crossing comparisons, and
unavailable suffixes have separate counts. Null onset is never converted to zero
or excluded without an explicit denominator. Reports include aggregate and
per-split diagnostics after the production choice is fixed; these do not feed
back into selection.

Both reference timestamps are **aligned numerical grid crossings**, potentially
at interpolated comparison coordinates. Neither is a verified physical fault
onset or injection time. Archived absolute simulator activation timestamps are
not converted using unverified time origins. Retained samples must equal source
observations and preserve stage endpoints and observation states.

For this frozen numerical threshold policy, all 1,309 stages with a raw crossing
also have a compressed crossing at the same grid timestamp: zero missed or
extra crossings and a maximum timing error of 0 s. Another 1,503 comparable
stages have no crossing, and 688 suffix stages are unavailable. This timing
result does not mean compression is lossless: reported maximum-delta errors
include 5.9229 N m for pickup joint torques and 0.04546 m for transport object
position. Consult the feature-level report before interpreting a sensor delta.

CPU tests cover scalar/vector metric offsets, shifted and differently sampled
analytic motion, equivalent `q`/`-q` rotations, finite onset error, missed/extra
crossings, unavailable suffixes, fabricated observations, immutable inputs,
deterministic BT publication, corruption rejection, and clean CLI failures:

```bash
PYTHONPATH=src .venv/bin/python -W error -m pytest \
  tests/test_parser.py tests/test_bt_pipeline.py \
  tests/test_compression_diagnostics.py tests/test_compression_tuning.py \
  tests/test_divergence.py tests/test_divergence_metrics.py \
  tests/test_partial_traces.py tests/test_onset_diagnostics.py \
  tests/test_production_parser.py
```
