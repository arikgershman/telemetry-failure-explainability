# Track 3 parser handoff

## Task structure and observed execution

Every pair uses one canonical `PairedBehaviorTree` and `SubtaskBT` contract.
The success trace must contain the complete ordered `TASK_SUBTASKS` template;
the failure trace may contain any nonempty ordered prefix, including the full
template. Pickup has four stages and transport has three. Skipped interior
stages, repeated visits, reordered stages, empty traces, malformed observations,
cadence violations, and inconsistent joint dimensions remain errors.

The tree always retains every expected task node, its semantic metadata, and
its success samples. A stage without failure observations has
`failure_samples: []`; it never receives fabricated rows or numeric sentinels.
These required fields describe coverage for all pairs:

| Field | Meaning |
| --- | --- |
| `failure_observation_status` | `observed` exactly when failure samples exist; otherwise `not_observed`. |
| `failure_segment_end` | `transitioned` when a later stage was observed; `trace_end` for the final observed stage; null for unobserved stages. |
| `divergence_status` | `uncomputed` with a null object for observed stages; `computed` with a divergence object; `not_comparable` with a null object for unobserved stages. |

The schema checks that observed failure stages form a nonempty prefix and
exactly its final node has `trace_end`. A transition establishes a stage
boundary, not physical goal achievement. The final sample does not establish
why recording stopped. Coverage states are derived from observations, never
from the injection label or failure mode. Raw CSV and metadata columns are
unchanged. `data/sample/pair_000_bt.json` uses this same contract.

## Construction and batch parsing

Run from the repository root, using the project environment:

```bash
PYTHONPATH=src .venv/bin/python scripts/parse_dataset.py --metadata DATA/metadata.csv --data-dir DATA --config configs/parser.yaml --output-dir OUTPUT/bt
```

Use a fresh or empty output directory. Task-local pair IDs in the supplied
archive overlap, so parse each task into a separate directory and join across
tasks with `(task, pair_id)`. Track 2's handoff preserves these original IDs;
never silently prefix or relabel metadata IDs.

```python
from telemetry_reasoning.parser import build_paired_bt, build_paired_bts, parse_dataset

tree = build_paired_bt(metadata_row, data_dir, compression_config)
batch = build_paired_bts(metadata_rows, data_dir, compression_config)
# batch.trees and batch.errors are ordered tuples.
result = parse_dataset(metadata_rows, data_dir, compression_config, output_dir)
# result.files, result.report_path, result.report
```

`validate_subtask_sequence()` is shared in `schemas.py` and exported from the
parser package. It validates consecutive segment labels with `allow_partial`
enabled for failures. `build_bt_from_segments()` provides pure construction
from consecutive compressed segments. Both construction paths validate order
before matching by subtask name, so revisits cannot overwrite segments and a
short failure cannot discard the remaining successful reference.

Batch APIs produce one tree or an explicit error for each pair. Invalid inputs
are recorded with source filenames, exception type, and message; sources are
never moved or repaired. Duplicate IDs reject a batch before pair reads;
unsafe filename IDs are rejected before serialization. Outputs are staged
before publication, and fatal storage errors do not publish a partial batch.
CLI exit codes are 0 for all pairs written, 1 for recorded pair rejections,
and 2 for fatal input/configuration/storage failures.

Parsing deliberately leaves observed-stage divergence uncomputed. Run the
separate enrichment stage when a comparison is required; there is no
`--divergence-config` flag on `parse_dataset.py`.

`parse_report.json` includes written/rejected inventories, source/output/config
hashes, complete-versus-partial failure coverage, observed/unobserved stage
counts, comparison availability, compression quality, and model-view payload
sizes. Reports contain no wall-clock fields and reproduce deterministically.
`termination_verified` remains false: parsing establishes structure, while
Track 2 verifies intentional termination, physical outcomes, and data integrity.

## Alignment

`align_subtask_samples(node)` returns detached comparison arrays and does not
modify the node or compute divergence scores:

- A `transitioned` stage uses the union of normalized retained timestamps.
  This removes pure duration differences without asserting physical success.
- A `trace_end` stage uses stage-relative elapsed seconds over the common
  observed interval. Its endpoint is never stretched to nominal completion;
  neither side is extrapolated. Singletons compare only at elapsed zero.
  Timeout tails remain in the original samples and observed durations.
- An unobserved stage returns `availability="not_comparable"` with empty
  timestamp and feature arrays. Absence is never a zero-valued comparison.

`alignment_mode` is `normalized_progress`, `common_elapsed_time`, or
`unavailable`. `progress` is dimensionless only in progress mode;
`elapsed_seconds` contains seconds only in time mode. The inactive coordinate
is null. Results include absolute timestamps, observed durations, and the
shared elapsed interval where applicable. Stage-relative time assumes paired
controller timing is a meaningful coordinate; it is not measured physical
phase. Interpolated times and values are comparisons, not observations.

Continuous features use linear interpolation, gripper closure uses the previous
value, and quaternions use normalized, sign-aligned shortest-path SLERP.
Queries within 1e-12 coordinate units of a retained sample snap to that sample
to avoid roundoff-induced closure transitions. Comparisons never cross stage
boundaries. Quaternion deltas are component differences, not angular distances.

## Compression diagnostics and tuning

Compression retains original stage endpoints, up to `k` uniform interior
observations, and clipped `t-1/t/t+1` neighborhoods around adjacent changes
strictly exceeding configured thresholds. No event comparison crosses a stage
boundary. `k=0` disables uniform interior sampling. The sample configuration
remains unchanged. Production selection is recorded separately in
`configs/parser-production-v1.yaml` and its selection receipt, based on fresh
dev-only sweeps. All 1,000 enriched outputs and numerical onset diagnostics are
documented in [the production handoff](../../../docs/production-parser.md).

Diagnostics compare retained rows to raw 50 Hz observations. They report raw
and retained counts, compression ratios, event-center and complete-neighborhood
retention, reconstruction errors, and prompt payload characters/UTF-8 bytes.
Payload measurements are not tokenizer counts and exclude prompt templates
and pair IDs. Reference event thresholds stay fixed across tuning candidates.

Missing stages have zero row/event counts and null ratios/errors. They do not
contribute fabricated zero reconstruction errors or event denominators. Raw
reconstruction compares each retained segment with its own recorded interval,
whose endpoints are preserved, independently of paired termination state.
Zero quaternions make only the corresponding reconstruction feature unavailable.

```bash
PYTHONPATH=src .venv/bin/python scripts/tune_compression.py --metadata DATA/metadata.csv --data-dir DATA --config configs/parser.yaml --manifest DATA/manifest.json --output runs/compression-dev.json
```

The manifest must cover every metadata pair with a `split` and
`initial_condition_id`, using either a pair-ID mapping or a list of records.
All groups are validated before telemetry is opened; `(task, initial_condition_id)`
groups cannot cross splits. Only the selected train/dev telemetry is read or
hashed. Test data never contributes to compression selection, and the tuner
does not invent missing Track 2 split assignments.

The default grid is `k=[0,2,5]` and threshold scales `[0.5,1,2]`. Candidate
trees preserve the complete task structure and coverage states, recompress
only observed rows, and clear any earlier computed divergence. The recommendation
minimizes paired payload bytes while retaining every fixed-reference event
neighborhood and not increasing per-pair maximum reconstruction errors over
the base beyond 1e-12 numerical tolerance. All candidates and tie rules are
reported. Configuration is never changed automatically.

The [Track 2 handoff](../../../docs/track2-dataset-handoff.md) records completed
nine-candidate sweeps on 50 dev pairs per task, recommending
`k=5;threshold_scale=1` for both tasks. The task manifests are the supported
tuning inputs. Confirm those archived inputs and record the final production
choice before publishing the enriched 1,000-pair dataset; the tool does not
promote a recommendation automatically.

## Deterministic divergence enrichment

From the repository root:

```bash
python scripts/enrich_divergence.py --bt-dir OUTPUT/bt --config configs/divergence.yaml --output-dir OUTPUT/enriched-bt
```

`parser/divergence_metrics.py` provides `DivergenceConfig`,
`load_divergence_config()`, `compute_subtask_divergence()`, and
`enrich_paired_bt()`. The separate YAML document has exactly one `divergence`
section containing a nonnegative `thresholds` entry for every
`TelemetryFeature`. Thresholds have each feature's native units. They compare
paired traces and are independent of compression's adjacent-row event
thresholds.

For each aligned grid point, a scalar feature contributes its absolute
difference and a vector feature contributes its largest absolute component
difference. `max_abs_deltas` is the maximum of those differences over the
aligned grid for every feature. Quaternion differences are sign-aligned
component deltas, not rotation angles. The first divergence is the earliest
grid point whose difference strictly exceeds any configured feature threshold,
expressed in the failure trace's absolute timestamp coordinate. Equality to a
threshold does not trigger an onset. This coordinate may be interpolated and
does not establish exact observed or physical onset.

Enrichment returns a validated deep copy, preserving source observations and
storage metadata. Every observed stage becomes `computed`; identical traces
have complete zero deltas and a null first timestamp. Every unobserved stage
stays `not_comparable` with a null divergence object. Main-model prompts require
computed divergence for observed stages and accept unavailable suffix stages.

The CLI validates the full batch before staged publication, requires a separate
fresh or empty output directory, and writes only matching
`pair_<pair_id>_bt.json` outputs plus `divergence_report.json`. The report records
source/output/config hashes and computed/unavailable counts. Exit codes are
0 for success and 2 for invalid input, configuration, or storage failures.

The known synthetic grip event is 5.10 s with a 12 N maximum delta under a
2 N grip threshold. Current metric tests also cover strict equality, identical
traces, main/baseline prompt compatibility, partial traces, and complete
configuration. Broader scalar/vector/timing metric cases and raw-reference
compressed-onset error reporting remain acceptance work; alignment tests alone
do not establish those metric-level guarantees.

## Model views and track boundaries

Storage retains scoring metadata. Model inputs use `to_failure_only()`,
`to_paired_without_divergence()`, or `to_full_paired()`, which remove ground-truth
labels, perturbations, source filenames, and seed. Failure-only views also omit
success samples. Views excluding divergence omit both its object and comparison
status. Factual failure coverage is available consistently across variants.

This implementation covers Track 3 and the necessary shared BT contract/sample.
Simulation, dataset acceptance, models, evaluation, plotting, and their tests
are owned by their respective tracks. Track 2 verifies valid termination and
physical outcomes. Track 4 should handle empty stages and interrupted endpoints
in plots and rules. Track 5 should distinguish uncomputed observed-stage
divergence from unobserved stages when building its prompts. Evidence must
always cite real failure observations. Numerical divergence scoring is the
explicit enrichment stage described above. Physical-outcome certification,
raw-reference onset error, and production dataset publication remain separate
acceptance gates.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -W error -m pytest tests/test_parser.py tests/test_bt_pipeline.py tests/test_partial_traces.py tests/test_divergence.py tests/test_divergence_metrics.py tests/test_compression_diagnostics.py tests/test_compression_tuning.py
python scripts/integration_smoke.py --backend stub --output-dir runs/integration-cpu
```

The suite covers both tasks and every failure-prefix length, invalid sequences,
required and contradictory states, source integrity, serialization, model-view
exclusions, elapsed-time censoring, singleton/quaternion/closure edge cases,
compression quality, and grouped tuning. Fresh sample generation reproduces
the checked-in files byte for byte.

The Nov. 18 CPU foundation verification passed the full suite with **826 passed
and 21 skipped** (15 archived-release checks and six opt-in real-physics checks).
The persisted CPU integration run passed seven public CLI stages with four
validated predictions and 10 PNG plots. See [the current acceptance receipt](../../../TODO.md#nov-18-cpu-foundation-acceptance-receipt)
for commands, artifact location, and scope. These checks do not complete
real-producer, real-model, or release certification gates.

### Historical partial-trace verification receipt

The partial-trace implementation passed all 231 Track 3 tests with warnings
treated as errors at that handoff. Its then-current full suite had 358 passes
and the Track 4 evaluator vocabulary failure for `excessive_speed`. These counts
describe that historical run; use the commands above for current verification.
All 1,000 supplied pairs also parsed
and schema-validated, including 390 partial failure prefixes, with all 2,007
source files unchanged. The report at
`runs/track3-partial-traces/current-xorunaw4/verification.json` records the
per-task counts and alignment modes. This verifies parser structure and
alignment bounds, not physical outcomes, calibrated settings, or model quality.

Additional interface verification is reproducible with:

```bash
MPLCONFIGDIR=/tmp/telemetry-parser-mpl PYTHONPATH=src:. .venv/bin/python -W error runs/track3-partial-traces/additional-verification/verify.py
```

Its report at
`runs/track3-partial-traces/additional-verification/run-grkot5wq/verification.json`
confirms that freshly rebuilt trees match all 1,000 stored trees exactly and
all 156,159 retained samples equal source observations. All 1,000 controller
sidecars agree with observed coverage; this is not a physical-causality check.
All four represented modes remain at 125 pairs per task, including the
transport inadequate-grip cases previously excluded by the complete-trace rule.

Both baseline CLIs load all 500 pairs per task with no input errors or invalid
prediction records. All 1,676 cited evidence items match real failure samples,
and the evaluator rejects fabricated citations from all 688 unobserved stages.
The evaluation API and CLI without BT evidence input also run successfully.
These checks establish interface and evidence validity, not diagnostic accuracy.

Seeded synthetic checks cover 100 source pairs, four compression settings, 400
tree builds, 1,400 aligned nodes, and 4,247 segment-label sequences. The terminal
analytic-motion checks retain closure events before asserting equal prefixes;
disabled event retention can deliberately lose interior transitions.

That historical report also captured an evaluator adapter failure with
`--bt-dir` and a plot failure on unobserved suffix stages. Those receipts remain
useful regression evidence; they are not the current list of blockers. Current
evidence-enabled CLI and partial-trace plotting checks are covered by Track 4's
tests and the CPU integration suite. The integration run writes stage logs and
an `integration_manifest.json` with status, assertions, and artifact hashes.
It validates synthetic plumbing without certifying the archived release or
real-model behavior.
