# Track 3: Nov. 6 handoff

Batch parsing, serialization, compression measurement, and a guarded tuning
sweep are implemented. `configs/parser.yaml` is a complete `ProjectConfig` and
deliberately preserves the mock settings. It is **provisional**, pending tuning
on Track 2's validated train/dev data. Shared schemas, default configuration,
and checked-in samples are unchanged.

Run from the repository root using the existing environment (or install the
project into another Python 3.10+ environment):

```bash
PYTHONPATH=src .venv/bin/python -W error -m pytest tests/test_parser.py tests/test_bt_pipeline.py tests/test_divergence.py tests/test_compression_diagnostics.py tests/test_compression_tuning.py
PYTHONPATH=src .venv/bin/python scripts/parse_dataset.py --metadata data/sample/metadata.csv --data-dir data/sample --config configs/parser.yaml --output-dir data/processed/track3-nov06/sample/bt
```

Use a fresh/empty output directory. Reusing a populated directory fails before
writing anything, preventing stale files from entering Track 4/5 glob readers.
The sample BT is byte-identical to `data/sample/pair_000_bt.json`: 31 success and
34 failure samples from 800 raw observations, including the failed-grasp
5.08/5.10/5.12 s neighborhood. This demonstrates reproducibility, not physical
performance or calibrated compression thresholds.

## Batch API and outputs

The existing `build_paired_bt()` and `build_paired_bts()` signatures and behavior
are unchanged. The new exported API is:

```python
from telemetry_reasoning.parser import parse_dataset

result = parse_dataset(metadata, data_dir, compression_config, output_dir)
# result.files: ordered tuple of written BT paths
# result.report_path: output_dir / "parse_report.json"
# result.report: detached JSON-compatible report
```

For each input pair, the pipeline either writes a schema-valid
`pair_<pair_id>_bt.json` or records its source filenames, exception type, and
message in `parse_report.json`. Source CSVs are never changed. Quarantine is an
explicit rejection record; no files are moved and no observations are invented.
Missing/repeated/reordered stages, malformed rows, cadence violations, and
inconsistent joint dimensions remain errors. Duplicate IDs reject the batch;
path separators/control newlines in IDs are refused for flat output filenames.
Both task templates and arbitrary consistent joint counts are supported.

Output is staged before publication. Fatal input/configuration/storage failures
do not publish a partial batch. CLI exit codes are 0 for all pairs written, 1
for pair rejections with valid outputs/report published, and 2 for fatal errors.
The report includes exact input/output inventories, source and configuration
hashes, and no runtime clock fields, allowing deterministic reproduction.

Every divergence object remains **null / not computed**. The Nov. 13 metric
stage, its separate `configs/divergence.yaml`, and the full frozen-dataset
conversion are still pending. This Nov. 6 command therefore does not accept
`--divergence-config`. Null must not be interpreted as zero. Track 4's baseline
and plotting consumers can read these BTs now; the main LLM variant must wait
for computed divergence.

## Measurements and tuning

`parser/diagnostics.py` compares the actual retained observations to raw 50 Hz
telemetry. Reports include stage/run/pair counts, raw/retained compression ratio,
strict adjacent-component reference events, event centers, complete clipped
`t-1/t/t+1` neighborhoods, and unique neighbor-row retention. Events are never
compared across subtask boundaries. No-event denominators are null.

Reconstruction uses the existing within-subtask progress alignment: linear
continuous features, previous-value closure, and normalized sign-aligned
shortest-path quaternion SLERP. It reports maximum and component-weighted mean
absolute errors for every feature. Quaternion errors are component differences,
not angles. Zero quaternions are permitted by the storage schema, so their
reconstruction metric is explicitly unavailable rather than an invented value.
These errors measure compression loss, not paired divergence or physical onset.

Prompt payload measurements use compact sorted-key JSON from the leakage-free
view helpers, with the opaque pair ID removed. They report Unicode characters
and UTF-8 bytes, excluding prompt templates. They are **not model token counts**.

```bash
PYTHONPATH=src .venv/bin/python scripts/tune_compression.py --metadata DATA/metadata.csv --data-dir DATA --config configs/parser.yaml --manifest DATA/manifest.json --output runs/track3-nov06/compression-dev.json
```

The supplied sidecar must contain every metadata pair exactly once, with
`split` (`train`, `dev`, or `test`) and `initial_condition_id`. Both evaluator
shapes are accepted: `{"pairs": {"000": {...}}}` and
`{"pairs": [{"pair_id": "000", ...}]}`. `(task, initial_condition_id)` groups must
not overlap splits. Integers and their string representations identify the
same group. Validation covers all entries before any telemetry is opened.
Only the requested train/dev split is read or hashed; test telemetry is never
used. Missing/invalid grouping fails explicitly; the tuner does not invent
Track 2's assignments.

The default grid is `k = [0, 2, 5]` and event-threshold scale
`[0.5, 1, 2]`, applied to every configured feature. Raw reference-event
thresholds stay fixed at the base settings for all nine candidates. The report
recommends the smallest paired payload that preserves all reference-event
neighborhoods and does not increase any feature's per-pair maximum
reconstruction error over the base (apart from a documented 1e-12 numerical
tolerance). All candidates, source/configuration hashes, checks, and tie rules
are retained. The sweep never updates configuration automatically.

## Real-data boundary

Gili's `dataset_v1.zip` (seed 723) supplies raw telemetry and diagnostics, but no
validated grouped split manifest. Its task-local pair IDs also collide across
the two metadata files. Parse them into **separate output directories** until
Track 2 supplies globally unique IDs; never silently prefix or relabel them.
Some injected runs are partial and cannot satisfy the frozen complete-template
schema. Track 3 reports them without changing Track 1/2's data or acceptance
policy. Full-archive parser compatibility evidence and synthetic tuning reports
are stored under ignored `runs/track3-nov06/`; neither establishes a validated
final dataset or real-data train/dev calibration.

The complete fixed-default preview on the supplied archive produced:

| Task | Input pairs | Written BTs | Rejected incomplete failures |
| --- | ---: | ---: | ---: |
| `pick_up_object` | 500 | 384 | 116 |
| `transport_object` | 500 | 226 | 274 |
| Total | 1,000 | 610 | 390 |

The compatible pairs retained 110,283 of 448,309 raw observations (24.60%,
4.065x reduction) and all 71,708 feature-event neighborhoods under the fixed
reference thresholds. Every output was schema-validated, and archive/source
hashes were unchanged. These 610 pairs are **not a replacement balanced
dataset**: compatibility filtering removes all transport inadequate-grip
examples. Track 2 must resolve partial-trace handling before final conversion.
The small real-source smoke also passed through Track 4's existing baseline
CLI without filename/schema errors.

The explicitly synthetic development sweep uses six pairs (both tasks, three
seeds) and all nine grid candidates. Its recommendation retains all six fixed
reference-event neighborhoods and reduces retained rows from 306 to 102 and
paired payload bytes from 105,138 to 36,939. These results apply only to that
synthetic corpus; no production settings were changed. The generator, command,
and byte-identical repeated report are saved in
`runs/track3-nov06/synthetic-tuning/`.

Verification with warnings treated as errors passes all 152 Track 3 tests.
The full repository suite passes 279 tests and retains one pre-existing Track 4
failure: `test_label_vocabularies_match_shared_schemas` because its evaluator
vocabulary omits `excessive_speed`. Track 3 does not change that owned code or
the shared valid failure-mode enum.
