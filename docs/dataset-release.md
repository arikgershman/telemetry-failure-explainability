# Dataset release restoration and certification

The [dataset-v1 release](https://github.com/arikgershman/telemetry-failure-explainability/releases/tag/dataset-v1)
contains the original `dataset_v1.zip` and two subsequently published clock
sidecars. [The release lock](../configs/dataset-v1.json) pins asset IDs, HTTPS
URLs, byte sizes, and SHA-256 digests. The ZIP digest is
`657560c5a42c21bc2770525cb0d7ad0c219f226dd669a119c13d63e6dc157ce4`.
The lock records tag commit `73eeb975c2593e7fe8ad7a8a31a6253693303784`
and Menagerie reference `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` as
source references. A tag cannot establish the exact working tree or environment
that produced archived telemetry.

## Restore the original inputs

Install the project, then use a fresh dataset root:

```bash
python scripts/restore_dataset.py \
  --asset-dir data/raw/releases/dataset-v1 \
  --dataset-root outputs/dataset \
  --download
```

For offline use, put all three lock-named assets in the asset directory and
omit `--download`. The command reuses a correctly hashed cache entry and refuses
to replace a corrupt entry or an existing dataset root. To repeat restoration,
choose a separate fresh root. Each installation contains the original batch
`batch_20261030_152125_e5ca72de` beneath that root.

Restoration verifies every asset before opening the ZIP, rejects unsafe or
ambiguous member paths, symlinks, unsupported entries, and excessive extraction
sizes, then publishes a staged directory. It preserves original archive bytes,
including the archive-root `.DS_Store`. The two task directories contain 2,006
original files; restoration adds no files inside either task directory.

Derived inputs are kept in `<batch>/provenance/`:

| File | Meaning |
| --- | --- |
| `release.json` | Asset identities, source references, installation scope and limitations. |
| `release-lock.json` | The exact lock used for restoration. |
| `published_clocks/*.jsonl` | Byte-identical copies of the published clock assets; their names do not certify their timing. |
| `recovered_failure_configs.json` | Configurations and accepted-pair counts recovered from the original diagnostics, with source hashes. |

The recovered catalogue supports inspection and later replay. It is not the
original calibration CSVs: the archive does not supply rejected candidates,
measured calibration failure/activation rates, or the complete selection inputs.
Do not synthesize those rates or relabel recovered accepted configurations as
original calibration evidence.

## Keep verification claims separate

| Claim | Required evidence |
| --- | --- |
| Release asset identity | Pinned size and SHA-256 for every downloaded or cached asset. |
| Stored trace acceptance | Canonical CSV/schema/cadence checks, task/mode/condition coverage, paired identities, activation diagnostics, parser compatibility. |
| Observed physical outcome | Independently calculated measurements from recorded object/robot states and explicit task-goal thresholds. Record sampling limitations. |
| Exact terminal outcome | Machine-readable terminal state or a sufficiently matched replay with terminal measurements. The final 50 Hz observation can precede termination. |
| Archived task origin | Per-run producer timing, or independently verified replay matching that run. A nominal replay does not by itself verify an injected run. |
| Full pre-injection agreement | Compare every paired observation strictly before verified activation, with explicit tolerance and matching time basis. A zero-length prefix is unavailable evidence. |
| Exact producer provenance | Producer-attested source/model/config/runtime fingerprints, original calibration inputs, and dirty-tree status. Release tags and current validator hashes are insufficient. |
| Frozen downstream handoff | Accepted trace inventory, deterministic grouped splits, task manifests, certification scope/limitations, and verification of the published artifacts. |

`duration_s - last_csv_timestamp` is not an exact origin calculation: the runner
steps after controller transitions and the logger samples at 50 Hz. Likewise,
`first_activation_s` uses absolute simulator time while CSV timestamps begin at
the first canonical task observation. Only populate an evaluator's trace-relative
`injection_timestamp` after verification of the corresponding origin.

Retain original metadata, telemetry, and diagnostics. Put recovered timing,
physical measurements, replay records, and source attestations in separate
sidecars that fingerprint their inputs. Refuse contradictory evidence and
represent unavailable evidence explicitly. A usable, checksummed release may
retain a partial certification status while missing producer facts are collected.

See [the Track 2 handoff](track2-dataset-handoff.md) for acceptance, freeze,
grouped-split, and downstream commands. `excessive_speed` remains a valid schema
mode but has no accepted examples in v1; four-mode release reporting must not
remove it from prediction validation.

## New-run interface and provenance

`run_episode(..., failure=None, control_hook=None)` returns `EpisodeResult` with
`rows`, `task_succeeded`, and `diagnostics` attributes; tuple unpacking yields
those three values in that order. New diagnostics include the absolute
`telemetry_start_sim_s`, terminal state, independent physical outcome, and actual
runtime/source/model fingerprints. Fingerprints report the current dirty state
rather than treating `HEAD` as the complete source identity.

The optional hook receives `(timestamp, subtask, nominal_command)` once per
500 Hz canonical task step, using trace-relative seconds. The detached command
contains exactly `ee_pos` (world metres), `ee_quat` (unit w/x/y/z),
`gripper_state` (closure, 0=open and 1=closed), `gripper_force` (total finger
generalized actuator-force limit in N), and positive `speed_scale` (joint-speed
limit multiplier). Force is a command limit, distinct from measured telemetry
force and the Panda's dimensionless 0–255 control. Return the complete validated
command, or `None` to retain the original nominal command. Unknown fields,
nonfinite values, invalid quaternions/closure, and force limits above hardware
capacity are rejected before applying overrides. Hooks change control commands;
they do not rewrite measured state or stored rows.

Identity hooks preserve nominal numerical replay. Motion hooks compose with
`FailureConfig`; actuator failure overrides are applied after the hook. The
Panda home keyframe initializes only robot joints, so the runner explicitly
restores the cube's scene-default free-joint pose before applying its seeded XY
offset. This fixes current initialization without claiming that the same source
or native runtime produced v1.

New failure diagnostics count `activation_steps` once per distinct active
simulator timestamp and declare `activation_steps_basis`. They separately retain
`activation_context_calls`, since the command loop can update context twice per
physics step. Archived records without the new basis retain their original
context-call meaning; replay compares them against the explicit legacy counter.
Unknown counter bases are not accepted as matched evidence. Source CSVs and
archived diagnostics are never rewritten to adopt the new interpretation.


## Nov. 18 restoration and replay receipt

The real v1 restore verified all three locked assets, preserved 2,006 original
task files, and recovered nine accepted configurations per task. The local
release tag was restored at the source reference above. Panda restoration
verified 80 pinned Git blobs; its receipt is
`data/raw/model-assets/panda-f3b60136.json`. Fresh native reproducibility and hook/
outcome checks passed with MuJoCo 3.3.7 (57 native checks). The optional current
validation profile, `configs/sim-validation.requirements.txt`, pins MuJoCo 3.3.7
and NumPy 2.5.4. These are current validation versions; the historical producer
environment remains unknown.

The representative archived replay is separate evidence. Eight pairs required
32 actual fresh nominal/injected executions. All 16 fresh repeatability checks
passed, while all 16 archived run records mismatched strict comparison at `atol=1e-8, rtol=0`. The source-bound receipt at
`runs/dataset-v1-replay-20261118-final/evidence.json` retains those mismatches and
unverified candidate clocks. It cannot establish archived onset, terminal state,
or exact historical runtime. The handoff at
`runs/dataset-v1-20261118/handoff` was published and verified with 1,000 accepted
pairs, 2,006 source files, and grouped train/dev/test counts of 700/100/200. Its
actual status is `parser_ready_certification_failed`: the known archived replay
failure is separate from the pending archival timing, terminal-outcome, and
producer-provenance gates. Use `--require-certified` wherever
complete release certification is required.

The final default suite passed 989 tests with 12 opt-in native tests skipped;
the separate native acceptance run passed 57 tests. The local verification
receipt, `runs/dataset-v1-20261118/verification.json`, records checks against
both the original source inventory and the new handoff, plus clean refusal of
`--require-certified` without publishing an output directory.
