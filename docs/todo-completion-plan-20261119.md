# Execution plan: complete TODO.md

Date: November 19, 2026. This is an implementation handoff, not a claim that the remaining work has run.

## 1. Objective and completion rules

Complete the engineering deliverables, real-data/model acceptance, reviewed-data optimization, adapter evaluation, final held-out study, and integration evidence described by TODO.md. Preserve the established contracts and every historical receipt.

The audited TODO snapshot has **88 checkboxes: 78 checked and 10 unchecked**. Its SHA-256 is 813cdfa9e5e6ffeef2ee895367ed619f84781122db9ba390ef32a14d176703c2. Git HEAD is b5bcff96618351019cd030f9a234c4a610fc461c, on track-5-llm-prompts; substantial implementation is currently uncommitted/untracked. A clone of HEAD alone is not this working implementation.

Use the accompanying [machine-readable task and checkbox map](todo-completion-plan-20261119.json). Original line numbers identify this snapshot only; match the original text after edits.

Completion requires evidence, not an arbitrary accuracy target. An honestly reported negative scientific result can complete an experiment. It cannot establish an improvement, physical certification, human review, or useful diagnosis. Do not force non-null predictions to make the quality gate pass.

Three requirements cannot be manufactured by an agent:

1. Exact historical producer inputs/records for the original v1 archive, if those records were never preserved.
2. Genuine human preference annotations and explanation review.
3. Actual optimizer/inference measurements on suitable GPU hardware, plus another member's actual integration review.

Prepare everything possible while those dependencies are pending. Do not repeatedly ask for authorization already given. Do not contact other people or send review requests without authorization.

### Immutable constraints

- Keep original dataset-v1 bytes, v1/v2/v3 model runs, manifests, model caches, and transfer archives intact.
- Never rewrite an old bundle's source hashes to make it pass with new code. Verify historical runs using their archived implementation; prepare a new bundle for every changed executable protocol.
- Keep task-local identities as (task, pair_id). Keep model variants as baseline_1, baseline_2, baseline_3, main.
- Baseline 1/2 consume failure-only observations; Baseline 3 adds reference success; main adds divergence. No storage labels, perturbations, filenames, seeds, or pair IDs enter predictive prompts.
- Preserve data/sample byte-for-byte. Reuse schemas.py, io.py, view helpers, parser stages, output validators and existing CLI behavior.
- Preserve all expected requests, including errors, abstentions and context refusals. Report full-cohort denominators.
- No silent truncation, invented observations, relabeled historical provenance, fake reviewed labels, hidden retries, or post hoc exclusion of inconvenient pairs.
- Test remains sealed for method selection. Existing structural BT publication across all splits is not permission to inspect test predictions for tuning.
- The instruction to execute this plan includes the final study after the freeze gate. Use the explicit final-study switch at that point; do not invent a second approval requirement solely for the switch.

## 2. Verified starting point

| Area | Existing evidence | Meaning |
| --- | --- | --- |
| Source regression suite | runs/alienware-v3-recovery-20261119/full-suite-final.log | 1,358 passed, zero skipped, with native MuJoCo and tiny CPU HF tests enabled |
| Historical dataset | outputs/dataset/batch_20261030_152125_e5ca72de | 1,000 pairs, 2,006 original task files |
| Dataset handoff | runs/dataset-v1-20261118/handoff | 700/100/200 grouped splits; actual status parser_ready_certification_failed |
| Archived replay | runs/dataset-v1-replay-20261118-final/evidence.json | All 16 representative archived records mismatch current-runtime replay; fresh repeats agree |
| Track 3 production | data/processed/production-v1-20261118/production_report.json | 1,000 enriched BTs, 2,004 artifacts; 2,812 computed observed comparisons, 688 unavailable suffixes |
| Compression diagnostics | Production report and onset reports | All 101,357 fixed-reference event neighborhoods retained; 1,309 matching numerical crossings; physical onset not established |
| Fresh native integration | runs/integration-mujoco-cpu-20261118/integration_manifest.json | 18 accepted configurations, 76 executions; 13 observed prehistories match, five unavailable |
| Real v3 smoke | runs/alienware-v3-return-20261119/returned/runs/alienware-fresh-smoke-hf-v3-20261119 | All 24 requests completed with valid output on eight real pairs |
| Mac evaluation | runs/alienware-fresh-smoke-v3-evaluation-20261119/evaluation_receipt.json | Passed independent hashes/settings/prompt/completion/qualification verification |
| Model quality | runs/alienware-v3-followup-20261119/returned_metrics_audit.json | All 24 LLM outputs abstain; all LLM mode/stage/joint accuracies zero; all explanations hit 512 characters |
| Training | runs/training-cpu-20261118/formatted-synthetic | Synthetic demonstration only: 24 train records, zero dev records; correctly refused by training launcher |

Hardware: RTX 5090, 31.336 GiB reported total CUDA memory; 62.248 GiB reported total host RAM. Passing inference used Torch 2.14.1+cu130 and Transformers 5.23.0. Smoke peaks were 23.10 GiB allocated and 29.49 GiB reserved. Allocation warnings occurred, but no request failed; allocator retry counters were not recorded.

The old training profile is a separate, unqualified Torch 2.6.0/Transformers 4.49.0/TRL 0.15.2/PEFT 0.14.0 environment. It requires 64 GiB host/32 GiB GPU even at 8,192 tokens. Archived dev main prompts begin at 25,309 tokens before completions. Do not run that profile blindly or lower its floors to admit this machine.

A newly identified checked-item implementation gap also needs repair: sim/calibrate_failures.py imports nonexistent dataset.configs_for, so its public CLI currently fails. Existing historical calibration evidence does not establish a working current calibration command.

## 3. Fastest safe schedule

Do not perform a full rewrite. Use four coordinated lanes, with a single GPU queue.

| Lane | Work | Parallelism and handoff |
| --- | --- | --- |
| Coordinator | P00, source snapshots, contract decisions, final verification, P13 | Maintains dependency/evidence ledger and integrates changes |
| Producer/dataset | P01 and P02 | CPU work and original-record recovery start immediately; independent of decoder/training |
| Model protocol | P03 then P04; P07 GPU diagnostics | Owns llm.py, constrained.py, prompts.py while changing those interfaces |
| Evaluation/training | P05, preparation for P08/P09/P10 | Review tools and runtime research proceed immediately; edits touching model interfaces wait for their owner |

Critical path:

- P00 -> P03 diagnostics -> P04 final input representation -> P07 small real validation.
- P05 review tools + stable P04/P07 prompts -> genuine annotations in P08.
- P08 + P09 training qualification + P10 adapter support -> P11 real optimization/dev evaluation.
- P02 certified study dataset + P06 production inputs + P07/P11 method freeze -> P12 held-out study -> P13 release review.
- Producer-record requests, human-review scheduling, and training-runtime compatibility work start at P00 rather than waiting for model fixes.

Avoid repeated source invalidation: execute long native/GPU jobs from a complete hashed source snapshot or commit, with the corresponding model assets. Keep that snapshot unchanged even while other lanes edit the development checkout. Never run receipts-producing jobs against files another agent is modifying.

On the Mac, begin native worker pools conservatively (two processes and one BLAS/OpenMP thread each); measure time/RAM before increasing. Do not run competing GPU qualification/training jobs simultaneously.

There is no defensible fixed completion date before original-record availability, human annotation capacity, and measured optimizer fit are known. Report measured durations and blockers, not invented time estimates.

## 4. Work packages

### P00 — Freeze the baseline, reconcile status, and open the external dependencies

**Runs:** Mac CPU. **Dependencies:** none.

Implementation and actions:

1. Inventory tracked and untracked intended source/config/tests/docs. Preserve the dirty working tree and create a recoverable source snapshot with file hashes. Do not use reset/clean/stash operations that could discard work. Keep weights, environments, raw data and generated runs out of Git.
2. Reverify the current returned v3 run before modifying hashed source. Use experiment_results.verify_gpu_results and verify the Mac evaluation inventory. Record the archive/checksum, qualification receipt, expected/actual request counts and quality audit.
3. Preserve a runnable original-code snapshot from the full Nov. 18 archive plus the v3 overlay, so historical artifacts remain independently verifiable after source changes.
4. Update TODO/README status text and attach an execution acceptance note:
   - Mark the literal real-model smoke requirement at original line 572 complete.
   - Record that the execution/output-delivery parts of 468 and 510 have real evidence; separate any remaining causal-quality requirement instead of rerunning them merely because the dates are old.
   - Keep final-study, optimization, human-review and archival-certification requirements open.
5. Maintain separate versioned status fields in new reports: execution_acceptance, request_coverage, diagnostic_quality_evaluated, explanation_review_complete, dataset_certification, optimizer_acceptance and final_study_complete. Preserve the semantics of historical real_model_acceptance; do not reinterpret old JSON.
6. Prepare an unsent producer-evidence request listing exact original calibration CSVs/results, generator working tree/patch, model assets, Python/MuJoCo/NumPy versions, scene/config inputs, generation command and original timing/terminal records. Ask the user for these unavailable records; do not send another group member a message without authorization.
7. Establish human reviewers and the review packet workflow. Do not create placeholder reviewer assertions or pass synthetic data as reviewed.

Verification:

- Reproduce the 24-request independent verification and all 28 evaluation artifact hashes.
- Default offline suite remains usable; run the full/native baseline once, then affected suites after changes.
- New acceptance-state tests must show that 24 valid abstentions pass execution but report zero diagnostic coverage, and that stubs never pass real execution.

**Output:** runs/<run-id>/baseline_snapshot.json, baseline_verification.json, external_dependencies.json; tracked status documentation with exact evidence references. No original receipt is rewritten.

### P01 — Repair calibration and strengthen the production generator

**Runs:** Mac CPU. **Dependencies:** P00; can run beside P03/P05.

Code:

- sim/calibrate_failures.py, sim/dataset.py, sim/outcomes.py, sim/runner.py.
- New versioned candidate catalogue, for example configs/failure-candidates-v1.json and sim/calibration_candidates.py.
- Tests: test_calibration.py, test_dataset_retries.py, test_dataset_acceptance.py, test_physical_outcomes.py and public CLI import tests.

Implement:

1. Search Git history/release sources for the original configs_for implementation before defining a new catalogue. Replace the missing import with a strict, versioned candidate-catalogue API. If the original catalogue is unavailable, identify a new proposed catalogue as such and calibrate it; do not claim recovered historical magnitudes. Validate all five mechanisms, task/subtask compatibility, axes, magnitude/delay units, finiteness and deterministic candidate ordering.
2. Make --help and CPU argument validation work without eagerly initializing physics. Add public CLI/import regression coverage so another missing symbol cannot hide behind passing runner tests.
3. Recompute physical pickup/transport outcomes from independent terminal measurements through sim/outcomes.py. Calibration must report activation rate, controller outcome, independent physical outcome and disagreement counts separately.
4. New dataset acceptance must require nominal physical success, activated injection, independently measured injected physical failure, valid telemetry, matching initial state, and the declared pre-injection history policy. Controller strings alone cannot admit a pair.
5. Preserve the fixed condition/mode slot schedule through retries. A rejected candidate cannot change the slot's condition, mode or quotas.
6. For a newly certified dataset, calibrate nonzero activation delays so there is measurable pre-activation history. Do not retroactively change archived onset times. A run with no stored pre-activation sample remains explicitly unavailable.
7. Record source tree/snapshot hashes, model/scene/asset hashes, exact runtime/package versions, candidate catalogue, calibration input hashes, full invocation, seed/spread, acceptance criteria, rejected attempts and physical measurements.
8. Publish calibration and completed batches atomically into absent destinations. Preserve incomplete work as explicitly failed/staging evidence, not a completed release.
9. Add source-bound checkpoints/shards only where they save material CPU time. Resume requires matching source/config/schedule hashes and disjoint completed slots; source changes start a new run. Never append duplicate accepted identities.

Verification:

- CPU fakes: inactive injection, controller/physical disagreement, physically successful injected run, missing terminal evidence, wrong units, empty telemetry, retries/exhaustion, nominal failure, source mutation, resume duplication, corrupted checkpoints.
- Native calibration pilot for both tasks/all five mechanisms before expanding.
- For a new production calibration, evaluate all 50 conditions/task and predeclare the reliability threshold (current threshold is 0.9).
- Preserve the four-mode release quota. Excessive speed stays supported and separately tested; inclusion in a new released dataset requires a declared protocol change and measured reliability, not a quiet balance change.

**Gate:** calibration CLI works; physical acceptance is independently tested; the exact new producer state is recorded. This repairs current reproducibility tooling; it does not supply historical calibration files.

### P02 — Close historical certification, or explicitly decide the replacement route

**Runs:** Mac CPU plus actual producer records. **Dependencies:** P00; P01 for a new dataset.

#### P02A: recover and verify original v1

1. Inventory supplied original materials without changing the v1 archive. Distinguish records genuinely captured at generation from retrospective recovery or inference.
2. Extend dataset_certification.py/validate_dataset.py narrowly to import authentic external timing/terminal/producer sidecars. Bind task/pair/run role, both trace hashes, diagnostic/metadata hashes, original sidecar hashes, runtime, source/model/config/calibration hashes and evidence origin.
3. Current producer_certification checks Git/source/model fields, but not every runtime/config/calibration requirement in TODO. Add explicit required provenance fields in a versioned evidence contract. A Git tag is not an original dirty working-tree snapshot.
4. Accept a complete dirty-tree snapshot only with exact files/patch hashes and reproducible state; never invent tree cleanliness. Do not set recorded_at_generation=true on a guessed reconstruction.
5. Restore an evidenced candidate historical environment separately from the working inference environment.
6. Run representative replay first: both tasks, each released mode, nominal and injected runs, two repeats, all raw fields and injection diagnostics at atol=1e-8/rtol=0. Compare complete trajectories, durations/stages and initial identities.
7. If representative replay still mismatches, retain the receipt and fix the evidenced environment/input mismatch. Do not spend the full replay budget on an already disproven environment or relax tolerances to obtain a pass.
8. After representative agreement, run full coverage. Current deduplication represents 734 unique episode inputs: 1,468 fresh executions at two repeats, covering 2,000 stored run records. If parallelizing, implement source-bound unique-input shards and a merger that rejects missing/duplicate/incompatible evidence.
9. Establish every run's original task origin independently. Nominal origins cannot certify injected runs; duration-minus-last-timestamp and sidecar filenames are not proof.
10. Compare every stored pre-injection observation, every field, from the common canonical origin to activation. Report number of rows, first mismatch and absolute tolerances. Zero observations does not pass vacuously.
11. Independently classify physical success/failure from genuine terminal measurements. Reconcile controller/outcome disagreement; do not overwrite original outcomes.
12. Publish a fresh certified handoff only after every required gate passes. Verify source bytes before and after.

Existing commands, with new absent output directories:

~~~bash
PYTHONPATH=src .venv/bin/python scripts/recover_clocks.py \
  --batch "$BATCH" \
  --output-dir "$RUN_ROOT/replay-representative" \
  --representative --repeats 2

# Run only after representative agreement:
PYTHONPATH=src .venv/bin/python scripts/recover_clocks.py \
  --batch "$BATCH" \
  --output-dir "$RUN_ROOT/replay-all" --repeats 2

PYTHONPATH=src .venv/bin/python \
  -m telemetry_reasoning.sim.freeze_robot_dataset "$BATCH" \
  --evidence "$VERIFIED_EVIDENCE" \
  --handoff-dir "$HANDOFF" --require-certified

PYTHONPATH=src .venv/bin/python \
  -m telemetry_reasoning.sim.freeze_robot_dataset "$BATCH" \
  --verify "$HANDOFF/dataset_manifest.json" --require-certified
~~~

Tests: forged/changed clocks, wrong run roles, nominal-only evidence, missing calibration/runtime fields, untrusted origin assertions, physical disagreements, strict trajectory mismatch, empty prehistory, duplicate/missing shard results, same-size corruption and failed atomic publication.

#### P02B: if original records do not exist

Prepare a concrete decision for the user: retain v1 as historical/uncertified and publish a separately versioned, fully recorded fresh dataset as the study source. A replacement can complete a revised reproducibility goal; it **cannot** make the original archive reproducible or close the original historical wording automatically.

Before asking for that decision, complete P01, the inventory of missing records, and a small physically validated new-producer demonstration. Present counts/configuration and expected CPU work, not a speculative permission request.

If replacement is authorized:

- Calibrate the new producer, then generate 500 accepted pairs/task, 50 conditions/task, ten pairs/condition and 125 pairs per represented mode.
- Record origins/terminal measurements/provenance at generation; require nonempty prehistory under the new certification policy.
- Repeat nominal success checks over all 100 task/condition combinations; verify hooks, activation, reset/replay, telemetry and physical goals.
- Freeze 700/100/200 grouped splits and certify the new version. Never overwrite or relabel v1.
- Rebuild/rebind downstream BTs, selection receipts, annotations and experiment bundles to the new dataset.
- Record the explicit requirement revision in TODO and release notes. Without original records or this explicit scope decision, the four historical certification checkboxes remain blocked.

**Closes:** original TODO lines 137, 199, 202, 213 only when their exact historical evidence requirements are met; otherwise records a truthful blocker/approved scope revision.

### P03 — Diagnose and correct the output protocol

**Runs:** CPU implementation/tests; a small Alienware diagnostic. **Dependencies:** P00.

Code:

- models/llm.py, models/constrained.py, models/prompts.py.
- New models/decoding_diagnostics.py and scripts/diagnose_decoding.py.
- New protocol config, for example configs/llm-production-v4.yaml.
- Tests: test_constrained.py, test_hf_execution_profile.py, test_llm_failure_cleanup.py and new diagnostic tests.

First measure, then choose the fix:

1. Add opt-in bounded diagnostics at failure-mode choice, stage choice, evidence choice and explanation termination.
2. Capture raw top-k scores before constraint masking, allowed token strings/IDs, masked top-k, chosen token/rank, retained probability mass, automaton state and termination reason. Verify the actual Transformers processor order so the supposedly raw scores are genuinely unmasked.
3. Retain CPU scalars/short lists only; never save full per-token GPU score tensors or long-vocabulary histories.
4. Measure full conditional likelihoods of all five mode continuations plus null on the same prompt/context. Report summed and length-normalized diagnostics; do not silently turn this into a new decision rule after seeing labels.
5. Bound the diagnostic workload and GPU memory: branch-only scoring, reuse or reconstruct bounded caches explicitly, and avoid storing every full vocabulary tensor for hundreds of steps. Do not clone the full 56K-token KV cache for each alternative. Use serial branch scoring with a tested cache rollback/crop, or re-prefill bounded diagnostic requests; compare cached branch scores against independent tiny-model forwards.
6. Verify diagnostics do not change generated token IDs, RNG, normal outputs or model settings.

Concrete findings to investigate:

- Independently encoded mode alternatives begin with different partial tokens: quote token 1 for most modes, token 55501 for the start of inadequate_grip_strength, token 2994 for null.
- The forced JSON prefix's token boundary differs from natural whole-object tokenization.
- Natural explanation endings may include combined punctuation/quote tokens, while the current grammar only allows a standalone closing quote. Every actual explanation reached the 512-character cap.
- These are verified tokenizer properties; the saved outputs contain no logits proving which property caused abstention. Do not assert a cause before measurement.

If diagnostics confirm a grammar restriction:

- Introduce a new protocol that evaluates token bytes across JSON grammar boundaries rather than assuming independently encoded fragments are the only valid tokenization.
- Accept valid boundary-spanning tokens and declared whitespace/escape forms; keep strict complete JSON, all modes/null, observed stages and exact observed citations.
- Prefer a compatible maintained grammar processor only after CPU tests prove its exact enum/value/termination behavior; otherwise implement the byte/grammar transition API explicitly. Record dependency/version if using a library.
- Retain genuine abstention. No rule/ground-truth fallback supplies a model diagnosis.
- Use a concise explanation instruction and permit natural termination. Record natural versus forced-budget termination; do not count syntactically closed but clipped prose as complete-quality output.
- Prove an output budget or use an explicit staged structured selection plus bounded rationale protocol. If a complete-output bound cannot be proven, reject the settings before weights rather than repairing a truncated response.

Tests with the pinned production tokenizer:

- Naturally tokenized whole JSON objects, including tokens spanning punctuation/field boundaries, all modes and null, both tasks, partial observations, one/two citations, vector values, exponents, escapes and adversarial strings.
- Canonical and alternative valid tokenizations of the same allowed bytes.
- Invalid/unobserved features/stages/values, extra JSON/prose, premature EOS, engine callback bypass and output-budget exhaustion.
- Actual tiny CPU HF generation through the callback, parser and manifest; legacy v3 tests remain intact.
- Diagnostics are bounded, CPU-only in persisted data, source-bound, and observational.

Prepare a new diagnostic bundle/source snapshot before these measurements, even when reusing the exact v3 messages/settings. Old v3 bundle hashes must not be bypassed to run changed diagnostic code. The small measurement in P03 precedes protocol selection; P07 reuses its evidence rather than repeating the same control for no reason.

**Gate:** explain the observed failure mechanism with measured evidence or explicitly retain uncertainty; pass the new protocol tests. No minimum scientific accuracy is invented.

### P04 — Resolve prompt/context size and freeze the final input representation

**Runs:** Mac CPU; GPU qualification after lengths are known. **Dependencies:** P03 interface ownership; can prototype independently.

Code:

- models/prompts.py; new models/prompt_serialization.py.
- experiments.py, fresh_experiments.py, models/training_data.py, models/training.py.
- New scripts/audit_token_budgets.py and representation/semantic round-trip tests.

Implement:

1. Audit train/dev exact token counts by task/variant/condition. Include complete chosen/rejected sequences and chat-template/EOS behavior for training; character estimates are insufficient.
2. Add an explicit versioned lossless compact serialization, initially columnar telemetry: declare feature names once, preserve each original timestamp/value/vector/quaternion, stage order/semantics, observation/end states and every divergence quantity.
3. Decode the compact representation and assert semantic equality to the allowed original view. No rounding, sample dropping, target invention or additional labels.
4. Keep the same serialization/instructions for all prompted variants; their only ablation difference remains available evidence. The formatter must call the same build_prompt path as inference.
5. Recount lengths before deciding context/precision settings. Columnar formatting is a candidate optimization, not a claim that 8K training becomes feasible.
6. If some records still cannot fit, choose an explicit policy:
   - Preserve explicit refusals and score all intended requests, with a prospectively defined common-fitting cohort as secondary analysis; or
   - Implement a separately versioned evidence-selection/compression or staged inference method, validate it on train/dev, and disclose coverage/losses; or
   - Use separately qualified quantization/offloading/larger hardware.
7. A method that removes observations must never be called lossless. Do not change only training prompts or only one model variant to make memory fit.
8. Freeze input representation before collecting preferences. A changed BT or prompt invalidates its annotation binding and requires genuine re-review; a new export alone cannot establish that review. Re-export without re-review only when the bound BT/prompt bytes and reviewed response content are unchanged. Never rewrite annotation hashes to follow changed inputs.
9. If parser compression itself changes, retain old production output and rerun dev selection, reconstruction/event-retention and raw numerical-onset diagnostics. A new model serialization alone should not rebuild or retune Track 3.

Verification:

- Exact view round-trip for real train/dev examples and synthetic edge cases.
- Recursive leakage tests, unequal joint dimensions, scalar/vector/quaternion precision, empty suffix stages, divergence null/computed distinctions.
- Token audits include every selected record; oversized examples are visible and no records disappear.
- Optional hardware profiles affect all three prompted variants consistently and receive new provenance.

**Gate:** a measured feasible intended cohort/profile, declared refusal policy, and one frozen prompt representation for inference, annotation and training.

### P05 — Finish quality evaluation, explanation review, and rule calibration

**Runs:** Mac CPU plus genuine human review. **Dependencies:** P00; final packets depend on P03/P04.

Code:

- eval/metrics.py, eval/contracts.py, scripts/evaluate.py, configs/eval.yaml.
- New eval/review.py, scripts/review_explanations.py, tests/test_human_review.py.
- models/rule_based.py/configs/rules.yaml only for evidenced defects/calibration.

Implement:

1. Separate execution acceptance, answer coverage, conditional accuracy, full-cohort accuracy, evidence support, explanation completeness and human review in new reports. Preserve strict prediction JSONL fields.
2. Extend blinded rubric export to include real abstention explanations; currently only status=ok is eligible, leaving zero HF review rows in this smoke.
3. Review factual consistency, unsupported success/failure claims, appropriateness of abstention, causal support when applicable and clarity/completeness. Preserve not-applicable separately from zero.
4. Build a review import/validation/aggregation path with item/run hashes, reviewer identity, timezone-aware review time, score ranges, missing/disputed items and any adjudication. No blank rubric is treated as completed review.
5. Use a small fixed smoke review first, then a predeclared representative dev/test review sample. Use two independent reviewers where feasible; report actual rater count/disagreement rather than claiming agreement that was not measured.
6. Audit Baseline 1 units, force semantics, persistence rules and available targets using train/dev only. Its current eight-pair smoke has seven abstentions and zero joint accuracy.
7. Add failure-only regression cases for each evidenced missed/incorrect diagnosis. If observables cannot identify a fault, retain abstention; do not import paired success or perturbation labels into B1.
8. Any rule-threshold tuning uses grouped train/dev, a recorded bounded search and a frozen selection before test.
9. Report onset only with independently verified reference time and a defined prediction timestamp. Keep injection, numerical divergence, and physical failure onset distinct. Never infer an onset from null evidence.

Tests:

- All-abstention real runs produce reviewable rows and zero diagnostic coverage; empty evidence cannot yield a misleading perfect aggregate.
- All intended variants remain in metrics when predictions are absent/refused.
- Review leakage/blinding, source mutation, invalid scores, missing items and reviewer provenance.
- Rule regressions, benign trajectories, all five mechanisms and competing symptoms.
- Group-aware uncertainty estimates with deterministic CPU resampling, grouping by (task, initial_condition_id), keeping paired/repeated executions together. Test against hand-computable fixtures.

**Gate:** quality reports are truthful, review packets can be completed/imported, and rule changes are justified by observables rather than labels.

### P06 — Reverify production data and prepare new immutable experiment bundles

**Runs:** Mac CPU. **Dependencies:** P02 for certified study source; P03/P04/P05 for final methods.

Track 3 is already complete. Reuse and verify it when its inputs/configuration are unchanged.

1. Verify the source dataset/handoff, 1,000 enriched BTs, 2,004 artifact hashes, condition-group split isolation, computed/unavailable stages and raw numerical-onset/compression reports.
2. If dataset, prompt representation or parser configuration changed, publish a new version with all affected input/source/config hashes; never patch old manifests.
3. Reuse production selection only when its exact dataset/config binding remains valid. A replacement dataset requires new train/dev selection evidence even when it selects the same k=5/scale=1.
4. Update experiments.SOURCE_PATHS and run_llm.implementation_provenance for every new executable helper. Extend portable source verification and historical replay handling deliberately. Config validation, GPU qualification, completed-output bounds, recorded-settings verification, receipt schemas and implementation identity must all dispatch the declared new protocol. They currently contain v3/observed_json_v1 assumptions; changing only the decoder would leave the new profile unlaunchable. Add integration tests that exercise the new profile through prepare, qualify, run and CPU score.
5. Prepare fresh smoke, complete fresh integration and dev bundles with one protocol/config. Preserve B1 outputs and every expected request. Do not silently select successful pairs.
6. Compile output constraints and count exact prompts on CPU. Store per-request input/output bounds, prompt hashes and context fit; do not open weights.
7. Rehearse each new execution/evaluation path using stubs, including new adapter-condition orchestration with fakes, then extract the transfer archive into a separate directory and verify every member hash/source binding.
8. Add a human-readable evidence summary to the tracked docs; keep large outputs under ignored runs/data paths.

Existing production CLI:

~~~bash
PYTHONPATH=src .venv/bin/python scripts/produce_bts.py \
  --batch "$BATCH" --handoff "$HANDOFF" \
  --config "$PARSER_CONFIG" --divergence-config configs/divergence.yaml \
  --selection "$SELECTION_RECEIPT" --output-dir "$BT_ROOT"
~~~

Do not execute it against an existing destination or merely to recreate already verified output.

**Gate:** all new bundles/transfer contents independently verify, and the portable CPU rehearsal succeeds.

### P07 — Run bounded real decoder validation and broader dev inference

**Runs:** Alienware generation only; Mac scoring/review. **Dependencies:** P03/P04/P05/P06 as applicable.

1. Predeclare a small diagnostic cohort from already observed fresh smoke pairs, spanning both tasks and four released modes. Keep test sealed.
2. Reuse P03's current-v3 branch measurements. Compare at most two prospectively declared candidate protocols, plus a memory-identical unconstrained/raw-response control where necessary. Retain invalid raw controls without treating them as accepted production output.
3. Avoid a large sweep. Use one resident checkpoint and a fixed sequential GPU queue. Record all conditions and failed candidates.
4. Compare full-cohort correctness, non-null coverage, abstention appropriateness, schema/evidence errors, natural/forced prose endings, token counts and memory. Forced diagnoses are not a fix.
5. If a mechanical bug is found, fix and verify it. If properly measured model behavior remains poor, report the negative result and proceed under a frozen, reviewed protocol rather than optimizing until a desired score appears.
6. Qualify each production bundle's longest fitting request with its exact source/config/model/device/runtime. Longer-context or changed attention/precision layouts need an appropriate new kernel probe.
7. Record CUDA allocated/reserved peaks, driver-free memory, allocator retry/OOM counters and host peaks as bounded diagnostics. The old logs cannot prove an internal allocator recovery mechanism.
8. Run the complete fresh cohort, including the separate excessive-speed cases; then the declared dev cohort.
9. Return only input-bound completions, receipts, logs and required metadata to the Mac. Run independent validation, metrics, plots and real explanation review there.
10. Preserve all context refusals in primary metrics. The old v3 fresh-all bundle has 44 fitting/10 refused requests; archived dev has 261 fitting/39 refused requests. Their longest fitting prompts are 61,864/62,329 tokens, longer than the successful 56,474-token smoke. Recount for the new representation; do not assume these counts persist.

Existing CLI shape after P06 prepares the named new bundle:

~~~bash
# Ubuntu, existing inference environment and verified model cache:
if python scripts/qualify_gpu_experiment.py \
  --bundle "$BUNDLE" --kernel-report "$KERNEL_REPORT" \
  --output-dir "$QUALIFICATION"
then
  python scripts/gpu_experiment.py run \
    --bundle "$BUNDLE" --qualification "$QUALIFICATION" \
    --output-dir "$GPU_RESULTS" --backend hf --inference-only
else
  echo 'Qualification failed; preserve its artifacts. Full inference was not started.'
fi

# Mac, after copying/checksumming the results:
PYTHONPATH=src .venv/bin/python scripts/evaluate_gpu_results.py \
  --bundle "$BUNDLE" --results "$RETURNED_GPU_RESULTS" \
  --output-dir "$CPU_EVALUATION"
~~~

Chain production generation only after qualification exits successfully, with pipefail if using tee. Every output directory is absent before execution.

**Gate:** real three-variant outputs/B1 comparisons, complete accounting, measured memory and completed declared quality review. Do not equate a protocol pass with an improvement.

### P08 — Create and publish genuine reviewed training/dev preferences

**Runs:** Mac CPU plus humans. **Dependencies:** stable P04/P07 prompts; P02 study dataset identity.

Code:

- New models/preference_review.py, scripts/prepare_preference_review.py and tests.
- Reuse models/training_data.py and scripts/prepare_dpo_data.py.

Implement:

1. Select prospective train/dev identities only, stratified by task/mode/condition where available; keep groups intact. Keep the annotation budget and intended experiment scope explicit.
2. For the first optimizer smoke, a bounded pilot such as 16 train and eight dev main preferences is sufficient to test the engineering path if genuinely reviewed; this is not a scientifically powered final training dataset. The full declared study cohort must be specified before optimization, using available review capacity.
3. Produce blinded candidate packets with exact variant-specific inference messages, source BT hash, prompt hash, source run/completion hashes, task semantics and evidence references. Candidate order is seeded.
4. Humans may choose, correct or write explanations grounded in actual observations. The agent may propose candidates but cannot claim it performed human review.
5. Import reviewer ID, timezone-aware timestamp, unique annotation ID, ordering rationale, exact BT/prompt hashes and chosen/rejected structured responses.
6. Both responses must satisfy the existing structured/evidence contract, differ semantically, and match the same task/pair/variant. Invalid JSON negatives are a different objective and must not quietly enter DPO.
7. Reject test identities, group/split changes, duplicates, missing review, stale prompts, unsupported evidence and synthetic/real mixtures.
8. Publish DPO and chosen-only SFT exports and verify their exact inference prompt equality. Preserve annotation provenance in sidecars, never model input.
9. Require nonempty reviewed train and dev rows for each trained variant. An empty dev export or synthetic demonstration cannot reach optimization.

Existing command, only after reviewed.jsonl exists and bound inputs are final:

~~~bash
PYTHONPATH=src .venv/bin/python scripts/prepare_dpo_data.py \
  --bt-dir "$BT_ROOT" \
  --preferences "$REVIEWED_PREFERENCES" \
  --manifest "$HANDOFF/dataset_manifest.json" \
  --config "$TRAINING_DATA_CONFIG" \
  --output-dir "$TRAINING_EXPORT"
~~~

Update the versioned training-data configuration to bind the final prompt/input protocol. Its current --config role is fingerprinting notes; real behavior is controlled by the formatter/API flags, so verify effective_configuration rather than assuming YAML notes change behavior.

**Gate:** training_acceptance=reviewed_annotations_supplied, immutable artifact hashes, genuine reviewer provenance, train/dev group isolation and exact complete-token audit.

### P09 — Implement and qualify an executable training profile

**Runs:** CPU implementation/tests; Alienware backward/optimizer qualification. **Dependencies:** P04 lengths; P08 for genuine project-data execution.

Code:

- models/training.py, scripts/train_adapter.py.
- New models/training_qualification.py and scripts/qualify_training.py.
- New versioned training configuration/requirements lock; retain old profiles as historical.
- Tests: test_training.py, new qualification tests and tiny CPU PEFT tests.

Implement:

1. Select mutually compatible Torch/CUDA, Transformers, TRL, PEFT, Accelerate, Datasets and optional quantization versions using official documentation and package metadata. Pin exact versions/builds and validate their actual APIs; do not invent replacement version numbers.
2. Preserve the working inference environment. Use a separate training environment and record Python/CUDA/device capability and package/build versions.
3. Audit exact complete chosen/rejected sequences, EOS/chat formatting, max prompt/sequence limits, collator masks and the actual trainer-processed IDs. Check the library's truncation defaults explicitly; reject any altered or silently truncated task evidence.
4. Start with BF16 LoRA only where measured lengths/memory make it plausible. If necessary, implement an explicit qualified CUDA quantized-training profile; quantization/offloading/reference-log-probability caching are declared method changes with correctness/provenance tests.
5. Verify frozen reference behavior and trainable parameters. If using reference-log-probability caching, bind it to exact prompt/response IDs, base revision, tokenizer and precision; compare cached and direct values on fixtures.
6. Replace generic admission assumptions only after measurements with a source/device/runtime/settings/sequence-bound qualification profile. Current total host/GPU memory is below the old 64/32 GiB floors. Do not simply lower those floors or add an unsafe override.
7. Implement a training qualification that performs actual chosen/rejected forward/reference passes, backward pass, optimizer update, dev evaluation, save, reload and adapter inference check. Include finite loss/gradients, expected changed trainable parameters, unchanged frozen base, optimizer state, and measured host/CUDA peaks.
8. Measure the actual longest complete alternatives and relevant length shapes for the intended export; input length alone is insufficient. Use fresh-process bounded probes and preserve failed receipts. A successful shorter example does not admit longer examples.
9. Make training manifests explicit about batch size, accumulation, checkpointing, optimizer/seed, precision/quantization, sequence limits, base/adapter/reference behavior and data hashes.
10. Keep the existing fresh-output-only launch behavior for the initial completion path. Implement resume only if measured run duration or interruptions make it necessary; require a fully matching source/config/export/tokenization checkpoint, including optimizer/RNG state. Source or data changes require a new run. Resume support is not a prerequisite for completing TODO 513.
11. If the full declared dataset cannot fit, report the precise resource/context blocker. Present a qualified alternative profile/cohort or larger hardware; do not pretend inference chunking proves training fit. Training uses use_cache=False and has gradients/optimizer requirements.

Tests:

- Real preference and synthetic-data boundaries, runtime drift, wrong GPU profile, token mismatch/truncation, labels/loss masks, frozen reference, cache misuse, nonfinite loss/gradients, zero optimizer updates, corrupt checkpoint, resume mismatch and save failures.
- Tiny actual CPU adapter save/reload and one-step optimizer tests where feasible, without loading the 8B model.
- Actual GPU optimizer qualification, not a fake backend test, before the declared run.

**Gate:** measured optimizer/save/reload success on genuine reviewed data at intended settings, or a precise unresolved hardware blocker. No unsupported claim that a single 5090 can train every current long prompt.

### P10 — Add source-bound adapter inference and evaluation

**Runs:** CPU implementation/tiny PEFT tests; Alienware real adapter inference. **Dependencies:** P03/P04 API freeze; P09 training receipt format.

Code:

- models/llm.py, scripts/run_llm.py, experiments.py, experiment_results.py, gpu_qualification.py.
- New adapter validation helper and versioned experiment-condition specification.
- Adapter/config/portable result tests.

Implement:

1. Add strict AdapterSpec or equivalent containing adapter location, training receipt/artifact digests, exact base revision, trained prompt variant, tokenizer/template/input representation, adapter type and precision expectations.
2. Verify inventory, safe paths, base model/revision, target modules/shapes, nonempty adapter weights and training/prompt compatibility before expensive loading.
3. Load PEFT adapters cache-only. Missing/incompatible adapters fail; no silent base-model fallback.
4. Preserve base inference when no adapter is configured. Keep a main-trained adapter confined to a separate main-only run; it must never silently modify baseline_2/3.
5. Retain model_variant as the input-view identity. Separate base and adapter conditions through run/experiment metadata/directories, not invented enum labels that break strict joins.
6. Extend portable experiment specs to declare intended variants/conditions explicitly. Existing gpu_experiment hardcodes all three LLM variants; do not reuse it for a main-only adapter without updating coverage/verification deliberately.
7. Record adapter hashes, training receipt, resolved base/tokenizer identity and runtime in each result; include new modules in source inventories.
8. Compare base main versus adapter main on the same dev identities, decoding, input representation, base precision, quantization, attention implementation and runtime. Run the adapter-disabled deployment base as the matched control, then enable the adapter. If deployment is quantized, the original BF16 baseline alone is not a matched adapter control. Keep the same-base four-variant ablation as a separate condition.
9. Do not compare a trained main against untrained baselines and claim divergence alone caused a difference.

Tests: wrong variant/base/revision/tokenizer, stale prompt protocol, missing/unexpected weights, corrupted receipts, base fallback refusal, disabling adapter restores base behavior, condition-aware joins and portable source verification.

**Gate:** tiny real PEFT round-trip plus actual 8B trained-adapter loading/generation/evidence verification. An adapter directory alone is not evaluated training.

### P11 — Execute reviewed-data optimization and dev evaluation

**Runs:** Alienware optimizer/generation; Mac evaluation. **Dependencies:** P08/P09/P10.

1. Run CPU data/hash/token preflight without loading weights.
2. Execute a small genuine reviewed-data optimizer smoke under the qualified profile. Save/reload the adapter and verify real parameter updates.
3. Run the declared training experiment with fixed seed, epochs, optimizer, accumulation and adapter settings; feed only train rows to optimization, dev only to evaluation/selection.
4. Start with one declared configuration. Avoid an open-ended tuning search just to obtain a favorable result.
5. Save adapter, tokenizer, source/export/config snapshots, checkpoints, token audit and completed training receipt.
6. Generate dev predictions for base and adapter under comparable conditions, return to Mac, independently verify and score them.
7. Complete the declared blinded human review, including abstentions and incorrect success claims.
8. Report full-cohort accuracy/F1, answer coverage, evidence support, explanation completeness, condition-group uncertainty, cost/tokens/memory and review results.
9. A quality regression or no improvement remains a real result. Record optimizer_acceptance=executed_on_reviewed_data separately from adapter_quality.
10. Mark TODO 513 complete only after genuine reviewed optimization and adapter evaluation both exist.

Existing CPU check shape; select the new qualified config, not the old incompatible default:

~~~bash
PYTHONPATH=src .venv/bin/python scripts/train_adapter.py \
  --data-dir "$TRAINING_EXPORT" --config "$TRAIN_CONFIG" --variant main \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-dir "$TRAIN_RUN" --check
~~~

The later --run uses the qualified training environment/profile. The planned qualification CLI must be implemented and documented in P09 before its commands are given to the user.

### P12 — Freeze and execute the held-out study once

**Runs:** Mac freeze/preparation; Alienware inference; Mac metrics/review. **Dependencies:** P02/P06/P07/P11 completed, or explicitly documented approved scope changes.

Before opening test predictions:

1. Freeze a tracked study specification: dataset/handoff/BT hashes; grouped split manifest; parser/compression/divergence choices; input/output protocol; model/tokenizer/adapter revisions; precision; exact decoding/budget/seed; selected checkpoint; training annotation/export hashes; metric/refusal policy; human-review sample/rubric and uncertainty method.
2. Declare primary full-cohort metrics and secondary common-fitting/answered-only analyses. Refusals/abstentions/errors remain in primary denominators.
3. Freeze same-base B1/B2/B3/main and the separately declared adapter condition. No post-test method/model selection.
4. Verify no test pair/initial-condition group entered annotations, training, threshold tuning or checkpoint selection. Structural parsing of test inputs is not a claim that held-out evaluation already occurred.
5. Prepare the 200 held-out pairs using the existing explicit boundary:

~~~bash
PYTHONPATH=src .venv/bin/python scripts/gpu_experiment.py prepare \
  --batch "$BATCH" --handoff "$HANDOFF" --bt-root "$BT_ROOT" \
  --tokenizer-dir "$TOKENIZER_DIR" --llm-config "$LLM_CONFIG" \
  --split test --final-study --output-dir "$TEST_BUNDLE"
~~~

Then:

- Qualify the exact longest fitting test request without using its correctness to change the method.
- Generate all intended base conditions and the declared adapter condition. Record every request/refusal and immutable raw output.
- Transfer to Mac and score independently. Compute paired, task/condition-group-aware uncertainty; do not bootstrap repeated episodes as independent initial conditions.
- Perform the predeclared human review and aggregate actual ratings.
- Report four represented release modes and the valid but unsupported excessive-speed class separately. Fresh excessive-speed engineering probes do not create held-out class support.
- Score onset only for independently validated timing references, with missing coverage reported.
- If execution fails, preserve the run and describe any justified rerun/source change. Do not erase a failed first study or silently tune after inspecting test results.

**Closes:** TODO 447 and the final-study requirement. A prospective method can produce disappointing results and still satisfy execution/reporting; do not claim improvement absent evidence.

### P13 — Assemble reproducible integration evidence and complete review

**Runs:** Mac CPU; authorized Git/PR workflow; actual human reviewer.

1. Run affected unit/contract suites after each change; run the full offline/native/tiny-HF suite at the final source freeze. Add tests for new entrypoints, including calibration --help.
2. Re-run fresh CPU contract/native integration when relevant producer/model-view interfaces changed. Verify public CLIs, same source/model state, strict evidence and nonempty inspected plots.
3. Verify certified dataset/handoff, all production artifacts, model/adapter/training receipts, final study inventories and split isolation.
4. Build a tracked lightweight integration report with exact commands, counts, hashes and limitations, linking to downloadable/checksummed large evidence rather than committing runs/weights.
5. Include representative pickup/transport raw/compressed/divergence plots, real four-variant plots, adapter comparison, human review, token/context coverage and memory diagnostics.
6. Preserve the existing dirty work, then create logically grouped commits including all intended untracked implementation files. Do not use git add . indiscriminately; inspect staged paths and exclude data/caches/weights/credentials.
7. Reproduce from a clean checkout/source archive using pinned dataset/model restoration, source/config locks and the documented CPU commands. Confirm no hidden untracked dependency is needed.
8. Prepare the integration PR description around final behavior and actual validation. Open/publish only within the user's authorization; otherwise leave the exact locally reviewable PR text/artifact list ready. Do not merge without authorization or fabricate another member's review.
9. Obtain the repository-required actual contract/evidence review. Do not send messages to reviewers unless instructed.
10. Reconcile every TODO checkbox against evidence. Distinguish historical deliverables from current runnable interfaces; update duplicate milestone entries consistently.
11. If exact historical inputs or real reviews remain unavailable, leave their items explicitly blocked with the missing input and owner. Do not mark the plan complete merely because all code was written.

**Closes:** TODO 575 and final integration review, after actual evidence attachment/review rather than a prepared draft alone.

## 5. TODO closure matrix

| Original line | Open requirement | Closure package / evidence |
| --- | --- | --- |
| 137 | Original calibration and producer/model/config revisions | P02A genuine original records + strict certification; P02B only with an explicit requirement revision |
| 199 | Archived origins and complete pre-injection agreement | P02A all-run timing/replay/prefix evidence; zero prehistory remains pending |
| 202 | Physical outcomes of all archived failure runs | P02A genuine terminal measurements + independent outcome verification |
| 213 | Exact original generator/model/calibration/time records | P02A complete versioned producer attestation and certified handoff |
| 447 | Comparable real HF and later held-out study | Existing smoke execution recorded in P00; dev P07 and final study P12 |
| 468 | Main/four-variant handoff and formatter | Execution/formatter already evidenced; reconcile P00, quality improvements/review P03-P08 |
| 510 | Main real inference/output delivery and causal explanation | Execution already evidenced; P03/P05/P07 address explanation quality and review |
| 513 | Reviewed preferences, GPU optimization, adapter evaluation | P08-P11 real human/GPU evidence |
| 572 | Real-model smoke | Already satisfied by verified v3 run; close in P00 without another GPU run |
| 575 | Release integration evidence attached/reviewed | P13 actual integration artifact/PR/review |

The 78 checked items are mapped in the companion JSON to preservation/regression work. Do not reopen completed parser/compression/hook implementations for cosmetic rewrites. The broken current calibration entrypoint is a concrete exception requiring P01 even though historical calibration generation was checked.

## 6. Execution commands, environment, and artifact conventions

All commands are from repository root unless noted. These variables are a run ledger, not arbitrary paths to overwrite:

~~~bash
# Mac
cd /Users/arikgershman/Desktop/CMSC723/telemetry-failure-explainability
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg
run_id=completion-20261119-r01
RUN_ROOT="$PWD/runs/$run_id"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

# Source/dependency baseline:
.venv/bin/python -m pytest -q
HF_TINY_NATIVE=1 TRACK2_REAL_SIM=1 .venv/bin/python -m pytest -q
~~~

Use a new run_id if a path exists. The full suite requires restored assets/dependencies; explicit resource skips in other environments are not native acceptance. Capture commands, exit codes, package versions and source/config hashes in receipts rather than relying on console prose.

At P06, write a run ledger resolving BATCH, HANDOFF, PARSER_CONFIG, SELECTION_RECEIPT, BT_ROOT, TOKENIZER_DIR, LLM_CONFIG and later TRAINING_EXPORT/TRAIN_CONFIG to verified artifacts. Freeze the ledger before running dependent jobs.

Ubuntu baseline:

~~~bash
cd "$HOME/CMSC723/telemetry-failure-explainability-alienware-20261118"
source .venv-gpu/bin/activate
export PYTHONPATH="$PWD/src"
export HF_HUB_CACHE="$HOME/CMSC723/telemetry-failure-explainability/weights/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
set -o pipefail
~~~

Verify the actual cache path; the dedicated checkout's weights/hub is the fallback if the original cache is absent. Reuse the existing checkpoint and successful inference environment. New dependencies belong in a pinned, tested environment; training uses a separate environment.

For each transfer:

- Source/config/bundle manifest plus archive SHA-256 and per-member inventory.
- No model weights, virtual environments, credentials or unrelated historical runs.
- Archive extraction/source verification and a portable CPU rehearsal before handing commands to the user.
- Fresh output paths, pipefail, and preservation of failed receipts.
- Mac performs all metrics, plots, annotation preparation, human-review aggregation and CPU token/data audits.

New names in this plan, including diagnose_decoding.py, audit_token_budgets.py, preference review helpers, qualify_training.py and AdapterSpec, are **planned deliverables**. Their proposed APIs are specifications to implement and test, not commands that already work. Existing command examples use checked current flags; replace configurations/paths only after the relevant package publishes them.

## 7. Verification checklist for the coding agent

Before declaring all TODO work finished, produce one completion report containing:

- A disposition/evidence link for all 88 original checkboxes, plus all new regression gaps discovered during implementation.
- Successful original-data certification or the user's explicit dataset/requirement revision; no pending/failed gate hidden as passed.
- 1,000 pair/2,000 telemetry counts, task-local identities, 50 conditions/task, balanced represented modes, immutable checksums and 700/100/200 grouped splits for the final study dataset.
- Verified parser/compression/divergence production artifacts with event retention and numerical-onset scope.
- Real same-base four-variant generation and independent scoring; all requests/refusals/abstentions/errors counted.
- Genuine completed human review, including abstention rationales.
- Reviewed train/dev preferences with exact prompt/BT hashes and no test leakage.
- Actual optimizer steps, saved/reloaded adapter and independently verified adapter dev/test inference.
- Exact runtime/source/model/tokenizer/adapter/config/data inventories.
- Frozen held-out study and all intended reported outcomes, including negative results.
- Full CPU/contract/native regression evidence and real GPU qualification receipts; fakes/stubs labeled synthetic.
- A clean-checkout reproduction and actual integration PR/evidence review.

If a dependency remains absent, report the precise blocked item rather than claiming “every checkbox complete.”

## 8. Primary API references for implementation

Consult the documentation for the versions actually selected and pin those versions in the run. Current pages are research starting points, not proof of compatibility with an untested environment:

- [TRL DPO trainer](https://huggingface.co/docs/trl/dpo_trainer): reference model, preference formatting, trainer configuration and token-length behavior.
- [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization): optional quantized adapter training; it still requires project/hardware qualification.
- [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation): logits processing, generation settings and supported execution controls.
- [PyTorch CUDA memory](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management): allocator diagnostics and memory measurements.

Do not upgrade working environments merely because newer packages exist. Choose a minimal compatible profile, test it, freeze it, and keep the evidence.

