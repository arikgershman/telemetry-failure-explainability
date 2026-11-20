# Source-bound preference data

`scripts/prepare_dpo_data.py` formats reviewed preferences entirely on the CPU.
It uses the exact `build_prompt()` function used by inference. It does not
generate human annotations, load model weights, tokenize, or run an optimizer.

The formatter accepts the frozen version-2 dataset manifest and task-scoped
enriched Behavior Trees:

```text
BT_ROOT/pick_up_object/enriched-bt/pair_<id>_bt.json
BT_ROOT/transport_object/enriched-bt/pair_<id>_bt.json
```

Pair IDs are task-local. Every preference must name both `task` and `pair_id`.
The formatter verifies embedded BT identity, source filenames, failure-mode
storage metadata, manifest inventory digest, initial-condition assignments, and
group isolation across the complete train/dev/test manifest. Labels and source
filenames stay in sidecars; the inference prompt removes them. The `main`
variant requires complete computed divergence on every observed stage.

## Annotation contract

Input is JSONL with exactly these outer fields:

```json
{
  "task": "pick_up_object",
  "pair_id": "000001",
  "model_variant": "main",
  "chosen": "{\"predicted_failure_mode\":null,\"predicted_subtask\":null,\"explanation\":\"Example only; replace with a reviewed response.\",\"evidence\":[],\"status\":\"abstain\"}",
  "rejected": "{\"predicted_failure_mode\":null,\"predicted_subtask\":null,\"explanation\":\"Different example response; replace before review.\",\"evidence\":[],\"status\":\"abstain\"}",
  "preference_provenance": {
    "kind": "human_reviewed",
    "annotation_id": "unique-review-record-id",
    "reviewer_id": "actual-reviewer-id",
    "reviewed_at": "2026-11-18T10:00:00-05:00",
    "rationale": "Replace with the reviewer's explanation for this ordering.",
    "bt_sha256": "REPLACE_WITH_EXACT_BT_FILE_SHA256",
    "prompt_sha256": "REPLACE_WITH_BUILD_PROMPT_MESSAGES_SHA256"
  }
}
```

This example is documentation, not a reviewed preference. Hash values must be
real lowercase SHA-256 digests. Obtain the prompt hash with
`build_prompt(validated_tree, variant).prompt_sha256`; the BT hash covers the
exact file bytes. Binding both hashes prevents an annotation from silently
following a changed parser, divergence result, template, or prompt view.

Each choice is a strict JSON string with the same five fields that inference
requests. Both choices must be valid `ok` or `abstain` responses. Every evidence
citation must match an actual retained failure-side observation. Invalid JSON,
duplicate keys, nonfinite numbers, fabricated measurements, extra identity
fields, unknown variants, blank choices, and semantically identical responses
are refused. Rejected responses can still be less persuasive or diagnostically
incorrect; this formatter requires structured, grounded responses on both
sides. A separate objective would be needed to train against malformed-output
negatives. Response fields are normalized through the shared validator before
export; original annotation bytes remain bound by the preferences-file hash.

`human_reviewed` requires a named reviewer, a timezone-aware ISO-8601 review
timestamp, a unique annotation ID, and rationale. These are retained reviewer
assertions. Formatting cannot independently establish reviewer identity or
annotation quality. Humans must actually review the pair and causal reasoning.

## Run and inspect

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_dpo_data.py \
  --bt-dir data/processed/production-v1-20261118 \
  --preferences data/processed/preferences/reviewed.jsonl \
  --manifest runs/dataset-v1-20261118/handoff/dataset_manifest.json \
  --config configs/training-data.yaml \
  --output-dir data/processed/training/reviewed-v1
```

The output directory must be absent. All records are validated before staging;
input and implementation hashes are rechecked before an atomic publication.
An invalid input or failed write publishes no dataset. Outputs must be outside
the BT input tree. Symlinked BT paths and hardlinked input files are refused.
`--bt-subdir` supports a different safe relative layout; `--template-dir`
supports the same controlled-template contract as inference. `--config`
fingerprints configuration notes; formatter behavior is recorded separately in
`effective_configuration` and is controlled by the explicit CLI arguments.

Published files:

| File | Purpose |
| --- | --- |
| `dpo/train.jsonl`, `dpo/dev.jsonl` | Conversational `prompt`, `chosen`, `rejected`; each value is a list of role/content messages. |
| `sft/train.jsonl`, `sft/dev.jsonl` | Chosen-only conversational `messages`. |
| `records.jsonl` | Task, pair, variant, condition, split, row index, annotation provenance, BT/prompt/completion hashes. |
| `training_manifest.json` | Input/source/config/template/output hashes, counts, validation scope, and explicit preference mode. |

Train and development rows remain separate. Empty split files are emitted with
zero rows. Held-out test preferences are refused, including test rows mixed
into an otherwise valid input. The formatter never changes split assignments.
The original dataset's unresolved certification remains visible in the receipt;
successful formatting does not repair historical simulation provenance.

## Synthetic CPU demonstration

Synthetic preferences use `kind: synthetic_demo`, `generator_id`, annotation ID,
rationale, and exact BT/prompt hashes. They forbid reviewer/timestamp claims and
require explicit `--synthetic-demo`. Reviewed and synthetic records cannot be
mixed. A synthetic receipt has `training_acceptance: synthetic_demo_only`; it
proves formatting and grounding contracts, not label quality or training-data
readiness. A real training run must require
`training_acceptance: reviewed_annotations_supplied` and verify the receipt's
artifact hashes first.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_training_data.py
```

Tests run with network and optional model runtimes blocked. They cover both
tasks and all three prompt variants, strict provenance, exact inference prompt
equality, leakage checks, split isolation, invalid inputs, source mutation,
nonmutation of BTs, and atomic publication failure.

## Later GPU work

`configs/training-data.yaml` records the handoff contract and intended DPO/LoRA
workflow. It deliberately leaves adapter and optimizer hyperparameters to an
explicit recorded experiment configuration. Apply the same pinned model,
tokenizer revision, and tokenizer chat template as inference to these message
lists. Do not concatenate a second system prompt or add ground-truth labels.
Perform a token-budget preflight on every chosen and rejected sequence; refuse
overlong examples rather than truncating retained task evidence. Feed only the
train export to optimization and use dev for model selection. Keep test sealed
until the held-out study. Record adapter parameters, optimizer/seed settings,
tokenization policy, runtime versions, source receipt, and resulting artifact
hashes in the training run. The SFT export supports an optional reviewed
chosen-response warm-up using the same split and input-view rules.

Human-reviewed preferences are still required before meaningful preference
training. Synthetic CPU demonstrations cannot supply that missing supervision.

## Verify a publication

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_dpo_data.py \
  --verify runs/training-cpu-20261118/formatted-synthetic
```

This checks all artifact hashes and row counts, refuses unexpected files, and
rechecks the original local annotation, manifest, BT, source, configuration, and
template fingerprints. After transferring only the published output, add
`--artifacts-only` to omit unavailable original-input checks. This reduced scope
does not change the reported annotation mode. GPU consumers can call
`verify_training_data()` and must additionally require reviewed annotation
acceptance before real preference optimization.
