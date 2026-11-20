# Reviewed-data GPU adapter training

The optional `scripts/train_adapter.py` entrypoint prepares one **LoRA/DPO
adapter for one explicit model variant**. Its default mode only checks CPU
inputs. Optimization, trained-adapter inference, and quality acceptance have not
run on this laptop. No reviewed project preference dataset exists yet; the
synthetic formatter demonstration is explicitly refused by this launcher.

The launcher uses the same immutable Llama-3.1-8B-Instruct checkpoint and
revision as inference. It supports a dedicated single-process CUDA/BF16 profile,
with a frozen base model as DPO reference by disabling the trainable PEFT
adapter. It does not merge adapters into base weights, upload artifacts, run on
the CPU/MPS, or mix baseline/main prompt variants.

## Before optimization

Create reviewed train and dev preferences using `scripts/prepare_dpo_data.py`.
Each annotation must retain its reviewer, timestamp, rationale, unique annotation
ID, and exact BT/prompt hashes. Test preferences never enter the export. The
formatter's `training_acceptance` must be `reviewed_annotations_supplied`;
`synthetic_demo_only` cannot pass checking or reach GPU preflight.

Run the default CPU check against a fresh, absent intended output path:

```bash
PYTHONPATH=src python scripts/train_adapter.py \
  --data-dir data/processed/reviewed-preferences \
  --config configs/training-gpu.yaml --variant main \
  --output-dir runs/main-dpo-v1 --check
```

This imports no Torch, Transformers, TRL, PEFT, or Datasets runtime and performs
no downloads or weight loading. It verifies portable export bytes/counts,
complete unique sidecar row coverage, reviewed annotation provenance, exact
prompt/response hashes, matching chosen-SFT views, and disjoint task/initial
condition train/dev groups. Every task-local pair retains its group across
variants. One explicit variant must have nonempty train and dev rows.
`--check-original-inputs` additionally verifies original local formatter inputs;
omit it after transferring an already verified export to another machine.
Neither mode upgrades the archived dataset's certification status.

After reviewed annotations exist, add
`--tokenizer-dir data/raw/tokenizers/llama31-8b-0e9e39f` to this CPU check. This
optional mode verifies the tokenizer's pinned byte inventory and computes exact
prompt/completion budgets locally, without model weights or device creation.
It can import tokenizer/runtime helpers, unlike the default dependency-light
check. Choose and record a training context budget using this audit before
launching optimization. The current production dev audit found a minimum main
prompt of 25,309 tokens before completions, so the default 8,192-token starting
profile cannot fit any of those main records. It is not a practical training
claim for the current main dataset or a 24 GiB GPU.

## Dedicated GPU environment and launch

Use a separate Python 3.11 environment with the CUDA build of Torch 2.6.0
appropriate for the machine, then install the pinned profile and this project:

```bash
python -m pip install -r configs/training-gpu.requirements.txt
python -m pip install -e .
```

These explicit API pins are separate from the inference environment; they are
not a claim that real optimization has been validated. The implementation was
checked against the official [TRL 0.15.2 DPO API](https://huggingface.co/docs/trl/v0.15.2/dpo_trainer),
its [tokenization implementation](https://github.com/huggingface/trl/blob/v0.15.2/trl/trainer/dpo_trainer.py),
and [PEFT 0.14.0 LoRA API](https://huggingface.co/docs/peft/v0.14.0/package_reference/lora).
The GPU process refuses incompatible installed versions.

Download/access the exact checkpoint on that machine before running. Loading
is cache-only (`local_files_only: true`) and does not execute remote model code:

```bash
hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659
PYTHONPATH=src python scripts/train_adapter.py \
  --data-dir data/processed/reviewed-preferences \
  --config configs/training-gpu.yaml --variant main \
  --output-dir runs/main-dpo-v1 --run
```

The default profile requires verified 64 GiB total/48 GiB currently available
host RAM before importing training runtimes, then a BF16 CUDA GPU and at least
32 GiB free/total VRAM. The resource floor rises to 48/64/80 GiB for maximum
sequence budgets above 8,192/16,384/32,768 tokens respectively. These are refusal
floors, not guarantees that a particular workload fits. Batch size is fixed at
one preference pair, with gradient accumulation and gradient checkpointing.

The default `max_length: 8192` is an explicit starting budget. Production BT
prompts can exceed it. Before weight loading, the exact pinned tokenizer renders
each system/user/assistant conversation and counts both complete alternatives,
including TRL's appended EOS token. Oversized records fail with their split,
row, and required token count. No record, task stage, or comparison is silently
truncated. Increase the declared budget only with sufficient hardware; changing
the profile does not justify tuning on held-out data. The actual trainer's
processed IDs are also checked against the preflight IDs before optimization.

## Receipts and acceptance

Setup, resource and context refusals create no training output directory.
Once optimization setup starts, the new directory retains
`training_run_manifest.json` with running/completed/failed status and, on
failure, the error and any partial checkpoints. Existing directories are always
refused. Use a fresh output for another attempt.

A completed run saves:

- `adapter/`: PEFT adapter configuration and safetensors weights.
- `tokenizer/`: the tokenizer used for exact formatting and token preflight.
- `training_config.yaml` and `training_data_manifest.json`: exact input snapshots.
- `checkpoints/`: trainer checkpoints/state under the fixed save policy.
- `training_run_manifest.json`: model revision, variant, annotation/split
  provenance, source/config/export hashes, runtime versions, resource admission,
  token counts, metrics, adapter/tokenizer artifact hashes, and unchanged inputs.

`optimizer_acceptance: executed_on_reviewed_data` records successful execution,
not an improvement claim. `adapter_quality` remains `not_evaluated`. Evaluate
the adapter on dev first under the same structured response and evidence
contracts; reserve held-out test evaluation for the later final study. The
chosen-SFT export remains available for future supervised training, while this
entrypoint intentionally exposes DPO only.

The CPU suite uses isolated fake CUDA/model/trainer objects to test successful
adapter publication, exact-token mismatch rejection, optimizer failures,
synthetic-data refusal, memory admission, and clean CLI errors. It never loads
weights or manufactures real reviewed project labels:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -W error -m pytest tests/test_training.py tests/test_training_data.py
```
