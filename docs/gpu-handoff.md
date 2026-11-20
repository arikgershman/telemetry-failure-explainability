# CPU completion and GPU handoff

The CPU work selects compression using dev data, publishes all 1,000 enriched
BTs, verifies fresh physical executions, validates the preference formatter,
and freezes exact-tokenized inference inputs. Stub responses establish pipeline
behavior only. The historical archive's failed/pending certification is deferred.

## Current portable inputs

| Bundle under `runs/` | Pairs | LLM requests | Context preflight |
| --- | ---: | ---: | --- |
| `alienware-fresh-smoke-v2-20261119` | 8 fresh physical pairs, one per released mode/task | 24 | All fit. Selection depends on prompt fit, not prediction accuracy. |
| `alienware-fresh-all-v2-20261119` | All 18 verified fresh physical pairs | 54 | 44 fit; 10 explicit refusals. No cases dropped. |
| `alienware-v1-dev-v2-20261119` | All 100 grouped archived dev pairs | 300 | 261 fit; 39 explicit refusals. Historical certification remains unresolved. |

The fresh bundles reverify all 362 original native artifacts, repeated raw
trajectories, physical success/failure goals, producer/model bytes, pre-injection
coverage, numerical onset origins, and BT reconstruction from stored telemetry.
Their IDs include task names. They are integration cases, not a held-out study.
No new simulator run is needed on the GPU machine.

Each bundle copies BTs, metadata, task manifests, CPU rule predictions, exact
configuration/templates, and byte-locked tokenizer inputs. Its manifest binds
source files, every input, and every prompt/token count. The eight-pair smoke
covers both tasks and all four released failure modes. Two excessive-speed
activation probes are covered separately by the native CPU receipt; they are
not invented accepted failure pairs.

## Ubuntu inference and Mac evaluation

Use the complete [Ubuntu 24.04 commands](alienware-ubuntu24.md), followed by the
[Nov. 19 inference fix](alienware-inference-fix-20261119.md) before generation. They transfer
this working source and the prepared inputs, reuse the Alienware checkpoint
cache, create a separate inference environment, and run only model generation
and response validation on the GPU:

```bash
python scripts/gpu_experiment.py verify --bundle runs/alienware-fresh-smoke-v2-20261119
python scripts/gpu_experiment.py run --inference-only \
  --bundle runs/alienware-fresh-smoke-v2-20261119 \
  --output-dir runs/alienware-fresh-smoke-hf-retry-20261119
```

After returning results to the Mac, run all four variants' metrics, blinded
review sheets, and plots without loading weights:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_gpu_results.py \
  --bundle runs/alienware-fresh-smoke-v2-20261119 \
  --results runs/alienware-retry-return-20261119/returned/runs/alienware-fresh-smoke-hf-retry-20261119 \
  --score-response-errors --output-dir runs/alienware-retry-evaluation-20261119
```

The CPU evaluator independently verifies the immutable GPU inventory, source
and input hashes, model revision, generation settings, CUDA/runtime/memory
provenance, exact prompts/token counts, and reparsed completions. It preserves
the returned inputs and publishes hashed success/failure receipts. It does not
require Transformers, Torch, a tokenizer runtime, network access, or a GPU.

All three LLM variants use the same pinned Llama-3.1-8B-Instruct checkpoint,
greedy decoding, seed 723, BF16, batch size 1, 65,536-token context budget and
512-token output limit. The production v2 response policy is `json_or_fenced_v1`; schema and evidence remain strict.
Unexpected malformed JSON, unsupported evidence, missing output, runtime
failure, or source mutation fails acceptance and retains diagnostics.

Full-bundle context refusals remain scored errors. A run may finish
`passed_with_context_refusals` while full `real_model_acceptance` is false.
Some archived dev prompts exceed even the model's native 131,072-token limit;
more VRAM cannot make those unchanged prompts fit. A future context policy
requires separate versioning and dev validation. Do not truncate or remove
observations to make the current acceptance gate appear complete.

The fresh rule baseline achieves only 1/18 joint accuracy. That is a measured
baseline-quality limitation. Valid JSON and faithful evidence also do not
establish explanation quality: human review remains a separate deliverable.

## Reproduce CPU preparation

Use new output directories. No weights or GPU are opened:

```bash
PYTHONPATH=src USE_TORCH=0 USE_TF=0 .venv/bin/python scripts/prepare_fresh_experiment.py \
  --native-run runs/integration-mujoco-cpu-20261118 \
  --tokenizer-dir data/raw/tokenizers/llama31-8b-0e9e39f \
  --smoke --output-dir runs/fresh-smoke-repeat
PYTHONPATH=src .venv/bin/python scripts/gpu_experiment.py run \
  --bundle runs/fresh-smoke-repeat --backend stub --inference-only \
  --output-dir runs/fresh-smoke-repeat-stub
PYTHONPATH=src .venv/bin/python scripts/evaluate_gpu_results.py \
  --bundle runs/fresh-smoke-repeat --results runs/fresh-smoke-repeat-stub \
  --output-dir runs/fresh-smoke-repeat-evaluation
```

Omit `--smoke` to retain all 18 physical cases. The archived dev preparation uses
`scripts/gpu_experiment.py prepare`, the restored batch, frozen dataset handoff,
and production BT root; see its `--help`. Test selection requires an explicit
`--final-study`; no prepared bundle here opens held-out test data.

Earlier `gpu-smoke-20261118` / `gpu-dev-20261118-final` bundles and CPU receipts
remain historical. Source changes intentionally invalidate them for current
execution; use the current `alienware-*` bundles.

## Reviewed preferences and training

The [formatter](training-data.md) is implemented and tested. Its 24-record local
demonstration is explicitly synthetic and cannot be optimized. Real reviewed
train/dev preferences remain required. Adapter training is outside steps 7–11;
there is no training command to run for this inference handoff. The separate
[training guide](gpu-training.md) records its resource/context requirements,
which the existing default configuration and hardware do not yet satisfy.
