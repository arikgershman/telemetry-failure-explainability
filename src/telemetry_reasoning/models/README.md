# Track 5 LLM harness, prompt baselines, and inference verification

Track 5 provides one model adapter, an offline stub, and controlled prompt
views. The Nov. 6 extension supports batched HF inference, records raw model
completions, validates structured predictions against observed telemetry, and
adds experimental Metal quantization support. The attempted full model load on
the local 16 GB Apple M4 made the laptop unresponsive and required a hard
reboot. That machine is unsupported for automatic full-model loading. The
November 16 Alienware run generated real HF responses for both baselines;
CPU-only response revalidation handles its single JSON code fence under an
explicit, versioned policy. The original strict run remains preserved as
incomplete. No local full-model retry is needed or authorized.

The Nov. 6 milestone is complete under the recorded response policy. See the
[acceptance receipt](../../../docs/track5-nov06-acceptance.md) for hashes,
verification commands, and explanation-quality limitations. No further GPU work
is required for that milestone.

The stub checks orchestration and contracts; it does not diagnose failures or
establish LLM accuracy. A real main-model run, training-data formatting,
preference collection, and LoRA/DPO training remain later Track 5 work.
Simulation, parser, dataset validation, and evaluation implementations remain
with their owning tracks.

## Local model-load incident

The failed command was the following. It is recorded for audit and must not be
rerun on the 16 GB laptop:

```text
PYTHONPATH=src HF_HUB_CACHE=weights/hub .venv/bin/python -u scripts/run_llm.py --bt-dir data/sample --config configs/llm-local.yaml --variants baseline_2 baseline_3 --backend hf --output-dir runs/track5-nov06/local-hf
```

The last captured output showed weight loading at approximately **87%
(253/291)** before the process stalled. The user reported that the laptop
became unresponsive and required a hard reboot. No completed result directory
exists at `runs/track5-nov06/local-hf`; there are no accepted prediction files
or completed run manifest for that attempt. The CLI publishes output only
after inference returns, so the absent directory establishes that publication
did not finish, not an exact internal failure point.

The later process audit found no matching inference task still running. An
after-reboot swap check showed zero current swap usage, and the available
read-only diagnostic files did not establish a relevant panic or confirmed
out-of-memory termination. Those later observations cannot reconstruct memory
pressure before the reboot.

The observed load stage and the 16 GB unified-memory limit make memory pressure
during model loading and on-the-fly quantization a plausible cause. Loading can
temporarily retain original tensors, converted tensors, and working buffers;
the final 4-bit weight size does not bound that peak. A later long-prompt
prefill would also require substantial memory, but the captured output does
not establish that prefill began. No OS report confirming the cause,
memory-pressure trace, or confirmed peak allocation was captured, so this is
an inference rather than a diagnosed OS failure.

Downloading the original checkpoint placed approximately **16 GB on disk** in
`weights/hub`. A cache file's disk size is different from a running process's
RAM/GPU allocation. The heavy step was the later `--backend hf` command, which
called `AutoModelForCausalLM.from_pretrained()` with MPS placement and runtime
Metal quantization. The cache download, package installation, isolated kernel
probe, and tokenizer checks did not prove that this full load would fit.
The readiness assessment was incomplete: it allowed full-model loading after
kernel/token checks without first establishing sufficient peak unified-memory
headroom. The local profile should not have been treated as ready to execute.

The earlier local profile was an unverified experiment. Successful tokenization
and kernel compatibility must not be presented as evidence that full-model
inference is viable on this laptop. Preserve the existing configuration,
dependency pins, and cache for audit without reloading the model. Current work
is limited to source/documentation inspection and CPU/offline validation that
does not import Torch, access the GPU, download files, or run model inference.
The subsequent Alienware generation used identical model/settings for both
baselines. Its saved responses can be revalidated on the Mac without opening
weights, importing model runtimes, or allocating GPU memory.

## Model identity and reproducibility

Use the instruction-tuned `meta-llama/Llama-3.1-8B-Instruct` checkpoint for all
three variants. Model and tokenizer both use the immutable commit
`0e9e39f249a16976918f6564b8830bc894c89659`, verified against the
[official Hugging Face model API](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct)
on November 12, 2026. Keep the SHA pinned when the upstream default branch
changes. A changed checkpoint, tokenizer, prompt, decoding setting, or precision
defines a new experiment and needs a new output directory.

Meta documents a 128k context and BF16 inference in the
[official model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
The reference configuration caps smoke tests at 65,536 total tokens,
deliberately below that model limit. That budget includes the chat template,
complete telemetry payload, and reserved output tokens. Count the actual model tokenizer's tokens before
generation; reject an oversized request instead of silently dropping stages or
evidence. A stub's approximate counts cannot establish that an HF prompt fits.

The reference configuration selects CUDA/BF16, greedy decoding, batch size 1,
a 512-token output allowance, and seed 723. The model and tokenizer share the
configured revision. `local_files_only: true` prevents an HF run from fetching
uncached files; make a separately named experiment configuration with that flag
set to `false` when intentionally retrieving the gated checkpoint. Quantization
is disabled in this reference configuration. The historical Metal
4-bit configuration below uses the same pinned original checkpoint and
tokenizer; its attempted full load failed on the 16 GB laptop.

The tokenizer's own chat template supplies special tokens and the assistant
generation prefix; see the
[Transformers chat-template guide](https://huggingface.co/docs/transformers/en/chat_templating).
Use identical instructions, output schema, token budget, decoding settings, and
seed across variants. Only the permitted telemetry view changes:

| Variant | Input view | Excluded information |
| --- | --- | --- |
| `baseline_2` | `tree.to_failure_only()` | Success samples and all divergence |
| `baseline_3` | `tree.to_paired_without_divergence()` | All divergence |
| `main` | `tree.to_full_paired()` | No additional telemetry exclusion; observed stages must have computed divergence |

All variants remove the opaque pair ID from prompt content and exclude answer
labels, perturbations, seed, and source filenames. The pair ID is retained only
for joining predictions to separate scoring metadata. A missing failure suffix
remains unavailable: a null divergence for an unobserved stage is permissible,
while null divergence for an observed stage rejects main-model input. Null
never means zero.

## Inference and output contracts

`LLMConfig` and `load_llm_config()` belong to Track 5; configuration stays under
the separate `llm:` YAML mapping. Paths such as `cache_dir` and `prompt_dir`
resolve relative to that YAML file. The same model, tokenizer revision, prompt
instructions, decoding settings, precision, quantization, and seed apply to
every variant in a run.

`batch_size` controls HF throughput without changing the allowed telemetry.
The adapter uses decoder-only left padding with explicit attention masks,
counts each unpadded input separately, and removes generated padding after the
first EOS. An overlong input becomes an error before generation; it is never
silently truncated or replaced by an incomplete task. Start with batch size 1
on memory-constrained hardware, then use a separately recorded configuration
to increase it. A batch execution failure remains an explicit failure for its
requests rather than silently retrying a different batch composition.

The model supplies exactly `predicted_failure_mode`, `predicted_subtask`,
`explanation`, `evidence`, and `status`. The adapter supplies the trusted join
fields `pair_id`, `model_variant`, and `backend`. It rejects malformed JSON,
duplicate/extra keys, invalid enum values, unavailable-stage predictions,
nonfinite/wrong-shape evidence, and citations that do not match a supplied
failure sample's timestamp and numeric value. A claimed `ok` prediction needs
both labels and supporting evidence. Abstentions retain null labels; errors
retain null labels and empty evidence. No response repair invents a diagnosis.

`response_policy` is separate from generation settings. `strict_json_v1` is
the default and the implicit policy for older manifests without this field.
`json_or_fenced_v1` accepts bare JSON or one whole-response lowercase JSON code
fence, with LF/CRLF marker lines and optional surrounding JSON whitespace.
It removes only the marker lines; the enclosed content must pass the same
duplicate-key, finite-number, schema, task, and exact observed-evidence checks.
It does not search prose for JSON, combine multiple objects/blocks, repair
syntax, change labels, or rewrite explanations. Prompts still request bare
JSON, and the selected policy applies uniformly to every variant.

Every new completion retains its exact raw text/hash plus
`response_normalization` containing the policy, envelope type, and SHA-256 of
the text submitted to the strict JSON parser. Live inference records the policy
in its config and manifest. Independent acceptance rechecks normalization as
well as rebuilding prompts and reparsing responses.

The CLI writes a fresh output directory containing:

- `<variant>.jsonl`: one validated prediction or explicit error per pair/variant.
- `completions.jsonl`: exact returned model text with parsing outcome, token
  counts, and a completion hash, including text rejected by validation.
- `run_manifest.json`: input/config/prompt/source hashes, requested and resolved
  model/tokenizer revisions, runtime/package/device/quantization provenance,
  per-request timing/counts, and separate execution and acceptance summaries.

Exit codes are 0 for a completed run of diagnoses/abstentions, 1 when any
prediction is an error, and 2 for configuration, input, or publication errors.
Selecting `--backend hf` alone does not establish real-model acceptance. A
passing inference smoke requires real generation and validated results for
every requested pair/variant; it does not establish diagnostic accuracy.

## Offline revalidation of saved HF responses

`scripts/revalidate_llm.py` reads a transferred run bundle containing
`config.yaml`, `input/pair_<id>_bt.json`, and `inference/` with the original
manifest, prediction JSONL, and exact raw completions. It verifies parent
artifact/config/input hashes, request identities, generation metadata, and
prompt/template compatibility before applying one declared response policy.
It never instantiates a backend, imports model runtimes, downloads weights, or
uses a GPU. Changing the prompt, model, decoding, precision, or observations
requires new inference instead of this replay.

Run from the repository root, choosing an absent or empty output directory:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/revalidate_llm.py \
  --bundle-dir runs/alienware-import-20261116T215259Z \
  --response-policy json_or_fenced_v1 \
  --output-dir runs/track5-nov06/accepted-baselines
```

Derived artifacts preserve raw completion text, generation provenance,
token/timing records, parent hashes, and original validation outcomes. New
validator source hashes, normalization records, and acceptance describe the
offline operation separately, with zero newly generated requests. The source
bundle and its original strict acceptance are never overwritten. Stub outputs
cannot establish real-HF acceptance, and unsupported evidence or malformed JSON
remain errors. Exit codes are 0 for accepted diagnoses/abstentions, 1 for
response errors, and 2 for invalid input/provenance or publication failures.

Passing this smoke on one mock pair establishes the inference/output contract,
not a dataset accuracy estimate or causal explanation quality. Preserve model
prose unchanged and record human review findings separately from format,
label, and structured-evidence validation.

## Offline CPU smoke

Use the existing project environment from the repository root. This path does
not need the `llm` extra, a Hugging Face account, network access, or model
weights. During safe recovery work, use installed packages without installing
or updating dependencies.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_prompts.py tests/test_llm.py
```

Run the two baselines directly against a copy of the checked-in fixture in a
fresh input directory. `run_llm.py` reads `pair_*_bt.json`, leaving report files
outside that explicit input pattern.

```bash
mkdir -p /tmp/track5-sample-bt
cp data/sample/pair_000_bt.json /tmp/track5-sample-bt/pair_000_bt.json
python scripts/run_llm.py \
  --bt-dir /tmp/track5-sample-bt \
  --config configs/llm.yaml \
  --variants baseline_2 baseline_3 \
  --backend stub \
  --output-dir runs/track5-stub-baselines
```

The checked-in fixture intentionally has uncomputed divergence. To exercise all
three variants without waiting for Track 3's real divergence handoff, make a
separate, explicitly synthetic copy. This snippet changes no source fixture and
does not implement or validate a divergence algorithm:

```bash
python - <<'PY'
from pathlib import Path
from telemetry_reasoning.schemas import (
    DeterministicDivergence, PairedBehaviorTree, Subtask, TelemetryFeature,
)

tree = PairedBehaviorTree.model_validate_json(
    Path("data/sample/pair_000_bt.json").read_text()
).model_copy(deep=True)
for node in tree.subtasks:
    if not node.failure_samples:
        continue
    deltas = {feature: 0.0 for feature in TelemetryFeature}
    onset = None
    if node.subtask in (Subtask.GRASP, Subtask.LIFT):
        deltas[TelemetryFeature.GRIPPER_FORCE] = 12.0
        onset = 5.10 if node.subtask == Subtask.GRASP else 6.0
    if node.subtask == Subtask.LIFT:
        deltas[TelemetryFeature.OBJ_POS] = 0.20
    node.divergence_status = "computed"
    node.deterministic_divergence = DeterministicDivergence(
        first_divergence_timestamp=onset,
        max_abs_deltas=deltas,
    )
PairedBehaviorTree.model_validate(tree.model_dump(mode="json"))
target = Path("/tmp/track5-synthetic-divergence-bt")
target.mkdir(parents=True, exist_ok=True)
(target / "pair_000_bt.json").write_text(tree.model_dump_json(indent=2) + "\n")
PY
python scripts/run_llm.py \
  --bt-dir /tmp/track5-synthetic-divergence-bt \
  --config configs/llm.yaml \
  --variants baseline_2 baseline_3 main \
  --backend stub \
  --output-dir runs/track5-stub-synthetic-divergence
```

Every prediction must identify `backend="stub"`; a deliberate abstention is the
expected offline result. Keep these files separate from real model results.
The stub bounds input tokens by UTF-8 bytes plus a 32-token framing allowance,
explicitly labeled `utf8_bytes_upper_bound` in its manifest. Its output count
also uses UTF-8 bytes. These counts deliberately differ from tokenizer counts;
the 65,536-token configuration admits the complete checked-in sample for all
three variants. The stub never loads the real tokenizer and does not silently
truncate oversized prompts.

The synthetic 5.10 s / 12 N value is a contract fixture, not a measured output
of a production divergence function. Do not report its predictions or metrics
as evidence that Llama identified the grasp event.

## Safe local workflow and compute gates

The current authorized work is source/documentation review and CPU/offline
validation only. Do not install packages, download more files, import Torch,
query or initialize the GPU, retry model loading, or provision external compute.
Use the existing environment and the stub. A passing offline test suite remains
separate from real-HF acceptance.

The resource preflight runs before optional model runtime imports or model
loading. It reads physical host memory and the selected configuration without
importing Torch. The policy blocks the 16 GB laptop. The conservative minimums
for any future approved execution environment are:

| Execution target | Minimum physical host RAM | Minimum currently available host RAM | Additional requirement |
| --- | --- | --- | --- |
| MPS / Apple Silicon | 32 GiB | 24 GiB | Matching supported runtime/kernel and a bounded allocator |
| CPU | 48 GiB | 36 GiB | Separate review of dtype, loading peaks, and prompt workload |
| CUDA | 32 GiB | 24 GiB | At least 24 GiB free and total GPU memory for FP16/BF16; 40 GiB for FP32 |

The table describes initial weight-loading admission. After weights are
resident, the adapter checks a separate workspace reserve and an estimate of
the requested batch's KV cache, rather than reserving the loading budget a
second time. These are screening estimates, not measured peak-memory bounds.

These thresholds are admission policy, not measured fit guarantees. A matching
kernel, an available device, a token count below the context limit, or a final
4-bit weight estimate does not establish peak load or inference memory safety.
The resource check must fail before importing Torch on an undersized host;
there is no automatic fallback to CPU, a smaller precision, a shorter prompt,
or a different batch composition. The local profile is diagnostic only on this
laptop.

Unknown physical or available RAM fails closed. Host admission is checked again
immediately before loading because other processes can change available
memory. Future authorized CUDA execution also checks actual free/total VRAM
before loading; it does not infer capacity from the GPU name. On an admitted
MPS host, the adapter requires a runtime allocator cap with fraction at most
0.8 and rejects disabled or invalid allocator-limit overrides. These guards
must not be bypassed to retry the 16 GB laptop.

`--check-access` now reports package/config and host-resource information
without loading a model, importing Torch, initializing a device, contacting
the Hub, or proving gated access. Its successful command exit does not mean
that model execution is permitted. Inspect `resources.host_admissible` and
`resources.reason`; the current laptop must report refusal. Device availability
remains unknown (`null`) with `device_probe` explicitly marked unperformed.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_llm.py --config configs/llm-local.yaml --check-access
```

The selected checkpoint remains gated. Existing account access and the cached
checkpoint are sufficient for the completed access audit; no new login or
access probe is needed during safe local work. The
[Hugging Face gated-model guide](https://huggingface.co/docs/hub/models-gated)
and [commit-pinned download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)
remain references for a separately authorized external run. Do not place tokens
in YAML, source, command arguments, or saved manifests.

New HF generation requires suitable external compute. The authorized
November 16 Alienware run supplied the Nov. 6 baseline responses. For future
runs, review the full load/inference memory budget, checkpoint/runtime identity,
and identical settings across variants. Keep raw completions and validated
predictions together with provenance. No local full-model retry is a
prerequisite for offline revalidation or later development.

## Historical Apple Silicon environment

`configs/llm-local.yaml` preserves the failed experiment's settings for audit:
MPS/FP16, Metal 4-bit quantization with group size 64, 32,768 total tokens,
1,024 output tokens, batch size 1, seed 723, and model cache `../weights/hub`
relative to the YAML. It is an unverified experimental configuration, not a
supported execution profile for the 16 GB laptop. Model execution must remain
disabled on this machine; there is no automatic retry.
The adapter may be evaluated on larger Apple hardware in a separately reviewed
experiment; no such full-model execution has been validated here.

The incident environment was Python 3.13.3 on macOS 26.5.2, Apple M4 with
16 GB unified memory. Its dependency versions are recorded in
`configs/llm-local.requirements.txt`: Torch 2.14.1, Transformers 5.23.0,
`kernels` 0.18.3, Accelerate 1.15.0, Hub 2.4.0, Tokenizers 0.23.3, and
Safetensors 0.8.0. These pins describe the experiment; they are not installation
or execution instructions for the current laptop.

The installed Transformers 5.23.0 Metal loader selected kernel API version 1.
Its inspected revision `6ed0fc4eff016df00830678eb96acb92a6a8f351` provided compiled
Torch 2.10–2.14 builds and no Torch 2.15 build, leading to the Torch 2.14.1
selection. The isolated kernel probe loaded successfully on MPS, and the actual
tokenizer counted 8,406 / 15,329 input tokens for Baseline 2 / Baseline 3.
Those observations verify compatibility and token counts only. They do not
measure full-model load peak memory, prefill memory, or operational safety.
The subsequent full load failed as described above.

The experimental adapter uses the upstream
[Transformers Metal implementation](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/metal.md)
and its [trusted kernel repository](https://huggingface.co/kernels/kernels-community/mlx-quantization-metal-kernels).
Kernel execution is separate from model `trust_remote_code`, as described in
the [official kernel trust documentation](https://huggingface.co/docs/kernels/main/en/security).
A matching kernel revision and source hash establish identity, not a safe
memory envelope. Retain the existing expected kernel SHA for audit without
fetching a replacement or bypassing the resource guard.

## Verified external environment and later main-model acceptance

The November 16 lab run verified Ubuntu 24.04, an RTX 5090 with approximately
31.34 GiB usable GPU memory, approximately 62.25 GiB OS-reported host RAM,
driver 595.99.02, Torch 2.14.1+cu130, and Transformers 5.23.0. Both baseline
generations ended at EOS in approximately 15 seconds total. Five-second GPU
samples recorded up to 21,213 MiB used; sampled values do not establish the
exact peak. Hardware and runtime checks must be repeated for future workloads.

Its dedicated configuration used CUDA/BF16, no quantization, cache-only loading,
32,768 total tokens, 1,024 output tokens, batch size 1, and seed 723 for both
baselines. The Mac dependency file remains an incident record, not a CUDA lock.
Keep lab results separate from the failed Mac attempt and derived response
validation separate from the original generation.

The following commands are a future handoff for an authorized, adequately
provisioned CUDA environment with the project installed. They use plain Python
commands and repository-relative paths so they do not assume a Unix shell.
They are not approved local commands and must not be run on this laptop.

```text
python scripts/run_llm.py --bt-dir data/sample --config configs/llm.yaml --variants baseline_2 baseline_3 --backend hf --output-dir runs/track5-hf-lab-baselines
```

The reference profile uses cache-only loading and CUDA/BF16. A later approved
run must confirm those choices against the actual hardware and supply a
separately reviewed YAML if loading/cache settings need to differ. All
resource, runtime, and model identity checks precede that run. No lighter-model
run is needed for current local validation: stub and fake-HF tests cover the
pipeline without allocating model memory. The completed lab generation and
offline replay supply Nov. 6 evidence; the main-model milestone remains open.

For the main variant, use Track 3's parser output with genuinely computed
observed-stage divergence, then run all three variants together into a new
directory. A synthetic enriched fixture can validate the HF transport path,
but it does not satisfy acceptance on real producer inputs.

```text
python scripts/run_llm.py --bt-dir data/processed/smoke/bt --config configs/llm.yaml --variants baseline_2 baseline_3 main --backend hf --output-dir runs/track5-hf-computed-divergence
```

Retain the per-variant JSONL and sibling run manifest. Check one unique
`(pair_id, model_variant)` record per requested pair/variant, `backend="hf"`,
valid structured outputs, factual citations to retained failure observations,
and explicit errors/abstentions. Confirm the manifest contains checkpoint and
tokenizer identity/revisions, decoding/device/dtype settings, seed, prompt and
configuration hashes, input identities, token counts, and latency. Inspect
errors before scoring; schema validity and one synthetic pair do not establish
causal quality or dataset accuracy. Track 4 owns evaluation and comparison
against the rule baseline; Track 5 supplies matching JSONL without changing
that evaluator or its rules.

The shared schema lists five possible `FailureMode` values, including
`excessive_speed`. Track 5 preserves that vocabulary. Track 4's evaluator owns
its accepted-class policy and dataset coverage; any vocabulary mismatch needs
that owner's follow-up rather than a Track 5 edit to evaluation code.

## Offline verification evidence

After the November 14 incident, safe verification completed with network
access disabled, CPU thread counts limited, and imports of actual Torch,
Transformers, Kernels, and Hub runtimes blocked by the verification process:

- The parser and Track 5 suites passed **284 tests**. No actual model runtime
  was imported, and fake runtime resources exercised both batching and refusal
  paths.
- The full repository suite reported **609 passed, 1 failed**. The remaining
  failure is the existing Track 4 `test_label_vocabularies_match_shared_schemas`
  mismatch concerning `excessive_speed`; no evaluator files were changed.
- `runs/track5-nov06/safe-local-preflight.json` records this machine's actual
  16 GiB physical RAM and refusal before runtime imports. Unknown or inadequate
  available memory is also rejected.
- `runs/track5-nov06/safe-stub/run_manifest.json` records a fresh baseline smoke
  with two stub abstentions and zero errors. Its acceptance is `synthetic`.
  These files were generated after the reboot and are separate from the
  interrupted `local-hf` attempt, which produced no accepted result directory.

No additional installation, model download, GPU initialization, full-model
load, or lighter-model run was performed during that safe verification.
Baseline 2/3 real-HF acceptance remains pending the lab machine.

The following evidence describes the earlier Oct. 30 foundation checks only.
The documented baseline smoke and synthetic all-variant smoke were executed on
November 12, 2026 in a base environment with no installed Torch, Transformers,
or Hugging Face Hub packages. Both exited successfully: respectively two and
three explicit stub abstentions, with zero errors and `weights_loaded: false`.
Local `--check-access` reported `remote_access: "not_checked"` and
`weights: "not_loaded"`. The synthetic run's conservative input counts were
19,534 / 33,892 / 35,124 for Baseline 2 / Baseline 3 / main; they are byte-based
bounds, not Llama tokenizer measurements. No authenticated access probe, model
weight download, or real-model inference was performed.
