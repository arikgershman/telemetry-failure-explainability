# Track 5 LLM harness and compute/access smoke plan

Track 5's Oct. 30 foundation provides one model adapter, an offline stub,
controlled prompt views, and a repeatable route to a real inference smoke test.
The stub checks orchestration and contracts; it does not diagnose failures or
establish LLM accuracy. Real Baseline 2/3 runs remain a Nov. 6 acceptance step,
and a real main-model run remains a Nov. 13 acceptance step. The training-data
formatter, preference collection, LoRA/DPO training, and other tracks' simulation,
parser, and evaluation implementations are outside this Oct. 30 change.

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
The harness caps smoke tests at 65,536 total tokens, deliberately below that
model limit. That budget includes the chat template, complete telemetry payload,
and reserved output tokens. Count the actual model tokenizer's tokens before
generation; reject an oversized request instead of silently dropping stages or
evidence. A stub's approximate counts cannot establish that an HF prompt fits.

The reference configuration selects CUDA/BF16, greedy decoding, batch size 1,
a 512-token output allowance, and seed 723. The model and tokenizer share the
configured revision. `local_files_only: true` prevents an HF run from fetching
uncached files; make a separately named experiment configuration with that flag
set to `false` when intentionally retrieving the gated checkpoint. Quantization
is disabled; batching and quantization extensions remain later work.

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

## Offline CPU smoke

Use the existing project environment from the repository root. Installing the
base package is sufficient; this path does not need the `llm` extra, a Hugging
Face account, network access, or model weights.

```bash
python -m pip install -e .
python -m pytest tests/test_prompts.py tests/test_llm.py
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

## Access and compute preflight

The selected repository is gated. Request access through the
[official checkpoint page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
using the account that will run inference; the public API currently reports
manual approval. After approval, authenticate locally using `hf auth login` or
provide a read token through `HF_TOKEN`, as described in the
[Hugging Face gated-model guide](https://huggingface.co/docs/hub/models-gated).
Do not put tokens in YAML, source, shell command arguments, or saved manifests.
Reading public metadata does not prove permission to download gated files.

Before reserving model compute:

1. Inspect local readiness with
   `python scripts/run_llm.py --config configs/llm.yaml --check-access`. This
   check reports installed libraries and the configured device without loading
   weights, reading credentials, or contacting Hugging Face. Its success does
   not establish gated repository access.
2. Install the optional environment with `python -m pip install -e '.[llm]'`.
   Check that Transformers supports Llama 3.1 (Meta's minimum is 4.43.0), and
   record the actual installed package versions in the run provenance. The
   repository's broad optional dependency minimum alone is not that check.
3. Confirm access to the pinned revision's `config.json` and tokenizer using
   the approved account. These are small downloads. Only the later inference
   run should load weights. Use the same SHA for every download; the
   [Hub download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download)
   documents commit-pinned downloads and cache reuse.
4. Reserve a device and sufficient free disk/cache space. An 8-billion-parameter
   model requires approximately 16 GB of BF16/FP16 weights or 32 GB of FP32
   weights before runtime overhead. These are planning estimates from parameter
   count and bytes per element, not measured memory requirements. KV cache,
   activations, tokenizer inputs, and library allocations add memory; longer
   prompts and larger batches increase the requirement. Start at batch size 1.
5. A CUDA GPU with at least 24 GB free memory is a practical initial BF16 smoke
   target, subject to the actual prompt budget and measured peak allocation.
   BF16 requires device support. Explicit CPU/FP32 inference is possible with
   substantial RAM and potentially long runtime; it is a real model run and
   is distinct from the weight-free CPU stub. Do not assume this workstation
   has a suitable accelerator or that automatic placement guarantees a fit.
6. Keep quantization disabled for the reference experiment unless a supported
   quantized configuration has been deliberately selected and verified on the
   target hardware. Any later quantization choice must be identical across
   variants and recorded in provenance. See the official
   [Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
   before adding that optional path; installing bitsandbytes alone does not
   establish hardware compatibility.

No approved-account access check, weight download, GPU allocation, or real
generation is claimed by the Oct. 30 offline smoke. Keep an explicit record of
access failure, insufficient memory, or missing dependencies instead of
substituting stub output for HF output.

An optional small-file access probe, run only after authentication, is:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
from telemetry_reasoning.models.llm import load_llm_config

config = load_llm_config("configs/llm.yaml")
hf_hub_download(
    repo_id=config.model_id,
    filename="config.json",
    revision=config.revision,
    local_files_only=False,
)
print("Pinned config file accessible; model weights and generation remain unchecked.")
PY
```

This downloads a small gated file and uses the existing local login. The
no-download `--check-access` command does neither. Never print the token or
place token material in provenance.

## Real-model acceptance, separately recorded

Run Baselines 2 and 3 first on the same schema-valid BT inputs and shared
configuration, with the tokenizer/model revision pinned as above:

```bash
python scripts/run_llm.py \
  --bt-dir /tmp/track5-sample-bt \
  --config configs/llm.yaml \
  --variants baseline_2 baseline_3 \
  --backend hf \
  --output-dir runs/track5-hf-baselines
```

These HF commands use the default cache-only policy. First populate the pinned
cache deliberately or pass the separately reviewed experiment YAML with
`local_files_only: false` as `--config`. Confirm the selected CUDA/BF16 device
is available before generation; changing it requires a new recorded config.

For the main variant, use Track 3's parser output with genuinely computed
observed-stage divergence, then run all three variants together into a new
directory. A synthetic enriched fixture can validate the HF transport path,
but it does not satisfy acceptance on real producer inputs.

```bash
python scripts/run_llm.py \
  --bt-dir data/processed/smoke/bt \
  --config configs/llm.yaml \
  --variants baseline_2 baseline_3 main \
  --backend hf \
  --output-dir runs/track5-hf-computed-divergence
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

The documented baseline smoke and synthetic all-variant smoke were executed on
November 12, 2026 in a base environment with no installed Torch, Transformers,
or Hugging Face Hub packages. Both exited successfully: respectively two and
three explicit stub abstentions, with zero errors and `weights_loaded: false`.
Local `--check-access` reported `remote_access: "not_checked"` and
`weights: "not_loaded"`. The synthetic run's conservative input counts were
19,534 / 33,892 / 35,124 for Baseline 2 / Baseline 3 / main; they are byte-based
bounds, not Llama tokenizer measurements. No authenticated access probe, model
weight download, or real-model inference was performed.
