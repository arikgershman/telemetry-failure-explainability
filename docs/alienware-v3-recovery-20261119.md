# Alienware v3 recovery — Nov. 19, 2026

Use this procedure instead of the earlier v2 retry. Keep the failed outputs.
The RTX 5090 has already passed all four fused-attention kernel cases with the
math backend disabled; the reported prefix cases peaked at about 1.51–1.63 GB.
The saved `runs/cuda-qualification-20261119.json` is reused below. Those kernel
checks did not load the checkpoint or measure a complete model request.

The retry exposed genuine CUDA OOMs on long prompts, truncated 512-token
responses, and invalid model citations. One failed 1,574,961,152-byte allocation
matches the allocator-rounded size of a 54,908-token × 14,336-channel BF16 MLP
intermediate; this is consistent with a full-prompt MLP allocation, but the
allocator log alone does not prove its stack location. V3 uses Transformers'
native prefill chunks of at most 2,048 tokens, dynamic KV cache, disabled
compilation, and fused-only SDPA. No math-attention fallback is permitted.
Finished failure traceback frames are released before another request, without
retrying inference or lowering memory thresholds.

`observed_json_v1` is a new decoding protocol. The model chooses the failure
mode, an observed failure stage, exact observed failure-side citations, and a
bounded explanation. The decoder permits at most two citations and 128
explanation tokens/512 printable ASCII characters, then requires complete bare
JSON and EOS within a precomputed bound. Null predictions produce an explicit
model-selected abstention. Strict schema/evidence validation remains; outputs
are not repaired and diagnoses are not supplied by rules. Constraints guarantee
format and citation membership, not diagnostic accuracy or truthful prose.
All three prompted variants use the same v3 settings. Do not combine their
scores with v1/v2 generations. The eight-pair smoke still contains 24 requests.

The existing environment and checkpoint cache are sufficient. No dependency,
driver, model download, simulator run, or training run is required. Full-model
GPU acceptance remains pending until the new qualification and smoke pass.

## 1. Mac: transfer the prepared implementation and frozen inputs

Replace only the SSH address:

```bash
cd /Users/arikgershman/Desktop/CMSC723/telemetry-failure-explainability/runs
alienware_host=a7l6@ALIENWARE_IP
shasum -a 256 -c telemetry-alienware-v3-20261119.tar.gz.sha256
ssh "$alienware_host" 'mkdir -p "$HOME/Downloads"'
scp telemetry-alienware-v3-20261119.tar.gz \
  telemetry-alienware-v3-20261119.tar.gz.sha256 \
  "$alienware_host:Downloads/"
```

## 2. Ubuntu: apply the update and verify the new bundle

Use the existing dedicated checkout. This overlays the implementation, not old
run directories. Do not pull or edit source after verifying a frozen bundle.
Every new qualification/result directory must be absent; no output is replaced.

```bash
set -euo pipefail
cd "$HOME/Downloads"
sha256sum -c telemetry-alienware-v3-20261119.tar.gz.sha256
cd "$HOME/CMSC723/telemetry-failure-explainability-alienware-20261118"
tar -xzf "$HOME/Downloads/telemetry-alienware-v3-20261119.tar.gz" \
  --strip-components=1
source .venv-gpu/bin/activate
export PYTHONPATH="$PWD/src"

if [ -d "$HOME/CMSC723/telemetry-failure-explainability/weights/hub" ]; then
  export HF_HUB_CACHE="$HOME/CMSC723/telemetry-failure-explainability/weights/hub"
else
  export HF_HUB_CACHE="$PWD/weights/hub"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

test -f runs/cuda-qualification-20261119.json
test ! -e runs/alienware-v3-qualification-20261119
test ! -e runs/alienware-fresh-smoke-hf-v3-20261119
test ! -e runs/alienware-v3-logs-20261119
mkdir runs/alienware-v3-logs-20261119
python scripts/gpu_experiment.py verify \
  --bundle runs/alienware-fresh-smoke-v3-20261119
```

## 3. Ubuntu: qualify one longest request, then run the smoke

Qualification executes the longest fitting frozen prompt through the actual
checkpoint, constraints and parser. It verifies the existing kernel report,
software/device binding, token count, complete response and measured CUDA peaks.
Only a passed receipt admits the full run. Leave other GPU workloads closed
while these commands execute; the existing desktop session can stay open.
Continue in the same Ubuntu terminal so the activated environment and exports
from step 2 remain active.

```bash
if python scripts/qualify_gpu_experiment.py \
  --bundle runs/alienware-fresh-smoke-v3-20261119 \
  --kernel-report runs/cuda-qualification-20261119.json \
  --output-dir runs/alienware-v3-qualification-20261119 \
  2>&1 | tee runs/alienware-v3-logs-20261119/qualification.log
then
  if python scripts/gpu_experiment.py run \
    --bundle runs/alienware-fresh-smoke-v3-20261119 \
    --qualification runs/alienware-v3-qualification-20261119 \
    --output-dir runs/alienware-fresh-smoke-hf-v3-20261119 \
    --backend hf --inference-only \
    2>&1 | tee runs/alienware-v3-logs-20261119/smoke.log
  then
    echo 'V3 smoke passed. Return the artifacts for Mac evaluation.'
  else
    echo 'Smoke retained failures. Return the artifacts; do not rerun or overwrite.'
  fi
else
  echo 'Qualification did not pass. Full smoke was not started. Return the artifacts.'
fi

if [ -f runs/alienware-v3-qualification-20261119/qualification_receipt.json ]; then
  python -m json.tool runs/alienware-v3-qualification-20261119/qualification_receipt.json
fi
if [ -f runs/alienware-fresh-smoke-hf-v3-20261119/experiment_receipt.json ]; then
  python -m json.tool runs/alienware-fresh-smoke-hf-v3-20261119/experiment_receipt.json
fi
```

Run this smoke first. The separate `alienware-fresh-all-v3-20261119` and
`alienware-v1-dev-v3-20261119` bundles each require their own longest-request
qualification before use. A smoke qualification cannot admit a larger bundle.
Their existing context refusals remain explicit; no prompts are shortened to
make them fit. Held-out test data remains unused.

## 4. Ubuntu: return qualification and inference evidence

Package evidence even when qualification or inference fails. This excludes
weights, environments and unrelated historical runs.

```bash
cd "$HOME/CMSC723/telemetry-failure-explainability-alienware-20261118"
result_paths=(runs/alienware-v3-logs-20261119)
for path in \
  runs/alienware-v3-qualification-20261119 \
  runs/alienware-fresh-smoke-hf-v3-20261119
do
  if [ -d "$path" ]; then result_paths+=("$path"); fi
done
test ! -e "$HOME/telemetry-alienware-v3-results-20261119.tar.gz"
test ! -e "$HOME/telemetry-alienware-v3-results-20261119.tar.gz.sha256"
tar -czf "$HOME/telemetry-alienware-v3-results-20261119.tar.gz" \
  "${result_paths[@]}"
cd "$HOME"
sha256sum telemetry-alienware-v3-results-20261119.tar.gz \
  > telemetry-alienware-v3-results-20261119.tar.gz.sha256
```

## 5. Mac: retrieve, verify and evaluate without loading weights

```bash
cd /Users/arikgershman/Desktop/CMSC723/telemetry-failure-explainability
alienware_host=a7l6@ALIENWARE_IP
test ! -e runs/alienware-v3-return-20261119
mkdir -p runs/alienware-v3-return-20261119/returned
scp "$alienware_host:telemetry-alienware-v3-results-20261119.tar.gz" \
  "$alienware_host:telemetry-alienware-v3-results-20261119.tar.gz.sha256" \
  runs/alienware-v3-return-20261119/
cd runs/alienware-v3-return-20261119
shasum -a 256 -c telemetry-alienware-v3-results-20261119.tar.gz.sha256
tar -xzf telemetry-alienware-v3-results-20261119.tar.gz -C returned
cd ../..
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg

returned_run=runs/alienware-v3-return-20261119/returned/runs/alienware-fresh-smoke-hf-v3-20261119
if [ -f "$returned_run/experiment_receipt.json" ]; then
  PYTHONPATH=src .venv/bin/python scripts/evaluate_gpu_results.py \
    --bundle runs/alienware-fresh-smoke-v3-20261119 \
    --results "$returned_run" \
    --output-dir runs/alienware-fresh-smoke-v3-evaluation-20261119
else
  echo 'Only qualification ran. Its receipt and logs are in the returned directory.'
fi
```

Each full run includes an immutable copy of its passed qualification. The Mac
verifies that copy, its receipt hash and all source/input/result provenance
automatically, then generates four-variant metrics, plots and blinded review
sheets. Human explanation review
remains required. A runtime failure must be diagnosed from its preserved evidence;
it is not converted into an accepted model result or a synthetic completion.
