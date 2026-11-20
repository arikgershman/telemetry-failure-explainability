# Nov. 19 Alienware inference fix

> Historical v2 procedure. The later retry exposed genuine CUDA OOMs, truncated responses and invalid citations. Use the [v3 recovery and measured qualification workflow](alienware-v3-recovery-20261119.md) for the next run. The earlier observations below describe the first smoke only.

The first fresh real-model smoke loaded weights and generated responses. Its
logs show independent strict JSON and later CUDA resource rejections, without
an actual CUDA OOM. The user subsequently supplied complete JSON-code-fenced
responses ending at EOS and resident memory diagnostics.

Two changes are now prepared:

- For a resident model only, when the existing CUDA guard would refuse, release
  inactive PyTorch cache once, remeasure driver-free memory, and apply exactly
  the same workspace/KV threshold. Genuine low memory still refuses; there is no
  automatic generation retry. Cleanup history is retained in provenance.
- `configs/llm-production-v2.yaml` explicitly selects the existing
  `json_or_fenced_v1` policy. A bare JSON object or one whole-response lowercase
  JSON fence is accepted as an envelope. Schema, observed evidence, labels and
  completeness checks remain strict. The checkpoint, prompts, decoding, context
  and token/refusal counts are unchanged. The original strict config remains.

Local checks of the three pasted payloads accepted one and rejected two for
unobserved evidence. These are transcribed semantic checks, not verification of
original completion bytes or model performance. Do not repair those citations.
The opt-in Mac `--score-response-errors` mode scores independently reproduced
parser/evidence failures while keeping acceptance false. Missing completions,
runtime failures, invalid token metadata and tampering still block verification.

Preserve `runs/alienware-fresh-smoke-hf` and the original bundles. The original
full `telemetry-alienware-20261118-v2.tar.gz` preserves the previous source.
Source binding intentionally invalidates earlier bundles with patched source;
use the new `*-v2-20261119` bundles. No dependency or model download is needed.

On the Mac (replace the SSH address):

```bash
cd /Users/arikgershman/Desktop/CMSC723/telemetry-failure-explainability/runs
alienware_host=a7l6@ALIENWARE_IP
shasum -a 256 -c telemetry-alienware-inference-fix-20261119.tar.gz.sha256
scp telemetry-alienware-inference-fix-20261119.tar.gz \
  telemetry-alienware-inference-fix-20261119.tar.gz.sha256 "$alienware_host:Downloads/"
```

On Ubuntu, apply to the existing dedicated checkout and verify the new bundle:

```bash
cd "$HOME/Downloads"
sha256sum -c telemetry-alienware-inference-fix-20261119.tar.gz.sha256
cd "$HOME/CMSC723/telemetry-failure-explainability-alienware-20261118"
tar -xzf "$HOME/Downloads/telemetry-alienware-inference-fix-20261119.tar.gz" --strip-components=1
source .venv-gpu/bin/activate
python scripts/gpu_experiment.py verify --bundle runs/alienware-fresh-smoke-v2-20261119

if [ -d "$HOME/CMSC723/telemetry-failure-explainability/weights/hub" ]; then
  export HF_HUB_CACHE="$HOME/CMSC723/telemetry-failure-explainability/weights/hub"
else
  export HF_HUB_CACHE="$PWD/weights/hub"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
set -o pipefail

if python scripts/gpu_experiment.py run \
  --bundle runs/alienware-fresh-smoke-v2-20261119 \
  --output-dir runs/alienware-fresh-smoke-hf-retry-20261119 \
  --backend hf --inference-only \
  2>&1 | tee runs/alienware-host-20261118/fresh-smoke-retry-20261119.log
then
  echo 'Smoke accepted; inspect the receipt.'
else
  echo 'Run retained failures; inspect and return the artifacts.'
fi
python -m json.tool runs/alienware-fresh-smoke-hf-retry-20261119/experiment_receipt.json
```

Run this small smoke first. Two further bundles, `alienware-fresh-all-v2-20261119`
and `alienware-v1-dev-v2-20261119`, are also prepared. Inspect the first retry
before launching them. No driver change, unsafe memory override or GPU training
is required. Actual Alienware cache recovery remains to be confirmed.

Package the retry even if the strict acceptance gate fails:

```bash
cd "$HOME/CMSC723/telemetry-failure-explainability-alienware-20261118"
test ! -e "$HOME/telemetry-alienware-retry-20261119.tar.gz"
tar -czf "$HOME/telemetry-alienware-retry-20261119.tar.gz" \
  runs/alienware-fresh-smoke-hf-retry-20261119 runs/alienware-host-20261118
cd "$HOME"
sha256sum telemetry-alienware-retry-20261119.tar.gz \
  > telemetry-alienware-retry-20261119.tar.gz.sha256
```

On the Mac, retrieve and score, including verified response errors:

```bash
cd /Users/arikgershman/Desktop/CMSC723/telemetry-failure-explainability
alienware_host=a7l6@ALIENWARE_IP
test ! -e runs/alienware-retry-return-20261119
mkdir -p runs/alienware-retry-return-20261119/returned
scp "$alienware_host:telemetry-alienware-retry-20261119.tar.gz" \
  "$alienware_host:telemetry-alienware-retry-20261119.tar.gz.sha256" \
  runs/alienware-retry-return-20261119/
cd runs/alienware-retry-return-20261119
shasum -a 256 -c telemetry-alienware-retry-20261119.tar.gz.sha256
tar -xzf telemetry-alienware-retry-20261119.tar.gz -C returned
cd ../..
PYTHONPATH=src .venv/bin/python scripts/evaluate_gpu_results.py \
  --bundle runs/alienware-fresh-smoke-v2-20261119 \
  --results runs/alienware-retry-return-20261119/returned/runs/alienware-fresh-smoke-hf-retry-20261119 \
  --score-response-errors --output-dir runs/alienware-retry-evaluation-20261119
```

The scorer exits 1 and retains a failed receipt if responses are rejected; look
for `metrics_generated` and the per-task `evaluation/` outputs. Such responses
remain scored errors. No model weights or GPU are used for this step.

PyTorch describes inactive cache and its release in
[CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management).
