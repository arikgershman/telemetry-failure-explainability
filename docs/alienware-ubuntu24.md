# Alienware Ubuntu 24.04: inference only, evaluation on the Mac

**Nov. 19 update:** after applying the original full source archive, apply the
[resident CUDA cache fix and updated bundles](alienware-inference-fix-20261119.md)
before real generation. The `*-20261118` bundles below are the original
handoff; the patched source uses `*-20261119` bundles and fresh retry outputs.

These commands transfer the exact local implementation and prepared inputs,
run the three model variants on the Alienware, and return completions for CPU
evaluation on the Mac. The CPU rule baseline is already in each bundle.
No simulation, compression sweep, model training, or metric plotting needs to
run on the Alienware.

The retained Alienware receipt identifies an RTX 5090 with 31.34 GiB VRAM,
driver `595.99.02`, and 62.25 GiB host RAM (approximately 64 GB). Its working
Torch build was `2.14.1+cu130`. Leave the working driver installed; these steps
do not install a system CUDA toolkit. The host meets the inference memory
minimum. This is not a claim that it meets the separate training requirements.

There are three prepared bundles:

| Bundle | Purpose |
| --- | --- |
| `runs/alienware-fresh-smoke-20261118` | Start here: a small comparison on newly verified physical traces. |
| `runs/alienware-fresh-all-20261118` | Recommended next: all 18 fresh physical pairs. |
| `runs/alienware-v1-dev-20261118` | Optional: all 100 grouped v1 dev pairs. Original v1 certification remains unresolved. |

Each bundle fixes the checkpoint revision, prompts, compression outputs,
tokenizer, three LLM variants, decoding settings, and rule predictions.
Context refusals remain explicit scored cases; neither extra VRAM nor a
successful process exit establishes full context coverage or diagnosis quality.
The frozen bundle manifests contain the exact request and refusal counts.
Held-out test data is not included.

## 1. Mac: check and transfer the package

The package contains source files that have not been pushed and generated
inputs that Git ignores. **Cloning or pulling alone does not supply this
implementation.** Use the archive overlay below as well.

Replace only `ALIENWARE_IP` with the Alienware's reachable IP address or SSH
hostname. `a7l6` is the username recorded by the previous Alienware run. If your
SSH account differs, change it in this one assignment.

```bash
cd /Users/arikgershman/Desktop/CMSC723/telemetry-failure-explainability/runs
shasum -a 256 -c telemetry-alienware-20261118-v2.tar.gz.sha256

alienware_host=a7l6@ALIENWARE_IP
ssh "$alienware_host" 'mkdir -p "$HOME/Downloads"'
scp telemetry-alienware-20261118-v2.tar.gz \
  telemetry-alienware-20261118-v2.tar.gz.sha256 \
  "$alienware_host:Downloads/"
```

The tarball has one top-level `telemetry-failure-explainability/` directory.
It contains no model weights or credentials. Keep the Mac checkout and these
prepared bundles unchanged until the returned results have been evaluated;
the evaluator verifies the implementation hashes.

## 2. Alienware: create an isolated checkout and verify the overlay

Run these in a Bash terminal on Ubuntu 24.04. Use the dedicated new directory
below so the earlier checkout and checkpoint cache remain available.

```bash
set -euo pipefail

sudo apt update
sudo apt install -y git python3.12 python3.12-venv ca-certificates

cd "$HOME/Downloads"
sha256sum -c telemetry-alienware-20261118-v2.tar.gz.sha256

mkdir -p "$HOME/CMSC723"
cd "$HOME/CMSC723"
test ! -e telemetry-failure-explainability-alienware-20261118
git clone https://github.com/arikgershman/telemetry-failure-explainability.git \
  telemetry-failure-explainability-alienware-20261118
cd telemetry-failure-explainability-alienware-20261118
git checkout --detach b5bcff96618351019cd030f9a234c4a610fc461c

tar -xzf "$HOME/Downloads/telemetry-alienware-20261118-v2.tar.gz" \
  --strip-components=1

python3.12 - <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath

root = Path.cwd().resolve()
manifest = json.loads((root / "HANDOFF_CONTENTS.json").read_text())
assert manifest["format_version"] == 1
assert isinstance(manifest["files"], dict) and manifest["files"]
for name, expected in manifest["files"].items():
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and ".." not in relative.parts, name
    path = root.joinpath(*relative.parts)
    assert not path.is_symlink() and path.is_file(), name
    assert path.resolve().is_relative_to(root), name
    data = path.read_bytes()
    assert len(data) == expected["size_bytes"], name
    assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
print(f"Verified {len(manifest['files'])} transferred files")
PY
```

`HANDOFF_CONTENTS.json` checks the transferred files; additional Git checkout
files and `.git/` are expected. Bundle verification below independently checks
the prepared inputs and the executable inference/evaluation source hashes.

If you already have this **dedicated** checkout from an earlier transfer and
prefer to update it, first ensure it has no work you need to preserve. A clean
checkout can be updated with `git pull --ff-only` before applying a fresh
archive. The archive overlay remains required. Do not pull or edit source
after verifying the frozen bundle, and do not reuse populated output folders.

## 3. Alienware: install the inference environment

This uses Ubuntu's Python 3.12 and the verified CUDA wheel. Torch is installed
explicitly from the CUDA 13.0 index before the remaining requirements, so pip
does not select a different Torch build. No project `[llm]` or training extra
is needed.

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate

python -m pip install --upgrade pip
python -m pip install 'torch==2.14.1+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r configs/llm-gpu.requirements.txt
python -m pip install -e .
python -m pip check

test ! -e runs/alienware-host-20261118
mkdir -p runs/alienware-host-20261118
python --version > runs/alienware-host-20261118/python-version.txt
python -m pip freeze > runs/alienware-host-20261118/pip-freeze.txt
nvidia-smi > runs/alienware-host-20261118/hardware-before.txt
cat runs/alienware-host-20261118/hardware-before.txt

python scripts/run_llm.py --config configs/llm.yaml --check-access \
  > runs/alienware-host-20261118/host-preflight.json

python - <<'PY' | tee runs/alienware-host-20261118/cuda-preflight.txt
import importlib.metadata as metadata
import psutil
import torch

GiB = 1024**3
assert metadata.version("torch") == "2.14.1+cu130"
assert torch.version.cuda == "13.0"
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 unavailable"
ram = psutil.virtual_memory()
free, total = torch.cuda.mem_get_info()
assert ram.total >= 32 * GiB and ram.available >= 24 * GiB, "Insufficient host memory"
assert min(free, total) >= 24 * GiB, "Free at least 24 GiB of GPU memory"
x = torch.ones((512, 512), device="cuda", dtype=torch.bfloat16)
assert x.sum().item() == 262144
print({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "cuda": torch.version.cuda,
       "host_available_gib": round(ram.available / GiB, 2),
       "gpu_free_gib": round(free / GiB, 2),
       "gpu_total_gib": round(total / GiB, 2)})
PY
```

These checks open no model weights. Close other GPU applications if memory
checks fail. If `nvidia-smi` or the CUDA test fails, preserve the output and
resolve that failure before loading the model. The runner also checks memory
again during loading and for each request; the minimum admission checks are
not a guarantee that every context will fit.

## 4. Alienware: reuse or download the pinned checkpoint

The old checkout's recorded cache can be reused without copying its weights.
Authentication is only needed if the local token is missing or no longer
valid. The account must already have access to the
[Llama 3.1 8B Instruct repository](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

```bash
if [ -d "$HOME/CMSC723/telemetry-failure-explainability/weights/hub" ]; then
  export HF_HUB_CACHE="$HOME/CMSC723/telemetry-failure-explainability/weights/hub"
else
  export HF_HUB_CACHE="$PWD/weights/hub"
fi
mkdir -p "$HF_HUB_CACHE"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
hf auth whoami || hf auth login
hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --cache-dir "$HF_HUB_CACHE" --exclude 'original/*' \
  2>&1 | tee runs/alienware-host-20261118/model-download.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_UPDATE_CHECK=1
export TOKENIZERS_PARALLELISM=false
```

The exact `hf` command was checked with `--dry-run`: it selects all four
safetensors shards and the required tokenizer/configuration files, and excludes
the unused duplicate original weights. Existing cache files are reused. A new
download is approximately 16 GB. Keep the cache environment variable in every
new terminal that runs inference.

Verify the cached snapshot without opening weights, then verify all bundles:

```bash
python - <<'PY' | tee runs/alienware-host-20261118/model-cache-check.txt
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

revision = "0e9e39f249a16976918f6564b8830bc894c89659"
snapshot = Path(snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", revision=revision,
    cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=True))
assert snapshot.name == revision, snapshot
for name in ("config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json",
             "model.safetensors.index.json"):
    assert (snapshot / name).is_file(), name
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
assert len(shards) == 4
for name in shards:
    assert (snapshot / name).is_file() and (snapshot / name).stat().st_size > 0, name
print({"snapshot": str(snapshot), "shards": len(shards),
       "weight_bytes": sum((snapshot / name).stat().st_size for name in shards)})
PY

python scripts/gpu_experiment.py verify --bundle runs/alienware-fresh-smoke-20261118
python scripts/gpu_experiment.py verify --bundle runs/alienware-fresh-all-20261118
python scripts/gpu_experiment.py verify --bundle runs/alienware-v1-dev-20261118
```

## 5. Alienware: run the small fresh smoke

This runs Baselines 2 and 3 and the main model with identical fixed model and
decoding settings. Structured-response checks stay enabled. Metrics and plots
are deferred to the Mac.

```bash
if python scripts/gpu_experiment.py run \
  --bundle runs/alienware-fresh-smoke-20261118 \
  --output-dir runs/alienware-fresh-smoke-hf --backend hf --inference-only \
  2>&1 | tee runs/alienware-host-20261118/fresh-smoke-console.log; then
  echo 'Fresh smoke finished; inspect its receipt below.'
else
  echo 'Fresh smoke failed. Keep the artifacts and skip to the return step.'
fi
python -m json.tool runs/alienware-fresh-smoke-hf/experiment_receipt.json
```

Proceed only when `real_generation_acceptance` is `true`. A failed smoke is
useful evidence: return its logs and exact completions before spending time
on larger runs. Invalid JSON, unsupported evidence, CUDA errors, and missing
outputs are failures. Do not edit outputs or loosen frozen settings to make
a receipt pass.

## 6. Alienware: run all fresh pairs; optionally run v1 dev

The first command below checks smoke acceptance before launching the full
fresh comparison. The full set includes both tasks and all four released
failure modes. The two excessive-speed activation probes were already verified
on CPU; they are not accepted failure pairs for model comparison. Ten of the
54 full-set requests have preflight context refusals; those remain in the returned artifacts
and in CPU scoring.

```bash
python - <<'PY'
import json
from pathlib import Path
receipt = json.loads(Path("runs/alienware-fresh-smoke-hf/experiment_receipt.json").read_text())
assert receipt["real_generation_acceptance"] is True, "Return the failed smoke first"
PY

if python scripts/gpu_experiment.py run \
  --bundle runs/alienware-fresh-all-20261118 \
  --output-dir runs/alienware-fresh-all-hf --backend hf --inference-only \
  2>&1 | tee runs/alienware-host-20261118/fresh-all-console.log; then
  echo 'Full fresh comparison finished.'
else
  echo 'Full fresh comparison failed; return its retained artifacts.'
fi
```

Optional v1 dev comparison, using all 100 dev pairs and keeping task-local
IDs separate:

```bash
if python scripts/gpu_experiment.py run \
  --bundle runs/alienware-v1-dev-20261118 \
  --output-dir runs/alienware-v1-dev-hf --backend hf --inference-only \
  2>&1 | tee runs/alienware-host-20261118/v1-dev-console.log; then
  echo 'v1 dev comparison finished.'
else
  echo 'v1 dev comparison failed; return its retained artifacts.'
fi
```

Run directories must be absent or empty. For a retry, choose a new output
name and adjust the return/evaluation commands accordingly. Keep the first
attempt for diagnosis. No training is required for this comparison; reviewed
preference annotations and any later training run remain separate work.

## 7. Alienware: package successful and failed attempts for return

Run this even if inference failed. It includes whichever result directories
exist and the host logs, without weights or credentials.

```bash
nvidia-smi > runs/alienware-host-20261118/hardware-after.txt
result_paths=(runs/alienware-host-20261118)
for path in runs/alienware-fresh-smoke-hf runs/alienware-fresh-all-hf runs/alienware-v1-dev-hf; do
  if [ -d "$path" ]; then result_paths+=("$path"); fi
done
test ! -e "$HOME/telemetry-alienware-results-20261118.tar.gz"
tar -czf "$HOME/telemetry-alienware-results-20261118.tar.gz" "${result_paths[@]}"
cd "$HOME"
sha256sum telemetry-alienware-results-20261118.tar.gz \
  > telemetry-alienware-results-20261118.tar.gz.sha256
```

## 8. Mac: fetch results, verify, and evaluate on CPU

In a Mac terminal, use the same SSH host value as step 1:

```bash
cd /Users/arikgershman/Desktop/CMSC723/telemetry-failure-explainability
alienware_host=a7l6@ALIENWARE_IP
test ! -e runs/alienware-return-20261118
mkdir -p runs/alienware-return-20261118/returned
scp "$alienware_host:telemetry-alienware-results-20261118.tar.gz" \
  "$alienware_host:telemetry-alienware-results-20261118.tar.gz.sha256" \
  runs/alienware-return-20261118/

cd runs/alienware-return-20261118
shasum -a 256 -c telemetry-alienware-results-20261118.tar.gz.sha256
tar -xzf telemetry-alienware-results-20261118.tar.gz -C returned
cd ../..

source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg
return_root="$PWD/runs/alienware-return-20261118/returned"

python scripts/evaluate_gpu_results.py \
  --bundle runs/alienware-fresh-smoke-20261118 \
  --results "$return_root/runs/alienware-fresh-smoke-hf" \
  --output-dir runs/alienware-fresh-smoke-evaluation

python scripts/evaluate_gpu_results.py \
  --bundle runs/alienware-fresh-all-20261118 \
  --results "$return_root/runs/alienware-fresh-all-hf" \
  --output-dir runs/alienware-fresh-all-evaluation
```

Run the second evaluation only if the full fresh run was attempted. If the
optional v1 dev run was attempted, evaluate it with:

```bash
python scripts/evaluate_gpu_results.py \
  --bundle runs/alienware-v1-dev-20261118 \
  --results "$return_root/runs/alienware-v1-dev-hf" \
  --output-dir runs/alienware-v1-dev-evaluation
```

CPU evaluation verifies the returned file inventory, request identities,
prompt/completion hashes, exact tokenizer counts, raw responses, structured
predictions, and the source-bound bundle. It scores all four variants and
creates plots and blinded review material. Context refusals remain errors in
metrics. A failed returned run remains failed; metrics do not turn it into
accepted real-model evidence. No model weights or CUDA are opened on the Mac.

Human explanation review is still needed after generation. The fresh cases
are integration evidence, and dev comparisons inform development; neither
replaces the held-out final study.

## Verified installation references

- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
  and the [CUDA 13.0 wheel index](https://download.pytorch.org/whl/cu130/torch/)
  provide the pinned Python 3.12 Linux build.
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)
  documents CUDA 13.x driver requirements; the retained working driver is newer
  than the required 580 series.
- [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli)
  documents authentication, immutable revisions, cache downloads, filtering,
  and dry runs.
- [Transformers 5.23.0](https://pypi.org/project/transformers/5.23.0/)
  supports Python 3.12; the remaining inference pins were checked against
  official package metadata, including the
  [psutil 7.2.2 Linux wheel](https://pypi.org/project/psutil/7.2.2/).
