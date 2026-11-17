# Track 5 Nov. 6 acceptance receipt

Completed and verified on November 16, 2026. This receipt covers Baseline 2 and
Baseline 3 inference/output acceptance. Main-model inference, DPO formatting,
training, and the full comparative study remain later work.

## Generation and response validation

The Alienware generated both actual HF responses from repository commit
`87d2697e6e8a8d6d2e3bc8e36871196d72c6183b`, using identical settings:
`meta-llama/Llama-3.1-8B-Instruct`, revision
`0e9e39f249a16976918f6564b8830bc894c89659`, CUDA/BF16, no quantization,
32,768 total tokens, 1,024 maximum new tokens, batch size 1, greedy decoding,
and seed 723. Both requests finished at EOS.

The original strict run remains unchanged and incomplete: Baseline 2 returned
one whole-response JSON Markdown fence; Baseline 3 returned bare JSON.
CPU-only replay applied `json_or_fenced_v1` uniformly to both exact responses.
It removed only the fence and applied the same strict JSON, schema, task, and
observed-evidence checks. It changed no JSON content, rationale, label, citation,
input, prompt, model, precision, or decoding setting. `strict_json_v1` remains
the default for legacy callers and the reference inference configuration.

The derived manifest reports `acceptance.status: passed`,
`nov6_baselines_covered: true`, two validated requests, and
`phase: offline_response_revalidation`, with zero newly generated requests and
no model loaded during replay. It separately preserves original acceptance,
statuses, requests, batches, generation sources, and raw text, alongside new
validator source and normalization hashes.

| Variant | Input/output tokens | Derived result | Structured citations |
| --- | --- | --- | --- |
| Baseline 2 | 8,406 / 243 | Correct mode and stage on pair `000` | All 4 match observed failure samples |
| Baseline 3 | 15,329 / 199 | Correct mode and stage on pair `000` | Both match observed failure samples |

## Artifact locations and hashes

Generated artifacts remain ignored by Git. Locations are relative to the
repository root:

- Original archive: `runs/track5-nov06-results.tar.gz`.
- Original bundle: `runs/alienware-import-20261116T215259Z/`.
- Accepted derived artifacts: `runs/track5-nov06/accepted-baselines/`.
- Independent verification: `runs/track5-nov06/accepted-baselines-verification.json`.
- Derived scores/plots: `runs/track5-nov06/accepted-evaluation/`.
- Original strict scores and causal review: `runs/track5-nov06/alienware-analysis-20261116/`.

| Artifact | SHA-256 |
| --- | --- |
| Original archive | `a3c6452ece457e1e4bc0b13cdc56b2f40e71213cd403670eea0448f1b6c4cbf8` |
| Original manifest | `2d1d6e5ec59e23ef3df31f199afbb6fcca1d1afcfa687b4e75e1745b859fe230` |
| Derived manifest | `edc27aee3c2638d219fbe278da1095f779e4449cba6ca167358a83889fb3b66f` |
| Baseline 2 exact raw text, before and after replay | `e2d06d9af6d8018f8b8ec5cc7a71e0984b771e3f78c4d7beb1d2f3df4c9d6037` |
| Baseline 3 exact raw text, before and after replay | `73109eb721d49392f9c17c5669b9d97a5bb2a6c4be93e7d7626359d6750e190b` |

The archive agrees with all 17 extracted files, and every original file's hash
was compared before/after replay. Historical implementation bytes were verified
from the recorded Git revision. Unchanged prompt/schema/evidence sources and
templates were checked against the current checkout. Hashes and metadata
establish internal provenance, not cryptographic attestation of model execution.

## Explanation limitations

This is one checked-in synthetic pickup pair, not a physics-generated dataset
or held-out accuracy estimate. Both generated explanations are preserved:

- Baseline 2 incorrectly says contact force stayed at 12 N after the grip-force
  decrease; its own 5.10 s citation is 2.4 N. This contradiction remains a quality
  defect even though the structured evidence and labels validate.
- Baseline 3 emphasizes initial contact behavior also present in the success
  reference, rather than the failure-side force drop at 5.10 s. Correct labels
  and observed citations do not establish a sound causal explanation.

Syntax, labels, evidence, and explanation quality are separate findings. The
unchanged Track 4 evaluator produced label scores and plots. Its existing
`--bt-dir` defect was not invoked; Track 5 independently checked every citation.
No rules, evaluator, simulation, parser, dataset, or shared-schema source changed.

## CPU verification and reproduction

All 367 parser/prompt/LLM/CLI/revalidation tests passed with model/GPU runtime
imports blocked and CPU math threads limited. The 54 dedicated replay tests
cover outside-directory CLI execution, legacy strict manifests, uniform policy,
parent preservation, damaged provenance/metadata/batches, duplicate/missing
identities, fresh outputs, evidence rejection, and synthetic-result safeguards.
The actual saved HF responses were also replayed with all model-runtime imports
blocked, and their raw text and explanations were independently compared.

With the original transferred bundle available, use a fresh destination:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/revalidate_llm.py \
  --bundle-dir runs/alienware-import-20261116T215259Z \
  --response-policy json_or_fenced_v1 \
  --output-dir runs/track5-nov06/replay-check
```

CPU tests:

```bash
PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_parser.py tests/test_prompts.py tests/test_llm.py \
  tests/test_llm_cli.py tests/test_llm_revalidation.py
```

No additional GPU run is required for this milestone. New prompts, generation
settings, checkpoints, examples, or the later main-model milestone require new
model execution on suitable external hardware.
