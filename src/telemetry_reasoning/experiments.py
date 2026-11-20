"""Portable CPU-prepared experiments; only real completion generation needs CUDA.

Preparation never opens weights. Every selected pair receives all four variants;
no oversized request or unsupported response is silently removed. Task-local IDs
stay in separate directories. Test data require an explicit final-study switch.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

import yaml

from telemetry_reasoning.eval.contracts import read_predictions, write_predictions
from telemetry_reasoning.io import read_metadata_csv, write_metadata_csv
from telemetry_reasoning.models.llm import Generation, StubBackend, load_llm_config, parse_response
from telemetry_reasoning.models.prompts import DEFAULT_TEMPLATE_DIR, TEMPLATE_FILENAMES, build_prompt
from telemetry_reasoning.models.rule_based import load_rule_config, predict
from telemetry_reasoning.parser.tuning import validate_tuning_manifest
from telemetry_reasoning.schemas import PairedBehaviorTree
from telemetry_reasoning.sim.freeze_robot_dataset import verify_dataset, verify_handoff

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("pick_up_object", "transport_object")
VARIANTS = ("baseline_2", "baseline_3", "main")
SOURCE_PATHS = (
    "src/telemetry_reasoning/experiments.py", "scripts/gpu_experiment.py",
    "scripts/run_llm.py", "scripts/evaluate.py",
    "src/telemetry_reasoning/models/llm.py", "src/telemetry_reasoning/models/prompts.py",
    "src/telemetry_reasoning/models/rule_based.py", "src/telemetry_reasoning/eval/contracts.py",
    "src/telemetry_reasoning/eval/metrics.py", "src/telemetry_reasoning/eval/plots.py",
    "src/telemetry_reasoning/schemas.py", "src/telemetry_reasoning/io.py",
    "src/telemetry_reasoning/experiment_results.py", "scripts/evaluate_gpu_results.py",
    "src/telemetry_reasoning/fresh_experiments.py", "scripts/prepare_fresh_experiment.py",
    "src/telemetry_reasoning/models/constrained.py",
    "src/telemetry_reasoning/gpu_qualification.py", "scripts/qualify_gpu_experiment.py",
    "scripts/probe_cuda_inference.py",
)


def _json(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=reject)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def _fingerprint(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular non-symlink file: {path}")
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _inventory(root, exclude=()):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink artifact: {path}")
        if path.is_file() and path.relative_to(root).as_posix() not in exclude:
            result[path.relative_to(root).as_posix()] = _fingerprint(path)
    return result


def _safe(root, relative):
    path = PurePosixPath(relative)
    if (not isinstance(relative, str) or not relative or path.is_absolute()
            or ".." in path.parts or "\\" in relative or str(path) != relative
            or any(ord(c) < 32 for c in relative)):
        raise ValueError(f"Unsafe relative artifact path: {relative!r}")
    result = Path(root) / relative
    if result.is_symlink() or any(p.is_symlink() for p in result.parents if p != Path(root).parent):
        raise ValueError(f"Symlink artifact path: {relative}")
    return result


def _fresh(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"Destination must be absent: {path}")
    return path


def _sources():
    return {name: _fingerprint(ROOT / name) for name in SOURCE_PATHS}


def load_cpu_tokenizer(directory, lock_path=ROOT / "configs/tokenizer-llama31.json"):
    """Load only a byte-locked local tokenizer, without loading model weights.

    Imports are lazy: the default test suite can inject a small token counter.
    Transformers may import its runtime helpers, but no model/device is created.
    """
    directory = Path(directory)
    lock = _json(lock_path)
    for name, expected in lock["files"].items():
        if _fingerprint(_safe(directory, name)) != expected:
            raise ValueError(f"Pinned tokenizer file changed: {name}")
    tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False,
    )
    if not tokenizer.chat_template:
        raise ValueError("Pinned instruction tokenizer has no chat template")
    return tokenizer, lock


def prompt_token_count(tokenizer, prompt):
    encoded = tokenizer.apply_chat_template(
        prompt.messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors=None, truncation=False,
    )
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if (not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids)
            or mask != [1] * len(ids)):
        raise ValueError("Unexpected unpadded tokenizer result")
    return len(ids)


def request_preflight(config, tokenizer, prompt, task, pair_id, variant):
    """Count the untouched prompt and prove the output grammar fits on CPU."""
    count = prompt_token_count(tokenizer, prompt)
    record = {"task": task, "pair_id": pair_id, "model_variant": variant,
              "prompt_sha256": prompt.prompt_sha256, "input_tokens": count,
              "reserved_output_tokens": config.max_new_tokens,
              "context_window_tokens": config.context_window_tokens,
              "fits": count + config.max_new_tokens <= config.context_window_tokens}
    if record["fits"] and config.decoding_constraint == "observed_json_v1":
        from telemetry_reasoning.models.constrained import ObservedResponseConstraint
        record["output_constraint"] = ObservedResponseConstraint(
            prompt, tokenizer, tokenizer.eos_token_id, config.max_new_tokens).provenance()
    return record


class TokenizerStubBackend(StubBackend):
    """The same synthetic abstention, with exact local tokenizer measurements."""

    token_count_method = "pinned_tokenizer_cpu"

    def __init__(self, config, tokenizer_dir):
        super().__init__(config)
        self.tokenizer, self.tokenizer_lock = load_cpu_tokenizer(tokenizer_dir)
        if (self.tokenizer_lock['model_id'], self.tokenizer_lock['revision']) != (config.model_id, config.revision):
            raise ValueError("Stub tokenizer lock differs from configured model identity")

    def count_tokens(self, prompt):
        return prompt_token_count(self.tokenizer, prompt)

    def generate(self, prompt):
        generated = super().generate(prompt)
        output_ids = self.tokenizer(generated.text, add_special_tokens=False)['input_ids']
        return Generation(generated.text, generated.input_tokens, len(output_ids), self.token_count_method)

    def provenance(self):
        return {"implementation": "fixed_abstention_pinned_tokenizer_v1", "weights_loaded": False,
                "token_count_method": self.token_count_method, "tokenizer_lock": self.tokenizer_lock}


def _verify_production(bt_root, dataset_manifest, manifest_path):
    report_path = bt_root / "production_report.json"
    report = _json(report_path)
    if (report.get('production_version') != 1 or report.get('status') != 'validated_parser_publication'
            or report.get('sources_unchanged') is not True
            or report.get('source_inventory_sha256') != dataset_manifest['source_inventory_sha256']
            or report.get('input_sha256', {}).get(str(manifest_path.resolve())) != _fingerprint(manifest_path)['sha256']
            or report.get('pair_count') != dataset_manifest['validation']['accepted_pairs']):
        raise ValueError("Production report does not bind these BTs to the verified dataset handoff")
    expected = report.get('outputs')
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Production report has no output inventory")
    for name in expected:
        _safe(bt_root, name)
    if _inventory(bt_root, ('production_report.json',)) != expected:
        raise ValueError("Production BT/report inventory changed")
    return report_path


def select_pairs(metadata, task_manifest, *, split="dev", smoke=False, final_study=False):
    if split not in ("train", "dev", "test"):
        raise ValueError("Split must be train, dev, or test")
    if split == "test" and not final_study:
        raise ValueError("Held-out test inputs require --final-study")
    identities = validate_tuning_manifest(metadata, task_manifest)
    selected = sorted((row for row in metadata if identities[row.pair_id]["split"] == split),
                      key=lambda row: row.pair_id)
    if not selected:
        raise ValueError(f"No pairs in split {split}")
    if smoke:
        by_mode = {}
        for row in selected:
            by_mode.setdefault(row.failure_mode.value, row)
        selected = [by_mode[mode] for mode in sorted(by_mode)]
    return selected


def prepare_experiment(*, batch, handoff, bt_root, tokenizer_dir, output_dir,
                       split="dev", smoke=False, final_study=False,
                       llm_config=ROOT / "configs/llm.yaml",
                       rules_config=ROOT / "configs/rules.yaml",
                       eval_config=ROOT / "configs/eval.yaml"):
    """Verify immutable sources, count exact prompts, run rules, publish a bundle."""
    batch, handoff, bt_root = map(Path, (batch, handoff, bt_root))
    output_dir = _fresh(output_dir)
    for protected in (batch, handoff, bt_root, Path(tokenizer_dir)):
        source, destination = protected.resolve(), output_dir.resolve()
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("Experiment output must be separate from all input trees")
    manifest = _json(handoff / "dataset_manifest.json")
    errors = verify_handoff(handoff) + verify_dataset(batch, manifest)
    if errors:
        raise ValueError(f"Invalid dataset handoff: {errors[:5]}")
    if manifest.get("manifest_version") != 2:
        raise ValueError("A version 2 source-bound handoff is required")
    production_report = _verify_production(bt_root, manifest, handoff / 'dataset_manifest.json')
    config = load_llm_config(llm_config)
    if config.device != "cuda":
        raise ValueError("GPU experiment config must explicitly select device=cuda")
    tokenizer, tokenizer_lock = load_cpu_tokenizer(tokenizer_dir)
    if (tokenizer_lock['model_id'], tokenizer_lock['revision']) != (config.model_id, config.revision):
        raise ValueError("Tokenizer lock differs from model identity")
    rules = load_rule_config(rules_config)
    source_before = _sources()
    upstream = {}
    counts, requests, tasks = {}, [], {}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".experiment-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        (staged / "configs").mkdir()
        (staged / "prompts").mkdir()
        (staged / "tokenizer").mkdir()
        for name in tokenizer_lock['files']:
            shutil.copyfile(Path(tokenizer_dir) / name, staged / "tokenizer" / name)
        # Resolve custom templates now, then make all paths portable.
        templates = config.prompt_dir or DEFAULT_TEMPLATE_DIR
        for name in TEMPLATE_FILENAMES.values():
            source = templates / name
            upstream[str(source.resolve())] = _fingerprint(source)
            shutil.copyfile(source, staged / "prompts" / name)
        config_value = config.model_dump(mode="json", exclude_none=True)
        config_value.update(prompt_dir="../prompts")
        config_value.pop("cache_dir", None)
        (staged / "configs/llm.yaml").write_text(yaml.safe_dump({"llm": config_value}), encoding="utf-8")
        for name, source in (("rules.yaml", rules_config), ("eval.yaml", eval_config)):
            shutil.copyfile(source, staged / "configs" / name)
        shutil.copyfile(handoff / "dataset_manifest.json", staged / "dataset_manifest.json")
        shutil.copyfile(handoff / "publication_receipt.json", staged / "dataset_publication_receipt.json")
        shutil.copyfile(production_report, staged / "production_report.json")
        for source in (Path(llm_config), Path(rules_config), Path(eval_config),
                       production_report, handoff / "dataset_manifest.json", handoff / "publication_receipt.json"):
            upstream[str(source.resolve())] = _fingerprint(source)
        for task in TASKS:
            task_manifest_path = handoff / "tasks" / f"{task}.json"
            task_manifest = _json(task_manifest_path)
            metadata_path = batch / task / "metadata.csv"
            metadata = read_metadata_csv(metadata_path)
            if any(row.task.value != task for row in metadata):
                raise ValueError(f"Metadata task mismatch: {task}")
            selected = select_pairs(metadata, task_manifest, split=split, smoke=smoke, final_study=final_study)
            task_dir = staged / "tasks" / task
            (task_dir / "bt").mkdir(parents=True)
            shutil.copyfile(task_manifest_path, task_dir / "task_manifest.json")
            write_metadata_csv(task_dir / "metadata.csv", selected)
            for source in (task_manifest_path, metadata_path):
                upstream[str(source.resolve())] = _fingerprint(source)
            predictions = []
            for row in selected:
                if any(c in row.pair_id for c in '/\\\x00\r\n'):
                    raise ValueError("Unsafe pair identity")
                source = bt_root / task / "enriched-bt" / f"pair_{row.pair_id}_bt.json"
                before = _fingerprint(source)
                tree = PairedBehaviorTree.model_validate_json(source.read_bytes())
                if tree.metadata != row:
                    raise ValueError(f"BT metadata mismatch: {task}/{row.pair_id}")
                for variant in VARIANTS:
                    prompt = build_prompt(tree, variant, template_dir=templates)
                    requests.append(request_preflight(config, tokenizer, prompt, task, row.pair_id, variant))
                predictions.append(predict(tree.to_failure_only(), rules))
                shutil.copyfile(source, task_dir / "bt" / source.name)
                upstream[str(source.resolve())] = before
            write_predictions(task_dir / "baseline_1.jsonl", predictions)
            _write(task_dir / "baseline_1.manifest.json", {
                "model_variant": "baseline_1", "backend": "rules", "n_inputs": len(predictions),
                "config": rules.model_dump(mode="json"), "config_sha256": _fingerprint(rules_config)["sha256"],
                "status_counts": dict(Counter(row["status"] for row in predictions)),
                "input_files": {p.name: _fingerprint(p) for p in sorted((task_dir / "bt").glob('*.json'))},
            })
            counts[task] = len(selected)
            tasks[task] = {"pair_ids": [row.pair_id for row in selected],
                           "failure_modes": dict(Counter(row.failure_mode.value for row in selected))}
        if source_before != _sources() or any(_fingerprint(path) != recorded for path, recorded in upstream.items()):
            raise ValueError("Experiment inputs changed during preparation")
        _verify_production(bt_root, manifest, handoff / 'dataset_manifest.json')
        report = {
            "bundle_version": 1, "split": split, "smoke": smoke,
            "scope": "final_study" if split == "test" else "development",
            "status": ("ready_for_gpu" if all(r['fits'] for r in requests) else
                       "ready_for_gpu_with_context_refusals" if any(r['fits'] for r in requests) else
                       "blocked_context_budget"),
            "dataset_release_status": manifest["release_status"],
            "dataset_certification": manifest["certification"],
            "pair_counts": counts, "tasks": tasks,
            "request_count": len(requests), "token_budget": requests,
            "context_refused_requests": sum(not r['fits'] for r in requests),
            "tokenizer": tokenizer_lock, "source_files": source_before,
            "upstream_inputs": upstream,
            "note": "Rules executed on CPU; no model weights opened or HF completions generated. Test labels are held out unless final-study was explicitly requested.",
            "files": _inventory(staged),
        }
        _write(staged / "bundle_manifest.json", report)
        _fresh(output_dir)
        staged.rename(output_dir)
    return report


def verify_experiment(bundle, *, check_sources=True):
    bundle = Path(bundle)
    if bundle.is_symlink():
        raise ValueError("Symlink experiment bundle")
    manifest = _json(bundle / "bundle_manifest.json")
    if manifest.get("bundle_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported experiment bundle")
    for name in manifest["files"]:
        _safe(bundle, name)
    if _inventory(bundle, ("bundle_manifest.json",)) != manifest["files"]:
        raise ValueError("Experiment bundle inventory changed")
    if check_sources and _sources() != manifest.get("source_files"):
        raise ValueError("Inference/evaluation implementation differs from the prepared experiment")
    return manifest


def _check_task_run(bundle, task, target, manifest, backend, *, allow_response_errors=False):
    """Accept budget refusals only when independently predicted on the CPU.

    Every pair/variant remains in scoring. An unrelated error, a missing result,
    changed prompt/token count, or invented completion fails this verification.
    """
    expected = {(r['pair_id'], r['model_variant']): r for r in manifest['token_budget'] if r['task'] == task}
    inference = _json(target / 'inference/run_manifest.json')
    parsed = read_predictions(*(target / 'inference' / f'{variant}.jsonl' for variant in VARIANTS))
    predictions = {(r.pair_id, r.model_variant): r.model_dump(mode='json') for r in parsed.records}
    requests = {(r['pair_id'], r['model_variant']): r for r in inference['requests']}
    completions = [json.loads(line) for line in (target / 'inference/completions.jsonl').read_text().splitlines() if line.strip()]
    generated = {(r['pair_id'], r['model_variant']): r for r in completions}
    fitting = {key for key, row in expected.items() if row['fits']}
    if (parsed.invalid or len(predictions) != len(parsed.records) or set(predictions) != set(expected)
            or len(requests) != len(inference['requests']) or set(requests) != set(expected)
            or len(generated) != len(completions) or set(generated) != fitting):
        raise ValueError(f'{task}: incomplete, invalid, duplicate, or unexpected inference coverage')
    config = load_llm_config(bundle / 'configs/llm.yaml')
    trees, response_errors = {}, 0
    for key, preflight in expected.items():
        pair_id, variant = key
        request, prediction = requests[key], predictions[key]
        if (request.get('input_tokens') != preflight['input_tokens']
                or request.get('prompt_sha256') != preflight['prompt_sha256']
                or prediction['backend'] != backend):
            raise ValueError(f'{task}/{key}: inference differs from CPU token/prompt preflight')
        if not preflight['fits']:
            if (prediction['status'] != 'error' or request.get('status') != 'error'
                    or request.get('error_type') != 'ValueError'
                    or request.get('error_message') != 'prompt plus reserved output exceeds context budget; no truncation performed'
                    or prediction['evidence']):
                raise ValueError(f'{task}/{key}: expected explicit context refusal')
        else:
            if prediction['status'] not in ('ok', 'abstain') and not allow_response_errors:
                raise ValueError(f'{task}/{key}: fitting input failed inference or response validation')
            if pair_id not in trees:
                trees[pair_id] = PairedBehaviorTree.model_validate_json(
                    (bundle / 'tasks' / task / 'bt' / f'pair_{pair_id}_bt.json').read_bytes())
            prompt = build_prompt(trees[pair_id], variant, template_dir=config.prompt_dir)
            completion = generated[key]
            if (completion.get('messages') != prompt.messages
                    or completion.get('prompt_sha256') != prompt.prompt_sha256
                    or completion.get('completion_sha256') != hashlib.sha256(completion['text'].encode()).hexdigest()):
                raise ValueError(f'{task}/{key}: completion does not reproduce the validated prediction')
            if completion.get('metadata_error_fields'):
                raise ValueError(f'{task}/{key}: invalid generation metadata cannot be scored as a response error')
            try:
                reproduced = parse_response(completion['text'], prompt, pair_id, backend,
                                            response_policy=config.response_policy)
            except (ValueError, TypeError) as exc:
                if not allow_response_errors:
                    raise
                error_type, error_message = type(exc).__name__, str(exc)
                rejected = {'pair_id': pair_id, 'model_variant': variant,
                            'predicted_failure_mode': None, 'predicted_subtask': None,
                            'explanation': f'Inference rejected ({error_type}): {error_message[:240]}',
                            'evidence': [], 'status': 'error', 'backend': backend}
                if (prediction != rejected or request.get('status') != 'error'
                        or completion.get('status') != 'error'
                        or any(record.get('error_type') != error_type
                               or record.get('error_message') != error_message[:1000]
                               for record in (request, completion))):
                    raise ValueError(f'{task}/{key}: response rejection does not reproduce the retained error') from exc
                response_errors += 1
            else:
                if reproduced['status'] not in ('ok', 'abstain') or reproduced != prediction:
                    raise ValueError(f'{task}/{key}: completion does not reproduce the validated prediction')
    if backend == 'hf' and fitting and inference.get('backend_provenance', {}).get('weights_loaded') is not True:
        raise ValueError(f'{task}: no evidence of real weight loading')
    coverage = {'expected_requests': len(expected), 'validated_fit_requests': len(fitting) - response_errors,
                'context_refused_requests': len(expected) - len(fitting),
                'full_request_acceptance': len(expected) == len(fitting) and not response_errors}
    if allow_response_errors:
        coverage['response_error_requests'] = response_errors
    return coverage


def run_experiment(bundle, output_dir, *, backend="hf", inference_only=False, qualification=None):
    """Generate with one frozen config; optionally defer CPU scoring to the Mac.

    Input and structured-response checks remain active during generation.
    Inference-only runs publish every raw response, hash, error and provenance
    record needed by ``evaluate_gpu_results.py`` without creating metrics/plots.
    """
    if backend not in ("hf", "stub"):
        raise ValueError("Backend must be hf or stub")
    bundle = Path(bundle).resolve()
    manifest = verify_experiment(bundle)
    if manifest["status"] not in ("ready_for_gpu", "ready_for_gpu_with_context_refusals"):
        raise ValueError("CPU token preflight failed; refusing model execution")
    config = load_llm_config(bundle / "configs/llm.yaml")
    qualification_sha256 = None
    qualification_root = None
    if backend == "hf" and config.prefill_chunk_size is not None:
        if qualification is None:
            raise ValueError("This GPU profile requires a passed longest-request --qualification receipt")
        from telemetry_reasoning.gpu_qualification import verify_qualification
        verify_qualification(bundle, qualification)
        qualification_path = Path(qualification)
        if qualification_path.is_dir():
            qualification_path = qualification_path / "qualification_receipt.json"
        qualification_sha256 = _fingerprint(qualification_path)["sha256"]
        qualification_root = qualification_path.parent
    output_dir = _fresh(output_dir).resolve()
    output_dir.mkdir(parents=True)
    stages, failures, complete, coverage = [], [], False, {}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MPLBACKEND"] = "Agg"
    try:
        if qualification_root is not None:
            shutil.copytree(qualification_root, output_dir / "qualification")
            if _fingerprint(output_dir / "qualification/qualification_receipt.json")["sha256"] != qualification_sha256:
                raise ValueError("Qualification changed while copying the immutable execution evidence")
            verify_qualification(bundle, output_dir / "qualification")
        for task in TASKS:
            source, target = bundle / "tasks" / task, output_dir / task
            target.mkdir()
            commands = [
                ("inference", ["scripts/run_llm.py", "--bt-dir", str(source / "bt"), "--config", str(bundle / "configs/llm.yaml"),
                               "--variants", *VARIANTS, "--backend", backend, "--output-dir", str(target / "inference")]),
                ("evaluation", ["scripts/evaluate.py", "--metadata", str(source / "metadata.csv"),
                                "--predictions", str(source / "baseline_1.jsonl"),
                                *(str(target / "inference" / f"{variant}.jsonl") for variant in VARIANTS),
                                "--bt-dir", str(source / "bt"), "--manifest", str(source / "task_manifest.json"),
                                "--split", manifest["split"], "--variants", "baseline_1", *VARIANTS,
                                "--config", str(bundle / "configs/eval.yaml"),
                                "--run-manifest", str(source / "baseline_1.manifest.json"), str(target / "inference/run_manifest.json"),
                                "--output-dir", str(target / "evaluation")]),
            ]
            if inference_only:
                commands = [(name, arguments) for name, arguments in commands if name == 'inference']
            for name, arguments in commands:
                if name == 'inference' and backend == 'stub':
                    arguments.extend(['--stub-tokenizer-dir', str(bundle / 'tokenizer')])
                with (target / f"{name}.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env,
                                            stdout=log, stderr=subprocess.STDOUT, check=False)
                stages.append({"task": task, "stage": name, "exit_code": result.returncode})
                expected_refusal_exit = False
                if name == 'inference' and result.returncode in (0, 1):
                    try:
                        coverage[task] = _check_task_run(bundle, task, target, manifest, backend)
                        expected_refusal_exit = result.returncode == 1 and coverage[task]['context_refused_requests'] > 0
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        failures.append(str(exc))
                if result.returncode and not expected_refusal_exit:
                    failures.append(f"{task}/{name}: exit {result.returncode}")
        verify_experiment(bundle)
        complete = not failures
    except BaseException as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _write(output_dir / "experiment_receipt.json", {
            "run_version": 1, "backend": backend,
            "inference_only": inference_only,
            "qualification_receipt_sha256": qualification_sha256,
            "qualification_path": "qualification/qualification_receipt.json" if qualification_root is not None else None,
            "evaluation": "deferred_to_cpu" if inference_only else "included_in_run",
            "status": (("synthetic" if backend == "stub" else "passed") +
                       ("_with_context_refusals" if manifest.get('context_refused_requests', 0) else '')) if complete else "failed",
            "real_model_acceptance": backend == "hf" and complete and not manifest.get('context_refused_requests', 0),
            "real_generation_acceptance": backend == "hf" and complete,
            "request_coverage": coverage,
            "bundle_manifest_sha256": _fingerprint(bundle / "bundle_manifest.json")["sha256"],
            "dataset_release_status": manifest["dataset_release_status"],
            "split": manifest["split"], "pair_counts": manifest["pair_counts"],
            "stages": stages, "failures": failures, "files": _inventory(output_dir),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
    return 0 if complete else 1
