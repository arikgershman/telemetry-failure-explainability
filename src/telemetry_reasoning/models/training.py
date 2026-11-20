"""Reviewed-data-only, CUDA LoRA/DPO launch boundary.

Checking exports uses the CPU and imports no training runtime. Actual training
is opt-in, cache-only, bounded by resource/context guards, and never accepts
the formatter's synthetic demonstrations as optimization data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from telemetry_reasoning.models.llm import MODEL_ID, MODEL_REVISION, _host_resources
from telemetry_reasoning.models.prompts import ModelVariant, SYSTEM_PROMPT
from telemetry_reasoning.models.training_data import PreferenceProvenance, verify_training_data
from telemetry_reasoning.schemas import Task

GIB = 1024 ** 3
RUNTIME_PINS = {"torch": "2.6.0", "transformers": "4.49.0", "trl": "0.15.2",
                "peft": "0.14.0", "datasets": "3.3.2", "accelerate": "1.4.0"}
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class TrainingConfig(BaseModel):
    """A separate pinned DPO profile; no shared inference defaults are changed."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    method: Literal["dpo"] = "dpo"
    model_id: Literal["meta-llama/Llama-3.1-8B-Instruct"] = MODEL_ID
    revision: Literal["0e9e39f249a16976918f6564b8830bc894c89659"] = MODEL_REVISION
    device: Literal["cuda"] = "cuda"
    dtype: Literal["bfloat16"] = "bfloat16"
    local_files_only: Literal[True] = True
    max_length: Annotated[int, Field(strict=True, ge=1024, le=65536)] = 8192
    learning_rate: Annotated[float, Field(gt=0, le=1e-3)] = 5e-6
    beta: Annotated[float, Field(gt=0, le=1)] = 0.1
    epochs: Annotated[float, Field(gt=0, le=20)] = 1.0
    gradient_accumulation_steps: PositiveInt = 8
    seed: Annotated[int, Field(strict=True, ge=0, lt=2**32)] = 723
    lora_rank: Annotated[int, Field(strict=True, ge=1, le=128)] = 16
    lora_alpha: PositiveInt = 32
    lora_dropout: Annotated[float, Field(ge=0, lt=1)] = 0.0
    target_modules: tuple[Literal["q_proj", "k_proj", "v_proj", "o_proj"], ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj")

    @model_validator(mode="after")
    def targets_are_unique(self):
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target_modules must be nonempty and unique")
        return self


def load_training_config(path: Path | str) -> TrainingConfig:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"training"}:
        raise ValueError("training config must contain exactly one 'training' mapping")
    return TrainingConfig.model_validate(document["training"])


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_rows(path: Path) -> list[dict]:
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    rows = [json.loads(line, parse_constant=invalid_constant)
            for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSONL objects: {path.name}")
    return rows


def _messages(value: Any, roles: list[str]) -> None:
    if (not isinstance(value, list) or len(value) != len(roles)
            or any(not isinstance(item, dict) or set(item) != {"role", "content"}
                   or item["role"] != role or not isinstance(item["content"], str)
                   or not item["content"].strip() for item, role in zip(value, roles))):
        raise ValueError("training messages must retain exact system/user/assistant roles and nonempty content")


def check_training(
    data_dir: Path | str, config_path: Path | str, variant: str, output_dir: Path | str,
    *, check_original_inputs: bool = False,
    tokenizer_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict]]]:
    """Check portable publication, provenance and split coverage without runtimes.

    A check establishes launch-input validity only. It neither tokenizes model
    inputs nor claims CUDA availability, optimizer execution, or adapter quality.
    All variants are checked for export integrity before one explicit variant
    is selected; no held-out test data can enter either split.
    """
    data_dir, config_path, output_dir = map(Path, (data_dir, config_path, output_dir))
    config = load_training_config(config_path)
    selected = ModelVariant(variant).value
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("training output directory must be absent")
    source, destination = data_dir.resolve(), output_dir.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("training output must be separate from the immutable export")
    receipt = verify_training_data(data_dir, check_inputs=check_original_inputs)
    if (receipt.get("training_acceptance") != "reviewed_annotations_supplied"
            or receipt.get("preference_mode") != "human_reviewed"
            or receipt.get("effective_configuration", {}).get("synthetic_demo") is not False):
        raise ValueError("GPU training requires reviewed_annotations_supplied; synthetic_demo_only exports cannot train")
    if (receipt.get("excluded_splits") != ["test"]
            or receipt.get("split_unit") != "(task, initial_condition_id)"):
        raise ValueError("training export must exclude test and preserve task/initial-condition groups")
    rows = {split: _read_rows(data_dir / f"dpo/{split}.jsonl") for split in ("train", "dev")}
    sft = {split: _read_rows(data_dir / f"sft/{split}.jsonl") for split in rows}
    sidecar = _read_rows(data_dir / "records.jsonl")
    indexes: set[tuple[str, int]] = set()
    identities, annotations = set(), set()
    groups: dict[tuple[str, str], str] = {}
    pair_groups: dict[tuple[str, str], tuple[str, str]] = {}
    variants = set()
    chosen_rows: dict[str, list[dict]] = {split: [] for split in rows}
    selected_records = []
    for record in sidecar:
        split, row_index = record.get("split"), record.get("row")
        if (split not in rows or type(row_index) is not int or not 0 <= row_index < len(rows[split])
                or (split, row_index) in indexes):
            raise ValueError("sidecar must uniquely cover every train/dev row without test entries")
        indexes.add((split, row_index))
        if record.get("dpo_file") != f"dpo/{split}.jsonl" or record.get("sft_file") != f"sft/{split}.jsonl":
            raise ValueError("sidecar filenames disagree with split")
        task, pair = Task(record["task"]).value, record["pair_id"]
        model_variant = ModelVariant(record["model_variant"]).value
        variants.add(model_variant)
        condition = record["initial_condition_id"]
        if (not isinstance(pair, str) or not pair.strip()
                or isinstance(condition, bool) or not isinstance(condition, (str, int))
                or not str(condition).strip()):
            raise ValueError("sidecar pair and condition IDs must be nonempty")
        identity = (task, pair, model_variant)
        if identity in identities:
            raise ValueError("duplicate task/pair/variant preference")
        identities.add(identity)
        group = (task, str(condition))
        if groups.get(group, split) != split:
            raise ValueError("initial-condition group overlaps train and dev")
        groups[group] = split
        assignment = (split, str(condition))
        if pair_groups.get((task, pair), assignment) != assignment:
            raise ValueError("a task-local pair cannot change split or initial condition across variants")
        pair_groups[task, pair] = assignment
        provenance = PreferenceProvenance.model_validate(record["preference_provenance"])
        if provenance.kind != "human_reviewed" or provenance.annotation_id in annotations:
            raise ValueError("every preference requires a unique human_reviewed annotation")
        annotations.add(provenance.annotation_id)
        row = rows[split][row_index]
        if set(row) != {"prompt", "chosen", "rejected"}:
            raise ValueError("DPO row must have exactly prompt/chosen/rejected")
        _messages(row["prompt"], ["system", "user"])
        _messages(row["chosen"], ["assistant"])
        _messages(row["rejected"], ["assistant"])
        if row["prompt"][0]["content"] != SYSTEM_PROMPT:
            raise ValueError("training system instructions differ from inference")
        if row["chosen"] == row["rejected"]:
            raise ValueError("chosen and rejected responses must differ")
        prompt_sha = _hash(_canonical(row["prompt"]).encode())
        if (prompt_sha != record["prompt_sha256"] or prompt_sha != provenance.prompt_sha256
                or provenance.bt_sha256 != record["bt"]["sha256"]
                or any(_hash(row[key][0]["content"].encode()) != record[f"{key}_sha256"]
                       for key in ("chosen", "rejected"))):
            raise ValueError("training row does not match reviewed prompt/response hashes")
        if row_index >= len(sft[split]) or sft[split][row_index] != {"messages": row["prompt"] + row["chosen"]}:
            raise ValueError("DPO and chosen-SFT views disagree")
        if model_variant == selected:
            chosen_rows[split].append(row)
            selected_records.append({key: record[key] for key in (
                "task", "pair_id", "model_variant", "split", "row", "initial_condition_id",
                "prompt_sha256", "chosen_sha256", "rejected_sha256", "preference_provenance")})
    expected = {(split, index) for split in rows for index in range(len(rows[split]))}
    if indexes != expected or any(len(sft[split]) != len(rows[split]) for split in rows):
        raise ValueError("sidecar/SFT coverage must match every DPO export row")
    if receipt.get("rows") != {split: len(values) for split, values in rows.items()}:
        raise ValueError("declared split counts disagree with exports")
    if variants != set(receipt.get("variants", [])):
        raise ValueError("declared variants disagree with sidecar")
    if not all(chosen_rows.values()):
        raise ValueError("selected variant requires nonempty reviewed train and dev data")
    inputs = {path.relative_to(data_dir).as_posix(): _hash(path.read_bytes())
              for path in data_dir.rglob("*") if path.is_file()}
    report = {
        "schema_version": 1, "status": "inputs_validated_gpu_execution_pending",
        "model_variant": selected, "model_id": config.model_id, "revision": config.revision,
        "training_config": config.model_dump(mode="json"), "config_sha256": _hash(config_path.read_bytes()),
        "input_sha256": dict(sorted(inputs.items())), "selected_records": selected_records,
        "rows": {split: len(values) for split, values in chosen_rows.items()},
        "training_acceptance": receipt["training_acceptance"],
        "dataset_release_status": receipt["dataset_release_status"],
        "dataset_certification": receipt["dataset_certification"],
        "dataset_source_inventory_sha256": receipt["dataset_source_inventory_sha256"],
        "original_inputs_checked": check_original_inputs,
        "checks_pending": ["pinned training environment", "CUDA/BF16 and available memory",
                           "exact tokenizer context budgets", "optimizer execution", "adapter evaluation"],
        "implementation_sha256": {path.name: _hash(path.read_bytes()) for path in (
            Path(__file__), Path(__file__).with_name("training_data.py"),
            Path(__file__).with_name("prompts.py"), Path(__file__).with_name("llm.py"))},
    }
    if tokenizer_dir is not None:
        # Optional CPU work only: an exact local tokenizer is useful once real
        # annotations exist. The default check never imports this optional stack.
        from telemetry_reasoning.experiments import load_cpu_tokenizer

        tokenizer, lock = load_cpu_tokenizer(tokenizer_dir)
        if (lock["model_id"], lock["revision"]) != (config.model_id, config.revision):
            raise ValueError("CPU tokenizer identity differs from training model")
        _, _, lengths = _render_rows(chosen_rows, tokenizer, config)
        report["tokenizer_preflight"] = {"lock": lock, "token_lengths": lengths,
                                          "weights_loaded": False, "device_initialized": False}
        report["checks_pending"].remove("exact tokenizer context budgets")
    return report, chosen_rows


def _minimum_vram_gib(max_length: int) -> int:
    # Refusal floors for BF16 8B + LoRA, batch1, chosen/rejected concatenation,
    # SDPA and gradient checkpointing. Admission never guarantees an OOM-free run.
    return 32 if max_length <= 8192 else 48 if max_length <= 16384 else 64 if max_length <= 32768 else 80


def _gpu_runtime(config: TrainingConfig) -> tuple[Any, dict]:
    host = _host_resources()
    if (type(host.host_memory_bytes) is not int or host.host_memory_bytes < 64 * GIB
            or type(host.available_host_memory_bytes) is not int or host.available_host_memory_bytes < 48 * GIB):
        raise ValueError("training requires verified 64 GiB host RAM and 48 GiB currently available before importing runtimes")
    versions = {}
    for name, pinned in RUNTIME_PINS.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"install the dedicated training profile; missing {name}=={pinned}") from exc
        if actual.split("+", 1)[0] != pinned:
            raise ValueError(f"training profile requires {name}=={pinned}; found {actual}")
        versions[name] = actual
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("training requires a CUDA GPU supporting BF16; CPU/MPS fallback is disabled")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this training profile supports one CUDA process only")
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    floor = _minimum_vram_gib(config.max_length) * GIB
    if free < floor or total < floor:
        raise ValueError(f"this {config.max_length}-token training profile requires at least {floor // GIB} GiB free/total VRAM")
    return torch, {"host_total_bytes": host.host_memory_bytes,
                   "host_available_bytes": host.available_host_memory_bytes,
                   "cuda_device": device, "cuda_name": torch.cuda.get_device_name(device),
                   "vram_free_bytes": free, "vram_total_bytes": total,
                   "minimum_vram_bytes": floor, "package_versions": versions,
                   "python": platform.python_version(), "platform": platform.platform(),
                   "admission_note": "Minimum resource floors do not guarantee that training fits."}


def _render_rows(rows: dict[str, list[dict]], tokenizer: Any, config: TrainingConfig):
    """Render once and count the exact pinned TRL separate-tokenization policy."""
    rendered, expected_ids, lengths = {}, {}, []
    if tokenizer.eos_token_id is None:
        raise ValueError("training tokenizer must define EOS")
    for split, records in rows.items():
        rendered[split], expected_ids[split] = [], []
        for index, record in enumerate(records):
            prompt = tokenizer.apply_chat_template(record["prompt"], tokenize=False, add_generation_prompt=True)
            values = {"prompt": prompt}
            for key in ("chosen", "rejected"):
                full = tokenizer.apply_chat_template(record["prompt"] + record[key], tokenize=False,
                                                      add_generation_prompt=False)
                if not full.startswith(prompt):
                    raise ValueError("chat-template prompt is not an exact prefix of the complete conversation")
                values[key] = full[len(prompt):]
            ids = {f"{key}_input_ids": tokenizer(value, add_special_tokens=False)["input_ids"]
                   for key, value in values.items()}
            # TRL0.15.2 tokenize_row always appends EOS to completions, including
            # when the rendered chat template already ended in EOS/EOT.
            for key in ("chosen", "rejected"):
                ids[f"{key}_input_ids"].append(tokenizer.eos_token_id)
            prompt_length = len(ids["prompt_input_ids"])
            maximum = prompt_length + max(len(ids["chosen_input_ids"]), len(ids["rejected_input_ids"]))
            if maximum > config.max_length:
                raise ValueError(f"{split} row {index} requires {maximum} tokens, above max_length={config.max_length}; no truncation is allowed")
            lengths.append({"split": split, "row": index, "prompt_tokens": prompt_length,
                            "chosen_tokens": len(ids["chosen_input_ids"]),
                            "rejected_tokens": len(ids["rejected_input_ids"]), "maximum_total_tokens": maximum})
            rendered[split].append(values)
            expected_ids[split].append(ids)
    return rendered, expected_ids, lengths


def _write_report(path: Path, report: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_training(data_dir: Path, config_path: Path, variant: str, output_dir: Path,
                 *, check_original_inputs: bool = False) -> dict:
    """Launch one reviewed variant on CUDA; retain success or failure receipts."""
    report, rows = check_training(data_dir, config_path, variant, output_dir,
                                 check_original_inputs=check_original_inputs)
    config = load_training_config(config_path)
    torch, runtime = _gpu_runtime(config)  # Before tokenizer or weight loading.
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_id, revision=config.revision, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("pinned tokenizer must provide its instruction chat template")
    tokenizer.pad_token = tokenizer.eos_token
    rendered, expected_ids, lengths = _render_rows(rows, tokenizer, config)
    output_dir = Path(output_dir)
    # Keep setup/context refusal completely free of partial training outputs.
    output_dir.mkdir(parents=True, exist_ok=False)
    report.update(status="running", runtime=runtime, token_lengths=lengths,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  tokenizer_chat_template_sha256=_hash(tokenizer.chat_template.encode()),
                  reference_policy="frozen_base_with_adapter_disabled_v1",
                  truncation_policy="reject_oversize_before_weight_load",
                  optimizer_acceptance="pending", adapter_quality="not_evaluated")
    receipt_path = output_dir / "training_run_manifest.json"
    _write_report(receipt_path, report)
    try:
        (output_dir / "training_config.yaml").write_bytes(Path(config_path).read_bytes())
        (output_dir / "training_data_manifest.json").write_bytes((Path(data_dir) / "training_manifest.json").read_bytes())
        datasets = importlib.import_module("datasets")
        peft = importlib.import_module("peft")
        trl = importlib.import_module("trl")
        transformers.set_seed(config.seed)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_id, revision=config.revision, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": runtime["cuda_device"]})
        model.config.use_cache = False
        lora = peft.LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
                               lora_dropout=config.lora_dropout, target_modules=list(config.target_modules),
                               bias="none", task_type="CAUSAL_LM")
        arguments = trl.DPOConfig(
            output_dir=str(output_dir / "checkpoints"), beta=config.beta, loss_type="sigmoid",
            max_length=config.max_length, max_prompt_length=None, max_completion_length=None,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate, num_train_epochs=config.epochs,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True, fp16=False, optim="adamw_torch", seed=config.seed, data_seed=config.seed,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            logging_steps=1, report_to=[], push_to_hub=False, remove_unused_columns=True,
            dataloader_num_workers=0, dataset_num_proc=1, generate_during_eval=False,
        )
        trainer = trl.DPOTrainer(
            model=model, ref_model=None, peft_config=lora, args=arguments, processing_class=tokenizer,
            train_dataset=datasets.Dataset.from_list(rendered["train"]),
            eval_dataset=datasets.Dataset.from_list(rendered["dev"]))
        # Assert that the actual SDK preprocessing preserves every preflighted
        # token; a version/configuration change must not introduce truncation.
        for split, dataset in (("train", trainer.train_dataset), ("dev", trainer.eval_dataset)):
            if len(dataset) != len(expected_ids[split]):
                raise ValueError("training runtime changed dataset row coverage")
            for actual, expected in zip(dataset, expected_ids[split], strict=True):
                if any(actual[key] != ids for key, ids in expected.items()):
                    raise ValueError("training runtime tokenization differs from the exact context preflight")
        result = trainer.train()
        metrics = {**result.metrics, **trainer.evaluate()}
        if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in metrics.values()):
            raise ValueError("training produced nonfinite metrics")
        adapter = output_dir / "adapter"
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(str(output_dir / "tokenizer"))
        trainer.save_state()
        # Verify original export/config/source bytes after the entire run.
        if (_hash(Path(config_path).read_bytes()) != report["config_sha256"]
                or any(_hash((Path(data_dir) / name).read_bytes()) != fingerprint
                       for name, fingerprint in report["input_sha256"].items())
                or any(_hash(Path(__file__).with_name(name).read_bytes()) != fingerprint
                       for name, fingerprint in report["implementation_sha256"].items())):
            raise ValueError("training inputs, configuration or implementation changed during optimization")
        if not (adapter / "adapter_config.json").is_file() or not any(adapter.glob("adapter_model*.safetensors")):
            raise ValueError("training runtime did not save a PEFT adapter")
        report.update(status="completed", optimizer_acceptance="executed_on_reviewed_data",
                      metrics=metrics, inputs_unchanged=True,
                      completed_at=datetime.now(timezone.utc).isoformat())
        report["artifacts"] = {
            path.relative_to(output_dir).as_posix(): {"sha256": _hash(path.read_bytes()), "size_bytes": path.stat().st_size}
            for path in [output_dir / "training_config.yaml", output_dir / "training_data_manifest.json",
                         *(path for directory in (adapter, output_dir / "tokenizer")
                           for path in sorted(directory.rglob("*")))]
            if path.is_file()}
        _write_report(receipt_path, report)
        return report
    except Exception as exc:
        report.update(status="failed", optimizer_acceptance="not_accepted",
                      failure={"type": type(exc).__name__, "message": str(exc)},
                      completed_at=datetime.now(timezone.utc).isoformat())
        _write_report(receipt_path, report)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training-gpu.yaml"))
    parser.add_argument("--variant", choices=[variant.value for variant in ModelVariant], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="default: CPU input checks, no runtime imports")
    action.add_argument("--run", action="store_true", help="opt in to CUDA optimization")
    parser.add_argument("--check-original-inputs", action="store_true", help="also verify original local sources; omit after transfer")
    parser.add_argument("--tokenizer-dir", type=Path,
                        help="optional --check only: exact byte-locked local tokenizer, CPU and no weights")
    args = parser.parse_args(argv)
    try:
        if args.run:
            if args.tokenizer_dir is not None:
                raise ValueError("--tokenizer-dir is a CPU --check option; --run preflights its pinned tokenizer independently")
            report = run_training(args.data_dir, args.config, args.variant, args.output_dir,
                                  check_original_inputs=args.check_original_inputs)
        else:
            report, _ = check_training(args.data_dir, args.config, args.variant, args.output_dir,
                                       check_original_inputs=args.check_original_inputs,
                                       tokenizer_dir=args.tokenizer_dir)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        print(f"training error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
