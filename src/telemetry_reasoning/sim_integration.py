"""Fresh, source-bound MuJoCo CPU acceptance through the public analysis CLIs.

Archived accepted configurations are candidate inputs, never outcome evidence.
Every candidate is rerun twice; only independently verified physical failure
pairs enter downstream evaluation. Excessive-speed activation is a separate
probe because activation alone cannot establish task failure. Importing this
module does not import MuJoCo or any model inference runtime.
"""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

from telemetry_reasoning import integration as contract
from telemetry_reasoning.eval.contracts import index_predictions, read_predictions
from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv, write_metadata_csv
from telemetry_reasoning.schemas import FailureMode, MetadataRow, PairedBehaviorTree, Subtask, TASK_SUBTASKS, Task
from telemetry_reasoning.sim.failures import FailureConfig
from telemetry_reasoning.sim.outcomes import assess_physical_outcome
from telemetry_reasoning.sim.replay import compare_telemetry


RELEASED_MODES = {mode.value for mode in FailureMode if mode is not FailureMode.EXCESSIVE_SPEED}
DEFAULT_BATCH = contract.ROOT / "outputs/dataset/batch_20261030_152125_e5ca72de"
CONTEXT_REFUSAL = "prompt plus reserved output exceeds context budget; no truncation performed"


def released_candidates(batch: Path) -> tuple[list[dict], dict[str, Any]]:
    """Read every accepted configuration with its archived representative inputs.

    The catalogue's diagnostics hashes are checked before its seeds/conditions
    are used. This recovers inputs only; historical calibration rates remain
    unknown and are not inferred from the accepted records.
    """
    catalogue_path = batch / "provenance/recovered_failure_configs.json"
    catalogue = contract._json(catalogue_path)
    sources = {str(catalogue_path.resolve()): contract._fingerprint(catalogue_path)}
    candidates = []
    for task in Task:
        part = catalogue["tasks"][task.value]
        diagnostics_path = batch / task.value / "diagnostics.jsonl"
        fingerprint = contract._fingerprint(diagnostics_path)
        contract._require(fingerprint["sha256"] == part["diagnostics_sha256"],
                          f"{task.value}: catalogue diagnostics checksum differs")
        sources[str(diagnostics_path.resolve())] = fingerprint
        records = [json.loads(line) for line in diagnostics_path.read_text().splitlines() if line.strip()]
        indexed = {str(record["pair_id"]): record for record in records}
        contract._require(len(indexed) == len(records), "Duplicate archived diagnostic identities")
        seen = set()
        for index, item in enumerate(part["configurations"]):
            config = item["failure_config"]
            record = indexed[str(item["representative_pair_id"])]
            contract._require(config == record["failure_config"], "Catalogue configuration differs from diagnostics")
            key = json.dumps(config, sort_keys=True)
            contract._require(key not in seen, "Duplicate catalogue configuration")
            seen.add(key)
            FailureConfig(mode=FailureMode(config["mode"]), subtask=Subtask(config["subtask"]),
                          magnitude=config["magnitude"], delay_s=config["delay_s"], axis=config["axis"])
            candidates.append({"pair_id": f"{task.value}_{index:03d}", "task": task.value,
                               "seed": record["nominal"]["seed"],
                               "initial_condition_id": record["initial_condition_id"],
                               "failure_config": config,
                               "archived_representative_pair_id": str(item["representative_pair_id"]),
                               "input_origin": "archived_accepted_configuration_only"})
        contract._require({item["failure_config"]["mode"] for item in candidates if item["task"] == task.value}
                          == RELEASED_MODES, f"{task.value}: expected all four released modes")
        contract._require(seen == {json.dumps(record["failure_config"], sort_keys=True) for record in records},
                          f"{task.value}: catalogue does not cover every unique accepted configuration")
    return candidates, sources


def _producer_inventory() -> dict:
    sources = list((contract.ROOT / "src/telemetry_reasoning/sim").glob("*.py"))
    sources += [contract.ROOT / "src/telemetry_reasoning" / name for name in ("schemas.py", "io.py")]
    models = [contract.ROOT / "assets/scenes/pickup.xml"]
    models += [path for path in (contract.ROOT / "assets/mujoco_menagerie/franka_emika_panda").rglob("*")
               if path.is_file()]
    return {kind: {path.relative_to(contract.ROOT).as_posix(): contract._fingerprint(path)["sha256"]
                   for path in sorted(set(paths))}
            for kind, paths in (("source_files", sources), ("model_files", models))}


def _verify_producer(producer: dict, expected: dict) -> None:
    contract._require(producer.get("recorded_at_generation") is True and bool(producer.get("mujoco_version")),
                      "Missing generation provenance")
    for kind in ("source_files", "model_files"):
        contract._require(producer.get(kind) == expected[kind] and bool(expected[kind]),
                          "Recorded producer source/model inventory differs from current inputs")
        for name, recorded_hash in producer[kind].items():
            path = Path(name)
            contract._require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
                              "Unsafe producer dependency path")
            contract._require(contract._fingerprint(contract.ROOT / path)["sha256"] == recorded_hash,
                              "Recorded producer dependency hash differs from actual bytes")
        digest = hashlib.sha256("\n".join(f"{name}:{value}" for name, value in sorted(expected[kind].items())).encode()).hexdigest()
        contract._require(producer.get("source_sha256" if kind == "source_files" else "model_sha256") == digest,
                          "Producer dependency summary hash differs from inventory")


def _terminal_outcome(task: str, diagnostics: dict, producer_inputs: dict) -> dict:
    state = diagnostics["terminal_state"]
    contract._require(state.get("terminal_observed") is True, "Missing observed terminal state")
    outcome = assess_physical_outcome(task, state["initial_position"], state["final_position"],
                                      state["final_velocity"], source="simulator_terminal_state")
    contract._require(outcome == diagnostics["physical_outcome"], "Producer physical outcome differs from recomputed terminal goal")
    _verify_producer(diagnostics["producer"], producer_inputs)
    return outcome


def _validate_episode(task: str, rows: list, diagnostics: dict, csv_path: Path, producer_inputs: dict) -> dict:
    stored = read_telemetry_csv(csv_path)
    contract._require(stored == rows, "Returned observations differ from canonical CSV")
    contract._require(bool(rows) and rows[0].timestamp == 0, "Canonical observations must start at zero")
    stages = []
    for row in rows:
        if not stages or stages[-1] != row.subtask:
            stages.append(row.subtask)
    contract._require(stages == list(TASK_SUBTASKS[Task(task)])[:len(stages)], "Observed stages are not a canonical prefix")
    contract._require([stage.value for stage in stages] == diagnostics["observed_subtasks"], "Diagnostic stage coverage differs from CSV")
    contract._require(diagnostics["complete_subtasks"] == (len(stages) == len(TASK_SUBTASKS[Task(task)])),
                      "Diagnostic completeness differs from CSV stage coverage")
    origin = diagnostics["telemetry_start_sim_s"]
    end = diagnostics["terminal_state"]["simulator_time_s"]
    contract._require(isinstance(origin, (float, int)) and math.isfinite(origin) and origin >= 0
                      and math.isfinite(end) and end + 1e-8 >= origin + rows[-1].timestamp
                      and abs(end - diagnostics["duration_s"]) <= 1e-8,
                      "Invalid canonical origin or terminal time")
    outcome = _terminal_outcome(task, diagnostics, producer_inputs)
    return {"row_count": len(rows), "prefix": [stage.value for stage in stages],
            "canonical_origin_sim_s": origin, "terminal_sim_s": end,
            "physical_status": outcome["status"], "csv": contract._fingerprint(csv_path)}


def pre_injection_agreement(nominal, injected, nominal_diagnostics: dict, diagnostics: dict) -> dict:
    activation = diagnostics["failure"]
    contract._require(activation.get("activated") is True and activation.get("activation_steps", 0) > 0,
                      "Requested perturbation did not activate")
    origin = diagnostics["telemetry_start_sim_s"]
    onset = activation["first_activation_s"] - origin
    contract._require(math.isfinite(onset) and onset >= -1e-8, "Activation precedes canonical task origin")
    contract._require(abs(origin - nominal_diagnostics["telemetry_start_sim_s"]) <= 1e-8,
                      "Paired canonical origins differ before injection")
    before_nominal = [row for row in nominal if row.timestamp < onset - 1e-8]
    before_injected = [row for row in injected if row.timestamp < onset - 1e-8]
    if not before_nominal and not before_injected:
        return {"status": "not_observed", "reason": "Activation at canonical task entry; no strictly prior stored rows.",
                "row_count": 0, "injection_timestamp": max(0.0, onset)}
    comparison = compare_telemetry(before_nominal, before_injected)
    contract._require(comparison["matched"], "Full pre-injection observations differ")
    return {"status": "passed", "injection_timestamp": max(0.0, onset), **comparison}


def _repeat_episode(case: dict, kind: str, directory: Path, runner, producer_inputs: dict) -> tuple[Any, dict]:
    config = case["failure_config"] if kind == "injected" else None
    failure = (FailureConfig(**{**config, "mode": FailureMode(config["mode"]),
                                "subtask": Subtask(config["subtask"])}) if config else None)
    results, checks = [], []
    for repeat in range(2):
        csv_path = directory / f"{kind}_{repeat}.csv"
        with (directory / f"{kind}_{repeat}.stdout.log").open("w") as stream, redirect_stdout(stream):
            result = runner(task=case["task"], seed=case["seed"],
                            initial_condition_id=case["initial_condition_id"], failure=failure,
                            output=csv_path)
        contract._require(result.diagnostics["task"] == case["task"]
                          and result.diagnostics["seed"] == case["seed"]
                          and result.diagnostics["initial_condition_id"] == case["initial_condition_id"]
                          and result.diagnostics["failure_config"] == config,
                          "Producer identity or requested configuration differs")
        contract._require(isinstance(result.task_succeeded, bool)
                          and result.task_succeeded == (result.diagnostics["controller_outcome"] == "success"),
                          "Controller outcome differs from returned success flag")
        checks.append({"file": csv_path.relative_to(directory.parent.parent).as_posix(),
                       **_validate_episode(case["task"], result.rows, result.diagnostics, csv_path, producer_inputs)})
        contract._write_json(directory / f"{kind}_{repeat}.diagnostics.json", result.diagnostics)
        results.append(result)
    comparison = compare_telemetry(results[0].rows, results[1].rows)
    contract._require(comparison["matched"] and results[0].diagnostics == results[1].diagnostics,
                      f"{case['pair_id']} {kind}: fresh executions did not repeat")
    return results[0], {"executions": checks, "repeatability": comparison,
                        "diagnostics_identical": True}


def generate_fresh_pairs(candidates: list[dict], output_dir: Path, *, episode_runner=None,
                         producer_inputs: dict | None = None) -> dict:
    """Generate and independently qualify candidates, retaining every attempt.

    ``output_dir`` must already be a fresh run directory. The injected runner is
    an offline test seam; the public CLI always resolves the real MuJoCo runner.
    No caller-supplied acceptance flag can replace the physical checks.
    """
    if episode_runner is None:
        from telemetry_reasoning.sim.runner import run_episode
        episode_runner = run_episode
    producer_inputs = producer_inputs or _producer_inventory()
    attempts = output_dir / "producer"
    attempts.mkdir()
    inputs = output_dir / "input"
    inputs.mkdir()
    metadata, records, references = [], [], {}
    coverage = {task.value: set() for task in Task}
    for case in candidates:
        directory = attempts / case["pair_id"]
        directory.mkdir()
        nominal, nominal_check = _repeat_episode(case, "nominal", directory, episode_runner, producer_inputs)
        injected, injected_check = _repeat_episode(case, "injected", directory, episode_runner, producer_inputs)
        prehistory = pre_injection_agreement(nominal.rows, injected.rows, nominal.diagnostics, injected.diagnostics)
        accepted = (nominal.task_succeeded and nominal.diagnostics["physical_outcome"]["status"] == "succeeded"
                    and not injected.task_succeeded and injected.diagnostics["physical_outcome"]["status"] == "failed")
        record = {**case, "accepted_for_paired_analysis": accepted,
                  "nominal": nominal_check, "injected": injected_check,
                  "pre_injection_agreement": prehistory,
                  "classification_basis": "fresh controller outcome AND independent terminal physical goal"}
        records.append(record)
        contract._write_json(attempts / "attempts.json", records)
        if accepted:
            names = {kind: f"{case['pair_id']}_{kind}.csv" for kind in ("success", "failure")}
            shutil.copyfile(directory / "nominal_0.csv", inputs / names["success"])
            shutil.copyfile(directory / "injected_0.csv", inputs / names["failure"])
            config = case["failure_config"]
            metadata.append(MetadataRow(pair_id=case["pair_id"], task=case["task"],
                                        success_file=names["success"], failure_file=names["failure"],
                                        failed_subtask=config["subtask"], failure_mode=config["mode"],
                                        perturbation_param="magnitude", perturbation_value=config["magnitude"],
                                        seed=case["seed"]))
            coverage[case["task"]].add(config["mode"])
            references[case["pair_id"]] = {"split": "fresh_cpu_acceptance",
                                           "injection_timestamp": prehistory["injection_timestamp"],
                                           "onset_source": "recorded_at_generation_canonical_origin_and_activation"}
    contract._require(all(modes == RELEASED_MODES for modes in coverage.values()),
                      f"Fresh physical-failure coverage is incomplete: {coverage}")
    write_metadata_csv(inputs / "metadata.csv", metadata)
    contract._write_json(output_dir / "fresh_manifest.json", {
        "dataset_kind": "fresh_mujoco_acceptance", "archive_certification": "not_asserted",
        "pairs": references})
    # Speed probes check the actuator path and observability independently of a
    # successful/failing physical outcome; no fake fifth-class pair is created.
    speed_probes = []
    for task in Task:
        source = next(case for case in candidates if case["task"] == task.value)
        probe = {**source, "pair_id": f"{task.value}_speed_probe",
                 "failure_config": {"mode": "excessive_speed", "subtask": (
                     "approach" if task is Task.PICK_UP_OBJECT else "move_along_path"),
                     "magnitude": 2.0, "axis": 0, "delay_s": 0.0},
                 "input_origin": "explicit_uncalibrated_activation_probe"}
        directory = attempts / probe["pair_id"]
        directory.mkdir()
        result, check = _repeat_episode(probe, "injected", directory, episode_runner, producer_inputs)
        failure = result.diagnostics["failure"]
        contract._require(failure["activated"] and failure["activation_steps"] > 0,
                          "Excessive-speed perturbation did not activate")
        speed_probes.append({**probe, **check, "activation": failure,
                             "physical_status": result.diagnostics["physical_outcome"]["status"],
                             "required_physical_failure": False})
    contract._write_json(attempts / "speed_probes.json", speed_probes)
    return {"candidate_count": len(candidates), "accepted_pair_count": len(metadata),
            "rejected_candidate_count": len(candidates) - len(metadata),
            "seed_set": sorted({case["seed"] for case in candidates}),
            "coverage": {task: sorted(modes) for task, modes in coverage.items()},
            "repeat_executions": 4 * len(candidates) + 2 * len(speed_probes),
            "pre_injection_status_counts": {status: sum(record["pre_injection_agreement"]["status"] == status
                                                        for record in records)
                                            for status in ("passed", "not_observed")},
            "speed_probe_count": len(speed_probes), "archive_certification": "not_asserted"}


def _validate_stub_outcome(record, request: dict, config: dict) -> bool:
    """Return whether this is a fully proven context refusal, or reject it.

    Stub errors may be expected negative cases only with the actual pinned
    tokenizer count and the matching context admission error. A generic runtime
    error, conservative byte estimate, or asserted refusal cannot pass this gate.
    """
    contract._require(record.backend == "stub" and not record.evidence and request["status"] == record.status,
                      "Stub predictions incorrectly claim model diagnosis")
    if record.status == "abstain":
        contract._require(request["input_tokens"] + config["max_new_tokens"] <= config["context_window_tokens"],
                          "Stub accepted an oversized prompt")
        return False
    contract._require(record.status == "error"
                      and record.explanation == f"Inference rejected (ValueError): {CONTEXT_REFUSAL}"
                      and request.get("error_type") == "ValueError"
                      and request.get("error_message") == CONTEXT_REFUSAL
                      and request.get("token_count_method") == "pinned_tokenizer_cpu"
                      and request.get("output_tokens") is None
                      and isinstance(request.get("input_tokens"), int)
                      and request["input_tokens"] + config["max_new_tokens"] > config["context_window_tokens"],
                      "An execution error was not an independently checked exact-token context refusal")
    return True


def _validate_analysis(output_dir: Path, *, stub_tokenizer_dir: Path | None = None) -> dict:
    from telemetry_reasoning.models.llm import StubBackend, load_llm_config

    config_path = contract.ROOT / "configs/llm.yaml"
    expected_config = load_llm_config(config_path)
    if stub_tokenizer_dir is not None:
        from telemetry_reasoning.experiments import TokenizerStubBackend
        counter = TokenizerStubBackend(expected_config, stub_tokenizer_dir)
    else:
        counter = StubBackend(expected_config)
    metadata = read_metadata_csv(output_dir / "input/metadata.csv")
    by_id = {row.pair_id: row for row in metadata}
    views, prompt_counts = {}, {}
    for pair_id, meta in by_id.items():
        before = PairedBehaviorTree.model_validate_json((output_dir / f"raw-bt/pair_{pair_id}_bt.json").read_bytes())
        after = PairedBehaviorTree.model_validate_json((output_dir / f"bt/pair_{pair_id}_bt.json").read_bytes())
        contract._require(before.metadata == after.metadata == meta, "Parsed metadata differs from fresh producer")
        for raw_node, node in zip(before.subtasks, after.subtasks):
            for kind in ("success", "failure"):
                source = read_telemetry_csv(output_dir / "input" / getattr(meta, f"{kind}_file"))
                observed = [sample for sample in source if sample.subtask == node.subtask]
                retained = getattr(raw_node, f"{kind}_samples")
                lookup = {sample.timestamp: sample for sample in observed}
                contract._require(bool(observed) == bool(retained)
                                  and all(sample == lookup.get(sample.timestamp) for sample in retained),
                                  "Compression fabricated or omitted an observed stage")
                if observed:
                    contract._require(retained[0] == observed[0] and retained[-1] == observed[-1],
                                      "Compression lost fresh stage endpoints")
                contract._require(retained == getattr(node, f"{kind}_samples"), "Enrichment changed observed telemetry")
            expected = "computed" if node.failure_samples else "not_comparable"
            contract._require(node.divergence_status == expected, "Enrichment fabricated or omitted a comparison")
        views[pair_id] = contract._validate_views(after)
        for variant in contract.VARIANTS[1:]:
            prompt_counts[pair_id, variant] = counter.count_tokens(contract.build_prompt(after, variant))
    report = contract._json(output_dir / "raw-bt/parse_report.json")
    contract._require(report["counts"] == {"input_pairs": len(metadata), "written_pairs": len(metadata), "rejected_pairs": 0},
                      "Fresh accepted pairs did not all parse")
    predictions = read_predictions(*contract._prediction_paths(output_dir))
    contract._require(not predictions.invalid, "Invalid prediction record")
    indexed = index_predictions(predictions, by_id)
    contract._require(set(indexed.by_key) == {(pair_id, variant) for pair_id in by_id for variant in contract.VARIANTS},
                      "Incomplete or extra pair/variant prediction coverage")
    llm = contract._json(output_dir / "predictions/llm/run_manifest.json")
    contract._require(llm["config"] == expected_config.model_dump(mode="json")
                      and llm["config_file_sha256"] == contract._fingerprint(config_path)["sha256"]
                      and llm["backend_provenance"] == counter.provenance(),
                      "Executed stub configuration/tokenizer provenance differs from intended inputs")
    contract._require(llm["acceptance"]["status"] == "synthetic"
                      and llm["execution"]["hf_weights_loaded"] is False
                      and llm["execution"]["hf_generation_completed"] is False,
                      "Stub execution incorrectly claims real inference")
    contract._require(len(llm["requests"]) == len(metadata) * 3, "Incomplete executed prompt provenance")
    requests = {(request["pair_id"], request["model_variant"]): request for request in llm["requests"]}
    contract._require(set(requests) == {(pair_id, variant) for pair_id in by_id for variant in contract.VARIANTS[1:]},
                      "Missing, extra, or repeated executed prompt provenance")
    for request in llm["requests"]:
        expected = views[request["pair_id"]][request["model_variant"]]
        contract._require(all(request[key] == expected[key] for key in ("prompt_sha256", "template_sha256")),
                          "Executed prompt differs from verified view")
        contract._require(request.get("input_tokens") == prompt_counts[request["pair_id"], request["model_variant"]]
                          and request.get("token_count_method") == counter.token_count_method,
                          "Executed token count differs from independent prompt recount")
    expected_inputs = {pair_id: contract._fingerprint(output_dir / f"bt/pair_{pair_id}_bt.json")["sha256"]
                       for pair_id in by_id}
    contract._require(len(llm["input_files"]) == len(expected_inputs)
                      and {item["pair_id"]: item["sha256"] for item in llm["input_files"]} == expected_inputs,
                      "LLM provenance differs from fresh enriched inputs")
    context_refusals = []
    for record in predictions.records:
        if record.model_variant == "baseline_1":
            contract._require(record.backend == "rules" and record.status in ("ok", "abstain"),
                              "Rule predictions contain an execution error or wrong backend")
            continue
        request = requests[(record.pair_id, record.model_variant)]
        if _validate_stub_outcome(record, request, llm["config"]):
            context_refusals.append({"pair_id": record.pair_id, "task": by_id[record.pair_id].task.value,
                                     "model_variant": record.model_variant, "input_tokens": request["input_tokens"],
                                     "reserved_output_tokens": llm["config"]["max_new_tokens"],
                                     "context_window_tokens": llm["config"]["context_window_tokens"]})
    metrics = contract._json(output_dir / "evaluation/metrics.json")
    contract._require(contract._finite_numbers(metrics) and metrics["n_truth_pairs"] == len(metadata)
                      and set(metrics["variants"]) == set(contract.VARIANTS), "Invalid evaluation coverage or numeric metrics")
    for variant, entry in metrics["variants"].items():
        overall = entry["slices"]["all"]
        expected_errors = sum(item["model_variant"] == variant for item in context_refusals)
        contract._require(overall["n"] == len(metadata)
                          and overall["n_error"] == expected_errors
                          and all(overall[key] == 0 for key in ("n_invalid", "n_missing")),
                          "Incomplete evaluation outcomes")
        if variant != "baseline_1":
            contract._require(overall["n_abstain"] == len(metadata) - expected_errors,
                              "Stub abstentions or context refusals disappeared from scoring")
        contract._require(entry["evidence"]["checked"] and entry["evidence"]["n_unverifiable"] == 0
                          and entry["evidence"]["n_unsupported"] == 0,
                          "Evidence was not checked against fresh BTs")
        contract._require(entry["synthetic"] == (variant != "baseline_1"), "Synthetic inference labeling differs")
    pngs = list((output_dir / "plots").glob("*.png")) + list((output_dir / "evaluation/plots").glob("*.png"))
    contract._require(len(list((output_dir / "plots").glob("telemetry_*.png"))) == len(metadata)
                      and len(pngs) >= len(metadata) + 1, "Missing fresh telemetry or evaluation plots")
    contract._require(all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in pngs), "Invalid plot PNG")
    return {"prediction_count": len(predictions.records), "metrics_finite": True,
            "evidence_checked": True, "views_checked": len(views),
            "stub_requests_complete": not context_refusals,
            "context_refusals": {"requested": len(requests), "refused": len(context_refusals),
                                 "stub_abstentions": len(requests) - len(context_refusals),
                                 "by_variant": {variant: sum(item["model_variant"] == variant for item in context_refusals)
                                                for variant in contract.VARIANTS[1:]},
                                 "by_task": {task.value: sum(item["task"] == task.value for item in context_refusals)
                                             for task in Task},
                                 "by_pair": {pair_id: sum(item["pair_id"] == pair_id for item in context_refusals)
                                             for pair_id in by_id}, "requests": context_refusals},
            "rule_metrics": metrics["variants"]["baseline_1"]["slices"]["all"],
            "plot_files": [path.relative_to(output_dir).as_posix() for path in sorted(pngs)]}


def run_mujoco_cpu_smoke(output_dir: Path | str, *, batch: Path | str = DEFAULT_BATCH,
                         parser_config: Path | str = contract.ROOT / "configs/parser.yaml",
                         stub_tokenizer_dir: Path | str | None = None,
                         stage_timeout_s: float = 180.0, episode_runner=None) -> dict:
    """Execute fresh physics and every analysis CLI; preserve honest failure receipts.

    This accepts current producer reproducibility, not the original archive or
    real language-model quality. Destination must be absent or empty. Each
    generated file is source-bound and source data are immutable.
    """
    if not math.isfinite(stage_timeout_s) or stage_timeout_s <= 0:
        raise ValueError("stage timeout must be finite and positive")
    batch = Path(batch).resolve()
    parser_config = Path(parser_config).resolve()
    configuration_inputs = {str(parser_config): contract._fingerprint(parser_config)}
    if stub_tokenizer_dir is not None:
        stub_tokenizer_dir = Path(stub_tokenizer_dir).resolve()
        if not stub_tokenizer_dir.is_dir():
            raise ValueError("stub tokenizer directory must be a local directory")
        tokenizer_inputs = contract._inventory(stub_tokenizer_dir)
        if not tokenizer_inputs:
            raise ValueError("stub tokenizer directory is empty")
    else:
        tokenizer_inputs = None
    output_path = Path(output_dir)
    if output_path.resolve().is_relative_to(batch):
        raise ValueError("Integration output must be outside the immutable dataset batch")
    candidates, release_sources = released_candidates(batch)
    output_dir = contract._prepare_output(output_path)
    manifest = {"manifest_version": 1, "status": "running", "synthetic": False,
                "scope": "fresh_mujoco_cpu_acceptance", "sim_backend": "mujoco", "backend": "stub",
                "language_model_execution": "synthetic_stub_only", "archive_certification": "not_asserted",
                "producer_acceptance": "running", "model_complete": False,
                "python": platform.python_version(), "output_dir": str(output_dir), "stages": [], "checks": {},
                "started_utc": datetime.now(timezone.utc).isoformat(), "release_input_files": release_sources,
                "configuration_inputs": configuration_inputs,
                "stub_tokenizer": {"directory": str(stub_tokenizer_dir), "files": tokenizer_inputs}
                                  if stub_tokenizer_dir is not None else None,
                "limitations": ["Fresh producer acceptance cannot certify historical archive bytes or clocks.",
                                "Accepted archive configurations do not recover original calibration rates.",
                                "Stub abstentions test handoffs; real HF inference and GPU training remain separate.",
                                "Small accepted sample is not a dataset-wide quality estimate."]}
    contract._write_json(output_dir / contract.MANIFEST_NAME, manifest)
    try:
        manifest["source_files"] = contract._source_inventory()
        producer_inputs = manifest["producer_inputs"] = _producer_inventory()
        sample_before = contract._inventory(contract.ROOT / "data/sample")
        generated = {"name": "generate_mujoco", "status": "running", "executions": 0}
        manifest["stages"].append(generated)
        started = time.perf_counter()
        try:
            manifest["checks"]["producer"] = generate_fresh_pairs(candidates, output_dir, episode_runner=episode_runner,
                                                                 producer_inputs=producer_inputs)
            generated.update(status="passed", executions=manifest["checks"]["producer"]["repeat_executions"])
            manifest["producer_acceptance"] = "passed"
        except BaseException:
            generated["status"] = "failed"
            raise
        finally:
            generated["runtime_s"] = time.perf_counter() - started
            contract._write_json(output_dir / contract.MANIFEST_NAME, manifest)
        environment = dict(os.environ)
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
                            "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(output_dir / "runtime/matplotlib")})
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(contract.ROOT / "src"), environment.get("PYTHONPATH")]))
        manifest["offline_environment"] = {key: environment[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES")}
        def stage(name, *arguments, expected_returncodes=(0,)):
            contract._run_stage(name, arguments, output_dir, manifest, environment, stage_timeout_s,
                                expected_returncodes=expected_returncodes)
        data = output_dir / "input"
        input_before = contract._inventory(data)
        stage("parse", "parse_dataset.py", "--metadata", str(data / "metadata.csv"), "--data-dir", str(data),
              "--config", str(parser_config), "--output-dir", str(output_dir / "raw-bt"))
        stage("divergence", "enrich_divergence.py", "--bt-dir", str(output_dir / "raw-bt"),
              "--config", str(contract.ROOT / "configs/divergence.yaml"), "--output-dir", str(output_dir / "bt"))
        bt_before = {name: contract._inventory(output_dir / name) for name in ("raw-bt", "bt")}
        stage("rules", "run_baseline.py", "--bt-dir", str(output_dir / "bt"), "--config", str(contract.ROOT / "configs/rules.yaml"),
              "--output", str(contract._prediction_paths(output_dir)[0]))
        stage("llm_stub", "run_llm.py", "--bt-dir", str(output_dir / "bt"), "--backend", "stub",
              "--config", str(contract.ROOT / "configs/llm.yaml"), "--variants", *contract.VARIANTS[1:],
              *(["--stub-tokenizer-dir", str(stub_tokenizer_dir)] if stub_tokenizer_dir is not None else []),
              "--output-dir", str(output_dir / "predictions/llm"), expected_returncodes=(0, 1))
        reference = output_dir / "fresh_manifest.json"
        stage("evaluate", "evaluate.py", "--metadata", str(data / "metadata.csv"),
              "--predictions", *(str(path) for path in contract._prediction_paths(output_dir)),
              "--bt-dir", str(output_dir / "bt"), "--config", str(contract.ROOT / "configs/eval.yaml"), "--manifest", str(reference),
              "--variants", *contract.VARIANTS, "--run-manifest", str(output_dir / "predictions/baseline_1.manifest.json"),
              str(output_dir / "predictions/llm/run_manifest.json"), "--output-dir", str(output_dir / "evaluation"))
        stage("plot_telemetry", "plot_telemetry.py", "--data-dir", str(data), "--bt-dir", str(output_dir / "bt"),
              "--manifest", str(reference), "--predictions", str(contract._prediction_paths(output_dir)[0]),
              "--output-dir", str(output_dir / "plots"))
        manifest["checks"].update(_validate_analysis(output_dir, stub_tokenizer_dir=stub_tokenizer_dir))
        contract._require(contract._inventory(data) == input_before, "Downstream analysis changed fresh CSVs")
        contract._require({name: contract._inventory(output_dir / name) for name in ("raw-bt", "bt")} == bt_before,
                          "Prediction/evaluation changed BT inputs")
        contract._require(contract._inventory(contract.ROOT / "data/sample") == sample_before, "Versioned samples changed")
        contract._require(all(contract._fingerprint(Path(path)) == fingerprint for path, fingerprint in release_sources.items()),
                          "Archived input records changed")
        contract._require(all(contract._fingerprint(Path(path)) == fingerprint for path, fingerprint in configuration_inputs.items()),
                          "Selected parser configuration changed")
        if stub_tokenizer_dir is not None:
            contract._require(contract._inventory(stub_tokenizer_dir) == tokenizer_inputs,
                              "Local stub tokenizer files changed")
        contract._require(contract._source_inventory() == manifest["source_files"], "Source/configuration changed during integration")
        contract._require(_producer_inventory() == producer_inputs, "Producer source/model assets changed during integration")
        manifest["checks"].update(source_files_unchanged=True, generated_inputs_unchanged=True,
                                  bt_inputs_unchanged=True, versioned_samples_unchanged=True,
                                  archived_configuration_records_unchanged=True)
        manifest["checks"]["producer_source_and_model_unchanged"] = True
        manifest["status"] = ("passed_with_context_refusals" if manifest["checks"]["context_refusals"]["refused"]
                              else "passed")
        if manifest["checks"]["context_refusals"]["refused"]:
            next(record for record in manifest["stages"] if record["name"] == "llm_stub")["status"] = "passed_with_context_refusals"
    except BaseException as exc:
        manifest["status"] = "failed"
        if manifest["producer_acceptance"] == "running":
            manifest["producer_acceptance"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise contract.SmokeFailure(f"Fresh MuJoCo CPU acceptance failed: {exc}. Receipt: {output_dir / contract.MANIFEST_NAME}") from exc
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts"] = contract._inventory(output_dir, exclude_runtime=True)
        contract._write_json(output_dir / contract.MANIFEST_NAME, manifest)
    return manifest
