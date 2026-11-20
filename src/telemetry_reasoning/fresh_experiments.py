"""Freeze verified fresh physical cases for GPU-only completion generation.

This consumes a completed native CPU smoke, without rerunning physics or
claiming certification of the historical archive. Context eligibility is a
resource selection policy for the small smoke; the full bundle retains all
accepted cases and all expected context refusals.
"""

from collections import Counter
import math
from pathlib import Path
import re
import shutil
import tempfile

import yaml

from telemetry_reasoning import experiments as exp
from telemetry_reasoning.config import load_config
from telemetry_reasoning.eval.contracts import read_predictions, write_predictions
from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv, write_metadata_csv
from telemetry_reasoning.models.llm import load_llm_config
from telemetry_reasoning.models.prompts import DEFAULT_TEMPLATE_DIR, TEMPLATE_FILENAMES, build_prompt
from telemetry_reasoning.parser.compressor import build_paired_bt
from telemetry_reasoning.parser.divergence_metrics import DivergenceConfig, enrich_paired_bt
from telemetry_reasoning.schemas import PairedBehaviorTree
from telemetry_reasoning.sim.outcomes import assess_physical_outcome
from telemetry_reasoning.sim.replay import compare_telemetry

RELEASED_MODES = frozenset(("position_error", "orientation_error", "inadequate_grip_strength", "early_gripper_release"))
SPLIT = "fresh_cpu_acceptance"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_native_inventory(native_run, receipt):
    expected = receipt.get("artifacts")
    _require(isinstance(expected, dict) and bool(expected), "Native receipt has no artifact inventory")
    for name in expected:
        exp._safe(native_run, name)
    actual = exp._inventory(native_run, ("integration_manifest.json",))
    _require(all(actual.get(name) == fingerprint for name, fingerprint in expected.items()),
             "Native artifact inventory changed")
    # Matplotlib can add a versioned font cache after the original receipt.
    # It is never physical/model evidence. Bind its current bytes in the new
    # preparation snapshot while keeping every original artifact mandatory.
    extra = set(actual) - set(expected)
    _require(all(re.fullmatch(r"runtime/matplotlib/fontlist-v[^/]+\.json", name) for name in extra),
             "Native artifact inventory contains unexpected files")


def verify_native_run(native_run):
    """Recheck stored replay, outcomes, identities and retained observations.

    Original producer fingerprints remain evidence about those executions. They
    are compared to their recorded receipt, not replaced with today's source
    revision. No native/model runtime is imported and no simulation is rerun.
    """
    native_run = Path(native_run)
    _require(not native_run.is_symlink(), "Symlink native run")
    receipt = exp._json(native_run / "integration_manifest.json")
    _require(receipt.get("sim_backend") == "mujoco" and receipt.get("synthetic") is False
             and receipt.get("producer_acceptance") == "passed"
             and receipt.get("status") in ("passed", "passed_with_context_refusals"),
             "Native run must pass real physical-producer acceptance")
    _require(receipt.get("archive_certification") == "not_asserted", "Fresh run must not assert archival certification")
    _verify_native_inventory(native_run, receipt)
    metadata = read_metadata_csv(native_run / "input/metadata.csv")
    by_id = {row.pair_id: row for row in metadata}
    _require(len(by_id) == len(metadata) and bool(metadata), "Native metadata identities must be unique and nonempty")
    references = exp._json(native_run / "fresh_manifest.json")
    _require(references.get("dataset_kind") == "fresh_mujoco_acceptance"
             and references.get("archive_certification") == "not_asserted"
             and set(references.get("pairs", {})) == set(by_id), "Fresh timing manifest identity mismatch")
    attempts = exp._json(native_run / "producer/attempts.json")
    _require(isinstance(attempts, list), "Native attempts must be a list")
    accepted = {row["pair_id"]: row for row in attempts if row.get("accepted_for_paired_analysis") is True}
    _require(len(accepted) == sum(row.get("accepted_for_paired_analysis") is True for row in attempts)
             and set(accepted) == set(by_id), "Accepted producer attempts differ from metadata")
    _require(receipt.get("checks", {}).get("producer", {}).get("accepted_pair_count") == len(metadata),
             "Native receipt count differs from accepted metadata")
    producer_inputs = receipt.get("producer_inputs", {})
    for kind in ("source_files", "model_files"):
        _require(isinstance(producer_inputs.get(kind), dict) and bool(producer_inputs[kind]),
                 "Native producer source/model inventory is unavailable")
        for name, digest in producer_inputs[kind].items():
            _require(exp._fingerprint(exp._safe(exp.ROOT, name))["sha256"] == digest,
                     f"Original producer source/model bytes changed: {name}")
    parser_report = exp._json(native_run / "raw-bt/parse_report.json")
    _require(parser_report.get("counts") == {"input_pairs": len(metadata), "written_pairs": len(metadata), "rejected_pairs": 0},
             "Native parser report does not account for all accepted pairs")
    provenance = parser_report.get("provenance", {})
    parser_path = Path(provenance.get("config_file", ""))
    _require(exp._fingerprint(parser_path)["sha256"] == provenance.get("config_sha256")
             and receipt.get("configuration_inputs", {}).get(str(parser_path)) == exp._fingerprint(parser_path)
             and provenance.get("metadata_sha256") == exp._fingerprint(native_run / "input/metadata.csv")["sha256"],
             "Native parser configuration/metadata provenance changed")
    parser_config = load_config(parser_path).parser
    _require(parser_config.model_dump(mode="json") == parser_report.get("compression_config"),
             "Native parser report configuration mismatch")
    divergence_report = exp._json(native_run / "bt/divergence_report.json")
    _require(divergence_report.get("status") == "computed" and divergence_report.get("pair_count") == len(metadata),
             "Native divergence report is incomplete")
    divergence_config = DivergenceConfig.model_validate(divergence_report["divergence_config"])
    trees = {}
    for pair_id, row in by_id.items():
        _require(row.task.value in exp.TASKS and row.failure_mode.value in RELEASED_MODES,
                 "Fresh metadata has unsupported task or released mode")
        _require(not any(character in row.pair_id for character in "/\\\x00\r\n"), "Unsafe native pair identity")
        exp._safe(native_run, f"bt/pair_{pair_id}_bt.json")
        attempt = accepted[pair_id]
        failure_config = attempt["failure_config"]
        _require(attempt.get("task") == row.task.value and attempt.get("seed") == row.seed
                 and failure_config.get("mode") == row.failure_mode.value
                 and failure_config.get("subtask") == row.failed_subtask.value
                 and failure_config.get("magnitude") == row.perturbation_value,
                 f"Producer attempt identity mismatch: {pair_id}")
        traces, diagnostics = {}, {}
        for kind, side, expected_status in (("nominal", "success", "succeeded"), ("injected", "failure", "failed")):
            directory = native_run / "producer" / pair_id
            samples = [read_telemetry_csv(exp._safe(directory, f"{kind}_{i}.csv")) for i in range(2)]
            diag = [exp._json(exp._safe(directory, f"{kind}_{i}.diagnostics.json")) for i in range(2)]
            _require(compare_telemetry(*samples)["matched"] and diag[0] == diag[1],
                     f"Stored fresh executions do not repeat: {pair_id}/{kind}")
            _require(samples[0] == read_telemetry_csv(exp._safe(native_run / "input", getattr(row, f"{side}_file"))),
                     f"Published trace differs from producer observations: {pair_id}/{side}")
            info = diag[0]
            _require(info.get("task") == row.task.value and info.get("seed") == row.seed
                     and info.get("initial_condition_id") == attempt.get("initial_condition_id")
                     and info.get("failure_config") == (failure_config if kind == "injected" else None)
                     and info.get("controller_outcome") == ("success" if kind == "nominal" else "failed"),
                     f"Stored producer identity or controller outcome mismatch: {pair_id}/{kind}")
            state = info["terminal_state"]
            _require(state.get("terminal_observed") is True, "Independent terminal evidence is unavailable")
            outcome = assess_physical_outcome(row.task, state["initial_position"], state["final_position"],
                                              state["final_velocity"], source="simulator_terminal_state")
            _require(outcome == info.get("physical_outcome") and outcome["status"] == expected_status,
                     f"Independent physical objective failed verification: {pair_id}/{kind}")
            producer = info.get("producer", {})
            _require(producer.get("recorded_at_generation") is True and bool(producer.get("mujoco_version"))
                     and all(producer.get(key) == producer_inputs.get(key) and bool(producer.get(key))
                             for key in ("source_files", "model_files")), "Original producer fingerprints are missing or inconsistent")
            traces[side], diagnostics[side] = samples[0], info
        failure = diagnostics["failure"]["failure"]
        origin = diagnostics["failure"]["telemetry_start_sim_s"]
        onset = failure["first_activation_s"] - origin
        _require(failure.get("activated") is True and failure.get("activation_steps", 0) > 0
                 and math.isfinite(onset) and onset >= -1e-8
                 and abs(origin - diagnostics["success"]["telemetry_start_sim_s"]) <= 1e-8,
                 f"Recorded injection timing mismatch: {pair_id}")
        reference = references["pairs"][pair_id]
        _require(reference.get("split") == SPLIT
                 and reference.get("onset_source") == "recorded_at_generation_canonical_origin_and_activation"
                 and reference.get("injection_timestamp") == max(0.0, onset), "Fresh onset reference differs from recorded activation")
        before = [[sample for sample in traces[side] if sample.timestamp < onset - 1e-8]
                  for side in ("success", "failure")]
        prehistory = attempt["pre_injection_agreement"]
        _require((not any(before) and prehistory.get("status") == "not_observed")
                 or (all(before) and prehistory.get("status") == "passed" and compare_telemetry(*before)["matched"]),
                 f"Complete stored pre-injection comparison failed: {pair_id}")
        tree = PairedBehaviorTree.model_validate_json((native_run / f"bt/pair_{pair_id}_bt.json").read_bytes())
        _require(tree.metadata == row, f"BT metadata differs from native metadata: {pair_id}")
        raw_tree = PairedBehaviorTree.model_validate_json((native_run / f"raw-bt/pair_{pair_id}_bt.json").read_bytes())
        _require(raw_tree == build_paired_bt(row, native_run / "input", parser_config)
                 and tree == enrich_paired_bt(raw_tree, divergence_config),
                 f"Native BT compression/divergence is not reproducible from stored observations: {pair_id}")
        for node in tree.subtasks:
            for side in ("success", "failure"):
                observed = [sample for sample in traces[side] if sample.subtask == node.subtask]
                retained = getattr(node, f"{side}_samples")
                lookup = {sample.timestamp: sample for sample in observed}
                _require(bool(retained) == bool(observed)
                         and all(lookup.get(sample.timestamp) == sample for sample in retained)
                         and (not observed or (retained[0] == observed[0] and retained[-1] == observed[-1])),
                         f"BT observations differ from measured source: {pair_id}/{side}")
            _require(node.divergence_status == ("computed" if node.failure_samples else "not_comparable")
                     and (node.deterministic_divergence is not None) == bool(node.failure_samples),
                     f"BT divergence coverage mismatch: {pair_id}")
        trees[pair_id] = tree
    predictions = read_predictions(native_run / "predictions/baseline_1.jsonl")
    rule_predictions = {record.pair_id: record.model_dump(mode="json") for record in predictions.records}
    _require(not predictions.invalid and len(rule_predictions) == len(predictions.records)
             and set(rule_predictions) == set(by_id)
             and all(record["model_variant"] == "baseline_1" and record["backend"] == "rules"
                     for record in rule_predictions.values()), "Native rule prediction coverage mismatch")
    rule_manifest = exp._json(native_run / "predictions/baseline_1.manifest.json")
    _require(rule_manifest.get("backend") == "rules" and rule_manifest.get("model_variant") == "baseline_1"
             and rule_manifest.get("n_inputs") == len(metadata), "Native rule manifest mismatch")
    _require(all({row.failure_mode.value for row in metadata if row.task.value == task} == RELEASED_MODES
                 for task in exp.TASKS), "Fresh physical cases do not cover all released modes on both tasks")
    return receipt, metadata, trees, references, rule_predictions, rule_manifest


def prepare_fresh_experiment(*, native_run, tokenizer_dir, output_dir, smoke=False,
                             llm_config=exp.ROOT / "configs/llm.yaml",
                             eval_config=exp.ROOT / "configs/eval.yaml"):
    """Publish a compatible portable experiment using previously verified physics."""
    native_run, tokenizer_dir, output_dir = Path(native_run), Path(tokenizer_dir), exp._fresh(output_dir)
    for protected in (native_run, tokenizer_dir):
        source, destination = protected.resolve(), output_dir.resolve()
        _require(source != destination and source not in destination.parents and destination not in source.parents,
                 "Fresh experiment output must be separate from input trees")
    receipt_hash = exp._fingerprint(native_run / "integration_manifest.json")
    native_snapshot = exp._inventory(native_run)
    receipt, metadata, trees, references, predictions, rule_manifest = verify_native_run(native_run)
    config = load_llm_config(llm_config)
    _require(config.device == "cuda", "GPU experiment config must explicitly select device=cuda")
    tokenizer, tokenizer_lock = exp.load_cpu_tokenizer(tokenizer_dir)
    _require((tokenizer_lock["model_id"], tokenizer_lock["revision"]) == (config.model_id, config.revision),
             "Tokenizer lock differs from model identity")
    templates = config.prompt_dir or DEFAULT_TEMPLATE_DIR
    source_before = exp._sources()
    upstream = {str(Path(path).resolve()): exp._fingerprint(path)
                for path in (llm_config, eval_config, *(templates / name for name in TEMPLATE_FILENAMES.values()))}
    for name, fingerprint in native_snapshot.items():
        upstream[str((native_run / name).resolve())] = fingerprint
    upstream[str((native_run / "integration_manifest.json").resolve())] = receipt_hash
    for name, expected in tokenizer_lock["files"].items():
        _require(exp._fingerprint(exp._safe(tokenizer_dir, name)) == expected, "Pinned tokenizer file changed")
        upstream[str((tokenizer_dir / name).resolve())] = expected
    request_map = {}
    for row in metadata:
        request_map[row.pair_id] = []
        for variant in exp.VARIANTS:
            prompt = build_prompt(trees[row.pair_id], variant, template_dir=templates)
            request_map[row.pair_id].append(exp.request_preflight(
                config, tokenizer, prompt, row.task.value, row.pair_id, variant))
    selected, eligibility = {}, {}
    for task in exp.TASKS:
        candidates = sorted((row for row in metadata if row.task.value == task), key=lambda row: row.pair_id)
        fitting = [row for row in candidates if all(request["fits"] for request in request_map[row.pair_id])]
        eligibility[task] = {"candidate_pair_ids": [row.pair_id for row in candidates],
                             "fully_fitting_pair_ids": [row.pair_id for row in fitting]}
        if smoke:
            by_mode = {}
            for row in fitting:
                by_mode.setdefault(row.failure_mode.value, row)
            _require(set(by_mode) == RELEASED_MODES,
                     f"No fully context-fitting pair for every released mode: {task}")
            selected[task] = [by_mode[mode] for mode in sorted(by_mode)]
        else:
            selected[task] = candidates
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fresh-experiment-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for name in ("configs", "prompts", "tokenizer"):
            (staged / name).mkdir()
        for name in tokenizer_lock["files"]:
            shutil.copyfile(tokenizer_dir / name, staged / "tokenizer" / name)
        for name in TEMPLATE_FILENAMES.values():
            shutil.copyfile(templates / name, staged / "prompts" / name)
        config_value = config.model_dump(mode="json", exclude_none=True)
        config_value.update(prompt_dir="../prompts")
        config_value.pop("cache_dir", None)
        (staged / "configs/llm.yaml").write_text(yaml.safe_dump({"llm": config_value}), encoding="utf-8")
        shutil.copyfile(eval_config, staged / "configs/eval.yaml")
        shutil.copyfile(native_run / "integration_manifest.json", staged / "native_integration_manifest.json")
        shutil.copyfile(native_run / "fresh_manifest.json", staged / "fresh_manifest.json")
        requests, tasks = [], {}
        for task, rows in selected.items():
            task_dir = staged / "tasks" / task
            (task_dir / "bt").mkdir(parents=True)
            write_metadata_csv(task_dir / "metadata.csv", rows)
            exp._write(task_dir / "task_manifest.json", {"dataset_kind": "fresh_mujoco_acceptance",
                "archive_certification": "not_asserted", "pairs": {row.pair_id: references["pairs"][row.pair_id] for row in rows}})
            for row in rows:
                shutil.copyfile(native_run / f"bt/pair_{row.pair_id}_bt.json", task_dir / f"bt/pair_{row.pair_id}_bt.json")
                requests.extend(request_map[row.pair_id])
            rules = [predictions[row.pair_id] for row in rows]
            write_predictions(task_dir / "baseline_1.jsonl", rules)
            exp._write(task_dir / "baseline_1.manifest.json", {**rule_manifest,
                "n_inputs": len(rows), "status_counts": dict(Counter(row["status"] for row in rules)),
                "predictions": "baseline_1.jsonl", "bt_dir": "bt",
                "original_manifest_sha256": exp._fingerprint(native_run / "predictions/baseline_1.manifest.json")["sha256"],
                "input_files": {path.name: exp._fingerprint(path) for path in sorted((task_dir / "bt").glob("*.json"))}})
            tasks[task] = {"pair_ids": [row.pair_id for row in rows],
                           "failure_modes": dict(Counter(row.failure_mode.value for row in rows))}
        _verify_native_inventory(native_run, receipt)
        _require(native_snapshot == exp._inventory(native_run) and source_before == exp._sources()
                 and all(exp._fingerprint(path) == fingerprint for path, fingerprint in upstream.items()),
                 "Fresh experiment inputs changed during preparation")
        report = {"bundle_version": 1, "split": SPLIT, "smoke": smoke, "scope": "fresh_physical_acceptance",
            "status": ("ready_for_gpu" if all(row["fits"] for row in requests) else
                       "ready_for_gpu_with_context_refusals" if any(row["fits"] for row in requests) else "blocked_context_budget"),
            "dataset_release_status": "fresh_physical_acceptance", "archive_certification": "not_asserted",
            "dataset_certification": {"fresh_producer_acceptance": "passed", "archival_certification": "not_asserted"},
            "pair_counts": {task: len(rows) for task, rows in selected.items()}, "tasks": tasks,
            "request_count": len(requests), "token_budget": requests,
            "context_refused_requests": sum(not row["fits"] for row in requests),
            "tokenizer": tokenizer_lock, "source_files": source_before, "upstream_inputs": upstream,
            "native_receipt": receipt_hash, "native_model_complete": receipt.get("model_complete"),
            "selection_policy": {"policy": "first_pair_per_mode_with_all_three_prompts_fitting" if smoke else "all_accepted_fresh_pairs",
                "eligibility": eligibility, "test_data_used": False,
                "note": "Smoke eligibility uses context budget only, never model predictions or accuracy. This is a producer/inference smoke, not an unbiased performance study."},
            "note": "Reused verified native physics and CPU rules. No weights opened or HF completions generated. Original archive certification is not asserted.",
            "files": exp._inventory(staged)}
        exp._write(staged / "bundle_manifest.json", report)
        exp._fresh(output_dir)
        staged.rename(output_dir)
    return report
