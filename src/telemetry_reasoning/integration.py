"""CPU/offline acceptance of the public telemetry-to-explanation CLIs.

This smoke uses one newly generated synthetic pickup pair. It checks contracts,
not scientific accuracy, physics, or real-model execution. Every stage runs in a
separate process so broken CLI handoffs cannot hide behind in-process fakes.
An exclusive run directory retains logs and a failed receipt when a stage fails;
only a completely verified run receives status ``passed``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from telemetry_reasoning.eval.contracts import index_predictions, read_predictions
from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.models.prompts import build_prompt
from telemetry_reasoning.schemas import PairedBehaviorTree, Subtask, TelemetryFeature


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("baseline_1", "baseline_2", "baseline_3", "main")
MANIFEST_NAME = "integration_manifest.json"


class SmokeFailure(RuntimeError):
    """A stage or an acceptance check failed; inspect the retained run receipt."""


def _require(condition: bool, message: str) -> None:
    # Acceptance checks must remain active under python -O.
    if not condition:
        raise SmokeFailure(message)


def _json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _inventory(folder: Path, *, exclude_runtime: bool = False) -> dict[str, Any]:
    return {
        path.relative_to(folder).as_posix(): _fingerprint(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
        and not (exclude_runtime and path.relative_to(folder).parts[0] == "runtime")
    }


def _source_inventory() -> dict[str, Any]:
    paths = [
        *(ROOT / "src/telemetry_reasoning").rglob("*.py"),
        *(ROOT / "scripts").glob("*.py"),
        *(ROOT / "configs").glob("*.yaml"),
        *(ROOT / "configs/prompts").glob("*.txt"),
    ]
    return {path.relative_to(ROOT).as_posix(): _fingerprint(path) for path in sorted(paths)}


def _prepare_output(output_dir: Path) -> Path:
    if output_dir.is_symlink() or (output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    )):
        raise ValueError("--output-dir must be absent or empty; choose a fresh directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reserve even an existing empty directory atomically. Two concurrent
    # launchers can both pass the empty-directory check; only one may publish
    # receipts or stages into this destination. The retained claim also makes
    # an interrupted pre-receipt setup visibly occupied on the next invocation.
    try:
        with (output_dir / ".integration_claim").open("x", encoding="utf-8") as claim:
            claim.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise ValueError("--output-dir is already reserved; choose a fresh directory") from exc
    return output_dir.resolve()


def _run_stage(
    name: str, arguments: Sequence[str], output_dir: Path,
    manifest: dict[str, Any], environment: Mapping[str, str], timeout_s: float,
    *, expected_returncodes: Sequence[int] = (0,),
) -> None:
    command = [sys.executable, str(ROOT / "scripts" / arguments[0]), *arguments[1:]]
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    stdout_path, stderr_path = logs / f"{name}.stdout.log", logs / f"{name}.stderr.log"
    record: dict[str, Any] = {
        "name": name, "command": command, "status": "running",
        "stdout": stdout_path.relative_to(output_dir).as_posix(),
        "stderr": stderr_path.relative_to(output_dir).as_posix(),
        "expected_returncodes": list(expected_returncodes),
    }
    manifest["stages"].append(record)
    _write_json(output_dir / MANIFEST_NAME, manifest)
    started = time.perf_counter()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command, cwd=ROOT, env=dict(environment), stdout=stdout,
                stderr=stderr, timeout=timeout_s, check=False,
            )
        record["returncode"] = completed.returncode
        _require(completed.returncode in expected_returncodes,
                 f"{name} exited {completed.returncode}; see {record['stderr']}")
        record["status"] = "passed"
    except Exception:
        record["status"] = "failed"
        raise
    finally:
        record["runtime_s"] = time.perf_counter() - started
        _write_json(output_dir / MANIFEST_NAME, manifest)


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(child) for child in value))
    return set()


def _validate_views(tree: PairedBehaviorTree) -> dict[str, Any]:
    original = tree.model_dump_json()
    forbidden = {
        "failed_subtask", "failure_mode", "perturbation_param", "perturbation_value",
        "success_file", "failure_file", "seed",
    }
    failure_view = tree.to_failure_only()
    _require(not _keys(failure_view) & (forbidden | {"success_samples", "deterministic_divergence", "divergence_status"}),
             "Baseline 1 view leaks restricted fields")
    prompts = {variant: build_prompt(tree, variant) for variant in VARIANTS[1:]}
    for variant, prompt in prompts.items():
        keys = _keys(prompt.payload)
        _require(not keys & (forbidden | {"pair_id"}), f"{variant} prompt leaks storage metadata")
        _require(("success_samples" in keys) == (variant != "baseline_2"),
                 f"{variant} has incorrect success-view access")
        _require(("deterministic_divergence" in keys) == (variant == "main"),
                 f"{variant} has incorrect divergence-view access")
        _require(variant == "main" or "divergence_status" not in keys,
                 f"{variant} leaks divergence availability")
    _require({key: value for key, value in failure_view.items() if key != "pair_id"}
             == prompts["baseline_2"].payload, "Baselines 1 and 2 receive different telemetry")
    _require(len({prompt.system for prompt in prompts.values()}) == 1
             and len({prompt.template_sha256 for prompt in prompts.values()}) == 1,
             "Prompt instructions differ between variants")
    _require(tree.model_dump_json() == original, "View formatting mutated the stored BT")
    return {variant: {"prompt_sha256": prompt.prompt_sha256,
                      "template_sha256": prompt.template_sha256} for variant, prompt in prompts.items()}


def _validate_trees(output_dir: Path) -> dict[str, Any]:
    raw = PairedBehaviorTree.model_validate_json((output_dir / "raw-bt/pair_000_bt.json").read_bytes())
    tree = PairedBehaviorTree.model_validate_json((output_dir / "bt/pair_000_bt.json").read_bytes())
    source_metadata, = read_metadata_csv(output_dir / "input/metadata.csv")
    _require(raw.metadata == tree.metadata == source_metadata, "BT metadata does not match the freshly generated pair")
    source_rows = {kind: read_telemetry_csv(output_dir / "input" / getattr(source_metadata, f"{kind}_file"))
                   for kind in ("success", "failure")}
    for before, after in zip(raw.subtasks, tree.subtasks):
        for kind, rows in source_rows.items():
            observed = [row for row in rows if row.subtask == before.subtask]
            retained = getattr(before, f"{kind}_samples")
            by_timestamp = {row.timestamp: row for row in observed}
            _require(bool(observed) == bool(retained), "Parsed stage coverage differs from the generated CSV")
            _require(all(sample == by_timestamp.get(sample.timestamp) for sample in retained),
                     "Retained telemetry does not match the freshly generated CSV")
            if observed:
                _require(retained[0] == observed[0] and retained[-1] == observed[-1],
                         "Compression lost a generated stage endpoint")
        _require(before.success_samples == after.success_samples and before.failure_samples == after.failure_samples,
                 "Enrichment changed observed telemetry")
        if after.failure_samples:
            _require(before.divergence_status == "uncomputed" and before.deterministic_divergence is None,
                     "Parser no longer preserves uncomputed divergence")
            _require(after.divergence_status == "computed" and after.deterministic_divergence is not None,
                     "Observed stage has missing divergence")
            _require(set(after.deterministic_divergence.max_abs_deltas) == set(TelemetryFeature),
                     "Computed divergence lacks canonical features")
        else:
            _require(after.divergence_status == "not_comparable" and after.deterministic_divergence is None,
                     "Unobserved stage has a fabricated comparison")
    grasp = next(node for node in tree.subtasks if node.subtask is Subtask.GRASP)
    divergence = grasp.deterministic_divergence
    _require(divergence is not None, "Missing grasp divergence")
    _require(math.isclose(divergence.first_divergence_timestamp or -1, 5.10, abs_tol=1e-9, rel_tol=0),
             "Known mock onset is not 5.10 s")
    _require(math.isclose(divergence.max_abs_deltas[TelemetryFeature.GRIPPER_FORCE], 12.0, abs_tol=1e-9, rel_tol=0),
             "Known mock grip delta is not 12 N")
    timestamps = {sample.timestamp for sample in grasp.failure_samples}
    _require({5.08, 5.10, 5.12} <= timestamps, "Compression lost the grip-drop neighborhood")
    counts = _json(output_dir / "raw-bt/parse_report.json")["counts"]
    _require(counts == {"input_pairs": 1, "written_pairs": 1, "rejected_pairs": 0},
             "Parser did not accept exactly the generated pair")
    report = _json(output_dir / "bt/divergence_report.json")
    _require(report["pair_count"] == 1 and report["computed_subtasks"] == 4,
             "Divergence report does not cover every observed mock stage")
    return {"first_grasp_divergence_s": divergence.first_divergence_timestamp,
            "max_grip_delta_n": divergence.max_abs_deltas[TelemetryFeature.GRIPPER_FORCE],
            "retained_event_timestamps_s": [5.08, 5.10, 5.12],
            "views": _validate_views(tree)}


def _prediction_paths(output_dir: Path) -> list[Path]:
    # Explicit files keep run manifests and unrelated previous outputs out of evaluation.
    return [output_dir / "predictions/baseline_1.jsonl",
            *(output_dir / f"predictions/llm/{variant}.jsonl" for variant in VARIANTS[1:])]


def _validate_predictions(output_dir: Path, views: Mapping[str, Any]) -> None:
    predictions = read_predictions(*_prediction_paths(output_dir))
    _require(not predictions.invalid, "Prediction JSONL contains invalid records")
    indexed = index_predictions(predictions, ["000"])
    _require(set(indexed.by_key) == {("000", variant) for variant in VARIANTS},
             "Missing or unexpected pair/variant predictions")
    for record in predictions.records:
        if record.model_variant == "baseline_1":
            _require(record.backend == "rules" and record.status == "ok"
                     and record.predicted_failure_mode == "inadequate_grip_strength"
                     and record.predicted_subtask == "grasp" and bool(record.evidence),
                     "Rule baseline missed the known mock diagnosis")
        else:
            _require(record.backend == "stub" and record.status == "abstain" and not record.evidence,
                     "Stub output must be an explicit synthetic abstention")
    manifest = _json(output_dir / "predictions/llm/run_manifest.json")
    _require(manifest["acceptance"]["status"] == "synthetic"
             and manifest["execution"]["hf_weights_loaded"] is False
             and manifest["execution"]["hf_generation_completed"] is False,
             "Stub execution incorrectly claims real-model acceptance")
    requests = manifest["requests"]
    _require(len(requests) == 3 and {(request["pair_id"], request["model_variant"]) for request in requests}
             == {("000", variant) for variant in VARIANTS[1:]}, "LLM provenance has incomplete request coverage")
    for request in requests:
        expected = views[request["model_variant"]]
        _require(request["prompt_sha256"] == expected["prompt_sha256"]
                 and request["template_sha256"] == expected["template_sha256"],
                 "Executed prompt provenance does not match the verified model view")
    inputs = manifest["input_files"]
    _require(len(inputs) == 1 and inputs[0]["pair_id"] == "000"
             and inputs[0]["sha256"] == _fingerprint(output_dir / "bt/pair_000_bt.json")["sha256"],
             "LLM provenance does not match the enriched input BT")


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_numbers(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def _validate_results(output_dir: Path) -> dict[str, Any]:
    metrics = _json(output_dir / "evaluation/metrics.json")
    _require(_finite_numbers(metrics), "Metrics contain nonfinite numbers")
    _require(metrics["n_truth_pairs"] == 1 and set(metrics["variants"]) == set(VARIANTS),
             "Evaluation did not score exactly the requested pairs and variants")
    _require(metrics["unattributable_invalid_lines"]["count"] == 0, "Unattributable predictions entered evaluation")
    for variant, entry in metrics["variants"].items():
        overall = entry["slices"]["all"]
        _require(overall["n"] == 1 and all(overall[key] == 0 for key in ("n_error", "n_invalid", "n_missing")),
                 f"{variant} has an incomplete or invalid evaluation")
        evidence = entry["evidence"]
        _require(evidence["checked"] and evidence["n_unsupported"] == 0 and evidence["n_unverifiable"] == 0,
                 f"{variant} evidence was unsupported or not verified")
        _require(entry["synthetic"] == (variant != "baseline_1"), f"{variant} synthetic labeling is incorrect")
    rules = metrics["variants"]["baseline_1"]
    _require(rules["slices"]["all"]["joint_accuracy"] == 1.0 and rules["evidence"]["n_supported"] > 0,
             "Mock rule diagnosis or evidence verification failed")
    _require(rules["onset"]["computed"] and rules["onset"]["n_scored"] == 1,
             "Known synthetic injection reference was not scored")
    for variant in VARIANTS[1:]:
        _require(metrics["variants"][variant]["slices"]["all"]["n_abstain"] == 1,
                 "Stub abstentions disappeared from scoring")
    pngs = [output_dir / "plots/telemetry_000.png", output_dir / "evaluation/plots/variant_comparison.png"]
    pngs.extend(output_dir / f"evaluation/plots/confusion_{variant}_{scope}.png"
                for variant in VARIANTS for scope in ("all", "pick_up_object"))
    for path in pngs:
        _require(path.stat().st_size > 8 and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
                 f"Missing or invalid plot: {path.relative_to(output_dir)}")
    for relative in ("evaluation/metrics.csv", "evaluation/confusion.csv",
                     "evaluation/rubric_blinded.csv", "evaluation/rubric_key.json"):
        _require((output_dir / relative).stat().st_size > 0, f"Missing report: {relative}")
    return {"prediction_count": 4, "evidence_verified": True, "metrics_finite": True,
            "plot_files": [path.relative_to(output_dir).as_posix() for path in pngs]}


def run_cpu_smoke(output_dir: Path | str, *, seed: int = 723, stage_timeout_s: float = 120.0) -> dict[str, Any]:
    """Run all seven public CLI stages and return the verified synthetic receipt.

    ``output_dir`` must be absent or empty. Failures preserve logs, partial
    artifacts, and a failed receipt, then raise :class:`SmokeFailure`. Reusing a
    populated destination is refused before any file is modified. Undefined
    class metrics may be null; every numeric metric must be finite.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(stage_timeout_s) or stage_timeout_s <= 0:
        raise ValueError("stage timeout must be finite and positive")
    output_dir = _prepare_output(Path(output_dir))
    manifest: dict[str, Any] = {
        "manifest_version": 1, "status": "running", "synthetic": True,
        "scope": "cpu_contract_smoke", "backend": "stub", "sim_backend": "mock",
        "seed": seed, "output_dir": str(output_dir), "python": platform.python_version(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "stages": [], "checks": {},
        "limitations": ["One synthetic pickup pair; not a dataset-wide performance estimate.",
                        "Real MuJoCo, real HF inference, and release certification remain separate gates."],
    }
    _write_json(output_dir / MANIFEST_NAME, manifest)
    try:
        sample_before = _inventory(ROOT / "data/sample")
        manifest["source_files"] = _source_inventory()
        environment = dict(os.environ)
        environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MPLBACKEND": "Agg",
                            "MPLCONFIGDIR": str(output_dir / "runtime/matplotlib"), "CUDA_VISIBLE_DEVICES": ""})
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), environment.get("PYTHONPATH")]))
        manifest["offline_environment"] = {key: environment[key] for key in (
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MPLBACKEND", "CUDA_VISIBLE_DEVICES")}

        def stage(name: str, *arguments: str) -> None:
            _run_stage(name, arguments, output_dir, manifest, environment, stage_timeout_s)

        data = output_dir / "input"
        stage("generate_mock", "generate_mock_pair.py", "--output-dir", str(data), "--seed", str(seed),
              "--config", str(ROOT / "configs/default.yaml"))
        metadata = read_metadata_csv(data / "metadata.csv")
        _require(len(metadata) == 1 and metadata[0].pair_id == "000" and metadata[0].seed == seed
                 and metadata[0].task.value == "pick_up_object", "Unexpected mock pair identity, seed, or task")
        rows = {kind: read_telemetry_csv(data / getattr(metadata[0], f"{kind}_file")) for kind in ("success", "failure")}
        _require(all(len(trace) == 400 for trace in rows.values()), "Mock traces do not have 400 schema-valid rows")
        input_before = _inventory(data)
        # This reference is known from the synthetic producer, not inferred physical onset.
        reference = output_dir / "mock_manifest.json"
        _write_json(reference, {"dataset_kind": "synthetic", "pairs": {
            "000": {"split": "smoke", "injection_timestamp": 5.10,
                    "onset_source": "known synthetic generator event"}}})
        manifest["checks"]["raw_row_counts"] = {kind: len(trace) for kind, trace in rows.items()}
        stage("parse", "parse_dataset.py", "--metadata", str(data / "metadata.csv"), "--data-dir", str(data),
              "--config", str(ROOT / "configs/parser.yaml"), "--output-dir", str(output_dir / "raw-bt"))
        stage("divergence", "enrich_divergence.py", "--bt-dir", str(output_dir / "raw-bt"),
              "--config", str(ROOT / "configs/divergence.yaml"), "--output-dir", str(output_dir / "bt"))
        manifest["checks"].update(_validate_trees(output_dir))
        bt_before = {name: _inventory(output_dir / name) for name in ("raw-bt", "bt")}
        stage("rules", "run_baseline.py", "--bt-dir", str(output_dir / "bt"),
              "--config", str(ROOT / "configs/rules.yaml"), "--output", str(_prediction_paths(output_dir)[0]))
        stage("llm_stub", "run_llm.py", "--bt-dir", str(output_dir / "bt"), "--backend", "stub",
              "--config", str(ROOT / "configs/llm.yaml"), "--variants", *VARIANTS[1:],
              "--output-dir", str(output_dir / "predictions/llm"))
        _validate_predictions(output_dir, manifest["checks"]["views"])
        stage("evaluate", "evaluate.py", "--metadata", str(data / "metadata.csv"),
              "--predictions", *(str(path) for path in _prediction_paths(output_dir)),
              "--bt-dir", str(output_dir / "bt"), "--config", str(ROOT / "configs/eval.yaml"),
              "--manifest", str(reference), "--variants", *VARIANTS,
              "--run-manifest", str(output_dir / "predictions/baseline_1.manifest.json"),
              str(output_dir / "predictions/llm/run_manifest.json"), "--output-dir", str(output_dir / "evaluation"))
        stage("plot_telemetry", "plot_telemetry.py", "--data-dir", str(data), "--bt-dir", str(output_dir / "bt"),
              "--manifest", str(reference), "--predictions", str(_prediction_paths(output_dir)[0]),
              "--output-dir", str(output_dir / "plots"))
        manifest["checks"].update(_validate_results(output_dir))
        _require(_inventory(data) == input_before, "A downstream stage modified generated source files")
        _require({name: _inventory(output_dir / name) for name in ("raw-bt", "bt")} == bt_before,
                 "A prediction/evaluation stage modified its BT inputs")
        _require(_inventory(ROOT / "data/sample") == sample_before, "The versioned samples were modified")
        _require(_source_inventory() == manifest["source_files"], "Source/configuration changed during the smoke")
        manifest["checks"]["source_files_unchanged"] = True
        manifest["checks"]["generated_inputs_unchanged"] = True
        manifest["checks"]["bt_inputs_unchanged"] = True
        manifest["checks"]["versioned_samples_unchanged"] = True
        manifest["status"] = "passed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise SmokeFailure(f"CPU integration smoke failed: {exc}. Receipt: {output_dir / MANIFEST_NAME}") from exc
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts"] = _inventory(output_dir, exclude_runtime=True)
        _write_json(output_dir / MANIFEST_NAME, manifest)
    return manifest
