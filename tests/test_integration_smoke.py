"""Tests for scripts/integration_smoke.py (the CPU contract smoke).

The helper tests are fast and pure. The end-to-end tests run the real public CLIs
as subprocesses on the mock pair, entirely on CPU and offline (stub backend only).

The real-hardware smokes (MuJoCo producer, Hugging Face model) are tested here with a fake
simulator and a test double for the Hugging Face CLI. Those tests prove the orchestration, the
checks and the labelling; they do NOT prove the code works on real hardware. The two tests that
would are at the bottom, gated on the hardware, and skip with the missing resource named.
"""

from __future__ import annotations

import collections
import importlib.util
import inspect
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from telemetry_reasoning.io import read_telemetry_csv, write_metadata_csv
from telemetry_reasoning.schemas import FailureMode, MetadataRow, PairedBehaviorTree, Subtask, Task

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
BATCH = ROOT / "outputs" / "dataset" / "batch_20261030_152125_e5ca72de"
HANDOFF = BATCH / "handoff"  # Track 2's real per-task handoff manifests: <task>.json
TASKS = ("pick_up_object", "transport_object")


def handoff_path(task):
    """The real per-task manifest, or skip the calling test when the released batch is not present."""
    path = HANDOFF / f"{task}.json"
    if not path.is_file():
        pytest.skip(f"released Track 2 manifest not found: {path}")
    return path


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load_script("integration_smoke")


@pytest.fixture(scope="module")
def sample_tree():
    return PairedBehaviorTree.model_validate_json((SAMPLE / "pair_000_bt.json").read_text())


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_find_view_leaks_reports_forbidden_keys_at_any_depth_and_forbidden_values():
    assert smoke.find_view_leaks({"pair_id": "000", "subtasks": [{"subtask": "grasp", "failure_samples": []}]}) == []
    deep = {"subtasks": [{"metadata": {"seed": 723}}], "failure_mode": "x"}
    leaks = smoke.find_view_leaks(deep)
    assert "key subtasks[0].metadata.seed" in leaks and "key failure_mode" in leaks
    by_value = smoke.find_view_leaks({"overall_goal": "recover from inadequate_grip_strength"}, ["inadequate_grip_strength", ""])
    assert by_value == ["value 'inadequate_grip_strength' at overall_goal"]


def test_check_views_pass_for_a_real_tree_and_cover_all_three_views(sample_tree):
    results = smoke.check_views(sample_tree, ["inadequate_grip_strength", "failure_000.csv", "success_000.csv"])
    names = [n for n, _, _ in results]
    assert len(names) == 9 and len(set(names)) == 9
    for view in ("failure_only", "paired_without_divergence", "full_paired"):
        assert {f"view.{view}.success_samples", f"view.{view}.divergence", f"view.{view}.no_label_leakage"} <= set(names)
    # structural checks only (keys present/absent); the sample's divergence is uncomputed, which the
    # end-to-end smoke covers separately on the enriched trees
    assert all(ok for _, ok, _ in results), [r for r in results if not r[1]]


def test_check_views_detect_a_leaky_or_wrongly_shaped_view():
    node = {"subtask": "grasp", "success_samples": [{}], "deterministic_divergence": None, "divergence_status": "computed"}

    class LeakyTree:
        def to_failure_only(self):  # leaks the label and still carries success samples + divergence
            return {"subtasks": [dict(node, failure_mode="x")]}

        def to_paired_without_divergence(self):  # divergence fields should have been removed
            return {"subtasks": [dict(node)]}

        def to_full_paired(self):  # success samples missing
            return {"subtasks": [{k: v for k, v in node.items() if k != "success_samples"}]}

    failed = {n for n, ok, _ in smoke.check_views(LeakyTree(), []) if not ok}
    assert {"view.failure_only.success_samples", "view.failure_only.divergence", "view.failure_only.no_label_leakage",
            "view.paired_without_divergence.divergence", "view.full_paired.success_samples"} <= failed
    assert "view.paired_without_divergence.success_samples" not in failed


def test_measure_grasp_event_finds_the_known_event_in_the_mock_pair():
    success, failure = read_telemetry_csv(SAMPLE / "success_000.csv"), read_telemetry_csv(SAMPLE / "failure_000.csv")
    when, drop = smoke.measure_grasp_event(success, failure)
    assert when == pytest.approx(smoke.GRASP_EVENT_S) and drop == pytest.approx(smoke.GRASP_FORCE_DROP_N)


def test_measure_grasp_event_rejects_misaligned_and_identical_runs():
    success = read_telemetry_csv(SAMPLE / "success_000.csv")
    with pytest.raises(ValueError, match="same time grid"):
        smoke.measure_grasp_event(success, success[:-1])
    with pytest.raises(ValueError, match="never differs"):
        smoke.measure_grasp_event(success, success)


def test_all_finite_and_png_and_fresh_dir_helpers(tmp_path):
    assert smoke.all_finite({"a": [1, 2.5, None, "x", True], "b": {"c": 0.0}})
    assert not smoke.all_finite({"a": [1, float("nan")]}) and not smoke.all_finite([math.inf])

    good = tmp_path / "ok.png"
    good.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 2000)
    empty, fake = tmp_path / "empty.png", tmp_path / "fake.png"
    empty.write_bytes(b"")
    fake.write_bytes(b"not a png" + b"0" * 2000)
    assert smoke.is_nonempty_png(good)
    assert not smoke.is_nonempty_png(empty) and not smoke.is_nonempty_png(fake) and not smoke.is_nonempty_png(tmp_path / "none.png")

    smoke.require_fresh_dir(tmp_path / "absent")  # fine
    (tmp_path / "blank").mkdir()
    smoke.require_fresh_dir(tmp_path / "blank")  # empty directory is fine
    with pytest.raises(smoke.SmokeError, match="not empty"):
        smoke.require_fresh_dir(tmp_path)


# --------------------------------------------------------------------------- #
# Command-line behaviour that needs no pipeline run
# --------------------------------------------------------------------------- #
def test_a_missing_stage_cli_is_a_setup_error_not_a_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(smoke, "SCRIPTS", tmp_path)  # no scripts here: the first stage cannot run
    out = tmp_path / "out"
    assert smoke.main(["--output-dir", str(out)]) == 2
    assert "stage 'generate_mock_pair'" in capsys.readouterr().err
    assert not (out / "smoke_report.json").exists()


# --------------------------------------------------------------------------- #
# End to end: real public CLIs, mock pair, stub backend
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cpu_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("integration") / "cpu"
    code = smoke.main(["--backend", "stub", "--output-dir", str(out)])
    return code, out, json.loads((out / "smoke_report.json").read_text())


def test_cpu_contract_smoke_passes_end_to_end(cpu_run):
    code, _, report = cpu_run
    failed = [c for c in report["checks"] if not c["passed"]]
    assert code == 0 and report["passed"] is True and not failed, failed
    assert report["n_checks"] > 40 and report["n_failed"] == 0 and report["pair_ids"] == ["000"]


def test_smoke_report_labels_everything_synthetic_and_real_checks_not_run(cpu_run):
    _, _, report = cpu_run
    assert report["synthetic"] is True and report["backend"] == "stub" and report["sim_backend"] == "mock"
    assert "SYNTHETIC" in report["banner"] and "not exercised" in report["banner"]
    assert report["real_checks"]["real_producer_smoke"] == "not_run"
    assert report["real_checks"]["real_model_smoke"] == "not_run"
    assert list(report["variants"]) == ["baseline_1", "baseline_2", "baseline_3", "main"]
    assert report["variants"]["baseline_1"]["synthetic"] is False and report["variants"]["baseline_1"]["backends"] == ["rules"]
    assert all(report["variants"][v]["synthetic"] and report["variants"][v]["backends"] == ["stub"]
               for v in ("baseline_2", "baseline_3", "main"))


def test_smoke_ran_every_public_cli_once_in_order_and_kept_logs(cpu_run):
    _, out, report = cpu_run
    assert [s["name"] for s in report["steps"]] == [
        "generate_mock_pair", "parse_dataset", "enrich_divergence", "run_baseline",
        "run_llm_stub", "evaluate", "plot_telemetry"]
    assert all(s["returncode"] == 0 and (out / s["log"]).is_file() for s in report["steps"])


def test_smoke_asserts_the_known_event_all_view_exclusions_and_one_result_per_variant(cpu_run):
    _, _, report = cpu_run
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["input.000.grasp_event_time"]["passed"] and "5.10" in by_name["input.000.grasp_event_time"]["detail"]
    assert by_name["input.000.grasp_force_drop"]["passed"] and "12.00" in by_name["input.000.grasp_force_drop"]["detail"]
    assert by_name["bt.000.event_sample_retained"]["passed"] and by_name["bt.000.computed_divergence_at_event"]["passed"]
    views = [n for n in by_name if n.startswith("bt.000.view.")]
    assert len(views) == 9  # three views x (success samples, divergence, label leakage)
    for variant in ("baseline_1", "baseline_2", "baseline_3", "main"):
        assert by_name[f"metrics.{variant}.one_result_per_pair"]["passed"]
        assert by_name[f"predictions.{variant}.one_per_pair"]["passed"]
    assert by_name["metrics.all_values_finite"]["passed"] and by_name["baseline_1.evidence_verified_and_supported"]["passed"]
    assert by_name["rubric.excludes_stub_output"]["passed"]


def test_smoke_evaluated_only_its_own_explicit_files_with_resources_and_plots(cpu_run):
    _, out, _ = cpu_run
    metrics = json.loads((out / "eval" / "metrics.json").read_text())
    assert list(metrics["variants"]) == ["baseline_1", "baseline_2", "baseline_3", "main"]
    assert metrics["split"] == "test" and metrics["n_truth_pairs"] == 1
    assert metrics["variants"]["main"]["resources"]["input_tokens_total"] > 0
    assert metrics["variants"]["baseline_1"]["resources"]["run_runtime_s"] >= 0
    assert (out / "plots" / "telemetry_000.png").stat().st_size > 1000
    assert (out / "eval" / "plots" / "variant_comparison.png").stat().st_size > 1000
    # stub output never reaches the human rubric
    assert {k["backend"] for k in json.loads((out / "eval" / "rubric_key.json").read_text())} == {"rules"}


def test_reusing_an_output_directory_is_refused(cpu_run, capsys):
    _, out, _ = cpu_run
    before = (out / "smoke_report.json").read_text()
    assert smoke.main(["--output-dir", str(out)]) == 2
    assert "not empty" in capsys.readouterr().err
    assert (out / "smoke_report.json").read_text() == before


def test_a_failed_contract_check_gives_exit_code_1_and_a_failing_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(smoke, "GRASP_EVENT_S", 4.0)  # the fixture's event is at 5.10 s, so these checks must fail
    out = tmp_path / "sabotaged"
    assert smoke.main(["--output-dir", str(out)]) == 1
    report = json.loads((out / "smoke_report.json").read_text())
    failed = {c["name"] for c in report["checks"] if not c["passed"]}
    assert report["passed"] is False and "input.000.grasp_event_time" in failed
    text = capsys.readouterr().out
    assert "FAILED input.000.grasp_event_time" in text and "-> FAILED (synthetic)" in text


# --------------------------------------------------------------------------- #
# Builders for the real-smoke tests
# --------------------------------------------------------------------------- #
# The real manifests carry no failed stage, so the stage labels below are SYNTHETIC. Pair IDs, splits,
# failure modes and initial conditions are the real manifests'. The stage sets deliberately give some
# modes two stages and others one, so "eligible stages" is exercised and is not just "one per mode".
SYNTHETIC_STAGES = {
    "pick_up_object": {"position_error": ("approach", "align"), "orientation_error": ("approach", "align"),
                       "inadequate_grip_strength": ("grasp",), "early_gripper_release": ("lift",)},
    "transport_object": {"position_error": ("maintain_grasp", "move_along_path"), "orientation_error": ("move_along_path",),
                         "inadequate_grip_strength": ("maintain_grasp",), "early_gripper_release": ("reach_target",)},
}
MAGNITUDES = {"position_error": 0.05, "orientation_error": 0.3, "inadequate_grip_strength": 60.0, "early_gripper_release": 0.5}
TRANSPORT_STAGE_OF = {"approach": "maintain_grasp", "align": "move_along_path", "grasp": "reach_target", "lift": "reach_target"}


def real_manifests():
    return {task: json.loads(handoff_path(task).read_text()) for task in TASKS}


def synthetic_stage(task, mode, pair_id):
    stages = SYNTHETIC_STAGES[task][mode]
    return stages[int(pair_id) % len(stages)]


def build_released_batch(root, *, tasks=TASKS):
    """A released-layout batch (metadata.csv + diagnostics.jsonl) over the REAL pair IDs. Returns ``root``."""
    for task, manifest in real_manifests().items():
        if task not in tasks:
            continue
        rows, lines = [], []
        for pair_id, pair in manifest["pairs"].items():
            mode, stage = pair["failure_mode"], synthetic_stage(task, pair["failure_mode"], pair_id)
            rows.append(MetadataRow(pair_id=pair_id, task=Task(task), success_file=f"success/{pair_id}.csv", failure_file=f"failure/{pair_id}.csv",
                                    failed_subtask=Subtask(stage), failure_mode=FailureMode(mode), perturbation_param="param",
                                    perturbation_value=MAGNITUDES[mode], seed=723))
            lines.append(json.dumps({"pair_id": pair_id, "initial_condition_id": int(pair["initial_condition_id"]),
                                     "failure_config": {"mode": mode, "subtask": stage, "magnitude": MAGNITUDES[mode], "delay_s": 0.0, "axis": 0}}))
        (root / task).mkdir(parents=True, exist_ok=True)
        write_metadata_csv(root / task / "metadata.csv", rows)
        (root / task / "diagnostics.jsonl").write_text("\n".join(lines) + "\n")
    return root


def add_released_csvs(root, task):
    """Released CSVs for exactly the pairs the smoke will select (the model-only path copies them)."""
    manifest = smoke.load_task_manifest(handoff_path(task), task)
    selected, _ = smoke.select_stage_pairs(smoke.load_released_pairs(root, task, manifest))
    for pair in selected:
        for rel, failed in ((pair["metadata"].success_file, False), (pair["metadata"].failure_file, True)):
            target = root / task / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SAMPLE / ("failure_000.csv" if failed else "success_000.csv"), target)
            if task == "transport_object":
                relabel_transport_csv(target)


def transport_rows(rows):
    return [r.model_copy(update={"subtask": Subtask(TRANSPORT_STAGE_OF[r.subtask.value])}) for r in rows]


def relabel_transport_csv(path):
    from telemetry_reasoning.io import write_telemetry_csv

    write_telemetry_csv(path, transport_rows(read_telemetry_csv(path)))


class FakeSim:
    """Stands in for ``sim.runner.run_episode``: deterministic, built from the sample CSVs, no MuJoCo.

    Switches (keyword arguments) sabotage one behaviour at a time so each acceptance check can be shown to fail.
    """

    def __init__(self, **flags):
        self.flags, self.calls, self._injected = flags, [], collections.Counter()
        success, failure = read_telemetry_csv(SAMPLE / "success_000.csv"), read_telemetry_csv(SAMPLE / "failure_000.csv")
        self.rows = {"pick_up_object": (success, failure), "transport_object": (transport_rows(success), transport_rows(failure))}

    def __call__(self, *, task, seed, initial_condition_id, spread=0.015, failure=None, output=None):
        self.calls.append({"task": task, "seed": seed, "condition": initial_condition_id, "spread": spread, "failure": failure})
        success, failed = self.rows[task]
        if failure is None:
            return SimpleNamespace(rows=list(success), task_succeeded=not self.flags.get("nominal_fails"),
                                   diagnostics={"complete_subtasks": True, "failure": {"activated": False}})
        key = (task, seed, initial_condition_id, failure.mode.value, failure.subtask.value)
        self._injected[key] += 1
        rows = list(failed)
        if self.flags.get("nondeterministic") and self._injected[key] == 2:  # the replay differs from the first run
            rows[10] = rows[10].model_copy(update={"gripper_force": rows[10].gripper_force + 1e-3})
        speed = failure.mode == FailureMode.EXCESSIVE_SPEED
        return SimpleNamespace(
            rows=rows, task_succeeded=bool(self.flags.get("injected_succeeds")) or speed,  # excessive_speed need not fail
            diagnostics={"complete_subtasks": False, "observed_subtasks": list(dict.fromkeys(r.subtask.value for r in rows)),
                         "failure": {"activated": not self.flags.get("no_activation"), "first_activation_s": 5.1}})


def relabel_as_hf(directory):
    """Rewrite a STUB run directory so it reads as a real Hugging Face run. Test double only."""
    import json
    from pathlib import Path

    directory = Path(directory)
    for path in directory.glob("*.jsonl"):
        if path.name == "completions.jsonl":
            continue
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for record in records:
            record["backend"] = "hf"
        path.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(backend="hf", synthetic=False)
    manifest["execution"].update(selected_backend="hf", hf_weights_loaded=True, hf_generation_completed=True)
    for request in manifest["requests"]:
        request["token_count_method"] = "tokenizer"
    manifest_path.write_text(json.dumps(manifest))


FAKE_RUN_LLM = '''
import json, runpy, sys
from pathlib import Path

REAL, RELABEL, ADMISSIBLE = {real!r}, {relabel!r}, {admissible!r}
{relabel_source}
argv = sys.argv[1:]
def arg(flag):
    return argv[argv.index(flag) + 1] if flag in argv else None
if "--check-access" in argv:
    print(json.dumps({{"model_id": "test/model", "revision": "r", "configured_device": "cuda", "configured_dtype": "bfloat16",
        "packages": {{"torch": "2", "transformers": "4"}} if ADMISSIBLE else {{"torch": None, "transformers": None}},
        "resources": {{"host_admissible": ADMISSIBLE, "reason": None if ADMISSIBLE else "cuda requires at least 32 GiB physical host RAM"}}}}))
    raise SystemExit(0)
argv[argv.index("--backend") + 1] = "stub"  # the real run_llm.py, on the stub backend: no model exists in CI
sys.argv = [REAL] + argv
code = 0
try:
    runpy.run_path(REAL, run_name="__main__")
except SystemExit as exc:
    code = exc.code or 0
if RELABEL and code in (0, 1):
    relabel_as_hf(arg("--output-dir"))
raise SystemExit(code)
'''


def write_fake_run_llm(directory, *, relabel=True, admissible=True):
    path = Path(directory) / "fake_run_llm.py"
    path.write_text(FAKE_RUN_LLM.format(real=str(ROOT / "scripts" / "run_llm.py"), relabel=relabel, admissible=admissible,
                                        relabel_source=inspect.getsource(relabel_as_hf)))
    return path


# --------------------------------------------------------------------------- #
# Task manifest loading: damage to the real manifest is caught
# --------------------------------------------------------------------------- #
def _write_mutated_manifest(tmp_path, task, mutate):
    data = json.loads(handoff_path(task).read_text())
    mutate(data)
    path = tmp_path / f"{task}.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["pairs"]["000001"].update(split="validation"), "unknown split"),
    (lambda d: d["pairs"]["000001"].update(failure_mode="excessive_speed"), "non-dataset failure_mode"),
    (lambda d: d.update(dataset_kind="mock"), "dataset_kind is 'mock'"),
    (lambda d: d.update(metadata="elsewhere/metadata.csv"), "metadata is 'elsewhere"),
    (lambda d: d["pairs"].update({"000002": dict(d["pairs"]["000002"], pair_id="999999")}), "has pair_id='999999'"),
    (lambda d: d["pairs"]["000003"].pop("success_file"), r"lacks \['success_file'\]"),
    (lambda d: d["pairs"]["000004"].update(injection_timestamp_s=-1.0), "invalid injection_timestamp_s"),
    (lambda d: d["pairs"]["000004"].update(injection_timestamp=True), "invalid injection_timestamp"),
    (lambda d: d.update(pairs={}), "missing 'pairs'"),
])
def test_load_task_manifest_rejects_damage_to_the_real_manifest(tmp_path, mutate, message):
    path = _write_mutated_manifest(tmp_path, "pick_up_object", mutate)
    with pytest.raises(smoke.SmokeError, match=message):
        smoke.load_task_manifest(path, "pick_up_object")


def test_load_task_manifest_reports_every_problem_but_truncates_the_list(tmp_path):
    def break_many(data):
        for pair in data["pairs"].values():
            pair["split"] = "bogus"

    with pytest.raises(smoke.SmokeError, match=r"and 495 more"):
        smoke.load_task_manifest(_write_mutated_manifest(tmp_path, "pick_up_object", break_many), "pick_up_object")


def test_load_task_manifest_rejects_unreadable_and_non_object_files(tmp_path):
    with pytest.raises(smoke.SmokeError, match="cannot read task manifest"):
        smoke.load_task_manifest(tmp_path / "absent.json", "pick_up_object")
    (tmp_path / "list.json").write_text("[1, 2]")
    with pytest.raises(smoke.SmokeError, match="must be a JSON object"):
        smoke.load_task_manifest(tmp_path / "list.json", "pick_up_object")
    (tmp_path / "nan.json").write_text('{"pairs": NaN}')
    with pytest.raises(smoke.SmokeError, match="cannot read"):
        smoke.load_task_manifest(tmp_path / "nan.json", "pick_up_object")


# --------------------------------------------------------------------------- #
# Released dataset: stages come from metadata.csv + diagnostics.jsonl
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def released_batch(tmp_path_factory):
    root = build_released_batch(tmp_path_factory.mktemp("released") / "batch")
    for task in TASKS:
        add_released_csvs(root, task)
    return root


def _released(batch_root, task):
    return smoke.load_released_pairs(batch_root, task, smoke.load_task_manifest(handoff_path(task), task))


@pytest.mark.parametrize("task", TASKS)
def test_released_pairs_join_all_three_sources_for_every_real_pair(released_batch, task):
    pairs = _released(released_batch, task)
    manifest = json.loads(handoff_path(task).read_text())["pairs"]
    assert [p["pair_id"] for p in pairs] == sorted(manifest) and len(pairs) == 500
    for p in pairs:
        info = manifest[p["pair_id"]]
        assert (p["split"], p["mode"], p["condition"]) == (info["split"], info["failure_mode"], int(info["initial_condition_id"]))
        assert p["subtask"] == synthetic_stage(task, p["mode"], p["pair_id"]) and p["seed"] == 723
        assert p["config"]["mode"] == p["mode"] and p["config"]["subtask"] == p["subtask"]


@pytest.mark.parametrize("task", TASKS)
def test_eligible_stages_are_exactly_the_stages_the_released_data_used(released_batch, task):
    expected = {(task, mode, stage) for mode, stages in SYNTHETIC_STAGES[task].items() for stage in stages}
    assert set(smoke.eligible_stages(_released(released_batch, task))) == expected
    assert smoke.eligible_stages(_released(released_batch, task)) == sorted(expected)


def test_a_stage_the_released_data_never_used_is_not_eligible(tmp_path):
    root = tmp_path / "batch"
    build_released_batch(root, tasks=("pick_up_object",))
    # rewrite the data so no pair ever failed in "align"
    manifest = smoke.load_task_manifest(handoff_path("pick_up_object"), "pick_up_object")
    from telemetry_reasoning.io import read_metadata_csv

    rows = [r.model_copy(update={"failed_subtask": Subtask.APPROACH}) if r.failed_subtask == Subtask.ALIGN else r
            for r in read_metadata_csv(root / "pick_up_object" / "metadata.csv")]
    write_metadata_csv(root / "pick_up_object" / "metadata.csv", rows)
    lines = []
    for line in (root / "pick_up_object" / "diagnostics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["failure_config"]["subtask"] == "align":
            record["failure_config"]["subtask"] = "approach"
        lines.append(json.dumps(record))
    (root / "pick_up_object" / "diagnostics.jsonl").write_text("\n".join(lines) + "\n")
    stages = {(m, s) for _, m, s in smoke.eligible_stages(smoke.load_released_pairs(root, "pick_up_object", manifest))}
    assert ("position_error", "align") not in stages and ("orientation_error", "align") not in stages
    assert ("position_error", "approach") in stages


def _corrupt(tmp_path, released_batch, task, edit_metadata=None, edit_diag=None):
    root = tmp_path / "corrupt"
    shutil.copytree(released_batch / task, root / task)
    if edit_metadata:
        from telemetry_reasoning.io import read_metadata_csv

        rows = read_metadata_csv(root / task / "metadata.csv")
        write_metadata_csv(root / task / "metadata.csv", [edit_metadata(r) for r in rows])
    if edit_diag:
        lines = [edit_diag(json.loads(line)) for line in (root / task / "diagnostics.jsonl").read_text().splitlines()]
        (root / task / "diagnostics.jsonl").write_text("\n".join(json.dumps(r) for r in lines if r is not None) + "\n")
    return root


def _set(record, **config):
    if record["pair_id"] == "000001":
        record["failure_config"].update(config)
    return record


@pytest.mark.parametrize("task, edit, message", [
    ("pick_up_object", {"edit_diag": lambda r: _set(r, mode="position_error" if r["failure_config"]["mode"] != "position_error" else "orientation_error")},
     "failure mode differs"),
    ("pick_up_object", {"edit_diag": lambda r: _set(r, subtask="lift" if r["failure_config"]["subtask"] != "lift" else "grasp")}, "failed stage differs"),
    ("pick_up_object", {"edit_diag": lambda r: None if r["pair_id"] == "000001" else r}, r"diagnostics.jsonl pair IDs differ from the manifest \(missing 1"),
    ("pick_up_object", {"edit_diag": lambda r: dict(r, initial_condition_id=9999) if r["pair_id"] == "000001" else r}, "initial condition differs"),
    ("transport_object", {"edit_diag": lambda r: _set(r, subtask="grasp"),
                          "edit_metadata": lambda m: m.model_copy(update={"failed_subtask": Subtask.GRASP}) if m.pair_id == "000001" else m},
     "failed_subtask must belong to task"),
])
def test_released_pairs_that_disagree_with_the_manifest_are_an_error_not_a_guess(tmp_path, released_batch, task, edit, message):
    root = _corrupt(tmp_path, released_batch, task, **edit)
    with pytest.raises(smoke.SmokeError, match=message):
        smoke.load_released_pairs(root, task, smoke.load_task_manifest(handoff_path(task), task))


def test_diagnostics_reader_rejects_duplicates_bad_json_and_missing_fields(tmp_path):
    good = json.dumps({"pair_id": "000001", "failure_config": {"mode": "position_error"}})
    for text, message in ((good + "\n" + good, "duplicate diagnostics"), ("{nope", "not valid JSON"),
                          ('{"pair_id": "000001"}', "expected an object with"), ('{"pair_id": 1, "failure_config": {}}', "expected an object with")):
        (tmp_path / "d.jsonl").write_text(text)
        with pytest.raises(smoke.SmokeError, match=message):
            smoke.read_diagnostics(tmp_path / "d.jsonl")
    (tmp_path / "ok.jsonl").write_text(good + "\n\n")
    assert list(smoke.read_diagnostics(tmp_path / "ok.jsonl")) == ["000001"]
    with pytest.raises(smoke.SmokeError, match="cannot read"):
        smoke.read_diagnostics(tmp_path / "absent.jsonl")


# --------------------------------------------------------------------------- #
# Selection: dev first, then train, never test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", TASKS)
def test_selection_takes_one_dev_pair_per_stage_and_is_deterministic(released_batch, task):
    pairs = _released(released_batch, task)
    selected, uncovered = smoke.select_stage_pairs(pairs)
    assert uncovered == [] and len(selected) == len(smoke.eligible_stages(pairs))
    assert {p["split"] for p in selected} <= {"dev"} or {p["split"] for p in selected} <= {"dev", "train"}
    assert [(p["task"], p["mode"], p["subtask"]) for p in selected] == smoke.eligible_stages(pairs)
    again, _ = smoke.select_stage_pairs(list(reversed(pairs)))
    assert [p["pair_id"] for p in again] == [p["pair_id"] for p in selected]  # independent of input order
    for stage_pair in selected:  # the lowest pair_id among that stage's dev pairs
        same = sorted(p["pair_id"] for p in pairs if (p["mode"], p["subtask"], p["split"]) == (stage_pair["mode"], stage_pair["subtask"], "dev"))
        assert stage_pair["pair_id"] == same[0]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("per_stage", [1, 3, 25, 10_000])
def test_selection_never_touches_the_held_out_test_split(released_batch, task, per_stage):
    pairs = _released(released_batch, task)
    selected, _ = smoke.select_stage_pairs(pairs, per_stage)
    assert "test" not in {p["split"] for p in selected}
    assert len(selected) <= sum(1 for p in pairs if p["split"] != "test")
    if per_stage == 10_000:  # everything outside test, and nothing else
        assert {p["pair_id"] for p in selected} == {p["pair_id"] for p in pairs if p["split"] in ("dev", "train")}


def test_selection_prefers_dev_falls_back_to_train_and_reports_test_only_stages():
    def pair(pid, split, subtask):
        return {"task": "pick_up_object", "pair_id": pid, "split": split, "mode": "position_error", "subtask": subtask}

    pairs = [pair("1", "train", "approach"), pair("2", "dev", "approach"), pair("3", "train", "align"), pair("4", "test", "lift")]
    selected, uncovered = smoke.select_stage_pairs(pairs)
    assert {p["pair_id"] for p in selected} == {"2", "3"}  # dev beats train; train is the fallback
    assert uncovered == [("pick_up_object", "position_error", "lift")]  # only a test pair exists: reported, never used
    with pytest.raises(smoke.SmokeError, match="at least 1"):
        smoke.select_stage_pairs(pairs, 0)


# --------------------------------------------------------------------------- #
# Episode checks: acceptance, replay agreement, schema validity
# --------------------------------------------------------------------------- #
def test_rows_agree_within_tolerance_and_says_exactly_what_differs():
    rows = read_telemetry_csv(SAMPLE / "success_000.csv")
    ok, detail = smoke.rows_agree(rows, list(rows))
    assert ok and "400 rows agree" in detail
    nudged = list(rows)
    nudged[7] = rows[7].model_copy(update={"gripper_force": rows[7].gripper_force + 1e-6})
    assert smoke.rows_agree(rows, nudged)[0] is False and "gripper_force differs" in smoke.rows_agree(rows, nudged)[1]
    assert "at row 7" in smoke.rows_agree(rows, nudged)[1]
    nudged[7] = rows[7].model_copy(update={"gripper_force": rows[7].gripper_force + 1e-9})
    assert smoke.rows_agree(rows, nudged)[0] is True  # inside atol 1e-8
    assert smoke.rows_agree(rows, nudged, atol=1e-12)[0] is False
    assert smoke.rows_agree(rows, rows[:-1]) == (False, "400 vs 399 rows")
    flipped = list(rows)
    flipped[3] = rows[3].model_copy(update={"subtask": Subtask.LIFT})
    assert smoke.rows_agree(rows, flipped) == (False, "subtask differs at row 3")
    nan = list(rows)
    nan[0] = rows[0].model_copy(update={"gripper_force": float("nan")})
    assert smoke.rows_agree(rows, nan)[0] is False


def _episode_trio(**flags):
    sim = FakeSim(**flags)
    failure = SimpleNamespace(mode=FailureMode.INADEQUATE_GRIP_STRENGTH, subtask=Subtask.GRASP)
    kwargs = {"task": "pick_up_object", "seed": 723, "initial_condition_id": 1}
    return sim(**kwargs), sim(**kwargs, failure=failure), sim(**kwargs, failure=failure)


def test_a_good_episode_pair_passes_every_check():
    results = smoke.check_episode_pair(*_episode_trio())
    assert [n for n, _, _ in results] == [
        "nominal_succeeded_complete", "injection_activated", "injected_run_failed", "failure_trace_observed",
        "nominal_trace_valid", "injected_trace_valid", "replay_rows_agree", "replay_diagnostics_agree"]
    assert all(ok for _, ok, _ in results), [r for r in results if not r[1]]


@pytest.mark.parametrize("flags, failing", [
    ({"nominal_fails": True}, {"nominal_succeeded_complete"}),
    ({"no_activation": True}, {"injection_activated"}),
    ({"injected_succeeds": True}, {"injected_run_failed"}),
    ({"nondeterministic": True}, {"replay_rows_agree"}),
])
def test_each_acceptance_check_fails_on_exactly_its_own_sabotage(flags, failing):
    results = smoke.check_episode_pair(*_episode_trio(**flags))
    assert {n for n, ok, _ in results if not ok} == failing


def test_an_empty_failure_trace_and_diverging_diagnostics_are_caught():
    nominal, injected, replay = _episode_trio()
    no_trace = SimpleNamespace(rows=injected.rows, task_succeeded=False, diagnostics=dict(injected.diagnostics, observed_subtasks=[]))
    assert {n for n, ok, _ in smoke.check_episode_pair(nominal, no_trace, no_trace) if not ok} == {"failure_trace_observed"}
    drifted = SimpleNamespace(rows=replay.rows, task_succeeded=False, diagnostics=dict(replay.diagnostics, failure={"activated": True, "first_activation_s": 5.2}))
    assert {n for n, ok, _ in smoke.check_episode_pair(nominal, injected, drifted) if not ok} == {"replay_diagnostics_agree"}
    broken = SimpleNamespace(rows=[], task_succeeded=False, diagnostics=injected.diagnostics)
    failed = {n for n, ok, _ in smoke.check_episode_pair(nominal, broken, broken) if not ok}
    assert "injected_trace_valid" in failed


def test_excessive_speed_probe_checks_activation_only_and_does_not_require_failure():
    sim = FakeSim()
    speed = SimpleNamespace(mode=FailureMode.EXCESSIVE_SPEED, subtask=Subtask.APPROACH)
    result = sim(task="pick_up_object", seed=723, initial_condition_id=1, failure=speed)
    assert result.task_succeeded is True  # the probe run succeeded at the task, and that is acceptable
    checks = dict((n, (ok, d)) for n, ok, d in smoke.check_excessive_speed_probe(result))
    assert checks["activated"][0] and "not required to fail" in checks["activated"][1] and checks["trace_valid"][0]
    inert = FakeSim(no_activation=True)(task="pick_up_object", seed=723, initial_condition_id=1, failure=speed)
    assert dict((n, ok) for n, ok, _ in smoke.check_excessive_speed_probe(inert))["activated"] is False


def test_excessive_speed_probes_use_valid_stages_and_a_magnitude_above_one():
    from telemetry_reasoning.schemas import TASK_SUBTASKS
    from telemetry_reasoning.sim.failures import FailureConfig, validate_config

    assert set(smoke.EXCESSIVE_SPEED_PROBES) == set(TASKS)
    for task, (subtask, magnitude) in smoke.EXCESSIVE_SPEED_PROBES.items():
        config = FailureConfig(mode=FailureMode.EXCESSIVE_SPEED, subtask=Subtask(subtask), magnitude=magnitude)
        validate_config(Task(task), config)  # raises if the stage is not the task's or the magnitude is <= 1
        assert Subtask(subtask) in TASK_SUBTASKS[Task(task)]


# --------------------------------------------------------------------------- #
# Response policy and real-run validation
# --------------------------------------------------------------------------- #
def test_policy_config_sets_json_or_fenced_and_changes_nothing_else(tmp_path):
    import yaml

    from telemetry_reasoning.models.llm import load_llm_config

    source = ROOT / "configs" / "llm.yaml"
    before = source.read_bytes()
    dest = tmp_path / "policy.yaml"
    llm = smoke.write_policy_config(source, dest)
    assert llm["response_policy"] == "json_or_fenced_v1" == smoke.RESPONSE_POLICY
    original, copy = yaml.safe_load(source.read_text())["llm"], yaml.safe_load(dest.read_text())["llm"]
    assert {k: v for k, v in copy.items() if k != "response_policy"} == {k: v for k, v in original.items() if k != "response_policy"}
    assert load_llm_config(source).response_policy == "strict_json_v1"  # the shared default: the override really changes it
    assert load_llm_config(dest).response_policy == "json_or_fenced_v1"  # the real loader accepts it
    assert source.read_bytes() == before  # the shared config is never modified


def test_policy_config_keeps_relative_paths_pointing_where_they_did(tmp_path):
    import yaml

    source = tmp_path / "conf" / "llm.yaml"
    source.parent.mkdir()
    source.write_text(yaml.safe_dump({"llm": {**yaml.safe_load((ROOT / "configs" / "llm.yaml").read_text())["llm"],
                                              "prompt_dir": "../prompts", "cache_dir": "cache"}}))
    (tmp_path / "prompts").mkdir()
    dest = tmp_path / "elsewhere" / "policy.yaml"
    dest.parent.mkdir()
    llm = smoke.write_policy_config(source, dest)
    assert llm["prompt_dir"] == str((tmp_path / "prompts").resolve()) and llm["cache_dir"] == str((source.parent / "cache").resolve())
    from telemetry_reasoning.models.llm import load_llm_config

    assert load_llm_config(dest).prompt_dir == (tmp_path / "prompts").resolve()  # still resolves to the original location


def test_policy_config_rejects_a_config_without_an_llm_mapping(tmp_path):
    (tmp_path / "bad.yaml").write_text("something: else\n")
    with pytest.raises(smoke.SmokeError, match="'llm' mapping"):
        smoke.write_policy_config(tmp_path / "bad.yaml", tmp_path / "out.yaml")


@pytest.fixture(scope="module")
def stub_llm_dir(cpu_run):
    """A genuine stub run directory (from the CPU contract smoke): 1 pair x 3 variants."""
    return cpu_run[1] / "runs" / "llm"


def _hf_copy(stub_llm_dir, tmp_path):
    copy = tmp_path / "llm"
    shutil.copytree(stub_llm_dir, copy)
    relabel_as_hf(copy)
    return copy


def test_a_stub_run_fails_exactly_the_real_run_checks_it_should(stub_llm_dir):
    failed = {n for n, ok, _ in smoke.check_real_llm_run(stub_llm_dir, ["000"]) if not ok}
    assert failed == {
        "manifest.backend_is_hf", "manifest.not_synthetic", "manifest.weights_loaded", "manifest.generation_completed",
        "manifest.real_token_counts", "manifest.response_policy",  # the stub run used the default strict policy
        "predictions.baseline_2.backend_is_hf", "predictions.baseline_3.backend_is_hf", "predictions.main.backend_is_hf"}


def test_a_run_that_looks_like_real_hf_under_json_or_fenced_passes_every_check(stub_llm_dir, tmp_path):
    copy = _hf_copy(stub_llm_dir, tmp_path)
    manifest = json.loads((copy / "run_manifest.json").read_text())
    manifest["response_policy"] = smoke.RESPONSE_POLICY
    (copy / "run_manifest.json").write_text(json.dumps(manifest))
    results = smoke.check_real_llm_run(copy, ["000"])
    assert len(results) == 8 + 1 + 3 * 4
    assert all(ok for _, ok, _ in results), [r for r in results if not r[1]]


@pytest.mark.parametrize("edit, failing", [
    (lambda m: m.update(response_policy="strict_json_v1"), {"manifest.response_policy"}),
    (lambda m: m["execution"].update(all_requests_validated=False), {"manifest.all_requests_validated"}),
    (lambda m: m["execution"].update(hf_weights_loaded=False), {"manifest.weights_loaded"}),
    (lambda m: m["execution"].update(generated_requests=2), {"manifest.request_counts"}),
    (lambda m: m.update(status_counts={"abstain": 2, "error": 1}), {"manifest.no_error_status"}),
    (lambda m: m.update(synthetic=True), {"manifest.not_synthetic"}),
    (lambda m: [r.update(token_count_method=smoke.TOKEN_BOUND_METHOD) for r in m["requests"]], {"manifest.real_token_counts"}),
    (lambda m: m.update(requests=[]), {"manifest.real_token_counts"}),
])
def test_each_real_run_manifest_check_fails_on_its_own_defect(stub_llm_dir, tmp_path, edit, failing):
    copy = _hf_copy(stub_llm_dir, tmp_path)
    manifest = json.loads((copy / "run_manifest.json").read_text())
    manifest["response_policy"] = smoke.RESPONSE_POLICY
    edit(manifest)
    (copy / "run_manifest.json").write_text(json.dumps(manifest))
    assert {n for n, ok, _ in smoke.check_real_llm_run(copy, ["000"]) if not ok} == failing


def test_prediction_level_real_run_checks_catch_stub_labels_errors_and_missing_pairs(stub_llm_dir, tmp_path):
    copy = _hf_copy(stub_llm_dir, tmp_path)
    manifest = json.loads((copy / "run_manifest.json").read_text())
    manifest["response_policy"] = smoke.RESPONSE_POLICY
    (copy / "run_manifest.json").write_text(json.dumps(manifest))
    records = [json.loads(line) for line in (copy / "main.jsonl").read_text().splitlines()]
    records[0].update(backend="stub", status="error")
    (copy / "main.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    failed = {n for n, ok, _ in smoke.check_real_llm_run(copy, ["000"]) if not ok}
    assert "predictions.main.backend_is_hf" in failed
    (copy / "baseline_2.jsonl").write_text("")
    failed = {n for n, ok, _ in smoke.check_real_llm_run(copy, ["000"]) if not ok}
    assert "predictions.baseline_2.one_per_pair" in failed
    # a pair the run was not asked about
    assert "manifest.request_counts" in {n for n, ok, _ in smoke.check_real_llm_run(copy, ["000", "001"]) if not ok}


# --------------------------------------------------------------------------- #
# Hardware preflight
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(importlib.util.find_spec("mujoco") is not None, reason="mujoco is installed here; this tests the not-installed message")
def test_mujoco_preflight_says_why_the_simulator_cannot_run_when_mujoco_is_missing():
    reasons = smoke.preflight_mujoco()
    assert len(reasons) == 1 and "cannot be imported" in reasons[0] and "mujoco" in reasons[0].lower()


def test_environment_report_never_imports_heavy_packages_and_reports_missing_ones_as_null():
    env = smoke._environment()
    assert set(env) == {"python", "platform", "mujoco", "torch", "transformers"} and env["python"].count(".") == 2
    assert all(v is None or isinstance(v, str) for v in env.values())


# --------------------------------------------------------------------------- #
# Orchestration: fake simulator + Hugging Face CLI double, real parser/divergence/baseline/evaluate/plots
# --------------------------------------------------------------------------- #
def run_real(tmp_path, batch, *, sim_backend="mujoco", backend="stub", tasks=TASKS, flags=None, llm=None,
             cpu_passed=True, name="out", per_stage=1, preflight_mujoco=None, use_fake_sim=True):
    """Run ``run_real_smoke`` with a fast stand-in for the CPU contract smoke (it has its own tests)."""
    out = Path(tmp_path) / name
    sim = FakeSim(**(flags or {})) if sim_backend == "mujoco" and use_fake_sim else None
    cpu_calls = []

    def fast_cpu(output_dir):
        cpu_calls.append(Path(output_dir))
        return {"passed": cpu_passed, "n_checks": 64, "n_failed": 0 if cpu_passed else 3}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(smoke, "run_smoke", fast_cpu)
        if llm is not None:
            mp.setitem(smoke.SCRIPT_OVERRIDES, "run_llm.py", llm)
        if preflight_mujoco is not None:
            mp.setattr(smoke, "preflight_mujoco", lambda: preflight_mujoco)
        report, code = smoke.run_real_smoke(out, dataset_dir=batch, task_manifest_dir=HANDOFF, tasks=tasks, sim_backend=sim_backend,
                                            backend=backend, per_stage=per_stage, episode_runner=sim)
    return SimpleNamespace(report=report, code=code, out=out, sim=sim, cpu_calls=cpu_calls, tasks=tasks)


def failed_names(run):
    return sorted({c["name"] for c in run.report["checks"] if not c["passed"]})


def step_names(run):
    return [s["name"] for s in run.report["steps"]]


@pytest.fixture(scope="module")
def producer_run(tmp_path_factory, released_batch):
    return run_real(tmp_path_factory.mktemp("producer"), released_batch, sim_backend="mujoco", backend="stub")


@pytest.fixture(scope="module")
def model_run(tmp_path_factory, released_batch):
    tmp = tmp_path_factory.mktemp("model")
    return run_real(tmp, released_batch, sim_backend="mock", backend="hf", llm=write_fake_run_llm(tmp))


@pytest.fixture(scope="module")
def combined_run(tmp_path_factory, released_batch):
    tmp = tmp_path_factory.mktemp("combined")
    return run_real(tmp, released_batch, sim_backend="mujoco", backend="hf", tasks=("pick_up_object",), llm=write_fake_run_llm(tmp))


# ---- producer smoke --------------------------------------------------------
def test_producer_smoke_passes_and_reports_only_itself_as_run(producer_run):
    assert producer_run.code == 0 and producer_run.report["passed"] is True, failed_names(producer_run)
    assert producer_run.report["real_checks"]["real_producer_smoke"] == "passed"
    assert producer_run.report["real_checks"]["real_model_smoke"] == "not_run"  # --backend stub: the model smoke never ran
    assert not [n for n in step_names(producer_run) if "run_llm" in n or "check_access" in n]
    assert not [c for c in producer_run.report["checks"] if c["name"].startswith("real.model.")]


def test_the_cpu_contract_smoke_runs_first_in_its_own_directory(producer_run):
    assert producer_run.cpu_calls == [producer_run.out / "cpu-contract"]
    cpu = producer_run.report["cpu_contract"]
    assert cpu["synthetic"] is True and cpu["passed"] is True and cpu["report"] == "cpu-contract/smoke_report.json"
    assert producer_run.report["checks"][0]["name"] == "real.cpu_contract_smoke_passed"


@pytest.mark.parametrize("task", TASKS)
def test_producer_smoke_replays_exactly_the_eligible_stages_with_the_released_pairs_own_inputs(producer_run, released_batch, task):
    info = producer_run.report["real"][task]
    pairs = {p["pair_id"]: p for p in _released(released_batch, task)}
    stages = smoke.eligible_stages(list(pairs.values()))
    assert [(task, s["mode"], s["subtask"]) for s in info["eligible_stages"]] == stages
    assert info["released_pairs"] == 500 and len(info["selected_pairs"]) == len(stages)
    assert all(pairs[pid]["split"] in ("dev", "train") for pid in info["selected_pairs"])
    injected = {(c["condition"], c["failure"].mode.value, c["failure"].subtask.value): c
                for c in producer_run.sim.calls if c["task"] == task and c["failure"] is not None and c["failure"].mode != FailureMode.EXCESSIVE_SPEED}
    for pid in info["selected_pairs"]:
        pair = pairs[pid]
        call = injected[(pair["condition"], pair["mode"], pair["subtask"])]
        config = pair["config"]
        assert (call["seed"], call["spread"]) == (723, smoke.SIM_SPREAD)
        assert (call["failure"].magnitude, call["failure"].delay_s, call["failure"].axis) == (config["magnitude"], config["delay_s"], config["axis"])


def test_producer_smoke_covers_all_four_modes_per_task_and_never_selects_test(producer_run):
    for task in TASKS:
        names = {c["name"]: c for c in producer_run.report["checks"] if c["name"].startswith(f"real.dataset.{task}.")}
        assert names[f"real.dataset.{task}.all_four_modes_covered"]["passed"]
        assert names[f"real.dataset.{task}.every_eligible_stage_covered"]["passed"]
        assert names[f"real.dataset.{task}.test_split_untouched"]["passed"]
    assert producer_run.report["split_policy"] == {"drawn_from": ["dev", "train"], "test_split_touched": False}
    used = {c["seed"] for c in producer_run.sim.calls}
    assert used == {723}


@pytest.mark.parametrize("task", TASKS)
def test_each_replayed_pair_runs_the_full_check_set_and_excessive_speed_only_activates(producer_run, task):
    per_pair = collections.defaultdict(set)
    for c in producer_run.report["checks"]:
        parts = c["name"].split(".")
        if parts[:3] == ["real", "producer", task] and len(parts) == 5 and parts[3].isdigit():
            per_pair[parts[3]].add(parts[4])
    assert len(per_pair) == len(producer_run.report["real"][task]["selected_pairs"])
    assert all(names == {"nominal_succeeded_complete", "injection_activated", "injected_run_failed", "failure_trace_observed",
                         "nominal_trace_valid", "injected_trace_valid", "replay_rows_agree", "replay_diagnostics_agree"} for names in per_pair.values())
    speed = [c for c in producer_run.report["checks"] if c["name"].startswith(f"real.producer.{task}.excessive_speed.")]
    assert {c["name"].rsplit(".", 1)[1] for c in speed} == {"activated", "trace_valid"} and all(c["passed"] for c in speed)
    subtask, magnitude = smoke.EXCESSIVE_SPEED_PROBES[task]
    probe = [c for c in producer_run.sim.calls if c["task"] == task and c["failure"] is not None and c["failure"].mode == FailureMode.EXCESSIVE_SPEED]
    assert len(probe) == 1 and (probe[0]["failure"].subtask.value, probe[0]["failure"].magnitude) == (subtask, magnitude)


def test_accepted_pairs_go_through_parse_divergence_baseline_evaluate_and_plots_per_task(producer_run):
    for task in TASKS:
        work = producer_run.out / "real" / "producer" / task
        assert {f"{task}_parse", f"{task}_enrich", f"{task}_baseline", f"{task}_evaluate_1v", f"{task}_plot_telemetry"} <= set(step_names(producer_run))
        n = len(producer_run.report["real"][task]["selected_pairs"])
        assert len(list((work / "bt").glob("pair_*_bt.json"))) == n and len(list((work / "plots").glob("telemetry_*.png"))) == n
        metrics = json.loads((work / "eval" / "metrics.json").read_text())
        assert list(metrics["variants"]) == ["baseline_1"] and metrics["variants"]["baseline_1"]["synthetic"] is False
        assert metrics["variants"]["baseline_1"]["evidence"]["n_unsupported"] == 0
    # regenerated pairs keep their released IDs, so the real task manifest applies to them
    ids = {p.name[len("pair_"):-len("_bt.json")] for p in (producer_run.out / "real" / "producer" / "pick_up_object" / "bt").glob("pair_*_bt.json")}
    assert ids == set(producer_run.report["real"]["pick_up_object"]["selected_pairs"])


@pytest.mark.parametrize("task", TASKS)
def test_onset_is_reported_unscored_because_the_real_manifests_have_no_evaluator_clock(producer_run, task):
    info = producer_run.report["real"][task]
    assert info["evaluator_onset_clock_in_manifest"] is False and "absolute-simulator-time" in info["onset_note"]
    metrics = json.loads((producer_run.out / "real" / "producer" / task / "eval" / "metrics.json").read_text())
    assert (metrics["variants"]["baseline_1"]["onset"].get("n_with_reference") or 0) == 0
    check = next(c for c in producer_run.report["checks"] if c["name"].endswith("onset_reference_matches_manifest_clock") and f".{task}." in c["name"])
    assert check["passed"] and "evaluator-clock injection_timestamp present: False" in check["detail"]


def test_report_is_strict_json_with_environment_and_banner(producer_run):
    text = (producer_run.out / "smoke_report.json").read_text()
    assert json.loads(text, parse_constant=lambda t: pytest.fail(f"non-finite {t}")) == producer_run.report
    assert producer_run.report["smoke"] == "real" and producer_run.report["response_policy"] == "json_or_fenced_v1"
    assert "REAL-HARDWARE" in producer_run.report["banner"] and "never reported as passed" in producer_run.report["banner"]
    assert set(producer_run.report["environment"]) >= {"python", "mujoco", "torch", "transformers"}


# ---- model smoke -----------------------------------------------------------
def test_model_smoke_passes_on_released_pairs_without_running_the_simulator(model_run):
    assert model_run.code == 0 and model_run.report["passed"] is True, failed_names(model_run)
    assert model_run.report["real_checks"] == {**model_run.report["real_checks"], "real_model_smoke": "passed", "real_producer_smoke": "not_run"}
    assert model_run.sim is None and not [c for c in model_run.report["checks"] if c["name"].startswith("real.producer.")]


def test_model_smoke_preflights_before_doing_any_work_and_records_the_capabilities(model_run):
    steps = step_names(model_run)
    assert steps[0] == "hf_check_access" and steps.index("hf_check_access") < min(i for i, n in enumerate(steps) if n.endswith("_parse"))
    caps = model_run.report["environment"]["hf_check_access"]
    assert caps["resources"]["host_admissible"] is True and caps["model_id"] == "test/model"


@pytest.mark.parametrize("task", TASKS)
def test_model_smoke_applies_json_or_fenced_and_checks_it_in_the_run_manifest(model_run, task):
    import yaml

    override = yaml.safe_load((model_run.out / "real" / "llm_policy.yaml").read_text())["llm"]
    assert override["response_policy"] == "json_or_fenced_v1"
    manifest = json.loads((model_run.out / "real" / "model" / task / "llm" / "run_manifest.json").read_text())
    assert manifest["response_policy"] == "json_or_fenced_v1" and manifest["backend"] == "hf" and manifest["config_path"].endswith("llm_policy.yaml")
    check = next(c for c in model_run.report["checks"] if c["name"] == f"real.model.{task}.manifest.response_policy")
    assert check["passed"] and "json_or_fenced_v1" in check["detail"]


@pytest.mark.parametrize("task", TASKS)
def test_model_smoke_scores_all_four_variants_on_the_same_pairs_as_real_not_synthetic(model_run, task):
    work = model_run.out / "real" / "model" / task
    metrics = json.loads((work / "eval" / "metrics.json").read_text())
    n = len(model_run.report["real"][task]["selected_pairs"])
    assert list(metrics["variants"]) == ["baseline_1", "baseline_2", "baseline_3", "main"]
    for variant, entry in metrics["variants"].items():
        assert entry["synthetic"] is False and entry["slices"]["all"]["n"] == n and entry["resources"]["reported"] is True
        assert entry["backends"] == (["rules"] if variant == "baseline_1" else ["hf"])
    for variant in ("baseline_2", "baseline_3", "main"):
        assert metrics["variants"][variant]["resources"]["input_tokens_total"] > 0
        assert metrics["variants"][variant]["resources"]["token_count_methods"] == ["tokenizer"]
    assert len(list((work / "plots").glob("telemetry_*.png"))) == n


def test_model_smoke_uses_released_dev_or_train_pairs_copied_into_a_per_task_directory(model_run, released_batch):
    for task in TASKS:
        data = model_run.out / "real" / "model" / task / "data"
        selected = model_run.report["real"][task]["selected_pairs"]
        from telemetry_reasoning.io import read_metadata_csv

        assert sorted(r.pair_id for r in read_metadata_csv(data / "metadata.csv")) == sorted(selected)
        for pid in selected:
            assert (data / "success" / f"{pid}.csv").is_file() and (data / "failure" / f"{pid}.csv").is_file()
    assert (model_run.out / "real" / "model" / "pick_up_object").is_dir() and (model_run.out / "real" / "model" / "transport_object").is_dir()


# ---- both together ---------------------------------------------------------
def test_with_both_flags_the_model_runs_on_the_producers_regenerated_pairs(combined_run):
    assert combined_run.code == 0, failed_names(combined_run)
    assert combined_run.report["real_checks"]["real_producer_smoke"] == combined_run.report["real_checks"]["real_model_smoke"] == "passed"
    assert not (combined_run.out / "real" / "model" / "pick_up_object" / "data").exists()  # no second copy of released pairs
    manifest = json.loads((combined_run.out / "real" / "model" / "pick_up_object" / "llm" / "run_manifest.json").read_text())
    assert Path(manifest["bt_dir"]).resolve() == (combined_run.out / "real" / "producer" / "pick_up_object" / "bt").resolve()
    steps = step_names(combined_run)
    assert steps.count("pick_up_object_parse") == 1 and steps.index("pick_up_object_evaluate_1v") < steps.index("pick_up_object_run_llm_hf")
    assert steps.index("hf_check_access") == 0  # preflight precedes any episode


# ---- negative paths: the smoke must never over-claim ------------------------
ONE = ("pick_up_object",)


def test_a_stub_labelled_run_can_never_pass_the_real_model_smoke(tmp_path, released_batch):
    run = run_real(tmp_path, released_batch, sim_backend="mock", backend="hf", tasks=ONE, llm=write_fake_run_llm(tmp_path, relabel=False))
    assert run.code == 1 and run.report["passed"] is False and run.report["real_checks"]["real_model_smoke"] == "failed"
    failed = set(failed_names(run))
    assert {f"real.model.pick_up_object.{n}" for n in (
        "manifest.backend_is_hf", "manifest.not_synthetic", "manifest.weights_loaded", "manifest.generation_completed",
        "manifest.real_token_counts", "predictions.main.backend_is_hf", "metrics.main.not_synthetic")} <= failed
    assert "real.model.pick_up_object.manifest.response_policy" not in failed  # the policy override still flowed through run_llm


@pytest.mark.parametrize("backend", ["stub", "hf"])
def test_blocked_mujoco_blocks_every_requested_stage_before_any_work(tmp_path, released_batch, backend):
    run = run_real(tmp_path, released_batch, sim_backend="mujoco", backend=backend, tasks=ONE, use_fake_sim=False,
                   llm=write_fake_run_llm(tmp_path), preflight_mujoco=["the MuJoCo runner cannot be imported (OSError: blocked by policy)"])
    assert run.code == 2 and run.report["passed"] is False
    requested = {"real_producer_smoke": "blocked", "real_model_smoke": "blocked" if backend == "hf" else "not_run"}
    assert {k: run.report["real_checks"][k] for k in requested} == requested
    assert [e["stage"] for e in run.report["errors"]] == ["preflight_mujoco"] and "blocked by policy" in run.report["errors"][0]["error"]
    assert run.report["steps"] == []  # nothing ran, not even the HF capability probe or the parser


def test_an_inadmissible_hf_host_blocks_both_stages_before_a_single_episode_runs(tmp_path, released_batch):
    run = run_real(tmp_path, released_batch, sim_backend="mujoco", backend="hf", tasks=ONE, llm=write_fake_run_llm(tmp_path, admissible=False))
    assert run.code == 2 and run.report["real_checks"]["real_producer_smoke"] == run.report["real_checks"]["real_model_smoke"] == "blocked"
    error = run.report["errors"][0]
    assert error["stage"] == "preflight_hf" and "torch is not installed" in error["error"] and "32 GiB" in error["error"]
    assert run.sim.calls == [] and step_names(run) == ["hf_check_access"]  # fail fast: the simulator was never called


def test_blocked_real_stages_are_never_reported_as_passed(tmp_path, released_batch):
    run = run_real(tmp_path, released_batch, sim_backend="mock", backend="hf", tasks=ONE, llm=write_fake_run_llm(tmp_path, admissible=False))
    assert run.code == 2 and run.report["real_checks"]["real_model_smoke"] == "blocked" and run.report["real_checks"]["real_producer_smoke"] == "not_run"
    assert run.report["passed"] is False
    assert json.loads((run.out / "smoke_report.json").read_text())["real_checks"]["real_model_smoke"] == "blocked"


def test_a_released_dataset_that_disagrees_with_the_manifest_blocks_the_smoke(tmp_path):
    root = build_released_batch(tmp_path / "batch", tasks=ONE)
    lines = [json.loads(line) for line in (root / ONE[0] / "diagnostics.jsonl").read_text().splitlines()]
    lines[3]["failure_config"]["mode"] = "excessive_speed"
    (root / ONE[0] / "diagnostics.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    run = run_real(tmp_path, root, sim_backend="mujoco", backend="stub", tasks=ONE)
    assert run.code == 2 and run.report["real_checks"]["real_producer_smoke"] == "blocked"
    assert run.report["errors"][0]["stage"] == "dataset" and "disagrees with the task manifest" in run.report["errors"][0]["error"]
    assert run.sim.calls == []


def test_a_missing_released_dataset_blocks_the_smoke_with_a_pointed_message(tmp_path):
    handoff_path(ONE[0])  # this test needs the real manifest to exist; only the batch directory is missing
    run = run_real(tmp_path, tmp_path / "no-such-batch", sim_backend="mujoco", backend="stub", tasks=ONE)
    assert run.code == 2 and "cannot read the released metadata" in run.report["errors"][0]["error"] and run.sim.calls == []


def test_a_failing_cpu_contract_smoke_stops_before_any_real_stage(tmp_path, released_batch):
    run = run_real(tmp_path, released_batch, sim_backend="mujoco", backend="hf", tasks=ONE, cpu_passed=False, llm=write_fake_run_llm(tmp_path))
    assert run.code == 1 and run.report["passed"] is False
    assert run.report["real_checks"]["real_producer_smoke"] == run.report["real_checks"]["real_model_smoke"] == "not_run"
    assert failed_names(run) == ["real.cpu_contract_smoke_passed"] and run.sim.calls == [] and run.report["steps"] == []
    assert run.report["real"] == {}  # the dataset was not even read


@pytest.mark.parametrize("flags, failing", [
    ({"injected_succeeds": True}, "injected_run_failed"),
    ({"no_activation": True}, "injection_activated"),
    ({"nominal_fails": True}, "nominal_succeeded_complete"),
    ({"nondeterministic": True}, "replay_rows_agree"),
])
def test_a_producer_defect_fails_the_producer_smoke_with_the_named_check(tmp_path, released_batch, flags, failing):
    run = run_real(tmp_path, released_batch, sim_backend="mujoco", backend="stub", tasks=ONE, flags=flags)
    assert run.code == 1 and run.report["passed"] is False and run.report["real_checks"]["real_producer_smoke"] == "failed"
    failed = failed_names(run)
    assert any(n.endswith(f".{failing}") for n in failed) and "real.producer.pick_up_object.all_selected_pairs_accepted" in failed
    assert not [n for n in step_names(run) if n.endswith("_parse")]  # rejected pairs never reach the parser


def test_when_the_producer_accepts_nothing_the_model_smoke_cannot_run_and_says_so(tmp_path, released_batch):
    run = run_real(tmp_path, released_batch, sim_backend="mujoco", backend="hf", tasks=ONE, flags={"injected_succeeds": True},
                   llm=write_fake_run_llm(tmp_path))
    assert run.code == 2 and run.report["real_checks"]["real_producer_smoke"] == "failed" and run.report["real_checks"]["real_model_smoke"] == "failed"
    assert "no accepted pairs to run the model on" in run.report["errors"][0]["error"]
    assert not [n for n in step_names(run) if "run_llm_hf" in n]


# ---- command line ----------------------------------------------------------
def _capture_real(monkeypatch, code=0):
    seen = {}

    def fake(output_dir, **kwargs):
        seen.update(kwargs, output_dir=output_dir)
        return ({"steps": [], "errors": [], "checks": [], "cpu_contract": {"n_checks": 1, "n_failed": 0},
                 "real_checks": {"real_producer_smoke": "passed", "real_model_smoke": "passed"}}, code)

    monkeypatch.setattr(smoke, "run_real_smoke", fake)
    monkeypatch.setattr(smoke, "run_smoke", lambda *a, **k: pytest.fail("the CPU-only path must not run for real flags"))
    return seen


def test_cli_planned_command_dispatches_to_the_real_smokes_with_defaults(monkeypatch, tmp_path):
    seen = _capture_real(monkeypatch)
    batch = tmp_path / "batch"
    assert smoke.main(["--sim-backend", "mujoco", "--backend", "hf", "--tasks", "all", "--cover-dataset-failure-modes",
                       "--dataset-dir", str(batch), "--output-dir", str(tmp_path / "o")]) == 0
    assert seen["tasks"] == smoke.DATASET_TASKS and seen["sim_backend"] == "mujoco" and seen["backend"] == "hf"
    assert seen["task_manifest_dir"] == batch / "handoff" and seen["per_stage"] == 1 and seen["llm_config"] == ROOT / "configs" / "llm.yaml"
    assert seen["dataset_dir"] == batch and seen["output_dir"] == tmp_path / "o"


def test_cli_passes_overrides_through_and_each_real_flag_works_alone(monkeypatch, tmp_path):
    seen = _capture_real(monkeypatch)
    base = ["--dataset-dir", str(tmp_path / "b"), "--output-dir", str(tmp_path / "o")]
    assert smoke.main(["--backend", "hf", "--tasks", "transport_object", "--per-stage", "2", "--task-manifest-dir", str(HANDOFF),
                       "--llm-config", str(tmp_path / "c.yaml"), *base]) == 0
    assert (seen["sim_backend"], seen["backend"], seen["tasks"], seen["per_stage"]) == ("mock", "hf", ("transport_object",), 2)
    assert seen["task_manifest_dir"] == HANDOFF and seen["llm_config"] == tmp_path / "c.yaml"
    assert smoke.main(["--sim-backend", "mujoco", *base]) == 0 and (seen["sim_backend"], seen["backend"]) == ("mujoco", "stub")


@pytest.mark.parametrize("code", [1, 2])
def test_cli_returns_the_real_smoke_exit_code(monkeypatch, tmp_path, code):
    _capture_real(monkeypatch, code=code)
    assert smoke.main(["--backend", "hf", "--dataset-dir", str(tmp_path), "--output-dir", str(tmp_path / "o")]) == code


@pytest.mark.parametrize("flags", [["--backend", "hf"], ["--sim-backend", "mujoco"], ["--backend", "hf", "--sim-backend", "mujoco"]])
def test_real_flags_without_a_dataset_dir_are_a_setup_error_and_create_nothing(tmp_path, capsys, flags):
    out = tmp_path / "real"
    assert smoke.main(["--output-dir", str(out), *flags]) == 2
    assert "need --dataset-dir" in capsys.readouterr().err and not out.exists()


def test_the_real_smokes_refuse_a_non_empty_output_directory(tmp_path, capsys):
    out = tmp_path / "used"
    out.mkdir()
    (out / "earlier-run.txt").write_text("keep me")
    assert smoke.main(["--backend", "hf", "--dataset-dir", str(tmp_path), "--output-dir", str(out)]) == 2
    assert "not empty" in capsys.readouterr().err and (out / "earlier-run.txt").read_text() == "keep me"


def test_the_cpu_only_report_still_says_the_real_checks_were_not_run(cpu_run):
    assert cpu_run[2]["real_checks"]["real_producer_smoke"] == cpu_run[2]["real_checks"]["real_model_smoke"] == "not_run"


# --------------------------------------------------------------------------- #
# Real hardware: opt-in, and skipped with the missing resource named (never a stub-backed pass)
# --------------------------------------------------------------------------- #
def _released_dataset_dir():
    for candidate in (os.environ.get("TRACK2_DATASET_DIR"), BATCH):
        if candidate and all((Path(candidate) / task / "diagnostics.jsonl").is_file() for task in TASKS):
            return Path(candidate)
    return None


def _real_skip_reason(flag, packages):
    if os.environ.get(flag) != "1":
        return f"opt-in: set {flag}=1 to run this check on real hardware"
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        return f"not installed: {', '.join(missing)}"
    if _released_dataset_dir() is None:
        return "released dataset batch not found (set TRACK2_DATASET_DIR to the directory holding pick_up_object/ and transport_object/)"
    return None


_SIM_SKIP, _HF_SKIP = _real_skip_reason("TRACK2_REAL_SIM", ["mujoco"]), _real_skip_reason("TRACK2_REAL_HF", ["torch", "transformers"])


def _run_real_cli(tmp_path, *flags):
    out = tmp_path / "real"
    code = smoke.main([*flags, "--dataset-dir", str(_released_dataset_dir()), "--task-manifest-dir", str(HANDOFF), "--output-dir", str(out)])
    return code, json.loads((out / "smoke_report.json").read_text())


@pytest.mark.skipif(_SIM_SKIP is not None, reason=_SIM_SKIP or "")
def test_real_mujoco_producer_smoke_on_the_released_dataset(tmp_path):
    code, report = _run_real_cli(tmp_path, "--sim-backend", "mujoco", "--tasks", "all", "--cover-dataset-failure-modes")
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert code == 0 and not failed and report["real_checks"]["real_producer_smoke"] == "passed", (report["errors"], failed)


@pytest.mark.skipif(_HF_SKIP is not None, reason=_HF_SKIP or "")
def test_real_hf_model_smoke_on_the_released_dataset(tmp_path):
    code, report = _run_real_cli(tmp_path, "--backend", "hf", "--tasks", "all")
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert code == 0 and not failed and report["real_checks"]["real_model_smoke"] == "passed", (report["errors"], failed)
    assert report["response_policy"] == "json_or_fenced_v1"
