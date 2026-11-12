"""Compression tuning is deterministic and excludes held-out groups/telemetry."""

import importlib.util
import json
from pathlib import Path

import pytest

from telemetry_reasoning.config import CompressionConfig
from telemetry_reasoning.io import write_metadata_csv, write_telemetry_csv
from telemetry_reasoning.parser import tuning
from telemetry_reasoning.parser.compressor import build_paired_bt
from telemetry_reasoning.schemas import (
    TASK_SUBTASKS, FailureMode, MetadataRow, RawTelemetryRow, Task,
)

ROOT = Path(__file__).resolve().parents[1]


def _metadata(pair_id, task=Task.TRANSPORT_OBJECT):
    return MetadataRow(
        pair_id=pair_id, task=task, success_file=f"success_{pair_id}.csv",
        failure_file=f"failure_{pair_id}.csv", failed_subtask=TASK_SUBTASKS[task][1],
        failure_mode=FailureMode.INADEQUATE_GRIP_STRENGTH,
        perturbation_param="force_n", perturbation_value=4.0, seed=723,
    )


def _run(task, failure=False):
    rows = []
    for stage_index, subtask in enumerate(TASK_SUBTASKS[task]):
        for index in range(11):
            rows.append(RawTelemetryRow(
                timestamp=round(len(rows) * 0.02, 8), subtask=subtask,
                joint_positions=[0.0, 0.0], joint_velocities=[0.0, 0.0],
                joint_torques=[0.0, 0.0], gripper_state=1.0,
                gripper_force=14.0 if failure and stage_index == 1 and index >= 4 else 10.0,
                ee_pos=[0.0, 0.0, 0.2], ee_quat=[1.0, 0.0, 0.0, 0.0],
                obj_pos=[0.0, 0.0, 0.1], obj_quat=[1.0, 0.0, 0.0, 0.0],
                obj_vel=[0.0, 0.0, 0.0], contact_force=10.0,
            ))
    return rows


def _write_pair(directory, row):
    write_telemetry_csv(directory / row.success_file, _run(row.task))
    write_telemetry_csv(directory / row.failure_file, _run(row.task, failure=True))


@pytest.fixture
def corpus(tmp_path):
    metadata = [_metadata("test"), _metadata("dev"), _metadata("train")]
    # The held-out paths intentionally do not exist. Tuning must succeed
    # without ever trying to load or hash them.
    for row in metadata[1:]:
        _write_pair(tmp_path, row)
    manifest = {"pairs": [
        {"pair_id": "test", "split": "test", "initial_condition_id": 1},
        {"pair_id": "dev", "split": "dev", "initial_condition_id": 2},
        {"pair_id": "train", "split": "train", "initial_condition_id": 3},
    ]}
    config = CompressionConfig(k_uniform_samples=5, major_event_thresholds={"gripper_force": 2.0})
    return metadata, manifest, config


def test_sweep_fixed_reference_events_strict_threshold_and_recommendation(tmp_path, corpus):
    metadata, manifest, config = corpus
    report = tuning.sweep_compression(metadata, tmp_path, config, manifest)
    assert report["selected_pair_ids"] == ["dev"]
    assert report["manifest_split_counts"] == {"dev": 1, "test": 1, "train": 1}
    assert report["grid"] == {"k_uniform_samples": [0, 2, 5], "threshold_scales": [0.5, 1.0, 2.0]}
    assert len(report["candidates"]) == 9
    assert [item["pair_id"] for item in report["source_hashes"]["telemetry"]] == ["dev"]
    assert all(len(value) == 64 for value in report["source_hashes"]["implementation"].values())
    assert report["reference_thresholds"] == {"gripper_force": 2.0}
    by_id = {item["candidate_id"]: item for item in report["candidates"]}
    low, equal = by_id["k=0;threshold_scale=1"], by_id["k=0;threshold_scale=2"]
    assert low["eligible"]
    assert low["summary"]["totals"]["event_retention"]["gripper_force"]["raw_events"] == 1
    assert low["summary"]["totals"]["event_retention"]["gripper_force"]["retained_event_neighborhoods"] == 1
    # The raw jump is exactly 4 N: candidate threshold 4 excludes it because
    # event detection is strictly greater, while the fixed 2 N reference does
    # still contain it. Candidates cannot redefine their own denominator.
    assert equal["config"]["major_event_thresholds"]["gripper_force"] == 4.0
    assert equal["summary"]["reference_thresholds"] == {"gripper_force": 2.0}
    assert equal["summary"]["totals"]["event_retention"]["gripper_force"]["raw_events"] == 1
    assert equal["summary"]["totals"]["event_retention"]["gripper_force"]["retained_event_neighborhoods"] == 0
    assert not equal["eligible"]
    assert report["recommendation"]["candidate_id"] == "k=0;threshold_scale=0.5"
    assert report["recommendation"]["paired_payload_utf8_bytes"] < report["base"]["summary"]["prompt_payload_sizes"]["paired_without_divergence"]["utf8_bytes"]
    assert report["base"]["summary"]["totals"]["raw_rows"] == 66
    assert "failure_mode" not in json.dumps(report)
    assert config == CompressionConfig(k_uniform_samples=5, major_event_thresholds={"gripper_force": 2.0})


def test_repeat_order_and_labels_do_not_change_tuning(tmp_path, corpus):
    metadata, manifest, config = corpus
    first = tuning.sweep_compression(metadata, tmp_path, config, manifest)
    changed_labels = [row.model_copy(update={"failure_mode": FailureMode.EARLY_GRIPPER_RELEASE})
                      for row in reversed(metadata)]
    reordered_manifest = {"pairs": list(reversed(manifest["pairs"]))}
    second = tuning.sweep_compression(changed_labels, tmp_path, config, reordered_manifest,
                                     k_grid=[5, 0, 2, 2], threshold_scales=[2, 1, 0.5, 1])
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)


def test_complete_pair_candidates_preserve_explicit_observation_states(tmp_path, corpus):
    metadata, manifest, config = corpus
    report = tuning.sweep_compression(metadata, tmp_path, config, manifest,
                                     k_grid=[0], threshold_scales=[1.0])
    dev = next(row for row in metadata if row.pair_id == "dev")
    base = build_paired_bt(dev, tmp_path, config)
    # Recompression changes the observations, so previous comparison results
    # cannot be carried into a tuning candidate.
    values = base.model_dump()
    for node in values["subtasks"]:
        node.update(divergence_status="computed", deterministic_divergence={
            "first_divergence_timestamp": None, "max_abs_deltas": {},
        })
    base = type(base).model_validate(values)
    before = base.model_dump_json()
    candidate = tuning._candidate_tree(base, _run(dev.task), _run(dev.task, failure=True), config)
    assert base.model_dump_json() == before
    assert all(node.failure_observation_status == "observed" for node in candidate.subtasks)
    assert all(node.divergence_status == "uncomputed" for node in candidate.subtasks)
    assert all(node.deterministic_divergence is None for node in candidate.subtasks)
    assert [node.failure_segment_end for node in candidate.subtasks] == [
        "transitioned", "transitioned", "trace_end",
    ]
    assert (report["candidates"][0]["summary"]["prompt_payload_sizes"]["paired_without_divergence"]["utf8_bytes"]
            > report["candidates"][0]["summary"]["prompt_payload_sizes"]["failure_only"]["utf8_bytes"])


@pytest.mark.parametrize("task,observed_stages", [
    (Task.PICK_UP_OBJECT, 1), (Task.PICK_UP_OBJECT, 3),
    (Task.TRANSPORT_OBJECT, 1), (Task.TRANSPORT_OBJECT, 2),
])
def test_partial_failure_candidates_preserve_schema_template_and_observation_states(tmp_path, task, observed_stages):
    row = _metadata("partial", task)
    success, failure = _run(task), _run(task, failure=True)[:11 * observed_stages - 4]
    write_telemetry_csv(tmp_path / row.success_file, success)
    write_telemetry_csv(tmp_path / row.failure_file, failure)
    base_config = CompressionConfig(k_uniform_samples=5, major_event_thresholds={"gripper_force": 2.0})
    candidate_config = CompressionConfig(k_uniform_samples=0, major_event_thresholds={"gripper_force": 2.0})
    base = build_paired_bt(row, tmp_path, base_config)
    before = base.model_dump_json()
    candidate = tuning._candidate_tree(base, success, failure, candidate_config)
    assert base.model_dump_json() == before
    assert tuple(node.subtask for node in candidate.subtasks) == TASK_SUBTASKS[task]
    for index, node in enumerate(candidate.subtasks):
        assert node.metadata == base.subtasks[index].metadata
        assert node.success_samples
        assert node.failure_observation_status == ("observed" if index < observed_stages else "not_observed")
        assert node.failure_segment_end == base.subtasks[index].failure_segment_end
        assert node.divergence_status == ("uncomputed" if index < observed_stages else "not_comparable")
        assert node.deterministic_divergence is None
        assert bool(node.failure_samples) == (index < observed_stages)
    manifest = {"pairs": {row.pair_id: {"split": "dev", "initial_condition_id": "partial"}}}
    report = tuning.sweep_compression([row], tmp_path, base_config, manifest,
                                     k_grid=[0], threshold_scales=[1.0])
    totals = report["candidates"][0]["summary"]["totals"]
    assert totals["raw_rows"] == len(success) + len(failure)
    assert totals["observed_stages"] == len(TASK_SUBTASKS[task]) + observed_stages
    assert totals["not_observed_stages"] == len(TASK_SUBTASKS[task]) - observed_stages
    # A missing suffix supplies no fake errors and cannot prevent tuning from
    # recommending a candidate that preserves the actual observations.
    assert report["candidates"][0]["selection_checks"][0]["reconstruction_available"]
    assert report["recommendation"] is not None
    if observed_stages == 1:
        events = totals["event_retention"]["gripper_force"]
        assert events["raw_events"] == 0
        assert events["event_retention_fraction"] is None
    json.dumps(report, allow_nan=False)


def test_only_explicit_train_or_dev_is_processed(tmp_path, corpus):
    metadata, manifest, config = corpus
    train = tuning.sweep_compression(metadata, tmp_path, config, manifest, split="train",
                                    k_grid=[0], threshold_scales=[1.0])
    assert train["selected_pair_ids"] == ["train"]
    with pytest.raises(ValueError, match="never test"):
        tuning.sweep_compression(metadata, tmp_path, config, manifest, split="test")


def test_all_manifest_groups_preflight_before_any_telemetry_read(tmp_path, corpus, monkeypatch):
    metadata, manifest, config = corpus
    # A duplicate initial condition crossing train/test is invalid even when
    # the dev pair being selected is in a different initial condition.
    manifest["pairs"][2]["initial_condition_id"] = "1"
    def forbidden(*args, **kwargs):
        pytest.fail("telemetry read before checking all split groups")
    monkeypatch.setattr(tuning, "read_telemetry_csv", forbidden)
    monkeypatch.setattr(tuning, "build_paired_bt", forbidden)
    with pytest.raises(ValueError, match="overlaps splits"):
        tuning.sweep_compression(metadata, tmp_path, config, manifest)


@pytest.mark.parametrize("problem,match", [
    ("missing_group", "initial_condition_id"), ("empty_group", "initial_condition_id"),
    ("bool_group", "initial_condition_id"), ("bad_split", "split train, dev, or test"),
    ("missing_pair", "missing=.*test"), ("unknown_pair", "unknown=.*other"),
    ("duplicate_pair", "duplicate pair_id"), ("mismatched_task", "task disagrees"),
])
def test_incomplete_or_invalid_manifests_refused(tmp_path, corpus, problem, match):
    metadata, manifest, config = corpus
    first = manifest["pairs"][0]
    if problem == "missing_group":
        del first["initial_condition_id"]
    elif problem == "empty_group":
        first["initial_condition_id"] = " "
    elif problem == "bool_group":
        first["initial_condition_id"] = True
    elif problem == "bad_split":
        first["split"] = "validation"
    elif problem == "missing_pair":
        manifest["pairs"].pop(0)
    elif problem == "unknown_pair":
        manifest["pairs"].append({"pair_id": "other", "split": "test", "initial_condition_id": 4})
    elif problem == "duplicate_pair":
        manifest["pairs"].append(dict(first))
    elif problem == "mismatched_task":
        first["task"] = Task.PICK_UP_OBJECT.value
    with pytest.raises(ValueError, match=match):
        tuning.sweep_compression(metadata, tmp_path, config, manifest)


def test_mapping_manifest_and_task_scoped_group_identity(corpus):
    metadata, manifest, _ = corpus
    metadata[0] = _metadata("test", Task.PICK_UP_OBJECT)
    manifest["pairs"][0]["initial_condition_id"] = manifest["pairs"][1]["initial_condition_id"]
    mapped = {"pairs": {item["pair_id"]: {key: value for key, value in item.items() if key != "pair_id"}
                         for item in manifest["pairs"]}}
    assert tuning.validate_tuning_manifest(metadata, mapped)["dev"]["split"] == "dev"


def test_absent_manifest_or_empty_requested_split_refused(tmp_path, corpus):
    metadata, manifest, config = corpus
    for missing in ({}, {"pairs": None}, []):
        with pytest.raises(ValueError, match="manifest"):
            tuning.sweep_compression(metadata, tmp_path, config, missing)
    for item in manifest["pairs"]:
        if item["split"] == "dev":
            item["split"] = "train"
    with pytest.raises(ValueError, match="no metadata pairs.*dev"):
        tuning.sweep_compression(metadata, tmp_path, config, manifest)


@pytest.mark.parametrize("k_grid,scales", [([], [1]), ([True], [1]), ([-1], [1]),
                                           ([0], []), ([0], [float("nan")]), ([0], [0])])
def test_invalid_grids_refused(tmp_path, corpus, k_grid, scales):
    metadata, manifest, config = corpus
    with pytest.raises(ValueError):
        tuning.sweep_compression(metadata, tmp_path, config, manifest,
                                 k_grid=k_grid, threshold_scales=scales)


def test_selected_invalid_telemetry_fails_instead_of_partial_tuning(tmp_path, corpus):
    metadata, manifest, config = corpus
    dev = next(row for row in metadata if row.pair_id == "dev")
    (tmp_path / dev.failure_file).unlink()
    with pytest.raises(FileNotFoundError):
        tuning.sweep_compression(metadata, tmp_path, config, manifest)


def test_reconstruction_constraint_rejects_event_preserving_candidate(tmp_path, corpus):
    metadata, manifest, config = corpus
    dev = next(row for row in metadata if row.pair_id == "dev")
    rows = _run(dev.task)
    rows[5] = rows[5].model_copy(update={"obj_pos": [0.01, 0.0, 0.1]})
    write_telemetry_csv(tmp_path / dev.success_file, rows)
    report = tuning.sweep_compression(metadata, tmp_path, config, manifest,
                                     k_grid=[0], threshold_scales=[1.0])
    check = report["candidates"][0]["selection_checks"][0]
    assert check["all_reference_event_neighborhoods_retained"]
    assert check["reconstruction_available"]
    assert not check["within_base_max_reconstruction_bounds"]
    assert report["recommendation"] is None


def test_missing_quaternion_reconstruction_never_recommends(tmp_path, corpus):
    metadata, manifest, config = corpus
    dev = next(row for row in metadata if row.pair_id == "dev")
    rows = _run(dev.task)
    rows[0] = rows[0].model_copy(update={"ee_quat": [0.0, 0.0, 0.0, 0.0]})
    write_telemetry_csv(tmp_path / dev.success_file, rows)
    report = tuning.sweep_compression(metadata, tmp_path, config, manifest,
                                     k_grid=[0], threshold_scales=[1.0])
    assert report["recommendation"] is None
    assert not report["candidates"][0]["selection_checks"][0]["reconstruction_available"]


def test_cli_writes_deterministic_report_and_requires_manifest(tmp_path, corpus, capsys):
    metadata, manifest, _ = corpus
    spec = importlib.util.spec_from_file_location("tune_compression_script", ROOT / "scripts/tune_compression.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    metadata_path, manifest_path = tmp_path / "metadata.csv", tmp_path / "manifest.json"
    write_metadata_csv(metadata_path, metadata)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "results" / "tuning.json"
    argv = ["--metadata", str(metadata_path), "--data-dir", str(tmp_path),
            "--config", str(ROOT / "configs/default.yaml"), "--output", str(output),
            "--k-grid", "0", "--threshold-scales", "1"]
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2 and not output.exists()
    argv += ["--manifest", str(manifest_path)]
    assert cli.main(argv) == 0
    first = output.read_bytes()
    assert cli.main(argv) == 0
    assert first == output.read_bytes()
    report = json.loads(first)
    assert report["selected_pair_ids"] == ["dev"]
    for key in ("metadata_sha256", "manifest_sha256", "config_file_sha256"):
        assert len(report["source_hashes"][key]) == 64
    assert "UTF-8 bytes" in capsys.readouterr().out


@pytest.mark.parametrize("bad_input", ["manifest_list", "manifest_null", "malformed_yaml", "overwrite_metadata"])
def test_cli_invalid_inputs_return_useful_error_without_overwriting_sources(tmp_path, corpus, capsys, bad_input):
    metadata, manifest, _ = corpus
    spec = importlib.util.spec_from_file_location("tune_compression_bad_input", ROOT / "scripts/tune_compression.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    metadata_path, manifest_path = tmp_path / "metadata.csv", tmp_path / "manifest.json"
    write_metadata_csv(metadata_path, metadata)
    before = metadata_path.read_bytes()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = ROOT / "configs/default.yaml"
    output = tmp_path / "report.json"
    if bad_input in ("manifest_list", "manifest_null"):
        manifest_path.write_text("[]" if bad_input == "manifest_list" else "null", encoding="utf-8")
    elif bad_input == "malformed_yaml":
        config_path = tmp_path / "config.yaml"
        config_path.write_text("parser: [broken", encoding="utf-8")
    else:
        output = metadata_path
    with pytest.raises(SystemExit) as error:
        cli.main(["--metadata", str(metadata_path), "--data-dir", str(tmp_path),
                  "--config", str(config_path), "--manifest", str(manifest_path), "--output", str(output)])
    assert error.value.code == 2
    assert metadata_path.read_bytes() == before
    assert "error:" in capsys.readouterr().err
