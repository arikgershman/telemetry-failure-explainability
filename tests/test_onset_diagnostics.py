"""Raw numerical reference diagnostics do not require physics or injection time."""

from itertools import groupby
from pathlib import Path

import pytest

from telemetry_reasoning.config import CompressionConfig, load_config
from telemetry_reasoning.io import read_metadata_csv, read_telemetry_csv
from telemetry_reasoning.parser.compressor import build_bt_from_segments, build_paired_bt
from telemetry_reasoning.parser.onset_diagnostics import (
    combine_onset_diagnostics, raw_reference_divergence_diagnostics,
)
from test_divergence_metrics import only_force_threshold

SAMPLE = Path(__file__).resolve().parents[1] / "data/sample"


def sample():
    metadata = read_metadata_csv(SAMPLE / "metadata.csv")[0]
    success = read_telemetry_csv(SAMPLE / metadata.success_file)
    failure = read_telemetry_csv(SAMPLE / metadata.failure_file)
    tree = build_paired_bt(metadata, SAMPLE, load_config(SAMPLE.parents[1] / "configs/parser.yaml").parser)
    return success, failure, tree


def test_sample_reference_preserves_known_onset_without_mutation():
    success, failure, tree = sample()
    before = tree.model_dump_json()
    report = raw_reference_divergence_diagnostics(success, failure, tree, only_force_threshold())
    grasp = report["stages"][2]
    assert grasp["raw_timestamp"] == pytest.approx(5.10)
    assert grasp["compressed_timestamp"] == pytest.approx(5.10)
    assert grasp["absolute_error_s"] == pytest.approx(0)
    assert report["summary"]["comparison_counts"]["both_crossing"] == 2
    assert "physical" in report["note"]
    assert tree.model_dump_json() == before


def test_lossy_trace_counts_missed_onset_without_zero_imputation():
    success, failure, tree = sample()
    # A brief grip dip that is not a retained endpoint disappears in the
    # deliberately lossy comparison; all other stages remain identical.
    failure = [row.model_copy(deep=True) for row in success]
    grasp = [row for row in failure if row.subtask.value == "grasp"]
    grasp[len(grasp) // 2].gripper_force = 0
    segments = lambda rows: [list(g) for _, g in groupby(rows, key=lambda row: row.subtask)]
    raw = build_bt_from_segments(tree.metadata, segments(success), segments(failure))
    compressed = raw.model_copy(deep=True)
    for node in compressed.subtasks:
        node.success_samples = [node.success_samples[0], node.success_samples[-1]]
        node.failure_samples = [node.failure_samples[0], node.failure_samples[-1]]
    report = raw_reference_divergence_diagnostics(success, failure, compressed, only_force_threshold())
    result = report["stages"][2]
    assert result["comparison"] == "missed_compressed_crossing"
    assert result["raw_timestamp"] is not None and result["compressed_timestamp"] is None
    assert result["signed_error_s"] is None
    assert report["summary"]["timing_error_count"] == 0
    assert report["summary"]["max_absolute_error_s"] is None


def test_lossy_reference_reports_nonzero_signed_timing_error():
    success, failure, original = sample()
    # Deliberately disable event retention to expose measurable loss of the
    # sample's 5.10 s grip event, instead of assuming every compression is exact.
    lossy = build_paired_bt(original.metadata, SAMPLE, CompressionConfig(k_uniform_samples=5))
    report = raw_reference_divergence_diagnostics(success, failure, lossy, only_force_threshold())
    grasp = report["stages"][2]
    assert grasp["raw_timestamp"] == pytest.approx(5.10)
    assert grasp["compressed_timestamp"] == pytest.approx(5.32)
    assert grasp["signed_error_s"] == pytest.approx(0.22)
    assert grasp["absolute_error_s"] == pytest.approx(0.22)


def test_asymmetric_compression_counts_extra_crossing():
    success, _, tree = sample()
    success = [row.model_copy(deep=True) for row in success]
    grasp = [row for row in success if row.subtask.value == "grasp"]
    grasp[len(grasp) // 2].gripper_force += 10
    failure = [row.model_copy(deep=True) for row in success]
    segments = lambda rows: [list(g) for _, g in groupby(rows, key=lambda row: row.subtask)]
    compressed = build_bt_from_segments(tree.metadata, segments(success), segments(failure))
    compressed.subtasks[2].success_samples = [grasp[0], grasp[-1]]
    report = raw_reference_divergence_diagnostics(success, failure, compressed, only_force_threshold())
    result = report["stages"][2]
    assert result["comparison"] == "extra_compressed_crossing"
    assert result["raw_timestamp"] is None and result["compressed_timestamp"] is not None
    assert result["absolute_error_s"] is None
    assert report["summary"]["comparison_counts"]["extra_compressed_crossing"] == 1


def test_partial_coverage_retains_unavailable_stage():
    success, failure, tree = sample()
    failure = [row for row in failure if row.subtask.value != "lift"]
    raw = build_bt_from_segments(tree.metadata,
         [list(g) for _, g in groupby(success, key=lambda row: row.subtask)],
         [list(g) for _, g in groupby(failure, key=lambda row: row.subtask)])
    report = raw_reference_divergence_diagnostics(success, failure, raw, only_force_threshold())
    assert report["stages"][-1]["comparison"] == "not_comparable"
    assert report["summary"]["comparison_counts"]["not_comparable"] == 1


@pytest.mark.parametrize("mutation", ["observation", "endpoint", "coverage"])
def test_reference_rejects_fabricated_samples_or_states(mutation):
    success, failure, tree = sample()
    if mutation == "observation":
        tree.subtasks[0].failure_samples[1].contact_force += 1
    elif mutation == "endpoint":
        tree.subtasks[0].failure_samples.pop()
    else:
        tree.subtasks[0].failure_segment_end = "trace_end"
    with pytest.raises(ValueError, match="raw"):
        raw_reference_divergence_diagnostics(success, failure, tree, only_force_threshold())


def test_empty_summary_has_null_errors():
    report = combine_onset_diagnostics([])
    assert report["stage_count"] == 0
    assert report["timing_error_count"] == 0
    assert report["mean_absolute_error_s"] is None
    assert all(value is None for value in report["max_delta_absolute_errors"].values())
