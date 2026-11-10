
"""Tests for deterministic, leakage-free dataset splitting."""

from collections import Counter

import pytest

from telemetry_reasoning.sim.split_robot_dataset import create_splits
from test_dataset_validation import dataset  # Reuse dataset fixture


def test_split_sizes(dataset):
    """Check the required 70/10/20 split."""
    manifest = create_splits(dataset)

    assert {
        split: len(pairs)
        for split, pairs in manifest["splits"].items()
    } == {
        "train": 700,
        "dev": 100,
        "test": 200,
    }


def test_no_initial_condition_leakage(dataset):
    """Within each task, an initial condition belongs to only one split."""
    manifest = create_splits(dataset)

    for task in ("pick_up_object", "transport_object"):
        condition_sets = {}

        for split, pairs in manifest["splits"].items():
            condition_sets[split] = {
                pair["initial_condition_id"]
                for pair in pairs
                if pair["task"] == task
            }

        assert condition_sets["train"].isdisjoint(condition_sets["dev"])
        assert condition_sets["train"].isdisjoint(condition_sets["test"])
        assert condition_sets["dev"].isdisjoint(condition_sets["test"])

        assert {
            split: len(conditions)
            for split, conditions in condition_sets.items()
        } == {
            "train": 35,
            "dev": 5,
            "test": 10,
        }


def test_every_pair_appears_once(dataset):
    """All 1,000 pairs should appear exactly once."""
    manifest = create_splits(dataset)

    pair_keys = [
        (pair["task"], pair["pair_id"])
        for pairs in manifest["splits"].values()
        for pair in pairs
    ]

    assert len(pair_keys) == 1000
    assert len(set(pair_keys)) == 1000


def test_reproducible_splits(dataset):
    """Same seed and dataset must produce identical assignments."""
    first = create_splits(dataset, seed=723)
    second = create_splits(dataset, seed=723)

    assert first == second


def test_different_seed_changes_assignments(dataset):
    """Changing the seed should change the condition assignments."""
    first = create_splits(dataset, seed=723)
    second = create_splits(dataset, seed=42)

    assert first["condition_assignments"] != second["condition_assignments"]
