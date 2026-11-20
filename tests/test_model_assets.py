"""Pinned model restoration without network, MuJoCo, or large binary fixtures."""

import hashlib
import json
from pathlib import Path

import pytest

from telemetry_reasoning.sim import model_assets


def inventory():
    data = b"<mujoco/>"
    sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    return {"sha": model_assets.MENAGERIE_REVISION, "truncated": False, "tree": [
        {"path": f"franka_emika_panda/{name}", "type": "blob", "mode": "100644", "sha": sha, "size": len(data)}
        for name in ("panda.xml", "scene.xml")
    ]}


@pytest.fixture(autouse=True)
def pin_small_fixture_inventory(monkeypatch):
    entries = inventory()["tree"]
    canonical = [{key: entry[key] for key in ("path", "type", "mode", "sha", "size")} for entry in entries]
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(model_assets, "PANDA_INVENTORY_SHA256", digest)


def test_restore_verifies_every_blob_and_refuses_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(model_assets, "_fetch", lambda url: b"<mujoco/>")
    output = tmp_path / "panda"
    receipt = model_assets.restore_panda(output, tree=inventory())
    assert set(receipt["files"]) == {"panda.xml", "scene.xml"}
    assert (output / "panda.xml").read_bytes() == b"<mujoco/>"
    with pytest.raises(ValueError, match="absent or empty"):
        model_assets.restore_panda(output, tree=inventory())


def test_corrupt_blob_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(model_assets, "_fetch", lambda url: b"corrupted")
    with pytest.raises(ValueError, match="blob mismatch"):
        model_assets.restore_panda(tmp_path / "panda", tree=inventory())
    assert not (tmp_path / "panda").exists()


@pytest.mark.parametrize("change", [
    {"sha": "other"}, {"truncated": True},
])
def test_wrong_or_truncated_revision_rejected(change):
    with pytest.raises(ValueError, match="release-pinned"):
        model_assets.panda_inventory({**inventory(), **change})


def test_incomplete_inventory_cannot_claim_complete_pinned_subtree(monkeypatch):
    monkeypatch.setattr(model_assets, "PANDA_INVENTORY_SHA256", "86365ae032e08870a8459f52e2e0194f7d00a379ade9f9641209747a8d48162b")
    with pytest.raises(ValueError, match="complete pinned subtree"):
        model_assets.panda_inventory(inventory())


@pytest.mark.parametrize("path,mode", [("franka_emika_panda/../../escape", "100644"),
                                        ("franka_emika_panda/link", "120000")])
def test_paths_and_symlinks_are_rejected(path, mode):
    tree = inventory()
    tree["tree"].append({**tree["tree"][0], "path": path, "mode": mode})
    with pytest.raises(ValueError, match="Unsafe"):
        model_assets.panda_inventory(tree)
