"""Offline release restoration regressions with small pinned archive fixtures."""
from contextlib import nullcontext
import json
import stat
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from telemetry_reasoning.sim.release import (
    load_release_lock, restore_release, sha256_file, validate_release_lock,
)

TASKS = ("pick_up_object", "transport_object")


def make_release(tmp_path, *, extras=(), sidecar=True, alter=None):
    cache = tmp_path / "assets"
    cache.mkdir()
    archive = cache / "dataset.zip"
    members = []
    for task in TASKS:
        prefix = f"batch_fixture/{task}"
        config = {"mode": "position_error", "subtask": "align" if task == TASKS[0] else "move_along_path", "magnitude": .1, "delay_s": 0, "axis": 0}
        members.extend([(prefix + "/metadata.csv", "opaque unchanged\r\n"),
                        (prefix + "/summary.json", "{}"),
                        (prefix + "/diagnostics.jsonl", json.dumps({"pair_id": "000", "failure_config": config}) + "\n"),
                        (prefix + "/success/000.csv", "success opaque bytes\r\n"),
                        (prefix + "/failure/000.csv", "failure opaque bytes\r\n")])
    members.extend(extras)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as target:
        for name, content in members:
            target.writestr(name, content)
    assets = [{"name": archive.name, "size_bytes": archive.stat().st_size,
               "sha256": sha256_file(archive), "url": "https://example.org/dataset.zip", "role": "archive"}]
    if sidecar:
        clock = cache / "published.clock.jsonl"
        clock.write_text('{"unverified":true}\n')
        assets.append({"name": clock.name, "size_bytes": clock.stat().st_size,
                       "sha256": sha256_file(clock), "url": "https://example.org/clock", "role": "published_clock_sidecar", "task": TASKS[0]})
    lock = {"lock_version": 1, "batch_name": "batch_fixture", "archive_asset": archive.name,
            "expected_task_files": 10, "release": {"tag": "fixture"},
            "source_reference": {"status": "release_tag_reference_only"}, "assets": assets,
            "extraction_limits": {"max_members": 100, "max_total_bytes": 1_000_000,
                                  "max_member_bytes": 100_000, "max_expansion_ratio": 1000}}
    if alter:
        alter(lock)
    return cache, lock, members


def test_restore_preserves_source_bytes_and_separates_derived_assets(tmp_path):
    cache, lock, members = make_release(tmp_path, extras=[("batch_fixture/.DS_Store", b"opaque")])
    batch = restore_release(lock, cache, tmp_path / "output")
    for name, content in members:
        expected = content.encode() if isinstance(content, str) else content
        assert (batch.parent / name).read_bytes() == expected
    assert len([p for task in TASKS for p in (batch / task).rglob("*") if p.is_file()]) == 10
    receipt = json.loads((batch / "provenance/release.json").read_text())
    assert receipt["status"] == "restored_verified_assets"
    assert receipt["producer_revision"] is None
    assert receipt["published_clock_sidecars"][0]["verification"] == "published_asset_identity_only"
    catalogue = json.loads((batch / "provenance/recovered_failure_configs.json").read_text())
    assert catalogue["status"] == "recovered_accepted_configurations_only"
    assert "failure_rate" not in json.dumps(catalogue)
    assert all(catalogue["tasks"][task]["accepted_pairs"] == 1 for task in TASKS)
    for task in TASKS:
        configs = catalogue["tasks"][task]["configurations"]
        assert configs[0]["accepted_pair_count"] == 1
    assert receipt["recovered_configurations"]["sha256"] == sha256_file(batch / receipt["recovered_configurations"]["path"])
    assert not list(tmp_path.glob(".output-restore-*"))


def test_existing_root_is_untouched(tmp_path):
    cache, lock, _ = make_release(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing")
    with pytest.raises(ValueError, match="must be fresh"):
        restore_release(lock, cache, output)
    assert marker.read_text() == "existing"


@pytest.mark.parametrize("corruption", ["same_size", "truncate", "missing", "symlink"])
def test_all_asset_hashes_checked_before_archive_is_opened(tmp_path, monkeypatch, corruption):
    cache, lock, _ = make_release(tmp_path)
    path = cache / "published.clock.jsonl"
    if corruption == "same_size":
        path.write_bytes(b"x" * path.stat().st_size)
    elif corruption == "truncate":
        path.write_bytes(b"x")
    elif corruption == "missing":
        path.unlink()
    else:
        content = path.read_bytes()
        path.unlink()
        other = tmp_path / "elsewhere"
        other.write_bytes(content)
        path.symlink_to(other)
    def never_open(*args, **kwargs):
        raise AssertionError("Archive opened before all assets were verified")
    monkeypatch.setattr("telemetry_reasoning.sim.release.ZipFile", never_open)
    with pytest.raises(ValueError, match="asset"):
        restore_release(lock, cache, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name", ["../escaped", "/absolute", "batch_fixture/../escaped", "batch_fixture/a\\b", "batch_fixture//double", "batch_fixture/C:drive", "other/file", "batch_fixture/./dot"])
def test_unsafe_zip_paths_are_rejected_atomically(tmp_path, name):
    cache, lock, _ = make_release(tmp_path, extras=[(name, "bad")])
    with pytest.raises(ValueError, match="Unsafe|unexpected"):
        restore_release(lock, cache, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output-restore-*"))


@pytest.mark.parametrize("extras", [
    [("batch_fixture/pick_up_object/metadata.csv", "duplicate")],
    [("batch_fixture/pick_up_object/METADATA.csv", "case collision")],
    [("batch_fixture/conflict", "file"), ("batch_fixture/conflict/child", "child")],
])
def test_duplicate_or_conflicting_members_are_rejected(tmp_path, extras):
    with pytest.warns(UserWarning) if "metadata.csv" in extras[0][0] else nullcontext():
        cache, lock, _ = make_release(tmp_path, extras=extras)
    with pytest.raises(ValueError, match="Duplicate|conflict"):
        restore_release(lock, cache, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_zip_symlink_is_rejected(tmp_path):
    link = ZipInfo("batch_fixture/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    cache, lock, _ = make_release(tmp_path, extras=[(link, "../../outside")])
    with pytest.raises(ValueError, match="regular file"):
        restore_release(lock, cache, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("limit,value", [("max_members", 1), ("max_total_bytes", 1), ("max_member_bytes", 1), ("max_expansion_ratio", 1)])
def test_resource_limits_block_zip_bombs(tmp_path, limit, value):
    cache, lock, _ = make_release(tmp_path, extras=[("batch_fixture/repetitive", "x" * 10000)])
    lock["extraction_limits"][limit] = value
    with pytest.raises(ValueError, match="limit"):
        restore_release(lock, cache, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_unexpected_task_inventory_is_rejected(tmp_path):
    cache, lock, _ = make_release(tmp_path)
    lock["expected_task_files"] += 1
    with pytest.raises(ValueError, match="file count"):
        restore_release(lock, cache, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_corrupt_zip_has_no_published_or_staging_output(tmp_path):
    cache, lock, _ = make_release(tmp_path)
    archive = cache / "dataset.zip"
    archive.write_bytes(b"not a zip")
    lock["assets"][0].update(size_bytes=archive.stat().st_size, sha256=sha256_file(archive))
    from zipfile import BadZipFile
    with pytest.raises(BadZipFile):
        restore_release(lock, cache, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output-restore-*"))


@pytest.mark.parametrize("mutation", [
    lambda lock: lock.update(lock_version=99),
    lambda lock: lock.update(batch_name="../bad"),
    lambda lock: lock["assets"].append(dict(lock["assets"][0])),
    lambda lock: lock["assets"][0].update(sha256="bad"),
    lambda lock: lock["assets"][0].update(size_bytes=True),
    lambda lock: lock["assets"][0].update(url="http://insecure"),
    lambda lock: lock["extraction_limits"].update(max_members=0),
])
def test_malformed_lock_is_rejected(tmp_path, mutation):
    _, lock, _ = make_release(tmp_path)
    mutation(lock)
    with pytest.raises(ValueError):
        validate_release_lock(lock)


def test_explicit_download_checks_streamed_bytes_and_preserves_cache(tmp_path, monkeypatch):
    from io import BytesIO
    cache, lock, _ = make_release(tmp_path)
    contents = {(asset["url"]): (cache / asset["name"]).read_bytes() for asset in lock["assets"]}
    for path in cache.iterdir():
        path.unlink()
    calls = []
    def fake_open(request, **kwargs):
        calls.append(request.full_url)
        return BytesIO(contents[request.full_url])
    monkeypatch.setattr("telemetry_reasoning.sim.release.urlopen", fake_open)
    restore_release(lock, cache, tmp_path / "output", download=True)
    assert calls == [asset["url"] for asset in lock["assets"]]
    for asset in lock["assets"]:
        assert sha256_file(cache / asset["name"]) == asset["sha256"]
    assert not list(cache.glob(".download-*"))


def test_bundled_lock_pins_release_assets():
    lock = load_release_lock()
    assert lock["source_reference"]["status"] == "release_tag_reference_only"
    assert lock["assets"][0]["sha256"] == "657560c5a42c21bc2770525cb0d7ad0c219f226dd669a119c13d63e6dc157ce4"
    assert lock["expected_task_files"] == 2006
    assert len(lock["assets"]) == 3


def test_restore_cli_offline(tmp_path):
    cache, lock, _ = make_release(tmp_path)
    lock_file = tmp_path / "lock.json"
    lock_file.write_text(json.dumps(lock))
    result = subprocess.run([sys.executable, "scripts/restore_dataset.py", "--asset-dir", str(cache), "--dataset-root", str(tmp_path / "output"), "--lock", str(lock_file)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Producer provenance" in result.stdout


@pytest.mark.parametrize("response", [b"short", b"x" * 100_000])
def test_invalid_download_never_publishes_partial_cache_or_dataset(tmp_path, monkeypatch, response):
    from io import BytesIO
    cache, lock, _ = make_release(tmp_path)
    (cache / lock["archive_asset"]).unlink()
    monkeypatch.setattr("telemetry_reasoning.sim.release.urlopen", lambda *args, **kwargs: BytesIO(response))
    with pytest.raises(ValueError, match="size|pinned"):
        restore_release(lock, cache, tmp_path / "output", download=True)
    assert not (cache / lock["archive_asset"]).exists()
    assert not list(cache.glob(".download-*"))
    assert not (tmp_path / "output").exists()


def test_download_refuses_to_overwrite_corrupt_existing_cache(tmp_path, monkeypatch):
    cache, lock, _ = make_release(tmp_path)
    path = cache / lock["archive_asset"]
    path.write_bytes(b"damaged")
    def no_download(*args, **kwargs):
        raise AssertionError("An existing corrupt cache entry must not be replaced")
    monkeypatch.setattr("telemetry_reasoning.sim.release.urlopen", no_download)
    with pytest.raises(ValueError, match="size"):
        restore_release(lock, cache, tmp_path / "output", download=True)
    assert path.read_bytes() == b"damaged"


def test_malformed_recovered_diagnostics_roll_back_publication(tmp_path):
    cache, lock, members = make_release(tmp_path)
    archive = cache / lock["archive_asset"]
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as target:
        for name, content in members:
            target.writestr(name, "not json" if name.endswith("diagnostics.jsonl") else content)
    lock["assets"][0].update(size_bytes=archive.stat().st_size, sha256=sha256_file(archive))
    with pytest.raises(ValueError):
        restore_release(lock, cache, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output-restore-*"))
