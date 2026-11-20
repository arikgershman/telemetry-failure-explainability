"""Pinned, staged restoration of immutable dataset release assets.

Restoration proves asset identity and preserves archived bytes. It does not
establish the producer working tree, certify published clocks, or replace the
separate dataset/certification validators. This module has no simulator imports.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import unicodedata
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

TASKS = ("pick_up_object", "transport_object")
DEFAULT_LOCK = Path(__file__).resolve().parents[3] / "configs/dataset-v1.json"
_LIMITS = {"max_members": 3000, "max_total_bytes": 1_000_000_000,
           "max_member_bytes": 10_000_000, "max_expansion_ratio": 1000}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(name: str, *, directory=False) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    stripped = name[:-1] if directory and name.endswith("/") else name
    parts = stripped.split("/")
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        raise ValueError(f"Unsafe archive path: {name!r}")
    result = PurePosixPath(stripped)
    if result.is_absolute():
        raise ValueError(f"Unsafe archive path: {name!r}")
    return result


def load_release_lock(path=DEFAULT_LOCK) -> dict:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_release_lock(lock)
    return lock


def validate_release_lock(lock: dict) -> None:
    """Reject ambiguous assets and malformed bounds before touching outputs."""
    if not isinstance(lock, dict) or lock.get("lock_version") != 1:
        raise ValueError("Unsupported release lock version")
    batch = _safe_path(lock.get("batch_name"))
    if len(batch.parts) != 1:
        raise ValueError("batch_name must be one directory name")
    assets = lock.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("Release lock needs assets")
    names = []
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("Invalid asset record")
        name = _safe_path(asset.get("name"))
        if len(name.parts) != 1:
            raise ValueError("Asset names must be plain filenames")
        names.append(str(name).casefold())
        size = asset.get("size_bytes")
        if type(size) is not int or size < 1:
            raise ValueError("Asset size must be a positive integer")
        if not isinstance(asset.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", asset["sha256"]):
            raise ValueError("Asset needs a lowercase SHA-256 digest")
        if asset.get("role") not in ("archive", "published_clock_sidecar"):
            raise ValueError("Unsupported asset role")
        if asset.get("role") == "published_clock_sidecar" and asset.get("task") not in TASKS:
            raise ValueError("Clock sidecar needs a canonical task")
        if urlparse(asset.get("url", "")).scheme != "https":
            raise ValueError("Asset URL must use HTTPS")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate release asset names")
    archives = [asset["name"] for asset in assets if asset["role"] == "archive"]
    if archives != [lock.get("archive_asset")]:
        raise ValueError("Release lock must identify exactly one archive")
    expected = lock.get("expected_task_files")
    if type(expected) is not int or expected < 1:
        raise ValueError("expected_task_files must be a positive integer")
    limits = lock.get("extraction_limits", _LIMITS)
    if set(limits) != set(_LIMITS) or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError("Extraction limits must be positive integers")
    for key in ("release", "source_reference"):
        if not isinstance(lock.get(key), dict):
            raise ValueError(f"Release lock needs {key}")


def verify_asset(path: Path, asset: dict) -> None:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular release asset: {path}")
    if path.stat().st_size != asset["size_bytes"]:
        raise ValueError(f"Release asset size mismatch: {path.name}")
    if sha256_file(path) != asset["sha256"]:
        raise ValueError(f"Release asset SHA-256 mismatch: {path.name}")


def _download(asset: dict, asset_dir: Path) -> None:
    destination = asset_dir / asset["name"]
    if destination.exists() or destination.is_symlink():
        verify_asset(destination, asset)
        return
    asset_dir.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=".download-", dir=asset_dir)
    temporary = Path(filename)
    try:
        request = Request(asset["url"], headers={"User-Agent": "telemetry-reasoning-release/1"})
        with os.fdopen(fd, "wb") as target, urlopen(request, timeout=60) as source:
            count = 0
            while block := source.read(1024 * 1024):
                count += len(block)
                if count > asset["size_bytes"]:
                    raise ValueError(f"Release download exceeds pinned size: {asset['name']}")
                target.write(block)
        verify_asset(temporary, asset)
        # Do not replace a cache entry created concurrently.
        if destination.exists() or destination.is_symlink():
            verify_asset(destination, asset)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                verify_asset(destination, asset)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_archive(archive: Path, output: Path, lock: dict) -> None:
    limits = lock.get("extraction_limits", _LIMITS)
    with ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > limits["max_members"]:
            raise ValueError("Archive exceeds member count limit")
        total = 0
        paths = {}
        for entry in entries:
            _safe_path(entry.orig_filename, directory=entry.is_dir())
            path = _safe_path(entry.filename, directory=entry.is_dir())
            if path.parts[0] != lock["batch_name"]:
                raise ValueError("Archive has an unexpected top-level directory")
            key = unicodedata.normalize("NFC", str(path)).casefold()
            if key in paths:
                raise ValueError(f"Duplicate archive path: {entry.filename}")
            paths[key] = entry.is_dir()
            mode = entry.external_attr >> 16
            kind = stat.S_IFMT(mode)
            type_mismatch = kind != 0 and ((kind == stat.S_IFDIR) != entry.is_dir())
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or type_mismatch:
                raise ValueError(f"Archive member is not a regular file/directory: {entry.filename}")
            if entry.flag_bits & 1 or entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
                raise ValueError("Encrypted or unsupported archive member")
            if entry.is_dir() and entry.file_size:
                raise ValueError("Archive directory has file content")
            total += entry.file_size
            if entry.file_size > limits["max_member_bytes"] or total > limits["max_total_bytes"]:
                raise ValueError("Archive exceeds uncompressed byte limit")
            if entry.file_size / max(1, entry.compress_size) > limits["max_expansion_ratio"]:
                raise ValueError("Archive exceeds expansion ratio limit")
        for name in paths:
            for parent in PurePosixPath(name).parents:
                if str(parent) in paths and paths[str(parent)] is False:
                    raise ValueError("Archive file/directory path conflict")
        # All names/types/declared sizes are checked before extraction begins.
        for entry in entries:
            path = output.joinpath(*_safe_path(entry.filename, directory=entry.is_dir()).parts)
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as incoming, path.open("xb") as target:
                count = 0
                while block := incoming.read(1024 * 1024):
                    count += len(block)
                    if count > entry.file_size:
                        raise ValueError("Archive member exceeds declared size")
                    target.write(block)
            if count != entry.file_size:
                raise ValueError("Archive member length mismatch")


def recovered_configuration_catalogue(batch: Path) -> dict:
    """Inventory accepted configurations without inventing calibration evidence."""
    catalogue = {"catalogue_version": 1, "status": "recovered_accepted_configurations_only",
                 "limitation": "Recovered from accepted diagnostics. These are not the original calibration CSVs; rejected candidates, activation/failure rates, and calibration selection provenance are unavailable.",
                 "tasks": {}}
    for task in TASKS:
        source = batch / task / "diagnostics.jsonl"
        configurations = Counter()
        examples = {}
        seen = set()
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            pair_id = record["pair_id"]
            if not isinstance(pair_id, str) or pair_id in seen:
                raise ValueError(f"Invalid/duplicate diagnostic pair ID for {task}")
            seen.add(pair_id)
            config = record["failure_config"]
            if not isinstance(config, dict) or not config:
                raise ValueError(f"Missing failure configuration for {task}/{pair_id}")
            key = json.dumps(config, sort_keys=True, allow_nan=False)
            configurations[key] += 1
            examples.setdefault(key, pair_id)
        catalogue["tasks"][task] = {"diagnostics_sha256": sha256_file(source),
            "accepted_pairs": len(seen), "configurations": [
                {"failure_config": json.loads(key), "accepted_pair_count": configurations[key],
                 "representative_pair_id": examples[key]}
                for key in sorted(configurations)]}
    return catalogue


def restore_release(lock: dict, asset_dir: Path, dataset_root: Path, *, download=False) -> Path:
    """Verify all assets, extract safely, and publish one fresh dataset root.

    Caller-supplied locks support offline fixtures/future releases. The bundled
    lock pins v1. Existing output roots and invalid cache assets are never replaced.
    """
    validate_release_lock(lock)
    asset_dir, dataset_root = Path(asset_dir).resolve(), Path(dataset_root).absolute()
    if dataset_root.exists() or dataset_root.is_symlink():
        raise ValueError(f"Dataset root must be fresh: {dataset_root}")
    for asset in lock["assets"]:
        if download:
            _download(asset, asset_dir)
        verify_asset(asset_dir / asset["name"], asset)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    publish_lock = dataset_root.parent / f".{dataset_root.name}.restore.lock"
    descriptor = os.open(publish_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    staging = None
    try:
        if dataset_root.exists() or dataset_root.is_symlink():
            raise ValueError(f"Dataset root must be fresh: {dataset_root}")
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset_root.name}-restore-", dir=dataset_root.parent))
        _extract_archive(asset_dir / lock["archive_asset"], staging, lock)
        batch = staging / lock["batch_name"]
        for task in TASKS:
            for name in ("metadata.csv", "diagnostics.jsonl", "summary.json"):
                if not (batch / task / name).is_file():
                    raise ValueError(f"Missing archived task input: {task}/{name}")
            if any(not (batch / task / name).is_dir() for name in ("success", "failure")):
                raise ValueError(f"Missing archived trace directories: {task}")
        files = [p for task in TASKS for p in (batch / task).rglob("*") if p.is_file()]
        if len(files) != lock["expected_task_files"]:
            raise ValueError(f"Unexpected archived task file count: {len(files)}")
        provenance = batch / "provenance"
        if provenance.exists():
            raise ValueError("Archive unexpectedly includes derived provenance directory")
        provenance.mkdir()
        sidecars = []
        for asset in lock["assets"]:
            if asset["role"] == "published_clock_sidecar":
                sidecar = provenance / "published_clocks" / asset["name"]
                sidecar.parent.mkdir(exist_ok=True)
                shutil.copyfile(asset_dir / asset["name"], sidecar)
                sidecars.append({"task": asset["task"], "path": sidecar.relative_to(batch).as_posix(),
                                 "sha256": asset["sha256"], "verification": "published_asset_identity_only"})
        catalogue = recovered_configuration_catalogue(batch)
        config_path = provenance / "recovered_failure_configs.json"
        config_path.write_text(json.dumps(catalogue, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        lock_path = provenance / "release-lock.json"
        lock_path.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        receipt = {"receipt_version": 1, "status": "restored_verified_assets",
                   "release": lock["release"], "source_reference": lock["source_reference"],
                   "producer_revision": None, "producer_provenance_status": "not_established_by_release_assets",
                   "archive": next(asset for asset in lock["assets"] if asset["role"] == "archive"),
                   "task_file_count": len(files), "published_clock_sidecars": sidecars,
                   "recovered_configurations": {"path": config_path.relative_to(batch).as_posix(), "sha256": sha256_file(config_path)},
                   "release_lock_sha256": sha256_file(lock_path),
                   "limitations": ["Release tag/source reference is not proof of the exact producer working tree, model bytes, runtime or original calibration inputs.",
                                   "Published clock sidecars require independent per-run verification before onset scoring or pre-injection certification.",
                                   "Restoration does not establish independent physical outcomes, deterministic replay, or dataset acceptance."]}
        (provenance / "release.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if dataset_root.exists() or dataset_root.is_symlink():
            raise ValueError(f"Dataset root appeared during restoration: {dataset_root}")
        staging.rename(dataset_root)
        return dataset_root / lock["batch_name"]
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        publish_lock.unlink(missing_ok=True)
