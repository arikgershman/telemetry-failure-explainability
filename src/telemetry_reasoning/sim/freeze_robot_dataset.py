
"""Freeze a dataset version with SHA-256 file checksums."""

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from telemetry_reasoning.sim.split_robot_dataset import TASKS, create_splits, verify_splits
from telemetry_reasoning.sim.dataset_certification import (
    inventory_sha256, inventory_sources, safe_source_path, sha256_file,
)


def source_provenance() -> dict:
    """Fingerprint the current validator inputs; not a claim of producer revision."""
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    paths = [*sorted((root / 'src/telemetry_reasoning').rglob('*.py')),
             *sorted((root / 'data/calibration').glob('*.csv')),
             *sorted(p for p in (root / 'assets').rglob('*') if p.is_file()),
             *sorted((root / 'configs').glob('*.yaml'))]
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip())
    return {'validation_revision': revision, 'python': platform.python_version(),
            'validation_worktree_dirty': dirty,
            'calibration_inputs_status': 'available' if list((root / 'data/calibration').glob('*.csv')) else 'missing',
            'producer_revision': None,
            'producer_revision_note': 'Not recorded in released diagnostics; must be supplied by Track 1.',
            'source_sha256': {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}}


def freeze_dataset(batch: Path, seed: int = 723, *, evidence: Path | dict | None = None,
                   require_certified: bool = False) -> dict:
    """Create a manifest recording splits and file fingerprints."""
    batch = batch.resolve()
    from telemetry_reasoning.sim.validate_dataset import validate_dataset
    before = inventory_sources(batch)
    validation = validate_dataset(batch, evidence=evidence)
    if validation['fatal_errors'] or validation['quarantined_pairs'] or validation['accepted_pairs'] != 1000:
        raise ValueError(
            f"Dataset cannot be frozen: {validation['accepted_pairs']} accepted, "
            f"{validation['quarantined_pairs']} quarantined; run audit_robot_dataset for details"
        )
    splits = create_splits(batch, seed=seed)
    certification = validation['certification']
    if require_certified and certification['status'] != 'certified':
        pending = [name for name, gate in certification['gates'].items() if gate['status'] != 'passed']
        raise ValueError(f"Dataset is parser-ready but not release-certified: {', '.join(pending)}")
    files = inventory_sources(batch)
    if before != files:
        raise ValueError("Dataset changed during validation; no manifest was frozen")

    return {
        "manifest_version": 2,
        "hash_algorithm": "sha256",
        "provenance": source_provenance(),
        "split_seed": seed,
        "validation": validation,
        "certification": certification,
        "release_status": ('certified' if certification['status'] == 'certified' else
                           'parser_ready_certification_failed' if certification['status'] == 'failed' else
                           'parser_ready_pending_certification'),
        "verification_scope": {key: value['status'] for key, value in certification['gates'].items()},
        "metadata_layout": {task: f'{task}/metadata.csv' for task in ('pick_up_object', 'transport_object')},
        "pair_identity": "(task, pair_id); IDs are task-local; parse tasks into separate output directories",
        "excluded_failure_modes": {"excessive_speed": "insufficient failure reliability in producer calibration"},
        "splits": splits,
        "files": files,
        "source_inventory_sha256": inventory_sha256(files),
        "certification_evidence": (
            {"path": str(Path(evidence).resolve()), "sha256": sha256_file(Path(evidence))}
            if isinstance(evidence, (Path, str)) else
            {"inline_sha256": inventory_sha256(evidence)} if isinstance(evidence, dict) else None
        ),
    }


def task_manifests(manifest: dict) -> dict[str, dict]:
    """Project a release manifest into the pair-ID layout used downstream."""
    validation = manifest["validation"]
    records = {(record["task"], record["pair_id"]): record for record in validation["pairs"]}
    result = {
        task: {
            "dataset_kind": "mujoco",
            "task": task,
            "metadata": manifest["metadata_layout"][task],
            "split_seed": manifest["split_seed"],
            "verification_scope": manifest["verification_scope"],
            "release_status": manifest.get("release_status", "legacy_unclassified"),
            "pairs": {},
        }
        for task in TASKS
    }
    for split, pairs in manifest["splits"]["splits"].items():
        for pair in pairs:
            task, pair_id = pair["task"], pair["pair_id"]
            record = records[task, pair_id]
            result[task]["pairs"][pair_id] = {
                **pair,
                "split": split,
                "injection_timestamp_s": record["injection_timestamp_s"],
                "injection_time_basis": record["injection_time_basis"],
            }
            # Evaluator consumes this unsuffixed key as CSV-relative time. Never
            # expose absolute simulator time under that key.
            if record.get("injection_conversion_verified") is True:
                result[task]["pairs"][pair_id].update({
                    "injection_timestamp": record["injection_timestamp"],
                    "injection_timestamp_basis": record["injection_timestamp_basis"],
                    "injection_conversion_verified": True,
                })
            if "certification" in record:
                result[task]["pairs"][pair_id]["certification"] = record["certification"]
    return result


def verify_dataset(batch: Path, manifest: dict) -> list[str]:
    """Report missing, changed, or unexpected dataset files."""
    batch = batch.resolve()
    errors = []
    if not isinstance(manifest, dict):
        return ['Dataset manifest must be an object']
    inventory = manifest.get("files")
    if not isinstance(inventory, dict):
        return ["Manifest files inventory must be an object"]
    if manifest.get('manifest_version', 1) not in (1, 2):
        return ["Unsupported dataset manifest_version"]
    # The Windows-generated v1 release used backslashes in its inventory. Read
    # those archived manifests without propagating platform-dependent paths
    # into any newly generated handoff.
    if manifest.get("manifest_version") == 1:
        normalized = {name.replace('\\', '/'): value for name, value in inventory.items()}
        if len(normalized) != len(inventory):
            return ["Duplicate inventory paths after legacy path normalization"]
        inventory = normalized
    expected = set(inventory)
    for relative in sorted(expected):
        try:
            safe_source_path(batch, relative)
        except ValueError:
            errors.append(f"Unsafe inventory path: {relative}")
    if errors:
        return errors
    if (manifest.get("source_inventory_sha256") is not None
            and manifest['source_inventory_sha256'] != inventory_sha256(manifest['files'])):
        return ["Manifest source inventory digest changed"]

    actual = {
        path.relative_to(batch).as_posix()
        for task in ("pick_up_object", "transport_object")
        for path in (batch / task).rglob("*")
        if path.is_file()
    }

    for relative in sorted(expected - actual):
        errors.append(f"Missing: {relative}")

    for relative in sorted(actual - expected):
        errors.append(f"Unexpected: {relative}")

    for relative in sorted(expected & actual):
        path = batch / relative
        recorded = inventory[relative]
        if (not isinstance(recorded, dict) or set(recorded) != {"size_bytes", "sha256"}
                or not isinstance(recorded['size_bytes'], int) or isinstance(recorded['size_bytes'], bool)
                or recorded['size_bytes'] < 0 or not isinstance(recorded['sha256'], str)
                or len(recorded['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in recorded['sha256'])):
            errors.append(f"Invalid fingerprint: {relative}")
            continue

        if path.stat().st_size != recorded["size_bytes"]:
            errors.append(f"Size changed: {relative}")
        elif sha256_file(path) != recorded["sha256"]:
            errors.append(f"Checksum changed: {relative}")

    if manifest.get('manifest_version', 1) >= 2:
        if 'splits' not in manifest or 'split_seed' not in manifest:
            errors.append('Version 2 manifest requires grouped splits and split_seed')
        else:
            errors.extend(verify_splits(batch, manifest['splits'], manifest['split_seed']))
    return errors


def is_release_certified(manifest: dict) -> bool:
    """A headline marker cannot override incomplete or failing gate evidence."""
    from telemetry_reasoning.sim.dataset_certification import GATES
    if not isinstance(manifest, dict):
        return False
    certification = manifest.get('certification', {})
    validation = manifest.get('validation', {})
    if not isinstance(certification, dict) or not isinstance(validation, dict):
        return False
    gates = certification.get('gates', {})
    if not isinstance(gates, dict) or any(not isinstance(gates.get(name), dict) for name in GATES):
        return False
    return (certification.get('status') == 'certified'
            and all(gates.get(name, {}).get('status') == 'passed' for name in GATES)
            and validation.get('accepted_pairs') == 1000
            and validation.get('quarantined_pairs') == 0
            and not validation.get('fatal_errors'))


def _write_json(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish_handoff(destination: Path, manifest: dict) -> Path:
    """Publish the complete immutable handoff directory with one atomic rename.

    The destination must not exist. Staging on its filesystem ensures readers
    see either no handoff or a complete manifest, splits, and both task views.
    """
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite handoff: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.publish.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    staged = None
    try:
        staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to overwrite handoff: {destination}")
        manifest = copy.deepcopy(manifest)
        evidence = manifest.get('certification_evidence')
        if evidence and evidence.get('path'):
            source = Path(evidence['path'])
            if sha256_file(source) != evidence['sha256']:
                raise ValueError('Certification evidence changed after validation')
            shutil.copyfile(source, staged / 'certification_evidence.json')
            if sha256_file(staged / 'certification_evidence.json') != evidence['sha256']:
                raise ValueError('Certification evidence changed while publishing')
            evidence['bundle_file'] = 'certification_evidence.json'
        _write_json(staged / "dataset_manifest.json", manifest)
        _write_json(staged / "dataset_splits.json", manifest["splits"])
        tasks = staged / "tasks"
        tasks.mkdir()
        for task, document in task_manifests(manifest).items():
            _write_json(tasks / f"{task}.json", document)
        files = {path.relative_to(staged).as_posix():
                 {'sha256': sha256_file(path), 'size_bytes': path.stat().st_size}
                 for path in sorted(staged.rglob('*.json'))}
        _write_json(staged / 'publication_receipt.json', {
            'publication_version': 1, 'release_status': manifest.get('release_status'),
            'files': files, 'inventory_sha256': inventory_sha256(files),
        })
        # Cooperating publishers serialize through the exclusive sibling lock;
        # the destination itself becomes visible only after it is complete.
        os.rename(staged, destination)
        return destination / "dataset_manifest.json"
    finally:
        if staged is not None and staged.exists():
            shutil.rmtree(staged)
        lock.unlink(missing_ok=True)


def verify_handoff(destination: Path) -> list[str]:
    """Verify all generated handoff artifacts as well as the source inventory."""
    destination = Path(destination)
    try:
        if destination.is_symlink() or (destination / 'publication_receipt.json').is_symlink():
            raise ValueError('symlink handoff directory or receipt')
        receipt = json.loads((destination / 'publication_receipt.json').read_text(encoding='utf-8'))
        files = receipt['files']
        if receipt['publication_version'] != 1 or inventory_sha256(files) != receipt['inventory_sha256']:
            raise ValueError('publication receipt inventory changed')
        required = {'dataset_manifest.json', 'dataset_splits.json',
                    *(f'tasks/{task}.json' for task in TASKS)}
        allowed = required | {'certification_evidence.json'}
        if not required <= set(files) or not set(files) <= allowed:
            raise ValueError('publication receipt has missing or unsafe artifact paths')
        actual = {p.relative_to(destination).as_posix()
                  for p in destination.rglob('*') if p.is_file()}
        if actual != set(files) | {'publication_receipt.json'}:
            raise ValueError('publication artifact set differs from receipt')
        errors = []
        for name, recorded in files.items():
            path = destination / name
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                        if destination in parent.parents):
                errors.append(f'Symlink handoff artifact: {name}')
            elif path.stat().st_size != recorded['size_bytes'] or sha256_file(path) != recorded['sha256']:
                errors.append(f'Handoff artifact changed: {name}')
        return errors
    except (OSError, TypeError, KeyError, ValueError) as exc:
        return [f'Invalid handoff publication: {exc}']


def write_manifest(path: Path, document: dict) -> None:
    """Atomically publish one JSON file and refuse all existing destinations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    staged = Path(temporary)
    try:
        staged.unlink()
        _write_json(staged, document)
        # link is an atomic create-if-absent operation on this filesystem.
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=723)
    parser.add_argument("--evidence", type=Path, help="Source-bound replay/clock/producer evidence JSON")
    parser.add_argument("--require-certified", action="store_true", help="Refuse a parser-ready handoff with pending certification gates")
    parser.add_argument("--handoff-dir", type=Path, help="Publish a complete new handoff directory atomically; refuses overwrite")
    parser.add_argument(
        "--task-manifests-dir",
        type=Path,
        help="Write one parser/evaluator manifest per task, using original pair IDs",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify the dataset against an existing manifest",
    )
    args = parser.parse_args(argv)
    if args.handoff_dir and (args.output or args.task_manifests_dir or args.verify):
        parser.error("--handoff-dir cannot be combined with --output, --task-manifests-dir, or --verify")

    if args.verify:
        if args.task_manifests_dir or args.evidence:
            parser.error("--task-manifests-dir and --evidence are only available when generating a manifest")
        manifest = json.loads(args.verify.read_text(encoding="utf-8"))
        errors = verify_dataset(args.batch, manifest)
        if (args.verify.parent / 'publication_receipt.json').exists():
            errors.extend(verify_handoff(args.verify.parent))
        if args.require_certified and not is_release_certified(manifest):
            errors.append('Manifest is not release-certified')

        if errors:
            for error in errors[:20]:
                print(error)
            print(f"Verification failed: {len(errors)} issue(s)")
            return 1

        print("Dataset verification passed!")
        return 0

    manifest = freeze_dataset(args.batch, seed=args.seed, evidence=args.evidence,
                              require_certified=args.require_certified)
    if args.handoff_dir:
        output = publish_handoff(args.handoff_dir, manifest)
        print(f"Saved complete handoff: {output.resolve()}")
        print(f"Release status: {manifest['release_status']}")
        return 0
    output = args.output or Path("dataset_manifest.json")
    # Check every legacy output before publishing the first file. Prefer
    # --handoff-dir when consumers require a transaction spanning all files.
    destinations = [output]
    if args.task_manifests_dir:
        destinations.extend(args.task_manifests_dir / f"{task}.json" for task in TASKS)
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("Refusing to overwrite an existing manifest; choose a new handoff path")

    print(f"Recorded {len(manifest['files'])} files")
    if args.task_manifests_dir:
        args.task_manifests_dir.mkdir(parents=True, exist_ok=True)
        for task, task_manifest in task_manifests(manifest).items():
            path = args.task_manifests_dir / f"{task}.json"
            write_manifest(path, task_manifest)
            print(f"Saved task manifest: {path.resolve()}")
    write_manifest(output, manifest)
    print(f"Saved manifest: {output.resolve()}")
    print(f"Release status: {manifest['release_status']}")
    return 0


def main(argv=None):
    """Return a clean CLI failure for expected input/setup/certification errors."""
    try:
        return _main(argv)
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Dataset freeze failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
