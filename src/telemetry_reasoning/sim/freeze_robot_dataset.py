
"""Freeze a dataset version with SHA-256 file checksums."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from telemetry_reasoning.sim.split_robot_dataset import create_splits


def sha256_file(path: Path) -> str:
    """Compute a file's SHA-256 hash without loading it all into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def source_provenance() -> dict:
    """Fingerprint the current validator inputs; not a claim of producer revision."""
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    paths = [*sorted((root / 'src/telemetry_reasoning/sim').glob('*.py')),
             *sorted((root / 'data/calibration').glob('*.csv')),
             *sorted((root / 'assets/scenes').glob('*.xml')),
             root / 'configs/default.yaml']
    return {'validation_revision': revision, 'python': platform.python_version(),
            'producer_revision': None,
            'producer_revision_note': 'Not recorded in released diagnostics; must be supplied by Track 1.',
            'source_sha256': {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}}


def freeze_dataset(batch: Path, seed: int = 723) -> dict:
    """Create a manifest recording splits and file fingerprints."""
    batch = batch.resolve()
    from telemetry_reasoning.sim.validate_dataset import validate_dataset
    validation = validate_dataset(batch)
    if validation['fatal_errors'] or validation['quarantined_pairs'] or validation['accepted_pairs'] != 1000:
        raise ValueError(
            f"Dataset cannot be frozen: {validation['accepted_pairs']} accepted, "
            f"{validation['quarantined_pairs']} quarantined; run audit_robot_dataset for details"
        )
    splits = create_splits(batch, seed=seed)

    files = {}

    for task in ("pick_up_object", "transport_object"):
        task_dir = batch / task

        for path in sorted(task_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(batch).as_posix()
                files[relative] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }

    return {
        "manifest_version": 1,
        "hash_algorithm": "sha256",
        "provenance": source_provenance(),
        "split_seed": seed,
        "validation": validation,
        "verification_scope": {
            "stored_trace_integrity_and_parser_compatibility": "checked",
            "independent_physical_outcomes": "not_certified_by_this_manifest",
            "deterministic_replay": "requires_separate_real_simulation_evidence",
            "pre_injection_trajectory_agreement": "requires_producer_task_time_origin",
        },
        "metadata_layout": {task: f'{task}/metadata.csv' for task in ('pick_up_object', 'transport_object')},
        "pair_identity": "(task, pair_id); IDs are task-local; parse tasks into separate output directories",
        "excluded_failure_modes": {"excessive_speed": "insufficient failure reliability in producer calibration"},
        "splits": splits,
        "files": files,
    }


def verify_dataset(batch: Path, manifest: dict) -> list[str]:
    """Report missing, changed, or unexpected dataset files."""
    batch = batch.resolve()
    errors = []
    expected = set(manifest["files"])
    for relative in sorted(expected):
        resolved = (batch / relative).resolve()
        if (not resolved.is_relative_to(batch)
                or not any(resolved.is_relative_to(batch / task)
                           for task in ('pick_up_object', 'transport_object'))):
            errors.append(f"Unsafe inventory path: {relative}")
    if errors:
        return errors

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
        recorded = manifest["files"][relative]

        if path.stat().st_size != recorded["size_bytes"]:
            errors.append(f"Size changed: {relative}")
        elif sha256_file(path) != recorded["sha256"]:
            errors.append(f"Checksum changed: {relative}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=723)
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify the dataset against an existing manifest",
    )
    args = parser.parse_args()

    if args.verify:
        manifest = json.loads(args.verify.read_text(encoding="utf-8"))
        errors = verify_dataset(args.batch, manifest)

        if errors:
            for error in errors[:20]:
                print(error)
            print(f"Verification failed: {len(errors)} issue(s)")
            raise SystemExit(1)

        print("Dataset verification passed!")
        return

    manifest = freeze_dataset(args.batch, seed=args.seed)
    args.output.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Recorded {len(manifest['files'])} files")
    print(f"Saved manifest: {args.output.resolve()}")


if __name__ == "__main__":
    main()
