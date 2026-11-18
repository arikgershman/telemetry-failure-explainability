
"""Freeze a dataset version with SHA-256 file checksums."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from telemetry_reasoning.sim.split_robot_dataset import TASKS, create_splits


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


def freeze_dataset(batch: Path, seed: int = 723, *, clock_sidecars=None) -> dict:
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

    for task in TASKS:
        task_dir = batch / task

        for path in sorted(task_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(batch).as_posix()
                files[relative] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }

    manifest = {
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
    if clock_sidecars is not None:
        from telemetry_reasoning.sim.clock_alignment import load_aligned_clocks

        clocks = {}
        for task, path in zip(TASKS, clock_sidecars, strict=True):
            pairs = load_aligned_clocks(batch / task, path)
            clocks[task] = {
                "source": path.name,
                "sha256": sha256_file(path),
                "pairs": pairs,
            }
        manifest["clock_alignment"] = clocks
        manifest["verification_scope"]["pre_injection_trajectory_agreement"] = (
            "compared_archived_samples_before_csv_injection_timestamp"
        )
    return manifest


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
            "pairs": {},
        }
        for task in TASKS
    }
    clocks = manifest.get("clock_alignment", {})
    for split, pairs in manifest["splits"]["splits"].items():
        for pair in pairs:
            task, pair_id = pair["task"], pair["pair_id"]
            record = records[task, pair_id]
            result[task]["pairs"][pair_id] = {
                **pair,
                "split": split,
                "injection_timestamp_s": record["injection_timestamp_s"],
                "injection_time_basis": record["injection_time_basis"],
                **clocks.get(task, {}).get("pairs", {}).get(pair_id, {}),
            }
    return result


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
        "--clock-sidecars", nargs=2, type=Path, metavar=("PICKUP", "TRANSPORT"),
        help="Track 1's aligned pickup and transport diagnostics, in that order",
    )
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
    args = parser.parse_args()

    if args.verify:
        if args.task_manifests_dir or args.clock_sidecars:
            parser.error("--task-manifests-dir and --clock-sidecars apply only when generating a manifest")
        manifest = json.loads(args.verify.read_text(encoding="utf-8"))
        errors = verify_dataset(args.batch, manifest)

        if errors:
            for error in errors[:20]:
                print(error)
            print(f"Verification failed: {len(errors)} issue(s)")
            raise SystemExit(1)

        print("Dataset verification passed!")
        return

    manifest = freeze_dataset(args.batch, seed=args.seed, clock_sidecars=args.clock_sidecars)
    args.output.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Recorded {len(manifest['files'])} files")
    print(f"Saved manifest: {args.output.resolve()}")
    if args.task_manifests_dir:
        args.task_manifests_dir.mkdir(parents=True, exist_ok=True)
        for task, task_manifest in task_manifests(manifest).items():
            path = args.task_manifests_dir / f"{task}.json"
            path.write_text(json.dumps(task_manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Saved task manifest: {path.resolve()}")


if __name__ == "__main__":
    main()
