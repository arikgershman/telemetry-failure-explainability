"""Evidence contracts for release certification, without importing MuJoCo.

Integrity and parser acceptance are deliberately separate from certification.
An evidence file is an audit receipt, bound to immutable source files; it cannot
turn an unverified clock estimate or a controller status into physical evidence.
"""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

TASKS = ("pick_up_object", "transport_object")
GATES = (
    "stored_trace_integrity_and_parser_compatibility",
    "independent_physical_outcomes",
    "complete_pre_injection_agreement",
    "deterministic_replay",
    "producer_provenance",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_source_path(batch: Path, relative: str) -> Path:
    """Only canonical, task-scoped relative POSIX paths can name sources."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"unsafe source path: {relative!r}")
    pure = PurePosixPath(relative)
    if (pure.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/"))
            or pure.parts[0] not in TASKS or len(pure.parts) < 2):
        raise ValueError(f"unsafe source path: {relative!r}")
    root = Path(batch).resolve()
    path = root.joinpath(*pure.parts)
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root and root in parent.parents):
        raise ValueError(f"symlink source path: {relative}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"unsafe source path: {relative}")
    return path


def inventory_sources(batch: Path) -> dict:
    files = {}
    for task in TASKS:
        folder = Path(batch) / task
        if folder.is_symlink():
            raise ValueError(f"symlink task directory: {task}")
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlink source path: {path.relative_to(batch)}")
            if path.is_file():
                relative = path.relative_to(batch).as_posix()
                safe_source_path(batch, relative)
                files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return files


def inventory_sha256(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def state(status: str, reason: str, **details) -> dict:
    return {"status": status, "reason": reason, **details}


def load_evidence(batch: Path, evidence: Path | dict | None) -> tuple[dict, dict, list[str]]:
    """Check source binding before any evidence is used. Never modify sources."""
    if evidence is None:
        return {}, {}, []
    try:
        document = (json.loads(Path(evidence).read_text(encoding="utf-8"))
                    if not isinstance(evidence, dict) else evidence)
        if not isinstance(document, dict):
            raise ValueError("certification evidence must be an object")
        if document.get("schema_version") != 1:
            raise ValueError("unsupported certification evidence schema_version")
        files = document.get("source_files")
        if not isinstance(files, dict) or not files:
            raise ValueError("evidence requires a nonempty source_files inventory")
        for relative, fingerprint in files.items():
            if (not isinstance(fingerprint, dict)
                    or not isinstance(fingerprint.get('size_bytes'), int)
                    or isinstance(fingerprint.get('size_bytes'), bool)
                    or fingerprint['size_bytes'] < 0
                    or not isinstance(fingerprint.get('sha256'), str)):
                raise ValueError(f"invalid evidence fingerprint: {relative}")
            path = safe_source_path(batch, relative)
            if (not path.is_file() or path.stat().st_size != fingerprint["size_bytes"]
                    or sha256_file(path) != fingerprint["sha256"]):
                raise ValueError(f"evidence source fingerprint mismatch: {relative}")
        expected_digest = document.get("source_inventory_sha256")
        if expected_digest is not None and expected_digest != inventory_sha256(files):
            raise ValueError("evidence source inventory digest mismatch")
        pairs = {}
        if not isinstance(document.get('pairs', []), list):
            raise ValueError("evidence pairs must be a list")
        if not isinstance(document.get('producer_provenance', {}), dict):
            raise ValueError("evidence producer_provenance must be an object")
        for pair in document.get("pairs", []):
            if not isinstance(pair, dict):
                raise ValueError("evidence pair must be an object")
            for key in ('nominal', 'injected'):
                run = pair.get(key, {})
                if not isinstance(run, dict) or any(not isinstance(run.get(field, {}), dict)
                                                   for field in ('origin', 'replay')):
                    raise ValueError(f"evidence {key}/origin/replay must be objects")
            key = (pair["task"], str(pair["pair_id"]))
            if key[0] not in TASKS or key in pairs:
                raise ValueError(f"invalid or duplicate evidence pair: {key}")
            # Bind metadata and diagnostics, not just the trace being replayed.
            for name in ("metadata.csv", "diagnostics.jsonl"):
                if f"{key[0]}/{name}" not in files:
                    raise ValueError(f"evidence missing identity binding: {key[0]}/{name}")
            pairs[key] = pair
        return document, pairs, []
    except (OSError, TypeError, KeyError, ValueError) as exc:
        return {}, {}, [str(exc)]


def replay_verified(run: dict, path: str, rows: list, files: dict) -> bool:
    replay = run.get("replay") or {}
    fingerprint = files.get(path, {})
    repeats = replay.get('repeats')
    comparisons = replay.get('repeat_comparisons')

    def complete_comparison(comparison):
        if not isinstance(comparison, dict):
            return False
        atol, rtol = comparison.get('atol'), comparison.get('rtol')
        return (
            comparison.get('matched') is True
            and comparison.get('all_rows') is True and comparison.get('all_fields') is True
            and comparison.get('row_count') == len(rows)
            and comparison.get('replayed_row_count') == len(rows)
            and comparison.get('identity_matched') is True
            and comparison.get('duration_matched') is True
            and comparison.get('injection_diagnostics_matched') is True
            and isinstance(atol, (int, float)) and not isinstance(atol, bool)
            and math.isfinite(atol) and 0 <= atol <= 1e-8
            and isinstance(rtol, (int, float)) and not isinstance(rtol, bool)
            and math.isfinite(rtol) and rtol == 0
        )

    return (bool(rows) and bool(fingerprint) and run.get('file') == path
            and run.get('source_sha256') == fingerprint.get('sha256')
            and type(repeats) is int and repeats >= 2
            and replay.get('fresh_repeatability_matched') is True
            and isinstance(comparisons, list) and len(comparisons) == repeats
            and complete_comparison(replay) and all(complete_comparison(item) for item in comparisons))


def replay_has_explicit_failure(run: dict) -> bool:
    """Distinguish a disproven replay from absent/incomplete audit evidence."""
    replay = run.get('replay') or {}
    comparisons = replay.get('repeat_comparisons', [])
    if not isinstance(comparisons, list):
        comparisons = []
    return any(isinstance(item, dict) and any(item.get(field) is False for field in (
        'matched', 'all_rows', 'all_fields', 'fresh_repeatability_matched',
        'identity_matched', 'duration_matched', 'injection_diagnostics_matched'))
        for item in [replay, *comparisons])


def verified_origin(diagnostic: dict, run: dict, replay_ok: bool) -> dict:
    recorded = diagnostic.get("telemetry_start_sim_s")
    alignment_source = diagnostic.get('clock_alignment_source')
    if (isinstance(recorded, (int, float)) and not isinstance(recorded, bool)
            and math.isfinite(recorded) and recorded >= 0
            and alignment_source in (None, 'recorded_at_generation', 'recorded_diagnostics')):
        return {"timestamp_s": float(recorded), "basis": "recorded_diagnostics", "verified": True}
    origin = run.get("origin") or {}
    value = origin.get("timestamp_s")
    verified = (origin.get("basis") == "verified_exact_replay" and origin.get("verified") is True
                and replay_ok and isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)
    if not origin and alignment_source is not None:
        return {'timestamp_s': recorded, 'basis': alignment_source, 'verified': False}
    return {**origin, "verified": verified}


def compare_pre_injection(nominal: list, injected: list, nominal_origin: dict,
                          injected_origin: dict, onset: float) -> dict:
    """Compare every available observation strictly before activation.

    Comparing the prefix is useful only when both independent clock origins are
    established. A nominal replay cannot establish an injected episode's origin.
    """
    if not nominal_origin.get("verified") or not injected_origin.get("verified"):
        return state("pending", "Both run origins require recorded or exact-replay verification.")
    if not isinstance(onset, (float, int)) or not math.isfinite(onset):
        return state("pending", "No finite activation timestamp.")
    a0, b0 = nominal_origin["timestamp_s"], injected_origin["timestamp_s"]
    if onset < b0 - 1e-8 or onset > b0 + injected[-1].timestamp + 0.022:
        return state("failed", "Activation lies outside the recorded run and final sampling interval.")
    if any(abs(rows[0].timestamp) > 1e-8 for rows in (nominal, injected)):
        return state("failed", "Canonical run time must start at zero.")
    before = [row for row in injected if b0 + row.timestamp < onset - 1e-9]
    if not before:
        return state("pending", "No stored observations strictly before activation.", compared_rows=0)
    # The complete prefix must begin at the same simulator instant. Merely
    # matching the overlap could hide an omitted beginning of either trajectory.
    if not math.isclose(a0, b0, abs_tol=1e-8, rel_tol=0):
        return state("failed", "Nominal and injected canonical origins differ.", compared_rows=0)
    for index, b in enumerate(before):
        if index >= len(nominal):
            return state("failed", "Nominal trajectory ends before injected pre-activation prefix.",
                         compared_rows=index)
        a = nominal[index]
        for field in type(a).model_fields:
            av, bv = getattr(a, field), getattr(b, field)
            mismatch = (av != bv if field == "subtask" else
                        np.shape(av) != np.shape(bv) or not np.allclose(av, bv, atol=1e-8, rtol=0))
            if mismatch:
                return state("failed", "Stored pre-injection trajectories disagree.",
                             compared_rows=index, first_mismatch={"row": index, "field": field,
                             "timestamp_s": b.timestamp})
    return state("passed", "Every stored pre-activation observation agrees across all fields.",
                 compared_rows=len(before), atol=1e-8, rtol=0,
                 coverage="stored_50hz_observations", activation_simulator_s=onset)


def producer_certification(evidence: dict) -> dict:
    provenance = evidence.get("producer_provenance") or {}
    revision = provenance.get("revision", "")
    hashes = provenance.get("source_files", provenance.get("source_sha256"))
    models = provenance.get("model_files", provenance.get("model_sha256"))
    # A retrospectively guessed commit cannot establish an old dirty tree.
    if (provenance.get("recorded_at_generation") is True and isinstance(revision, str)
            and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
            and isinstance(provenance.get("tree_clean"), bool)
            and isinstance(hashes, dict) and hashes
            and isinstance(models, dict) and models
            and all(isinstance(v, str) and len(v) == 64
                    and all(c in "0123456789abcdef" for c in v)
                    for v in [*hashes.values(), *models.values()])):
        return state("passed", "Producer revision and exact source/model hashes were recorded at generation.",
                     evidence=provenance)
    return state("pending", "Exact producer source/model/calibration state was not recorded at generation.",
                 available_evidence=provenance)


def summarize_certification(records: list[dict], fatal: list, evidence: dict) -> dict:
    gates = {}
    for name in GATES:
        if name == "producer_provenance":
            global_producer = producer_certification(evidence)
            if global_producer['status'] == 'passed':
                gates[name] = global_producer
            else:
                available = [r['certification'].get(name, {}).get('status') for r in records]
                gates[name] = (state('passed', 'Both run producers were recorded for every pair.')
                               if available and all(value == 'passed' for value in available)
                               else global_producer)
            continue
        counts = {key: sum(r["certification"][name]["status"] == key for r in records)
                  for key in ("passed", "pending", "failed")}
        status = ("failed" if counts["failed"] else
                  "passed" if records and counts["passed"] == len(records) else "pending")
        if name == GATES[0] and fatal:
            status = "failed"
        gates[name] = state(status, "Per-pair evidence coverage.", counts=counts)
    return {"certification_version": 1,
            "status": "certified" if not fatal and all(g["status"] == "passed" for g in gates.values())
                      else "failed" if fatal or any(g['status'] == 'failed' for g in gates.values()) else "pending",
            "gates": gates}
