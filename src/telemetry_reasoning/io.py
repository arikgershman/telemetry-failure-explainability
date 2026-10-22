"""Canonical CSV encoding: one JSON array cell per vector, scalar numbers otherwise."""

import csv
import json
from math import isclose
from pathlib import Path
from typing import Sequence

from telemetry_reasoning.schemas import (
    LOGGING_DT_S, MetadataRow, RawTelemetryRow, validate_observation_sequence,
)

VECTOR_COLUMNS = (
    "joint_positions", "joint_velocities", "joint_torques", "ee_pos", "ee_quat",
    "obj_pos", "obj_quat", "obj_vel",
)
SCALAR_COLUMNS = ("timestamp", "gripper_state", "gripper_force", "contact_force")


def _read_csv(path: str | Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if (reader.fieldnames is None or len(reader.fieldnames) != len(columns)
                or set(reader.fieldnames) != set(columns)):
            raise ValueError(f"{path}: expected exactly these CSV columns: {list(columns)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV contains no observations")
    return rows


def validate_raw_run(rows: Sequence[RawTelemetryRow]) -> None:
    validate_observation_sequence(rows)
    if any(not isclose(b.timestamp - a.timestamp, LOGGING_DT_S, rel_tol=0, abs_tol=1e-8)
           for a, b in zip(rows, rows[1:])):
        raise ValueError("raw telemetry must be sampled every 0.02 seconds (50 Hz)")


def read_telemetry_csv(path: str | Path) -> list[RawTelemetryRow]:
    rows = []
    for line, record in enumerate(_read_csv(path, tuple(RawTelemetryRow.model_fields)), start=2):
        try:
            data = dict(record)
            for column in VECTOR_COLUMNS:
                data[column] = json.loads(data[column])
            for column in SCALAR_COLUMNS:
                data[column] = float(data[column])
            rows.append(RawTelemetryRow.model_validate(data))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line}: invalid telemetry row: {exc}") from exc
    validate_raw_run(rows)
    return rows


def write_telemetry_csv(path: str | Path, rows: Sequence[RawTelemetryRow]) -> None:
    validate_raw_run(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RawTelemetryRow.model_fields))
        writer.writeheader()
        for row in rows:
            record = row.model_dump(mode="json")
            for column in VECTOR_COLUMNS:
                record[column] = json.dumps(record[column], allow_nan=False, separators=(",", ":"))
            writer.writerow(record)


def read_metadata_csv(path: str | Path) -> list[MetadataRow]:
    rows = []
    for line, record in enumerate(_read_csv(path, tuple(MetadataRow.model_fields)), start=2):
        try:
            data = dict(record)
            data["perturbation_value"] = float(data["perturbation_value"])
            data["seed"] = int(data["seed"])
            rows.append(MetadataRow.model_validate(data))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line}: invalid metadata row: {exc}") from exc
    if len({row.pair_id for row in rows}) != len(rows):
        raise ValueError("metadata pair_id values must be unique")
    return rows


def write_metadata_csv(path: str | Path, rows: Sequence[MetadataRow]) -> None:
    if not rows or len({row.pair_id for row in rows}) != len(rows):
        raise ValueError("metadata needs at least one row and unique pair_id values")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(MetadataRow.model_fields))
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in rows)
