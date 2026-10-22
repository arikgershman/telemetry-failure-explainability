"""Generate an eight-second synthetic contract fixture, without physics or ML."""

import argparse
from pathlib import Path

import numpy as np

from telemetry_reasoning.config import ProjectConfig, load_config
from telemetry_reasoning.io import read_metadata_csv, write_metadata_csv, write_telemetry_csv
from telemetry_reasoning.parser import build_paired_bt
from telemetry_reasoning.schemas import (
    TASK_SUBTASKS, FailureMode, MetadataRow, PairedBehaviorTree, RawTelemetryRow,
    Subtask, Task,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_mock_rows(config: ProjectConfig, seed: int) -> tuple[list[RawTelemetryRow], list[RawTelemetryRow]]:
    """Use identical starting conditions, then lower grasp force in the failure."""
    rng = np.random.default_rng(seed)
    dt = config.simulation.logging_dt_s
    count = round(config.simulation.mock_duration_s / dt)
    if count < 8:
        raise ValueError("mock data needs at least two samples per subtask")
    segments = np.array_split(np.arange(count), 4)
    success, failure = [], []
    for subtask, indices in zip(TASK_SUBTASKS[Task.PICK_UP_OBJECT], segments):
        for offset, index in enumerate(indices):
            timestamp = round(int(index) * dt, 10)
            progress = offset / (len(indices) - 1)
            joints = [round(float(0.2 * np.sin(timestamp + j)), 6) for j in range(6)]
            velocities = [round(float(0.2 * np.cos(timestamp + j)), 6) for j in range(6)]
            torques = [round(float(0.5 * np.sin(timestamp + j)), 6) for j in range(6)]
            force = round(float(0.2 + rng.uniform(-0.02, 0.02)), 6)
            closed = 0.0
            ee_pos = [0.4, 0.0, 0.1]
            obj_pos = [0.4, 0.0, 0.05]
            obj_vel = [0.0, 0.0, 0.0]
            if subtask == Subtask.APPROACH:
                ee_pos = [round(0.1 + 0.3 * progress, 6), 0.0, 0.2]
            elif subtask == Subtask.ALIGN:
                ee_pos[2] = round(0.2 - 0.1 * progress, 6)
            elif subtask == Subtask.GRASP and progress >= 0.2:
                closed, force = 1.0, 15.0
            elif subtask == Subtask.LIFT:
                closed, force = 1.0, 15.0
                ee_pos[2] = round(0.1 + 0.2 * progress, 6)
                obj_pos[2] = round(0.05 + 0.2 * progress, 6)
                obj_vel[2] = round(0.2 / ((len(indices) - 1) * dt), 6)
            row = RawTelemetryRow(
                timestamp=timestamp, subtask=subtask,
                joint_positions=joints, joint_velocities=velocities, joint_torques=torques,
                gripper_state=closed, gripper_force=force,
                ee_pos=ee_pos, ee_quat=[1.0, 0.0, 0.0, 0.0],
                obj_pos=obj_pos, obj_quat=[1.0, 0.0, 0.0, 0.0], obj_vel=obj_vel,
                contact_force=round(force * 0.8, 6) if closed else 0.0,
            )
            success.append(row)
            failed_data = row.model_dump()
            # Abrupt force loss midway through grasp exercises event retention.
            # The failed gripper stays closed but the object remains on support.
            if (subtask == Subtask.GRASP and progress >= 0.55) or subtask == Subtask.LIFT:
                failed_data.update(gripper_force=3.0, contact_force=2.4,
                                   obj_pos=[0.4, 0.0, 0.05], obj_vel=[0.0, 0.0, 0.0])
            failure.append(RawTelemetryRow.model_validate(failed_data))
    return success, failure


def generate_mock_pair(output_dir: Path, config: ProjectConfig, seed: int = 723) -> PairedBehaviorTree:
    metadata = MetadataRow(
        pair_id="000", task=Task.PICK_UP_OBJECT,
        success_file="success_000.csv", failure_file="failure_000.csv",
        failed_subtask=Subtask.GRASP, failure_mode=FailureMode.INADEQUATE_GRIP_STRENGTH,
        perturbation_param="gripper_force", perturbation_value=3.0, seed=seed,
    )
    success, failure = make_mock_rows(config, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_telemetry_csv(output_dir / metadata.success_file, success)
    write_telemetry_csv(output_dir / metadata.failure_file, failure)
    write_metadata_csv(output_dir / "metadata.csv", [metadata])
    # Read the serialized metadata and telemetry back through the public APIs.
    metadata = read_metadata_csv(output_dir / "metadata.csv")[0]
    tree = build_paired_bt(metadata, output_dir, config.parser)
    (output_dir / "pair_000_bt.json").write_text(tree.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/default.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/sample")
    parser.add_argument("--seed", type=int, default=723)
    args = parser.parse_args()
    config = load_config(args.config)
    tree = generate_mock_pair(args.output_dir, config, args.seed)
    count = round(config.simulation.mock_duration_s / config.simulation.logging_dt_s)
    print(f"Wrote {count} rows per run to {args.output_dir.resolve()}")
    for node in tree.subtasks:
        print(f"  {node.subtask.value}: success={len(node.success_samples)}, failure={len(node.failure_samples)}")
    print(f"Validated BT: {args.output_dir / 'pair_000_bt.json'}")


if __name__ == "__main__":
    main()
