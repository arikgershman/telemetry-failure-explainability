
"""Canonical 50 Hz MuJoCo telemetry logger."""

import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from telemetry_reasoning.schemas import RawTelemetryRow, Subtask


class TelemetryLogger:
    """Log canonical task subtasks, with timestamps relative to first sample."""

    def __init__(self, model, output, hz=50):
        self.model = model
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)

        self.file = self.output.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file, fieldnames=list(RawTelemetryRow.model_fields)
        )
        self.writer.writeheader()

        self.interval = 1.0 / hz
        self.origin = None
        self.sample_index = 0

        self.hand_id = model.body("hand").id
        self.cube_id = model.body("cube").id
        self.cube_geom_id = model.geom("cube_geom").id

        self.jids = np.array([
            model.joint(f"joint{i}").id for i in range(1, 8)
        ])
        self.qids = model.jnt_qposadr[self.jids]
        self.dids = model.jnt_dofadr[self.jids]

        # Panda's two finger slide joints.
        self.finger_jids = np.array([
            model.joint("finger_joint1").id,
            model.joint("finger_joint2").id,
        ])
        self.finger_qids = model.jnt_qposadr[self.finger_jids]
        self.finger_dids = model.jnt_dofadr[self.finger_jids]

        self.contact_buffer = np.zeros(6)

    def _contact_force(self, data):
        """Sum normal forces on contacts involving the cube, in N."""
        total = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            if self.cube_geom_id not in (contact.geom1, contact.geom2):
                continue
            self.contact_buffer.fill(0)
            mujoco.mj_contactForce(
                self.model, data, i, self.contact_buffer
            )
            total += abs(float(self.contact_buffer[0]))
        return total

    def _gripper_state(self, data):
        """Normalized closure: 0=open, 1=closed."""
        opening = float(np.sum(data.qpos[self.finger_qids]))
        # Panda finger travel is approximately 0 to 0.04 m each.
        return float(np.clip(1.0 - opening / 0.08, 0.0, 1.0))

    def _gripper_force(self, data):
        """Sum magnitudes of finger actuator generalized forces, in N."""
        return float(np.sum(np.abs(
            data.qfrc_actuator[self.finger_dids]
        )))

    def _make_row(self, data, stage, timestamp):
        # Position of the actual IK end-effector reference point.
        rot = data.xmat[self.hand_id].reshape(3, 3)
        ee_pos = (
            data.xpos[self.hand_id] +
            rot @ np.array([0.0, 0.0, 0.103])
        )

        # Object free-joint translational velocity in world coordinates.
        cube_joint = self.model.body("cube").jntadr[0]
        cube_dof = self.model.jnt_dofadr[cube_joint]
        obj_vel = data.qvel[cube_dof: cube_dof + 3]

        return RawTelemetryRow(
            timestamp=float(timestamp),
            subtask=Subtask(stage),
            joint_positions=data.qpos[self.qids].astype(float).tolist(),
            joint_velocities=data.qvel[self.dids].astype(float).tolist(),
            joint_torques=data.qfrc_actuator[self.dids].astype(float).tolist(),
            gripper_state=self._gripper_state(data),
            gripper_force=self._gripper_force(data),
            ee_pos=ee_pos.astype(float).tolist(),
            ee_quat=data.xquat[self.hand_id].astype(float).tolist(),
            obj_pos=data.xpos[self.cube_id].astype(float).tolist(),
            obj_quat=data.xquat[self.cube_id].astype(float).tolist(),
            obj_vel=obj_vel.astype(float).tolist(),
            contact_force=self._contact_force(data),
        )

    def record(self, data, stage):
        """Return True if a sample was recorded."""
        if stage not in Subtask._value2member_map_:
            return False

        if self.origin is None:
            self.origin = float(data.time)

        elapsed = float(data.time) - self.origin
        target = self.sample_index * self.interval

        if elapsed + 1e-8 < target:
            return False

        if abs(elapsed - target) > 1e-6:
            raise ValueError(
                f"Missed 50 Hz sample: elapsed={elapsed:.6f}, "
                f"expected={target:.6f}"
            )

        # Use the exact sample index to avoid floating-point timestamp drift.
        row = self._make_row(
            data, stage, round(target, 8)
        )
        record = row.model_dump(mode="json")

        for field in (
            "joint_positions", "joint_velocities", "joint_torques",
            "ee_pos", "ee_quat", "obj_pos", "obj_quat", "obj_vel",
        ):
            record[field] = json.dumps(
                record[field], separators=(",", ":"), allow_nan=False
            )

        self.writer.writerow(record)
        self.sample_index += 1
        return True

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
