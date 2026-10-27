"""Sample robot/object telemetry at 50Hz independently of physics timestep."""
import csv
from pathlib import Path
import numpy as np


class TelemetryLogger:
    def __init__(self, model, output, hz=50):
        self.model = model
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output.open('w', newline='')
        self.writer = None
        self.interval = 1.0 / hz
        self.next_sample = 0.0
        self.hand_id = model.body('hand').id
        self.cube_id = model.body('cube').id
        self.jids = [model.joint(f'joint{i}').id for i in range(1, 8)]
        self.qids = model.jnt_qposadr[self.jids]
        self.dids = model.jnt_dofadr[self.jids]

    def record(self, data, stage):
        if data.time + 1e-9 < self.next_sample:
            return
        row = {'time': round(float(data.time), 6), 'subtask': stage}
        for i, (q, d) in enumerate(zip(self.qids, self.dids), 1):
            row[f'joint{i}_position'] = float(data.qpos[q])
            row[f'joint{i}_velocity'] = float(data.qvel[d])
            row[f'joint{i}_actuator_force'] = float(data.qfrc_actuator[d])
            row[f'joint{i}_target'] = float(data.ctrl[i - 1])
        for prefix, xyz in (('hand', data.xpos[self.hand_id]),
                            ('cube', data.xpos[self.cube_id])):
            for axis, value in zip('xyz', xyz):
                row[f'{prefix}_{axis}'] = float(value)
        for i, value in enumerate(data.xquat[self.hand_id]):
            row[f'hand_quat_{i}'] = float(value)
        for i, value in enumerate(data.xquat[self.cube_id]):
            row[f'cube_quat_{i}'] = float(value)
        for axis, value in zip('xyz', data.cvel[self.cube_id, 3:]):
            row[f'cube_velocity_{axis}'] = float(value)
        row['gripper_command'] = float(data.ctrl[7])
        row['contact_count'] = int(data.ncon)
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.next_sample += self.interval

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
