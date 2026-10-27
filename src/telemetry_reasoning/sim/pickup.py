"""Physics-based pickup state machine. No teleporting or synthetic successes."""
import numpy as np
import mujoco
from telemetry_reasoning.sim.ik import IKSolver


class PickupController:
    STAGES = ('approach', 'align', 'grasp', 'lift', 'success', 'failed')

    def __init__(self, model, data, timeout=18.0, grip_close=255.0):
        self.model = model
        self.ik = IKSolver(model)
        self.cube_id = model.body('cube').id
        self.timeout = timeout
        self.grip_open = 255.0
        self.grip_close = 0.0
        home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home)
        else:
            mujoco.mj_resetData(model, data)
        # The cube starts at rest on the table. Let it settle before commanding.
        mujoco.mj_forward(model, data)
        self.stage = 'settle'
        self.since = data.time
        self.start_time = data.time
        self.reason = ''
        self.cube_initial = data.xpos[self.cube_id].copy()
        self.cube_anchor = self.cube_initial.copy()
        self.lift_target = None
        self.align_target = None
        # Local gripper +z points down, with its fingers parallel to world x.
        self.target_rot = np.diag([1.0, -1.0, -1.0])
        data.ctrl[7] = self.grip_open

    @property
    def done(self):
        return self.stage in ('success', 'failed')

    def change(self, stage, data, reason=''):
        self.stage = stage
        self.since = data.time
        self.reason = reason
        print(f'{data.time:6.2f}s  {stage}' + (f' ({reason})' if reason else ''))

    def update(self, data):
        if self.done:
            return
        if data.time - self.start_time > self.timeout:
            self.change('failed', data, 'timeout')
            return
        cube = data.xpos[self.cube_id].copy()
        dt = self.model.opt.timestep
        elapsed = data.time - self.since
        if self.stage == 'settle':
            data.ctrl[7] = self.grip_open
            if elapsed > 0.5:
                self.cube_anchor = cube.copy()
                self.change('approach', data)
            return
        if self.stage == 'approach':
            data.ctrl[7] = self.grip_open
            target = self.cube_anchor + [0, 0, 0.18]
            pe, re = self.ik.move(data, target, self.target_rot, dt)
            if pe < 0.025 and re < 0.18:
                self.change('align', data)
        elif self.stage == 'align':
            data.ctrl[7] = self.grip_open
            # The grasp reference point is near the center of the fingertips.
            # For a 5cm cube, aim near its mid-height, not its top surface.
            target = self.cube_anchor.copy()
            self.align_target = target
            pe, re = self.ik.move(data, target, self.target_rot, dt)
            if pe < 0.012 and re < 0.12:
                self.change('grasp', data)
        elif self.stage == 'grasp':
            self.ik.move(data, self.align_target, self.target_rot, dt)
            data.ctrl[7] = self.grip_close
            if elapsed > 1.0:
                self.lift_target = self.align_target + [0, 0, 0.20]
                self.change('lift', data)
        elif self.stage == 'lift':
            data.ctrl[7] = self.grip_close
            pe, _ = self.ik.move(data, self.lift_target, self.target_rot, dt)
            rise = cube[2] - self.cube_anchor[2]
            if rise > 0.12 and pe < 0.04:
                self.change('success', data, f'cube rise={rise:.3f}m')
            elif elapsed > 5.0:
                self.change('failed', data, f'cube rise={rise:.3f}m')
