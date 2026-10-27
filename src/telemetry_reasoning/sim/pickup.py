"""
- first controller, not yet a robust grasp planner
- grasp orientation and finger closure may need calibration against the Panda model
"""

import numpy as np
import mujoco

from telemetry_reasoning.sim.ik import IKSolver


class PickupController:
    """Approach -> align -> grasp -> lift."""

    def __init__(self, model, data):
        self.model = model
        self.ik = IKSolver(model)

        self.cube_id = model.body("cube").id
        self.stage = "approach"
        self.stage_start = data.time

        mujoco.mj_forward(model, data)

        self.cube_start = data.xpos[self.cube_id].copy()

        # Desired downward-facing gripper orientation.
        self.target_rot = np.diag([1.0, -1.0, -1.0])

        # Position targets relative to cube.
        self.approach_height = 0.18
        self.grasp_height = 0.005
        self.lift_height = 0.20

        # The gripper actuator uses a 0-255 command.
        self.gripper_open = 255.0
        self.gripper_closed = 0.0

        # Start from the model's home configuration.
        home = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if home >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home)
        else:
            mujoco.mj_resetData(model, data)

        mujoco.mj_forward(model, data)

        self.cube_start = data.xpos[self.cube_id].copy()
        data.ctrl[7] = self.gripper_open

    def transition(self, stage, data):
        self.stage = stage
        self.stage_start = data.time
        print(f"{data.time:.2f}s: {stage}")

    def update(self, data):
        cube_pos = data.xpos[self.cube_id].copy()
        elapsed = data.time - self.stage_start

        target = cube_pos.copy()

        if self.stage == "approach":
            target[2] += self.approach_height
            data.ctrl[7] = self.gripper_open

            pos_err, rot_err = self.ik.step(
                data, target, self.target_rot
            )

            if pos_err < 0.025 and rot_err < 0.15:
                self.transition("align", data)

        elif self.stage == "align":
            target[2] += self.grasp_height
            data.ctrl[7] = self.gripper_open

            pos_err, rot_err = self.ik.step(
                data, target, self.target_rot
            )

            if pos_err < 0.015 and rot_err < 0.12:
                self.transition("grasp", data)

        elif self.stage == "grasp":
            # Hold pose while closing the gripper.
            target[2] += self.grasp_height
            self.ik.step(data, target, self.target_rot)

            data.ctrl[7] = self.gripper_closed

            if elapsed > 1.0:
                self.lift_target = cube_pos.copy()
                self.lift_target[2] += self.lift_height
                self.transition("lift", data)

        elif self.stage == "lift":
            data.ctrl[7] = self.gripper_closed

            pos_err, rot_err = self.ik.step(
                data, self.lift_target, self.target_rot
            )

            if pos_err < 0.025 and rot_err < 0.15:
                cube_rise = (
                    data.xpos[self.cube_id, 2]
                    - self.cube_start[2]
                )

                if cube_rise > 0.10:
                    self.transition("success", data)
                    print("Pickup successful!")
                else:
                    self.transition("failed", data)
                    print("Pickup failed: cube was not lifted.")

        elif self.stage in ("success", "failed"):
            # Hold the final arm configuration.
            pass