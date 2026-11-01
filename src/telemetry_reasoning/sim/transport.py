
"""Physics-based Panda object transport and placement controller."""

import numpy as np

from telemetry_reasoning.sim.pickup import PickupController


class TransportController:
    """Pickup, maintain grasp, move along path, and place at target."""

    def __init__(self, model, data, timeout=45.0):
        self.model = model
        self.pickup = PickupController(
            model, data, grip_close=0.0
        )
        self.ik = self.pickup.ik
        self.cube_id = model.body("cube").id

        self.stage = "pickup"
        self.since = data.time
        self.start_time = data.time
        self.timeout = timeout
        self.reason = ""

        self.target_rot = self.pickup.target_rot.copy()
        self.displacement = np.array([0.0, 0.15, 0.0])

        self.target_cube_pos = None
        self.path = []
        self.waypoint_index = 0
        self.grasp_offset = None

        self.position_tolerance = 0.025
        self.grasp_tolerance = 0.045

        # Internal phases of reach_target.
        self.placement_phase = None
        self.phase_start = None
        self.lower_target = None
        self.retreat_target = None

        self.cube_half_height = 0.025
        self.table_height = 0.4

    @property
    def done(self):
        return self.stage in ("success", "failed")

    def change(self, stage, data, reason=""):
        self.stage = stage
        self.since = data.time
        self.reason = reason
        print(
            f"{data.time:6.2f}s  {stage}"
            + (f" ({reason})" if reason else "")
        )

    def change_phase(self, phase, data):
        self.placement_phase = phase
        self.phase_start = data.time
        print(f"{data.time:6.2f}s  placement: {phase}")

    def cube_position(self, data):
        return data.xpos[self.cube_id].copy()

    def grasp_is_stable(self, data):
        """Check whether the cube stays near its grasped position."""
        grasp_pos, _ = self.ik.pose(data)
        cube_pos = self.cube_position(data)

        current_offset = cube_pos - grasp_pos

        error = np.linalg.norm(
            current_offset - self.grasp_offset
        )
        return error < self.grasp_tolerance

    def initialize_transport(self, data):
        """Build the carrying path and destination."""
        cube_pos = self.cube_position(data)
        grasp_pos, _ = self.ik.pose(data)

        self.grasp_offset = cube_pos - grasp_pos

        self.target_cube_pos = cube_pos + self.displacement

        self.path = [
            grasp_pos + self.displacement * 0.5,
            grasp_pos + self.displacement,
        ]
        self.waypoint_index = 0

        # Lower the cube to just above the table surface.
        desired_cube_z = (
            self.table_height + self.cube_half_height + 0.002
        )

        lower_distance = cube_pos[2] - desired_cube_z

        self.lower_target = self.path[-1].copy()
        self.lower_target[2] -= max(0.0, lower_distance)

        # After release, lift the gripper clear of the cube.
        self.retreat_target = (
            self.lower_target + np.array([0.0, 0.0, 0.12])
        )

        # The final target is on the table, not in the air.
        self.target_cube_pos = np.array([
            cube_pos[0] + self.displacement[0],
            cube_pos[1] + self.displacement[1],
            desired_cube_z,
        ])

    def update(self, data):
        if self.done:
            return

        if data.time - self.start_time > self.timeout:
            self.change("failed", data, "timeout")
            return

        elapsed = data.time - self.since
        dt = self.model.opt.timestep

        if self.stage == "pickup":
            self.pickup.update(data)

            if self.pickup.stage == "success":
                self.initialize_transport(data)
                self.change("maintain_grasp", data)

            elif self.pickup.stage == "failed":
                self.change(
                    "failed", data,
                    f"pickup: {self.pickup.reason}"
                )
            return

        # Keep holding the cube until the release phase.
        if self.stage != "reach_target" or (
            self.placement_phase not in ("release", "retreat", "verify")
        ):
            data.ctrl[7] = self.pickup.grip_close

            if not self.grasp_is_stable(data):
                self.change("failed", data, "lost grasp")
                return

        if self.stage == "maintain_grasp":
            # Hold the arm's current position.
            grasp_pos, _ = self.ik.pose(data)
            self.ik.move(
                data, grasp_pos, self.target_rot, dt
            )

            if elapsed > 0.5:
                self.change("move_along_path", data)

        elif self.stage == "move_along_path":
            target = self.path[self.waypoint_index]

            pos_error, rot_error = self.ik.move(
                data, target, self.target_rot, dt
            )

            if pos_error < self.position_tolerance:
                self.waypoint_index += 1

                if self.waypoint_index >= len(self.path):
                    self.change("reach_target", data)
                    self.change_phase("lower", data)

        elif self.stage == "reach_target":
            self.update_placement(data, dt)

    def update_placement(self, data, dt):
        """Lower, release, retreat, and verify placement."""
        phase_elapsed = data.time - self.phase_start

        if self.placement_phase == "lower":
            data.ctrl[7] = self.pickup.grip_close

            pos_error, _ = self.ik.move(
                data, self.lower_target,
                self.target_rot, dt
            )

            if pos_error < 0.015:
                self.change_phase("release", data)

        elif self.placement_phase == "release":
            # Open the fingers while holding position.
            data.ctrl[7] = self.pickup.grip_open

            self.ik.move(
                data, self.lower_target,
                self.target_rot, dt
            )

            if phase_elapsed > 0.8:
                self.change_phase("retreat", data)

        elif self.placement_phase == "retreat":
            data.ctrl[7] = self.pickup.grip_open

            pos_error, _ = self.ik.move(
                data, self.retreat_target,
                self.target_rot, dt
            )

            if pos_error < 0.025:
                self.change_phase("verify", data)

        elif self.placement_phase == "verify":
            data.ctrl[7] = self.pickup.grip_open

            self.ik.move(
                data, self.retreat_target,
                self.target_rot, dt
            )

            cube_pos = self.cube_position(data)

            xy_error = np.linalg.norm(
                cube_pos[:2] - self.target_cube_pos[:2]
            )
            z_error = abs(
                cube_pos[2] - self.target_cube_pos[2]
            )

            if phase_elapsed > 0.5:
                if xy_error < 0.04 and z_error < 0.025:
                    self.change(
                        "success", data,
                        f"placement error={xy_error:.3f}m"
                    )
                else:
                    self.change(
                        "failed", data,
                        f"placement error={xy_error:.3f}m, "
                        f"height error={z_error:.3f}m"
                    )
