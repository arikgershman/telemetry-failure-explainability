# inverse kinematics: how to change the seven joint targets to reduce the gripper's position and orientation errors

import numpy as np
import mujoco


class IKSolver:
    """Damped least-squares Cartesian pose controller."""

    def __init__(self, model, hand_name="hand"):
        self.model = model
        self.hand_id = model.body(hand_name).id

        self.joint_ids = [
            model.joint(f"joint{i}").id
            for i in range(1, 8)
        ]

        self.qpos_ids = np.array([
            model.jnt_qposadr[j] for j in self.joint_ids
        ])
        self.dof_ids = np.array([
            model.jnt_dofadr[j] for j in self.joint_ids
        ])

        self.lower = model.jnt_range[self.joint_ids, 0]
        self.upper = model.jnt_range[self.joint_ids, 1]

        # Grasp center relative to the hand frame.
        self.grasp_offset = np.array([0.0, 0.0, 0.10])

    def grasp_pose(self, data):
        """Current grasp-center position and orientation."""
        rotation = data.xmat[self.hand_id].reshape(3, 3)
        position = (
            data.xpos[self.hand_id]
            + rotation @ self.grasp_offset
        )
        return position, rotation

    def step(self, data, target_pos, target_rot,
             gain=2.0, damping=0.08):
        """Compute updated joint targets toward a desired 6D pose."""

        mujoco.mj_forward(self.model, data)

        position, rotation = self.grasp_pose(data)

        # Translational error.
        pos_error = target_pos - position

        # Rotational error, expressed in world coordinates.
        relative_rot = target_rot @ rotation.T
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, relative_rot.ravel())

        rot_error = np.zeros(3)
        mujoco.mju_quat2Vel(rot_error, quat, 1.0)

        error = np.concatenate([pos_error, rot_error])

        # Jacobian at the grasp center.
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        mujoco.mj_jac(
            self.model,
            data,
            jac_pos,
            jac_rot,
            position,
            self.hand_id,
        )

        jacobian = np.vstack([
            jac_pos[:, self.dof_ids],
            jac_rot[:, self.dof_ids],
        ])

        # Damped least-squares inverse.
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T
            + damping**2 * np.eye(6),
            error,
        )

        # Limit commanded joint motion per simulation step.
        dq = np.clip(dq * gain * self.model.opt.timestep,
                     -0.005, 0.005)

        targets = data.qpos[self.qpos_ids] + dq
        targets = np.clip(targets, self.lower, self.upper)

        data.ctrl[:7] = targets

        return np.linalg.norm(pos_error), np.linalg.norm(rot_error)