
"""Damped least-squares IK with an optional failure-injection target transform."""
import numpy as np
import mujoco


class IKSolver:
    def __init__(self, model, offset=(0, 0, 0.103), max_speed=0.65):
        self.model = model
        self.hand_id = model.body('hand').id
        self.offset = np.asarray(offset, dtype=float)
        self.jids = np.array([model.joint(f'joint{i}').id for i in range(1, 8)])
        self.qids = model.jnt_qposadr[self.jids]
        self.dids = model.jnt_dofadr[self.jids]
        self.limits = model.jnt_range[self.jids]
        self.max_speed = max_speed
        self.target_q = None
        self.failure_engine = None

    def pose(self, data):
        rot = data.xmat[self.hand_id].reshape(3, 3).copy()
        return data.xpos[self.hand_id].copy() + rot @ self.offset, rot

    def move(self, data, target_pos, target_rot, dt):
        speed = self.max_speed
        if self.failure_engine is not None:
            target_pos, target_rot, speed = self.failure_engine.transform_target(
                target_pos, target_rot, speed
            )
        pos, rot = self.pose(data)
        ep = np.asarray(target_pos) - pos
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, (np.asarray(target_rot) @ rot.T).ravel())
        if q[0] < 0:
            q *= -1
        er = np.empty(3)
        mujoco.mju_quat2Vel(er, q, 1.0)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jp, jr, pos, self.hand_id)
        J = np.vstack((jp[:, self.dids], jr[:, self.dids]))
        e = np.r_[3.0 * ep, 2.0 * er]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.06**2 * np.eye(6), e)
        dq = np.clip(dq, -speed, speed)
        if self.target_q is None:
            self.target_q = data.qpos[self.qids].copy()
        self.target_q = np.clip(
            self.target_q + dq * dt,
            self.limits[:, 0] + 0.02,
            self.limits[:, 1] - 0.02,
        )
        data.ctrl[:7] = self.target_q
        return float(np.linalg.norm(ep)), float(np.linalg.norm(er))
