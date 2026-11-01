"""Deterministic, physically interpretable perturbations of Panda commands."""
from dataclasses import dataclass
from math import cos, sin
import numpy as np
from telemetry_reasoning.schemas import FailureMode, Subtask, TASK_SUBTASKS, Task


@dataclass(frozen=True)
class FailureConfig:
    mode: FailureMode
    subtask: Subtask
    magnitude: float
    delay_s: float = 0.0
    axis: int = 0

    def __post_init__(self):
        if not np.isfinite(self.magnitude) or self.magnitude <= 0:
            raise ValueError('magnitude must be finite and positive')
        if self.delay_s < 0 or self.axis not in (0, 1, 2):
            raise ValueError('invalid delay or axis')


class FailureEngine:
    """Activated only in the requested canonical subtask after delay_s."""
    def __init__(self, config: FailureConfig):
        self.config = config
        self.stage = None
        self.stage_start = None
        self.activated = False
        self.first_activation_s = None
        self.activation_steps = 0
        self._active = False

    def set_context(self, stage: str | None, time_s: float):
        if stage != self.stage:
            self.stage = stage
            self.stage_start = time_s
        self._active = bool(
            stage == self.config.subtask.value
            and self.stage_start is not None
            and time_s - self.stage_start + 1e-9 >= self.config.delay_s
        )
        if self._active:
            self.activated = True
            self.activation_steps += 1
            if self.first_activation_s is None:
                self.first_activation_s = float(time_s)

    def transform_target(self, position, rotation, speed):
        pos = np.asarray(position, dtype=float).copy()
        rot = np.asarray(rotation, dtype=float).copy()
        if not self._active:
            return pos, rot, speed
        c = self.config
        if c.mode == FailureMode.POSITION_ERROR:
            pos[c.axis] += c.magnitude  # meters
        elif c.mode == FailureMode.ORIENTATION_ERROR:
            angle = c.magnitude  # radians
            C, S = cos(angle), sin(angle)
            if c.axis == 0:
                delta = np.array([[1, 0, 0], [0, C, -S], [0, S, C]])
            elif c.axis == 1:
                delta = np.array([[C, 0, S], [0, 1, 0], [-S, 0, C]])
            else:
                delta = np.array([[C, -S, 0], [S, C, 0], [0, 0, 1]])
            rot = delta @ rot
        elif c.mode == FailureMode.EXCESSIVE_SPEED:
            speed = speed * c.magnitude  # multiplicative speed factor
        return pos, rot, speed

    def transform_commands(self, data, stage):
        if not self._active or stage != self.config.subtask.value:
            return
        c = self.config
        if c.mode == FailureMode.INADEQUATE_GRIP_STRENGTH:
            # Panda actuator ctrl: 0=closed, 255=open. Limits closure.
            data.ctrl[7] = max(float(data.ctrl[7]), min(255.0, c.magnitude))
        elif c.mode == FailureMode.EARLY_GRIPPER_RELEASE:
            data.ctrl[7] = 255.0

    def diagnostics(self):
        return {
            'activated': self.activated,
            'first_activation_s': self.first_activation_s,
            'activation_steps': self.activation_steps,
        }


def validate_config(task: Task, config: FailureConfig):
    if config.subtask not in TASK_SUBTASKS[task]:
        raise ValueError(f'{config.subtask.value} does not belong to {task.value}')
    if config.mode == FailureMode.EXCESSIVE_SPEED and config.magnitude <= 1:
        raise ValueError('excessive_speed magnitude must exceed 1')
    if config.mode == FailureMode.INADEQUATE_GRIP_STRENGTH and config.magnitude > 255:
        raise ValueError('gripper command must be <= 255')
