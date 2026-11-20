"""Pure validation for the optional 500 Hz command-hook boundary.

The hook receives a detached dictionary. ``ee_pos`` is world metres;
``ee_quat`` is a unit world quaternion in MuJoCo's wxyz convention;
``gripper_state`` is commanded closure (0=open, 1=closed);
``gripper_force`` is a total finger generalized actuator-force *limit* in N,
not measured force or the Panda's dimensionless 0..255 position control;
``speed_scale`` multiplies the nominal joint-velocity limit. Returning None
retains the original command, even if the supplied dictionary was mutated.
"""

from collections.abc import Mapping
from copy import deepcopy
import numpy as np


COMMAND_FIELDS = frozenset({'ee_pos', 'ee_quat', 'gripper_state',
                            'gripper_force', 'speed_scale'})


def validate_command(command, *, maximum_gripper_force):
    """Return a detached canonical command or fail before applying any override.

    All fields are required. Unknown fields are rejected to expose misspelled
    perturbations. Force may reduce the hardware limit but cannot exceed it.
    Quaternion norms must already be within 1e-6 of one; accepted values are
    retained exactly so an identity hook does not perturb numerical replay.
    """
    if not isinstance(command, Mapping) or set(command) != COMMAND_FIELDS:
        raise ValueError(f'control command must contain exactly {sorted(COMMAND_FIELDS)}')
    result = {}
    for name, length in (('ee_pos', 3), ('ee_quat', 4)):
        raw = np.asarray(command[name])
        if raw.dtype.kind not in 'iuf':
            raise ValueError(f'{name} must contain {length} finite real numbers')
        value = raw.astype(float)
        if value.shape != (length,) or not np.all(np.isfinite(value)):
            raise ValueError(f'{name} must contain {length} finite values')
        result[name] = value.tolist()
    if abs(float(np.linalg.norm(result['ee_quat'])) - 1.0) > 1e-6:
        raise ValueError('ee_quat must be a unit quaternion in wxyz order')
    for name in ('gripper_state', 'gripper_force', 'speed_scale'):
        value = command[name]
        if isinstance(value, (bool, str, bytes)) or not np.isscalar(value):
            raise ValueError(f'{name} must be a finite number')
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
        result[name] = value
    if not 0.0 <= result['gripper_state'] <= 1.0:
        raise ValueError('gripper_state closure must be in [0, 1]')
    if not 0.0 <= result['gripper_force'] <= maximum_gripper_force + 1e-9:
        raise ValueError('gripper_force must be between zero and the nominal hardware limit in N')
    if result['speed_scale'] <= 0.0:
        raise ValueError('speed_scale must be positive')
    return result


def invoke_control_hook(hook, timestamp, subtask, nominal_command):
    """Invoke once with detached input, validating the complete returned command."""
    nominal = validate_command(nominal_command,
                               maximum_gripper_force=nominal_command['gripper_force'])
    returned = hook(float(timestamp), str(subtask), deepcopy(nominal))
    if returned is None:
        return nominal
    return validate_command(returned, maximum_gripper_force=nominal['gripper_force'])
