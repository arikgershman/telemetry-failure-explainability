"""Independent, simulator-free task outcome measurements.

These criteria deliberately do not inspect controller state, failure labels, or
reason strings. A final observed CSV sample is not necessarily the terminal
simulator state: callers must establish that separately before certifying an
episode outcome.
"""

import numpy as np

from telemetry_reasoning.schemas import Task


OUTCOME_VERSION = 1
PICKUP_MIN_RISE_M = 0.12
TRANSPORT_DISPLACEMENT_M = (0.0, 0.15, 0.0)
TRANSPORT_TARGET_HEIGHT_M = 0.427
TRANSPORT_MAX_XY_ERROR_M = 0.04
TRANSPORT_MAX_HEIGHT_ERROR_M = 0.025
TRANSPORT_MAX_FINAL_SPEED_M_S = 0.02


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain three finite values')
    return array


def assess_physical_outcome(task, initial_position, final_position,
                            final_velocity=None, *, terminal_observed=True,
                            source='simulator_state'):
    """Measure the physical objective in metres/seconds, independently of labels.

    Pickup requires an object rise strictly greater than 12 cm. Transport
    requires placement within 4 cm XY of the 15 cm world-Y displacement, height
    within 2.5 cm of the tabletop destination, and final speed at most 2 cm/s.
    An absent terminal observation or absent transport velocity yields unknown,
    even when the final available position satisfies the position thresholds.
    """
    task = Task(task)
    initial = _vector(initial_position, 'initial_position')
    final = _vector(final_position, 'final_position')
    velocity = None if final_velocity is None else _vector(final_velocity, 'final_velocity')
    if not isinstance(terminal_observed, bool):
        raise ValueError('terminal_observed must be a boolean')
    measurements = {
        'initial_position': initial.tolist(),
        'final_position': final.tolist(),
        'final_velocity': None if velocity is None else velocity.tolist(),
        'object_rise_m': float(final[2] - initial[2]),
        'final_speed_m_s': None if velocity is None else float(np.linalg.norm(velocity)),
    }
    if task == Task.PICK_UP_OBJECT:
        criteria = {'minimum_object_rise_m_exclusive': PICKUP_MIN_RISE_M}
        objective_met = measurements['object_rise_m'] > PICKUP_MIN_RISE_M
        incomplete = False
    else:
        target = initial + np.asarray(TRANSPORT_DISPLACEMENT_M)
        target[2] = TRANSPORT_TARGET_HEIGHT_M
        measurements.update({
            'target_position': target.tolist(),
            'placement_xy_error_m': float(np.linalg.norm(final[:2] - target[:2])),
            'placement_height_error_m': float(abs(final[2] - target[2])),
        })
        criteria = {
            'world_displacement_m': list(TRANSPORT_DISPLACEMENT_M),
            'target_height_m': TRANSPORT_TARGET_HEIGHT_M,
            'maximum_xy_error_m_exclusive': TRANSPORT_MAX_XY_ERROR_M,
            'maximum_height_error_m_exclusive': TRANSPORT_MAX_HEIGHT_ERROR_M,
            'maximum_final_speed_m_s_inclusive': TRANSPORT_MAX_FINAL_SPEED_M_S,
        }
        incomplete = velocity is None
        objective_met = None if incomplete else (
            measurements['placement_xy_error_m'] < TRANSPORT_MAX_XY_ERROR_M
            and measurements['placement_height_error_m'] < TRANSPORT_MAX_HEIGHT_ERROR_M
            and measurements['final_speed_m_s'] <= TRANSPORT_MAX_FINAL_SPEED_M_S
        )
    if not terminal_observed:
        status, reason = 'unknown', 'Terminal simulator state has not been independently established.'
    elif incomplete:
        status, reason = 'unknown', 'Transport terminal velocity is required to verify settled placement.'
    elif objective_met:
        status, reason = 'succeeded', 'Independent physical objective satisfied at the terminal state.'
    else:
        status, reason = 'failed', 'Independent physical objective not satisfied at the terminal state.'
    return {'outcome_version': OUTCOME_VERSION, 'task': task.value,
            'source': str(source), 'terminal_observed': terminal_observed,
            'status': status, 'objective_met': objective_met,
            'criteria': criteria, 'measurements': measurements, 'reason': reason}


def assess_telemetry_outcome(task, rows, *, terminal_observed=False):
    """Assess raw observations; never infer terminality from the last CSV row.

    ``terminal_observed=True`` is appropriate only when external evidence proves
    that the final stored observation represents the stopped simulator state.
    Neither a complete stage prefix nor a controller label supplies that proof.
    """
    if not rows:
        return {'outcome_version': OUTCOME_VERSION, 'task': Task(task).value,
                'source': 'stored_telemetry', 'terminal_observed': False,
                'status': 'unknown', 'objective_met': None, 'criteria': {},
                'measurements': {}, 'reason': 'No object observations are available.'}
    return assess_physical_outcome(task, rows[0].obj_pos, rows[-1].obj_pos,
                                   rows[-1].obj_vel,
                                   terminal_observed=terminal_observed,
                                   source='stored_telemetry')
