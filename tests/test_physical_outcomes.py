"""Independent physical criteria do not require MuJoCo or controller labels."""

from types import SimpleNamespace

import numpy as np
import pytest

from telemetry_reasoning.sim.outcomes import (
    assess_physical_outcome, assess_telemetry_outcome,
)


@pytest.mark.parametrize('height,expected', [(0.119, 'failed'), (0.12, 'failed'), (0.121, 'succeeded')])
def test_pickup_strict_rise_threshold(height, expected):
    # Choose an exactly representable initial z so the 12 cm boundary is exact.
    result = assess_physical_outcome('pick_up_object', [0, 0, 0], [0, 0, height])
    assert result['status'] == expected
    assert result['measurements']['object_rise_m'] == height


def test_pickup_exact_boundary_and_no_input_mutation():
    initial, final = np.array([0, 0, 0]), np.array([0, 0, .12])
    result = assess_physical_outcome('pick_up_object', initial, final)
    assert result['status'] == 'failed'
    result['measurements']['initial_position'][0] = 9
    np.testing.assert_array_equal(initial, [0, 0, 0])


@pytest.mark.parametrize('final,velocity,expected', [
    ([.5, .15, .427], [0, 0, 0], 'succeeded'),
    ([.5, .15, .427], [.02, 0, 0], 'succeeded'),
    ([.5, .15, .427], [.021, 0, 0], 'failed'),
    ([.5, .20, .427], [0, 0, 0], 'failed'),
    ([.5, .15, .5], [0, 0, 0], 'failed'),
    ([.5, .15, .427], None, 'unknown'),
])
def test_transport_requires_xy_height_and_settled_velocity(final, velocity, expected):
    result = assess_physical_outcome('transport_object', [.5, 0, .56], final, velocity)
    assert result['status'] == expected
    assert result['measurements']['target_position'] == [.5, .15, .427]


@pytest.mark.parametrize('task', ['pick_up_object', 'transport_object'])
def test_final_csv_sample_does_not_establish_termination(task):
    rows = [SimpleNamespace(obj_pos=[.5, 0, .3], obj_vel=[0, 0, 0]),
            SimpleNamespace(obj_pos=[.5, .15, .427], obj_vel=[0, 0, 0])]
    observed = assess_telemetry_outcome(task, rows)
    assert observed['status'] == 'unknown'
    assert observed['objective_met'] is True
    assert assess_telemetry_outcome(task, rows, terminal_observed=True)['status'] == 'succeeded'


def test_empty_trace_is_explicitly_unknown():
    result = assess_telemetry_outcome('pick_up_object', [])
    assert result['status'] == 'unknown' and result['objective_met'] is None


@pytest.mark.parametrize('field,bad', [('initial_position', [0, 0]),
                                      ('final_position', [0, float('nan'), 0]),
                                      ('final_velocity', [0, 0, float('inf')])])
def test_malformed_physical_measurements_are_rejected(field, bad):
    args = dict(task='pick_up_object', initial_position=[0, 0, 0],
                final_position=[0, 0, 1], final_velocity=[0, 0, 0])
    args[field] = bad
    with pytest.raises(ValueError, match=field):
        assess_physical_outcome(**args)
