"""CPU command-boundary checks plus opt-in native hook acceptance."""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from telemetry_reasoning.sim.controls import invoke_control_hook, validate_command
from telemetry_reasoning.sim.runner import EpisodeResult, _HookAdapter, _initialize_cube, _moment_row, run_episode


@pytest.fixture
def command():
    return {'ee_pos': [.4, .1, .5], 'ee_quat': [1., 0., 0., 0.],
            'gripper_state': .5, 'gripper_force': 100., 'speed_scale': 1.}


def test_hook_receives_detached_input_and_none_discards_mutations(command):
    before = deepcopy(command)
    calls = []
    def hook(timestamp, stage, nominal):
        calls.append((timestamp, stage))
        nominal['ee_pos'][0] = 99
        nominal['gripper_force'] = 1
        return None
    result = invoke_control_hook(hook, .002, 'grasp', command)
    assert calls == [(.002, 'grasp')]
    assert command == before == result
    result['ee_pos'][0] = -1
    assert command == before


def test_hook_return_is_validated_and_detached(command):
    output = deepcopy(command)
    output['ee_pos'][0] += .05
    output['gripper_state'] = 0.
    output['gripper_force'] = 3.
    output['speed_scale'] = 2.
    result = invoke_control_hook(lambda *_: output, 0, 'grasp', command)
    assert result == output and command['ee_pos'][0] == .4
    output['ee_pos'][0] = 99
    assert result['ee_pos'][0] == pytest.approx(.45)


@pytest.mark.parametrize('field,value', [
    ('ee_pos', [1, 2]), ('ee_pos', [1, float('inf'), 3]),
    ('ee_pos', ['1', '2', '3']), ('ee_pos', [True, False, True]),
    ('ee_pos', [1j, 0, 0]),
    ('ee_quat', [0, 0, 0, 0]), ('ee_quat', [1, 0, 0, .1]),
    ('gripper_state', -0.1), ('gripper_state', 1.1),
    ('gripper_force', -1), ('gripper_force', 101),
    ('gripper_force', float('nan')), ('speed_scale', 0),
    ('speed_scale', -1), ('speed_scale', True), ('speed_scale', '2'),
])
def test_bad_commands_are_rejected(command, field, value):
    command[field] = value
    with pytest.raises(ValueError):
        validate_command(command, maximum_gripper_force=100)


@pytest.mark.parametrize('change', ['missing', 'unknown', 'not_mapping'])
def test_hook_cannot_silently_drop_or_misspell_fields(command, change):
    if change == 'missing':
        del command['ee_pos']
    elif change == 'unknown':
        command['gripper_froce'] = 3
    else:
        command = []
    with pytest.raises(ValueError, match='exactly'):
        validate_command(command, maximum_gripper_force=100)


def test_result_supports_attributes_and_three_item_unpacking():
    result = EpisodeResult(['row'], True, {'task': 'pick_up_object'})
    rows, succeeded, diagnostics = result
    assert rows is result.rows and succeeded is result.task_succeeded
    assert diagnostics is result.diagnostics


def test_runner_rejects_invalid_hook_before_importing_native_backend():
    with pytest.raises(ValueError, match='control_hook'):
        run_episode(task='pick_up_object', seed=723, initial_condition_id=0,
                    control_hook='not callable')


def test_runner_contract_is_importable_without_optional_native_simulator(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'src/telemetry_reasoning/sim/runner.py'
    name = 'telemetry_reasoning.sim._runner_without_native'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'mujoco', None)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert list(module.EpisodeResult([], False, {})) == [[], False, {}]


def test_cube_initialization_preserves_robot_home_and_scene_quaternion():
    defaults = np.r_[np.zeros(9), [.5, 0, .425, 1, 0, 0, 0]]
    model = SimpleNamespace(qpos0=defaults, jnt_qposadr=[9],
                            body=lambda _name: SimpleNamespace(jntadr=[0]))
    home = np.r_[np.arange(9.), [0, 0, 0, 1, 0, 0, 0]]
    data = SimpleNamespace(qpos=home.copy())
    _initialize_cube(model, data, [.01, -.02])
    np.testing.assert_array_equal(data.qpos[:9], home[:9])
    np.testing.assert_array_equal(data.qpos[9:], [.51, -.02, .425, 1, 0, 0, 0])
    np.testing.assert_array_equal(model.qpos0, defaults)


@pytest.mark.parametrize('sparse', [False, True])
def test_panda_transmission_supports_dense_and_sparse_native_arrays(sparse):
    model = SimpleNamespace(nu=2, nv=3)
    if sparse:
        data = SimpleNamespace(actuator_moment=np.array([.5, .5]),
                               moment_rowadr=[0, 0], moment_rownnz=[0, 2],
                               moment_colind=[1, 2])
    else:
        data = SimpleNamespace(actuator_moment=np.array([[0, 0, 0], [0, .5, .5]]))
    np.testing.assert_array_equal(_moment_row(model, data, 1), [0, .5, .5])


@pytest.fixture
def adapter_inputs():
    # Force transmission .25 + .25 converts a 100 actuator-unit cap into 50 N
    # summed generalized finger force. ctrl remains an independent position.
    model = SimpleNamespace(nu=8, nv=9, opt=SimpleNamespace(timestep=.002),
                            actuator_forcerange=np.tile([-100., 100.], (8, 1)),
                            actuator_forcelimited=np.ones(8), jnt_dofadr=np.arange(9),
                            joint=lambda name: SimpleNamespace(id=7 if name.endswith('1') else 8))
    moments = np.zeros((8, 9)); moments[7, 7:] = .25
    data = SimpleNamespace(ctrl=np.r_[np.zeros(7), 255.], actuator_moment=moments,
                           qpos=np.arange(9), qvel=np.arange(9))
    class IK:
        target_q = np.arange(7.)
        last_command = (np.array([.4, .1, .5]), np.eye(3))
        calls = []
        def pose(self, _data):
            return self.last_command
        def move(self, *args, **kwargs):
            self.calls.append((args, kwargs))
    native = SimpleNamespace(mju_mat2Quat=lambda quat, _matrix: quat.__setitem__(slice(None), [1, 0, 0, 0]),
                             mju_quat2Mat=lambda matrix, _quat: matrix.__setitem__(slice(None), np.eye(3).ravel()))
    return model, data, IK(), native


def test_adapter_changes_position_and_force_commands_without_faking_measurements(adapter_inputs):
    model, data, ik, native = adapter_inputs
    before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
    previous_target = ik.target_q.copy()
    seen = []
    def hook(_time, _stage, nominal):
        seen.append(deepcopy(nominal))
        nominal.update(gripper_state=.8, gripper_force=3., speed_scale=2.)
        nominal['ee_pos'][0] += .05
        return nominal
    adapter = _HookAdapter(model, data, ik, native)
    adapter.apply(hook, .002, 'grasp', model, data, previous_target)
    assert seen[0]['gripper_force'] == 50.0
    assert data.ctrl[7] == pytest.approx(51.)
    np.testing.assert_array_equal(model.actuator_forcerange[7], [-6., 6.])
    np.testing.assert_array_equal(data.qpos, before_qpos)
    np.testing.assert_array_equal(data.qvel, before_qvel)
    assert len(ik.calls) == 1 and ik.calls[0][1] == {'speed_scale': 2.}
    np.testing.assert_array_equal(ik.target_q, previous_target)
    adapter.close()
    np.testing.assert_array_equal(model.actuator_forcerange[7], [-100., 100.])


def test_identity_hook_does_not_integrate_ik_twice(adapter_inputs):
    model, data, ik, native = adapter_inputs
    adapter = _HookAdapter(model, data, ik, native)
    before = data.ctrl.copy()
    adapter.apply(lambda _t, _s, command: command, 0., 'approach', model, data,
                  ik.target_q.copy())
    assert not ik.calls
    np.testing.assert_array_equal(data.ctrl, before)


@pytest.mark.skipif(os.environ.get('TRACK2_REAL_SIM') != '1',
                    reason='Set TRACK2_REAL_SIM=1 for native hook acceptance')
@pytest.mark.parametrize('task', ['pick_up_object', 'transport_object'])
def test_real_identity_hook_matches_no_hook_every_canonical_step(task):
    pytest.importorskip('mujoco')
    from telemetry_reasoning.sim.runner import MODEL_PATH
    if not MODEL_PATH.is_file():
        pytest.skip('MuJoCo scene absent')
    nominal = run_episode(task=task, seed=723, initial_condition_id=0)
    calls = []
    def hook(timestamp, stage, command):
        calls.append((timestamp, stage))
        return command
    hooked = run_episode(task=task, seed=723, initial_condition_id=0, control_hook=hook)
    assert nominal.task_succeeded and hooked.task_succeeded
    assert nominal.task_succeeded == hooked.task_succeeded
    assert [row.model_dump() for row in nominal.rows] == [row.model_dump() for row in hooked.rows]
    assert nominal.diagnostics['terminal_state'] == hooked.diagnostics['terminal_state']
    assert calls[0][0] == 0.
    np.testing.assert_allclose(np.diff([time for time, _ in calls]), .002, atol=1e-10, rtol=0)
    assert hooked.diagnostics['control_hook']['calls'] == len(calls)


@pytest.mark.skipif(os.environ.get('TRACK2_REAL_SIM') != '1',
                    reason='Set TRACK2_REAL_SIM=1 for native hook acceptance')
@pytest.mark.parametrize('perturbation', ['position', 'orientation', 'gripper', 'speed'])
def test_real_hook_changes_physics_instead_of_rewriting_observations(perturbation):
    pytest.importorskip('mujoco')
    nominal = run_episode(task='pick_up_object', seed=723, initial_condition_id=0)
    def hook(_timestamp, stage, command):
        if perturbation == 'position' and stage == 'align':
            command['ee_pos'][0] += .06
        elif perturbation == 'orientation' and stage == 'align':
            command['ee_quat'] = [1., 0., 0., 0.]
        elif perturbation == 'gripper' and stage in ('grasp', 'lift'):
            command['gripper_state'] = 0.
            command['gripper_force'] = 3.
        elif perturbation == 'speed':
            command['speed_scale'] = 1.5
        return command
    injected = run_episode(task='pick_up_object', seed=723, initial_condition_id=0,
                           control_hook=hook)
    assert nominal.task_succeeded
    assert injected.diagnostics['control_hook']['calls'] > 0
    assert nominal.rows[0] == injected.rows[0]
    assert ([row.model_dump() for row in nominal.rows]
            != [row.model_dump() for row in injected.rows])
    # These are observations of physical finger joints/actuator force; a 3 N
    # commanded cap cannot synthesize a constant force or closed-state reading.
    if perturbation == 'gripper':
        assert not injected.task_succeeded
        assert injected.diagnostics['physical_outcome']['status'] == 'failed'
        assert injected.rows[-1].gripper_state < .1
        assert abs(injected.rows[-1].gripper_force - 3.) > .01
