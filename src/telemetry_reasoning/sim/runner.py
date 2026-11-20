"""Unified, reproducible pickup/transport simulation episodes."""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import platform
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from telemetry_reasoning.io import read_telemetry_csv
from telemetry_reasoning.schemas import Task, TASK_SUBTASKS
from telemetry_reasoning.sim.failures import FailureConfig, FailureEngine, validate_config
from telemetry_reasoning.sim.controls import invoke_control_hook
from telemetry_reasoning.sim.outcomes import assess_physical_outcome

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / 'assets/scenes/pickup.xml'


@dataclass
class EpisodeResult:
    rows: list
    task_succeeded: bool
    diagnostics: dict

    def __iter__(self):
        """Allow ``rows, task_succeeded, diagnostics = run_episode(...)``."""
        yield self.rows
        yield self.task_succeeded
        yield self.diagnostics


@lru_cache(maxsize=8)
def _fingerprints(files):
    # The cache key includes size and nanosecond mtime for every dependency,
    # rather than silently reusing hashes after a source or asset edit.
    return {name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name, _size, _mtime in files}


def runtime_provenance(mujoco_version):
    """Fingerprint the source and actual scene/robot files used by this runner.

    This identifies the current producer; it makes no claim that these sources
    generated any previously archived CSVs. Asset contents, not a mutable branch
    name or checkout label, determine the model fingerprint.
    """
    sources = list((ROOT / 'src/telemetry_reasoning/sim').glob('*.py'))
    sources += [ROOT / 'src/telemetry_reasoning' / name
                for name in ('schemas.py', 'io.py')]
    model_files = [MODEL_PATH]
    model_files += [path for path in (ROOT / 'assets/mujoco_menagerie/franka_emika_panda').rglob('*')
                    if path.is_file()]
    def inventory(paths):
        key = tuple((path.relative_to(ROOT).as_posix(), path.stat().st_size,
                     path.stat().st_mtime_ns) for path in sorted(set(paths)))
        return _fingerprints(key)
    source_hashes, model_hashes = inventory(sources), inventory(model_files)
    def digest(values):
        return sha256('\n'.join(f'{name}:{value}' for name, value in sorted(values.items())).encode()).hexdigest()
    try:
        revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                  check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain',
                                '--untracked-files=all'], check=True,
                               capture_output=True, text=True).stdout
        tree_clean = not bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, tree_clean = None, None
    return {'provenance_version': 1, 'recorded_at_generation': True,
            'revision': revision, 'tree_clean': tree_clean,
            'python_version': platform.python_version(),
            'numpy_version': np.__version__, 'mujoco_version': str(mujoco_version),
            'source_sha256': digest(source_hashes), 'model_sha256': digest(model_hashes),
            'source_files': source_hashes, 'model_files': model_hashes}


def _moment_row(model, data, actuator):
    """Read dense (older MuJoCo) or sparse actuator transmissions."""
    moments = np.asarray(data.actuator_moment)
    if moments.size == model.nu * model.nv:
        return moments.reshape(model.nu, model.nv)[actuator].copy()
    row = np.zeros(model.nv)
    start = int(data.moment_rowadr[actuator])
    end = start + int(data.moment_rownnz[actuator])
    row[np.asarray(data.moment_colind[start:end], dtype=int)] = moments[start:end]
    return row


class _HookAdapter:
    """Map physical commands to Panda targets without editing measured state."""

    def __init__(self, model, data, ik, mujoco):
        self.model, self.ik, self.mujoco = model, ik, mujoco
        self.force_range = model.actuator_forcerange[7].copy()
        if not model.actuator_forcelimited[7]:
            raise ValueError('control_hook requires a force-limited Panda gripper actuator')
        self.finger_dids = [int(model.jnt_dofadr[model.joint(name).id])
                            for name in ('finger_joint1', 'finger_joint2')]
        self.calls = 0

    def apply(self, hook, timestamp, stage, model, data, previous_target_q):
        if self.ik.last_command is None:
            position, rotation = self.ik.pose(data)
        else:
            position, rotation = self.ik.last_command
        quat = np.empty(4)
        self.mujoco.mju_mat2Quat(quat, np.asarray(rotation).ravel())
        # Sum of absolute finger generalized forces per actuator force unit.
        # In the pinned Panda split tendon this is .5 + .5 = 1. This conversion
        # avoids equating dimensionless ctrl[7] with force measured in newtons.
        transmission = float(np.sum(np.abs(_moment_row(model, data, 7)[self.finger_dids])))
        if not np.isfinite(transmission) or transmission <= 0:
            raise ValueError('Cannot resolve Panda finger actuator force transmission')
        maximum_force = float(np.max(np.abs(self.force_range)) * transmission)
        nominal = {'ee_pos': np.asarray(position).tolist(), 'ee_quat': quat.tolist(),
                   'gripper_state': float(np.clip(1.0 - data.ctrl[7] / 255.0, 0.0, 1.0)),
                   'gripper_force': maximum_force, 'speed_scale': 1.0}
        changed = invoke_control_hook(hook, timestamp, stage, nominal)
        self.calls += 1
        # Identity hooks retain the original solver result bit for bit. When a
        # motion command changes, undo only the solver target integration before
        # solving once from the same measured state. No qpos/qvel is overwritten.
        if any(changed[name] != nominal[name] for name in ('ee_pos', 'ee_quat', 'speed_scale')):
            self.ik.target_q = None if previous_target_q is None else previous_target_q.copy()
            rotation = np.empty(9)
            quaternion = np.asarray(changed['ee_quat'], dtype=float)
            quaternion = quaternion / np.linalg.norm(quaternion)
            self.mujoco.mju_quat2Mat(rotation, quaternion)
            self.ik.move(data, changed['ee_pos'], rotation.reshape(3, 3),
                         model.opt.timestep, speed_scale=changed['speed_scale'])
        if changed['gripper_state'] != nominal['gripper_state']:
            data.ctrl[7] = (1.0 - changed['gripper_state']) * 255.0
        if changed['gripper_force'] == nominal['gripper_force']:
            model.actuator_forcerange[7] = self.force_range
        else:
            cap = changed['gripper_force'] / transmission
            model.actuator_forcerange[7] = [max(self.force_range[0], -cap),
                                           min(self.force_range[1], cap)]

    def close(self):
        self.model.actuator_forcerange[7] = self.force_range


def _initialize_cube(model, data, offset):
    """Restore the scene's free-object pose after the robot-only home keyframe.

    Menagerie's Panda home keyframe specifies nine robot positions; MuJoCo fills
    appended free-body translations with zero. Restore only the cube's seven
    free-joint coordinates from the compiled scene default, preserving robot
    home joints and applying the deterministic offset in world XY metres.
    """
    joint = int(model.body('cube').jntadr[0])
    qadr = int(model.jnt_qposadr[joint])
    data.qpos[qadr:qadr + 7] = model.qpos0[qadr:qadr + 7]
    data.qpos[qadr:qadr + 2] += offset


def _run(task, seed, initial_condition_id, failure, output, spread, control_hook=None):
    # Pure outcome/contract helpers and EpisodeResult remain importable without
    # installing the optional native simulator backend.
    import mujoco
    from telemetry_reasoning.sim.pickup import PickupController
    from telemetry_reasoning.sim.transport import TransportController
    from telemetry_reasoning.sim.telemetry import TelemetryLogger

    task = Task(task)
    if failure is not None:
        validate_config(task, failure)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    if not np.isclose(model.opt.timestep, 0.002):
        raise ValueError('Expected 500 Hz physics timestep')

    controller = (PickupController(model, data) if task == Task.PICK_UP_OBJECT
                  else TransportController(model, data))
    rng = np.random.default_rng(np.random.SeedSequence([seed, initial_condition_id]))
    offset = rng.uniform(-spread, spread, 2)
    _initialize_cube(model, data, offset)
    mujoco.mj_forward(model, data)
    pickup = controller if task == Task.PICK_UP_OBJECT else controller.pickup
    pickup.cube_initial = data.xpos[pickup.cube_id].copy()
    pickup.cube_anchor = pickup.cube_initial.copy()
    provenance = runtime_provenance(mujoco.__version__)
    adapter = (_HookAdapter(model, data, controller.ik, mujoco)
               if control_hook is not None else None)
    initial_task_position = None

    engine = FailureEngine(failure) if failure is not None else None
    if engine is not None:
        controller.ik.failure_engine = engine

    allowed = {s.value for s in TASK_SUBTASKS[task]}
    previous = controller.stage
    with TelemetryLogger(model, output) as log:
        while not controller.done:
            # The transport preparation pickup is intentionally unperturbed.
            stage_before = controller.stage
            if engine is not None:
                engine.set_context(stage_before if stage_before in allowed else None,
                                   float(data.time))
            previous_target_q = (None if controller.ik.target_q is None
                                 else controller.ik.target_q.copy()) if adapter else None
            if adapter:
                controller.ik.last_command = None
            controller.update(data)
            stage = controller.stage

            # Record at t=0 of the canonical task, before the first physics step.
            if (task == Task.PICK_UP_OBJECT and previous == 'settle'
                    and stage == 'approach') or (
                    task == Task.TRANSPORT_OBJECT and previous == 'pickup'
                    and stage == 'maintain_grasp'):
                log.record(data, stage)
                initial_task_position = data.xpos[pickup.cube_id].copy()

            if adapter and log.origin is not None and (stage in allowed or stage_before in allowed):
                adapter.apply(control_hook, float(data.time) - log.origin,
                              stage if stage in allowed else stage_before,
                              model, data, previous_target_q)

            if engine is not None:
                # Update context for stage transitions before applying actuator overrides.
                engine.set_context(stage if stage in allowed else None, float(data.time))
                engine.transform_commands(data, stage)

            mujoco.mj_step(model, data)
            if stage in allowed:
                log.record(data, stage)
            previous = stage

    if adapter:
        adapter.close()
    # mj_step integrates qpos after computing xpos. Recompute kinematics once
    # after stopping to record the actual terminal state, without changing any
    # already recorded 50 Hz observations or the no-hook controller trajectory.
    mujoco.mj_forward(model, data)
    cube_joint = model.body('cube').jntadr[0]
    cube_dof = model.jnt_dofadr[cube_joint]
    terminal_state = {
        'initial_position': (initial_task_position.tolist()
                             if initial_task_position is not None else None),
        'final_position': data.xpos[pickup.cube_id].astype(float).tolist(),
        'final_velocity': data.qvel[cube_dof:cube_dof + 3].astype(float).tolist(),
        'terminal_observed': True,
        'simulator_time_s': float(data.time),
    }
    physical_outcome = (assess_physical_outcome(
        task, terminal_state['initial_position'], terminal_state['final_position'],
        terminal_state['final_velocity'], source='simulator_terminal_state')
        if initial_task_position is not None else {
            'outcome_version': 1, 'task': task.value, 'status': 'unknown',
            'terminal_observed': True, 'objective_met': None,
            'source': 'simulator_terminal_state', 'criteria': {}, 'measurements': {},
            'reason': 'Canonical task was never entered.'})

    # A transport preparation failure legitimately yields no canonical task
    # observations. Preserve that outcome for generator rejection/diagnostics;
    # the strict public CSV reader still rejects header-only dataset inputs.
    rows = read_telemetry_csv(output) if log.sample_index else []
    observed = []
    for row in rows:
        if not observed or observed[-1] != row.subtask:
            observed.append(row.subtask)
    expected = list(TASK_SUBTASKS[task])
    prefix_ok = observed == expected[:len(observed)]
    if not prefix_ok:
        raise ValueError(f'Unexpected subtask order: {observed}')

    diagnostics = {
        'task': task.value,
        'seed': seed,
        'initial_condition_id': initial_condition_id,
        'spread_m': float(spread),
        'telemetry_start_sim_s': log.origin,
        'initial_offset_xy': offset.tolist(),
        'initialization_policy': 'robot_home_keyframe_then_scene_cube_free_joint_v1',
        'controller_outcome': controller.stage,
        'controller_reason': controller.reason,
        'duration_s': float(data.time),
        'observed_subtasks': [s.value for s in observed],
        'complete_subtasks': observed == expected,
        'failure': engine.diagnostics() if engine else None,
        'failure_config': ({'mode': failure.mode.value, 'subtask': failure.subtask.value,
                            'magnitude': float(failure.magnitude),
                            'delay_s': float(failure.delay_s), 'axis': int(failure.axis)}
                           if failure is not None else None),
        'terminal_state': terminal_state,
        'physical_outcome': physical_outcome,
        'producer': provenance,
        'control_hook': {'enabled': adapter is not None,
                         'calls': adapter.calls if adapter else 0,
                         'timestamp_basis': 'seconds_since_canonical_task_onset',
                         'frequency_hz': 500},
    }
    return EpisodeResult(rows, controller.stage == 'success', diagnostics)


def run_episode(*, task, seed, initial_condition_id, failure: FailureConfig | None = None,
                output=None, spread=0.015, control_hook=None):
    """Run an episode and return ``EpisodeResult`` (also a three-item iterable).

    ``control_hook(timestamp, subtask, nominal_command)`` is called once before
    each 500 Hz canonical-task physics step, including entry at timestamp zero.
    Preparation/settling receive no hook. Times are seconds relative to canonical
    task onset, subtasks are strings, and commands are detached dictionaries with
    ee_pos (world m), ee_quat (unit wxyz), gripper_state (0=open, 1=closed),
    gripper_force (total finger generalized actuator-force limit, N), and
    speed_scale (positive multiplier of the nominal joint velocity limit).
    Return a complete command or None to preserve the nominal command. No hook
    changes measured telemetry directly. Hook commands precede FailureConfig
    actuator overrides; motion hooks and FailureConfig target transforms compose.

    If output is supplied, retain the canonical CSV. Recorded terminal physical
    outcomes are independent of controller outcome strings. Producer fingerprints
    describe this execution and do not establish provenance of historical CSVs.
    """
    if control_hook is not None and not callable(control_hook):
        raise ValueError('control_hook must be callable or None')
    if not np.isfinite(spread) or spread < 0:
        raise ValueError('spread must be finite and nonnegative')
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        return _run(task, seed, initial_condition_id, failure, output, spread, control_hook)
    with TemporaryDirectory() as tmp:
        return _run(task, seed, initial_condition_id, failure,
                    Path(tmp) / 'episode.csv', spread, control_hook)
