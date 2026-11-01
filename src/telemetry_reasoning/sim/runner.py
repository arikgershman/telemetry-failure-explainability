"""Unified, reproducible pickup/transport simulation episodes."""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from telemetry_reasoning.io import read_telemetry_csv
from telemetry_reasoning.schemas import Task, TASK_SUBTASKS
from telemetry_reasoning.sim.failures import FailureConfig, FailureEngine, validate_config
from telemetry_reasoning.sim.pickup import PickupController
from telemetry_reasoning.sim.transport import TransportController
from telemetry_reasoning.sim.telemetry import TelemetryLogger

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / 'assets/scenes/pickup.xml'


@dataclass
class EpisodeResult:
    rows: list
    task_succeeded: bool
    diagnostics: dict


def _run(task, seed, initial_condition_id, failure, output, spread):
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
    cube_joint = model.body('cube').jntadr[0]
    qadr = model.jnt_qposadr[cube_joint]
    data.qpos[qadr:qadr + 2] += offset
    mujoco.mj_forward(model, data)
    pickup = controller if task == Task.PICK_UP_OBJECT else controller.pickup
    pickup.cube_initial = data.xpos[pickup.cube_id].copy()
    pickup.cube_anchor = pickup.cube_initial.copy()

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
            controller.update(data)
            stage = controller.stage

            # Record at t=0 of the canonical task, before the first physics step.
            if (task == Task.PICK_UP_OBJECT and previous == 'settle'
                    and stage == 'approach') or (
                    task == Task.TRANSPORT_OBJECT and previous == 'pickup'
                    and stage == 'maintain_grasp'):
                log.record(data, stage)

            if engine is not None:
                # Update context for stage transitions before applying actuator overrides.
                engine.set_context(stage if stage in allowed else None, float(data.time))
                engine.transform_commands(data, stage)

            mujoco.mj_step(model, data)
            if stage in allowed:
                log.record(data, stage)
            previous = stage

    rows = read_telemetry_csv(output)
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
        'initial_offset_xy': offset.tolist(),
        'controller_outcome': controller.stage,
        'controller_reason': controller.reason,
        'duration_s': float(data.time),
        'observed_subtasks': [s.value for s in observed],
        'complete_subtasks': observed == expected,
        'failure': engine.diagnostics() if engine else None,
    }
    return EpisodeResult(rows, controller.stage == 'success', diagnostics)


def run_episode(*, task, seed, initial_condition_id, failure: FailureConfig | None = None,
                output=None, spread=0.015):
    """Return EpisodeResult. If output is supplied, retain the canonical CSV."""
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        return _run(task, seed, initial_condition_id, failure, output, spread)
    with TemporaryDirectory() as tmp:
        return _run(task, seed, initial_condition_id, failure,
                    Path(tmp) / 'episode.csv', spread)
