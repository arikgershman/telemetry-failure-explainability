# to visualize each failure


"""Visualize nominal and failure-injected MuJoCo episodes.

Run with mjpython on macOS.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from telemetry_reasoning.schemas import FailureMode, Subtask, Task
from telemetry_reasoning.sim.failures import (
    FailureConfig,
    FailureEngine,
    validate_config,
)
from telemetry_reasoning.sim.pickup import PickupController
from telemetry_reasoning.sim.transport import TransportController


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "assets/scenes/pickup.xml"

# Initial configurations for visualization.
# These are not yet calibrated to guarantee physical failure.
DEFAULTS = {
    "position_error": (0.10, 0),
    "orientation_error": (0.7, 0),
    "inadequate_grip_strength": (150.0, 0),
    "early_gripper_release": (1.0, 0),
    "excessive_speed": (4.0, 0),
}

DEFAULT_SUBTASKS = {
    "pickup": {
        "position_error": "align",
        "orientation_error": "align",
        "inadequate_grip_strength": "grasp",
        "early_gripper_release": "lift",
        "excessive_speed": "approach",
    },
    "transport": {
        "position_error": "move_along_path",
        "orientation_error": "move_along_path",
        "inadequate_grip_strength": "maintain_grasp",
        "early_gripper_release": "move_along_path",
        "excessive_speed": "move_along_path",
    },
}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", choices=["pickup", "transport"], required=True
    )
    parser.add_argument(
        "--failure",
        choices=[mode.value for mode in FailureMode] + ["none"],
        required=True,
    )
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--magnitude", type=float, default=None)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--seed", type=int, default=723)
    parser.add_argument("--condition", type=int, default=0)
    parser.add_argument("--spread", type=float, default=0.015)
    parser.add_argument("--speed", type=float, default=1.0)

    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.condition < 0 or args.seed < 0:
        parser.error("--seed and --condition must be nonnegative")
    if args.spread < 0:
        parser.error("--spread must be nonnegative")

    task = (
        Task.PICK_UP_OBJECT
        if args.task == "pickup"
        else Task.TRANSPORT_OBJECT
    )

    engine = None

    if args.failure != "none":
        magnitude = (
            args.magnitude
            if args.magnitude is not None
            else DEFAULTS[args.failure][0]
        )

        subtask = (
            args.subtask
            or DEFAULT_SUBTASKS[args.task][args.failure]
        )

        config = FailureConfig(
            mode=FailureMode(args.failure),
            subtask=Subtask(subtask),
            magnitude=magnitude,
            delay_s=args.delay,
            axis=args.axis,
        )

        validate_config(task, config)
        engine = FailureEngine(config)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    controller = (
        PickupController(model, data)
        if args.task == "pickup"
        else TransportController(model, data)
    )

    # Match runner.py's deterministic initial conditions.
    rng = np.random.default_rng(
        np.random.SeedSequence([args.seed, args.condition])
    )
    offset = rng.uniform(-args.spread, args.spread, 2)

    cube_joint = model.body("cube").jntadr[0]
    qadr = model.jnt_qposadr[cube_joint]
    data.qpos[qadr:qadr + 2] += offset

    mujoco.mj_forward(model, data)

    pickup = (
        controller
        if args.task == "pickup"
        else controller.pickup
    )
    pickup.cube_initial = data.xpos[pickup.cube_id].copy()
    pickup.cube_anchor = pickup.cube_initial.copy()

    if engine is not None:
        controller.ik.failure_engine = engine

    print(f"Task: {args.task}")
    print(f"Failure: {args.failure}")
    print(f"Initial offset: {offset}")
    if engine is not None:
        print(f"Target subtask: {config.subtask.value}")
        print(f"Magnitude: {config.magnitude}")
        print(f"Delay: {config.delay_s}s")

    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        while viewer.is_running() and not controller.done:
            start = time.perf_counter()

            stage_before = controller.stage

            # Transport preparation pickup is always nominal.
            if engine is not None:
                active_stage = (
                    stage_before
                    if stage_before != "pickup"
                    and stage_before != "settle"
                    else None
                )
                engine.set_context(active_stage, float(data.time))

            controller.update(data)

            if engine is not None:
                stage_after = controller.stage
                active_stage = (
                    stage_after
                    if stage_after != "pickup"
                    and stage_after != "settle"
                    else None
                )
                engine.set_context(active_stage, float(data.time))
                engine.transform_commands(data, stage_after)

            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time playback.
            remaining = dt / args.speed - (
                time.perf_counter() - start
            )
            if remaining > 0:
                time.sleep(remaining)

        print("\nEpisode finished")
        print("Outcome:", controller.stage)
        print("Reason:", controller.reason)

        if engine is not None:
            print("Injection:", engine.diagnostics())

        print("Close the viewer window to exit.")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
