
"""Headless pickup reliability evaluation."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np

from telemetry_reasoning.sim.pickup import PickupController
from telemetry_reasoning.sim.telemetry import TelemetryLogger

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "assets/scenes/pickup.xml"

PICKUP_STAGES = {"approach", "align", "grasp", "lift"}


def run_one(index, rng, output, spread, grip_close):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    controller = PickupController(model, data, grip_close=grip_close)

    cube_joint = model.body("cube").jntadr[0]
    qadr = model.jnt_qposadr[cube_joint]
    data.qpos[qadr:qadr + 2] += rng.uniform(-spread, spread, 2)

    mujoco.mj_forward(model, data)
    controller.cube_initial = data.xpos[controller.cube_id].copy()
    controller.cube_anchor = controller.cube_initial.copy()

    output_file = output / f"pickup_{index:04d}.csv"

    with TelemetryLogger(model, output_file) as log:
        previous_stage = controller.stage

        while not controller.done:
            controller.update(data)

            # Capture the first sample exactly at the approach transition.
            if (
                previous_stage == "settle"
                and controller.stage == "approach"
            ):
                log.record(data, "approach")

            mujoco.mj_step(model, data)

            if controller.stage in PICKUP_STAGES:
                log.record(data, controller.stage)

            previous_stage = controller.stage

    return {
        "run": index,
        "result": controller.stage,
        "reason": controller.reason,
        "duration": round(float(data.time), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spread", type=float, default=0.015)
    parser.add_argument("--grip-close", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/pickup"
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    batch_id = (
        f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid4().hex[:8]}"
    )
    output = args.output / batch_id
    output.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(args.seed)

    results = [
        run_one(i, rng, output, args.spread, args.grip_close)
        for i in range(args.runs)
    ]

    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    successes = sum(r["result"] == "success" for r in results)
    print(
        f"Pickup success: {successes}/{len(results)} "
        f"({successes / len(results):.1%})"
    )
    print("Output:", output)


if __name__ == "__main__":
    main()
