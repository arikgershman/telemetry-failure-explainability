
"""Headless transport batch evaluation."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import mujoco

from telemetry_reasoning.sim.transport import TransportController
from telemetry_reasoning.sim.telemetry import TelemetryLogger


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "assets/scenes/pickup.xml"


def run_one(index, rng, output, spread):
    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )
    data = mujoco.MjData(model)

    controller = TransportController(model, data)

    # Randomize the cube's initial x/y coordinates.
    cube_joint = model.body("cube").jntadr[0]
    qadr = model.jnt_qposadr[cube_joint]

    offset = rng.uniform(-spread, spread, 2)
    data.qpos[qadr:qadr + 2] += offset

    mujoco.mj_forward(model, data)

    controller.pickup.cube_initial = (
        data.xpos[controller.cube_id].copy()
    )
    controller.pickup.cube_anchor = (
        controller.pickup.cube_initial.copy()
    )

    output_file = output / f"transport_{index:04d}.csv"

    with TelemetryLogger(model, output_file) as log:
        while not controller.done:
            controller.update(data)
            mujoco.mj_step(model, data)

            stage = (
                controller.pickup.stage
                if controller.stage == "pickup"
                else controller.stage
            )

            log.record(data, stage)

    return {
        "run": index,
        "result": controller.stage,
        "reason": controller.reason,
        "duration": round(float(data.time), 3),
        "initial_offset_x": float(offset[0]),
        "initial_offset_y": float(offset[1]),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spread", type=float, default=0.015)

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    run_id = (
        f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid4().hex[:8]}"
    )

    output = ROOT / "outputs/transport" / run_id
    output.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(args.seed)

    results = []

    for i in range(args.runs):
        result = run_one(
            i, rng, output, args.spread
        )
        results.append(result)

        print(
            f"Run {i}: {result['result']} "
            f"({result['reason']})"
        )

    summary = output / "summary.csv"

    with summary.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    successes = sum(
        r["result"] == "success"
        for r in results
    )

    print(
        f"Transport success: {successes}/{len(results)} "
        f"({successes / len(results):.1%})"
    )
    print("Output:", output)


if __name__ == "__main__":
    main()
