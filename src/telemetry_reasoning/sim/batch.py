"""Headless pickup reliability evaluation with uniquely named batches."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import mujoco

from telemetry_reasoning.sim.pickup import PickupController
from telemetry_reasoning.sim.telemetry import TelemetryLogger

ROOT = Path(__file__).resolve().parents[3]


def run_one(index, rng, output, spread, grip_close):
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/scenes/pickup.xml'))
    data = mujoco.MjData(model)
    controller = PickupController(model, data, grip_close=grip_close)

    cube_joint = model.body('cube').jntadr[0]
    qadr = model.jnt_qposadr[cube_joint]
    data.qpos[qadr:qadr + 2] += rng.uniform(-spread, spread, 2)
    mujoco.mj_forward(model, data)
    controller.cube_initial = data.xpos[controller.cube_id].copy()
    controller.cube_anchor = controller.cube_initial.copy()

    with TelemetryLogger(model, output / f'pickup_{index:04d}.csv') as log:
        while not controller.done:
            controller.update(data)
            mujoco.mj_step(model, data)
            log.record(data, controller.stage)

    return {
        'run': index,
        'result': controller.stage,
        'reason': controller.reason,
        'duration': round(float(data.time), 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--spread', type=float, default=0.015)
    p.add_argument('--grip-close', type=float, default=0.0)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/pickup',
                   help='Parent directory for uniquely named batch folders')
    args = p.parse_args()
    if args.runs < 1:
        p.error('--runs must be at least 1')

    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    output = args.output / batch_id
    output.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(args.seed)
    rows = [run_one(i, rng, output, args.spread, args.grip_close)
            for i in range(args.runs)]

    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(row['result'] == 'success' for row in rows)
    print(f'Pickup success: {successes}/{len(rows)} ({successes / len(rows):.1%})')
    print('Output:', output)


if __name__ == '__main__':
    main()
