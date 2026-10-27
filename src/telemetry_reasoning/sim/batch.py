"""Headless pickup reliability evaluation with CSV trajectories."""
import argparse
import csv
from pathlib import Path
import numpy as np
import mujoco
from telemetry_reasoning.sim.pickup import PickupController
from telemetry_reasoning.sim.telemetry import TelemetryLogger

ROOT = Path(__file__).resolve().parents[3]


def run_one(index, rng, output, spread, grip_close):
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/scenes/pickup.xml'))
    data = mujoco.MjData(model)
    controller = PickupController(model, data, grip_close=grip_close)
    # Free joint position is stored in qpos; change only its x/y coordinates.
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
    return {'run': index, 'result': controller.stage, 'reason': controller.reason,
            'duration': round(float(data.time), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--spread', type=float, default=0.015)
    p.add_argument('--grip-close', type=float, default=255)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/pickup_batch')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = [run_one(i, rng, args.output, args.spread, args.grip_close)
            for i in range(args.runs)]
    with (args.output / 'summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    successes = sum(r['result'] == 'success' for r in rows)
    print(f'Pickup success: {successes}/{len(rows)} ({successes / len(rows):.1%})')
    print('Output:', args.output)


if __name__ == '__main__':
    main()
