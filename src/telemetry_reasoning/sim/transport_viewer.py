
"""Interactive transport simulation (macOS: use mjpython)."""

import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import mujoco
import mujoco.viewer

from telemetry_reasoning.sim.transport import TransportController
from telemetry_reasoning.sim.telemetry import TelemetryLogger


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "assets/scenes/pickup.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

controller = TransportController(model, data)

run_id = (
    f"viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    f"_{uuid4().hex[:8]}"
)

output = ROOT / "outputs/transport" / f"{run_id}.csv"
output.parent.mkdir(parents=True, exist_ok=True)

print("Starting transport simulation...")

with TelemetryLogger(model, output) as log:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            controller.update(data)

            mujoco.mj_step(model, data)

            # Preserve pickup subtask labels in the telemetry.
            stage = (
                controller.pickup.stage
                if controller.stage == "pickup"
                else controller.stage
            )

            log.record(data, stage)
            viewer.sync()

            time.sleep(model.opt.timestep)

            if controller.done:
                print(
                    "Result:",
                    controller.stage,
                    controller.reason,
                )
                break

print("Output:", output)
