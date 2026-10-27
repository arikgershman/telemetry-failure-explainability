
"""Interactive pickup playback (on macOS run with mjpython)."""

import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import mujoco
import mujoco.viewer

from telemetry_reasoning.sim.pickup import PickupController
from telemetry_reasoning.sim.telemetry import TelemetryLogger


ROOT = Path(__file__).resolve().parents[3]

model = mujoco.MjModel.from_xml_path(
    str(ROOT / "assets/scenes/pickup.xml")
)
data = mujoco.MjData(model)

controller = PickupController(model, data, grip_close=0.0)

# Generate a unique filename for each execution.
run_id = (
    f"viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    f"_{uuid4().hex[:8]}"
)

output = ROOT / "outputs/pickup" / f"{run_id}.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with TelemetryLogger(model, output) as log:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            controller.update(data)
            mujoco.mj_step(model, data)
            log.record(data, controller.stage)
            viewer.sync()
            time.sleep(model.opt.timestep)

            if controller.done:
                print("Result:", controller.stage, controller.reason)
                break

print("Output:", output)
