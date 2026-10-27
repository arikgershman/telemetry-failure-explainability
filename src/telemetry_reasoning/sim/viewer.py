"""Interactive pickup playback (on macOS run with mjpython)."""
import time
from pathlib import Path
import mujoco
import mujoco.viewer
from telemetry_reasoning.sim.pickup import PickupController
from telemetry_reasoning.sim.telemetry import TelemetryLogger

ROOT = Path(__file__).resolve().parents[3]
model = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/scenes/pickup.xml'))
data = mujoco.MjData(model)
controller = PickupController(model, data)
with TelemetryLogger(model, ROOT / 'outputs/pickup_viewer.csv') as log:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            controller.update(data)
            mujoco.mj_step(model, data)
            log.record(data, controller.stage)
            viewer.sync()
            time.sleep(model.opt.timestep)
            if controller.done:
                print('Result:', controller.stage, controller.reason)
                break
