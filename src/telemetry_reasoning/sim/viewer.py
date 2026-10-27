import time
from pathlib import Path

import mujoco
import mujoco.viewer

from telemetry_reasoning.sim.pickup import PickupController


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "assets/scenes/pickup.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

controller = PickupController(model, data)

print("Starting autonomous pickup...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        controller.update(data)

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)