from pathlib import Path
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "assets/mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

import time

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)