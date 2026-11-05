"""
motosim — CBR500R v0.1 interactive viewer

Usage:
    cd /Users/jared/motosim
    source /path/to/venv/bin/activate  # the Python 3.12 venv with mujoco installed
    python view.py

Keyboard controls (in MuJoCo viewer):
    space       pause/resume
    right-drag  rotate camera
    scroll      zoom
    ctrl+right-drag  pan
    r           reset
    tab         toggle UI panel

What to look for in v0.1:
    1. The bike stands upright at t=0 (not falling through the ground).
    2. If you release the bike at zero velocity, it falls sideways
       (expected — a stationary bike has no gyroscopic stability).
    3. Run `python validate.py` for the self-stability test
       (bike pushed to 5 m/s should NOT fall within 10 seconds).
"""

import mujoco
import mujoco.viewer


def main():
    model = mujoco.MjModel.from_xml_path("cbr500r.xml")
    data = mujoco.MjData(model)

    # Launch interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Pull the camera back so you see the whole bike + a bit of ground
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 135      # degrees, orbital angle
        viewer.cam.elevation = -20    # degrees, tilt down slightly
        viewer.cam.lookat[:] = [0.3, 0, 0.5]  # focus roughly at bike CoM

        import time
        start = time.time()
        while viewer.is_running() and time.time() - start < 300:
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
