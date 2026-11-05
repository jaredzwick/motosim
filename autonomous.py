"""
motosim — perception-driven autonomous slalom (v0.5).

Unlike drive.py which uses a hardcoded target_lean_at(x) schedule, this script
drives the bike using ONLY what the FPV camera sees. The controller does not
know where the cones are — it only knows where its perception module says
cones are.

Pipeline every 20 Hz (every 25 physics steps at timestep=0.002s):
    1. Render the FPV camera frame
    2. Detect cones in the frame (HSV threshold -> bounding boxes)
    3. Project each detection to world (x, y) via ground-plane ray intersection
    4. Pick the nearest cone ahead of the bike -> pick lean direction
    5. Hold the chosen lean target for the next 25 physics steps

The controller runs at 500 Hz (physics timestep) applying the most recent
lean target. This matches real hardware: camera 30 FPS, inference 10-15 ms,
control loop faster.

Usage:
    mjpython autonomous.py                  # interactive, chase cam
    mjpython autonomous.py --drop 0.3       # drop 30% of frames (perception failure test)
    mjpython autonomous.py --noise 0.5      # add 0.5 m gaussian noise to each detection

Keyboard:
    space        pause/resume
    R            reset bike to start
    [ / ]        cycle fixed cameras (chase | side | overhead | fpv)
    D            toggle a HUD line in the terminal showing the chosen lean + detections
"""

import argparse
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import mujoco
import mujoco.viewer

from perception import (
    Detection,
    detect_cones_hsv,
    project_detections_to_world,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SPEED       = 5.0
MAX_LEAN_RAD       = 0.22
WHEEL_RADIUS       = 0.312
FINISH_X           = 65.0

MAX_THROTTLE       = 180.0
MAX_REAR_BRAKE     = 180.0
MAX_FRONT_BRAKE    = 350.0
FRONT_BRAKE_BIAS   = 0.65
BRAKE_DEADBAND     = 0.5

# Perception loop rate: run detection every N physics steps
PERCEPTION_HZ      = 20
PHYSICS_HZ         = 500   # 1 / timestep (0.002)
PERCEPTION_EVERY   = PHYSICS_HZ // PERCEPTION_HZ

# FPV render size. 640x480 matches see.py where detection was verified working.
# At 320x240 the cones become 4-5 pixels wide at 10m distance and HSV misses them.
FPV_WIDTH, FPV_HEIGHT = 640, 480

# Controller gain for lean target from lateral offset of nearest cone
LATERAL_LEAN_GAIN  = 0.11   # rad per meter of lateral offset
LOOKAHEAD_MAX      = 15.0   # meters of forward lookahead


# ---------------------------------------------------------------------------
# Lean target from perception
# ---------------------------------------------------------------------------
def target_lean_from_cones(
    cones_world: List[Tuple[float, float]],
    bike_x: float,
    bike_y: float,
) -> float:
    """
    Pick a lean target based on perceived cones.

    Strategy: find the nearest cone AHEAD of the bike (world_x > bike_x + 0.5,
    within LOOKAHEAD_MAX). Lean away from its lateral offset. Lean strength
    scales with proximity (close cones get stronger reaction).
    """
    ahead = [
        (cx, cy) for (cx, cy) in cones_world
        if bike_x + 0.5 < cx < bike_x + LOOKAHEAD_MAX
    ]
    if not ahead:
        return 0.0

    # Nearest by forward distance
    nearest_x, nearest_y = min(ahead, key=lambda c: c[0] - bike_x)

    lateral_offset = nearest_y - bike_y
    distance       = nearest_x - bike_x
    urgency        = max(0.0, 1.0 - distance / LOOKAHEAD_MAX)

    # Lean sign: cone on left (+y) -> lean right (positive roll)
    # (Same sign convention as drive.py)
    lean = np.sign(lateral_offset) * LATERAL_LEAN_GAIN * abs(lateral_offset) * urgency
    return float(np.clip(lean, -MAX_LEAN_RAD, MAX_LEAN_RAD))


def lean_from_quat(q: np.ndarray) -> float:
    qw, qx, qy, qz = q
    return float(np.arctan2(2.0 * (qw * qx + qy * qz),
                             1.0 - 2.0 * (qx * qx + qy * qy)))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(drop_rate: float = 0.0, noise_std: float = 0.0, max_duration: float = 60.0):
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data  = mujoco.MjData(model)

    throttle_id    = model.actuator("throttle").id
    rear_brake_id  = model.actuator("rear_brake").id
    front_brake_id = model.actuator("front_brake").id
    steer_motor_id = model.actuator("steer_motor").id

    fpv_cam_id       = model.camera("fpv").id
    fpv_fovy_deg     = float(model.cam_fovy[fpv_cam_id])
    fpv_fovy_rad     = np.deg2rad(fpv_fovy_deg)
    chase_cam_id     = model.camera("chase").id
    rear_frame_body_id = model.body("rear_frame").id

    renderer = mujoco.Renderer(model, width=FPV_WIDTH, height=FPV_HEIGHT)

    # Initial state: cruise speed
    data.qvel[0] = TARGET_SPEED
    omega = TARGET_SPEED / WHEEL_RADIUS
    data.joint("rear_spin").qvel[0]  = omega
    data.joint("front_spin").qvel[0] = omega

    # Compute forward kinematics so cam_xpos / cam_xmat are populated before
    # the first perception cycle. Without this, the FIRST render fires with
    # an uninitialized camera pose and projects to garbage.
    mujoco.mj_forward(model, data)

    # Latest perceived world cones (updated by perception loop)
    latest_cones: List[Tuple[float, float]] = []

    # Pause / reset state. Verbose ON by default -- turn off with D once you
    # confirm perception is working.
    state = {"paused": False, "reset": False, "verbose": True}

    def key_callback(keycode: int) -> None:
        if keycode == 32:        # space
            state["paused"] = not state["paused"]
            print(f"[{'PAUSED' if state['paused'] else 'RESUMED'}]")
        elif keycode == 82:      # R
            state["reset"] = True
        elif keycode == 68:      # D
            state["verbose"] = not state["verbose"]
            print(f"[verbose={'on' if state['verbose'] else 'off'}]")

    def reset_bike():
        mujoco.mj_resetData(model, data)
        data.qvel[0] = TARGET_SPEED
        data.joint("rear_spin").qvel[0]  = omega
        data.joint("front_spin").qvel[0] = omega

    rng = np.random.default_rng(42)
    tick_count = 0

    with mujoco.viewer.launch_passive(model, data,
                                       key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = rear_frame_body_id
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        print("=" * 60)
        print("motosim autonomous (v0.5) — perception-driven slalom")
        print("  space  = pause/resume")
        print("  R      = reset bike to start")
        print("  D      = toggle verbose terminal output")
        print("  [ / ]  = cycle cameras (chase | side | overhead | fpv)")
        print(f"  perception rate : {PERCEPTION_HZ} Hz (every {PERCEPTION_EVERY} physics steps)")
        print(f"  fpv render      : {FPV_WIDTH}x{FPV_HEIGHT}  fovy={fpv_fovy_deg:.1f} deg")
        print(f"  drop rate       : {drop_rate*100:.0f}% of perception cycles drop all detections")
        print(f"  noise           : {noise_std:.2f} m gaussian added to each world xy")
        print("=" * 60)

        current_lean_target = 0.0
        start = time.time()

        while viewer.is_running() and (time.time() - start) < max_duration:
            step_start = time.time()

            if state["reset"]:
                reset_bike()
                latest_cones = []
                state["reset"] = False
                start = time.time()
                tick_count = 0

            if state["paused"]:
                viewer.sync()
                time.sleep(0.01)
                continue

            # ---- perception loop (20 Hz) ----
            if tick_count % PERCEPTION_EVERY == 0:
                # Decide whether to skip this perception cycle (failure test)
                if drop_rate > 0.0 and rng.random() < drop_rate:
                    # Drop: do not update latest_cones. Controller keeps stale data.
                    pass
                else:
                    renderer.update_scene(data, camera="fpv")
                    frame = renderer.render()  # (H, W, 3) uint8
                    dets = detect_cones_hsv(frame)
                    project_detections_to_world(
                        dets,
                        img_width=FPV_WIDTH,
                        img_height=FPV_HEIGHT,
                        fovy_rad=fpv_fovy_rad,
                        cam_pos=data.cam_xpos[fpv_cam_id],
                        cam_xmat=data.cam_xmat[fpv_cam_id].reshape(3, 3),
                        ground_z=0.0,
                    )
                    cones_world = []
                    for d in dets:
                        if d.world_xy is None:
                            continue
                        wx, wy = d.world_xy
                        if noise_std > 0.0:
                            wx += rng.normal(0.0, noise_std)
                            wy += rng.normal(0.0, noise_std)
                        cones_world.append((wx, wy))
                    latest_cones = cones_world

                if state["verbose"]:
                    bx, by = float(data.qpos[0]), float(data.qpos[1])
                    print(f"t={data.time:6.2f} bike=({bx:+5.2f},{by:+5.2f}) "
                          f"cones_seen={len(latest_cones):2d} "
                          f"lean_target={current_lean_target:+.3f}")

            # ---- compute controls every physics step ----
            bike_x        = float(data.qpos[0])
            bike_y        = float(data.qpos[1])
            lean          = lean_from_quat(data.qpos[3:7])
            forward_speed = float(data.qvel[0])

            current_lean_target = target_lean_from_cones(latest_cones, bike_x, bike_y)

            # Speed controller (unchanged from drive.py)
            speed_cmd   = TARGET_SPEED if bike_x < FINISH_X else 0.0
            speed_error = speed_cmd - forward_speed
            if speed_error > 0:
                throttle_cmd   = float(np.clip(15.0 * speed_error, 0.0, MAX_THROTTLE))
                brake_fraction = 0.0
            else:
                throttle_cmd   = 0.0
                brake_fraction = float(np.clip(-speed_error / 3.0, 0.0, 1.0))

            rear_omega  = data.joint("rear_spin").qvel[0]
            front_omega = data.joint("front_spin").qvel[0]
            rear_brake_cmd = 0.0
            front_brake_cmd = 0.0
            if abs(rear_omega) > BRAKE_DEADBAND:
                rear_brake_cmd  = -(1.0 - FRONT_BRAKE_BIAS) * brake_fraction * MAX_REAR_BRAKE * np.sign(rear_omega)
            if abs(front_omega) > BRAKE_DEADBAND:
                front_brake_cmd = -FRONT_BRAKE_BIAS * brake_fraction * MAX_FRONT_BRAKE * np.sign(front_omega)

            # Lean controller (same PD, but target now comes from perception)
            steer_torque = float(np.clip(6.0 * (current_lean_target - lean), -10.0, 10.0))

            data.ctrl[throttle_id]    = throttle_cmd
            data.ctrl[rear_brake_id]  = rear_brake_cmd
            data.ctrl[front_brake_id] = front_brake_cmd
            data.ctrl[steer_motor_id] = steer_torque

            mujoco.mj_step(model, data)
            viewer.sync()
            tick_count += 1

            # Real-time pacing
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", type=float, default=0.0,
                        help="fraction of perception cycles to drop (0.0-1.0). 0.3 = 30%% blind")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="gaussian noise std (meters) added to each world (x, y) estimate")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="max wall-clock seconds before auto-exit")
    args = parser.parse_args()
    main(drop_rate=args.drop, noise_std=args.noise, max_duration=args.duration)
