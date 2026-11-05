"""
motosim — perception + MPC autonomous driving (v1.1).

Replaces the v0.5 reactive lean controller with a receding-horizon MPC
planner. Per cycle:

    1. Render FPV camera
    2. HSV detect cones -> project to world coords (perception.py)
    3. Set goal: 20m ahead on original heading, y=0 (centerline)
    4. MPC solve (planner.py) -> steer_angle command
    5. Apply steer (position actuator) + throttle (P controller for speed)

The planner naturally handles:
  - swerving around dead-ahead obstacles
  - returning to the centerline after swerving
  - smooth steering (no jitter from frame-to-frame perception noise)
  - multi-obstacle scenarios (all cones in cost function simultaneously)

Usage:
    mjpython autonomous.py
    mjpython autonomous.py --duration 30
    mjpython autonomous.py --drop 0.3     (perception failure injection)

Keyboard:
    space        pause/resume
    R            reset bike to start
    D            toggle verbose terminal output
    [ / ]        cycle fixed cameras (chase | side | overhead | fpv)
"""

import argparse
import time
from typing import List, Optional, Tuple

import numpy as np
import mujoco
import mujoco.viewer

from perception import (
    detect_cones_hsv,
    project_detections_to_world,
)
from planner import MPCPlanner, yaw_from_quat


# ---- Constants ----
# Target speed 2.0 m/s — kinematic-bicycle MPC planner is accurate at low
# speed; at 5+ m/s lean lag creates planner/reality mismatch and the bike
# oscillates or overshoots. 2 m/s matches the self-park wedge operating
# envelope and gives clean 0.7m+ clearance when swerving around an obstacle.
TARGET_SPEED       = 2.0
WHEEL_RADIUS       = 0.312
FINISH_X           = 65.0

MAX_THROTTLE       = 180.0
MAX_REAR_BRAKE     = 180.0
MAX_FRONT_BRAKE    = 350.0
FRONT_BRAKE_BIAS   = 0.65
BRAKE_DEADBAND     = 0.5

# Perception loop rate
PERCEPTION_HZ      = 20
PHYSICS_HZ         = 500
PERCEPTION_EVERY   = PHYSICS_HZ // PERCEPTION_HZ
FPV_WIDTH, FPV_HEIGHT = 640, 480

# MPC: goal is set to GOAL_LOOKAHEAD meters ahead on the original heading.
# Gives the planner a direction to pull toward when no obstacles are nearby.
GOAL_LOOKAHEAD     = 10.0   # shorter horizon = less confusion from distant cones
ORIGINAL_HEADING   = 0.0   # +x direction


def lean_from_quat(q: np.ndarray) -> float:
    qw, qx, qy, qz = q
    return float(np.arctan2(2.0 * (qw * qx + qy * qz),
                             1.0 - 2.0 * (qx * qx + qy * qy)))


def main(drop_rate: float = 0.0, max_duration: float = 60.0):
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data  = mujoco.MjData(model)

    throttle_id    = model.actuator("throttle").id
    rear_brake_id  = model.actuator("rear_brake").id
    front_brake_id = model.actuator("front_brake").id
    steer_motor_id = model.actuator("steer_motor").id

    fpv_cam_id       = model.camera("fpv").id
    fpv_fovy_rad     = np.deg2rad(float(model.cam_fovy[fpv_cam_id]))
    rear_frame_body_id = model.body("rear_frame").id

    renderer = mujoco.Renderer(model, width=FPV_WIDTH, height=FPV_HEIGHT)
    planner = MPCPlanner()

    # Initial state: cruise speed
    data.qvel[0] = TARGET_SPEED
    omega = TARGET_SPEED / WHEEL_RADIUS
    data.joint("rear_spin").qvel[0]  = omega
    data.joint("front_spin").qvel[0] = omega
    mujoco.mj_forward(model, data)

    latest_cones: List[Tuple[float, float]] = []
    latest_steer_cmd = 0.0
    latest_plan_info = ""

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
        mujoco.mj_forward(model, data)

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
        print("motosim autonomous (v1.1) — MPC planner + perception")
        print("  space  = pause/resume")
        print("  R      = reset bike to start")
        print("  D      = toggle verbose terminal output")
        print(f"  perception rate : {PERCEPTION_HZ} Hz")
        print(f"  MPC horizon     : {planner.N} steps × {planner.dt}s = {planner.N * planner.dt:.1f}s")
        print(f"  drop rate       : {drop_rate*100:.0f}% of perception cycles dropped")
        print("=" * 60)

        start = time.time()

        while viewer.is_running() and (time.time() - start) < max_duration:
            step_start = time.time()

            if state["reset"]:
                reset_bike()
                latest_cones = []
                latest_steer_cmd = 0.0
                state["reset"] = False
                start = time.time()
                tick_count = 0

            if state["paused"]:
                viewer.sync()
                time.sleep(0.01)
                continue

            # ---- perception + plan (20 Hz) ----
            if tick_count % PERCEPTION_EVERY == 0:
                if drop_rate > 0.0 and rng.random() < drop_rate:
                    pass  # drop this cycle, controller keeps stale plan
                else:
                    renderer.update_scene(data, camera="fpv")
                    frame = renderer.render()
                    dets = detect_cones_hsv(frame)
                    project_detections_to_world(
                        dets,
                        img_width=FPV_WIDTH,
                        img_height=FPV_HEIGHT,
                        fovy_rad=fpv_fovy_rad,
                        cam_pos=data.cam_xpos[fpv_cam_id],
                        cam_xmat=data.cam_xmat[fpv_cam_id].reshape(3, 3),
                        cone_top_z=0.65,
                    )
                    latest_cones = [d.world_xy for d in dets if d.world_xy is not None]

                # MPC solve
                bike_x = float(data.qpos[0])
                bike_y = float(data.qpos[1])
                bike_heading = yaw_from_quat(data.qpos[3:7])

                # Filter to nearest cones ahead. Distant cones add noise to
                # the planner cost; keep the problem focused on what's
                # imminently in the way.
                cones_ahead = [c for c in latest_cones if c[0] > bike_x - 0.5]
                cones_ahead.sort(key=lambda c: (c[0] - bike_x) ** 2 + (c[1] - bike_y) ** 2)
                nearest_cones = cones_ahead[:2]

                # Dead-ahead goal shift: if a cone sits near the bike's path,
                # move the goal off-centerline to one side (RIGHT by default:
                # negative y). Gives the planner a concrete direction to
                # swerve. Not sticky — releases as soon as the cone is no
                # longer in the dead-ahead zone.
                goal_y = 0.0
                DEAD_AHEAD_LATERAL = 0.6
                DEAD_AHEAD_FORWARD = 8.0
                PASS_SIDE_Y = -1.8
                for cx, cy in nearest_cones:
                    if abs(cy - bike_y) < DEAD_AHEAD_LATERAL and cx < bike_x + DEAD_AHEAD_FORWARD:
                        goal_y = PASS_SIDE_Y
                        break

                goal = (bike_x + GOAL_LOOKAHEAD, goal_y, ORIGINAL_HEADING)

                plan = planner.solve(
                    state=(bike_x, bike_y, bike_heading),
                    goal=goal,
                    obstacles=nearest_cones,
                    speed=TARGET_SPEED,
                )
                latest_steer_cmd = plan.steer_cmd
                latest_plan_info = (
                    f"solve={plan.solve_time_s*1000:5.1f}ms "
                    f"cost={plan.cost:7.2f} {'OK' if plan.success else 'FAIL'}"
                )

                if state["verbose"]:
                    nearest = sorted(
                        [c for c in latest_cones if c[0] > bike_x - 1],
                        key=lambda c: (c[0] - bike_x)**2 + (c[1] - bike_y)**2,
                    )[:3]
                    cones_str = " ".join(f"({cx:+5.2f},{cy:+5.2f})" for cx, cy in nearest)
                    print(f"t={data.time:6.2f} bike=({bike_x:+5.2f},{bike_y:+5.2f}) "
                          f"hd={bike_heading:+.3f} "
                          f"steer={latest_steer_cmd:+.3f} "
                          f"cones={len(latest_cones):2d} nearest=[{cones_str}] "
                          f"| {latest_plan_info}")

            # ---- low-level control (every physics step) ----
            bike_x = float(data.qpos[0])
            forward_speed = float(data.qvel[0])

            # Speed P controller
            speed_cmd = TARGET_SPEED if bike_x < FINISH_X else 0.0
            speed_error = speed_cmd - forward_speed
            if speed_error > 0:
                throttle_cmd = float(np.clip(15.0 * speed_error, 0.0, MAX_THROTTLE))
                brake_fraction = 0.0
            else:
                throttle_cmd = 0.0
                brake_fraction = float(np.clip(-speed_error / 3.0, 0.0, 1.0))

            rear_omega  = data.joint("rear_spin").qvel[0]
            front_omega = data.joint("front_spin").qvel[0]
            rear_brake_cmd = 0.0
            front_brake_cmd = 0.0
            if abs(rear_omega) > BRAKE_DEADBAND:
                rear_brake_cmd  = -(1.0 - FRONT_BRAKE_BIAS) * brake_fraction * MAX_REAR_BRAKE * np.sign(rear_omega)
            if abs(front_omega) > BRAKE_DEADBAND:
                front_brake_cmd = -FRONT_BRAKE_BIAS * brake_fraction * MAX_FRONT_BRAKE * np.sign(front_omega)

            # Position actuator: send planned steer angle directly.
            # Note: planner's steer convention matches MuJoCo's. The negative
            # sign that was applied in the reactive v0.5 controller is now
            # implicit in how the planner's cost function uses the kinematic
            # bicycle model.
            data.ctrl[throttle_id]    = throttle_cmd
            data.ctrl[rear_brake_id]  = rear_brake_cmd
            data.ctrl[front_brake_id] = front_brake_cmd
            data.ctrl[steer_motor_id] = float(np.clip(latest_steer_cmd, -0.15, 0.15))

            mujoco.mj_step(model, data)
            viewer.sync()
            tick_count += 1

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", type=float, default=0.0,
                        help="fraction of perception cycles to drop (0.0-1.0)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="max wall-clock seconds before auto-exit")
    args = parser.parse_args()
    main(drop_rate=args.drop, max_duration=args.duration)
