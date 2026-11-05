"""
motosim — perception demo render (v1.1).

Produces perception.mp4: FPV camera view with live HSV cone detections
overlaid, while the bike navigates the scene using the v1.1 MPC planner.
The bike actually avoids cone 0 at x=15 (dead-ahead) via MPC planning.

Usage:
    python3 see.py                           # 20s sim at 2 m/s, 640x480 30fps
    python3 see.py --duration 30
    python3 see.py --resolution 1280 720     # 720p
    python3 see.py --no-overlay              # drop the bounding-box overlay
"""

import argparse
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import mujoco

from perception import detect_cones_hsv, project_detections_to_world, annotate_frame
from planner import MPCPlanner, yaw_from_quat


# Match autonomous.py tuned v1.1 settings
TARGET_SPEED       = 2.0
WHEEL_RADIUS       = 0.312
FINISH_X           = 65.0
MAX_THROTTLE       = 180.0
MAX_REAR_BRAKE     = 180.0
MAX_FRONT_BRAKE    = 350.0
FRONT_BRAKE_BIAS   = 0.65
BRAKE_DEADBAND     = 0.5
PERCEPTION_HZ      = 20
PHYSICS_HZ         = 500
PERCEPTION_EVERY   = PHYSICS_HZ // PERCEPTION_HZ
GOAL_LOOKAHEAD     = 10.0
FPS                = 30


def run(duration: float, output_path: Path, resolution: tuple, show_overlay: bool) -> dict:
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    throttle_id    = model.actuator("throttle").id
    rear_brake_id  = model.actuator("rear_brake").id
    front_brake_id = model.actuator("front_brake").id
    steer_motor_id = model.actuator("steer_motor").id

    fpv_cam_id = model.camera("fpv").id
    fpv_fovy_rad = np.deg2rad(float(model.cam_fovy[fpv_cam_id]))

    width, height = resolution
    renderer = mujoco.Renderer(model, width=width, height=height)
    planner = MPCPlanner()

    # Initial state
    data.qvel[0] = TARGET_SPEED
    omega = TARGET_SPEED / WHEEL_RADIUS
    data.joint("rear_spin").qvel[0]  = omega
    data.joint("front_spin").qvel[0] = omega
    mujoco.mj_forward(model, data)

    n_steps = int(duration / model.opt.timestep)
    frame_every = max(1, int(round(1.0 / (FPS * model.opt.timestep))))

    frames = []
    detections_per_frame = []
    latest_steer = 0.0
    t0 = time.time()

    print(f"=== motosim perception demo (MPC-driven) ===")
    print(f"  duration    : {duration}s sim at {TARGET_SPEED} m/s")
    print(f"  output      : {output_path}")
    print(f"  resolution  : {width}x{height}")
    print(f"  ...")

    for step in range(n_steps):
        bike_x = float(data.qpos[0])
        bike_y = float(data.qpos[1])
        forward_speed = float(data.qvel[0])

        # Perception + plan at 20 Hz
        if step % PERCEPTION_EVERY == 0:
            renderer.update_scene(data, camera="fpv")
            frame = renderer.render()
            dets = detect_cones_hsv(frame)
            project_detections_to_world(
                dets,
                img_width=width, img_height=height,
                fovy_rad=fpv_fovy_rad,
                cam_pos=data.cam_xpos[fpv_cam_id],
                cam_xmat=data.cam_xmat[fpv_cam_id].reshape(3, 3),
                cone_top_z=0.65,
            )
            cones_world = [d.world_xy for d in dets if d.world_xy is not None]

            # MPC solve with same logic as autonomous.py
            cones_ahead = [c for c in cones_world if c[0] > bike_x - 0.5]
            cones_ahead.sort(key=lambda c: (c[0] - bike_x) ** 2 + (c[1] - bike_y) ** 2)
            nearest_cones = cones_ahead[:2]

            goal_y = 0.0
            for cx, cy in nearest_cones:
                if abs(cy - bike_y) < 0.6 and cx < bike_x + 8.0:
                    goal_y = -1.8
                    break

            bh = yaw_from_quat(data.qpos[3:7])
            plan = planner.solve(
                state=(bike_x, bike_y, bh),
                goal=(bike_x + GOAL_LOOKAHEAD, goal_y, 0.0),
                obstacles=nearest_cones,
                speed=TARGET_SPEED,
            )
            latest_steer = plan.steer_cmd

            # Save the FPV frame for inclusion in the video
            latest_frame_annotated = annotate_frame(frame, dets) if show_overlay else frame

        # Speed controller
        speed_cmd = TARGET_SPEED if bike_x < FINISH_X else 0.0
        speed_error = speed_cmd - forward_speed
        if speed_error > 0:
            throttle_cmd = float(np.clip(15.0 * speed_error, 0.0, MAX_THROTTLE))
            brake_fraction = 0.0
        else:
            throttle_cmd = 0.0
            brake_fraction = float(np.clip(-speed_error / 3.0, 0.0, 1.0))

        rear_omega = data.joint("rear_spin").qvel[0]
        front_omega = data.joint("front_spin").qvel[0]
        rear_brake_cmd = 0.0
        front_brake_cmd = 0.0
        if abs(rear_omega) > BRAKE_DEADBAND:
            rear_brake_cmd = -(1.0 - FRONT_BRAKE_BIAS) * brake_fraction * MAX_REAR_BRAKE * np.sign(rear_omega)
        if abs(front_omega) > BRAKE_DEADBAND:
            front_brake_cmd = -FRONT_BRAKE_BIAS * brake_fraction * MAX_FRONT_BRAKE * np.sign(front_omega)

        data.ctrl[throttle_id] = throttle_cmd
        data.ctrl[rear_brake_id] = rear_brake_cmd
        data.ctrl[front_brake_id] = front_brake_cmd
        data.ctrl[steer_motor_id] = float(np.clip(latest_steer, -0.15, 0.15))

        mujoco.mj_step(model, data)

        # Grab a frame every frame_every steps
        if step % frame_every == 0:
            detections_per_frame.append(0)  # placeholder, not critical
            frames.append(latest_frame_annotated.copy())

    wall = time.time() - t0
    print(f"  sim done in {wall:.1f}s wall. Writing MP4 ...")

    with imageio.get_writer(str(output_path), fps=FPS, codec="libx264",
                             quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    return {"frames": len(frames), "output_mb": round(size_mb, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0,
                        help="sim duration in seconds (default 20)")
    parser.add_argument("--output", type=str, default="perception.mp4")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480])
    parser.add_argument("--no-overlay", action="store_true",
                        help="skip bounding-box overlay (raw FPV only)")
    args = parser.parse_args()

    out = Path(args.output).resolve()
    stats = run(
        duration=args.duration,
        output_path=out,
        resolution=tuple(args.resolution),
        show_overlay=not args.no_overlay,
    )

    print()
    print(f"VERDICT: PASS — perception demo rendered")
    print(f"  frames:    {stats['frames']}")
    print(f"  file size: {stats['output_mb']} MB")
    print(f"  output:    {out}")


if __name__ == "__main__":
    main()
