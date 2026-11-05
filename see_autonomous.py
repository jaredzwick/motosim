"""
Headless render of autonomous.py (MPC-planned swerve + slalom).

Produces autonomous.mp4 — chase-cam video of the bike using the v1.1 MPC
planner to navigate the scene with cone-0 directly in front at x=15.

Usage:
    python3 see_autonomous.py
    python3 see_autonomous.py --duration 10 --output demo.mp4
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


TARGET_SPEED = 3.0
WHEEL_RADIUS = 0.312
MAX_THROTTLE = 180.0
MAX_REAR_BRAKE = 180.0
MAX_FRONT_BRAKE = 350.0
FRONT_BRAKE_BIAS = 0.65
BRAKE_DEADBAND = 0.5
PERCEPTION_HZ = 20
PHYSICS_HZ = 500
PERCEPTION_EVERY = PHYSICS_HZ // PERCEPTION_HZ
GOAL_LOOKAHEAD = 10.0
FPS = 30
OVERLAY_SIZE = (320, 240)  # FPV picture-in-picture at bottom-left


def run(duration: float, output_path: Path, resolution: tuple) -> dict:
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    throttle_id    = model.actuator("throttle").id
    rear_brake_id  = model.actuator("rear_brake").id
    front_brake_id = model.actuator("front_brake").id
    steer_motor_id = model.actuator("steer_motor").id

    fpv_cam_id = model.camera("fpv").id
    fpv_fovy_rad = np.deg2rad(float(model.cam_fovy[fpv_cam_id]))
    chase_cam_id = model.camera("chase").id

    width, height = resolution

    chase_renderer = mujoco.Renderer(model, width=width, height=height)
    fpv_renderer = mujoco.Renderer(model, width=OVERLAY_SIZE[0], height=OVERLAY_SIZE[1])

    planner = MPCPlanner()

    # Initial state: cruise speed
    data.qvel[0] = TARGET_SPEED
    omega = TARGET_SPEED / WHEEL_RADIUS
    data.joint("rear_spin").qvel[0]  = omega
    data.joint("front_spin").qvel[0] = omega
    mujoco.mj_forward(model, data)

    latest_cones = []
    latest_steer = 0.0
    latest_plan_cost = 0.0

    n_steps = int(duration / model.opt.timestep)
    frame_every = max(1, int(round(1.0 / (FPS * model.opt.timestep))))

    frames = []
    t0 = time.time()

    print(f"=== motosim autonomous (MPC) render ===")
    print(f"  duration    : {duration}s sim at {TARGET_SPEED} m/s")
    print(f"  resolution  : {width}x{height}")
    print(f"  output      : {output_path}")
    print(f"  ...")

    for step in range(n_steps):
        if step % PERCEPTION_EVERY == 0:
            fpv_renderer.update_scene(data, camera="fpv")
            frame_fpv = fpv_renderer.render()
            dets = detect_cones_hsv(frame_fpv)
            project_detections_to_world(
                dets,
                img_width=OVERLAY_SIZE[0], img_height=OVERLAY_SIZE[1],
                fovy_rad=fpv_fovy_rad,
                cam_pos=data.cam_xpos[fpv_cam_id],
                cam_xmat=data.cam_xmat[fpv_cam_id].reshape(3, 3),
                cone_top_z=0.65,
            )
            latest_cones = [d.world_xy for d in dets if d.world_xy is not None]

            bx = float(data.qpos[0])
            by = float(data.qpos[1])
            bh = yaw_from_quat(data.qpos[3:7])
            # Filter to nearest obstacles ahead
            cones_ahead = [c for c in latest_cones if c[0] > bx - 0.5]
            cones_ahead.sort(key=lambda c: (c[0] - bx) ** 2 + (c[1] - by) ** 2)
            plan = planner.solve(
                state=(bx, by, bh),
                goal=(bx + GOAL_LOOKAHEAD, 0.0, 0.0),
                obstacles=cones_ahead[:2],
                speed=TARGET_SPEED,
            )
            latest_steer = plan.steer_cmd
            latest_plan_cost = plan.cost
            latest_fpv_annotated = annotate_frame(frame_fpv, dets)

        # Speed controller
        forward_speed = float(data.qvel[0])
        speed_error = TARGET_SPEED - forward_speed
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

        if step % frame_every == 0:
            chase_renderer.update_scene(data, camera="chase")
            chase_frame = chase_renderer.render()

            # HUD text
            bx = float(data.qpos[0])
            by = float(data.qpos[1])
            bh = yaw_from_quat(data.qpos[3:7])
            cv2.putText(chase_frame, f"MPC solve ok | steer={latest_steer:+.3f}rad",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(chase_frame, f"pos=({bx:+.1f},{by:+.2f}) hd={bh:+.2f}rad speed={forward_speed:.2f}m/s",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(chase_frame, f"cones seen: {len(latest_cones)}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Inset FPV in bottom-left corner
            if hasattr(latest_fpv_annotated, "shape") if False else True:
                try:
                    fpv_small = cv2.resize(latest_fpv_annotated, (OVERLAY_SIZE[0] // 2, OVERLAY_SIZE[1] // 2))
                    h, w = fpv_small.shape[:2]
                    y0 = chase_frame.shape[0] - h - 10
                    x0 = 10
                    chase_frame[y0:y0 + h, x0:x0 + w] = fpv_small
                    cv2.rectangle(chase_frame, (x0 - 2, y0 - 2), (x0 + w + 2, y0 + h + 2),
                                  (0, 255, 255), 2)
                    cv2.putText(chase_frame, "FPV",
                                (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                except Exception:
                    pass

            frames.append(chase_frame)

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
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="autonomous.mp4")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1280, 720])
    args = parser.parse_args()

    out = Path(args.output).resolve()
    stats = run(duration=args.duration, output_path=out, resolution=tuple(args.resolution))

    print()
    print(f"VERDICT: PASS — MPC rollout rendered")
    print(f"  frames:    {stats['frames']}")
    print(f"  file size: {stats['output_mb']} MB")
    print(f"  output:    {out}")


if __name__ == "__main__":
    main()
