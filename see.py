"""
motosim — perception demo (v0.4).

Runs the slalom drive loop, grabs frames from the FPV camera, runs the HSV
cone detector on each frame, overlays bounding boxes, and saves the result
as perception.mp4.

This is OFFLINE — the detector output does NOT feed the controller yet. That's
v0.5 (perception-in-the-loop). For v0.4, perception just observes and records.

Usage:
    python see.py                        # saves perception.mp4 at 30 fps
    python see.py --duration 30          # run for 30 seconds of sim time
    python see.py --output demo.mp4
    python see.py --resolution 1280 720  # higher res for YouTube

Headless; no window opens. Writes the MP4 and a one-line summary when done.

Dependencies:
    pip install opencv-python imageio-ffmpeg
"""

import argparse
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import mujoco

from perception import detect_cones_hsv, annotate_frame


# Controller constants — kept in sync with drive.py so a future refactor
# can extract the controller into a shared module.
TARGET_SPEED       = 5.0
MAX_LEAN_RAD       = 0.22
LEAN_LOOKAHEAD     = 4.0
CONE_XS            = [10, 17, 24, 31, 38, 45, 52, 59]
CONE_YS            = [+2, -2, +2, -2, +2, -2, +2, -2]
FINISH_X           = 65.0
MAX_THROTTLE       = 180.0
MAX_REAR_BRAKE     = 180.0
MAX_FRONT_BRAKE    = 350.0
FRONT_BRAKE_BIAS   = 0.65
BRAKE_DEADBAND     = 0.5
WHEEL_RADIUS       = 0.312
FPS                = 30


def target_lean_at(x: float) -> float:
    if x < CONE_XS[0] - LEAN_LOOKAHEAD or x > FINISH_X:
        return 0.0
    for i, cx in enumerate(CONE_XS):
        if x < cx + 2.0:
            cone_y = CONE_YS[i]
            dist = cx - x
            if dist > LEAN_LOOKAHEAD or dist < -1.5:
                frac = 0.0
            else:
                frac = max(0.0, 1.0 - abs(dist) / LEAN_LOOKAHEAD)
            return np.sign(cone_y) * MAX_LEAN_RAD * frac
    return 0.0


def lean_from_quat(q: np.ndarray) -> float:
    qw, qx, qy, qz = q
    return np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))


def run(duration: float, output_path: Path, resolution: tuple, show_overlay: bool) -> dict:
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    throttle_id    = model.actuator("throttle").id
    rear_brake_id  = model.actuator("rear_brake").id
    front_brake_id = model.actuator("front_brake").id
    steer_motor_id = model.actuator("steer_motor").id

    fpv_cam_name = "fpv"

    # Initial state: at cruise speed
    data.qvel[0] = TARGET_SPEED
    omega = TARGET_SPEED / WHEEL_RADIUS
    data.joint("rear_spin").qvel[0]  = omega
    data.joint("front_spin").qvel[0] = omega

    width, height = resolution
    renderer = mujoco.Renderer(model, width=width, height=height)

    n_steps = int(duration / model.opt.timestep)
    frame_every = max(1, int(round(1.0 / (FPS * model.opt.timestep))))

    frames: list[np.ndarray] = []
    detections_per_frame: list[int] = []
    t0 = time.time()

    print(f"=== motosim perception demo ===")
    print(f"  duration       : {duration}s sim")
    print(f"  output         : {output_path}")
    print(f"  resolution     : {width}x{height}")
    print(f"  sim timestep   : {model.opt.timestep}s")
    print(f"  frames to write: {n_steps // frame_every}")
    print(f"  ...")

    for step in range(n_steps):
        chassis_x     = float(data.qpos[0])
        chassis_quat  = data.qpos[3:7].copy()
        lean          = lean_from_quat(chassis_quat)
        forward_speed = float(data.qvel[0])

        # speed controller
        speed_cmd   = TARGET_SPEED if chassis_x < FINISH_X else 0.0
        speed_error = speed_cmd - forward_speed
        if speed_error > 0:
            throttle_cmd = float(np.clip(15.0 * speed_error, 0.0, MAX_THROTTLE))
            brake_fraction = 0.0
        else:
            throttle_cmd = 0.0
            brake_fraction = float(np.clip(-speed_error / 3.0, 0.0, 1.0))

        rear_omega  = data.joint("rear_spin").qvel[0]
        front_omega = data.joint("front_spin").qvel[0]
        rear_brake_cmd  = 0.0
        front_brake_cmd = 0.0
        if abs(rear_omega) > BRAKE_DEADBAND:
            rear_brake_cmd  = -(1.0 - FRONT_BRAKE_BIAS) * brake_fraction * MAX_REAR_BRAKE * np.sign(rear_omega)
        if abs(front_omega) > BRAKE_DEADBAND:
            front_brake_cmd = -FRONT_BRAKE_BIAS * brake_fraction * MAX_FRONT_BRAKE * np.sign(front_omega)

        # lean controller: command steer ANGLE (position actuator). Negative
        # gain for correct sign convention — see autonomous.py comment.
        lean_target  = target_lean_at(chassis_x)
        steer_angle_cmd = float(np.clip(-0.3 * lean_target, -0.15, 0.15))

        data.ctrl[throttle_id]    = throttle_cmd
        data.ctrl[rear_brake_id]  = rear_brake_cmd
        data.ctrl[front_brake_id] = front_brake_cmd
        data.ctrl[steer_motor_id] = steer_angle_cmd

        mujoco.mj_step(model, data)

        # Grab a frame every frame_every steps
        if step % frame_every == 0:
            renderer.update_scene(data, camera=fpv_cam_name)
            frame = renderer.render()  # (H, W, 3) uint8 RGB

            cones = detect_cones_hsv(frame)
            if show_overlay:
                frame = annotate_frame(frame, cones)

            frames.append(frame)
            detections_per_frame.append(len(cones))

    wall = time.time() - t0
    print(f"  sim done in {wall:.1f}s wall. Writing MP4 ...")

    # Write the video
    with imageio.get_writer(str(output_path), fps=FPS, codec="libx264",
                             quality=8, macro_block_size=1) as writer:
        for frame in frames:
            # imageio expects uint8 RGB
            writer.append_data(frame)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    total_detections = sum(detections_per_frame)
    unique_detections_per_frame_avg = total_detections / max(1, len(detections_per_frame))

    return {
        "frames": len(frames),
        "total_detections": total_detections,
        "avg_cones_in_view": round(unique_detections_per_frame_avg, 2),
        "output_mb": round(size_mb, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0,
                        help="sim duration in seconds (default 15)")
    parser.add_argument("--output", type=str, default="perception.mp4")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                        help="width height, e.g. --resolution 1280 720")
    parser.add_argument("--no-overlay", action="store_true",
                        help="skip bounding-box overlay (raw fpv only)")
    args = parser.parse_args()

    out_path = Path(args.output).resolve()
    stats = run(
        duration=args.duration,
        output_path=out_path,
        resolution=tuple(args.resolution),
        show_overlay=not args.no_overlay,
    )

    print()
    print(f"VERDICT: PASS — perception demo rendered")
    print(f"  frames:           {stats['frames']}")
    print(f"  total detections: {stats['total_detections']}")
    print(f"  avg cones/frame:  {stats['avg_cones_in_view']}")
    print(f"  file size:        {stats['output_mb']} MB")
    print(f"  output:           {out_path}")


if __name__ == "__main__":
    main()
