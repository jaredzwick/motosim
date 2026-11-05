"""
Headless rollout: run autonomous.py's control loop forward in time without
the viewer. Print bike state + actual_lean + lean_target + whether cone-0
is hit.

Run: python3 debug_rollout.py
"""

import numpy as np
import mujoco
from perception import detect_cones_hsv, project_detections_to_world
from autonomous import target_lean_from_cones, lean_from_quat


FPV_WIDTH, FPV_HEIGHT = 640, 480
TARGET_SPEED = 5.0
WHEEL_RADIUS = 0.312
MAX_THROTTLE = 180.0
MAX_REAR_BRAKE = 180.0
MAX_FRONT_BRAKE = 350.0
FRONT_BRAKE_BIAS = 0.65
BRAKE_DEADBAND = 0.5
KP_LEAN = 6.0
PERCEPTION_EVERY = 25  # match autonomous.py (20 Hz at timestep 0.002)

CONE_0_WORLD = (15.0, 0.0)
CONE_HIT_RADIUS = 0.3  # meters; if bike passes within this distance, treats as hit


def run_rollout(duration: float = 6.0, print_every: float = 0.25) -> None:
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    throttle_id    = model.actuator("throttle").id
    rear_brake_id  = model.actuator("rear_brake").id
    front_brake_id = model.actuator("front_brake").id
    steer_motor_id = model.actuator("steer_motor").id

    fpv_cam_id = model.camera("fpv").id
    fpv_fovy_rad = np.deg2rad(float(model.cam_fovy[fpv_cam_id]))

    renderer = mujoco.Renderer(model, width=FPV_WIDTH, height=FPV_HEIGHT)

    # Initial state
    data.qvel[0] = TARGET_SPEED
    omega = TARGET_SPEED / WHEEL_RADIUS
    data.joint("rear_spin").qvel[0] = omega
    data.joint("front_spin").qvel[0] = omega
    mujoco.mj_forward(model, data)

    latest_cones = []
    current_lean_target = 0.0
    n_steps = int(duration / model.opt.timestep)
    print_every_steps = int(print_every / model.opt.timestep)

    # Track min distance to cone-0
    min_cone_dist = float("inf")
    cone_passed = False
    hit_cone = False

    print(f"{'t':>5s} {'bike_x':>7s} {'bike_y':>7s} {'lean_act':>9s} {'lean_tgt':>9s} "
          f"{'steer_q':>8s} {'steer_tq':>9s} {'cones':>5s} {'dist_c0':>7s} {'status':>10s}")

    for step in range(n_steps):
        chassis_x = float(data.qpos[0])
        chassis_y = float(data.qpos[1])
        chassis_quat = data.qpos[3:7].copy()
        lean = lean_from_quat(chassis_quat)
        forward_speed = float(data.qvel[0])

        # Perception at 20 Hz
        if step % PERCEPTION_EVERY == 0:
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
            cones_world = [d.world_xy for d in dets if d.world_xy is not None]
            latest_cones = cones_world

        current_lean_target = target_lean_from_cones(latest_cones, chassis_x, chassis_y)

        # Speed controller
        speed_cmd = TARGET_SPEED if chassis_x < 65.0 else 0.0
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

        # Position-controlled steer. Positive gain: countersteering means
        # steer LEFT (positive steer angle) to lean RIGHT (positive lean).
        # Small gain — the physics is extremely sensitive at speed; 0.1 rad
        # steer at 5 m/s produces aggressive lean.
        steer_torque = float(np.clip(-0.3 * current_lean_target, -0.15, 0.15))
        steer_angle = float(data.joint("steer").qpos[0])

        data.ctrl[throttle_id] = throttle_cmd
        data.ctrl[rear_brake_id] = rear_brake_cmd
        data.ctrl[front_brake_id] = front_brake_cmd
        data.ctrl[steer_motor_id] = steer_torque

        # Track closest approach to cone 0
        cone_dist = np.sqrt((chassis_x - CONE_0_WORLD[0])**2 + (chassis_y - CONE_0_WORLD[1])**2)
        if cone_dist < min_cone_dist:
            min_cone_dist = cone_dist
        if cone_dist < CONE_HIT_RADIUS:
            hit_cone = True
        if chassis_x > CONE_0_WORLD[0] + 0.5 and not cone_passed:
            cone_passed = True
            status = "PASSED" if min_cone_dist > CONE_HIT_RADIUS else "HIT"
            print(f"\n>>> cone-0 traversal: min dist = {min_cone_dist:.3f}m, status = {status}\n")

        # Print state at regular intervals
        if step % print_every_steps == 0:
            status = ""
            if cone_dist < 1.0:
                status = "NEAR"
            if hit_cone:
                status = "HIT"
            print(f"{data.time:5.2f} {chassis_x:7.2f} {chassis_y:7.2f} "
                  f"{lean:+9.3f} {current_lean_target:+9.3f} "
                  f"{steer_angle:+8.3f} {steer_torque:+9.3f} "
                  f"{len(latest_cones):5d} {cone_dist:7.2f} {status:>10s}")

        mujoco.mj_step(model, data)

    print(f"\nFinal: min cone-0 distance = {min_cone_dist:.3f}m, hit = {hit_cone}")


if __name__ == "__main__":
    run_rollout()
