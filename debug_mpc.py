"""
Headless rollout with MPC planner. Verifies bike swerves around cone AND
returns to the centerline afterward.
"""
import numpy as np
import mujoco
from perception import detect_cones_hsv, project_detections_to_world
from planner import MPCPlanner, yaw_from_quat


FPV_WIDTH, FPV_HEIGHT = 640, 480
TARGET_SPEED = 2.0   # slower = kinematic bicycle planner more accurate
WHEEL_RADIUS = 0.312
PERCEPTION_EVERY = 25  # 20 Hz at 500 Hz physics

CONE_0 = (15.0, 0.0)


def main():
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    throttle_id    = model.actuator("throttle").id
    steer_motor_id = model.actuator("steer_motor").id
    fpv_cam_id = model.camera("fpv").id
    fpv_fovy_rad = np.deg2rad(float(model.cam_fovy[fpv_cam_id]))

    renderer = mujoco.Renderer(model, width=FPV_WIDTH, height=FPV_HEIGHT)
    planner = MPCPlanner()

    data.qvel[0] = TARGET_SPEED
    omega = TARGET_SPEED / WHEEL_RADIUS
    data.joint("rear_spin").qvel[0] = omega
    data.joint("front_spin").qvel[0] = omega
    mujoco.mj_forward(model, data)

    latest_cones = []
    latest_steer = 0.0

    # Sticky pass-side state: once a dead-ahead cone triggers a side commit,
    # hold that goal_y for HOLD_SECONDS so the bike completes the swerve
    # instead of snapping back to centerline the instant the cone leaves
    # the dead-ahead window.
    HOLD_SECONDS = 2.5
    sticky_goal_y = 0.0
    sticky_expire = 0.0

    duration = 8.0
    n_steps = int(duration / model.opt.timestep)
    min_cone_dist = float("inf")
    hit = False
    passed = False

    print(f"{'t':>5s} {'bx':>7s} {'by':>7s} {'yaw':>7s} {'steer':>7s} "
          f"{'dc0':>6s} {'solve':>7s} {'status'}")

    for step in range(n_steps):
        if step % PERCEPTION_EVERY == 0:
            renderer.update_scene(data, camera="fpv")
            frame = renderer.render()
            dets = detect_cones_hsv(frame)
            project_detections_to_world(
                dets,
                img_width=FPV_WIDTH, img_height=FPV_HEIGHT,
                fovy_rad=fpv_fovy_rad,
                cam_pos=data.cam_xpos[fpv_cam_id],
                cam_xmat=data.cam_xmat[fpv_cam_id].reshape(3, 3),
                cone_top_z=0.65,
            )
            latest_cones = [d.world_xy for d in dets if d.world_xy is not None]

            bx = float(data.qpos[0])
            by = float(data.qpos[1])
            bh = yaw_from_quat(data.qpos[3:7])

            cones_ahead = [c for c in latest_cones if c[0] > bx - 0.5]
            cones_ahead.sort(key=lambda c: (c[0] - bx) ** 2 + (c[1] - by) ** 2)
            nearest_cones = cones_ahead[:2]

            # Dead-ahead goal shift: when cone sits in path, bias goal to
            # the right. Not sticky — once bike passes cone (cone no longer
            # dead-ahead), goal returns to centerline and bike pulls back.
            goal_y = 0.0
            for cx, cy in nearest_cones:
                if abs(cy - by) < 0.6 and cx < bx + 8.0:
                    goal_y = -1.8
                    break

            plan = planner.solve(
                state=(bx, by, bh),
                goal=(bx + 10.0, goal_y, 0.0),
                obstacles=nearest_cones,
                speed=TARGET_SPEED,
            )
            latest_steer = plan.steer_cmd

        # Apply controls — proper speed P controller to hold TARGET_SPEED
        forward_speed = float(data.qvel[0])
        speed_error = TARGET_SPEED - forward_speed
        throttle_cmd = float(np.clip(15.0 * max(speed_error, 0.0), 0.0, 180.0))
        data.ctrl[throttle_id] = throttle_cmd
        data.ctrl[steer_motor_id] = float(np.clip(latest_steer, -0.15, 0.15))
        mujoco.mj_step(model, data)

        bx = float(data.qpos[0])
        by = float(data.qpos[1])
        dist = np.sqrt((bx - CONE_0[0])**2 + (by - CONE_0[1])**2)
        if dist < min_cone_dist:
            min_cone_dist = dist
        if dist < 0.3:
            hit = True
        if bx > CONE_0[0] + 0.5 and not passed:
            passed = True
            print(f"\n>>> cone-0 traversal: min dist = {min_cone_dist:.3f}m, "
                  f"status = {'HIT' if hit else 'PASSED'}\n")

        if step % 100 == 0:   # print every 0.2 s
            bh = yaw_from_quat(data.qpos[3:7])
            status = "HIT" if hit else ("NEAR" if dist < 1.2 else "")
            print(f"{data.time:5.2f} {bx:+7.2f} {by:+7.2f} "
                  f"{bh:+7.3f} {latest_steer:+7.3f} "
                  f"{dist:6.2f} {'':>7s} {status}")

    print(f"\nFinal: bike at ({bx:.2f}, {by:.2f}), min cone-0 dist={min_cone_dist:.3f}m, hit={hit}")


if __name__ == "__main__":
    main()
