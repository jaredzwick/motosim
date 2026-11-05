"""
motosim — drive the CBR500R through a cone slalom.

Loads scene.xml (bike + track), launches an interactive viewer with the chase
camera, and runs a two-channel controller:

    - Speed: P controller on forward velocity -> rear wheel motor torque
    - Lean:  P controller on target lean angle -> steer motor torque

The slalom target-lean schedule is derived from the bike's current x position:
cones alternate y=+2 and y=-2 at x=10,17,24,...,59. Bike leans AWAY from each
cone as it approaches, flipping direction between cones.

Usage:
    mjpython drive.py

Press space to pause, [ and ] to cycle cameras in the viewer.

To record video for YouTube: screen-record the viewer window with QuickTime
(cmd+shift+5 on macOS -> "Record Selected Portion"). Aim for a 30-45 second
clip. Trim out the first second of settling.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer


# ---------------------------------------------------------------------------
# Scene constants (keep in sync with scene.xml)
# ---------------------------------------------------------------------------
CONE_XS = [10, 17, 24, 31, 38, 45, 52, 59]
CONE_YS = [+2, -2, +2, -2, +2, -2, +2, -2]
FINISH_X = 65.0

TARGET_SPEED = 5.0           # m/s cruise speed
MAX_LEAN_RAD = 0.22          # ~12.5 deg peak lean through cones
LEAN_LOOKAHEAD = 4.0         # start leaning this many meters before the cone


# ---------------------------------------------------------------------------
# Lean-target schedule
# ---------------------------------------------------------------------------
def target_lean_at(x: float) -> float:
    """Return the commanded lean (rad) given the bike's forward x position.

    Leaning sign convention: positive lean = bike rolled so the bike's +y side
    (left) goes down. To avoid a cone at y=+2, the bike should pass on the
    right (y<0), which means it leans toward -y = negative lean.
    """
    if x < CONE_XS[0] - LEAN_LOOKAHEAD:
        return 0.0
    if x > FINISH_X:
        return 0.0

    # Find the cone the bike is approaching or just passed
    for i, cx in enumerate(CONE_XS):
        if x < cx + 2.0:
            cone_y = CONE_YS[i]
            # Smooth in: ramp from 0 to full lean over LEAN_LOOKAHEAD meters
            dist_to_cone = cx - x
            if dist_to_cone > LEAN_LOOKAHEAD:
                frac = 0.0
            elif dist_to_cone < -1.5:
                frac = 0.0  # already passed, straighten out
            else:
                # bell curve: peak lean at cone, zero before/after
                frac = max(0.0, 1.0 - abs(dist_to_cone) / LEAN_LOOKAHEAD)

            # Lean TOWARD -y when cone is at +y (so bike curves right, passes
            # the cone on the right). Sign convention: positive roll about world
            # +x = bike's left side rises = bike leans right = path curves right
            # toward -y. So to avoid a +y cone, we want positive lean.
            return np.sign(cone_y) * MAX_LEAN_RAD * frac

    return 0.0


# ---------------------------------------------------------------------------
# Lean extraction from the chassis quaternion
# ---------------------------------------------------------------------------
def lean_from_quat(q: np.ndarray) -> float:
    """Extract lean (roll about world x) from a chassis quat (w,x,y,z)."""
    qw, qx, qy, qz = q
    return np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    throttle_id    = model.actuator("throttle").id
    rear_brake_id  = model.actuator("rear_brake").id
    front_brake_id = model.actuator("front_brake").id
    steer_motor_id = model.actuator("steer_motor").id

    chase_cam_id = model.camera("chase").id
    rear_frame_body_id = model.body("rear_frame").id

    # Torque caps (match cbr500r.xml ctrlrange values)
    MAX_THROTTLE    = 180.0
    MAX_REAR_BRAKE  = 180.0
    MAX_FRONT_BRAKE = 350.0
    # Brake distribution: motorcycles use more front than rear.
    FRONT_BRAKE_BIAS = 0.65
    # Minimum wheel angular velocity (rad/s) before brakes engage. Below this
    # the wheel is treated as stationary and brakes apply zero torque.
    BRAKE_DEADBAND_OMEGA = 0.5

    # Start the bike already at cruise speed. Starting from rest is a bad
    # idea for a motorcycle sim: countersteering only works at speed, and
    # the bike would flop sideways before the P controller could ramp up.
    # Launching at cruise bypasses the "stand up from a stop" problem that's
    # a year-two concern (Honda Riding Assist solves this mechanically, not
    # with software).
    data.qvel[0] = TARGET_SPEED
    wheel_omega_0 = TARGET_SPEED / 0.312
    data.joint("rear_spin").qvel[0]  = wheel_omega_0
    data.joint("front_spin").qvel[0] = wheel_omega_0

    # Pause / reset keyboard state. Space toggles pause, R resets.
    state = {"paused": False, "reset": False}

    def key_callback(keycode: int) -> None:
        # MuJoCo passes GLFW keycodes. Space = 32, R = 82.
        if keycode == 32:        # space
            state["paused"] = not state["paused"]
            print(f"[{'PAUSED' if state['paused'] else 'RESUMED'}]")
        elif keycode == 82:      # R (uppercase in GLFW keycode)
            state["reset"] = True
            print("[RESET]")

    with mujoco.viewer.launch_passive(model, data,
                                       key_callback=key_callback) as viewer:
        # Start in TRACKING mode: camera follows the bike AND can be
        # orbited / zoomed with the mouse. Press [ or ] to jump to the
        # locked chase / side / overhead fixed cameras for recording.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = rear_frame_body_id
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 135      # degrees, orbit around vertical
        viewer.cam.elevation = -20    # degrees, tilt down slightly

        print("=" * 60)
        print("motosim drive")
        print("  space        = pause / resume")
        print("  R            = reset bike to start")
        print("  [ / ]        = cycle fixed cameras (chase | side | overhead)")
        print("  mouse drag   = orbit the tracking camera")
        print("  scroll       = zoom the tracking camera")
        print("  ESC          = return to free/tracking camera from a fixed one")
        print(f"Target speed: {TARGET_SPEED} m/s")
        print(f"Slalom: 8 cones at x = {CONE_XS}")
        print("=" * 60)

        def reset_bike():
            mujoco.mj_resetData(model, data)
            data.qvel[0] = TARGET_SPEED
            data.joint("rear_spin").qvel[0]  = wheel_omega_0
            data.joint("front_spin").qvel[0] = wheel_omega_0

        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            if state["reset"]:
                reset_bike()
                state["reset"] = False
                start = time.time()

            # ---- read state ----
            chassis_x       = float(data.qpos[0])
            chassis_quat    = data.qpos[3:7].copy()
            lean            = lean_from_quat(chassis_quat)
            forward_speed   = float(data.qvel[0])

            # ---- speed controller: split into throttle + friction-model brakes ----
            speed_cmd       = TARGET_SPEED if chassis_x < FINISH_X else 0.0
            speed_error     = speed_cmd - forward_speed

            if speed_error > 0:
                # Accelerating: throttle only, no brake
                throttle_cmd = np.clip(15.0 * speed_error, 0.0, MAX_THROTTLE)
                brake_fraction = 0.0
            else:
                # Decelerating: brake only, no throttle
                throttle_cmd = 0.0
                brake_fraction = np.clip(-speed_error / 3.0, 0.0, 1.0)

            # Friction-model brakes: only apply torque if the wheel is actually
            # spinning forward. Direction of brake torque always opposes motion.
            rear_omega  = data.joint("rear_spin").qvel[0]
            front_omega = data.joint("front_spin").qvel[0]
            rear_brake_cmd  = 0.0
            front_brake_cmd = 0.0
            if abs(rear_omega) > BRAKE_DEADBAND_OMEGA:
                rear_brake_cmd  = -(1.0 - FRONT_BRAKE_BIAS) * brake_fraction * MAX_REAR_BRAKE * np.sign(rear_omega)
            if abs(front_omega) > BRAKE_DEADBAND_OMEGA:
                front_brake_cmd = -FRONT_BRAKE_BIAS * brake_fraction * MAX_FRONT_BRAKE * np.sign(front_omega)

            # ---- lean controller ----
            lean_target     = target_lean_at(chassis_x)
            lean_error      = lean_target - lean
            KP_LEAN         = 6.0
            steer_torque    = np.clip(KP_LEAN * lean_error, -10.0, 10.0)

            # ---- apply controls ----
            data.ctrl[throttle_id]    = throttle_cmd
            data.ctrl[rear_brake_id]  = rear_brake_cmd
            data.ctrl[front_brake_id] = front_brake_cmd
            data.ctrl[steer_motor_id] = steer_torque

            # Only step physics when not paused. Still sync() each frame so
            # the window can redraw if you orbit the camera while paused.
            if not state["paused"]:
                mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

            # Exit after 45 seconds (long enough for the run + recording)
            if time.time() - start > 45.0:
                break


if __name__ == "__main__":
    main()
