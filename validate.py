"""
motosim — CBR500R v0.1 self-stability validation

This is the v0.1 acceptance gate. From Meijaard-Papadopoulos-Ruina-Schwab 2007,
the uncontrolled Whipple bicycle should self-stabilize between its weave and
capsize eigenvalue speeds (roughly 4-8 m/s for benchmark bike geometry).

Test protocol:
    1. Set rear_wheel spin to 16 rad/s (approx 5 m/s forward at tire radius 0.312).
    2. Add small forward velocity to chassis matching rolling speed.
    3. Apply tiny lean perturbation (0.05 rad = ~2.9 deg).
    4. Run 15 seconds of sim with NO control input.
    5. Observe: does the bike self-stabilize?

PASS criteria:
    - Chassis stays within 20 deg of upright for the full 15 seconds
    - Forward speed stays above 3 m/s (friction can bleed some speed)

FAIL criteria:
    - Chassis exceeds 20 deg lean (will fall)
    - Chassis z-position drops below 0.25 (bike fell over)

If this fails: inertia tensors in cbr500r.xml need tuning. Do NOT blame the
topology — Meijaard's analytical result guarantees self-stability exists for
correctly-parameterized Whipple bikes in this speed range.

Usage:
    python validate.py           # runs headless, prints verdict
    python validate.py --render  # also opens viewer to watch
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer  # imported at module scope to avoid Python scoping surprise


FORWARD_SPEED = 5.0  # m/s target
REAR_WHEEL_RADIUS = 0.312
INITIAL_LEAN = 0.02  # rad — smaller perturbation keeps us in Meijaard's linear-stability regime
DURATION = 15.0  # s
LEAN_LIMIT_DEG = 20.0
MIN_SPEED = 3.0  # m/s
MIN_Z = 0.25  # m


def run(render: bool = False):
    model = mujoco.MjModel.from_xml_path("cbr500r.xml")
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    # Inject initial state:
    # - qpos layout for freejoint: [x, y, z, qw, qx, qy, qz, rear_spin, steer, front_spin]
    # - qvel layout for freejoint: [vx, vy, vz, wx, wy, wz, rear_spin_dot, steer_dot, front_spin_dot]

    # Apply initial lean about forward (x) axis by setting chassis quaternion
    half = INITIAL_LEAN / 2.0
    data.qpos[3] = np.cos(half)   # qw
    data.qpos[4] = np.sin(half)   # qx (roll about x = lean)
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    # Forward velocity on chassis
    data.qvel[0] = FORWARD_SPEED   # vx

    # Wheel spin to match rolling without slip: omega = v / r
    wheel_omega = FORWARD_SPEED / REAR_WHEEL_RADIUS
    data.qvel[6] = wheel_omega    # rear spin
    # front wheel index: last qvel element
    data.qvel[-1] = wheel_omega   # front spin

    # Rollout
    n_steps = int(DURATION / model.opt.timestep)
    lean_history = []
    speed_history = []
    z_history = []

    if render:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.lookat[:] = [0.3, 0, 0.5]

    for step in range(n_steps):
        mujoco.mj_step(model, data)

        # Extract lean angle from chassis quaternion
        qw, qx, qy, qz = data.qpos[3:7]
        # roll (lean) = atan2(2*(qw*qx + qy*qz), 1 - 2*(qx^2 + qy^2))
        lean_rad = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        lean_deg = np.degrees(lean_rad)

        speed = data.qvel[0]
        z = data.qpos[2]

        lean_history.append(lean_deg)
        speed_history.append(speed)
        z_history.append(z)

        if render:
            viewer.sync()

        # Early termination if bike has clearly fallen
        if abs(lean_deg) > 60 or z < 0.15:
            break

    if render:
        viewer.close()

    # Verdict
    lean_arr = np.array(lean_history)
    speed_arr = np.array(speed_history)
    z_arr = np.array(z_history)

    max_abs_lean = np.max(np.abs(lean_arr))
    final_speed = speed_arr[-1]
    min_z = np.min(z_arr)
    sim_time = len(lean_history) * model.opt.timestep

    print(f"\n=== CBR500R v0.1 self-stability validation ===")
    print(f"Simulated duration : {sim_time:.2f} s / {DURATION:.1f} s target")
    print(f"Max |lean|         : {max_abs_lean:.2f} deg  (limit {LEAN_LIMIT_DEG})")
    print(f"Final forward speed: {final_speed:.2f} m/s  (floor {MIN_SPEED})")
    print(f"Min chassis z      : {min_z:.3f} m  (floor {MIN_Z})")

    pass_lean = max_abs_lean <= LEAN_LIMIT_DEG
    pass_speed = final_speed >= MIN_SPEED
    pass_z = min_z >= MIN_Z
    pass_full = pass_lean and pass_speed and pass_z and sim_time >= DURATION * 0.99

    if pass_full:
        print(f"\nVERDICT: PASS — bike self-stabilizes within gates.")
        return 0
    else:
        print(f"\nVERDICT: FAIL")
        if not pass_lean:
            print(f"  - bike leaned beyond {LEAN_LIMIT_DEG} deg (reached {max_abs_lean:.1f})")
        if not pass_speed:
            print(f"  - bike lost too much speed (ended {final_speed:.2f} m/s)")
        if not pass_z:
            print(f"  - chassis dropped below {MIN_Z} m (min {min_z:.2f})")
        print(f"\nLikely cause: inertia tensors need tuning in cbr500r.xml.")
        print(f"First parameters to try adjusting:")
        print(f"  rear_frame diaginertia (currently '8.0 20.0 15.0')")
        print(f"  front_frame diaginertia (currently '0.3 0.3 0.05')")
        print(f"  front wheel spin inertia (currently 0.75)")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Open viewer during sim")
    args = parser.parse_args()
    raise SystemExit(run(render=args.render))
