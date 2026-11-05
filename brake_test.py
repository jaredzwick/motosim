"""
motosim — brake + throttle actuator validation (v0.3).

Four tests, all on a straight line (no slalom, no steering). All run headless
unless --render is passed.

    1. ACCEL    throttle=full from standstill, measure time to 10 m/s
    2. REAR_ONLY  from 10 m/s, apply rear_brake=full. Measure stopping distance.
    3. FRONT_ONLY from 10 m/s, apply front_brake=full. Measure stopping distance.
    4. COMBINED   from 10 m/s, apply 60% front + 40% rear. Measure distance.

Expected behavior (matches real motorcycle physics):
    - ACCEL     : ~3-6 seconds 0-10 m/s (CBR500R is ~150 kW/ton, so plausible)
    - REAR_ONLY : longest stop. Rear can lock and skid or just under-brake.
    - FRONT_ONLY: shortest stop in theory but risks going over the bars.
    - COMBINED  : should beat both single-wheel tests. This is what real riders do.

Usage:
    python brake_test.py            # headless, table output
    python brake_test.py --render   # watch each test in the viewer (need mjpython)
"""

import argparse
import numpy as np
import mujoco
import mujoco.viewer  # module scope to avoid Python local-shadowing when main() does its conditional import


REAR_WHEEL_RADIUS = 0.312
TARGET_SPEED = 10.0     # m/s = ~36 km/h = ~22 mph
BRAKE_TRIGGER_SPEED = 10.0  # start braking once speed reaches this
HOLD_FINISH_BELOW = 0.1  # m/s, stop condition
MAX_TEST_DURATION = 15.0  # seconds


def reset_bike(model, data, initial_speed: float = 0.0):
    mujoco.mj_resetData(model, data)
    data.qvel[0] = initial_speed
    omega = initial_speed / REAR_WHEEL_RADIUS
    data.joint("rear_spin").qvel[0]  = omega
    data.joint("front_spin").qvel[0] = omega


def run_accel_test(model, data, viewer=None):
    """Full throttle from rest until forward speed >= 10 m/s."""
    reset_bike(model, data, initial_speed=0.0)
    throttle_id    = model.actuator("throttle").id
    rear_brake_id  = model.actuator("rear_brake").id
    front_brake_id = model.actuator("front_brake").id

    t_start = data.time
    reached_10 = False
    MAX_THROTTLE = 180.0
    while data.time - t_start < MAX_TEST_DURATION:
        data.ctrl[throttle_id]    = MAX_THROTTLE
        data.ctrl[rear_brake_id]  = 0.0
        data.ctrl[front_brake_id] = 0.0
        mujoco.mj_step(model, data)
        if viewer: viewer.sync()
        if data.qvel[0] >= TARGET_SPEED:
            reached_10 = True
            break

    elapsed = data.time - t_start
    return {
        "test": "accel 0->10 m/s",
        "pass": reached_10,
        "time_s": round(elapsed, 2),
        "final_speed_mps": round(float(data.qvel[0]), 2),
        "distance_m": round(float(data.qpos[0]), 2),
    }


def run_brake_test(model, data, label: str, rear_strength: float, front_strength: float, viewer=None):
    """Accelerate to 10 m/s, then apply brakes at given strengths [0..1], measure stop distance."""
    reset_bike(model, data, initial_speed=TARGET_SPEED)
    throttle_id    = model.actuator("throttle").id
    rear_brake_id  = model.actuator("rear_brake").id
    front_brake_id = model.actuator("front_brake").id

    MAX_REAR_BRAKE  = 180.0
    MAX_FRONT_BRAKE = 350.0
    # Low deadband: brake stays engaged until wheel is almost stopped (~0.03 m/s
    # linear). Otherwise the bike coasts on near-zero rolling friction and
    # never crosses the test's "stopped" threshold.
    BRAKE_DEADBAND = 0.1

    x_start = float(data.qpos[0])
    t_start = data.time
    stopped = False
    while data.time - t_start < MAX_TEST_DURATION:
        rear_omega  = data.joint("rear_spin").qvel[0]
        front_omega = data.joint("front_spin").qvel[0]

        rear_brake_cmd  = 0.0
        front_brake_cmd = 0.0
        if abs(rear_omega) > BRAKE_DEADBAND:
            rear_brake_cmd  = -rear_strength  * MAX_REAR_BRAKE  * np.sign(rear_omega)
        if abs(front_omega) > BRAKE_DEADBAND:
            front_brake_cmd = -front_strength * MAX_FRONT_BRAKE * np.sign(front_omega)

        data.ctrl[throttle_id]    = 0.0
        data.ctrl[rear_brake_id]  = rear_brake_cmd
        data.ctrl[front_brake_id] = front_brake_cmd
        mujoco.mj_step(model, data)
        if viewer: viewer.sync()

        if abs(data.qvel[0]) < HOLD_FINISH_BELOW:
            stopped = True
            break

    stop_distance = float(data.qpos[0]) - x_start
    chassis_z = float(data.qpos[2])
    return {
        "test": label,
        "pass": stopped and chassis_z > 0.3,  # not fallen over
        "stop_distance_m": round(stop_distance, 2),
        "stop_time_s": round(data.time - t_start, 2),
        "chassis_z_final": round(chassis_z, 3),
    }


def main(render: bool = False):
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -15

    results = []

    print("\n=== motosim v0.3 actuator validation ===\n")

    r = run_accel_test(model, data, viewer=viewer)
    print(f"[1/4] {r['test']:30s} time={r['time_s']}s  dist={r['distance_m']}m  final={r['final_speed_mps']} m/s  pass={r['pass']}")
    results.append(r)

    r = run_brake_test(model, data, "brake: rear only",     rear_strength=1.0, front_strength=0.0, viewer=viewer)
    print(f"[2/4] {r['test']:30s} distance={r['stop_distance_m']:5.2f}m  time={r['stop_time_s']}s  chassis_z={r['chassis_z_final']}  pass={r['pass']}")
    results.append(r)

    r = run_brake_test(model, data, "brake: front only",    rear_strength=0.0, front_strength=1.0, viewer=viewer)
    print(f"[3/4] {r['test']:30s} distance={r['stop_distance_m']:5.2f}m  time={r['stop_time_s']}s  chassis_z={r['chassis_z_final']}  pass={r['pass']}")
    results.append(r)

    r = run_brake_test(model, data, "brake: max both (emergency)", rear_strength=1.0, front_strength=1.0, viewer=viewer)
    print(f"[4/4] {r['test']:30s} distance={r['stop_distance_m']:5.2f}m  time={r['stop_time_s']}s  chassis_z={r['chassis_z_final']}  pass={r['pass']}")
    results.append(r)

    print()
    all_pass = all(r["pass"] for r in results)
    if all_pass:
        rear_d  = results[1]["stop_distance_m"]
        front_d = results[2]["stop_distance_m"]
        combo_d = results[3]["stop_distance_m"]
        ordered = combo_d < front_d < rear_d  # emergency < front < rear
        print("VERDICT: PASS — all actuators behave physically.")
        print("Expected ranking: emergency_combined < front_only < rear_only")
        print(f"  rear_only:         {rear_d}m")
        print(f"  front_only:        {front_d}m")
        print(f"  emergency combo:   {combo_d}m  (should be shortest)")
        print(f"  ordering correct:  {ordered}")
        return 0
    else:
        print("VERDICT: FAIL")
        for r in results:
            if not r["pass"]:
                print(f"  - {r['test']}: {r}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(render=args.render))
