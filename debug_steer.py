"""
Ignore the controller. Just apply +10 N*m steer torque from t=0 and observe
whether the steer joint actually moves. If steer angle stays at zero even
with max torque, there's a joint config bug (range, locked, bad axis).
"""
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

throttle_id = model.actuator("throttle").id
steer_motor_id = model.actuator("steer_motor").id

# Launch at cruise
data.qvel[0] = 5.0
omega = 5.0 / 0.312
data.joint("rear_spin").qvel[0] = omega
data.joint("front_spin").qvel[0] = omega
mujoco.mj_forward(model, data)

# Check steer joint config
steer_jnt_id = model.joint("steer").id
print(f"Steer joint id={steer_jnt_id}")
print(f"  range: {model.jnt_range[steer_jnt_id]}")
print(f"  limited: {model.jnt_limited[steer_jnt_id]}")
print(f"  damping: {model.dof_damping[model.jnt_dofadr[steer_jnt_id]]}")
print(f"  armature: {model.dof_armature[model.jnt_dofadr[steer_jnt_id]]}")
print(f"  stiffness: {model.jnt_stiffness[steer_jnt_id]}")
print(f"  axis (local): {model.jnt_axis[steer_jnt_id]}")

print(f"\nSteer motor id={steer_motor_id}")
print(f"  ctrlrange: {model.actuator_ctrlrange[steer_motor_id]}")
print(f"  gear: {model.actuator_gear[steer_motor_id]}")

print("\nt     bike_x    steer_q  steer_dq  steer_tq_cmd")
for step in range(1500):   # 3 seconds at 0.002s
    data.ctrl[throttle_id] = 0.0
    # Apply MAX positive steer torque
    data.ctrl[steer_motor_id] = 10.0
    mujoco.mj_step(model, data)
    if step % 100 == 0:
        sq = float(data.joint("steer").qpos[0])
        sdq = float(data.joint("steer").qvel[0])
        print(f"{data.time:5.2f} {data.qpos[0]:+7.2f} {sq:+9.4f} {sdq:+9.4f} +10.00")
