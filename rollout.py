"""
motosim — play a trained PPO policy in the native MuJoCo viewer (v1.0).

Loads a pickled policy checkpoint produced by train.py and rolls it out in
the interactive viewer so you can watch the learned controller balance the
bike. The sim runs in native MuJoCo (not MJX) so the viewer works — the
policy params transfer directly since the model is identical.

Usage:
    mjpython rollout.py                              # loads checkpoints/balance_latest.pkl
    mjpython rollout.py --checkpoint path/to/ckpt.pkl
    mjpython rollout.py --command 0.1                # fixed commanded lean 0.1 rad
    mjpython rollout.py --command sine               # oscillating lean command
"""

import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer

import motosim_env  # noqa: F401 — triggers env registration
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks


TARGET_SPEED    = 5.0
WHEEL_RADIUS    = 0.312
MAX_STEER_TORQUE = 10.0
MAX_THROTTLE    = 180.0
SPEED_P_GAIN    = 15.0


def _lean_from_quat(q):
    qw, qx, qy, qz = q
    return float(np.arctan2(2.0 * (qw * qx + qy * qz),
                             1.0 - 2.0 * (qx * qx + qy * qy)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/balance_latest.pkl")
    parser.add_argument("--command", type=str, default="sine",
                        help="'sine' for oscillation, or a float (rad) for a fixed lean target")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}\n"
                          f"Run `python train.py` first to produce one.")

    # Load params
    with ckpt_path.open("rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    print(f"loaded checkpoint: {ckpt_path}")

    # Rebuild the inference function from the same architecture used in training
    env = envs.get_environment("motorcycle_balance")
    network_factory = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        policy_hidden_layer_sizes=(64, 64),
        value_hidden_layer_sizes=(64, 64),
    )
    make_inference_fn = ppo_networks.make_inference_fn(network_factory)
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference = jax.jit(inference_fn)

    # Load the model in native MuJoCo for the viewer
    model = mujoco.MjModel.from_xml_path("cbr500r.xml")
    data  = mujoco.MjData(model)

    throttle_id    = model.actuator("throttle").id
    steer_motor_id = model.actuator("steer_motor").id

    # Initial state
    data.qvel[0] = TARGET_SPEED
    omega = TARGET_SPEED / WHEEL_RADIUS
    data.joint("rear_spin").qvel[0]  = omega
    data.joint("front_spin").qvel[0] = omega
    mujoco.mj_forward(model, data)

    # Command generator
    def command_at(t: float) -> float:
        if args.command == "sine":
            return 0.12 * float(np.sin(0.4 * t))
        try:
            return float(args.command)
        except ValueError:
            return 0.0

    rng = jax.random.PRNGKey(0)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("rear_frame").id
        viewer.cam.distance = 6.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        print(f"rolling out learned policy — command: {args.command}")
        print("Watch the bike track the commanded lean angle.")
        print()

        start = time.time()
        last_print = 0.0
        while viewer.is_running() and (time.time() - start) < args.duration:
            step_start = time.time()

            cmd = command_at(data.time)

            # Build observation (6-dim)
            quat = data.qpos[3:7]
            lean = _lean_from_quat(quat)
            lean_rate = float(data.qvel[3])
            steer_angle = float(data.joint("steer").qpos[0])
            steer_rate  = float(data.joint("steer").qvel[0])
            forward_speed = float(data.qvel[0])
            obs = jp.array([lean, lean_rate, steer_angle, steer_rate, forward_speed, cmd])

            # Policy inference
            rng, key = jax.random.split(rng)
            action, _ = jit_inference(obs, key)
            steer_torque = float(action[0]) * MAX_STEER_TORQUE

            # Speed controller (fixed P, same as during training)
            speed_error = TARGET_SPEED - forward_speed
            throttle_torque = max(0.0, min(SPEED_P_GAIN * speed_error, MAX_THROTTLE))

            data.ctrl[throttle_id]    = throttle_torque
            data.ctrl[steer_motor_id] = steer_torque

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print > 0.5:
                print(f"t={data.time:5.2f}  cmd={cmd:+.3f}  "
                      f"lean={lean:+.3f}  err={lean-cmd:+.3f}  "
                      f"steer_torque={steer_torque:+.2f}")
                last_print = data.time

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
