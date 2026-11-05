"""
motosim — Brax + MJX environment for learned motorcycle control (v1.0).

Task: LEAN-COMMAND TRACKING.
    Given a commanded lean angle, the policy must apply steer torque to track
    it without falling over. Everything else (throttle, brakes, slalom path
    planning) is NOT learned in v1.0.

Observation (6-dim, proprioceptive only, no vision):
    [lean, lean_rate, steer_angle, steer_rate, forward_speed, commanded_lean]

Action (1-dim):
    steer_torque in [-1, +1]  (scaled to [-10, +10] N*m by the env)

Reward per step:
    +1                               alive bonus
    -10 * (lean - commanded_lean)^2  tracking error
    -0.01 * action^2                 smooth control
    -100 and done                    on fall (chassis z < 0.3)

Episode:
    5 seconds = 2500 physics steps at timestep 0.002
    commanded_lean resampled every 1s to Uniform(-0.15, +0.15) rad

Only the steer_motor is learned. The rear_motor is driven by a fixed P
controller on forward_speed = 5 m/s, so the bike cruises at constant
speed while the policy focuses on balance.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from brax import base, envs, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


# ---- env constants (keep in sync with cbr500r.xml defaults) ----
TARGET_SPEED        = 5.0
WHEEL_RADIUS        = 0.312
MAX_STEER_TORQUE    = 10.0
MAX_THROTTLE        = 180.0
SPEED_P_GAIN        = 15.0
EPISODE_SECONDS     = 5.0
LEAN_CMD_RESAMPLE_S = 1.0
LEAN_CMD_RANGE      = 0.15      # radians, ~8.6 deg peak
FALL_Z              = 0.30
ALIVE_BONUS         = 1.0
TRACK_WEIGHT        = 10.0
ACTION_WEIGHT       = 0.01
FALL_PENALTY        = 100.0


def _lean_from_quat(qw, qx, qy, qz):
    """Extract roll (lean) about world x from a chassis quat."""
    return jp.arctan2(2.0 * (qw * qx + qy * qz),
                      1.0 - 2.0 * (qx * qx + qy * qy))


class MotorcycleBalance(PipelineEnv):
    """Lean-command tracking env for the motosim CBR500R.

    Backend defaults to 'mjx' for GPU/CPU-vectorized training.
    """

    def __init__(self, mjcf_path: str = "cbr500r.xml", backend: str = "mjx",
                 n_frames: int = 1, **kwargs: Any):
        # Load the model via brax's MJCF loader (handles MJX conversion)
        sys = mjcf.load(mjcf_path)

        # Use a larger physics timestep for training throughput.
        # Default MJCF is 0.002s, which is excellent for native MuJoCo but
        # expensive for MJX rollouts. 0.004s is still numerically stable
        # with RK4 for this topology.
        sys = sys.tree_replace({'opt.timestep': 0.004})

        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)

        # Cache indices for actuators + joints by name
        self._throttle_id    = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "throttle")
        self._steer_motor_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer_motor")
        self._rear_brake_id  = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_brake")
        self._front_brake_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_brake")

        self._steer_jnt_id     = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "steer")
        self._rear_spin_jnt_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "rear_spin")
        self._front_spin_jnt_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "front_spin")

        # Qpos/qvel addresses for named joints
        self._steer_qpos_adr   = int(sys.mj_model.jnt_qposadr[self._steer_jnt_id])
        self._steer_qvel_adr   = int(sys.mj_model.jnt_dofadr[self._steer_jnt_id])
        self._rear_spin_qvel_adr  = int(sys.mj_model.jnt_dofadr[self._rear_spin_jnt_id])
        self._front_spin_qvel_adr = int(sys.mj_model.jnt_dofadr[self._front_spin_jnt_id])

        # freejoint always at qpos[0:7], qvel[0:6] for a top-level freejoint
        self._max_steps = int(EPISODE_SECONDS / sys.opt.timestep)
        self._cmd_resample_every = max(1, int(LEAN_CMD_RESAMPLE_S / sys.opt.timestep))

    # ----- Brax API -----

    def reset(self, rng: jax.Array) -> State:
        """Spawn the bike at cruise speed, upright, with a random lean command."""
        rng, rng_cmd = jax.random.split(rng)

        qpos = jp.array(self.sys.mj_model.qpos0)
        # Default qpos from MJCF is already upright at rear_frame z=0.5.

        qvel = jp.zeros(self.sys.qd_size())
        # Forward velocity on chassis (freejoint vx)
        qvel = qvel.at[0].set(TARGET_SPEED)
        # Wheel spin matching forward motion
        omega = TARGET_SPEED / WHEEL_RADIUS
        qvel = qvel.at[self._rear_spin_qvel_adr].set(omega)
        qvel = qvel.at[self._front_spin_qvel_adr].set(omega)

        pipeline_state = self.pipeline_init(qpos, qvel)

        commanded_lean = jax.random.uniform(
            rng_cmd, shape=(), minval=-LEAN_CMD_RANGE, maxval=LEAN_CMD_RANGE)

        obs = self._get_obs(pipeline_state, commanded_lean)
        reward, done = jp.zeros(2)

        metrics: Dict[str, jax.Array] = {
            "reward_alive":   jp.zeros(()),
            "reward_track":   jp.zeros(()),
            "reward_action":  jp.zeros(()),
            "reward_fall":    jp.zeros(()),
            "lean_error":     jp.zeros(()),
            "forward_speed":  jp.zeros(()),
        }

        info = {
            "rng": rng,
            "commanded_lean": commanded_lean,
            "step_count": jp.zeros((), dtype=jp.int32),
        }

        return State(pipeline_state=pipeline_state, obs=obs,
                     reward=reward, done=done, metrics=metrics, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        """One env step: apply actuator commands, step physics, compute reward."""
        # Action is in [-1, +1]; scale to steer torque range
        steer_torque = jp.clip(action[0], -1.0, 1.0) * MAX_STEER_TORQUE

        # Fixed P controller on forward speed (maintains cruise)
        forward_speed = state.pipeline_state.qvel[0]
        speed_error = TARGET_SPEED - forward_speed
        throttle_torque = jp.clip(SPEED_P_GAIN * jp.maximum(speed_error, 0.0),
                                   0.0, MAX_THROTTLE)

        ctrl = jp.zeros(self.sys.nu)
        ctrl = ctrl.at[self._throttle_id].set(throttle_torque)
        ctrl = ctrl.at[self._steer_motor_id].set(steer_torque)
        # rear_brake and front_brake default to 0 (no braking during balance task)

        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)

        # Resample commanded lean periodically
        step_count = state.info["step_count"] + 1
        resample = (step_count % self._cmd_resample_every) == 0
        rng = state.info["rng"]
        rng, rng_cmd = jax.random.split(rng)
        new_cmd = jax.random.uniform(rng_cmd, shape=(),
                                      minval=-LEAN_CMD_RANGE, maxval=LEAN_CMD_RANGE)
        commanded_lean = jp.where(resample, new_cmd, state.info["commanded_lean"])

        obs = self._get_obs(pipeline_state, commanded_lean)

        # ----- reward -----
        quat = pipeline_state.qpos[3:7]
        lean = _lean_from_quat(quat[0], quat[1], quat[2], quat[3])
        chassis_z = pipeline_state.qpos[2]

        lean_err = lean - commanded_lean

        reward_alive  = ALIVE_BONUS
        reward_track  = -TRACK_WEIGHT * lean_err * lean_err
        reward_action = -ACTION_WEIGHT * action[0] * action[0]

        fallen = chassis_z < FALL_Z
        reward_fall = jp.where(fallen, -FALL_PENALTY, 0.0)

        reward = reward_alive + reward_track + reward_action + reward_fall
        done = jp.where(fallen, 1.0, 0.0)

        # Episode time limit
        time_up = step_count >= self._max_steps
        done = jp.where(time_up, 1.0, done)

        new_metrics = {
            "reward_alive":   reward_alive,
            "reward_track":   reward_track,
            "reward_action":  reward_action,
            "reward_fall":    reward_fall,
            "lean_error":     jp.abs(lean_err),
            "forward_speed":  forward_speed,
        }

        new_info = {
            "rng": rng,
            "commanded_lean": commanded_lean,
            "step_count": step_count,
        }

        return state.replace(
            pipeline_state=pipeline_state, obs=obs,
            reward=reward, done=done,
            metrics=new_metrics, info=new_info)

    def _get_obs(self, pipeline_state: base.State,
                 commanded_lean: jax.Array) -> jax.Array:
        """Build the 6-dim observation."""
        quat = pipeline_state.qpos[3:7]
        lean = _lean_from_quat(quat[0], quat[1], quat[2], quat[3])

        # Chassis angular velocity about world x (lean rate)
        # freejoint angvel is qvel[3:6]
        lean_rate = pipeline_state.qvel[3]

        steer_angle = pipeline_state.qpos[self._steer_qpos_adr]
        steer_rate  = pipeline_state.qvel[self._steer_qvel_adr]

        forward_speed = pipeline_state.qvel[0]

        return jp.array([
            lean, lean_rate, steer_angle, steer_rate, forward_speed, commanded_lean,
        ])

    @property
    def action_size(self) -> int:
        return 1

    @property
    def observation_size(self) -> int:
        return 6


# Register for brax.envs.get_environment("motorcycle_balance")
envs.register_environment("motorcycle_balance", MotorcycleBalance)
