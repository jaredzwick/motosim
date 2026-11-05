"""
motosim — PPO training for the learned lean controller (v1.0).

Trains a small MLP via PPO (brax.training.agents.ppo.train) on the
MotorcycleBalance MJX env. Saves a pickled policy params checkpoint.

Usage:
    python train.py                              # default: short local run on CPU
    python train.py --num-timesteps 2_000_000    # longer run
    python train.py --num-envs 2048 --num-timesteps 10_000_000  # GPU-scale

CPU training is slow (brax PPO on M4 Mac ~1k steps/sec per env).
For a trained policy worth rolling out, run on cloud GPU:
    Lambda Labs or vast.ai A100: ~$1-2/hour, finishes in ~10-15 min.

Output:
    checkpoints/balance_<timestamp>.pkl  — pickled flax params tree
    metrics/train_<timestamp>.jsonl      — per-eval reward + loss metrics

Then load the checkpoint with rollout.py to see the policy in the native
MuJoCo viewer.

Dependencies:
    pip install brax jax jaxlib flax
    (brax may lag Python versions — if install fails on 3.14+, use 3.12.)
"""

import argparse
import functools
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import jax

# Import the env to trigger registration
import motosim_env  # noqa: F401
from brax import envs
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks


def make_progress_fn(metrics_path: Path):
    """Return a brax progress callback that prints + logs each eval."""
    metrics_file = metrics_path.open("w")

    def progress(num_steps: int, metrics: dict) -> None:
        # Strip jax device arrays to floats for logging
        clean = {k: float(v) for k, v in metrics.items() if hasattr(v, "shape") or isinstance(v, (int, float))}
        clean["num_steps"] = num_steps
        json.dump(clean, metrics_file)
        metrics_file.write("\n")
        metrics_file.flush()

        reward = metrics.get("eval/episode_reward", 0.0)
        reward_std = metrics.get("eval/episode_reward_std", 0.0)
        episode_length = metrics.get("eval/episode_length", 0.0)
        print(f"  steps={num_steps:>10,d}  "
              f"reward={float(reward):+7.2f} ± {float(reward_std):5.2f}  "
              f"episode_len={float(episode_length):5.1f}")

    return progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-timesteps", type=int, default=500_000,
                        help="total env-steps to train for (default 500k)")
    parser.add_argument("--num-envs", type=int, default=32,
                        help="parallel envs (default 32; 2048+ for GPU)")
    parser.add_argument("--num-evals", type=int, default=10,
                        help="number of eval checkpoints (default 10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print("motosim v1.0 — PPO training")
    print(f"  env:             motorcycle_balance")
    print(f"  backend:         mjx")
    print(f"  num_envs:        {args.num_envs}")
    print(f"  num_timesteps:   {args.num_timesteps:,}")
    print(f"  num_evals:       {args.num_evals}")
    print(f"  seed:            {args.seed}")
    print(f"  devices:         {jax.devices()}")
    print("=" * 60)

    env = envs.get_environment("motorcycle_balance")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = metrics_dir / f"train_{timestamp}.jsonl"
    checkpoint_path = output_dir / f"balance_{timestamp}.pkl"

    # PPO hyperparameters — tuned for a simple 1-D control task.
    # Conservative defaults that converge on CPU in reasonable time.
    make_networks = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(64, 64),
        value_hidden_layer_sizes=(64, 64),
    )

    t_start = time.time()
    make_inference_fn, params, metrics = ppo_train.train(
        environment=env,
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1.0,
        episode_length=2500,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=8,
        num_updates_per_batch=4,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=args.num_envs,
        batch_size=128,
        seed=args.seed,
        network_factory=make_networks,
        progress_fn=make_progress_fn(metrics_path),
    )
    elapsed = time.time() - t_start

    print()
    print(f"training done in {elapsed/60:.1f} min")
    print(f"final reward: {float(metrics.get('eval/episode_reward', 0)):+.2f}")
    print(f"saving checkpoint: {checkpoint_path}")

    # Save params for later rollout
    with checkpoint_path.open("wb") as f:
        pickle.dump({"params": params, "metrics": dict(metrics)}, f)

    # Also save latest symlink for rollout.py convenience
    latest_link = output_dir / "balance_latest.pkl"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(checkpoint_path.name)

    print(f"symlink: {latest_link} -> {checkpoint_path.name}")
    print(f"metrics: {metrics_path}")
    print()
    print("Next: python rollout.py (plays the latest trained policy in the viewer)")


if __name__ == "__main__":
    main()
