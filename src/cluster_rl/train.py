"""Train a MaskablePPO agent on the container-placement environment.

    python -m cluster_rl.train --config configs/smoke.yaml --rl-config configs/rl.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from cluster_rl.baselines import default_policies, run_episode
from cluster_rl.baselines.sb3_policy import SB3Policy
from cluster_rl.env import EnvConfig, PlacementEnv

DEFAULT_RL_CONFIG = {
    "total_timesteps": 50_000,
    "n_envs": 4,
    "learning_rate": 3.0e-4,
    "gamma": 0.99,
    "batch_size": 128,
    "n_steps": 512,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "net_arch": [128, 128],
    "eval_episodes": 10,
    "checkpoint_every": 10_000,
}


def mask_fn(env) -> np.ndarray:
    return env.action_masks()


def build_vec_env(cfg: EnvConfig, n_envs: int, seed: int) -> DummyVecEnv:
    def thunk(env_seed: int):
        def _make():
            env = PlacementEnv(cfg)
            env._rng = np.random.default_rng(env_seed)
            return ActionMasker(env, mask_fn)

        return _make

    return DummyVecEnv([thunk(seed + 1000 * i) for i in range(n_envs)])


def load_rl_config(path: str | None) -> dict:
    rl = dict(DEFAULT_RL_CONFIG)
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
        rl.update(data.get("train", data))
    return rl


def train(args: argparse.Namespace) -> None:
    cfg = EnvConfig.from_yaml(args.config)
    rl = load_rl_config(args.rl_config)
    rl.update(
        {
            "total_timesteps": int(args.total_timesteps or rl["total_timesteps"]),
            "n_envs": int(args.n_envs or rl["n_envs"]),
            "run_name": args.run_name or rl.get("run_name", "ppo_default"),
            "seed": int(args.seed),
        }
    )

    save_dir = Path(args.save_dir) / rl["run_name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = build_vec_env(cfg, rl["n_envs"], rl["seed"])
    policy_kwargs = {"net_arch": list(rl.get("net_arch", [128, 128]))}

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(rl["learning_rate"]),
        gamma=float(rl["gamma"]),
        batch_size=int(rl["batch_size"]),
        n_steps=int(rl["n_steps"]),
        ent_coef=float(rl["ent_coef"]),
        clip_range=float(rl["clip_range"]),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(save_dir / "tb"),
        seed=rl["seed"],
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=int(rl.get("checkpoint_every", 10_000)),
        save_path=str(save_dir),
        name_prefix="ppo",
    )

    print(f"[train] run={rl['run_name']} n_envs={rl['n_envs']} "
          f"total_timesteps={rl['total_timesteps']} seed={rl['seed']}")
    t0 = time.time()
    model.learn(
        total_timesteps=rl["total_timesteps"],
        callback=checkpoint_cb,
        progress_bar=False,
    )
    print(f"[train] done in {time.time() - t0:.1f}s")

    final_path = save_dir / "ppo_final"
    model.save(final_path)
    vec_env.close()
    print(f"[train] saved -> {final_path}.zip")

    ep = int(rl.get("eval_episodes", 10))
    print("[eval] comparing learned policy vs baselines ...")
    env = PlacementEnv(cfg)
    learned = SB3Policy(MaskablePPO.load(f"{final_path}.zip"), name="rl_ppo")
    results = {"rl_ppo": evaluate_policy(env, learned, ep, rl["seed"])}
    for p in default_policies(seed=rl["seed"]):
        results[p.name] = evaluate_policy(env, p, ep, rl["seed"])

    (save_dir / "eval.json").write_text(json.dumps(results, indent=2))
    print_table(results)
    print(f"[eval] wrote -> {save_dir / 'eval.json'}")


def evaluate_policy(env, policy, episodes: int, seed: int) -> dict:
    import numpy as np

    runs = [run_episode(env, policy, seed=seed + e) for e in range(episodes)]
    keys = [
        "failure_rate", "sla_violation_rate", "cpu_util_mean", "imbalance_mean",
        "wait_p95_ticks", "traffic_imbalance", "mean_latency", "total_reward",
    ]
    agg = {k: float(np.nanmean([r[k] for r in runs])) for k in keys}
    agg["scheduled"] = float(np.mean([r["scheduled"] for r in runs]))
    return agg


def print_table(results: dict) -> None:
    header = (
        f"{'policy':16s} {'sched':>6s} {'fail':>6s} {'sla':>6s} {'cpu':>6s} "
        f"{'imbal':>6s} {'wait95':>7s} {'traffic':>8s} {'reward':>9s}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:16s} {m['scheduled']:6.1f} {m['failure_rate']:6.3f} "
            f"{m['sla_violation_rate']:6.3f} {m['cpu_util_mean']:6.3f} "
            f"{m['imbalance_mean']:6.3f} {m['wait_p95_ticks']:7.1f} "
            f"{m['traffic_imbalance']:8.3f} {m['total_reward']:9.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train MaskablePPO for container placement")
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--rl-config", default="configs/rl.yaml")
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-dir", default="runs")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()