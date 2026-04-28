import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import DummyGridEnv


def load_config(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_env(config: dict, seed: int) -> DummyGridEnv:
    env_config = config["env"]
    return DummyGridEnv(
        max_steps=env_config.get("max_steps", 200),
        disturbance_step=env_config.get("disturbance_step", 50),
        disturbance_magnitude=env_config.get("disturbance_magnitude", 0.35),
        action_low=env_config.get("action_low", -0.02),
        action_high=env_config.get("action_high", 0.02),
        reward_coefficients=env_config.get("reward", {}),
        seed=seed,
    )


def compute_action_smoothness(actions) -> float:
    actions = np.asarray(actions, dtype=np.float32)
    if len(actions) <= 1:
        return 0.0
    return float(np.mean(np.diff(actions) ** 2))


def plot_random_policy_trajectories(df: pd.DataFrame, output_path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    colors = {
        "df": "tab:blue",
        "V": "tab:green",
        "action": "tab:orange",
        "reward": "tab:red",
    }

    for episode, episode_df in df.groupby("episode"):
        alpha = 0.35
        label = f"episode {episode}" if episode <= 3 else None
        axes[0].plot(episode_df["step"], episode_df["df"], color=colors["df"], alpha=alpha, label=label)
        axes[1].plot(episode_df["step"], episode_df["V"], color=colors["V"], alpha=alpha)
        axes[2].plot(episode_df["step"], episode_df["action"], color=colors["action"], alpha=alpha)
        axes[3].plot(episode_df["step"], episode_df["reward"], color=colors["reward"], alpha=alpha)

    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("df")
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("V")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("action")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("reward")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Random Policy Evaluation on DummyGridEnv")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_random_policy(config_path, episodes: int = 10) -> None:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)
    base_seed = int(config.get("seed", 42)) + 2000

    rows = []
    episode_metrics = []

    for episode in range(1, episodes + 1):
        env = build_env(config, base_seed + episode)
        obs = env.reset(seed=base_seed + episode)

        episode_reward = 0.0
        episode_rows = []
        actions = []

        for step in range(1, env.max_steps + 1):
            action = env.sample_action()
            next_obs, reward, done, info = env.step(action)

            action_value = float(action[0])
            row = {
                "episode": episode,
                "step": step,
                "df": float(next_obs[0]),
                "V": float(next_obs[1]),
                "P": float(next_obs[2]),
                "Q": float(next_obs[3]),
                "I": float(next_obs[4]),
                "action": action_value,
                "reward": float(reward),
                "load_disturbance": float(info["load_disturbance"]),
                "done": bool(done),
                "done_reason": info["done_reason"],
            }
            episode_rows.append(row)
            actions.append(action_value)
            episode_reward += reward
            obs = next_obs

            if done:
                break

        episode_df = pd.DataFrame(episode_rows)
        max_abs_df = float(episode_df["df"].abs().max())
        max_abs_v_deviation = float((episode_df["V"] - 1.0).abs().max())
        action_smoothness = compute_action_smoothness(actions)

        metric = {
            "episode": episode,
            "episode_reward": float(episode_reward),
            "max_abs_df": max_abs_df,
            "max_abs_v_deviation": max_abs_v_deviation,
            "action_smoothness": action_smoothness,
            "episode_steps": len(episode_rows),
        }
        episode_metrics.append(metric)

        for row in episode_rows:
            row.update(metric)
        rows.extend(episode_rows)

    result_dir = PROJECT_ROOT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "random_policy_evaluation.csv"
    plot_path = result_dir / "random_policy_evaluation.png"

    result_df = pd.DataFrame(rows)
    result_df.to_csv(csv_path, index=False)
    plot_random_policy_trajectories(result_df, plot_path)

    metrics_df = pd.DataFrame(episode_metrics)
    mean_episode_reward = float(metrics_df["episode_reward"].mean())
    max_abs_df = float(metrics_df["max_abs_df"].max())
    max_abs_v_deviation = float(metrics_df["max_abs_v_deviation"].max())
    mean_action_smoothness = float(metrics_df["action_smoothness"].mean())

    print(f"Saved random policy CSV to: {csv_path}")
    print(f"Saved random policy plot to: {plot_path}")
    print(f"average episode reward: {mean_episode_reward:.6f}")
    print(f"max abs(df): {max_abs_df:.6f}")
    print(f"max abs(V-1.0): {max_abs_v_deviation:.6f}")
    print(f"action smoothness: {mean_action_smoothness:.8f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate random policy on DummyGridEnv.")
    parser.add_argument("--config", type=str, default="configs/sac_dummy.yaml", help="Path to YAML config.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of random-policy episodes.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_random_policy(args.config, args.episodes)
