from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards: Sequence[float], output_path, window: int = 10, title: str = "SAC Training Reward") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rewards_array = np.asarray(rewards, dtype=np.float32)
    episodes = np.arange(1, len(rewards_array) + 1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, rewards_array, label="Episode reward", linewidth=1.2)
    if len(rewards_array) >= window:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        moving_avg = np.convolve(rewards_array, kernel, mode="valid")
        plt.plot(np.arange(window, len(rewards_array) + 1), moving_avg, label=f"{window}-episode mean")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_evaluation(history: Dict[str, Iterable[float]], output_path, title: str = "SAC DummyGridEnv Evaluation") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = np.asarray(list(history["step"]))
    df = np.asarray(list(history["df"]))
    voltage = np.asarray(list(history["V"]))
    action = np.asarray(list(history["action"]))
    reward = np.asarray(list(history["reward"]))

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(steps, df, color="tab:blue")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_ylabel("df")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, voltage, color="tab:green")
    axes[1].axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("V")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, action, color="tab:orange")
    axes[2].set_ylabel("action")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(steps, reward, color="tab:red")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("reward")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
