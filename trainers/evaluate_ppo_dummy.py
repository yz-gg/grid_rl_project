import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainers.train_ppo_dummy import build_agent, build_env
from utils import ensure_dir, get_device, load_config, plot_evaluation, set_seed


def evaluate(config_path, model_path=None) -> None:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)
    seed = int(config.get("seed", 42)) + 1000
    set_seed(seed)
    device = get_device(config.get("device", "auto"))

    env = build_env(config, seed)
    agent = build_agent(config, env, device)

    eval_config = config["evaluation"]
    if model_path is None:
        model_path = eval_config.get("model_path", "checkpoints/ppo_dummy_final.pt")
    model_path = Path(model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    agent.load_model(model_path, load_optimizers=False)

    obs = env.reset()
    history = {
        "step": [],
        "df": [],
        "V": [],
        "P": [],
        "Q": [],
        "I": [],
        "action": [],
        "reward": [],
        "load_disturbance": [],
    }

    episode_reward = 0.0
    for step in range(1, env.max_steps + 1):
        action, _, _ = agent.select_action(obs, evaluate=True)
        next_obs, reward, done, info = env.step(action)

        history["step"].append(step)
        history["df"].append(float(next_obs[0]))
        history["V"].append(float(next_obs[1]))
        history["P"].append(float(next_obs[2]))
        history["Q"].append(float(next_obs[3]))
        history["I"].append(float(next_obs[4]))
        history["action"].append(float(action[0]))
        history["reward"].append(float(reward))
        history["load_disturbance"].append(float(info["load_disturbance"]))

        episode_reward += reward
        obs = next_obs
        if done:
            break

    result_dir = ensure_dir(PROJECT_ROOT / eval_config.get("result_dir", "results"))
    plot_path = result_dir / eval_config.get("plot_file", "ppo_dummy_evaluation.png")
    csv_path = result_dir / eval_config.get("csv_file", "ppo_dummy_evaluation.csv")

    pd.DataFrame(history).to_csv(csv_path, index=False)
    plot_evaluation(history, plot_path, title="PPO DummyGridEnv Evaluation")

    df_values = np.asarray(history["df"], dtype=np.float32)
    voltage_values = np.asarray(history["V"], dtype=np.float32)
    action_values = np.asarray(history["action"], dtype=np.float32)
    action_smoothness = 0.0
    if len(action_values) > 1:
        action_smoothness = float(np.mean(np.diff(action_values) ** 2))

    print(f"Loaded model: {model_path}")
    print(f"Saved evaluation CSV to: {csv_path}")
    print(f"Saved evaluation plot to: {plot_path}")
    print(f"max abs(df): {float(np.max(np.abs(df_values))):.6f}")
    print(f"max abs(V-1.0): {float(np.max(np.abs(voltage_values - 1.0))):.6f}")
    print(f"action smoothness: {action_smoothness:.8f}")
    print(f"episode reward: {episode_reward:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on DummyGridEnv.")
    parser.add_argument("--config", type=str, default="configs/ppo_dummy.yaml", help="Path to YAML config.")
    parser.add_argument("--model", type=str, default=None, help="Path to trained checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.model)
