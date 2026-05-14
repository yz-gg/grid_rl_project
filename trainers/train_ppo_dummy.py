import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import trange


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import PPOAgent
from buffers import RolloutBuffer
from envs import DummyGridEnv
from utils import ensure_dir, get_device, load_config, plot_rewards, save_records_csv, set_seed


def build_env(config, seed):
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


def build_agent(config, env, device):
    agent_config = config["agent"]
    return PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
        hidden_dims=tuple(agent_config.get("hidden_dims", [256, 256])),
        actor_lr=agent_config.get("actor_lr", 3e-4),
        critic_lr=agent_config.get("critic_lr", 3e-4),
        clip_range=agent_config.get("clip_range", 0.2),
        entropy_coef=agent_config.get("entropy_coef", 0.01),
        value_coef=agent_config.get("value_coef", 0.5),
        max_grad_norm=agent_config.get("max_grad_norm", 0.5),
        initial_log_std=agent_config.get("initial_log_std", -0.5),
    )


def train(config_path) -> None:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = get_device(config.get("device", "auto"))

    env = build_env(config, seed)
    agent = build_agent(config, env, device)

    agent_config = config["agent"]
    train_config = config["training"]
    result_dir = ensure_dir(PROJECT_ROOT / train_config.get("result_dir", "results"))
    checkpoint_dir = ensure_dir(PROJECT_ROOT / train_config.get("checkpoint_dir", "checkpoints"))

    rollout_steps = int(train_config.get("rollout_steps", 2048))
    total_updates = int(train_config.get("total_updates", 50))
    batch_size = int(train_config.get("batch_size", 256))
    update_epochs = int(train_config.get("update_epochs", 10))
    log_interval = int(train_config.get("log_interval", 10))
    save_interval = int(train_config.get("save_interval", 50))

    rollout_buffer = RolloutBuffer(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        capacity=rollout_steps,
        gamma=agent_config.get("gamma", 0.99),
        gae_lambda=agent_config.get("gae_lambda", 0.95),
        device=device,
        seed=seed,
    )

    episode_records = []
    loss_records = []
    episode_rewards = []
    total_steps = 0
    episode = 1
    episode_reward = 0.0
    episode_steps = 0
    obs = env.reset()

    progress = trange(1, total_updates + 1, desc="Training PPO", unit="update")
    for update in progress:
        rollout_buffer.reset()
        for _ in range(rollout_steps):
            action, log_prob, value = agent.select_action(obs, evaluate=False)
            next_obs, reward, done, _ = env.step(action)
            rollout_buffer.add(obs, action, reward, done, log_prob, value)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_records.append(
                    {
                        "episode": episode,
                        "reward": episode_reward,
                        "steps": episode_steps,
                        "total_steps": total_steps,
                    }
                )
                obs = env.reset()
                episode += 1
                episode_reward = 0.0
                episode_steps = 0

        last_value = 0.0 if done else agent.get_value(obs)
        rollout_buffer.compute_returns_and_advantages(last_value=last_value, last_done=done)
        losses = agent.update(rollout_buffer, batch_size, update_epochs)
        loss_records.append({"update": update, "total_steps": total_steps, **losses})

        recent_reward = episode_rewards[-1] if episode_rewards else episode_reward
        progress.set_postfix(
            {
                "reward": f"{recent_reward:.3f}",
                "policy_loss": f"{losses['policy_loss']:.3f}",
                "value_loss": f"{losses['value_loss']:.3f}",
            }
        )

        if update % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            if recent_rewards:
                mean_reward = sum(recent_rewards) / len(recent_rewards)
                print(f"Update {update:4d} | mean reward {mean_reward:9.3f} | total steps {total_steps}")

        if save_interval > 0 and update % save_interval == 0:
            agent.save_model(checkpoint_dir / f"ppo_dummy_update{update}.pt")

    if episode_steps > 0:
        episode_records.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "steps": episode_steps,
                "total_steps": total_steps,
            }
        )
        episode_rewards.append(episode_reward)

    reward_csv = result_dir / train_config.get("reward_csv", "ppo_dummy_rewards.csv")
    loss_csv = result_dir / train_config.get("loss_csv", "ppo_dummy_losses.csv")
    reward_plot = result_dir / train_config.get("reward_plot", "ppo_dummy_rewards.png")
    final_model = checkpoint_dir / "ppo_dummy_final.pt"

    save_records_csv(episode_records, reward_csv)
    if loss_records:
        pd.DataFrame(loss_records).to_csv(loss_csv, index=False)
    plot_rewards(episode_rewards, reward_plot, title="PPO Training Reward")
    agent.save_model(final_model)

    print(f"Saved final model to: {final_model}")
    print(f"Saved rewards to: {reward_csv}")
    print(f"Saved reward plot to: {reward_plot}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on DummyGridEnv.")
    parser.add_argument("--config", type=str, default="configs/ppo_dummy.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
