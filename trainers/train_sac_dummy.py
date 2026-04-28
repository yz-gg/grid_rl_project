import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import trange


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import SACAgent
from buffers import ReplayBuffer
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
    return SACAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        action_low=env.action_low,
        action_high=env.action_high,
        device=device,
        hidden_dims=tuple(agent_config.get("hidden_dims", [256, 256])),
        gamma=agent_config.get("gamma", 0.99),
        tau=agent_config.get("tau", 0.005),
        actor_lr=agent_config.get("actor_lr", 3e-4),
        critic_lr=agent_config.get("critic_lr", 3e-4),
        alpha_lr=agent_config.get("alpha_lr", 3e-4),
        automatic_entropy_tuning=agent_config.get("automatic_entropy_tuning", True),
        target_entropy=agent_config.get("target_entropy"),
        initial_alpha=agent_config.get("initial_alpha", 0.2),
        grad_clip_norm=agent_config.get("grad_clip_norm", 10.0),
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

    train_config = config["training"]
    result_dir = ensure_dir(PROJECT_ROOT / train_config.get("result_dir", "results"))
    checkpoint_dir = ensure_dir(PROJECT_ROOT / train_config.get("checkpoint_dir", "checkpoints"))

    replay_buffer = ReplayBuffer(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        capacity=train_config.get("replay_size", 100000),
        device=device,
        seed=seed,
    )

    total_episodes = int(train_config.get("total_episodes", 200))
    batch_size = int(train_config.get("batch_size", 256))
    warmup_steps = int(train_config.get("warmup_steps", 1000))
    start_updates = int(train_config.get("start_updates", 1000))
    updates_per_step = int(train_config.get("updates_per_step", 1))
    log_interval = int(train_config.get("log_interval", 10))
    save_interval = int(train_config.get("save_interval", 50))

    episode_records = []
    loss_records = []
    episode_rewards = []
    total_steps = 0

    progress = trange(1, total_episodes + 1, desc="Training SAC", unit="episode")
    for episode in progress:
        obs = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        last_losses = None

        for _ in range(env.max_steps):
            if total_steps < warmup_steps:
                action = env.sample_action()
            else:
                action = agent.select_action(obs, evaluate=False)

            next_obs, reward, done, info = env.step(action)
            replay_buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

            if total_steps >= start_updates and len(replay_buffer) >= batch_size:
                for _ in range(updates_per_step):
                    last_losses = agent.update(replay_buffer, batch_size)
                    loss_records.append(
                        {
                            "global_step": total_steps,
                            "episode": episode,
                            **last_losses,
                        }
                    )

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_records.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "steps": episode_steps,
                "total_steps": total_steps,
            }
        )

        postfix = {
            "reward": f"{episode_reward:.3f}",
            "buffer": len(replay_buffer),
            "alpha": f"{last_losses['alpha']:.3f}" if last_losses else "-",
        }
        progress.set_postfix(postfix)

        if episode % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            mean_reward = sum(recent_rewards) / len(recent_rewards)
            print(f"Episode {episode:4d} | mean reward {mean_reward:9.3f} | total steps {total_steps}")

        if save_interval > 0 and episode % save_interval == 0:
            agent.save_model(checkpoint_dir / f"sac_dummy_ep{episode}.pt")

    reward_csv = result_dir / train_config.get("reward_csv", "sac_dummy_rewards.csv")
    loss_csv = result_dir / train_config.get("loss_csv", "sac_dummy_losses.csv")
    reward_plot = result_dir / train_config.get("reward_plot", "sac_dummy_rewards.png")
    final_model = checkpoint_dir / "sac_dummy_final.pt"

    save_records_csv(episode_records, reward_csv)
    if loss_records:
        pd.DataFrame(loss_records).to_csv(loss_csv, index=False)
    plot_rewards(episode_rewards, reward_plot)
    agent.save_model(final_model)

    print(f"Saved final model to: {final_model}")
    print(f"Saved rewards to: {reward_csv}")
    print(f"Saved reward plot to: {reward_plot}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on DummyGridEnv.")
    parser.add_argument("--config", type=str, default="configs/sac_dummy.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
