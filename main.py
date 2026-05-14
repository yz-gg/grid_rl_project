import argparse
from pathlib import Path

from trainers.evaluate_ppo_dummy import evaluate as evaluate_ppo
from trainers.evaluate_sac_dummy import evaluate as evaluate_sac
from trainers.train_ppo_dummy import train as train_ppo
from trainers.train_sac_dummy import train as train_sac


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS = {
    "sac": "configs/sac_dummy.yaml",
    "ppo": "configs/ppo_dummy.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="grid_rl_project entry point.")
    parser.add_argument("mode", choices=["train", "evaluate"], help="Run training or evaluation.")
    parser.add_argument("--algo", choices=["sac", "ppo"], default="sac", help="Algorithm to run.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--model", type=str, default=None, help="Model path for evaluation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config or DEFAULT_CONFIGS[args.algo])
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if args.mode == "train":
        if args.algo == "ppo":
            train_ppo(config_path)
        else:
            train_sac(config_path)
    else:
        if args.algo == "ppo":
            evaluate_ppo(config_path, args.model)
        else:
            evaluate_sac(config_path, args.model)
