import argparse
from pathlib import Path

from trainers.evaluate_sac_dummy import evaluate
from trainers.train_sac_dummy import train


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="grid_rl_project entry point.")
    parser.add_argument("mode", choices=["train", "evaluate"], help="Run training or evaluation.")
    parser.add_argument("--config", type=str, default="configs/sac_dummy.yaml", help="Path to YAML config.")
    parser.add_argument("--model", type=str, default=None, help="Model path for evaluation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if args.mode == "train":
        train(config_path)
    else:
        evaluate(config_path, args.model)
