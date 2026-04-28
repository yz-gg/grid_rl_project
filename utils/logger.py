import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
import yaml


def load_config(path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def save_records_csv(records: Iterable[Dict], path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(list(records)).to_csv(path, index=False)
