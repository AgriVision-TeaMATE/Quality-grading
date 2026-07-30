"""
Shared utilities used across the project: config loading, path resolution,
logging setup, and reproducibility helpers.
"""
import os
import random
import yaml
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str = None) -> dict:
    """Load the central YAML config. Defaults to configs/config.yaml at project root."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> str:
    """Resolve a path from config (relative to project root) to an absolute path."""
    return os.path.join(PROJECT_ROOT, relative_path)


def set_global_seed(seed: int = 42):
    """Set random seeds across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path
