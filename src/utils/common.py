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


def get_px_per_mm(cfg: dict, grade: str = None) -> float:
    """
    Resolve the correct pixel-to-millimetre calibration factor for a grade.

    Different grades are captured at different camera zoom levels (see
    configs/config.yaml -> traditional_cv.px_per_mm_by_grade), so a single
    global px_per_mm silently corrupts size features for any grade shot at
    a different zoom. This looks up the grade-specific value, falling back
    to px_per_mm_default (or the legacy single px_per_mm key) if the grade
    isn't listed or wasn't provided (e.g. for a mixed sample where the
    per-particle grade isn't known yet).
    """
    cfg_cv = cfg["traditional_cv"]
    by_grade = cfg_cv.get("px_per_mm_by_grade")
    if by_grade and grade is not None and grade in by_grade:
        return by_grade[grade]
    return cfg_cv.get("px_per_mm_default", cfg_cv.get("px_per_mm"))
