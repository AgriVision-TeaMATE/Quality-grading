"""
Model Cache
=============
Loads trained model bundles (traditional-CV) and checkpoints (CNN) ONCE and
keeps them in memory, keyed by name. Without this, a naive FastAPI endpoint
that calls classify_with_traditional()/classify_with_cnn() per-request would
reload the model from disk on every single API call - fine for the one-shot
CLI scripts (estimate_mixture.py), unacceptable for a live API.

Import this module from the FastAPI app rather than duplicating loading logic.
"""
import os
import joblib
import numpy as np

_TRADITIONAL_CACHE = {}
_CNN_CACHE = {}
_TORCH_DEVICE = None


def get_traditional_bundle(cfg, model_name: str):
    """Returns {"model", "scaler", "label_encoder", "feature_cols"}, cached after first load."""
    if model_name in _TRADITIONAL_CACHE:
        return _TRADITIONAL_CACHE[model_name]

    from common import resolve_path
    models_dir = resolve_path(cfg["paths"]["results_models"])
    bundle_path = os.path.join(models_dir, f"traditional_{model_name}.joblib")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}. Train it first.")

    bundle = joblib.load(bundle_path)
    _TRADITIONAL_CACHE[model_name] = bundle
    return bundle


def get_cnn_model(cfg, backbone_name: str):
    """Returns (model, classes, transform, device), cached after first load."""
    if backbone_name in _CNN_CACHE:
        return _CNN_CACHE[backbone_name]

    import torch
    from common import resolve_path
    from model import build_model
    from dataset import build_transforms

    global _TORCH_DEVICE
    if _TORCH_DEVICE is None:
        _TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = _TORCH_DEVICE

    models_dir = resolve_path(cfg["paths"]["results_models"])
    checkpoint_path = os.path.join(models_dir, f"cnn_{backbone_name}_best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train it first.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    num_classes = checkpoint["num_classes"]

    cfg_dl = dict(cfg["deep_learning"])
    cfg_dl["backbone"] = backbone_name
    model = build_model(cfg_dl, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transforms(cfg_dl, split="test")

    result = (model, classes, transform, device)
    _CNN_CACHE[backbone_name] = result
    return result


def warm_up(cfg, traditional_models=("random_forest",), cnn_backbones=("resnet18",)):
    """Optionally pre-load models at app startup so the first real request isn't slow."""
    for name in traditional_models:
        try:
            get_traditional_bundle(cfg, name)
        except FileNotFoundError:
            pass
    for name in cnn_backbones:
        try:
            get_cnn_model(cfg, name)
        except FileNotFoundError:
            pass
