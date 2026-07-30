"""
Train CNN Classifier (Approach B)
====================================
Trains the configured backbone on the particle dataset, with early stopping
on validation macro-F1, and saves training curves + final test metrics so
results are directly comparable against the traditional CV pipeline.

Usage:
    python train_cnn.py
    python train_cnn.py --backbone resnet18
"""
import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import load_config, resolve_path, ensure_dir, set_global_seed
from dataset import build_dataloaders
from model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels


def evaluate_full(model, loader, criterion, device, label_encoder):
    loss, acc, f1, preds, labels = run_epoch(model, loader, criterion, None, device, train=False)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, target_names=label_encoder.classes_,
                                    zero_division=0, output_dict=True)
    cm = confusion_matrix(labels, preds)
    return {
        "loss": loss, "accuracy": acc, "macro_precision": precision,
        "macro_recall": recall, "macro_f1": f1,
        "per_class_report": report, "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train CNN for tea grade classification.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None,
                         help="Override backbone from config (simple_cnn, resnet18, efficientnet_b0, mobilenet_v3_small)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_dl = cfg["deep_learning"]
    if args.backbone:
        cfg_dl["backbone"] = args.backbone

    set_global_seed(cfg["split"]["random_seed"])
    device = get_device()
    print(f"Using device: {device}")

    manifest_path = os.path.join(resolve_path(cfg["paths"]["manifests"]), "full_manifest.csv")
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found at {manifest_path}. Run build_manifests.py first.")
        return

    grades = cfg["classes"]["grades"]
    label_encoder = LabelEncoder()
    label_encoder.fit(grades)

    loaders = build_dataloaders(manifest_path, cfg_dl, label_encoder)
    num_classes = len(grades)

    model = build_model(cfg_dl, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_dl["learning_rate"], weight_decay=cfg_dl["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    models_dir = resolve_path(cfg["paths"]["results_models"])
    logs_dir = resolve_path(cfg["paths"]["results_logs"])
    metrics_dir = resolve_path(cfg["paths"]["results_metrics"])
    ensure_dir(models_dir); ensure_dir(logs_dir); ensure_dir(metrics_dir)

    history = []
    best_val_f1 = -1.0
    patience_counter = 0
    backbone_name = cfg_dl["backbone"]
    checkpoint_path = os.path.join(models_dir, f"cnn_{backbone_name}_best.pt")

    print(f"\n=== Training {backbone_name} ===")
    start_time = time.time()

    for epoch in range(cfg_dl["num_epochs"]):
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, loaders["train"], criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, loaders["val"], criterion, optimizer, device, train=False
        )
        scheduler.step(val_f1)

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "train_f1": train_f1, "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })
        print(f"  Epoch {epoch+1:3d}/{cfg_dl['num_epochs']}  "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f}  "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": backbone_name,
                "num_classes": num_classes,
                "classes": list(label_encoder.classes_),
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg_dl["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch+1} (no val improvement for "
                      f"{cfg_dl['early_stopping_patience']} epochs).")
                break

    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time:.1f}s. Best val macro-F1: {best_val_f1:.4f}")

    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(logs_dir, f"cnn_{backbone_name}_history.csv"), index=False)

    # Load best checkpoint and evaluate on test set
    # weights_only=False: this is our own checkpoint saved a few lines above
    # in this same run, not an externally sourced file, so it's safe to load
    # with full unpickling (needed because PyTorch >=2.6 defaults to
    # weights_only=True, which rejects the metadata dict we stored alongside
    # the model weights).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_full(model, loaders["test"], criterion, device, label_encoder)
    val_metrics = evaluate_full(model, loaders["val"], criterion, device, label_encoder)

    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}  macro-F1: {test_metrics['macro_f1']:.4f}")

    num_params = sum(p.numel() for p in model.parameters())
    results = {
        "backbone": backbone_name,
        "training_time_seconds": training_time,
        "num_parameters": num_params,
        "best_epoch": len(history) - patience_counter,
        "val": val_metrics,
        "test": test_metrics,
    }
    out_path = os.path.join(metrics_dir, f"cnn_{backbone_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
