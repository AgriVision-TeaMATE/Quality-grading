"""
Train Traditional ML Classifiers (Approach A)
================================================
Trains Random Forest, SVM, and XGBoost on the handcrafted feature matrix,
evaluates each on val/test, and saves the best model + a comparison table.

Usage:
    python train_traditional_models.py
    python train_traditional_models.py --model random_forest   # train just one
"""
import os
import sys
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import load_config, resolve_path, ensure_dir, set_global_seed

NON_FEATURE_COLS = {"filepath", "label", "split"}


def load_feature_matrix(manifest_dir):
    path = os.path.join(manifest_dir, "feature_matrix.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run build_feature_matrix.py first."
        )
    return pd.read_csv(path)


def prepare_splits(df):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_val, y_val = val_df[feature_cols].values, val_df["label"].values
    X_test, y_test = test_df[feature_cols].values, test_df["label"].values

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols


def build_model(name, cfg_models, random_seed):
    if name == "random_forest":
        params = cfg_models["random_forest"]
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=random_seed,
            n_jobs=-1,
        )
    elif name == "svm":
        params = cfg_models["svm"]
        return SVC(
            kernel=params["kernel"], C=params["C"], gamma=params["gamma"],
            probability=True, random_state=random_seed,
        )
    elif name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed. pip install xgboost")
        params = cfg_models["xgboost"]
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=random_seed,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def evaluate(model, X, y_encoded, label_encoder):
    preds = model.predict(X)
    acc = accuracy_score(y_encoded, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_encoded, preds, average="macro", zero_division=0
    )
    report = classification_report(
        y_encoded, preds, target_names=label_encoder.classes_,
        zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_encoded, preds)
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
    }, preds


def main():
    parser = argparse.ArgumentParser(description="Train traditional ML classifiers on handcrafted features.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="all",
                         choices=["all", "random_forest", "svm", "xgboost"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["split"]["random_seed"])

    manifest_dir = resolve_path(cfg["paths"]["manifests"])
    models_dir = resolve_path(cfg["paths"]["results_models"])
    metrics_dir = resolve_path(cfg["paths"]["results_metrics"])
    ensure_dir(models_dir)
    ensure_dir(metrics_dir)

    df = load_feature_matrix(manifest_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = prepare_splits(df)

    # Encode labels and scale features (fit on train only, applied to val/test)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model_names = ["random_forest", "svm", "xgboost"] if args.model == "all" else [args.model]
    if "xgboost" in model_names and not XGBOOST_AVAILABLE:
        print("[WARN] xgboost not installed, skipping. Run: pip install xgboost --break-system-packages")
        model_names = [m for m in model_names if m != "xgboost"]

    all_results = {}
    for name in model_names:
        print(f"\n=== Training {name} ===")
        model = build_model(name, cfg["traditional_cv"]["models"], cfg["split"]["random_seed"])
        model.fit(X_train_scaled, y_train_enc)

        val_metrics, _ = evaluate(model, X_val_scaled, y_val_enc, label_encoder)
        test_metrics, test_preds = evaluate(model, X_test_scaled, y_test_enc, label_encoder)

        print(f"  Val  accuracy: {val_metrics['accuracy']:.4f}  macro-F1: {val_metrics['macro_f1']:.4f}")
        print(f"  Test accuracy: {test_metrics['accuracy']:.4f}  macro-F1: {test_metrics['macro_f1']:.4f}")

        # Save model + scaler + encoder bundled together
        bundle = {"model": model, "scaler": scaler, "label_encoder": label_encoder,
                  "feature_cols": feature_cols}
        joblib.dump(bundle, os.path.join(models_dir, f"traditional_{name}.joblib"))

        all_results[name] = {"val": val_metrics, "test": test_metrics}

    # Save comparison metrics
    out_path = os.path.join(metrics_dir, "traditional_cv_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {out_path}")
    print("\n=== Summary (Test Set) ===")
    for name, res in all_results.items():
        t = res["test"]
        print(f"  {name:15s}  acc={t['accuracy']:.4f}  macro-F1={t['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
