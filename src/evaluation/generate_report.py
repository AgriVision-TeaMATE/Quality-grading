"""
Generate Comparative Report (Phase 6)
========================================
Pulls together all saved metrics JSONs:
    - traditional_cv_results.json (RF, SVM, XGBoost classification metrics)
    - cnn_<backbone>_results.json (CNN classification metrics)
    - mixture_evaluation_comparison.json (proportion estimation MAE/RMSE)

Produces:
    - results/figures/classification_accuracy_comparison.png
    - results/figures/mixture_estimation_error_comparison.png
    - results/figures/confusion_matrices.png
    - results/metrics/final_comparison_table.csv

Usage:
    python generate_report.py
"""
import os
import sys
import json
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import load_config, resolve_path, ensure_dir


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def collect_classification_results(metrics_dir):
    rows = []

    trad = load_json(os.path.join(metrics_dir, "traditional_cv_results.json"))
    if trad:
        for model_name, res in trad.items():
            t = res["test"]
            rows.append({
                "approach": "Traditional CV", "model": model_name,
                "accuracy": t["accuracy"], "macro_precision": t["macro_precision"],
                "macro_recall": t["macro_recall"], "macro_f1": t["macro_f1"],
            })

    for path in glob.glob(os.path.join(metrics_dir, "cnn_*_results.json")):
        res = load_json(path)
        if res:
            t = res["test"]
            rows.append({
                "approach": "Deep Learning", "model": res["backbone"],
                "accuracy": t["accuracy"], "macro_precision": t["macro_precision"],
                "macro_recall": t["macro_recall"], "macro_f1": t["macro_f1"],
                "num_parameters": res.get("num_parameters"),
                "training_time_seconds": res.get("training_time_seconds"),
            })

    return pd.DataFrame(rows)


def plot_classification_comparison(df, out_path):
    if df.empty:
        print("  [SKIP] No classification results found yet.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    df_plot = df.melt(id_vars=["approach", "model"],
                       value_vars=["accuracy", "macro_precision", "macro_recall", "macro_f1"],
                       var_name="metric", value_name="value")
    df_plot["label"] = df_plot["approach"] + " - " + df_plot["model"]

    sns.barplot(data=df_plot, x="metric", y="value", hue="label", ax=ax)
    ax.set_title("Classification Performance: Traditional CV vs Deep Learning")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_mixture_comparison(mixture_data, out_path):
    if not mixture_data:
        print("  [SKIP] No mixture evaluation results found yet.")
        return

    rows = []
    for key, summary in mixture_data.items():
        if summary["num_samples"] > 0:
            rows.append({"method": key, "MAE": summary["mean_mae"], "RMSE": summary["mean_rmse"]})

    if not rows:
        print("  [SKIP] Mixture evaluation file exists but has no evaluated samples.")
        return

    df = pd.DataFrame(rows)
    df_plot = df.melt(id_vars="method", value_vars=["MAE", "RMSE"], var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df_plot, x="method", y="value", hue="metric", ax=ax)
    ax.set_title("Mixture Proportion Estimation Error by Method")
    ax.set_ylabel("Error")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_confusion_matrices(metrics_dir, grades, out_path):
    sources = {}

    trad = load_json(os.path.join(metrics_dir, "traditional_cv_results.json"))
    if trad:
        for model_name, res in trad.items():
            sources[f"Traditional - {model_name}"] = res["test"]["confusion_matrix"]

    for path in glob.glob(os.path.join(metrics_dir, "cnn_*_results.json")):
        res = load_json(path)
        if res:
            sources[f"Deep Learning - {res['backbone']}"] = res["test"]["confusion_matrix"]

    if not sources:
        print("  [SKIP] No confusion matrices found yet.")
        return

    n = len(sources)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (title, cm) in zip(axes, sources.items()):
        cm = np.array(cm)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=grades, yticklabels=grades, cbar=False)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate final comparative report.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics_dir = resolve_path(cfg["paths"]["results_metrics"])
    figures_dir = resolve_path(cfg["paths"]["results_figures"])
    ensure_dir(figures_dir)

    grades = cfg["classes"]["grades"]

    print("Collecting classification results...")
    class_df = collect_classification_results(metrics_dir)
    if not class_df.empty:
        table_path = os.path.join(metrics_dir, "final_comparison_table.csv")
        class_df.to_csv(table_path, index=False)
        print(f"  Saved comparison table to {table_path}")
        print(class_df.to_string(index=False))

    print("\nGenerating plots...")
    plot_classification_comparison(
        class_df, os.path.join(figures_dir, "classification_accuracy_comparison.png")
    )

    mixture_data = load_json(os.path.join(metrics_dir, "mixture_evaluation_comparison.json"))
    plot_mixture_comparison(
        mixture_data, os.path.join(figures_dir, "mixture_estimation_error_comparison.png")
    )

    plot_confusion_matrices(
        metrics_dir, grades, os.path.join(figures_dir, "confusion_matrices.png")
    )

    print("\nReport generation complete. Check results/figures/ and results/metrics/")


if __name__ == "__main__":
    main()
