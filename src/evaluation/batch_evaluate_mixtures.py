"""
Batch Mixture Evaluation
===========================
Runs mixture proportion estimation across ALL mixed-sample images (each with
a known ground-truth composition, read from a manifest CSV), for BOTH the
traditional-CV and CNN classifiers, and aggregates MAE/RMSE across samples.

Expects a CSV at data/mixed_samples/ground_truth.csv with columns:
    image_filename, <grade_1>, <grade_2>, ..., <grade_n>
where grade columns hold the true proportion (0-1) of each grade in that sample.

Usage:
    python batch_evaluate_mixtures.py
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "segmentation"))

from common import load_config, resolve_path, ensure_dir
from segment_particles import segment_image
from estimate_mixture import (
    classify_with_traditional, classify_with_cnn,
    compute_proportions, compute_errors,
)


def run_batch(cfg, ground_truth_df, method, model_name_or_backbone):
    grades = cfg["classes"]["grades"]
    mixed_dir = resolve_path(cfg["paths"]["mixed_samples"])
    weighting = cfg["mixture_estimation"]["weighting"]

    per_sample_results = []

    for _, row in ground_truth_df.iterrows():
        image_path = os.path.join(mixed_dir, row["image_filename"])
        image = cv2.imread(image_path)
        if image is None:
            print(f"  [WARN] Could not read {image_path}, skipping.")
            continue

        ground_truth = {g: row[g] for g in grades if g in row}

        crops, _, _ = segment_image(image, cfg["segmentation"])
        if not crops:
            print(f"  [WARN] No particles detected in {row['image_filename']}, skipping.")
            continue

        if method == "traditional":
            predictions, areas = classify_with_traditional(crops, cfg, model_name_or_backbone)
        else:
            predictions, areas = classify_with_cnn(crops, cfg, model_name_or_backbone)

        proportions = compute_proportions(predictions, areas, grades, weighting)
        errors, mae, rmse = compute_errors(proportions, ground_truth, grades)

        per_sample_results.append({
            "image": row["image_filename"],
            "num_particles": len(crops),
            "mae": mae,
            "rmse": rmse,
            "predicted_proportions": proportions,
            "ground_truth": ground_truth,
        })
        print(f"  {row['image_filename']:30s}  particles={len(crops):4d}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    return per_sample_results


def summarize(per_sample_results, label):
    if not per_sample_results:
        return {"method": label, "num_samples": 0, "mean_mae": None, "mean_rmse": None}

    maes = [r["mae"] for r in per_sample_results]
    rmses = [r["rmse"] for r in per_sample_results]
    return {
        "method": label,
        "num_samples": len(per_sample_results),
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "per_sample": per_sample_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate mixture proportion estimation.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--traditional_model", type=str, default="random_forest")
    parser.add_argument("--cnn_backbone", type=str, default="resnet18")
    parser.add_argument("--skip_traditional", action="store_true")
    parser.add_argument("--skip_cnn", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mixed_dir = resolve_path(cfg["paths"]["mixed_samples"])
    gt_path = os.path.join(mixed_dir, "ground_truth.csv")

    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth CSV not found at {gt_path}")
        print("Expected columns: image_filename, <grade_1>, <grade_2>, ... (proportions summing to 1.0)")
        return

    ground_truth_df = pd.read_csv(gt_path)
    metrics_dir = resolve_path(cfg["paths"]["results_metrics"])
    ensure_dir(metrics_dir)

    all_summaries = {}

    if not args.skip_traditional:
        print(f"\n=== Evaluating traditional CV ({args.traditional_model}) ===")
        results = run_batch(cfg, ground_truth_df, "traditional", args.traditional_model)
        all_summaries["traditional_" + args.traditional_model] = summarize(
            results, f"traditional_{args.traditional_model}"
        )

    if not args.skip_cnn:
        print(f"\n=== Evaluating CNN ({args.cnn_backbone}) ===")
        results = run_batch(cfg, ground_truth_df, "cnn", args.cnn_backbone)
        all_summaries["cnn_" + args.cnn_backbone] = summarize(
            results, f"cnn_{args.cnn_backbone}"
        )

    out_path = os.path.join(metrics_dir, "mixture_evaluation_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n=== Final Comparison ===")
    for key, summary in all_summaries.items():
        if summary["num_samples"] > 0:
            print(f"  {key:25s}  n={summary['num_samples']:3d}  "
                  f"mean_MAE={summary['mean_mae']:.4f}  mean_RMSE={summary['mean_rmse']:.4f}")
        else:
            print(f"  {key:25s}  no samples evaluated")

    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
