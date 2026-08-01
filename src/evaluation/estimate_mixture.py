"""
Mixture Proportion Estimation (Phase 5)
==========================================
Given a mixed tea sample image:
    1. Segment individual particles
    2. Classify each particle (using EITHER the traditional-CV model OR the CNN)
    3. Aggregate predictions into grade proportions
    4. Compare against ground-truth proportions (if known) using MAE/RMSE

This is the script that answers the research question's second half:
"how does classification accuracy translate into mixture-estimation accuracy?"

Usage:
    # Traditional CV model
    python estimate_mixture.py --image data/mixed_samples/sample1.jpg \\
        --method traditional --model_name random_forest \\
        --ground_truth '{"BOP": 0.4, "OP": 0.3, "Dust": 0.3}'

    # CNN model
    python estimate_mixture.py --image data/mixed_samples/sample1.jpg \\
        --method cnn --backbone resnet18 \\
        --ground_truth '{"BOP": 0.4, "OP": 0.3, "Dust": 0.3}'
"""
import os
import sys
import json
import argparse
import joblib
import cv2
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "segmentation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "traditional_cv"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "deep_learning"))

from common import load_config, resolve_path, ensure_dir
from segment_particles import segment_image
from feature_extraction import extract_all_features
from model import build_model
from dataset import build_transforms


def classify_with_traditional(crops, cfg, model_name):
    """Classify a list of particle crop images (numpy arrays) using a saved traditional-CV bundle."""
    models_dir = resolve_path(cfg["paths"]["results_models"])
    bundle_path = os.path.join(models_dir, f"traditional_{model_name}.joblib")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}. Train it first.")

    bundle = joblib.load(bundle_path)
    model, scaler, label_encoder, feature_cols = (
        bundle["model"], bundle["scaler"], bundle["label_encoder"], bundle["feature_cols"]
    )

    cfg_cv = cfg["traditional_cv"]
    predictions, areas = [], []

    tmp_dir = "/tmp/mixture_crops"
    ensure_dir(tmp_dir)

    for i, crop in enumerate(crops):
        if crop.size == 0:
            continue
        tmp_path = os.path.join(tmp_dir, f"crop_{i}.png")
        cv2.imwrite(tmp_path, crop)

        feats = extract_all_features(tmp_path, cfg_cv)
        if feats is None:
            continue

        x = np.array([[feats[c] for c in feature_cols]])
        x_scaled = scaler.transform(x)
        pred_idx = model.predict(x_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        predictions.append(pred_label)
        # "area" was renamed to "area_mm2" (calibrated) / "area_px" (fallback)
        # when geometric features were fixed to measure the original, un-resized
        # crop - fall back through both keys so this keeps working either way.
        areas.append(feats.get("area_mm2", feats.get("area_px", 0.0)))

    return predictions, areas


def classify_with_cnn(crops, cfg, backbone_name):
    """Classify a list of particle crop images using a saved CNN checkpoint."""
    models_dir = resolve_path(cfg["paths"]["results_models"])
    checkpoint_path = os.path.join(models_dir, f"cnn_{backbone_name}_best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train it first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: loading our own training checkpoint (see train_cnn.py),
    # which stores metadata (class names etc.) alongside model weights.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    num_classes = checkpoint["num_classes"]

    cfg_dl = dict(cfg["deep_learning"])
    cfg_dl["backbone"] = backbone_name
    model = build_model(cfg_dl, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transforms(cfg_dl, split="test")  # no augmentation at inference

    predictions, areas = [], []
    with torch.no_grad():
        for crop in crops:
            if crop.size == 0:
                continue
            area = crop.shape[0] * crop.shape[1]  # bounding-box area as proxy
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = transform(pil_img).unsqueeze(0).to(device)

            output = model(tensor)
            pred_idx = output.argmax(dim=1).item()
            predictions.append(classes[pred_idx])
            areas.append(area)

    return predictions, areas


def compute_proportions(predictions, areas, grades, weighting="count"):
    """Convert a list of per-particle predicted labels into grade proportions."""
    proportions = {g: 0.0 for g in grades}

    if weighting == "area":
        total = sum(areas) if areas else 0
        if total == 0:
            return proportions
        for pred, area in zip(predictions, areas):
            if pred in proportions:
                proportions[pred] += area / total
    else:  # count-based
        total = len(predictions)
        if total == 0:
            return proportions
        for pred in predictions:
            if pred in proportions:
                proportions[pred] += 1.0 / total

    return proportions


def compute_errors(predicted, ground_truth, grades):
    """MAE and RMSE between predicted and actual proportions, per grade and overall."""
    errors = {}
    sq_errors = []
    abs_errors = []
    for g in grades:
        pred_val = predicted.get(g, 0.0)
        true_val = ground_truth.get(g, 0.0)
        err = pred_val - true_val
        errors[g] = {"predicted": pred_val, "actual": true_val, "error": err}
        abs_errors.append(abs(err))
        sq_errors.append(err ** 2)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    return errors, mae, rmse


def main():
    parser = argparse.ArgumentParser(description="Estimate tea grade mixture proportions from a sample image.")
    parser.add_argument("--image", type=str, required=True, help="Path to mixed sample image.")
    parser.add_argument("--method", type=str, required=True, choices=["traditional", "cnn"])
    parser.add_argument("--model_name", type=str, default="random_forest",
                         choices=["random_forest", "svm", "xgboost"],
                         help="Used when --method traditional")
    parser.add_argument("--backbone", type=str, default="resnet18",
                         help="Used when --method cnn")
    parser.add_argument("--ground_truth", type=str, default=None,
                         help='JSON string of actual proportions, e.g. \'{"BOP":0.4,"OP":0.6}\'')
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    grades = cfg["classes"]["grades"]

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    print("Segmenting particles...")
    crops, contours, _ = segment_image(image, cfg["segmentation"])
    print(f"Found {len(crops)} particles.")

    if not crops:
        print("[ERROR] No particles detected. Check segmentation parameters.")
        return

    print(f"Classifying particles using {args.method}...")
    if args.method == "traditional":
        predictions, areas = classify_with_traditional(crops, cfg, args.model_name)
        method_label = f"traditional_{args.model_name}"
    else:
        predictions, areas = classify_with_cnn(crops, cfg, args.backbone)
        method_label = f"cnn_{args.backbone}"

    weighting = cfg["mixture_estimation"]["weighting"]
    proportions = compute_proportions(predictions, areas, grades, weighting)

    print(f"\nPredicted composition ({weighting}-weighted):")
    for g, p in proportions.items():
        print(f"  {g:10s}: {p*100:.1f}%")

    result = {
        "image": args.image,
        "method": method_label,
        "weighting": weighting,
        "num_particles_detected": len(crops),
        "num_particles_classified": len(predictions),
        "predicted_proportions": proportions,
    }

    if args.ground_truth:
        ground_truth = json.loads(args.ground_truth)
        errors, mae, rmse = compute_errors(proportions, ground_truth, grades)
        result["ground_truth"] = ground_truth
        result["per_grade_errors"] = errors
        result["mae"] = mae
        result["rmse"] = rmse

        print(f"\nComparison with ground truth:")
        for g, e in errors.items():
            print(f"  {g:10s}: predicted={e['predicted']*100:.1f}%  actual={e['actual']*100:.1f}%  "
                  f"error={e['error']*100:+.1f}pp")
        print(f"\nMAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

    metrics_dir = resolve_path(cfg["paths"]["results_metrics"])
    ensure_dir(metrics_dir)
    image_name = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(metrics_dir, f"mixture_{image_name}_{method_label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result to {out_path}")


if __name__ == "__main__":
    main()
