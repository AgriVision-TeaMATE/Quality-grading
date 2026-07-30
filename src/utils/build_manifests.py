"""
Build Dataset Manifests
=========================
Scans the segmented particle directory (data/segmented/<grade>/*.png) and
creates stratified train/val/test split CSVs. Both the traditional-CV
pipeline and the deep-learning pipeline read from these SAME manifests,
which is essential for a fair, reproducible comparison between approaches.

Usage:
    python build_manifests.py
"""
import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import load_config, resolve_path, ensure_dir


def scan_segmented_data(segmented_dir, grades):
    """Walk data/segmented/<grade>/ and build a flat list of (filepath, label)."""
    records = []
    for grade in grades:
        grade_dir = os.path.join(segmented_dir, grade)
        if not os.path.isdir(grade_dir):
            print(f"  [WARN] No directory found for grade '{grade}' at {grade_dir}")
            continue
        files = [
            f for f in os.listdir(grade_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for f in files:
            records.append({
                "filepath": os.path.join(grade_dir, f),
                "label": grade,
            })
    return pd.DataFrame(records)


def stratified_split(df, train_frac, val_frac, test_frac, seed, stratify=True):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    strat_col = df["label"] if stratify else None
    train_df, temp_df = train_test_split(
        df, train_size=train_frac, random_state=seed, stratify=strat_col
    )

    remaining_frac = val_frac + test_frac
    strat_col_temp = temp_df["label"] if stratify else None
    val_df, test_df = train_test_split(
        temp_df, train_size=val_frac / remaining_frac,
        random_state=seed, stratify=strat_col_temp
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Build train/val/test manifests.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    segmented_dir = resolve_path(cfg["paths"]["segmented_data"])
    manifest_dir = resolve_path(cfg["paths"]["manifests"])
    ensure_dir(manifest_dir)

    grades = cfg["classes"]["grades"]
    df = scan_segmented_data(segmented_dir, grades)

    if df.empty:
        print("\n[ERROR] No particle images found. Did you run segmentation first?")
        print(f"Expected structure: {segmented_dir}/<grade>/*.png")
        return

    print("Class distribution found:")
    print(df["label"].value_counts().to_string())

    split_cfg = cfg["split"]
    full_df = stratified_split(
        df,
        train_frac=split_cfg["train"],
        val_frac=split_cfg["val"],
        test_frac=split_cfg["test"],
        seed=split_cfg["random_seed"],
        stratify=split_cfg["stratify"],
    )

    out_path = os.path.join(manifest_dir, "full_manifest.csv")
    full_df.to_csv(out_path, index=False)

    print(f"\nSaved manifest with {len(full_df)} particles to {out_path}")
    print("\nSplit sizes:")
    print(full_df["split"].value_counts().to_string())
    print("\nPer-split class distribution:")
    print(full_df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
