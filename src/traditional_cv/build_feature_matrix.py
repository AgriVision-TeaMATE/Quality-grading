"""
Build Feature Matrix
======================
Runs feature_extraction.py over every particle in the manifest and saves
a single CSV: one row per particle, columns = handcrafted features + label + split.

This is a SEPARATE step from model training so that feature extraction
(slow, CPU-bound) only needs to run once; training/tuning experiments can
then quickly re-load the cached feature matrix.

Usage:
    python build_feature_matrix.py
"""
import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import load_config, resolve_path, ensure_dir
from feature_extraction import extract_all_features


def main():
    parser = argparse.ArgumentParser(description="Extract handcrafted features for all particles.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None,
                         help="Override path to manifest CSV (default: data/manifests/full_manifest.csv)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_dir = resolve_path(cfg["paths"]["manifests"])
    manifest_path = args.manifest or os.path.join(manifest_dir, "full_manifest.csv")

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found at {manifest_path}")
        print("Run src/utils/build_manifests.py first.")
        return

    df = pd.read_csv(manifest_path)
    cfg_cv = cfg["traditional_cv"]

    rows = []
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        feats = extract_all_features(row["filepath"], cfg_cv)
        if feats is None:
            failed += 1
            continue
        feats["filepath"] = row["filepath"]
        feats["label"] = row["label"]
        feats["split"] = row["split"]
        rows.append(feats)

    feature_df = pd.DataFrame(rows)
    out_path = os.path.join(manifest_dir, "feature_matrix.csv")
    feature_df.to_csv(out_path, index=False)

    print(f"\nExtracted features for {len(feature_df)} particles ({failed} failed/skipped).")
    print(f"Saved to: {out_path}")
    print(f"Feature columns: {len([c for c in feature_df.columns if c not in ('filepath', 'label', 'split')])}")


if __name__ == "__main__":
    main()
