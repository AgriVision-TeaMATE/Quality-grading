"""Same as build_feature_matrix.py but processes a row range at a time and
APPENDS to the output CSV, so it can be run across several short calls
(useful in sandboxes with a hard per-command time limit).

Usage:
    python build_feature_matrix_chunked.py --start 0 --end 200
    python build_feature_matrix_chunked.py --start 200 --end 400
    ...
    python build_feature_matrix_chunked.py --finalize   # sorts/dedupes at the end (optional)
"""
import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import load_config, resolve_path, get_px_per_mm
from feature_extraction import extract_all_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_dir = resolve_path(cfg["paths"]["manifests"])
    manifest_path = os.path.join(manifest_dir, "full_manifest.csv")
    out_path = os.path.join(manifest_dir, "feature_matrix.csv")

    df = pd.read_csv(manifest_path)
    end = args.end if args.end is not None else len(df)
    chunk = df.iloc[args.start:end]

    cfg_cv = cfg["traditional_cv"]
    rows = []
    failed = 0
    for _, row in chunk.iterrows():
        # Each grade may be captured at a different zoom level - look up the
        # calibration that matches THIS particle's grade, not one global value.
        cfg_cv_row = dict(cfg_cv)
        cfg_cv_row["px_per_mm"] = get_px_per_mm(cfg, row["label"])
        feats = extract_all_features(row["filepath"], cfg_cv_row)
        if feats is None:
            failed += 1
            continue
        feats["filepath"] = row["filepath"]
        feats["label"] = row["label"]
        feats["split"] = row["split"]
        rows.append(feats)

    chunk_df = pd.DataFrame(rows)
    write_header = not os.path.exists(out_path)
    chunk_df.to_csv(out_path, mode="a", header=write_header, index=False)
    print(f"Rows {args.start}:{end} -> extracted {len(chunk_df)} ({failed} failed). Appended to {out_path}")


if __name__ == "__main__":
    main()
