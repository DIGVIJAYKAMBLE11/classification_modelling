"""
Score New Data — Deployment Script
====================================
Loads saved artefacts and scores raw data through the same pipeline
used during training.

All data (main + external columns) is expected in a single CSV file.
The script automatically splits the columns into main and external
based on the saved column_metadata.json.

Usage:
  python score_new_data.py --data new_data.csv --output predictions.csv
  python score_new_data.py --data new_data.csv --output predictions.csv --threshold 0.4

The script expects artefacts in ~/ml_artefacts (or override with --artefact-dir).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def score_new_data(raw_df, artefact_dir, ext_raw_df=None):
    """Score raw data using saved artefacts. Identical to deployment.

    raw_df    : DataFrame with main columns (from COLUMNS_LIST)
    ext_raw_df: DataFrame with external columns (from EXT_COLUMNS), same index as raw_df
    """
    model     = joblib.load(os.path.join(artefact_dir, "xgb_model.joblib"))
    ohe_enc   = joblib.load(os.path.join(artefact_dir, "ohe_encoder.joblib"))
    bin_edges = json.load(open(os.path.join(artefact_dir, "bin_edges.json")))
    col_meta  = json.load(open(os.path.join(artefact_dir, "column_metadata.json")))
    rare_map  = json.load(open(os.path.join(artefact_dir, "rare_mappings.json")))
    features  = json.load(open(os.path.join(artefact_dir, "final_features.json")))

    raw_df = raw_df.copy()

    # Bin numerical columns
    for col, info in bin_edges.items():
        edges = info["edges"]
        if col in raw_df.columns:
            raw_df[col + "_bin"] = pd.cut(
                raw_df[col], bins=edges, labels=False, include_lowest=True
            )
    raw_df.drop(
        columns=[c for c in col_meta["original_num_cols"] if c in raw_df.columns],
        inplace=True, errors="ignore",
    )

    # Handle rare categories
    for col, rares in rare_map.items():
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].apply(
                lambda x: "__rare__" if x in rares else x
            )

    # Fill nulls & OHE main features
    enc_cols = col_meta["encode_cols"]
    for col in enc_cols:
        if col in raw_df.columns and raw_df[col].isnull().any():
            raw_df[col] = raw_df[col].fillna("__NULL__")
        if col not in raw_df.columns:
            raw_df[col] = "__NULL__"
    raw_df[enc_cols] = raw_df[enc_cols].astype(str)
    encoded = ohe_enc.transform(raw_df[enc_cols])
    enc_df = pd.DataFrame(
        encoded,
        columns=ohe_enc.get_feature_names_out(enc_cols),
        index=raw_df.index,
    )

    # External features
    if col_meta.get("has_external") and ext_raw_df is not None:
        ext_raw_df = ext_raw_df.copy()
        ext_ohe_enc = joblib.load(
            os.path.join(artefact_dir, "ext_ohe_encoder.joblib")
        )
        ext_bin_edges = json.load(
            open(os.path.join(artefact_dir, "ext_bin_edges.json"))
        )
        ext_rare_map = json.load(
            open(os.path.join(artefact_dir, "ext_rare_mappings.json"))
        )

        # Indices already aligned (same CSV, same rows)
        ext_raw_df.index = raw_df.index

        for col, info in ext_bin_edges.items():
            if col in ext_raw_df.columns:
                ext_raw_df[col + "_bin"] = pd.cut(
                    ext_raw_df[col],
                    bins=info["edges"],
                    labels=False,
                    include_lowest=True,
                )
        ext_raw_df.drop(
            columns=[
                c for c in col_meta["ext_num_cols"] if c in ext_raw_df.columns
            ],
            inplace=True,
            errors="ignore",
        )

        for col, rares in ext_rare_map.items():
            if col in ext_raw_df.columns:
                ext_raw_df[col] = ext_raw_df[col].apply(
                    lambda x: "__rare__" if x in rares else x
                )

        ext_enc_cols = col_meta["ext_encode_cols"]
        for col in ext_enc_cols:
            if col in ext_raw_df.columns and ext_raw_df[col].isnull().any():
                ext_raw_df[col] = ext_raw_df[col].fillna("__NULL__")
            if col not in ext_raw_df.columns:
                ext_raw_df[col] = "__NULL__"
        ext_raw_df[ext_enc_cols] = ext_raw_df[ext_enc_cols].astype(str)
        ext_encoded = ext_ohe_enc.transform(ext_raw_df[ext_enc_cols])
        ext_enc_df = pd.DataFrame(
            ext_encoded,
            columns=ext_ohe_enc.get_feature_names_out(ext_enc_cols),
            index=raw_df.index,
        )
        enc_df = pd.concat([enc_df, ext_enc_df], axis=1)

    # Align to training features and predict
    for col in features:
        if col not in enc_df.columns:
            enc_df[col] = 0
    enc_df = enc_df[features]

    probabilities = model.predict_proba(enc_df)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return probabilities, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Score new data using saved classification pipeline artefacts."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the input CSV file (single file with all columns — main + external).",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to save the output CSV with predictions (default: predictions.csv).",
    )
    parser.add_argument(
        "--artefact-dir",
        default=str(Path.home() / "ml_artefacts"),
        help="Path to the artefact directory (default: ~/ml_artefacts).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.isfile(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    if not os.path.isdir(args.artefact_dir):
        print(f"Error: Artefact directory not found: {args.artefact_dir}")
        sys.exit(1)

    # Load column metadata
    col_meta_path = os.path.join(args.artefact_dir, "column_metadata.json")
    with open(col_meta_path) as f:
        col_meta = json.load(f)

    # Load data (single CSV with all columns)
    print(f"Loading data from: {args.data}")
    full_df = pd.read_csv(args.data)
    print(f"  Shape: {full_df.shape}")

    # Split into main and external columns automatically
    # Main columns = whatever's not in ext_columns and not the target
    ext_columns = col_meta.get("ext_columns", [])
    target_col = col_meta.get("target_col", "target")

    # Remove target if present (we don't need it for scoring)
    main_df = full_df.drop(columns=[target_col], errors="ignore")

    ext_raw_df = None
    if col_meta.get("has_external") and ext_columns:
        available_ext = [c for c in ext_columns if c in full_df.columns]
        if available_ext:
            ext_raw_df = full_df[available_ext].copy()
            print(f"  External columns found: {len(available_ext)}/{len(ext_columns)}")
        else:
            print(f"  WARNING: No external columns found in data.")

    # Score
    print("Scoring ...")
    probabilities, predictions = score_new_data(
        main_df, args.artefact_dir, ext_raw_df=ext_raw_df
    )

    # Apply custom threshold if not default
    if args.threshold != 0.5:
        predictions = (probabilities >= args.threshold).astype(int)
        print(f"  Using custom threshold: {args.threshold}")

    # Build output
    output_df = full_df.copy()
    output_df["prediction_probability"] = probabilities
    output_df["prediction"] = predictions

    # Save
    output_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {args.output}")
    print(f"  Total rows: {len(output_df)}")
    print(f"  Predicted positive: {predictions.sum()} ({predictions.mean()*100:.1f}%)")
    print(f"  Predicted negative: {(1-predictions).sum()} ({(1-predictions).mean()*100:.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
