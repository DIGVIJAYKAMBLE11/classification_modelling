"""
Score New Data — Deployment Script
====================================
Loads saved artefacts and scores raw data through the same pipeline
used during training.

Usage:
  python score_new_data.py --data new_data.csv --output predictions.csv
  python score_new_data.py --data new_data.csv --ext-data external.csv --output predictions.csv

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


def score_new_data(raw_df, artefact_dir, ext_raw_df=None, merge_key=None):
    """Score raw data using saved artefacts. Identical to deployment."""
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

    # External features (if applicable)
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

        if merge_key and merge_key in raw_df.columns:
            ext_raw_df = raw_df[[merge_key]].merge(
                ext_raw_df, on=merge_key, how="left"
            )
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
        "--data", required=True, help="Path to the input CSV file with raw features."
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
        "--ext-data",
        default=None,
        help="Path to external data CSV (optional).",
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

    # Load column metadata for merge key
    col_meta_path = os.path.join(args.artefact_dir, "column_metadata.json")
    with open(col_meta_path) as f:
        col_meta = json.load(f)
    merge_key = col_meta.get("ext_merge_key")

    # Load data
    print(f"Loading data from: {args.data}")
    raw_df = pd.read_csv(args.data)
    print(f"  Shape: {raw_df.shape}")

    # Load external data if provided
    ext_raw_df = None
    if args.ext_data is not None:
        if not os.path.isfile(args.ext_data):
            print(f"Error: External data file not found: {args.ext_data}")
            sys.exit(1)
        print(f"Loading external data from: {args.ext_data}")
        ext_raw_df = pd.read_csv(args.ext_data)
        print(f"  Shape: {ext_raw_df.shape}")

    # Score
    print("Scoring ...")
    probabilities, predictions = score_new_data(
        raw_df, args.artefact_dir, ext_raw_df=ext_raw_df, merge_key=merge_key
    )

    # Apply custom threshold if not default
    if args.threshold != 0.5:
        predictions = (probabilities >= args.threshold).astype(int)
        print(f"  Using custom threshold: {args.threshold}")

    # Build output
    output_df = raw_df.copy()
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
