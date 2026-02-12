"""
Classification Training Pipeline
=================================
Standalone script that replicates the notebook pipeline end-to-end.

Steps:
  1. Load data & holdout split (15% true out-of-sample)
  2. Column selection (drop / keep)
  3. Null analysis & column removal
  4. Class-wise variation analysis & low-variation column removal
  5. Identify categorical vs numerical columns
  6. Numerical binning
  7. One-hot encoding
  8. External data pipeline (optional)
  9. Save transformation artefacts
  10. Stratified train / test split
  11. XGBoost GridSearchCV
  12. Feature importance — smart auto-flag
  13. Evaluation metrics
  14. Save model & artefacts
  15. Holdout evaluation via deployment pipeline

Usage:
  python train_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.inspection import permutation_importance
from scipy.stats import ks_2samp, chi2_contingency
from xgboost import XGBClassifier


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    USER CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Data ───────────────────────────────────────────────────────────
DATA_PATH  = "connected_dataset_dpd0_20260207_141810.csv"
TARGET_COL = "target"
TARGET_RENAME = {"target_dpd0": "target"}  # rename applied after load

# ── Holdout ────────────────────────────────────────────────────────
HOLDOUT_PCT  = 0.15
HOLDOUT_SEED = 99

# ── Column selection ───────────────────────────────────────────────
MODE = "keep"  # "drop" or "keep"
COLUMNS_LIST = [
    "target", "l_principal", "l_partner_disburcement_amount", "l_discount",
    "l_term", "cl_cupo", "cl_cupo_disponible", "cedula", "platam_score",
    "hybrid_score", "platam_rating", "hybrid_rating", "categoria_madurez",
    "edad", "ingresos_smlv", "nivel_ingresos_encoded", "cuota_mensual",
    "ratio_cuota_ingreso", "creditos_vigentes", "creditos_mora",
    "hist_neg_12m", "departamento", "experian_score", "total_debt",
    "queries_6m", "queries_12m", "active_credits", "closed_credits",
    "total_entities", "negative_entities", "genero", "edad_promedio",
    "num_accounts", "total_account_balance", "total_arrears_balance",
    "num_accounts_in_arrears", "clr_type", "clr_doc_type",
    "clr_bus_relation", "clr_city", "clr_bus_type",
    "clr_bus_num_locations", "clr_bus_num_employees", "clr_bus_seniority",
    "clr_bus_monthly_purchases", "clr_bus_current_purchases",
    "clr_bus_monthly_income", "clr_bus_monthly_expenses",
    "clr_declares_rent", "clr_has_rent", "clr_rent", "clr_hcpn_status",
    "clr_credit_study_score", "clr_credit_study_result",
    "clr_credit_study_loc", "clr_requested_loc", "clr_risk_profile",
    "clr_sr_opinion_relationship_duration", "clr_agent_recommendation",
    "clr_agent_loc",
]

# ── Null threshold ─────────────────────────────────────────────────
NULL_THRESHOLD_PCT = 40

# ── Variation threshold & manual drops ─────────────────────────────
VARIATION_THRESHOLD = 0.02
DROP_LOW_VARIATION = [
    "clr_bus_current_purchases",
    "clr_requested_loc",
    "clr_rent",
]

# ── Binning ────────────────────────────────────────────────────────
N_BINS           = 5
BINNING_STRATEGY = "quantile"
CUSTOM_BINS      = {}

# ── OHE ────────────────────────────────────────────────────────────
DROP_FIRST     = False
MAX_CATEGORIES = None

# ── External data ──────────────────────────────────────────────────
EXTERNAL_DATA_PATH    = "final_enriched_output.csv"  # or None
EXT_MERGE_KEY         = "loan_id"
EXT_MODE              = "drop"
EXT_COLUMNS_LIST      = []
EXT_NULL_THRESHOLD_PCT    = NULL_THRESHOLD_PCT
EXT_VARIATION_THRESHOLD   = VARIATION_THRESHOLD
EXT_DROP_LOW_VARIATION    = []
EXT_N_BINS                = N_BINS
EXT_BINNING_STRATEGY      = BINNING_STRATEGY
EXT_CUSTOM_BINS           = {}
EXT_DROP_FIRST            = DROP_FIRST
EXT_MAX_CATEGORIES        = MAX_CATEGORIES

# ── Train / test split ─────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── GridSearchCV ───────────────────────────────────────────────────
CV_FOLDS = 5
SCORING  = "roc_auc"
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}

# ── Feature importance ─────────────────────────────────────────────
GAIN_TOP_PCT       = 50
PERM_MIN_THRESHOLD = 0.001
ADD_BACK_FEATURES  = []
EXTRA_DROP_FEATURES = []

# ── Artefact directory ─────────────────────────────────────────────
ARTEFACT_DIR = str(Path.home() / "ml_artefacts")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     HELPER FUNCTIONS                            ║
# ╚══════════════════════════════════════════════════════════════════╝

def cramers_v(col, target):
    ct = pd.crosstab(col, target)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0


def null_analysis(dataframe, columns, threshold_pct, target_col):
    null_pct = (dataframe[columns].isnull().sum() / len(dataframe) * 100).sort_values(ascending=False)
    cols_to_drop = null_pct[null_pct > threshold_pct].index.tolist()
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)
    return cols_to_drop


def variation_analysis(dataframe, columns, target_col, threshold):
    scores = {}
    for col in columns:
        if dataframe[col].nunique() < 2:
            scores[col] = 0.0
            continue
        if dataframe[col].dtype in ["object", "category"] or dataframe[col].nunique() <= 10:
            scores[col] = cramers_v(dataframe[col].fillna("__NULL__"), dataframe[target_col])
        else:
            classes = dataframe[target_col].unique()
            grp0 = dataframe.loc[dataframe[target_col] == classes[0], col].dropna()
            grp1 = dataframe.loc[dataframe[target_col] == classes[1], col].dropna()
            scores[col] = ks_2samp(grp0, grp1)[0] if len(grp0) > 0 and len(grp1) > 0 else 0.0
    flagged = [c for c, v in scores.items() if v < threshold]
    return flagged, scores


def identify_col_types(dataframe, feature_cols):
    cat = [c for c in feature_cols if dataframe[c].dtype in ["object", "category"] or dataframe[c].nunique() <= 10]
    num = [c for c in feature_cols if c not in cat]
    return cat, num


def bin_columns(dataframe, num_cols, n_bins, strategy, custom_bins, bin_store):
    for col in num_cols:
        series = dataframe[col].dropna()
        if col in custom_bins:
            edges = custom_bins[col]
            dataframe[col + "_bin"] = pd.cut(dataframe[col], bins=edges, labels=False, include_lowest=True)
            bin_store[col] = {"type": "custom", "edges": edges}
        else:
            binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy, subsample=None)
            valid_mask = dataframe[col].notna()
            dataframe.loc[valid_mask, col + "_bin"] = binner.fit_transform(
                dataframe.loc[valid_mask, [col]]
            ).ravel()
            bin_store[col] = {"type": strategy, "edges": binner.bin_edges_[0].tolist(), "n_bins": n_bins}
        print(f"  Binned: {col}")
    dataframe.drop(columns=num_cols, inplace=True)
    return dataframe, bin_store


def ohe_columns(dataframe, target_col, drop_first, max_categories, rare_map_store):
    encode_cols = [c for c in dataframe.columns if c != target_col]
    if max_categories is not None:
        for col in encode_cols:
            counts = dataframe[col].value_counts()
            rare_cats = counts[counts < max_categories].index.tolist()
            if rare_cats:
                dataframe[col] = dataframe[col].apply(lambda x: "__rare__" if x in rare_cats else x)
                rare_map_store[col] = rare_cats
    for col in encode_cols:
        if dataframe[col].isnull().any():
            dataframe[col] = dataframe[col].fillna("__NULL__")
    dataframe[encode_cols] = dataframe[encode_cols].astype(str)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                        drop="first" if drop_first else None)
    encoded = ohe.fit_transform(dataframe[encode_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(encode_cols),
                              index=dataframe.index)
    result = pd.concat([encoded_df, dataframe[[target_col]]], axis=1)
    return result, encode_cols, ohe, rare_map_store


def score_new_data(raw_df, artefact_dir, ext_raw_df=None, merge_key=None):
    """Score raw data using saved artefacts. Identical to deployment."""
    model     = joblib.load(os.path.join(artefact_dir, "xgb_model.joblib"))
    ohe_enc   = joblib.load(os.path.join(artefact_dir, "ohe_encoder.joblib"))
    bin_edges = json.load(open(os.path.join(artefact_dir, "bin_edges.json")))
    col_meta  = json.load(open(os.path.join(artefact_dir, "column_metadata.json")))
    rare_map  = json.load(open(os.path.join(artefact_dir, "rare_mappings.json")))
    features  = json.load(open(os.path.join(artefact_dir, "final_features.json")))

    raw_df = raw_df.copy()
    for col, info in bin_edges.items():
        if col in raw_df.columns:
            raw_df[col + "_bin"] = pd.cut(raw_df[col], bins=info["edges"], labels=False, include_lowest=True)
    raw_df.drop(columns=[c for c in col_meta["original_num_cols"] if c in raw_df.columns], inplace=True, errors="ignore")

    for col, rares in rare_map.items():
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].apply(lambda x: "__rare__" if x in rares else x)

    enc_cols = col_meta["encode_cols"]
    for col in enc_cols:
        if col in raw_df.columns and raw_df[col].isnull().any():
            raw_df[col] = raw_df[col].fillna("__NULL__")
        if col not in raw_df.columns:
            raw_df[col] = "__NULL__"
    raw_df[enc_cols] = raw_df[enc_cols].astype(str)
    encoded = ohe_enc.transform(raw_df[enc_cols])
    enc_df = pd.DataFrame(encoded, columns=ohe_enc.get_feature_names_out(enc_cols), index=raw_df.index)

    if col_meta.get("has_external") and ext_raw_df is not None:
        ext_raw_df = ext_raw_df.copy()
        ext_ohe_enc   = joblib.load(os.path.join(artefact_dir, "ext_ohe_encoder.joblib"))
        ext_bin_edges = json.load(open(os.path.join(artefact_dir, "ext_bin_edges.json")))
        ext_rare_map  = json.load(open(os.path.join(artefact_dir, "ext_rare_mappings.json")))

        if merge_key and merge_key in raw_df.columns:
            ext_raw_df = raw_df[[merge_key]].merge(ext_raw_df, on=merge_key, how="left")
            ext_raw_df.index = raw_df.index

        for col, info in ext_bin_edges.items():
            if col in ext_raw_df.columns:
                ext_raw_df[col + "_bin"] = pd.cut(ext_raw_df[col], bins=info["edges"], labels=False, include_lowest=True)
        ext_raw_df.drop(columns=[c for c in col_meta["ext_num_cols"] if c in ext_raw_df.columns], inplace=True, errors="ignore")

        for col, rares in ext_rare_map.items():
            if col in ext_raw_df.columns:
                ext_raw_df[col] = ext_raw_df[col].apply(lambda x: "__rare__" if x in rares else x)

        ext_enc_cols = col_meta["ext_encode_cols"]
        for col in ext_enc_cols:
            if col in ext_raw_df.columns and ext_raw_df[col].isnull().any():
                ext_raw_df[col] = ext_raw_df[col].fillna("__NULL__")
            if col not in ext_raw_df.columns:
                ext_raw_df[col] = "__NULL__"
        ext_raw_df[ext_enc_cols] = ext_raw_df[ext_enc_cols].astype(str)
        ext_encoded = ext_ohe_enc.transform(ext_raw_df[ext_enc_cols])
        ext_enc_df = pd.DataFrame(ext_encoded, columns=ext_ohe_enc.get_feature_names_out(ext_enc_cols), index=raw_df.index)
        enc_df = pd.concat([enc_df, ext_enc_df], axis=1)

    for col in features:
        if col not in enc_df.columns:
            enc_df[col] = 0
    enc_df = enc_df[features]
    return model.predict_proba(enc_df)[:, 1]


# ╔══════════════════════════════════════════════════════════════════╗
# ║                      MAIN PIPELINE                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    os.makedirs(ARTEFACT_DIR, exist_ok=True)
    print(f"Artefact directory: {ARTEFACT_DIR}")

    # ── 1. Load data ───────────────────────────────────────────────
    print("\n[1] Loading data ...")
    df = pd.read_csv(DATA_PATH)
    df.rename(columns=TARGET_RENAME, inplace=True)
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution:\n{df[TARGET_COL].value_counts(normalize=True)}")

    # ── 1a. Holdout split ──────────────────────────────────────────
    print(f"\n[1a] Holdout split ({HOLDOUT_PCT*100:.0f}%) ...")
    df_dev, df_holdout = train_test_split(
        df, test_size=HOLDOUT_PCT, random_state=HOLDOUT_SEED, stratify=df[TARGET_COL]
    )
    holdout_raw = df_holdout.copy()
    df = df_dev.reset_index(drop=True)
    print(f"  Dev: {len(df)} rows | Holdout: {len(holdout_raw)} rows")

    # ── 2. Column selection ────────────────────────────────────────
    print("\n[2] Column selection ...")
    if MODE == "drop":
        df.drop(columns=[c for c in COLUMNS_LIST if c in df.columns], inplace=True)
    elif MODE == "keep":
        keep = list(set(COLUMNS_LIST + [TARGET_COL]))
        df = df[[c for c in keep if c in df.columns]]
    print(f"  Shape after selection: {df.shape}")

    # ── 3. Null analysis ───────────────────────────────────────────
    print("\n[3] Null analysis ...")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    null_drops = null_analysis(df, feature_cols, NULL_THRESHOLD_PCT, TARGET_COL)
    df.drop(columns=null_drops, inplace=True)
    print(f"  Dropped {len(null_drops)} columns: {null_drops}")

    # ── 4. Variation analysis ──────────────────────────────────────
    print("\n[4] Variation analysis ...")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    flagged, var_scores = variation_analysis(df, feature_cols, TARGET_COL, VARIATION_THRESHOLD)
    print(f"  Flagged: {flagged}")
    df.drop(columns=[c for c in DROP_LOW_VARIATION if c in df.columns], inplace=True)
    print(f"  Dropped: {DROP_LOW_VARIATION}")

    # ── 5. Identify column types ───────────────────────────────────
    print("\n[5] Identifying column types ...")
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    cat_cols, num_cols = identify_col_types(df, feature_cols)
    print(f"  Categorical: {len(cat_cols)} | Numerical: {len(num_cols)}")

    # ── 6. Binning ─────────────────────────────────────────────────
    print("\n[6] Binning numerical columns ...")
    bin_edges_store = {}
    df, bin_edges_store = bin_columns(df, num_cols, N_BINS, BINNING_STRATEGY, CUSTOM_BINS, bin_edges_store)

    # ── 7. OHE ─────────────────────────────────────────────────────
    print("\n[7] One-hot encoding ...")
    rare_mappings = {}
    df, encode_cols, ohe, rare_mappings = ohe_columns(df, TARGET_COL, DROP_FIRST, MAX_CATEGORIES, rare_mappings)
    main_feature_cols = [c for c in df.columns if c != TARGET_COL]
    print(f"  Shape after OHE: {df.shape}")

    # ── 8. External data ───────────────────────────────────────────
    ext_bin_edges_store = {}
    ext_rare_mappings = {}
    ext_ohe = None
    ext_encode_cols = []
    ext_cat_cols = []
    ext_num_cols = []
    ext_df = None
    holdout_ext_raw = None

    if EXTERNAL_DATA_PATH is not None and EXT_MERGE_KEY is not None:
        print("\n[8] External data pipeline ...")
        ext_raw = pd.read_csv(EXTERNAL_DATA_PATH)
        main_raw_full = pd.read_csv(DATA_PATH)
        main_raw_full.rename(columns=TARGET_RENAME, inplace=True)

        main_dev_raw, main_holdout_raw = train_test_split(
            main_raw_full, test_size=HOLDOUT_PCT, random_state=HOLDOUT_SEED,
            stratify=main_raw_full[TARGET_COL]
        )
        main_dev_raw = main_dev_raw.reset_index(drop=True)

        # Dev external
        ext_merged = main_dev_raw[[EXT_MERGE_KEY, TARGET_COL]].merge(ext_raw, on=EXT_MERGE_KEY, how="left")
        ext_merged.index = df.index
        ext_feature_names = [c for c in ext_merged.columns if c not in [EXT_MERGE_KEY, TARGET_COL]]
        ext_df = ext_merged[ext_feature_names + [TARGET_COL]].copy()

        # Holdout external
        holdout_ext_merged = main_holdout_raw[[EXT_MERGE_KEY, TARGET_COL]].merge(ext_raw, on=EXT_MERGE_KEY, how="left")
        holdout_ext_raw = holdout_ext_merged.drop(columns=[TARGET_COL]).reset_index(drop=True)

        # Drop/keep
        if EXT_MODE == "drop":
            ext_df.drop(columns=[c for c in EXT_COLUMNS_LIST if c in ext_df.columns], inplace=True)
        elif EXT_MODE == "keep":
            keep = list(set(EXT_COLUMNS_LIST + [TARGET_COL]))
            ext_df = ext_df[[c for c in keep if c in ext_df.columns]]

        # Null analysis
        ext_feature_cols = [c for c in ext_df.columns if c != TARGET_COL]
        ext_null_drops = null_analysis(ext_df, ext_feature_cols, EXT_NULL_THRESHOLD_PCT, TARGET_COL)
        ext_df.drop(columns=ext_null_drops, inplace=True)
        print(f"  Ext null drops: {len(ext_null_drops)}")

        # Variation
        ext_feature_cols = [c for c in ext_df.columns if c != TARGET_COL]
        ext_flagged, _ = variation_analysis(ext_df, ext_feature_cols, TARGET_COL, EXT_VARIATION_THRESHOLD)
        ext_df.drop(columns=[c for c in EXT_DROP_LOW_VARIATION if c in ext_df.columns], inplace=True)

        # Types, binning, OHE
        ext_feature_cols = [c for c in ext_df.columns if c != TARGET_COL]
        if len(ext_feature_cols) > 0:
            ext_cat_cols, ext_num_cols = identify_col_types(ext_df, ext_feature_cols)
            if ext_num_cols:
                ext_df, ext_bin_edges_store = bin_columns(
                    ext_df, ext_num_cols, EXT_N_BINS, EXT_BINNING_STRATEGY, EXT_CUSTOM_BINS, ext_bin_edges_store
                )
            ext_df, ext_encode_cols, ext_ohe, ext_rare_mappings = ohe_columns(
                ext_df, TARGET_COL, EXT_DROP_FIRST, EXT_MAX_CATEGORIES, ext_rare_mappings
            )
            # Concatenate
            ext_only = ext_df.drop(columns=[TARGET_COL])
            df = pd.concat([df.drop(columns=[TARGET_COL]), ext_only, df[[TARGET_COL]]], axis=1)
            print(f"  Combined shape: {df.shape}")
        else:
            ext_df = None
    else:
        print("\n[8] No external data — skipping.")

    # ── 9. Save artefacts ──────────────────────────────────────────
    print("\n[9] Saving transformation artefacts ...")
    with open(os.path.join(ARTEFACT_DIR, "bin_edges.json"), "w") as f:
        json.dump(bin_edges_store, f, indent=2)
    joblib.dump(ohe, os.path.join(ARTEFACT_DIR, "ohe_encoder.joblib"))
    with open(os.path.join(ARTEFACT_DIR, "rare_mappings.json"), "w") as f:
        json.dump(rare_mappings, f, indent=2)

    if ext_df is not None:
        with open(os.path.join(ARTEFACT_DIR, "ext_bin_edges.json"), "w") as f:
            json.dump(ext_bin_edges_store, f, indent=2)
        joblib.dump(ext_ohe, os.path.join(ARTEFACT_DIR, "ext_ohe_encoder.joblib"))
        with open(os.path.join(ARTEFACT_DIR, "ext_rare_mappings.json"), "w") as f:
            json.dump(ext_rare_mappings, f, indent=2)

    col_meta = {
        "original_cat_cols": cat_cols, "original_num_cols": num_cols,
        "encode_cols": encode_cols, "main_feature_cols": main_feature_cols,
        "ext_cat_cols": ext_cat_cols, "ext_num_cols": ext_num_cols,
        "ext_encode_cols": ext_encode_cols,
        "final_feature_cols": [c for c in df.columns if c != TARGET_COL],
        "target_col": TARGET_COL, "has_external": ext_df is not None,
        "ext_merge_key": EXT_MERGE_KEY if ext_df is not None else None,
    }
    with open(os.path.join(ARTEFACT_DIR, "column_metadata.json"), "w") as f:
        json.dump(col_meta, f, indent=2)
    with open(os.path.join(ARTEFACT_DIR, "holdout_meta.json"), "w") as f:
        json.dump({"holdout_pct": HOLDOUT_PCT, "holdout_seed": HOLDOUT_SEED,
                    "holdout_rows": len(holdout_raw), "dev_rows": len(df)}, f, indent=2)

    # ── 10. Train / test split ─────────────────────────────────────
    print("\n[10] Stratified train / test split ...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 11. GridSearchCV ───────────────────────────────────────────
    print("\n[11] XGBoost GridSearchCV ...")
    neg, pos = np.bincount(y_train.astype(int))
    scale_pos_weight = neg / pos if pos > 0 else 1
    print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=xgb_base, param_grid=PARAM_GRID, scoring=SCORING,
        cv=skf, n_jobs=-1, verbose=1, refit=True,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"  Best {SCORING}: {grid_search.best_score_:.4f}")
    print(f"  Best params: {grid_search.best_params_}")

    # ── 12. Feature importance ─────────────────────────────────────
    print("\n[12] Feature importance & auto-flag ...")
    imp = pd.Series(best_model.feature_importances_, index=X_train.columns)
    perm_result = permutation_importance(
        best_model, X_test, y_test, n_repeats=10,
        random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )
    perm_imp = pd.Series(perm_result.importances_mean, index=X_test.columns)

    gain_cutoff = np.percentile(imp.values, 100 - GAIN_TOP_PCT)
    suspicious = imp.index[(imp >= gain_cutoff) & (perm_imp < PERM_MIN_THRESHOLD)].tolist()
    print(f"  Suspicious features: {suspicious}")

    auto_drop = [f for f in suspicious if f not in ADD_BACK_FEATURES]
    final_drop = list(set(auto_drop + EXTRA_DROP_FEATURES))
    if final_drop:
        X_train = X_train.drop(columns=final_drop, errors="ignore")
        X_test  = X_test.drop(columns=final_drop, errors="ignore")
        print(f"  Dropped {len(final_drop)} features, retraining ...")
        best_model.fit(X_train, y_train)
    print(f"  Final feature count: {X_train.shape[1]}")

    # ── 13. Evaluation metrics ─────────────────────────────────────
    print("\n[13] Dev test metrics ...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy":          accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "F1 Score":          f1_score(y_test, y_pred),
        "AUC (ROC)":         roc_auc_score(y_test, y_prob),
        "Gini":              2 * roc_auc_score(y_test, y_prob) - 1,
    }
    for name, val in metrics.items():
        print(f"  {name:<20s}  {val:.4f}")

    # ── 14. Save model ─────────────────────────────────────────────
    print("\n[14] Saving model & artefacts ...")
    joblib.dump(best_model, os.path.join(ARTEFACT_DIR, "xgb_model.joblib"))
    with open(os.path.join(ARTEFACT_DIR, "final_features.json"), "w") as f:
        json.dump(X_train.columns.tolist(), f, indent=2)
    with open(os.path.join(ARTEFACT_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ARTEFACT_DIR, "best_params.json"), "w") as f:
        json.dump(grid_search.best_params_, f, indent=2)
    with open(os.path.join(ARTEFACT_DIR, "dropped_features.json"), "w") as f:
        json.dump({"auto_flagged_suspicious": suspicious, "added_back": ADD_BACK_FEATURES,
                    "extra_manual_drops": EXTRA_DROP_FEATURES, "final_dropped": final_drop}, f, indent=2)

    # ── 15. Holdout evaluation ─────────────────────────────────────
    print("\n[15] Holdout (out-of-sample) evaluation ...")
    holdout_df = holdout_raw.copy()
    if MODE == "keep":
        keep_cols = list(set(COLUMNS_LIST + [TARGET_COL]))
        holdout_df = holdout_df[[c for c in keep_cols if c in holdout_df.columns]]
    elif MODE == "drop":
        holdout_df.drop(columns=[c for c in COLUMNS_LIST if c in holdout_df.columns], inplace=True)

    y_holdout = holdout_df[TARGET_COL]
    holdout_features = holdout_df.drop(columns=[TARGET_COL])

    h_ext = holdout_ext_raw if EXTERNAL_DATA_PATH is not None else None
    h_merge = EXT_MERGE_KEY if EXTERNAL_DATA_PATH is not None else None

    h_prob = score_new_data(holdout_features, ARTEFACT_DIR, ext_raw_df=h_ext, merge_key=h_merge)
    h_pred = (h_prob >= 0.5).astype(int)

    holdout_metrics = {
        "Accuracy":          accuracy_score(y_holdout, h_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_holdout, h_pred),
        "F1 Score":          f1_score(y_holdout, h_pred),
        "AUC (ROC)":         roc_auc_score(y_holdout, h_prob),
        "Gini":              2 * roc_auc_score(y_holdout, h_prob) - 1,
    }

    print("\n  " + "=" * 55)
    print("  DEV TEST vs HOLDOUT")
    print("  " + "=" * 55)
    for name in metrics:
        diff = holdout_metrics[name] - metrics[name]
        print(f"  {name:<20s}  Dev={metrics[name]:.4f}  Holdout={holdout_metrics[name]:.4f}  Diff={diff:+.4f}")
    print("  " + "=" * 55)

    with open(os.path.join(ARTEFACT_DIR, "holdout_metrics.json"), "w") as f:
        json.dump(holdout_metrics, f, indent=2)

    print(f"\nAll artefacts saved to: {ARTEFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
