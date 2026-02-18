"""
Classification Training Pipeline
=================================
Standalone script that replicates the notebook pipeline end-to-end.

All data (main + external columns) lives in a single CSV file.
- Main columns go through the main pipeline (sections 2-7)
- External columns go through the external pipeline (section 8)
- Both are concatenated before train/test split

Steps:
  1. Load data & holdout split (15% true out-of-sample)
  2. Column selection (keep COLUMNS_LIST)
  3. Null analysis & column removal
  4. Class-wise variation analysis & low-variation column removal
  5. Identify categorical vs numerical columns
  6. Numerical binning
  7. One-hot encoding
  8. External columns pipeline (from same CSV)
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

# ── Data (single combined CSV) ────────────────────────────────────
DATA_PATH  = "final_enriched_output.csv"
TARGET_COL = "target"
TARGET_RENAME = {"target_dpd0": "target"}  # rename applied after load

# ── Holdout ────────────────────────────────────────────────────────
HOLDOUT_PCT  = 0.15
HOLDOUT_SEED = 99

# ── Main column selection ─────────────────────────────────────────
MODE = "keep"  # "drop" or "keep"
COLUMNS_LIST = [
    "target",
    "l_principal",
    "l_partner_disburcement_amount",
    "l_discount",
    "l_term",
    "cl_cupo",
    "cl_cupo_disponible",
    "cedula",
    "platam_score",
    "hybrid_score",
    "platam_rating",
    "hybrid_rating",
    "categoria_madurez",
    "edad",
    "ingresos_smlv",
    "nivel_ingresos_encoded",
    "cuota_mensual",
    "ratio_cuota_ingreso",
    "creditos_vigentes",
    "creditos_mora",
    "hist_neg_12m",
    "departamento",
    "experian_score",
    "total_debt",
    "queries_6m",
    "queries_12m",
    "active_credits",
    "closed_credits",
    "total_entities",
    "negative_entities",
    "genero",
    "edad_promedio",
    "num_accounts",
    "total_account_balance",
    "total_arrears_balance",
    "num_accounts_in_arrears",
    "clr_type",
    "clr_doc_type",
    "clr_bus_relation",
    "clr_city",
    "clr_bus_type",
    "clr_bus_num_locations",
    "clr_bus_num_employees",
    "clr_bus_seniority",
    "clr_bus_monthly_purchases",
    "clr_bus_current_purchases",
    "clr_bus_monthly_income",
    "clr_bus_monthly_expenses",
    "clr_declares_rent",
    "clr_has_rent",
    "clr_rent",
    "clr_hcpn_status",
    "clr_credit_study_score",
    "clr_credit_study_result",
    "clr_credit_study_loc",
    "clr_requested_loc",
    "clr_risk_profile",
    "clr_sr_opinion_relationship_duration",
    "clr_agent_recommendation",
    "clr_agent_loc",
    "industry",
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

# ── Column type overrides ─────────────────────────────────────────
# Columns with ≤10 unique values are auto-classified as categorical.
# List columns here that should stay numeric despite low cardinality.
CAT_NUNIQUE_THRESHOLD = 10
FORCE_NUMERIC = []   # e.g. ["creditos_mora", "hist_neg_12m"]

# ── Binning ────────────────────────────────────────────────────────
N_BINS           = 5
BINNING_STRATEGY = "quantile"
CUSTOM_BINS      = {}
BINNING_FALLBACK = "categorical"  # "categorical" = move problem columns to OHE, "drop" = remove them

# ── OHE ────────────────────────────────────────────────────────────
DROP_FIRST     = False
MAX_CATEGORIES = None

# ── External columns (from the same CSV) ──────────────────────────
# Set to None or [] to skip external pipeline entirely
EXT_COLUMNS = [
    "rating",
    "user_ratings_total",
    "photo_count",
    "industry_confidence",
    "gross_revenue_midpoint",
    "gross_revenue_low",
    "gross_revenue_high",
    "net_profit_midpoint",
    "owner_income_midpoint",
    "financial_strength_score",
    "strengths_count",
    "weaknesses_count",
    "opportunities_count",
    "threats_count",
    "total_swot_factors",
    "digital_health_score",
    "trust_score",
    "online_presence_score",
    "business_health_score",
    "monnai_final_score",
    "monnai_score_group",
    "fraud_risk_score",
    "google_reviews_count",
    "social_googlemaps_place_general_rating",
    "social_googlemaps_overall_place_riviews",
    "social_googlemaps_local_guide_reviewer_count",
    "social_googlemaps_category",
    "social_googlemaps_positive_reviews_share",
    "social_googlemaps_negative_reviews_share",
    "social_googlemaps_neutral_reviews_share",
    "social_facebook_num_comments_Total",
    "social_facebook_num_shares_Total",
    "social_facebook_page_category",
    "social_facebook_page_followers",
    "social_facebook_page_is_verified",
    "social_facebook_video_view_count_Total",
    "social_facebook_likes_post_Total",
    "social_facebook_positive_post_share",
    "social_facebook_negative_posts_share",
    "social_facebook_neutral_post_share",
    "social_instagram_posts_count",
    "social_instagram_is_business_account",
    "social_instagram_is_professional_account",
    "social_instagram_is_verified",
    "social_instagram_avg_engagement",
    "social_instagram_category_name",
    "social_instagram_following",
    "social_instagram_highlights_count",
    "social_instagram_full_name",
    "social_instagram_is_private",
    "social_instagram_is_joined_recently",
    "social_instagram_has_channel",
    "social_instagram_post_likes_total_count",
    "social_instagram_post_likes_average_count",
    "social_instagram_post_total_comments",
    "social_instagram_post_average_comments",
    "social_instagram_content_type_Image",
    "social_instagram_content_type_Video",
    "social_instagram_content_type_Reel",
    "social_instagram_content_type_Carousel",
    "social_instagram_video_view_count_total",
    "social_instagram_video_view_count_average",
    "social_instagram_positive_post_share",
    "social_instagram_negative_posts_share",
    "social_instagram_neutral_post_share",
]

EXT_NULL_THRESHOLD_PCT    = NULL_THRESHOLD_PCT
EXT_VARIATION_THRESHOLD   = VARIATION_THRESHOLD
EXT_DROP_LOW_VARIATION    = []
EXT_CAT_NUNIQUE_THRESHOLD = CAT_NUNIQUE_THRESHOLD
EXT_FORCE_NUMERIC         = []   # e.g. ["trust_score", "strengths_count"]
EXT_N_BINS                = N_BINS
EXT_BINNING_STRATEGY      = BINNING_STRATEGY
EXT_CUSTOM_BINS           = {}
EXT_BINNING_FALLBACK      = BINNING_FALLBACK
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
ARTEFACT_DIR = "artefacts"


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
        if not pd.api.types.is_numeric_dtype(dataframe[col]) or dataframe[col].nunique() <= 10:
            scores[col] = cramers_v(dataframe[col].fillna("__NULL__"), dataframe[target_col])
        else:
            classes = dataframe[target_col].unique()
            grp0 = dataframe.loc[dataframe[target_col] == classes[0], col].dropna()
            grp1 = dataframe.loc[dataframe[target_col] == classes[1], col].dropna()
            scores[col] = ks_2samp(grp0, grp1)[0] if len(grp0) > 0 and len(grp1) > 0 else 0.0
    flagged = [c for c, v in scores.items() if v < threshold]
    return flagged, scores


def identify_col_types(dataframe, feature_cols, nunique_threshold=10,
                       force_numeric=None):
    force_numeric = set(force_numeric or [])
    cat = [c for c in feature_cols
           if c not in force_numeric
           and (not pd.api.types.is_numeric_dtype(dataframe[c])
                or dataframe[c].nunique() <= nunique_threshold)]
    num = [c for c in feature_cols if c not in cat]
    return cat, num


def bin_columns(dataframe, num_cols, n_bins, strategy, custom_bins, bin_store,
                fallback="categorical"):
    """Bin numerical columns. Columns that fail binning are handled by fallback.

    fallback: "categorical" moves problem columns to categorical (OHE),
              "drop" removes them entirely.
    Returns: dataframe, bin_store, moved_to_cat, dropped
    """
    moved_to_cat = []
    dropped = []
    successfully_binned = []

    for col in num_cols:
        series = dataframe[col].dropna()

        # Pre-check: non-numeric columns cannot be binned
        if not pd.api.types.is_numeric_dtype(dataframe[col]):
            if fallback == "categorical":
                moved_to_cat.append(col)
                print(f"  WARNING: {col} is not numeric (dtype={dataframe[col].dtype}) — moved to categorical")
            else:
                dropped.append(col)
                print(f"  WARNING: {col} is not numeric (dtype={dataframe[col].dtype}) — dropped")
            continue

        # Pre-check: constant or single-value columns cannot be binned
        if series.nunique() < 2:
            if fallback == "categorical":
                moved_to_cat.append(col)
                print(f"  WARNING: {col} has {series.nunique()} unique value(s) — moved to categorical")
            else:
                dropped.append(col)
                print(f"  WARNING: {col} has {series.nunique()} unique value(s) — dropped")
            continue

        # Pre-check: fewer unique values than requested bins
        if series.nunique() < n_bins and col not in custom_bins:
            if fallback == "categorical":
                moved_to_cat.append(col)
                print(f"  WARNING: {col} has only {series.nunique()} unique values (< {n_bins} bins) — moved to categorical")
            else:
                dropped.append(col)
                print(f"  WARNING: {col} has only {series.nunique()} unique values (< {n_bins} bins) — dropped")
            continue

        try:
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
            successfully_binned.append(col)
            print(f"  Binned: {col}")
        except Exception as e:
            if fallback == "categorical":
                moved_to_cat.append(col)
                print(f"  WARNING: {col} failed binning ({e}) — moved to categorical")
            else:
                dropped.append(col)
                print(f"  WARNING: {col} failed binning ({e}) — dropped")
            # Clean up partial bin column if created
            if col + "_bin" in dataframe.columns:
                dataframe.drop(columns=[col + "_bin"], inplace=True)

    # Drop originals of successfully binned columns
    dataframe.drop(columns=successfully_binned, inplace=True)
    # Drop columns marked for removal
    if dropped:
        dataframe.drop(columns=dropped, inplace=True)
    # Columns moved to categorical stay in the dataframe as-is

    return dataframe, bin_store, moved_to_cat, dropped


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


def train_and_evaluate(df, target_col, test_size, random_state, cv_folds, scoring,
                       param_grid, gain_top_pct, perm_min_threshold,
                       add_back_features, extra_drop_features, label=""):
    """Train XGBoost with GridSearchCV, evaluate, and return results dict.

    Returns dict with keys: best_model, X_train, X_test, y_test, metrics,
    best_params, feature_drop_info, grid_search.
    """
    tag = f" [{label}]" if label else ""

    # ── Train / test split ──────────────────────────────────────────
    print(f"\n  {tag} Stratified train / test split ...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"    Train: {X_train.shape} | Test: {X_test.shape}")

    # ── GridSearchCV ────────────────────────────────────────────────
    print(f"\n  {tag} XGBoost GridSearchCV ...")
    neg, pos = np.bincount(y_train.astype(int))
    scale_pos_weight = neg / pos if pos > 0 else 1
    print(f"    scale_pos_weight = {scale_pos_weight:.2f}")

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight, use_label_encoder=False,
        eval_metric="logloss", random_state=random_state, verbosity=0,
    )
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=xgb_base, param_grid=param_grid, scoring=scoring,
        cv=skf, n_jobs=-1, verbose=1, refit=True,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"    Best {scoring}: {grid_search.best_score_:.4f}")
    print(f"    Best params: {grid_search.best_params_}")

    # ── Feature importance & auto-flag ──────────────────────────────
    print(f"\n  {tag} Feature importance & auto-flag ...")
    imp = pd.Series(best_model.feature_importances_, index=X_train.columns)
    perm_result = permutation_importance(
        best_model, X_test, y_test, n_repeats=10,
        random_state=random_state, scoring="roc_auc", n_jobs=-1,
    )
    perm_imp = pd.Series(perm_result.importances_mean, index=X_test.columns)

    gain_cutoff = np.percentile(imp.values, 100 - gain_top_pct)
    suspicious = imp.index[(imp >= gain_cutoff) & (perm_imp < perm_min_threshold)].tolist()
    print(f"    Suspicious features: {suspicious}")

    auto_drop = [f for f in suspicious if f not in add_back_features]
    final_drop = list(set(auto_drop + extra_drop_features))
    if final_drop:
        X_train = X_train.drop(columns=final_drop, errors="ignore")
        X_test  = X_test.drop(columns=final_drop, errors="ignore")
        print(f"    Dropped {len(final_drop)} features, retraining ...")
        best_model.fit(X_train, y_train)
    print(f"    Final feature count: {X_train.shape[1]}")

    # ── Evaluation metrics ──────────────────────────────────────────
    print(f"\n  {tag} Dev test metrics ...")
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
        print(f"    {name:<20s}  {val:.4f}")

    return {
        "best_model": best_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "metrics": metrics,
        "best_params": grid_search.best_params_,
        "feature_drop_info": {
            "auto_flagged_suspicious": suspicious,
            "added_back": add_back_features,
            "extra_manual_drops": extra_drop_features,
            "final_dropped": final_drop,
        },
        "grid_search": grid_search,
        "xgb_gain": imp,
        "perm_importance": perm_imp,
        "gain_cutoff": gain_cutoff,
        "perm_min_threshold": perm_min_threshold,
    }


def score_new_data(raw_df, artefact_dir, ext_raw_df=None,
                   model_file="xgb_model.joblib",
                   features_file="final_features.json"):
    """Score raw data using saved artefacts. Identical to deployment.

    raw_df        : DataFrame with main columns (from COLUMNS_LIST)
    ext_raw_df    : DataFrame with external columns (from EXT_COLUMNS), same index as raw_df
    model_file    : model joblib filename (default: augmented model)
    features_file : features json filename (default: augmented features)
    """
    model     = joblib.load(os.path.join(artefact_dir, model_file))
    ohe_enc   = joblib.load(os.path.join(artefact_dir, "ohe_encoder.joblib"))
    bin_edges = json.load(open(os.path.join(artefact_dir, "bin_edges.json")))
    col_meta  = json.load(open(os.path.join(artefact_dir, "column_metadata.json")))
    rare_map  = json.load(open(os.path.join(artefact_dir, "rare_mappings.json")))
    features  = json.load(open(os.path.join(artefact_dir, features_file)))

    raw_df = raw_df.copy()

    # Bin numerical columns
    for col, info in bin_edges.items():
        if col in raw_df.columns:
            raw_df[col + "_bin"] = pd.cut(raw_df[col], bins=info["edges"], labels=False, include_lowest=True)
    raw_df.drop(columns=[c for c in col_meta["original_num_cols"] if c in raw_df.columns], inplace=True, errors="ignore")

    # Handle rare categories
    for col, rares in rare_map.items():
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].apply(lambda x: "__rare__" if x in rares else x)

    # Fill nulls & OHE main features
    enc_cols = col_meta["encode_cols"]
    for col in enc_cols:
        if col in raw_df.columns and raw_df[col].isnull().any():
            raw_df[col] = raw_df[col].fillna("__NULL__")
        if col not in raw_df.columns:
            raw_df[col] = "__NULL__"
    raw_df[enc_cols] = raw_df[enc_cols].astype(str)
    encoded = ohe_enc.transform(raw_df[enc_cols])
    enc_df = pd.DataFrame(encoded, columns=ohe_enc.get_feature_names_out(enc_cols), index=raw_df.index)

    # External features
    if col_meta.get("has_external") and ext_raw_df is not None:
        ext_raw_df = ext_raw_df.copy()
        ext_ohe_enc   = joblib.load(os.path.join(artefact_dir, "ext_ohe_encoder.joblib"))
        ext_bin_edges = json.load(open(os.path.join(artefact_dir, "ext_bin_edges.json")))
        ext_rare_map  = json.load(open(os.path.join(artefact_dir, "ext_rare_mappings.json")))

        # Indices already aligned (same CSV, same rows)
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

    # Align to training features and predict
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
    df_full = pd.read_csv(DATA_PATH)
    df_full.rename(columns=TARGET_RENAME, inplace=True)
    print(f"  Shape: {df_full.shape}")
    print(f"  Target distribution:\n{df_full[TARGET_COL].value_counts(normalize=True)}")

    # ── 1a. Holdout split ──────────────────────────────────────────
    print(f"\n[1a] Holdout split ({HOLDOUT_PCT*100:.0f}%) ...")
    df_dev, df_holdout = train_test_split(
        df_full, test_size=HOLDOUT_PCT, random_state=HOLDOUT_SEED, stratify=df_full[TARGET_COL]
    )
    holdout_raw = df_holdout.copy()
    df_dev = df_dev.reset_index(drop=True)
    print(f"  Dev: {len(df_dev)} rows | Holdout: {len(holdout_raw)} rows")

    # Save full dev data before column selection (needed for external columns)
    df_dev_full = df_dev.copy()

    # ── 2. Column selection (main columns) ─────────────────────────
    print("\n[2] Column selection (main columns) ...")
    df = df_dev.copy()
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
    cat_cols, num_cols = identify_col_types(df, feature_cols, CAT_NUNIQUE_THRESHOLD, FORCE_NUMERIC)
    original_cat_cols = list(cat_cols)  # snapshot before binning modifies these
    original_num_cols = list(num_cols)
    print(f"  Categorical: {len(cat_cols)} | Numerical: {len(num_cols)}")
    if FORCE_NUMERIC:
        forced = [c for c in FORCE_NUMERIC if c in num_cols]
        if forced:
            print(f"  Forced numeric (override): {forced}")

    # ── 6. Binning ─────────────────────────────────────────────────
    print("\n[6] Binning numerical columns ...")
    bin_edges_store = {}
    df, bin_edges_store, binning_moved_to_cat, binning_dropped = bin_columns(
        df, num_cols, N_BINS, BINNING_STRATEGY, CUSTOM_BINS, bin_edges_store,
        fallback=BINNING_FALLBACK,
    )
    if binning_moved_to_cat:
        cat_cols.extend(binning_moved_to_cat)
        num_cols = [c for c in num_cols if c not in binning_moved_to_cat]
        print(f"  Moved to categorical: {binning_moved_to_cat}")
    if binning_dropped:
        num_cols = [c for c in num_cols if c not in binning_dropped]
        print(f"  Dropped (binning fallback): {binning_dropped}")

    # ── 7. OHE ─────────────────────────────────────────────────────
    print("\n[7] One-hot encoding ...")
    rare_mappings = {}
    df, encode_cols, ohe, rare_mappings = ohe_columns(df, TARGET_COL, DROP_FIRST, MAX_CATEGORIES, rare_mappings)
    main_feature_cols = [c for c in df.columns if c != TARGET_COL]
    print(f"  Shape after OHE: {df.shape}")

    # Save main-only dataframe before external augmentation
    df_main_only = df.copy()

    # ── 8. External columns pipeline ───────────────────────────────
    ext_bin_edges_store = {}
    ext_rare_mappings = {}
    ext_ohe = None
    ext_encode_cols = []
    ext_cat_cols = []
    ext_num_cols = []
    ext_df = None
    ext_null_drops = []
    ext_flagged = []
    ext_binning_moved = []
    ext_binning_dropped = []
    ext_original_cat_cols = []
    ext_original_num_cols = []

    if EXT_COLUMNS and len(EXT_COLUMNS) > 0:
        print("\n[8] External columns pipeline ...")

        # Extract external columns from the same dev data
        available_ext = [c for c in EXT_COLUMNS if c in df_dev_full.columns]
        missing_ext = [c for c in EXT_COLUMNS if c not in df_dev_full.columns]
        if missing_ext:
            print(f"  WARNING: {len(missing_ext)} external columns not found: {missing_ext}")

        ext_df = df_dev_full[available_ext + [TARGET_COL]].copy()
        ext_df.index = df.index  # align with main df
        print(f"  External columns selected: {len(available_ext)}")

        # Null analysis
        ext_feature_cols = [c for c in ext_df.columns if c != TARGET_COL]
        ext_null_drops = null_analysis(ext_df, ext_feature_cols, EXT_NULL_THRESHOLD_PCT, TARGET_COL)
        ext_df.drop(columns=ext_null_drops, inplace=True)
        print(f"  Ext null drops: {len(ext_null_drops)} columns: {ext_null_drops}")

        # Variation
        ext_feature_cols = [c for c in ext_df.columns if c != TARGET_COL]
        ext_flagged, _ = variation_analysis(ext_df, ext_feature_cols, TARGET_COL, EXT_VARIATION_THRESHOLD)
        print(f"  Ext flagged (low variation): {ext_flagged}")
        ext_df.drop(columns=[c for c in EXT_DROP_LOW_VARIATION if c in ext_df.columns], inplace=True)

        # Types, binning, OHE
        ext_feature_cols = [c for c in ext_df.columns if c != TARGET_COL]
        if len(ext_feature_cols) > 0:
            ext_cat_cols, ext_num_cols = identify_col_types(
                ext_df, ext_feature_cols, EXT_CAT_NUNIQUE_THRESHOLD, EXT_FORCE_NUMERIC)
            ext_original_cat_cols = list(ext_cat_cols)
            ext_original_num_cols = list(ext_num_cols)
            print(f"  Ext categorical: {len(ext_cat_cols)} | Ext numerical: {len(ext_num_cols)}")
            if EXT_FORCE_NUMERIC:
                forced_ext = [c for c in EXT_FORCE_NUMERIC if c in ext_num_cols]
                if forced_ext:
                    print(f"  Ext forced numeric (override): {forced_ext}")
            if ext_num_cols:
                ext_df, ext_bin_edges_store, ext_binning_moved, ext_binning_dropped = bin_columns(
                    ext_df, ext_num_cols, EXT_N_BINS, EXT_BINNING_STRATEGY, EXT_CUSTOM_BINS, ext_bin_edges_store,
                    fallback=EXT_BINNING_FALLBACK,
                )
                if ext_binning_moved:
                    ext_cat_cols.extend(ext_binning_moved)
                    ext_num_cols = [c for c in ext_num_cols if c not in ext_binning_moved]
                    print(f"  Ext moved to categorical: {ext_binning_moved}")
                if ext_binning_dropped:
                    ext_num_cols = [c for c in ext_num_cols if c not in ext_binning_dropped]
                    print(f"  Ext dropped (binning fallback): {ext_binning_dropped}")
            ext_df, ext_encode_cols, ext_ohe, ext_rare_mappings = ohe_columns(
                ext_df, TARGET_COL, EXT_DROP_FIRST, EXT_MAX_CATEGORIES, ext_rare_mappings
            )
            # Concatenate with main
            ext_only = ext_df.drop(columns=[TARGET_COL])
            df = pd.concat([df.drop(columns=[TARGET_COL]), ext_only, df[[TARGET_COL]]], axis=1)
            print(f"  Combined shape: {df.shape}")
        else:
            ext_df = None
            print("  No external columns survived preprocessing.")
    else:
        print("\n[8] No external columns — skipping.")

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
        "binning_moved_to_cat": binning_moved_to_cat,
        "binning_dropped": binning_dropped,
        "ext_cat_cols": ext_cat_cols, "ext_num_cols": ext_num_cols,
        "ext_encode_cols": ext_encode_cols,
        "ext_columns": EXT_COLUMNS if EXT_COLUMNS else [],
        "final_feature_cols": [c for c in df.columns if c != TARGET_COL],
        "target_col": TARGET_COL, "has_external": ext_df is not None,
    }
    with open(os.path.join(ARTEFACT_DIR, "column_metadata.json"), "w") as f:
        json.dump(col_meta, f, indent=2)
    with open(os.path.join(ARTEFACT_DIR, "holdout_meta.json"), "w") as f:
        json.dump({"holdout_pct": HOLDOUT_PCT, "holdout_seed": HOLDOUT_SEED,
                    "holdout_rows": len(holdout_raw), "dev_rows": len(df)}, f, indent=2)

    # ══════════════════════════════════════════════════════════════
    #  PHASE A — MAIN DATA ONLY (before external augmentation)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE A: Training on MAIN DATA ONLY")
    print("=" * 60)

    main_result = train_and_evaluate(
        df_main_only, TARGET_COL, TEST_SIZE, RANDOM_STATE, CV_FOLDS, SCORING,
        PARAM_GRID, GAIN_TOP_PCT, PERM_MIN_THRESHOLD,
        ADD_BACK_FEATURES, EXTRA_DROP_FEATURES, label="MAIN ONLY",
    )

    # Save main-only artefacts
    print("\n  Saving MAIN-ONLY model & metrics ...")
    joblib.dump(main_result["best_model"], os.path.join(ARTEFACT_DIR, "xgb_model_main_only.joblib"))
    with open(os.path.join(ARTEFACT_DIR, "final_features_main_only.json"), "w") as f:
        json.dump(main_result["X_train"].columns.tolist(), f, indent=2)
    with open(os.path.join(ARTEFACT_DIR, "test_metrics_main_only.json"), "w") as f:
        json.dump(main_result["metrics"], f, indent=2)
    with open(os.path.join(ARTEFACT_DIR, "best_params_main_only.json"), "w") as f:
        json.dump(main_result["best_params"], f, indent=2)
    with open(os.path.join(ARTEFACT_DIR, "dropped_features_main_only.json"), "w") as f:
        json.dump(main_result["feature_drop_info"], f, indent=2)

    # ══════════════════════════════════════════════════════════════
    #  PHASE B — AUGMENTED (main + external data)
    # ══════════════════════════════════════════════════════════════
    if ext_df is not None:
        print("\n" + "=" * 60)
        print("  PHASE B: Training on MAIN + EXTERNAL DATA (augmented)")
        print("=" * 60)

        # df already has main + external concatenated from step 8
        aug_result = train_and_evaluate(
            df, TARGET_COL, TEST_SIZE, RANDOM_STATE, CV_FOLDS, SCORING,
            PARAM_GRID, GAIN_TOP_PCT, PERM_MIN_THRESHOLD,
            ADD_BACK_FEATURES, EXTRA_DROP_FEATURES, label="AUGMENTED",
        )

        # Save augmented artefacts
        print("\n  Saving AUGMENTED model & metrics ...")
        joblib.dump(aug_result["best_model"], os.path.join(ARTEFACT_DIR, "xgb_model.joblib"))
        with open(os.path.join(ARTEFACT_DIR, "final_features.json"), "w") as f:
            json.dump(aug_result["X_train"].columns.tolist(), f, indent=2)
        with open(os.path.join(ARTEFACT_DIR, "test_metrics_augmented.json"), "w") as f:
            json.dump(aug_result["metrics"], f, indent=2)
        with open(os.path.join(ARTEFACT_DIR, "test_metrics.json"), "w") as f:
            json.dump(aug_result["metrics"], f, indent=2)
        with open(os.path.join(ARTEFACT_DIR, "best_params_augmented.json"), "w") as f:
            json.dump(aug_result["best_params"], f, indent=2)
        with open(os.path.join(ARTEFACT_DIR, "best_params.json"), "w") as f:
            json.dump(aug_result["best_params"], f, indent=2)
        with open(os.path.join(ARTEFACT_DIR, "dropped_features_augmented.json"), "w") as f:
            json.dump(aug_result["feature_drop_info"], f, indent=2)
        with open(os.path.join(ARTEFACT_DIR, "dropped_features.json"), "w") as f:
            json.dump(aug_result["feature_drop_info"], f, indent=2)

        # Use augmented model for holdout & final artefacts
        best_model = aug_result["best_model"]
        metrics = aug_result["metrics"]
        X_train = aug_result["X_train"]
    else:
        print("\n  No external data — skipping PHASE B.")
        # Use main-only results as the final model
        best_model = main_result["best_model"]
        metrics = main_result["metrics"]
        X_train = main_result["X_train"]

        # Save as the primary artefacts too
        joblib.dump(best_model, os.path.join(ARTEFACT_DIR, "xgb_model.joblib"))
        with open(os.path.join(ARTEFACT_DIR, "final_features.json"), "w") as f:
            json.dump(X_train.columns.tolist(), f, indent=2)
        with open(os.path.join(ARTEFACT_DIR, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(ARTEFACT_DIR, "best_params.json"), "w") as f:
            json.dump(main_result["best_params"], f, indent=2)
        with open(os.path.join(ARTEFACT_DIR, "dropped_features.json"), "w") as f:
            json.dump(main_result["feature_drop_info"], f, indent=2)

    # ── Holdout evaluation ──────────────────────────────────────────
    print("\n[Holdout] Out-of-sample evaluation ...")

    # Main columns from holdout
    holdout_df = holdout_raw.copy()
    if MODE == "keep":
        keep_cols = list(set(COLUMNS_LIST + [TARGET_COL]))
        holdout_df = holdout_df[[c for c in keep_cols if c in holdout_df.columns]]
    elif MODE == "drop":
        holdout_df.drop(columns=[c for c in COLUMNS_LIST if c in holdout_df.columns], inplace=True)

    y_holdout = holdout_df[TARGET_COL]
    holdout_main = holdout_df.drop(columns=[TARGET_COL])

    # External columns from holdout (same CSV, so they're already there)
    holdout_ext = None
    if EXT_COLUMNS and ext_df is not None:
        available_ext_h = [c for c in EXT_COLUMNS if c in holdout_raw.columns]
        holdout_ext = holdout_raw[available_ext_h].reset_index(drop=True)

    # --- Holdout scored with MAIN-ONLY model ---
    h_prob_main = score_new_data(
        holdout_main, ARTEFACT_DIR, ext_raw_df=None,
        model_file="xgb_model_main_only.joblib",
        features_file="final_features_main_only.json",
    )
    h_pred_main = (h_prob_main >= 0.5).astype(int)

    holdout_main_metrics = {
        "Accuracy":          accuracy_score(y_holdout, h_pred_main),
        "Balanced Accuracy": balanced_accuracy_score(y_holdout, h_pred_main),
        "F1 Score":          f1_score(y_holdout, h_pred_main),
        "AUC (ROC)":         roc_auc_score(y_holdout, h_prob_main),
        "Gini":              2 * roc_auc_score(y_holdout, h_prob_main) - 1,
    }

    # --- Holdout scored with AUGMENTED model ---
    h_prob_aug = score_new_data(holdout_main, ARTEFACT_DIR, ext_raw_df=holdout_ext)
    h_pred_aug = (h_prob_aug >= 0.5).astype(int)

    holdout_aug_metrics = {
        "Accuracy":          accuracy_score(y_holdout, h_pred_aug),
        "Balanced Accuracy": balanced_accuracy_score(y_holdout, h_pred_aug),
        "F1 Score":          f1_score(y_holdout, h_pred_aug),
        "AUC (ROC)":         roc_auc_score(y_holdout, h_prob_aug),
        "Gini":              2 * roc_auc_score(y_holdout, h_prob_aug) - 1,
    }

    # ── Comparison table ────────────────────────────────────────────
    print("\n  " + "=" * 100)
    print("  DEV-TEST (Main vs Aug)  &  HOLDOUT (Main vs Aug)")
    print("  " + "=" * 100)
    main_metrics = main_result["metrics"]
    aug_metrics = metrics  # augmented if available, else same as main
    print(f"  {'Metric':<20s}  {'Dev_Main':>10s}  {'Dev_Aug':>10s}  {'Hold_Main':>10s}  {'Hold_Aug':>10s}")
    print("  " + "-" * 100)
    for name in main_metrics:
        main_val     = main_metrics[name]
        aug_val      = aug_metrics[name]
        hold_main_v  = holdout_main_metrics[name]
        hold_aug_v   = holdout_aug_metrics[name]
        print(f"  {name:<20s}  {main_val:>10.4f}  {aug_val:>10.4f}  {hold_main_v:>10.4f}  {hold_aug_v:>10.4f}")
    print("  " + "=" * 100)

    with open(os.path.join(ARTEFACT_DIR, "holdout_metrics_main_only.json"), "w") as f:
        json.dump(holdout_main_metrics, f, indent=2)
    with open(os.path.join(ARTEFACT_DIR, "holdout_metrics.json"), "w") as f:
        json.dump(holdout_aug_metrics, f, indent=2)

    # ── KS Statistic ──────────────────────────────────────────────
    print("\n[KS] Computing KS statistic ...")

    def compute_ks(y_true, y_prob):
        """KS statistic: max separation between positive and negative CDFs."""
        pos = y_prob[y_true == 1]
        neg = y_prob[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.0
        return ks_2samp(pos, neg)[0]

    ks_results = {}

    # Dev test KS — main-only
    main_y_pred_prob = main_result["best_model"].predict_proba(
        main_result["X_test"])[:, 1]
    ks_dev_main = compute_ks(main_result["y_test"].values, main_y_pred_prob)
    ks_results["Dev_Main_KS"] = round(ks_dev_main, 6)
    print(f"  Dev Main-Only  KS = {ks_dev_main:.4f}")

    # Dev test KS — augmented
    if ext_df is not None:
        aug_y_pred_prob = aug_result["best_model"].predict_proba(
            aug_result["X_test"])[:, 1]
        ks_dev_aug = compute_ks(aug_result["y_test"].values, aug_y_pred_prob)
        ks_results["Dev_Aug_KS"] = round(ks_dev_aug, 6)
        print(f"  Dev Augmented  KS = {ks_dev_aug:.4f}")
    else:
        ks_results["Dev_Aug_KS"] = ks_results["Dev_Main_KS"]
        print(f"  Dev Augmented  KS = {ks_results['Dev_Main_KS']:.4f} (same as main)")

    # Holdout KS
    ks_hold_main = compute_ks(y_holdout.values, h_prob_main)
    ks_results["Hold_Main_KS"] = round(ks_hold_main, 6)
    print(f"  Holdout Main   KS = {ks_hold_main:.4f}")

    ks_hold_aug = compute_ks(y_holdout.values, h_prob_aug)
    ks_results["Hold_Aug_KS"] = round(ks_hold_aug, 6)
    print(f"  Holdout Aug    KS = {ks_hold_aug:.4f}")

    with open(os.path.join(ARTEFACT_DIR, "ks_statistics.json"), "w") as f:
        json.dump(ks_results, f, indent=2)

    # Add KS to the comparison table
    print(f"\n  {'KS Statistic':<20s}  {ks_results['Dev_Main_KS']:>10.4f}  "
          f"{ks_results['Dev_Aug_KS']:>10.4f}  "
          f"{ks_results['Hold_Main_KS']:>10.4f}  "
          f"{ks_results['Hold_Aug_KS']:>10.4f}")

    # ── Save comprehensive metrics summary ─────────────────────────
    print("\n[Summary] Saving comprehensive metrics & column tracking ...")

    all_metrics_summary = {
        "dev_main": main_result["metrics"],
        "holdout_main": holdout_main_metrics,
        "ks_dev_main": ks_results["Dev_Main_KS"],
        "ks_holdout_main": ks_results["Hold_Main_KS"],
    }
    if ext_df is not None:
        all_metrics_summary["dev_augmented"] = aug_result["metrics"]
        all_metrics_summary["holdout_augmented"] = holdout_aug_metrics
        all_metrics_summary["ks_dev_augmented"] = ks_results["Dev_Aug_KS"]
        all_metrics_summary["ks_holdout_augmented"] = ks_results["Hold_Aug_KS"]

    with open(os.path.join(ARTEFACT_DIR, "all_metrics_summary.json"), "w") as f:
        json.dump(all_metrics_summary, f, indent=2)

    # ── Save column tracking ───────────────────────────────────────
    column_tracking = {
        "initial_columns": COLUMNS_LIST,
        "dropped_by_null_analysis": null_drops,
        "flagged_low_variation": flagged,
        "dropped_by_variation": DROP_LOW_VARIATION,
        "categorical_columns": cat_cols,
        "numerical_columns_binned": list(bin_edges_store.keys()),
        "binning_moved_to_categorical": binning_moved_to_cat,
        "binning_dropped": binning_dropped,
        "columns_after_ohe": main_feature_cols,
        "main_model_features_used": main_result["X_train"].columns.tolist(),
        "main_model_features_dropped": main_result["feature_drop_info"]["final_dropped"],
    }
    if ext_df is not None:
        column_tracking["external_columns_input"] = EXT_COLUMNS
        column_tracking["external_dropped_by_null"] = [c for c in EXT_COLUMNS
            if c not in [col for col in ext_df.columns if col != TARGET_COL]]
        column_tracking["external_columns_binned"] = list(ext_bin_edges_store.keys())
        column_tracking["augmented_model_features_used"] = aug_result["X_train"].columns.tolist()
        column_tracking["augmented_model_features_dropped"] = aug_result["feature_drop_info"]["final_dropped"]

    with open(os.path.join(ARTEFACT_DIR, "column_tracking.json"), "w") as f:
        json.dump(column_tracking, f, indent=2)

    # ── Predictor report (CSV for sharing) ────────────────────────
    def _build_report(result, label, encode_columns):
        """Build a per-OHE-feature report with importance, status & drop reason."""
        used_feats   = result["X_train"].columns.tolist()
        drop_info    = result["feature_drop_info"]
        gain         = result["xgb_gain"]          # Series — all features (pre-drop)
        perm         = result["perm_importance"]    # Series — all features (pre-drop)
        gain_cut     = result["gain_cutoff"]
        perm_thresh  = result["perm_min_threshold"]
        suspicious   = set(drop_info["auto_flagged_suspicious"])
        added_back   = set(drop_info["added_back"])
        extra_manual = set(drop_info["extra_manual_drops"])
        final_dropped = set(drop_info["final_dropped"])

        # Sort encode columns by length descending so longer prefixes match first
        # e.g., 'social_facebook_page' matches before 'social_facebook'
        sorted_encode_cols = sorted(encode_columns, key=len, reverse=True)

        all_feats = sorted(set(gain.index.tolist()) | set(perm.index.tolist()))
        rows = []
        for feat in all_feats:
            g = gain.get(feat, np.nan)
            p = perm.get(feat, np.nan)
            status = "USED" if feat in used_feats else "DROPPED"

            # Determine drop reason
            reason = ""
            if feat in final_dropped:
                if feat in extra_manual:
                    reason = "Manual drop (EXTRA_DROP_FEATURES)"
                elif feat in suspicious and feat not in added_back:
                    reason = (f"Auto-flagged: high XGBoost gain (>= {gain_cut:.6f}) "
                              f"but perm importance < {perm_thresh}")
                else:
                    reason = "Auto-flagged suspicious"

            # Map back to original column name (longest prefix match)
            base_col = feat
            for col in sorted_encode_cols:
                if feat == col or feat.startswith(col + "_"):
                    base_col = col
                    break

            rows.append({
                "model": label,
                "ohe_feature": feat,
                "original_column": base_col,
                "status": status,
                "xgb_gain": round(g, 6) if not np.isnan(g) else np.nan,
                "perm_importance_auc_drop": round(p, 6) if not np.isnan(p) else np.nan,
                "drop_reason": reason,
            })

        rdf = pd.DataFrame(rows)
        # Compute ranks as new columns (avoids .loc dtype issues with string-backed frames)
        rdf["xgb_gain_rank"] = rdf["xgb_gain"].rank(ascending=False, na_option="keep")
        rdf["perm_importance_rank"] = rdf["perm_importance_auc_drop"].rank(ascending=False, na_option="keep")
        return rdf

    # Use only the relevant encode columns for each model's report
    main_encode_list = list(encode_cols)
    ext_encode_list  = list(ext_encode_cols) if ext_df is not None else []
    all_encode_list  = main_encode_list + ext_encode_list

    report_main = _build_report(main_result, "MAIN_ONLY", main_encode_list)
    report_aug  = _build_report(aug_result, "AUGMENTED", all_encode_list) if ext_df is not None else None

    if report_aug is not None:
        report_full = pd.concat([report_main, report_aug], ignore_index=True)
    else:
        report_full = report_main

    report_path = os.path.join(ARTEFACT_DIR, "predictor_report.csv")
    report_full.to_csv(report_path, index=False)

    # ── Column Trail Report (comprehensive CSV) ───────────────────
    print("\n  Building column trail report ...")

    def _ohe_features_by_col(encoder, enc_cols):
        """Map each encode-column to its list of OHE feature names."""
        if encoder is None or not enc_cols:
            return {}
        all_feats = encoder.get_feature_names_out(enc_cols)
        sorted_enc = sorted(enc_cols, key=len, reverse=True)
        mapping = {}
        for feat in all_feats:
            for col in sorted_enc:
                if feat == col or feat.startswith(col + "_"):
                    mapping.setdefault(col, []).append(feat)
                    break
        return mapping

    def _process_columns_for_trail(
        source_label, initial_cols, target_col,
        null_drops_list, flagged_var_list, dropped_var_list,
        orig_cat_list, orig_num_list,
        bin_store, binning_moved_list, binning_dropped_list,
        enc_cols, encoder,
        main_result, aug_result, include_in_main,
    ):
        """Build trail rows for one source (Main or InsightGenie)."""
        rows = []
        null_set = set(null_drops_list)
        flagged_set = set(flagged_var_list)
        dropped_var_set = set(dropped_var_list)
        cat_set = set(orig_cat_list)
        num_set = set(orig_num_list)
        binned_set = set(bin_store.keys())
        moved_set = set(binning_moved_list)
        bin_drop_set = set(binning_dropped_list)

        main_used = set(main_result["X_train"].columns.tolist())
        main_sus = set(main_result["feature_drop_info"]["auto_flagged_suspicious"])
        main_drop = set(main_result["feature_drop_info"]["final_dropped"])

        aug_used, aug_sus, aug_drop = set(), set(), set()
        if aug_result is not None:
            aug_used = set(aug_result["X_train"].columns.tolist())
            aug_sus = set(aug_result["feature_drop_info"]["auto_flagged_suspicious"])
            aug_drop = set(aug_result["feature_drop_info"]["final_dropped"])

        ohe_map = _ohe_features_by_col(encoder, enc_cols)

        cols = [c for c in initial_cols if c != target_col]
        for col in cols:
            is_null = col in null_set
            is_flag_var = col in flagged_set
            is_drop_var = col in dropped_var_set

            # --- Dropped before type classification ---
            if is_null or is_drop_var:
                rows.append({
                    "source": source_label,
                    "original_column": col,
                    "column_type": "—",
                    "dropped_null_analysis": is_null,
                    "flagged_low_variation": is_flag_var,
                    "dropped_low_variation": is_drop_var,
                    "binning_result": "—",
                    "ohe_feature": "—",
                    "flagged_suspicious_main": False,
                    "dropped_suspicious_main": False,
                    "finally_used_main": False,
                    "flagged_suspicious_aug": False,
                    "dropped_suspicious_aug": False,
                    "finally_used_aug": False,
                })
                continue

            # --- Column type (from pre-binning snapshot) ---
            col_type = "categorical" if col in cat_set else (
                "numerical" if col in num_set else "unknown")

            # --- Binning result ---
            if col in num_set:
                if col in binned_set:
                    bin_result = "binned"
                    ohe_key = col + "_bin"
                elif col in moved_set:
                    bin_result = "moved_to_categorical"
                    ohe_key = col
                elif col in bin_drop_set:
                    bin_result = "dropped"
                    rows.append({
                        "source": source_label,
                        "original_column": col,
                        "column_type": col_type,
                        "dropped_null_analysis": False,
                        "flagged_low_variation": is_flag_var,
                        "dropped_low_variation": False,
                        "binning_result": "dropped",
                        "ohe_feature": "—",
                        "flagged_suspicious_main": False,
                        "dropped_suspicious_main": False,
                        "finally_used_main": False,
                        "flagged_suspicious_aug": False,
                        "dropped_suspicious_aug": False,
                        "finally_used_aug": False,
                    })
                    continue
                else:
                    bin_result = "binned"
                    ohe_key = col + "_bin"
            else:
                bin_result = "N/A (categorical)"
                ohe_key = col

            # --- OHE features ---
            ohe_feats = ohe_map.get(ohe_key, [])

            if not ohe_feats:
                rows.append({
                    "source": source_label,
                    "original_column": col,
                    "column_type": col_type,
                    "dropped_null_analysis": False,
                    "flagged_low_variation": is_flag_var,
                    "dropped_low_variation": False,
                    "binning_result": bin_result,
                    "ohe_feature": "—",
                    "flagged_suspicious_main": False,
                    "dropped_suspicious_main": False,
                    "finally_used_main": False,
                    "flagged_suspicious_aug": False,
                    "dropped_suspicious_aug": False,
                    "finally_used_aug": False,
                })
            else:
                for feat in ohe_feats:
                    rows.append({
                        "source": source_label,
                        "original_column": col,
                        "column_type": col_type,
                        "dropped_null_analysis": False,
                        "flagged_low_variation": is_flag_var,
                        "dropped_low_variation": False,
                        "binning_result": bin_result,
                        "ohe_feature": feat,
                        "flagged_suspicious_main": (
                            feat in main_sus if include_in_main else False),
                        "dropped_suspicious_main": (
                            feat in main_drop if include_in_main else False),
                        "finally_used_main": (
                            feat in main_used if include_in_main else False),
                        "flagged_suspicious_aug": feat in aug_sus,
                        "dropped_suspicious_aug": feat in aug_drop,
                        "finally_used_aug": feat in aug_used,
                    })
        return rows

    trail_rows = _process_columns_for_trail(
        "Main", COLUMNS_LIST, TARGET_COL,
        null_drops, flagged, DROP_LOW_VARIATION,
        original_cat_cols, original_num_cols,
        bin_edges_store, binning_moved_to_cat, binning_dropped,
        encode_cols, ohe,
        main_result, aug_result if ext_df is not None else None,
        include_in_main=True,
    )

    if EXT_COLUMNS and ext_df is not None:
        trail_rows += _process_columns_for_trail(
            "InsightGenie", EXT_COLUMNS, TARGET_COL,
            ext_null_drops, ext_flagged, EXT_DROP_LOW_VARIATION,
            ext_original_cat_cols, ext_original_num_cols,
            ext_bin_edges_store, ext_binning_moved, ext_binning_dropped,
            ext_encode_cols, ext_ohe,
            main_result, aug_result,
            include_in_main=False,
        )

    trail_df = pd.DataFrame(trail_rows)

    # ── Append summary trail (running counts at each step) ─────
    def _summary_for_source(label, tdf):
        """Build summary rows showing column counts at each pipeline step."""
        src = tdf[tdf["source"] == label]
        if len(src) == 0:
            return []
        # Count unique original columns at each stage
        total = src["original_column"].nunique()
        null_dropped = src[src["dropped_null_analysis"]]["original_column"].nunique()
        after_null = total - null_dropped
        var_dropped = src[src["dropped_low_variation"]]["original_column"].nunique()
        after_var = after_null - var_dropped
        bin_dropped = src[src["binning_result"] == "dropped"]["original_column"].nunique()
        after_bin = after_var - bin_dropped
        # OHE features produced
        ohe_produced = len(src[(src["ohe_feature"] != "—")])
        # Post-training (main model)
        used_main = len(src[src["finally_used_main"]])
        dropped_main = len(src[src["dropped_suspicious_main"]])
        # Post-training (augmented model)
        used_aug = len(src[src["finally_used_aug"]])
        dropped_aug = len(src[src["dropped_suspicious_aug"]])

        summary = [
            {"step": "1. Initial columns", "columns_dropped": "—",
             "columns_remaining": str(total), "source": label},
            {"step": "2. Null analysis (>40% null)", "columns_dropped": str(null_dropped),
             "columns_remaining": str(after_null), "source": label},
            {"step": "3. Low-variation drop (manual)", "columns_dropped": str(var_dropped),
             "columns_remaining": str(after_var), "source": label},
            {"step": "4. Binning failures dropped", "columns_dropped": str(bin_dropped),
             "columns_remaining": str(after_bin), "source": label},
            {"step": "5. After OHE (feature count)", "columns_dropped": "—",
             "columns_remaining": str(ohe_produced), "source": label},
            {"step": "6. Suspicious dropped (Main)", "columns_dropped": str(dropped_main),
             "columns_remaining": str(used_main), "source": label},
            {"step": "7. Suspicious dropped (Aug)", "columns_dropped": str(dropped_aug),
             "columns_remaining": str(used_aug), "source": label},
        ]
        return summary

    summary_rows = _summary_for_source("Main", trail_df)
    if EXT_COLUMNS and ext_df is not None:
        summary_rows += _summary_for_source("InsightGenie", trail_df)

    summary_df = pd.DataFrame(summary_rows)

    trail_path = os.path.join(ARTEFACT_DIR, "column_trail_report.csv")
    summary_path = os.path.join(ARTEFACT_DIR, "column_trail_summary.csv")
    trail_df.to_csv(trail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"  Column trail report: {trail_path}")
    print(f"  Column trail summary: {summary_path}")

    # ── Console summary ───────────────────────────────────────────
    main_used = report_main[report_main["status"] == "USED"]
    main_drop = report_main[report_main["status"] == "DROPPED"]
    print("\n  " + "=" * 100)
    print("  PREDICTOR REPORT  (saved to predictor_report.csv)")
    print("  " + "=" * 100)
    print(f"\n  MAIN-ONLY model : {len(main_used)} used, {len(main_drop)} dropped")
    print(f"    Original columns used   : {sorted(main_used['original_column'].unique())}")
    print(f"    Original columns dropped : {sorted(main_drop['original_column'].unique())}")
    if len(main_drop) > 0:
        print(f"    Drop reasons:")
        for _, r in main_drop.iterrows():
            print(f"      - {r['ohe_feature']:<45s}  {r['drop_reason']}")

    if report_aug is not None:
        aug_used = report_aug[report_aug["status"] == "USED"]
        aug_drop = report_aug[report_aug["status"] == "DROPPED"]
        print(f"\n  AUGMENTED model : {len(aug_used)} used, {len(aug_drop)} dropped")
        print(f"    Original columns used   : {sorted(aug_used['original_column'].unique())}")
        print(f"    Original columns dropped : {sorted(aug_drop['original_column'].unique())}")
        if len(aug_drop) > 0:
            print(f"    Drop reasons:")
            for _, r in aug_drop.iterrows():
                print(f"      - {r['ohe_feature']:<45s}  {r['drop_reason']}")

    print("\n  " + "=" * 100)
    print(f"  Full report: {report_path}")
    print(f"  Columns: model | ohe_feature | original_column | status |")
    print(f"           xgb_gain | xgb_gain_rank | perm_importance_auc_drop |")
    print(f"           perm_importance_rank | drop_reason")
    print("  " + "=" * 100)

    print(f"\nAll artefacts saved to: {ARTEFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
