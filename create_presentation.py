"""
Generate a PowerPoint presentation summarising the classification pipeline,
metrics, feature importance, and augmented data analysis.

Usage:
    python create_presentation.py

Reads artefact JSON files from artefacts/ (if they exist) and embeds
the actual numbers.  If artefacts are not yet available it uses placeholder
text so the deck structure is ready for Monday.
"""

import json
import os
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ARTEFACT_DIR = "artefacts"
OUTPUT_PATH = "Classification_Model_Presentation.pptx"

# ── Colour palette ──────────────────────────────────────────────────
DARK_BLUE   = RGBColor(0x1B, 0x2A, 0x4A)
MED_BLUE    = RGBColor(0x2C, 0x5F, 0x8A)
LIGHT_BLUE  = RGBColor(0x4A, 0x90, 0xD9)
ACCENT_ORANGE = RGBColor(0xE8, 0x7C, 0x2A)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY  = RGBColor(0xF2, 0xF2, 0xF2)
DARK_GREY   = RGBColor(0x33, 0x33, 0x33)
GREEN       = RGBColor(0x27, 0xAE, 0x60)
RED         = RGBColor(0xC0, 0x39, 0x2B)


def _load_json(filename, default=None):
    path = os.path.join(ARTEFACT_DIR, filename)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return default


def _add_bg(slide, colour=DARK_BLUE):
    """Fill slide background with a solid colour."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = colour


def _add_shape_bg(slide, colour=DARK_BLUE):
    """Add a full-slide rectangle as background (works more reliably)."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(0),
        Inches(13.333), Inches(7.5),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = colour
    shape.line.fill.background()
    # Send to back
    sp = shape._element
    sp.getparent().remove(sp)
    slide.shapes._spTree.insert(2, sp)


def _title_slide(prs, title, subtitle=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    _add_shape_bg(slide, DARK_BLUE)

    # Title
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.5), Inches(1.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT

    # Subtitle
    if subtitle:
        txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(3.8), Inches(11.5), Inches(1))
        tf2 = txBox2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = subtitle
        p2.font.size = Pt(20)
        p2.font.color.rgb = LIGHT_BLUE
        p2.alignment = PP_ALIGN.LEFT

    # Accent line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.6), Inches(2), Inches(0.05))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT_ORANGE
    line.line.fill.background()

    return slide


def _section_slide(prs, title, subtitle=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_shape_bg(slide, MED_BLUE)

    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(2.8), Inches(11.5), Inches(1.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT

    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(18)
        p2.font.color.rgb = LIGHT_GREY
        p2.alignment = PP_ALIGN.LEFT
        p2.space_before = Pt(12)

    # Accent line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.6), Inches(1.5), Inches(0.05))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT_ORANGE
    line.line.fill.background()

    return slide


def _content_slide(prs, title, bullets, two_col=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_shape_bg(slide, WHITE)

    # Top bar
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.9))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()

    # Title in bar
    txBox = slide.shapes.add_textbox(Inches(0.6), Inches(0.15), Inches(12), Inches(0.6))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = WHITE

    if two_col and len(bullets) >= 2:
        mid = len(bullets) // 2
        left_bullets = bullets[:mid]
        right_bullets = bullets[mid:]

        for col_idx, col_bullets in enumerate([(0.6, left_bullets), (6.8, right_bullets)]):
            x, items = col_bullets
            tb = slide.shapes.add_textbox(Inches(x), Inches(1.2), Inches(5.8), Inches(5.8))
            tf2 = tb.text_frame
            tf2.word_wrap = True
            for i, bullet in enumerate(items):
                if i == 0:
                    p2 = tf2.paragraphs[0]
                else:
                    p2 = tf2.add_paragraph()
                p2.text = bullet
                p2.font.size = Pt(14)
                p2.font.color.rgb = DARK_GREY
                p2.space_before = Pt(6)
                p2.space_after = Pt(4)
                p2.level = 0
    else:
        tb = slide.shapes.add_textbox(Inches(0.6), Inches(1.2), Inches(12), Inches(5.8))
        tf2 = tb.text_frame
        tf2.word_wrap = True
        for i, bullet in enumerate(bullets):
            if i == 0:
                p2 = tf2.paragraphs[0]
            else:
                p2 = tf2.add_paragraph()
            # Support bold headers with "**text**" prefix
            if bullet.startswith("**") and "**" in bullet[2:]:
                end = bullet.index("**", 2)
                bold_part = bullet[2:end]
                rest = bullet[end+2:]
                run1 = p2.add_run()
                run1.text = bold_part
                run1.font.size = Pt(15)
                run1.font.bold = True
                run1.font.color.rgb = DARK_BLUE
                if rest:
                    run2 = p2.add_run()
                    run2.text = rest
                    run2.font.size = Pt(14)
                    run2.font.color.rgb = DARK_GREY
            else:
                p2.text = bullet
                p2.font.size = Pt(14)
                p2.font.color.rgb = DARK_GREY
            p2.space_before = Pt(6)
            p2.space_after = Pt(4)

    return slide


def _table_slide(prs, title, headers, rows, col_widths=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_shape_bg(slide, WHITE)

    # Top bar
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.9))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()

    txBox = slide.shapes.add_textbox(Inches(0.6), Inches(0.15), Inches(12), Inches(0.6))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = WHITE

    n_rows = len(rows) + 1
    n_cols = len(headers)
    table_width = Inches(12) if col_widths is None else sum(col_widths)
    left = Inches(0.6)
    top = Inches(1.3)
    row_height = Inches(0.4)
    table_height = row_height * n_rows

    tbl_shape = slide.shapes.add_table(n_rows, n_cols, left, top, table_width, table_height)
    tbl = tbl_shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = w

    # Header row
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE
        for para in cell.text_frame.paragraphs:
            para.font.size = Pt(12)
            para.font.bold = True
            para.font.color.rgb = WHITE
            para.alignment = PP_ALIGN.CENTER
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    # Data rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.text = str(val)
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GREY
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
            for para in cell.text_frame.paragraphs:
                para.font.size = Pt(11)
                para.font.color.rgb = DARK_GREY
                para.alignment = PP_ALIGN.CENTER
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    return slide


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Load artefacts (if available)
    test_metrics_main = _load_json("test_metrics_main_only.json", {})
    test_metrics_aug  = _load_json("test_metrics_augmented.json", {})
    hold_metrics_main = _load_json("holdout_metrics_main_only.json", {})
    hold_metrics_aug  = _load_json("holdout_metrics.json", {})
    ks_stats          = _load_json("ks_statistics.json", {})
    col_tracking      = _load_json("column_tracking.json", {})
    col_meta          = _load_json("column_metadata.json", {})
    best_params_main  = _load_json("best_params_main_only.json", {})
    best_params_aug   = _load_json("best_params_augmented.json", best_params_main)
    drop_info_main    = _load_json("dropped_features_main_only.json", {})
    drop_info_aug     = _load_json("dropped_features_augmented.json", {})
    all_metrics       = _load_json("all_metrics_summary.json", {})

    has_data = bool(test_metrics_main)

    def fmt(v, dec=4):
        if isinstance(v, (int, float)):
            return f"{v:.{dec}f}"
        return str(v) if v else "N/A"

    # ================================================================
    # SLIDE 1: Title
    # ================================================================
    _title_slide(
        prs,
        "Classification Model — Platam Performance Enhancement",
        "Data Augmentation Analysis & Model Evaluation  |  February 2026"
    )

    # ================================================================
    # SLIDE 2: Agenda / Table of Contents
    # ================================================================
    _content_slide(prs, "Agenda", [
        "**1. Objective** — Why we are doing this",
        "**2. Data & Process Overview** — End-to-end pipeline",
        "**3. Feature Selection & Preprocessing** — What was kept, dropped, and why",
        "**4. Model Training** — XGBoost with GridSearchCV",
        "**5. Metric Definitions** — What each metric means",
        "**6. Model Performance — Main Data** — Baseline model results",
        "**7. Model Performance — Augmented Data** — With InsightGenie data",
        "**8. Performance Comparison** — Main vs Augmented, Dev vs Holdout",
        "**9. KS Statistic** — Discrimination power",
        "**10. Feature Importance — Main Model** — XGB Gain & Permutation",
        "**11. Feature Importance — Augmented Model** — Impact of InsightGenie features",
        "**12. Column Tracking** — Full audit trail of columns",
        "**13. Features Used in Each Model** — Final feature lists",
        "**14. Key Findings & Recommendations**",
    ])

    # ================================================================
    # SLIDE 3: Objective
    # ================================================================
    _content_slide(prs, "1. Objective", [
        "**Goal:** Improve the Platam classification model performance by augmenting the existing "
        "internal data with InsightGenie data sources.",
        "",
        "**Approach:**",
        "  - Train a baseline model using only internal/main data features",
        "  - Train an augmented model using internal + InsightGenie data features",
        "  - Compare performance across Dev-Test and Holdout (true out-of-sample) sets",
        "  - Evaluate whether InsightGenie data provides meaningful lift in discrimination power",
        "",
        "**InsightGenie Data Sources:**",
        "  - Business financials (revenue, profit, financial strength)",
        "  - Digital presence scores (trust score, online presence, digital health)",
        "  - Social media data (Google Maps reviews/ratings, Facebook engagement, Instagram metrics)",
        "  - Fraud risk scoring (InsightGenie scores, fraud risk score)",
        "  - Industry classification",
    ])

    # ================================================================
    # SLIDE 4: Process Overview
    # ================================================================
    _content_slide(prs, "2. Data & Process Overview", [
        "**Step 1 — Data Loading & Holdout Split**",
        "  15% of data held out before any preprocessing (true OOS test)",
        "",
        "**Step 2 — Column Selection**",
        "  Keep only relevant features from the full dataset",
        "",
        "**Step 3 — Null Analysis**",
        "  Drop columns with > 40% null values",
        "",
        "**Step 4 — Variation Analysis**",
        "  KS test (numerical) & Cramer's V (categorical) to flag low-information columns",
        "  Threshold: 0.02 — columns below this are flagged for review",
        "",
        "**Step 5 — Binning & Encoding**",
        "  Numerical columns binned into 5 quantile bins, then one-hot encoded",
        "",
        "**Step 6 — InsightGenie Data Pipeline**",
        "  Same preprocessing applied to InsightGenie features, then concatenated",
        "",
        "**Step 7 — Model Training**",
        "  XGBoost with GridSearchCV (5-fold stratified CV, ROC AUC scoring)",
        "",
        "**Step 8 — Feature Importance & Selection**",
        "  Auto-flag suspicious features (high XGB gain but low permutation importance)",
    ])

    # ================================================================
    # SLIDE 5: Feature Selection & Preprocessing
    # ================================================================
    initial_cols = col_tracking.get("initial_columns", [])
    null_drops = col_tracking.get("dropped_by_null_analysis", [])
    var_drops = col_tracking.get("dropped_by_variation", [])
    bin_moved = col_tracking.get("binning_moved_to_categorical", [])

    _content_slide(prs, "3. Feature Selection & Preprocessing", [
        f"**Initial Main Columns:** {len(initial_cols)} features selected",
        f"**Dropped by Null Analysis (>40% nulls):** {len(null_drops)} columns"
        + (f" — {', '.join(null_drops)}" if null_drops else ""),
        f"**Dropped by Low Variation:** {len(var_drops)} columns"
        + (f" — {', '.join(var_drops)}" if var_drops else ""),
        f"**Binning Moved to Categorical:** {len(bin_moved)} columns"
        + (f" — {', '.join(bin_moved)}" if bin_moved else ""),
        "",
        "**InsightGenie Columns Pipeline:**",
        f"  Input: {len(col_tracking.get('external_columns_input', []))} InsightGenie features",
        f"  Covering: Business financials, SWOT analysis, digital scores, social media metrics",
        f"  Same preprocessing: null analysis, variation analysis, binning, OHE",
    ])

    # ================================================================
    # SLIDE 6: Model Training Details
    # ================================================================
    param_bullets = []
    if best_params_main:
        for k, v in best_params_main.items():
            param_bullets.append(f"  {k}: {v}")

    _content_slide(prs, "4. Model Training — XGBoost with GridSearchCV", [
        "**Algorithm:** XGBoost (Gradient Boosted Trees)",
        "**Cross-Validation:** 5-fold Stratified K-Fold",
        "**Scoring Metric:** ROC AUC",
        "**Class Imbalance Handling:** scale_pos_weight = negative_count / positive_count",
        "",
        "**Hyperparameter Search Space:**",
        "  n_estimators: [100, 200, 300]",
        "  max_depth: [3, 5, 7]",
        "  learning_rate: [0.01, 0.05, 0.1]",
        "  subsample: [0.8, 1.0]",
        "  colsample_bytree: [0.8, 1.0]",
        "  min_child_weight: [1, 3, 5]",
        "",
        "**Best Parameters (Main Model):**",
    ] + (param_bullets if param_bullets else ["  (will be populated after training)"]),
    )

    # ================================================================
    # SLIDE 7: Metric Definitions
    # ================================================================
    _content_slide(prs, "5. Metric Definitions", [
        "**Accuracy** — Proportion of correctly classified instances (TP+TN) / Total. "
        "Simple but can be misleading with imbalanced data.",
        "",
        "**Balanced Accuracy** — Average of recall for each class: (Sensitivity + Specificity) / 2. "
        "Better for imbalanced datasets as it gives equal weight to both classes.",
        "",
        "**F1 Score** — Harmonic mean of Precision and Recall: 2 x (P x R) / (P + R). "
        "Balances the trade-off between false positives and false negatives.",
        "",
        "**AUC (ROC)** — Area Under the ROC Curve. Measures the model's ability to "
        "rank-order predictions. 0.5 = random, 1.0 = perfect separation.",
        "",
        "**Gini Coefficient** — 2 x AUC - 1. Common in credit scoring. "
        "0 = no discrimination, 1 = perfect discrimination.",
        "",
        "**KS Statistic (Kolmogorov-Smirnov)** — Maximum separation between the cumulative "
        "distribution functions of positive and negative classes. Higher = better discrimination. "
        "Widely used in credit risk to measure model power.",
    ])

    # ================================================================
    # SLIDE 8: Main Model Performance
    # ================================================================
    if has_data:
        main_rows = []
        for metric_name in ["Accuracy", "Balanced Accuracy", "F1 Score", "AUC (ROC)", "Gini"]:
            dev_val = fmt(test_metrics_main.get(metric_name))
            hold_val = fmt(hold_metrics_main.get(metric_name))
            main_rows.append([metric_name, dev_val, hold_val])
        # Add KS
        main_rows.append(["KS Statistic",
                          fmt(ks_stats.get("Dev_Main_KS")),
                          fmt(ks_stats.get("Hold_Main_KS"))])
    else:
        main_rows = [
            ["Accuracy", "—", "—"],
            ["Balanced Accuracy", "—", "—"],
            ["F1 Score", "—", "—"],
            ["AUC (ROC)", "—", "—"],
            ["Gini", "—", "—"],
            ["KS Statistic", "—", "—"],
        ]

    _table_slide(prs,
        "6. Model Performance — Main Data Only (Baseline)",
        ["Metric", "Dev Test", "Holdout (OOS)"],
        main_rows,
        col_widths=[Inches(4), Inches(4), Inches(4)],
    )

    # ================================================================
    # SLIDE 9: Augmented Model Performance
    # ================================================================
    if has_data and test_metrics_aug:
        aug_rows = []
        for metric_name in ["Accuracy", "Balanced Accuracy", "F1 Score", "AUC (ROC)", "Gini"]:
            dev_val = fmt(test_metrics_aug.get(metric_name))
            hold_val = fmt(hold_metrics_aug.get(metric_name))
            aug_rows.append([metric_name, dev_val, hold_val])
        aug_rows.append(["KS Statistic",
                         fmt(ks_stats.get("Dev_Aug_KS")),
                         fmt(ks_stats.get("Hold_Aug_KS"))])
    else:
        aug_rows = [
            ["Accuracy", "—", "—"],
            ["Balanced Accuracy", "—", "—"],
            ["F1 Score", "—", "—"],
            ["AUC (ROC)", "—", "—"],
            ["Gini", "—", "—"],
            ["KS Statistic", "—", "—"],
        ]

    _table_slide(prs,
        "7. Model Performance — Augmented Data (Main + InsightGenie)",
        ["Metric", "Dev Test", "Holdout (OOS)"],
        aug_rows,
        col_widths=[Inches(4), Inches(4), Inches(4)],
    )

    # ================================================================
    # SLIDE 10: Performance Comparison (4 columns)
    # ================================================================
    if has_data:
        comp_rows = []
        for metric_name in ["Accuracy", "Balanced Accuracy", "F1 Score", "AUC (ROC)", "Gini"]:
            comp_rows.append([
                metric_name,
                fmt(test_metrics_main.get(metric_name)),
                fmt(test_metrics_aug.get(metric_name, test_metrics_main.get(metric_name))),
                fmt(hold_metrics_main.get(metric_name)),
                fmt(hold_metrics_aug.get(metric_name, hold_metrics_main.get(metric_name))),
            ])
        comp_rows.append([
            "KS Statistic",
            fmt(ks_stats.get("Dev_Main_KS")),
            fmt(ks_stats.get("Dev_Aug_KS")),
            fmt(ks_stats.get("Hold_Main_KS")),
            fmt(ks_stats.get("Hold_Aug_KS")),
        ])
    else:
        comp_rows = [
            ["Accuracy", "—", "—", "—", "—"],
            ["Balanced Accuracy", "—", "—", "—", "—"],
            ["F1 Score", "—", "—", "—", "—"],
            ["AUC (ROC)", "—", "—", "—", "—"],
            ["Gini", "—", "—", "—", "—"],
            ["KS Statistic", "—", "—", "—", "—"],
        ]

    _table_slide(prs,
        "8. Performance Comparison — Main vs Augmented, Dev vs Holdout",
        ["Metric", "Dev Main", "Dev Augmented", "Holdout Main", "Holdout Augmented"],
        comp_rows,
        col_widths=[Inches(3), Inches(2.3), Inches(2.3), Inches(2.3), Inches(2.3)],
    )

    # ================================================================
    # SLIDE 11: KS Statistic
    # ================================================================
    ks_bullets = [
        "**KS Statistic** measures the maximum distance between the cumulative distribution "
        "functions of good (non-default) and bad (default) populations.",
        "",
        "**Interpretation:**",
        "  KS > 0.40 — Excellent discrimination",
        "  KS 0.30 - 0.40 — Good discrimination",
        "  KS 0.20 - 0.30 — Acceptable discrimination",
        "  KS < 0.20 — Weak discrimination",
        "",
        "**Results:**",
    ]
    if ks_stats:
        ks_bullets.extend([
            f"  Dev Main:       KS = {fmt(ks_stats.get('Dev_Main_KS'))}",
            f"  Dev Augmented:  KS = {fmt(ks_stats.get('Dev_Aug_KS'))}",
            f"  Holdout Main:   KS = {fmt(ks_stats.get('Hold_Main_KS'))}",
            f"  Holdout Aug:    KS = {fmt(ks_stats.get('Hold_Aug_KS'))}",
        ])
    else:
        ks_bullets.append("  (KS values will be populated after training)")

    _content_slide(prs, "9. KS Statistic — Model Discrimination Power", ks_bullets)

    # ================================================================
    # SLIDE 12: Feature Importance — Main Model
    # ================================================================
    # Try to load predictor report for detailed importance
    import pandas as pd
    report_path = os.path.join(ARTEFACT_DIR, "predictor_report.csv")
    report_df = None
    if os.path.isfile(report_path):
        report_df = pd.read_csv(report_path)
        # Rename monnai -> insightgenie in displayed feature names
        for col in ["ohe_feature", "original_column"]:
            report_df[col] = report_df[col].str.replace("monnai", "insightgenie", regex=False)

    if report_df is not None:
        main_report = report_df[report_df["model"] == "MAIN_ONLY"].copy()
        main_used = main_report[main_report["status"] == "USED"].copy()
        main_used = main_used.sort_values("xgb_gain", ascending=False)

        top_n = min(15, len(main_used))
        fi_rows = []
        for _, row in main_used.head(top_n).iterrows():
            fi_rows.append([
                row["ohe_feature"],
                row["original_column"],
                fmt(row["xgb_gain"], 6) if row["xgb_gain"] != "" else "—",
                str(int(row["xgb_gain_rank"])) if row["xgb_gain_rank"] != "" else "—",
                fmt(row["perm_importance_auc_drop"], 6) if row["perm_importance_auc_drop"] != "" else "—",
                str(int(row["perm_importance_rank"])) if row["perm_importance_rank"] != "" else "—",
            ])

        _table_slide(prs,
            "10. Feature Importance — Main Model (Top 15 by XGB Gain)",
            ["OHE Feature", "Original Column", "XGB Gain", "Gain Rank", "Perm Imp (AUC Drop)", "Perm Rank"],
            fi_rows,
            col_widths=[Inches(3.5), Inches(2.5), Inches(1.5), Inches(1.2), Inches(2), Inches(1.5)],
        )
    else:
        _content_slide(prs,
            "10. Feature Importance — Main Model",
            [
                "**XGBoost Gain:** Measures how much each feature contributes to the model's splits. "
                "Higher gain = feature is used more frequently and effectively in tree splits.",
                "",
                "**Permutation Importance (AUC Drop):** Measures the actual drop in ROC AUC "
                "when a feature's values are randomly shuffled. Shows true predictive value.",
                "",
                "**Suspicious Feature Detection:**",
                "  Features with high XGB gain but low permutation importance are flagged.",
                "  These may be high-cardinality noise that trees memorise but don't genuinely help.",
                "  Such features are auto-dropped to prevent overfitting.",
                "",
                "(Detailed feature importance table will be populated after training)",
            ])

    # ================================================================
    # SLIDE 13: Feature Importance — Augmented Model
    # ================================================================
    if report_df is not None:
        aug_report = report_df[report_df["model"] == "AUGMENTED"].copy()
        if len(aug_report) > 0:
            aug_used = aug_report[aug_report["status"] == "USED"].copy()
            aug_used = aug_used.sort_values("xgb_gain", ascending=False)

            top_n = min(15, len(aug_used))
            fi_aug_rows = []
            for _, row in aug_used.head(top_n).iterrows():
                fi_aug_rows.append([
                    row["ohe_feature"],
                    row["original_column"],
                    fmt(row["xgb_gain"], 6) if row["xgb_gain"] != "" else "—",
                    str(int(row["xgb_gain_rank"])) if row["xgb_gain_rank"] != "" else "—",
                    fmt(row["perm_importance_auc_drop"], 6) if row["perm_importance_auc_drop"] != "" else "—",
                    str(int(row["perm_importance_rank"])) if row["perm_importance_rank"] != "" else "—",
                ])

            _table_slide(prs,
                "11. Feature Importance — Augmented Model (Top 15 by XGB Gain)",
                ["OHE Feature", "Original Column", "XGB Gain", "Gain Rank", "Perm Imp (AUC Drop)", "Perm Rank"],
                fi_aug_rows,
                col_widths=[Inches(3.5), Inches(2.5), Inches(1.5), Inches(1.2), Inches(2), Inches(1.5)],
            )

            # Extra slide: InsightGenie features contribution
            ext_columns_input = col_tracking.get("external_columns_input", [])
            ext_encode_cols_list = col_meta.get("ext_encode_cols", [])
            ext_features_in_model = aug_used[aug_used["original_column"].isin(
                ext_encode_cols_list)].copy()

            if len(ext_features_in_model) > 0:
                ext_fi_rows = []
                ext_features_in_model = ext_features_in_model.sort_values(
                    "xgb_gain", ascending=False)
                for _, row in ext_features_in_model.head(15).iterrows():
                    ext_fi_rows.append([
                        row["ohe_feature"],
                        row["original_column"],
                        fmt(row["xgb_gain"], 6) if row["xgb_gain"] != "" else "—",
                        fmt(row["perm_importance_auc_drop"], 6) if row["perm_importance_auc_drop"] != "" else "—",
                    ])

                _table_slide(prs,
                    "11b. InsightGenie Features — Contribution to Augmented Model",
                    ["OHE Feature", "Original Column", "XGB Gain", "Perm Importance (AUC Drop)"],
                    ext_fi_rows,
                    col_widths=[Inches(4), Inches(3), Inches(2.5), Inches(2.5)],
                )
        else:
            _content_slide(prs,
                "11. Feature Importance — Augmented Model",
                ["No augmented model results available (no InsightGenie data was used)."])
    else:
        _content_slide(prs,
            "11. Feature Importance — Augmented Model",
            [
                "**With InsightGenie Data Augmentation:**",
                "  The augmented model includes InsightGenie features from business financials,",
                "  digital presence, social media, and fraud risk scoring.",
                "",
                "**Key analysis:**",
                "  - XGB Gain: Which InsightGenie features the model uses most in tree splits",
                "  - Perm Importance (AUC Drop): Which InsightGenie features truly improve predictions",
                "  - Features with high gain but low perm importance are noise — auto-dropped",
                "",
                "(Detailed table will be populated after training)",
            ])

    # ================================================================
    # SLIDE 14: Column Trail Summary (running counts per step)
    # ================================================================
    trail_summary_path = os.path.join(ARTEFACT_DIR, "column_trail_summary.csv")
    trail_report_path = os.path.join(ARTEFACT_DIR, "column_trail_report.csv")
    trail_summary_df = None
    trail_report_df = None
    if os.path.isfile(trail_summary_path):
        trail_summary_df = pd.read_csv(trail_summary_path)
    if os.path.isfile(trail_report_path):
        trail_report_df = pd.read_csv(trail_report_path)

    if trail_summary_df is not None:
        for src_label in trail_summary_df["source"].unique():
            src_data = trail_summary_df[trail_summary_df["source"] == src_label]
            summary_rows = []
            for _, row in src_data.iterrows():
                summary_rows.append([
                    row["step"], str(row["columns_dropped"]),
                    str(row["columns_remaining"]),
                ])
            _table_slide(prs,
                f"12. Column Trail — {src_label} (Running Counts)",
                ["Pipeline Step", "Columns Dropped", "Columns Remaining"],
                summary_rows,
                col_widths=[Inches(5), Inches(3.5), Inches(3.5)],
            )
    elif col_tracking:
        tracking_rows = [
            ["Initial Main Features", str(len(col_tracking.get("initial_columns", [])))],
            ["Dropped by Null Analysis", str(len(col_tracking.get("dropped_by_null_analysis", [])))],
            ["Dropped by Low Variation", str(len(col_tracking.get("dropped_by_variation", [])))],
            ["Columns After OHE (Main)", str(len(col_tracking.get("columns_after_ohe", [])))],
            ["Main Model Features Used", str(len(col_tracking.get("main_model_features_used", [])))],
        ]
        _table_slide(prs,
            "12. Column Tracking — Summary",
            ["Stage", "Count"],
            tracking_rows,
            col_widths=[Inches(7), Inches(5)],
        )
    else:
        _content_slide(prs, "12. Column Tracking",
            ["(Column tracking data will be populated after training)"])

    # ================================================================
    # SLIDE 14b: Detailed Column Trail (per-column boolean breakdown)
    # ================================================================
    if trail_report_df is not None:
        for src_label in trail_report_df["source"].unique():
            src = trail_report_df[trail_report_df["source"] == src_label].copy()
            # Collapse to one row per original column (show OHE count)
            grouped = src.groupby("original_column", sort=False).agg(
                column_type=("column_type", "first"),
                dropped_null=("dropped_null_analysis", "first"),
                flagged_var=("flagged_low_variation", "first"),
                dropped_var=("dropped_low_variation", "first"),
                binning_result=("binning_result", "first"),
                ohe_count=("ohe_feature", lambda x: sum(1 for v in x if v != "—")),
                used_main=("finally_used_main", "sum"),
                used_aug=("finally_used_aug", "sum"),
            ).reset_index()

            trail_rows = []
            for _, row in grouped.iterrows():
                yn = lambda v: "Yes" if v else "No"
                trail_rows.append([
                    row["original_column"],
                    str(row["column_type"]),
                    yn(row["dropped_null"]),
                    yn(row["flagged_var"]),
                    yn(row["dropped_var"]),
                    str(row["binning_result"]),
                    str(int(row["ohe_count"])),
                    str(int(row["used_main"])),
                    str(int(row["used_aug"])),
                ])

            MAX_PER_SLIDE = 16
            num_slides = max(1, (len(trail_rows) + MAX_PER_SLIDE - 1) // MAX_PER_SLIDE)
            for slide_idx in range(num_slides):
                start = slide_idx * MAX_PER_SLIDE
                end = min(start + MAX_PER_SLIDE, len(trail_rows))
                chunk = trail_rows[start:end]
                suffix = f" (Page {slide_idx + 1}/{num_slides})" if num_slides > 1 else ""
                _table_slide(prs,
                    f"12b. Column Trail — {src_label} Detail{suffix}",
                    ["Column", "Type", "Null\nDrop", "Var\nFlag",
                     "Var\nDrop", "Binning", "OHE\nCols",
                     "Used\nMain", "Used\nAug"],
                    chunk,
                    col_widths=[Inches(2.6), Inches(1.2), Inches(0.8),
                                Inches(0.8), Inches(0.8), Inches(2),
                                Inches(0.8), Inches(0.8), Inches(0.8)],
                )

    # ================================================================
    # SLIDE 15a: Features Used — Main Model (Every OHE column)
    # ================================================================
    main_features = col_tracking.get("main_model_features_used", [])
    main_dropped = col_tracking.get("main_model_features_dropped", [])

    if report_df is not None:
        main_report = report_df[report_df["model"] == "MAIN_ONLY"].copy()
        main_used = main_report[main_report["status"] == "USED"].copy()
        main_used = main_used.sort_values(
            ["original_column", "ohe_feature"])

        total_ohe = len(main_used)
        total_orig = main_used["original_column"].nunique()

        feat_rows = []
        for i, (_, row) in enumerate(main_used.iterrows(), 1):
            feat_rows.append([
                str(i), row["ohe_feature"], row["original_column"],
            ])

        MAX_PER_SLIDE = 20
        num_slides = max(1, (len(feat_rows) + MAX_PER_SLIDE - 1) // MAX_PER_SLIDE)
        for slide_idx in range(num_slides):
            start = slide_idx * MAX_PER_SLIDE
            end = min(start + MAX_PER_SLIDE, len(feat_rows))
            chunk = feat_rows[start:end]
            suffix = f" (Page {slide_idx + 1}/{num_slides})" if num_slides > 1 else ""
            _table_slide(prs,
                f"13a. All OHE Features Used — Main Model "
                f"({total_ohe} features from {total_orig} columns){suffix}",
                ["#", "OHE Feature", "Original Column"],
                chunk,
                col_widths=[Inches(1), Inches(7), Inches(4)],
            )
    else:
        # Fallback: list features from column_tracking
        main_feat_bullets = [
            f"**Total features used:** {len(main_features)}",
            f"**Features dropped (auto-flag):** {len(main_dropped)}",
            "",
            "**Features used:**",
        ]
        for f_name in main_features[:40]:
            main_feat_bullets.append(f"  - {f_name}")
        if len(main_features) > 40:
            main_feat_bullets.append(f"  ... and {len(main_features) - 40} more")
        _content_slide(prs, "13a. Features — Main Model", main_feat_bullets)

    # ================================================================
    # SLIDE 15b: Features Used — Augmented Model (Every OHE column)
    # ================================================================
    aug_features = col_tracking.get("augmented_model_features_used", [])
    aug_dropped = col_tracking.get("augmented_model_features_dropped", [])

    if report_df is not None:
        aug_report = report_df[report_df["model"] == "AUGMENTED"].copy()
        if len(aug_report) > 0:
            aug_used = aug_report[aug_report["status"] == "USED"].copy()
            ext_encode_cols_set = set(col_meta.get("ext_encode_cols", []))

            aug_used = aug_used.copy()
            aug_used["source"] = aug_used["original_column"].apply(
                lambda x: "InsightGenie" if x in ext_encode_cols_set else "Main"
            )
            aug_used = aug_used.sort_values(
                ["source", "original_column", "ohe_feature"])

            total_ohe = len(aug_used)
            total_orig = aug_used["original_column"].nunique()
            n_main_ohe = len(aug_used[aug_used["source"] == "Main"])
            n_ext_ohe = len(aug_used[aug_used["source"] == "InsightGenie"])

            feat_rows = []
            for i, (_, row) in enumerate(aug_used.iterrows(), 1):
                feat_rows.append([
                    str(i), row["ohe_feature"], row["original_column"],
                    row["source"],
                ])

            MAX_PER_SLIDE = 20
            num_slides = max(1, (len(feat_rows) + MAX_PER_SLIDE - 1) // MAX_PER_SLIDE)
            for slide_idx in range(num_slides):
                start = slide_idx * MAX_PER_SLIDE
                end = min(start + MAX_PER_SLIDE, len(feat_rows))
                chunk = feat_rows[start:end]
                suffix = f" (Page {slide_idx + 1}/{num_slides})" if num_slides > 1 else ""
                _table_slide(prs,
                    f"13b. All OHE Features Used — Augmented Model "
                    f"({n_main_ohe} Main + {n_ext_ohe} InsightGenie = "
                    f"{total_ohe} features){suffix}",
                    ["#", "OHE Feature", "Original Column", "Source"],
                    chunk,
                    col_widths=[Inches(1), Inches(5), Inches(3), Inches(3)],
                )
        else:
            _content_slide(prs, "13b. Features — Augmented Model",
                ["No augmented model results available."])
    else:
        # Fallback: list features from column_tracking
        aug_feat_bullets = [
            f"**Total features used:** {len(aug_features)}",
            f"**Features dropped (auto-flag):** {len(aug_dropped)}",
            "",
            "**Features used:**",
        ]
        for f_name in aug_features[:40]:
            aug_feat_bullets.append(f"  - {f_name}")
        if len(aug_features) > 40:
            aug_feat_bullets.append(f"  ... and {len(aug_features) - 40} more")
        if not aug_features and not aug_dropped:
            aug_feat_bullets = ["No augmented model (no InsightGenie data used)."]
        _content_slide(prs, "13b. Features — Augmented Model", aug_feat_bullets)

    # ================================================================
    # SLIDE 17: Key Findings & Recommendations
    # ================================================================
    findings = [
        "**Baseline Model (Main Data Only):**",
    ]
    if test_metrics_main:
        findings.append(
            f"  AUC = {fmt(test_metrics_main.get('AUC (ROC)'))}, "
            f"Gini = {fmt(test_metrics_main.get('Gini'))}, "
            f"KS = {fmt(ks_stats.get('Dev_Main_KS'))}"
        )
    else:
        findings.append("  (Metrics pending)")

    findings.extend(["", "**Augmented Model (Main + InsightGenie Data):**"])
    if test_metrics_aug:
        findings.append(
            f"  AUC = {fmt(test_metrics_aug.get('AUC (ROC)'))}, "
            f"Gini = {fmt(test_metrics_aug.get('Gini'))}, "
            f"KS = {fmt(ks_stats.get('Dev_Aug_KS'))}"
        )
        # Compute lift
        main_auc = test_metrics_main.get("AUC (ROC)", 0)
        aug_auc = test_metrics_aug.get("AUC (ROC)", 0)
        if main_auc and aug_auc:
            lift = aug_auc - main_auc
            findings.append(
                f"  AUC Lift from augmentation: {'+' if lift >= 0 else ''}{lift:.4f}"
            )
    else:
        findings.append("  (Metrics pending)")

    findings.extend([
        "",
        "**Holdout Validation:**",
    ])
    if hold_metrics_main:
        hold_main_auc = hold_metrics_main.get("AUC (ROC)", 0)
        dev_main_auc = test_metrics_main.get("AUC (ROC)", 0)
        if hold_main_auc and dev_main_auc:
            diff = hold_main_auc - dev_main_auc
            findings.append(
                f"  Main model: Dev AUC {fmt(dev_main_auc)} vs Holdout AUC {fmt(hold_main_auc)} "
                f"(diff: {'+' if diff >= 0 else ''}{diff:.4f})"
            )
    else:
        findings.append("  (Pending)")

    findings.extend([
        "",
        "**Recommendations:**",
        "  - Review the InsightGenie features with high permutation importance for production inclusion",
        "  - Monitor model stability with periodic holdout validation",
        "  - Consider feature engineering on top-contributing InsightGenie features",
    ])

    _content_slide(prs, "14. Key Findings & Recommendations", findings)

    # ================================================================
    # SLIDE 18: Thank You / Q&A
    # ================================================================
    _title_slide(prs, "Thank You", "Questions & Discussion")

    # ── Save ──────────────────────────────────────────────────────
    prs.save(OUTPUT_PATH)
    print(f"Presentation saved to: {OUTPUT_PATH}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
