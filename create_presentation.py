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


CODE_BG    = RGBColor(0x1E, 0x1E, 0x2E)  # dark background for code
CODE_FG    = RGBColor(0xCD, 0xD6, 0xF4)  # light text
CODE_TITLE = RGBColor(0xA6, 0xE3, 0xA1)  # green for title accent


def _code_slide(prs, title, code_text, note=""):
    """Slide with a monospace code block on a dark background."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_shape_bg(slide, CODE_BG)

    # Top bar
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.8))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RGBColor(0x11, 0x11, 0x1B)
    bar.line.fill.background()

    # Title
    txBox = slide.shapes.add_textbox(Inches(0.6), Inches(0.12), Inches(12), Inches(0.55))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = CODE_TITLE

    # Code block background
    code_box_bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(1.0),
        Inches(12.5), Inches(5.6))
    code_box_bg.fill.solid()
    code_box_bg.fill.fore_color.rgb = RGBColor(0x16, 0x16, 0x25)
    code_box_bg.line.color.rgb = RGBColor(0x31, 0x31, 0x44)
    code_box_bg.line.width = Pt(1)

    # Code text
    code_tb = slide.shapes.add_textbox(Inches(0.7), Inches(1.15), Inches(12), Inches(5.4))
    tf2 = code_tb.text_frame
    tf2.word_wrap = True

    lines = code_text.strip().split("\n")
    for i, line in enumerate(lines):
        if i == 0:
            p2 = tf2.paragraphs[0]
        else:
            p2 = tf2.add_paragraph()
        p2.text = line
        p2.font.size = Pt(11)
        p2.font.name = "Consolas"
        p2.font.color.rgb = CODE_FG
        p2.space_before = Pt(1)
        p2.space_after = Pt(1)

    # Optional note
    if note:
        note_tb = slide.shapes.add_textbox(Inches(0.6), Inches(6.7), Inches(12), Inches(0.6))
        tf3 = note_tb.text_frame
        tf3.word_wrap = True
        p3 = tf3.paragraphs[0]
        p3.text = note
        p3.font.size = Pt(12)
        p3.font.italic = True
        p3.font.color.rgb = RGBColor(0x94, 0x9C, 0xBB)

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
            # Check if __NULL__ drop columns exist
            has_null_drop = "dropped_null_feature_main" in src.columns

            agg_dict = {
                "column_type": ("column_type", "first"),
                "dropped_null": ("dropped_null_analysis", "first"),
                "flagged_var": ("flagged_low_variation", "first"),
                "dropped_var": ("dropped_low_variation", "first"),
                "binning_result": ("binning_result", "first"),
                "ohe_count": ("ohe_feature", lambda x: sum(1 for v in x if v != "—")),
                "used_main": ("finally_used_main", "sum"),
                "used_aug": ("finally_used_aug", "sum"),
            }
            if has_null_drop:
                agg_dict["null_drop_main"] = ("dropped_null_feature_main", "sum")
                agg_dict["null_drop_aug"] = ("dropped_null_feature_aug", "sum")
            grouped = src.groupby("original_column", sort=False).agg(**agg_dict).reset_index()

            trail_rows = []
            for _, row in grouped.iterrows():
                yn = lambda v: "Yes" if v else "No"
                base = [
                    row["original_column"],
                    str(row["column_type"]),
                    yn(row["dropped_null"]),
                    yn(row["flagged_var"]),
                    yn(row["dropped_var"]),
                    str(row["binning_result"]),
                    str(int(row["ohe_count"])),
                    str(int(row["used_main"])),
                ]
                if has_null_drop:
                    base.append(str(int(row.get("null_drop_main", 0))))
                base.append(str(int(row["used_aug"])))
                if has_null_drop:
                    base.append(str(int(row.get("null_drop_aug", 0))))
                trail_rows.append(base)

            headers = ["Column", "Type", "Null\nDrop", "Var\nFlag",
                       "Var\nDrop", "Binning", "OHE\nCols", "Used\nMain"]
            widths = [Inches(2.2), Inches(1.1), Inches(0.7), Inches(0.7),
                      Inches(0.7), Inches(1.6), Inches(0.7), Inches(0.7)]
            if has_null_drop:
                headers.append("NULL\nMain")
                widths.append(Inches(0.7))
            headers.append("Used\nAug")
            widths.append(Inches(0.7))
            if has_null_drop:
                headers.append("NULL\nAug")
                widths.append(Inches(0.7))

            MAX_PER_SLIDE = 16
            num_slides = max(1, (len(trail_rows) + MAX_PER_SLIDE - 1) // MAX_PER_SLIDE)
            for slide_idx in range(num_slides):
                start = slide_idx * MAX_PER_SLIDE
                end = min(start + MAX_PER_SLIDE, len(trail_rows))
                chunk = trail_rows[start:end]
                suffix = f" (Page {slide_idx + 1}/{num_slides})" if num_slides > 1 else ""
                _table_slide(prs,
                    f"12b. Column Trail — {src_label} Detail{suffix}",
                    headers,
                    chunk,
                    col_widths=widths,
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


INTERNAL_OUTPUT_PATH = "Internal_Pipeline_Guide.pptx"


def create_internal_deck():
    """Generate a separate PPT explaining the pipeline for the internal team."""
    prs = Presentation()
    prs.slide_width  = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ── 1. Title ──────────────────────────────────────────────────
    _title_slide(prs, "Classification Pipeline — Internal Guide",
                 "Process, Functions, Parameters, Hyperparameters & Code")

    # ── 2. Pipeline Overview ──────────────────────────────────────
    _content_slide(prs, "1. Pipeline Overview — End-to-End Flow", [
        "**Step 1 — Load & Holdout Split:** Read raw CSV, create a 15% holdout set (never seen during training).",
        "**Step 2 — Column Selection:** Keep only pre-approved COLUMNS_LIST features (MODE='keep').",
        "**Step 3 — Null Analysis:** Drop columns where >40% of values are null (configurable via NULL_THRESHOLD_PCT).",
        "**Step 4 — Variation Analysis:** Compute Cramer's-V between each feature and the target. "
            "Flag low-variation columns; manually drop columns listed in DROP_LOW_VARIATION.",
        "**Step 5 — Column Typing:** Auto-classify columns as categorical or numerical "
            "(threshold: CAT_NUNIQUE_THRESHOLD). FORCE_NUMERIC overrides selected columns.",
        "**Step 6 — Binning:** KBinsDiscretizer on numerical columns (quantile strategy by default). "
            "Columns that fail binning fall back to categorical or are dropped.",
        "**Step 7 — One-Hot Encoding (OHE):** All features converted to binary dummy columns. "
            "Rare categories (<MAX_CATEGORIES) are collapsed to '__rare__'. Nulls become '__NULL__'.",
        "**Step 8 — External (InsightGenie) Pipeline:** Same steps 3-7 applied to external columns, "
            "then concatenated with main features.",
        "**Step 9 — Save Artefacts:** Bin edges, OHE encoder, rare mappings, column metadata saved to artefacts/.",
        "**Step 10 — Train Main Model:** XGBoost + GridSearchCV on main features only.",
        "**Step 11 — Train Augmented Model:** XGBoost + GridSearchCV on main + InsightGenie features.",
        "**Step 12 — Reports:** Predictor report, column trail CSV, timing, holdout evaluation.",
    ])

    # ── Pipeline main() code ──────────────────────────────────────
    _code_slide(prs, "Code: Pipeline main() — Steps 1-4", """\
def main():
    progress = PipelineProgress()
    os.makedirs(ARTEFACT_DIR, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────
    progress.start_step(0)
    df_full = pd.read_csv(DATA_PATH)
    df_full.rename(columns=TARGET_RENAME, inplace=True)

    # Holdout split (15% true out-of-sample, never used during training)
    holdout_raw, df_dev_full = train_test_split(
        df_full, test_size=(1 - HOLDOUT_PCT), random_state=HOLDOUT_SEED,
        stratify=df_full[TARGET_COL])

    # ── 2. Column selection ───────────────────────────────────
    progress.start_step(1)
    if MODE == "keep":
        keep = list(set(COLUMNS_LIST + [TARGET_COL]))
        df = df[[c for c in keep if c in df.columns]]

    # ── 3. Null analysis ──────────────────────────────────────
    progress.start_step(2)
    null_drops = null_analysis(df, feature_cols, NULL_THRESHOLD_PCT, TARGET_COL)
    df.drop(columns=null_drops, inplace=True)

    # ── 4. Variation analysis ─────────────────────────────────
    progress.start_step(3)
    flagged, var_scores = variation_analysis(df, feature_cols, TARGET_COL, VARIATION_THRESHOLD)
    df.drop(columns=[c for c in DROP_LOW_VARIATION if c in df.columns], inplace=True)""",
        note="File: train_pipeline.py — main() function (simplified for readability)")

    _code_slide(prs, "Code: Pipeline main() — Steps 5-8", """\
    # ── 5. Identify column types ──────────────────────────────
    progress.start_step(4)
    cat_cols, num_cols = identify_col_types(df, feature_cols, CAT_NUNIQUE_THRESHOLD, FORCE_NUMERIC)

    # ── 6. Binning ────────────────────────────────────────────
    progress.start_step(5)
    df, bin_edges_store, binning_moved_to_cat, binning_dropped = bin_columns(
        df, num_cols, N_BINS, BINNING_STRATEGY, CUSTOM_BINS, bin_edges_store,
        fallback=BINNING_FALLBACK)

    # ── 7. OHE ────────────────────────────────────────────────
    progress.start_step(6)
    df, encode_cols, ohe, rare_mappings = ohe_columns(
        df, TARGET_COL, DROP_FIRST, MAX_CATEGORIES, rare_mappings)

    # ── 8. External columns pipeline ──────────────────────────
    progress.start_step(7)
    if EXT_COLUMNS and len(EXT_COLUMNS) > 0:
        ext_df = df_dev_full[available_ext + [TARGET_COL]].copy()
        # Same steps: null → variation → types → binning → OHE
        ext_null_drops = null_analysis(ext_df, ext_feature_cols, EXT_NULL_THRESHOLD_PCT, TARGET_COL)
        ext_flagged, _ = variation_analysis(ext_df, ext_feature_cols, TARGET_COL, EXT_VARIATION_THRESHOLD)
        ext_cat_cols, ext_num_cols = identify_col_types(ext_df, ...)
        ext_df, ext_bin_edges_store, ext_binning_moved, ext_binning_dropped = bin_columns(...)
        ext_df, ext_encode_cols, ext_ohe, ext_rare_mappings = ohe_columns(...)
        # Concatenate with main
        df = pd.concat([df.drop(columns=[TARGET_COL]), ext_only, df[[TARGET_COL]]], axis=1)""",
        note="File: train_pipeline.py — main() function (simplified for readability)")

    # ── 3. Helper Functions ───────────────────────────────────
    _content_slide(prs, "2. Key Functions — What Each One Does", [
        "**cramers_v(col, target):** Measures association between a categorical feature and the target "
            "(chi-squared based). Used in variation analysis to flag weak predictors.",
        "",
        "**null_analysis(df, cols, threshold, target):** Returns list of columns where null % > threshold. "
            "These are dropped to prevent sparse-data issues.",
        "",
        "**variation_analysis(df, cols, target, threshold):** Computes Cramer's-V for each column vs target. "
            "Returns flagged columns (below threshold) and their scores. "
            "Columns in DROP_LOW_VARIATION are actually removed.",
        "",
        "**identify_col_types(df, cols, cat_threshold, force_numeric):** "
            "Categorical if nunique <= cat_threshold (default 20). "
            "Columns in FORCE_NUMERIC are always treated as numerical.",
        "",
        "**bin_columns(df, num_cols, n_bins, strategy, custom_bins, ...):** "
            "Discretizes numerical columns into n_bins bins (default 5, quantile strategy). "
            "Creates col_bin column, drops original. Falls back to categorical or drops on failure.",
        "",
        "**ohe_columns(df, target, drop_first, max_categories, ...):** "
            "Applies OneHotEncoder (sklearn). Rare categories collapsed first. "
            "Returns encoded DataFrame + encoder object for deployment reuse.",
    ])

    # ── Code: cramers_v + null_analysis ───────────────────────
    _code_slide(prs, "Code: cramers_v() + null_analysis()", """\
def cramers_v(col, target):
    \"\"\"Cramer's V statistic — measures association between two categorical variables.\"\"\"
    ct = pd.crosstab(col, target)
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0


def null_analysis(dataframe, columns, threshold_pct, target_col):
    \"\"\"Return columns where null percentage exceeds threshold.\"\"\"
    null_pct = (dataframe[columns].isnull().sum() / len(dataframe) * 100)
    null_pct = null_pct.sort_values(ascending=False)
    cols_to_drop = null_pct[null_pct > threshold_pct].index.tolist()
    if target_col in cols_to_drop:
        cols_to_drop.remove(target_col)
    return cols_to_drop""",
        note="cramers_v: higher = stronger association.  null_analysis: returns list of column names to drop.")

    # ── Code: variation_analysis ──────────────────────────────
    _code_slide(prs, "Code: variation_analysis()", """\
def variation_analysis(dataframe, columns, target_col, threshold):
    \"\"\"Score each feature's predictive variation vs the target.
    Categorical: Cramer's V.   Numerical: KS statistic between classes.\"\"\"
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
    return flagged, scores""",
        note="Flagged columns have score < VARIATION_THRESHOLD. Only DROP_LOW_VARIATION columns are actually removed.")

    # ── Code: identify_col_types ──────────────────────────────
    _code_slide(prs, "Code: identify_col_types() + bin_columns()", """\
def identify_col_types(dataframe, feature_cols, nunique_threshold=10, force_numeric=None):
    \"\"\"Classify columns as categorical or numerical.\"\"\"
    force_numeric = set(force_numeric or [])
    cat = [c for c in feature_cols
           if c not in force_numeric
           and (not pd.api.types.is_numeric_dtype(dataframe[c])
                or dataframe[c].nunique() <= nunique_threshold)]
    num = [c for c in feature_cols if c not in cat]
    return cat, num


def bin_columns(dataframe, num_cols, n_bins, strategy, custom_bins, bin_store,
                fallback="categorical"):
    \"\"\"Bin numerical columns. Falls back to categorical or drops on failure.\"\"\"
    moved_to_cat = []
    dropped = []
    successfully_binned = []
    for col in num_cols:
        try:
            if col in custom_bins:
                edges = custom_bins[col]
                dataframe[col + "_bin"] = pd.cut(dataframe[col], bins=edges, ...)
            else:
                binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
                dataframe.loc[valid_mask, col + "_bin"] = binner.fit_transform(...)
            successfully_binned.append(col)
        except Exception:
            if fallback == "categorical": moved_to_cat.append(col)
            else: dropped.append(col)
    dataframe.drop(columns=successfully_binned, inplace=True)   # drop originals, keep _bin
    return dataframe, bin_store, moved_to_cat, dropped""",
        note="After binning: original 'col' dropped, 'col_bin' kept. Failed cols go to categorical or are dropped.")

    # ── Code: ohe_columns ─────────────────────────────────────
    _code_slide(prs, "Code: ohe_columns()", """\
def ohe_columns(dataframe, target_col, drop_first, max_categories, rare_map_store):
    \"\"\"One-hot encode all feature columns. Collapse rare categories first.\"\"\"
    encode_cols = [c for c in dataframe.columns if c != target_col]

    # Collapse rare categories → '__rare__'
    if max_categories is not None:
        for col in encode_cols:
            counts = dataframe[col].value_counts()
            rare_cats = counts[counts < max_categories].index.tolist()
            if rare_cats:
                dataframe[col] = dataframe[col].apply(
                    lambda x: "__rare__" if x in rare_cats else x)
                rare_map_store[col] = rare_cats

    # Fill nulls → '__NULL__'
    for col in encode_cols:
        if dataframe[col].isnull().any():
            dataframe[col] = dataframe[col].fillna("__NULL__")
    dataframe[encode_cols] = dataframe[encode_cols].astype(str)

    # Fit & transform
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                        drop="first" if drop_first else None)
    encoded = ohe.fit_transform(dataframe[encode_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(encode_cols),
                              index=dataframe.index)
    result = pd.concat([encoded_df, dataframe[[target_col]]], axis=1)
    return result, encode_cols, ohe, rare_map_store""",
        note="Produces binary columns like 'genero_M', 'platam_score_bin_3.0'. Encoder saved for deployment reuse.")

    # ── 4. train_and_evaluate ─────────────────────────────────
    _content_slide(prs, "3. Training Function — train_and_evaluate()", [
        "**Purpose:** End-to-end training, feature selection, and evaluation in one call.",
        "",
        "**Flow:**",
        "  1. Stratified train/test split (80/20 by default)",
        "  2. XGBoost with GridSearchCV (5-fold stratified CV, scoring = roc_auc)",
        "  3. Compute XGB gain importance + permutation importance (10 repeats)",
        "  4. Auto-flag suspicious features: high gain but low perm importance "
            "(may indicate data leakage or noise overfitting)",
        "  5. Drop __NULL__ OHE features with negligible gain AND perm importance "
            "(toggled by DROP_NULL_FEATURES)",
        "  6. Retrain on cleaned feature set",
        "  7. Evaluate: Accuracy, Balanced Accuracy, F1, AUC, Gini",
        "",
        "**Returns:** Model, train/test sets, metrics, params, feature drop audit trail, "
            "importance Series (gain + perm).",
    ])

    # ── Code: train_and_evaluate — signature + GridSearchCV ───
    _code_slide(prs, "Code: train_and_evaluate() — Signature & GridSearchCV", """\
def train_and_evaluate(df, target_col, test_size, random_state, cv_folds, scoring,
                       param_grid, gain_top_pct, perm_min_threshold,
                       add_back_features, extra_drop_features, label="",
                       drop_null_features=False, null_gain_thr=0.001,
                       null_perm_thr=0.0005):
    \"\"\"Train XGBoost with GridSearchCV, evaluate, and return results dict.\"\"\"

    # Stratified train / test split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    # XGBoost with auto class-imbalance handling
    neg, pos = np.bincount(y_train.astype(int))
    scale_pos_weight = neg / pos if pos > 0 else 1

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight, use_label_encoder=False,
        eval_metric="logloss", random_state=random_state, verbosity=0)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=xgb_base, param_grid=param_grid, scoring=scoring,
        cv=skf, n_jobs=-1, verbose=1, refit=True)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_""",
        note="scale_pos_weight auto-handles class imbalance. GridSearchCV tries all param combinations with 5-fold CV.")

    # ── Code: train_and_evaluate — importance & auto-flag ─────
    _code_slide(prs, "Code: train_and_evaluate() — Feature Importance & Auto-Flag", """\
    # Feature importance & auto-flag
    imp = pd.Series(best_model.feature_importances_, index=X_train.columns)    # XGB gain
    perm_result = permutation_importance(
        best_model, X_test, y_test, n_repeats=10,
        random_state=random_state, scoring="roc_auc", n_jobs=-1)
    perm_imp = pd.Series(perm_result.importances_mean, index=X_test.columns)   # perm importance

    # Suspicious: high gain but low permutation importance
    gain_cutoff = np.percentile(imp.values, 100 - gain_top_pct)   # top 50% by default
    suspicious = imp.index[(imp >= gain_cutoff) & (perm_imp < perm_min_threshold)].tolist()

    auto_drop = [f for f in suspicious if f not in add_back_features]

    # Drop __NULL__ OHE features with negligible importance
    null_features_dropped = []
    if drop_null_features:
        null_feats = [f for f in imp.index if "__NULL__" in f]
        null_low = [f for f in null_feats
                    if imp[f] < null_gain_thr and perm_imp.get(f, 0) < null_perm_thr]
        null_features_dropped = [f for f in null_low if f not in add_back_features]

    final_drop = list(set(auto_drop + extra_drop_features + null_features_dropped))
    if final_drop:
        X_train = X_train.drop(columns=final_drop, errors="ignore")
        X_test  = X_test.drop(columns=final_drop, errors="ignore")
        best_model.fit(X_train, y_train)    # retrain on cleaned feature set""",
        note="Two-pass approach: train → measure importance → prune → retrain. Ensures clean final model.")

    # ── Code: train_and_evaluate — evaluation ─────────────────
    _code_slide(prs, "Code: train_and_evaluate() — Evaluation Metrics & Return", """\
    # Evaluation metrics
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy":          accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "F1 Score":          f1_score(y_test, y_pred),
        "AUC (ROC)":         roc_auc_score(y_test, y_prob),
        "Gini":              2 * roc_auc_score(y_test, y_prob) - 1,
    }

    return {
        "best_model": best_model,
        "X_train": X_train,  "X_test": X_test,  "y_test": y_test,
        "metrics": metrics,
        "best_params": grid_search.best_params_,
        "feature_drop_info": {
            "auto_flagged_suspicious": suspicious,
            "added_back": add_back_features,
            "extra_manual_drops": extra_drop_features,
            "null_features_dropped": null_features_dropped,
            "final_dropped": final_drop,
        },
        "grid_search": grid_search,
        "xgb_gain": imp,
        "perm_importance": perm_imp,
        "gain_cutoff": gain_cutoff,
        "perm_min_threshold": perm_min_threshold,
    }""",
        note="Gini = 2*AUC - 1. feature_drop_info provides full audit trail of every drop decision.")

    # ── 5. User Configuration Parameters ──────────────────────
    param_rows = [
        ["DATA_PATH", "Path to input CSV file", "str"],
        ["TARGET_COL", "Name of the binary target column", "str"],
        ["HOLDOUT_PCT", "Fraction of data reserved for true out-of-sample test", "0.15"],
        ["NULL_THRESHOLD_PCT", "Max % nulls before a column is dropped", "40"],
        ["VARIATION_THRESHOLD", "Cramer's-V below which columns are flagged", "0.02"],
        ["DROP_LOW_VARIATION", "Manually confirmed list of columns to drop", "list"],
        ["CAT_NUNIQUE_THRESHOLD", "Max unique values to classify as categorical", "20"],
        ["FORCE_NUMERIC", "Columns forced to numerical despite low cardinality", "list"],
        ["N_BINS", "Number of bins for KBinsDiscretizer", "5"],
        ["BINNING_STRATEGY", "Binning strategy (quantile / uniform / kmeans)", "quantile"],
        ["CUSTOM_BINS", "Per-column custom bin edges (dict)", "{}"],
        ["BINNING_FALLBACK", "Behaviour on binning failure: 'categorical' or 'drop'", "categorical"],
        ["DROP_FIRST", "Whether OHE drops the first category (avoids multicollinearity)", "False"],
        ["MAX_CATEGORIES", "Minimum frequency to keep a category (else -> __rare__)", "5"],
        ["MODE", "Column selection mode: 'keep' (whitelist) or 'drop' (blacklist)", "keep"],
        ["DROP_NULL_FEATURES", "Toggle: drop __NULL__ OHE features with negligible importance", "True"],
    ]
    _table_slide(prs,
        "4. Configuration Parameters — Data & Preprocessing",
        ["Parameter", "Description", "Default"],
        param_rows,
        col_widths=[Inches(3), Inches(7), Inches(2)],
    )

    # ── Code: Configuration block ─────────────────────────────
    _code_slide(prs, "Code: User Configuration Block", """\
# ── Data ──────────────────────────────────────────────────────
DATA_PATH   = "data/all_data.csv"
TARGET_COL  = "y"
HOLDOUT_PCT = 0.15

# ── Null / Variation thresholds ───────────────────────────────
NULL_THRESHOLD_PCT  = 40
VARIATION_THRESHOLD = 0.02
DROP_LOW_VARIATION  = ["col_a", "col_b", ...]   # manually confirmed drops

# ── Column types ──────────────────────────────────────────────
CAT_NUNIQUE_THRESHOLD = 20
FORCE_NUMERIC         = []

# ── Binning ───────────────────────────────────────────────────
N_BINS            = 5
BINNING_STRATEGY  = "quantile"
CUSTOM_BINS       = {}           # e.g. {"age": [0, 18, 30, 50, 65, 100]}
BINNING_FALLBACK  = "categorical"

# ── OHE ───────────────────────────────────────────────────────
DROP_FIRST     = False
MAX_CATEGORIES = 5               # categories with < 5 rows -> '__rare__'
MODE           = "keep"

# ── __NULL__ feature handling ─────────────────────────────────
DROP_NULL_FEATURES  = True
NULL_GAIN_THRESHOLD = 0.001
NULL_PERM_THRESHOLD = 0.0005""",
        note="All parameters are at the top of train_pipeline.py. Change these to tune the pipeline.")

    # ── 6. Hyperparameters (GridSearchCV) ─────────────────────
    hp_rows = [
        ["n_estimators", "Number of boosting rounds (trees)", "[100, 200, 300]"],
        ["max_depth", "Max tree depth (controls complexity)", "[3, 5, 7]"],
        ["learning_rate", "Step-size shrinkage per round", "[0.01, 0.05, 0.1]"],
        ["subsample", "Fraction of rows used per tree", "[0.8, 1.0]"],
        ["colsample_bytree", "Fraction of features used per tree", "[0.8, 1.0]"],
        ["min_child_weight", "Min observations in a leaf node", "[1, 3, 5]"],
        ["scale_pos_weight", "Auto-computed: neg / pos ratio (handles class imbalance)", "auto"],
    ]
    _table_slide(prs,
        "5. XGBoost Hyperparameters — Grid Search Space",
        ["Hyperparameter", "What it controls", "Search Grid"],
        hp_rows,
        col_widths=[Inches(3), Inches(5.5), Inches(3.5)],
    )

    # ── Code: PARAM_GRID ──────────────────────────────────────
    _code_slide(prs, "Code: Hyperparameter Grid & Feature Importance Config", """\
# ── GridSearchCV ──────────────────────────────────────────────
CV_FOLDS = 5
SCORING  = "roc_auc"

PARAM_GRID = {
    "n_estimators":     [100, 200, 300],    # more trees = slower but often better
    "max_depth":        [3, 5, 7],          # deeper = more complex; risk overfitting
    "learning_rate":    [0.01, 0.05, 0.1],  # lower = needs more trees but generalizes better
    "subsample":        [0.8, 1.0],         # row sampling per tree (regularization)
    "colsample_bytree": [0.8, 1.0],         # feature sampling per tree (regularization)
    "min_child_weight": [1, 3, 5],          # min obs per leaf (prevents overfitting)
}
# Total combinations: 3 * 3 * 3 * 2 * 2 * 3 = 324
# Each tested with 5-fold CV = 1,620 model fits per training call

# ── Feature importance ────────────────────────────────────────
GAIN_TOP_PCT       = 50      # suspicious if gain >= top 50th percentile
PERM_MIN_THRESHOLD = 0.001   # ... AND perm importance < this
ADD_BACK_FEATURES  = []      # override: never drop these
EXTRA_DROP_FEATURES = []     # hard-coded: always drop these""",
        note="324 combinations x 5 folds x 2 models = 3,240 total fits. This is the slowest step.")

    # ── 7. Feature Importance & Auto-Flag ─────────────────────
    _content_slide(prs, "6. Feature Importance — How Suspicious Features Are Detected", [
        "**Two independent importance measures are computed after training:**",
        "",
        "**1. XGBoost Gain Importance:**",
        "  - Built-in: how much each feature contributes to reducing the loss function",
        "  - Fast but can overweight high-cardinality or noisy features",
        "",
        "**2. Permutation Importance (AUC-based):**",
        "  - Shuffle each feature 10 times, measure the AUC drop",
        "  - Slower but more reliable; shows true predictive contribution",
        "",
        "**Auto-flag rule:**",
        "  Feature is 'suspicious' if: gain >= top-50-percentile AND perm importance < 0.001",
        "  (High gain but the model doesn't actually lose AUC when the feature is shuffled)",
        "",
        "**__NULL__ feature rule:**",
        "  OHE features ending in '__NULL__' are dropped if BOTH gain < 0.001 AND perm < 0.0005.",
        "  These represent 'the value was missing' indicators that add noise.",
        "",
        "**ADD_BACK_FEATURES:** Override list — features that should never be auto-dropped",
        "**EXTRA_DROP_FEATURES:** Hard-coded additional drops (domain knowledge)",
    ])

    # ── 8. Deployment / Scoring Pipeline ──────────────────────
    _content_slide(prs, "7. Deployment — score_new_data() Function", [
        "**Purpose:** Apply the exact same transformations to new data and produce predictions.",
        "",
        "**Artefacts loaded at scoring time:**",
        "  - bin_edges.json — Binning boundaries (same buckets as training)",
        "  - ohe_encoder.joblib — Fitted OneHotEncoder (same columns, same categories)",
        "  - rare_mappings.json — Which categories map to '__rare__'",
        "  - final_features.json — Exact list of features the model expects",
        "  - xgb_model.joblib — Trained XGBoost model",
        "",
        "**Flow:**",
        "  1. Read raw data, apply same null fills / rare mappings",
        "  2. Apply saved bin edges (pd.cut with same boundaries)",
        "  3. Apply saved OHE encoder (handle_unknown='ignore' for new categories)",
        "  4. Align columns to final_features (add missing as 0, drop extra)",
        "  5. Predict probabilities with saved model",
        "",
        "**Key guarantee:** Training and scoring transformations are identical — "
            "no train/serve skew.",
    ])

    # ── Code: score_new_data ──────────────────────────────────
    _code_slide(prs, "Code: score_new_data() — Deployment Scoring", """\
def score_new_data(raw_df, artefact_dir, ext_raw_df=None,
                   model_file="xgb_model.joblib",
                   features_file="final_features.json"):
    \"\"\"Score raw data using saved artefacts. Identical to deployment.\"\"\"

    # Load all saved artefacts
    model     = joblib.load(os.path.join(artefact_dir, model_file))
    ohe_enc   = joblib.load(os.path.join(artefact_dir, "ohe_encoder.joblib"))
    bin_edges = json.load(open(os.path.join(artefact_dir, "bin_edges.json")))
    col_meta  = json.load(open(os.path.join(artefact_dir, "column_metadata.json")))
    rare_map  = json.load(open(os.path.join(artefact_dir, "rare_mappings.json")))
    features  = json.load(open(os.path.join(artefact_dir, features_file)))

    # Apply same transformations as training:
    # 1. Fill nulls with '__NULL__', apply rare mappings
    for col in encode_cols:
        if col in rare_map:
            raw_df[col] = raw_df[col].apply(lambda x: "__rare__" if x in rare_map[col] else x)

    # 2. Apply saved bin edges (same boundaries)
    for col, info in bin_edges.items():
        raw_df[col + "_bin"] = pd.cut(raw_df[col], bins=info["edges"], labels=False)

    # 3. OHE transform (handle_unknown='ignore' for new categories)
    encoded = ohe_enc.transform(raw_df[encode_cols])

    # 4. Align to exact features the model expects
    for f in features:
        if f not in scored_df.columns: scored_df[f] = 0   # missing → 0
    scored_df = scored_df[features]

    # 5. Predict
    return model.predict_proba(scored_df)[:, 1]""",
        note="handle_unknown='ignore' means unseen categories in new data produce all-zero OHE columns (safe).")

    # ── 9. Artefact Directory ─────────────────────────────────
    artefact_rows = [
        ["bin_edges.json", "Bin boundaries for numerical columns"],
        ["ohe_encoder.joblib", "Fitted OneHotEncoder (sklearn)"],
        ["rare_mappings.json", "Category -> __rare__ mappings"],
        ["column_metadata.json", "Cat/num classification, all column lists"],
        ["column_tracking.json", "Full audit trail: which columns survived each step"],
        ["xgb_model_main_only.joblib", "Trained XGBoost (main features only)"],
        ["xgb_model.joblib", "Trained XGBoost (augmented: main + InsightGenie)"],
        ["final_features_main_only.json", "Feature list for main-only model"],
        ["final_features.json", "Feature list for augmented model"],
        ["test_metrics_*.json", "Dev test metrics (AUC, Gini, F1, etc.)"],
        ["predictor_report.csv", "Per-feature importance, status, drop reason"],
        ["column_trail_report.csv", "Per-OHE-feature boolean trail through every step"],
        ["column_trail_summary.csv", "Running counts at each pipeline step"],
        ["pipeline_timing.json", "Elapsed time per step and total runtime"],
        ["holdout_metrics_*.json", "Out-of-sample holdout metrics"],
    ]
    _table_slide(prs,
        "8. Artefact Directory — What Gets Saved",
        ["File", "Description"],
        artefact_rows,
        col_widths=[Inches(4.5), Inches(7.5)],
    )

    # ── 10. External / InsightGenie Pipeline ──────────────────
    _content_slide(prs, "9. InsightGenie (External) Features Pipeline", [
        "**Separate but identical preprocessing** to main columns:",
        "  - Own null threshold (EXT_NULL_THRESHOLD_PCT, default same as main)",
        "  - Own variation threshold (EXT_VARIATION_THRESHOLD)",
        "  - Own binning config (EXT_N_BINS, EXT_BINNING_STRATEGY, EXT_CUSTOM_BINS)",
        "  - Own OHE encoder (separate from main OHE)",
        "",
        "**Why separate?**",
        "  - External features may have different data quality / distributions",
        "  - Separate OHE prevents category leakage between main and external",
        "  - Allows tuning thresholds independently (stricter null threshold for external, etc.)",
        "",
        "**After preprocessing:**",
        "  - External OHE features are concatenated with main OHE features",
        "  - Both go into the AUGMENTED model together",
        "  - The MAIN-ONLY model never sees external features (clean comparison)",
        "",
        "**Current external columns:** Defined in EXT_COLUMNS list (rating, photo_count, "
            "industry_confidence, sentiment, trust_score, etc.)",
    ])

    # ── Code: Progress tracker ────────────────────────────────
    _code_slide(prs, "Code: PipelineProgress — Progress Bar & Timing", """\
class PipelineProgress:
    \"\"\"Lightweight progress bar + per-step timing tracker.\"\"\"

    STEPS = [
        "Load data & holdout split",  "Column selection",
        "Null analysis",              "Variation analysis",
        "Identify column types",      "Binning",
        "One-hot encoding",           "External columns pipeline",
        "Save transformation artefacts",
        "Train - Main model",         "Train - Augmented model",
        "Build reports & trail CSV",
    ]
    BAR_WIDTH = 40

    def start_step(self, step_idx):
        \"\"\"Begin a numbered pipeline step.\"\"\"
        if self._step_start is not None and self.current < len(self.STEPS):
            self.step_times[self.STEPS[self.current]] = time.time() - self._step_start
        self.current = step_idx
        self._step_start = time.time()
        self._print_bar()

    def _print_bar(self):
        done = self.current
        filled = int(self.BAR_WIDTH * (done / self.total))
        bar = "\\u2588" * filled + "\\u2591" * (self.BAR_WIDTH - filled)
        elapsed = time.time() - self._pipeline_start
        eta = elapsed / done * (self.total - done) if done > 0 else 0
        print(f"  [{bar}] {done}/{self.total}  ETA {eta:.0f}s", flush=True)

    # Output example:
    #   [████████████████░░░░░░░░░░░░░░░░░░░░░░░░]  5/12  ETA 3m 22s  │ Binning""",
        note="No external dependencies — pure Python. Timing saved to artefacts/pipeline_timing.json.")

    # ── 11. Suggestions & Best Practices ──────────────────────
    _content_slide(prs, "10. Suggestions & Best Practices", [
        "**For the team (development):**",
        "  - Run the pipeline with HOLDOUT_PCT=0.15 to always have an unseen test set",
        "  - Review column_trail_report.csv after every run to catch unexpected drops",
        "  - Use FORCE_NUMERIC for ordinal-looking columns (e.g., scores that appear categorical)",
        "  - Keep CUSTOM_BINS for domain-specific splits (e.g., age ranges, income brackets)",
        "  - Tune VARIATION_THRESHOLD carefully — too aggressive drops useful features",
        "",
        "**For the client (presentation):**",
        "  - Lead with AUC/Gini lift from augmentation (Main vs Augmented model)",
        "  - Show the column trail summary to explain feature reduction transparency",
        "  - Highlight permutation importance over gain (more interpretable for business)",
        "  - Use holdout metrics to demonstrate generalization (no overfitting)",
        "",
        "**Production readiness checklist:**",
        "  - Validate score_new_data() on holdout set matches holdout_metrics",
        "  - Monitor prediction distribution drift monthly",
        "  - Re-train quarterly or when AUC degrades > 0.02 on monitoring set",
        "  - Version artefacts directory with each retrain (artefacts_v1, _v2, ...)",
    ])

    # ── 12. Quick Reference — Config Cheat Sheet ─────────────
    _content_slide(prs, "11. Quick Reference — What to Change When", [
        "**'Model AUC is too low':**",
        "  - Expand PARAM_GRID (add more n_estimators, try learning_rate=0.005)",
        "  - Lower VARIATION_THRESHOLD to keep more features",
        "  - Check if good columns are in DROP_LOW_VARIATION (remove them)",
        "",
        "**'Too many features / model is slow':**",
        "  - Increase VARIATION_THRESHOLD to drop more weak features",
        "  - Lower MAX_CATEGORIES to collapse more rare categories",
        "  - Increase NULL_THRESHOLD_PCT to keep fewer sparse columns",
        "",
        "**'Augmented model is not helping':**",
        "  - Check EXT_COLUMNS — remove noisy external features",
        "  - Review predictor_report.csv for external features with 0 gain",
        "  - Try stricter EXT_VARIATION_THRESHOLD",
        "",
        "**'__NULL__ features dominating importance':**",
        "  - Ensure DROP_NULL_FEATURES=True",
        "  - Lower NULL_GAIN_THRESHOLD / NULL_PERM_THRESHOLD",
        "  - Investigate why so many nulls exist in the source data",
    ])

    # ── 13. Thank You ─────────────────────────────────────────
    _title_slide(prs, "Internal Pipeline Guide", "For questions, refer to artefacts/ or this deck")

    # ── Save ──────────────────────────────────────────────────
    prs.save(INTERNAL_OUTPUT_PATH)
    print(f"Internal guide saved to: {INTERNAL_OUTPUT_PATH}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
    create_internal_deck()
