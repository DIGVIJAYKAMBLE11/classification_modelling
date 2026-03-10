# Scorecard Developer — Complete Interview & Role Preparation Guide

> Tailored to the pipeline in `train_pipeline.py` / `classification_pipeline.ipynb`
> Dataset: `final_enriched_dataset.csv` (Colombia SME credit scoring)

---

## TABLE OF CONTENTS

1. [Keyword Glossary](#1-keyword-glossary)
2. [Metrics Reference](#2-metrics-reference)
3. [SQL Refresher — Beginner to Professional](#3-sql-refresher)
4. [Tech Stack Modules — Beginner to Professional](#4-tech-stack-modules)
5. [2-Day Preparation Plan](#5-2-day-preparation-plan)
6. [Resume Pointers](#6-resume-pointers)

---

## 1. KEYWORD GLOSSARY

Every term that appears in the job description, defined precisely in a credit scoring context.

---

### A

**Application Scoring**
A scorecard built at the point of a credit application. Uses information available at origination: bureau data, application form fields, identity data, and sometimes alternative data. Predicts whether an applicant will default within a defined performance window (e.g., 12 months). In this pipeline, features like `experian_score`, `queries_6m`, `edad`, `ingresos_smlv` are all application-time variables.

**Artefact (Model Artefact)**
Any file produced during training that is required to reproduce or serve predictions in production. Examples: `xgb_model.joblib`, `bin_edges.json`, `ohe_encoder.joblib`, `final_features.json`. These must be version-controlled and paired with the training code for reproducibility.

**Approval Rate**
The proportion of applicants who receive a credit offer. Computed as: `approved / total_applicants`. A scorecard cut-off is set to balance approval rate against expected bad rate.

---

### B

**Bad Rate**
The proportion of accounts that become "bad" (i.e., default) within the performance window. For a given score band: `bad_rate = bads / (goods + bads)`. In the pipeline, `target = 1` marks a bad account. The target variable is derived from `target_dpd0`.

**Behavioural Scoring (Behavioural Scorecard)**
A scorecard applied to existing customers using repayment history, transaction patterns, and bureau updates. Differs from application scoring because it uses time-series on-book data. `platam_score` and `hybrid_score` in the dataset are examples of behavioural risk signals.

**Binning / Supervised Binning**
Converting a continuous variable into discrete buckets. *Supervised* binning aligns bin boundaries with bad rate patterns (e.g., using decision trees or quantiles that maximise IV). In the pipeline, `KBinsDiscretizer` with `strategy="quantile"` is used, creating 5 equal-frequency bins. Binning improves interpretability, handles outliers, and enables WoE transformation downstream.

**Bureau Data**
Credit information provided by a credit reference agency (CRA) such as Experian, TransUnion, or Equifax. Includes credit history, outstanding balances, delinquency flags, and enquiry counts. In the dataset: `experian_score`, `queries_6m`, `queries_12m`, `active_credits`, `closed_credits`, `total_debt`, `total_arrears_balance`, `negative_entities`.

---

### C

**Calibration (PD Calibration)**
The process of mapping a model's raw score or predicted probability to an observed bad rate. A well-calibrated model produces predicted PDs that closely match actual default rates across the score distribution. Methods include isotonic regression, Platt scaling, or binned actual-vs-predicted plots.

**Cohort / Vintage Analysis**
Grouping accounts by their origination period (e.g., month of booking) to track how each cohort matures over time. Useful for detecting if model performance degrades in newer cohorts.

**Cramér's V**
A measure of association between two categorical variables, derived from the chi-squared statistic. Used in the pipeline's `variation_analysis()` function to assess how much a categorical feature varies by target class. Range: 0 (no association) to 1 (perfect association). Feature flagged if Cramér's V < `VARIATION_THRESHOLD` (0.02).

**Credit Decisioning**
The automated or semi-automated process of making a lending decision (approve/decline/refer) based on risk score, credit policy rules, and affordability checks.

**Cross-Validation (Stratified K-Fold)**
A model validation technique splitting training data into K folds, training on K-1 and validating on the held-out fold. *Stratified* preserves the target class ratio in each fold — critical for imbalanced credit datasets. Used in the pipeline with `StratifiedKFold(n_splits=5)`.

**Cut-off / Decision Threshold**
The score or probability above/below which an application is approved or declined. Moving the cut-off shifts the approval rate vs bad rate trade-off. Setting it requires business context (loss appetite, volume targets).

---

### D

**Data Pipeline**
The end-to-end sequence of steps that transforms raw data into model-ready features and then into scored outputs. In the pipeline: load → holdout split → column selection → null analysis → variation analysis → binning → OHE → model → evaluate.

**Data Quality**
Fitness of data for its intended use. Key dimensions: completeness (nulls), consistency (same variable defined the same way across sources), accuracy (matches ground truth), timeliness (data lag).

**DPD (Days Past Due)**
The number of days a payment is overdue. `DPD0` = any missed payment. `DPD30`, `DPD60`, `DPD90` are common performance definitions. The target variable `target_dpd0` uses a DPD0 definition of default.

**Drift (Feature/Score Drift)**
A shift in the statistical distribution of an input feature or model score over time. Causes model performance to degrade. Measured via PSI (Population Stability Index). Requires scheduled monitoring and potential model recalibration or rebuild.

---

### E

**Edge-Case Handling**
Logic to handle unusual or boundary input values in production. Examples: null values for `experian_score` (applicant has no bureau file), extreme outliers in `total_debt`, or unknown categories in `departamento` not seen in training. Production specs must explicitly define fallback values for every edge case.

**Ensemble Methods**
Machine learning models that combine predictions from multiple base learners to improve accuracy and robustness. Key examples: Random Forest (bagging), XGBoost / LightGBM (gradient boosting), Stacking. XGBoost (`XGBClassifier`) is the core model in this pipeline.

---

### F

**Feature Catalogue**
A structured document listing every input feature: its name, source system, definition, transformation logic, missing value handling, and known data quality issues. Required for model governance and audit.

**Feature Engineering**
The process of creating, transforming, and selecting variables from raw data to maximise predictive power. Examples: creating `ratio_cuota_ingreso` (monthly repayment / monthly income), binning `edad`, encoding `departamento`, generating enquiry recency ratios.

**Feature Importance**
A metric indicating how much each feature contributes to the model's predictions. In XGBoost, measured as "gain" (average improvement in loss reduction). The pipeline also uses permutation importance (model-agnostic) to cross-validate gain-based rankings and flag suspicious features.

**Feature Selection**
Reducing the feature set to those most predictive and stable, removing noise, correlated variables, and data leakage. The pipeline auto-flags features with high gain but low permutation importance as suspicious.

**Feature Store**
A centralised repository where computed features are stored, versioned, and served both for training and real-time inference. Ensures training-serving consistency (avoids train-serve skew).

**Feature Specification**
An implementation document that exactly defines how each feature is computed in production: SQL logic, transformation steps, edge cases, and expected data types.

---

### G

**Gini Coefficient**
A model discrimination metric: `Gini = 2 × AUC − 1`. Range: 0 (no discrimination) to 1 (perfect discrimination). Equivalent to the area between the Lorenz curve and the line of equality. Used in credit scoring as the primary discrimination KPI (often called "Gini" rather than AUC).

**Governance (Model Risk Governance)**
The policies, procedures, and controls ensuring models are developed, validated, approved, and monitored appropriately. Includes model documentation standards, independent validation, change control logs, and periodic review schedules.

---

### H

**Holdout Set**
A portion of data withheld entirely from training and validation, used only for final unbiased evaluation. In the pipeline: `HOLDOUT_PCT = 0.15`, split with `train_test_split(..., stratify=...)` before any preprocessing to prevent data leakage.

**Hyperparameter Optimisation**
Searching for the model configuration (not learned from data) that maximises validation performance. In the pipeline: `GridSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight` scored by ROC-AUC.

---

### I

**Information Value (IV)**
Measures predictive power of a feature based on WoE. `IV = Σ (P(Goods)_i − P(Bads)_i) × WoE_i`. Ranges: <0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, 0.3–0.5 strong, >0.5 suspicious (possible leakage).

**Interpretability / Explainability**
The degree to which a model's outputs can be understood and justified. Critical in credit scoring for regulatory compliance (e.g., providing decline reasons). SHAP values and WoE-based scorecards are common interpretability tools.

---

### K

**KS Statistic (Kolmogorov-Smirnov)**
Maximum separation between the cumulative distribution of goods and bads across the score range. `KS = max |CDF_goods − CDF_bads|`. Used in the pipeline's `variation_analysis()` for numerical features. Good models typically show KS > 20%.

---

### L

**Limit Strategy**
The process of determining how much credit to extend to approved applicants, using calibrated risk outputs to balance expected loss against revenue.

**Loss (Expected Loss)**
`EL = PD × LGD × EAD` where PD = Probability of Default, LGD = Loss Given Default, EAD = Exposure at Default. Scorecard developers focus primarily on PD.

---

### M

**Monitoring Pack**
A standardised report tracking model health metrics: score distribution shifts (PSI), feature stability (CSI), approval rate trends, bad rate by score band, vintage curves, and data pipeline health.

**Missing Value Logic**
Explicit rules for handling null inputs at inference time. In the pipeline: numerical nulls → excluded from binning (bin is null); categorical nulls → `"__NULL__"` category; production scorecards often assign a specific WoE or bin for missing.

**Model Validation (Independent)**
A review of the model by a team separate from development (second-line function). Assesses methodology, data quality, performance, stability, and compliance with policy.

---

### O

**OHE (One-Hot Encoding)**
Converting a categorical variable with N categories into N binary columns (or N-1 with `drop_first=True`). Used in the pipeline after binning. Each bin of a numerical variable becomes a categorical column, then is one-hot encoded.

**Out-of-Sample / Out-of-Time Validation**
Evaluation on data not used in training. *Out-of-sample* = held-out subset from same period (holdout set). *Out-of-time* = data from a later time period — stronger test of generalisability.

---

### P

**PD (Probability of Default)**
The probability that a borrower will default within a defined time horizon (typically 12 months). The raw output of a classification model (`.predict_proba()[:, 1]`) is the model's estimated PD for each application.

**Performance Window**
The observation period over which default is measured. E.g., if a loan is originated in Jan and the performance window is 12 months, you check whether the account went bad by Jan of the following year.

**Permutation Importance**
Model-agnostic feature importance: shuffle one feature's values and measure the drop in model performance. Features that cause large drops are important. Used in the pipeline alongside XGBoost's native gain importance to cross-validate.

**PSI (Population Stability Index)**
Measures the shift in a variable's distribution between two populations (e.g., training vs current). `PSI = Σ (Actual% − Expected%) × ln(Actual%/Expected%)`. PSI < 0.1: stable; 0.1–0.25: minor shift; >0.25: major shift, investigate.

---

### R

**Recalibration**
Updating the PD-to-score mapping to reflect changes in the population bad rate without rebuilding the full model. Typically done by fitting a new logistic calibration on recent data.

**Repayment Data**
Transaction-level records of payments made against a credit account. Includes payment dates, amounts, and missed payments. Source for behavioural features like `creditos_mora`, `hist_neg_12m`.

**Risk Signal**
Any variable that carries predictive information about default risk. May be from bureau, application, behavioural, or alternative data sources.

---

### S

**Scale Pos Weight**
An XGBoost parameter that handles class imbalance: `scale_pos_weight = count(negatives) / count(positives)`. Effectively up-weights the minority class (bads) during training, improving recall on bads.

**Scorecard**
Historically a linear model where each variable contributes a score (points) and the total score maps to a probability of default. Modern scorecards may use XGBoost or other ML methods but must remain interpretable, stable, and calibrated.

**Segmentation**
Dividing the portfolio into sub-populations with distinct risk characteristics to develop separate or adjusted models for each (e.g., new-to-credit vs credit-active, consumer vs SME).

**SHAP (SHapley Additive exPlanations)**
A game-theory-based method to explain individual predictions by attributing a contribution value to each feature for each prediction.

**Stability**
A model's ability to produce consistent performance over time, across cohorts, and across markets. A stable model's score distribution and bad rate by band do not shift dramatically as new data flows in.

**Stratified Split**
A train/test or train/validation split that preserves the proportion of the target class in each partition. Critical when the target is imbalanced (e.g., 5% bads).

---

### T

**Transactional Data**
Records of financial transactions (purchases, payments, transfers). Used to build behavioural signals such as spending patterns, repayment velocity, and cash flow estimates.

---

### V

**Variation Analysis**
Testing whether a feature shows meaningful difference between goods and bads. In the pipeline: for numerical features, KS statistic; for categorical features, Cramér's V. Features below `VARIATION_THRESHOLD = 0.02` are flagged.

**Vintage Analysis**
See *Cohort Analysis*. Tracks performance of accounts by origination month to understand maturation curves.

---

### W

**WoE (Weight of Evidence)**
Transforms a feature into a log-odds signal: `WoE_i = ln(Distribution_Goods_i / Distribution_Bads_i)`. Monotonic after supervised binning. Scales with IV. Standard in traditional scorecard development; less common in tree-based models but useful for monitoring.

---

## 2. METRICS REFERENCE

### Discrimination Metrics (How well the model separates goods from bads)

| Metric | Formula | Good Range | Notes |
|--------|---------|------------|-------|
| **Gini** | `2 × AUC − 1` | >0.30 | Primary KPI in credit. 0.45–0.65 is strong |
| **AUC-ROC** | Area under ROC curve | >0.65 | Same info as Gini |
| **KS Statistic** | `max\|CDF_goods − CDF_bads\|` | >0.20 | Higher = better separation |
| **F1 Score** | `2 × (P×R)/(P+R)` | Context-dependent | Harmonic mean of precision and recall |
| **Accuracy** | `(TP+TN)/(Total)` | Misleading if imbalanced | Use Balanced Accuracy for imbalanced data |
| **Balanced Accuracy** | `(Sensitivity+Specificity)/2` | >0.65 | Better for imbalanced credit datasets |
| **Precision** | `TP/(TP+FP)` | Context | Of predicted bads, how many are actually bad |
| **Recall / Sensitivity** | `TP/(TP+FN)` | Context | Of actual bads, how many are caught |

---

### Calibration Metrics (How accurate are the predicted PDs)

| Metric | Formula / Description | Good Range |
|--------|----------------------|------------|
| **Brier Score** | `mean((predicted_PD − actual)²)` | Lower is better; <0.05 for typical credit |
| **Log Loss** | `−mean(y×log(p) + (1−y)×log(1−p))` | Lower is better |
| **Hosmer-Lemeshow** | Chi-square test of calibration across deciles | p > 0.05 (not significant = well-calibrated) |
| **Reliability Diagram** | Plot of predicted PD vs actual bad rate by score band | Points on diagonal = well-calibrated |
| **Expected vs Actual Bad Rate** | By score decile/band | Should align within ±20% per band |

---

### Stability Metrics (Is the model still working over time)

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **PSI (score)** | `Σ(A% − E%) × ln(A%/E%)` | <0.10 stable; 0.10–0.25 minor shift; >0.25 rebuild |
| **CSI (feature)** | Same as PSI applied per feature | Identifies root cause of score drift |
| **Vintage Curve** | Bad rate by origination cohort over time | Consistent maturation patterns expected |
| **Approval Rate Trend** | Monthly approved% | Should be stable if cut-off fixed |
| **Bad Rate Trend** | Monthly bad rate of approved cohort | Increasing = model degrading or pop shift |

---

### Business / Decision Metrics

| Metric | Description | Used For |
|--------|-------------|---------|
| **Bad Rate by Score Band** | Bads/(Goods+Bads) per band | Cut-off setting, strategy design |
| **Lift / Odds** | How much better than random | Score band ranking quality |
| **Approval Rate vs Bad Rate Curve** | Trade-off at different cut-offs | Business strategy decisions |
| **Expected Loss (EL)** | PD × LGD × EAD | Portfolio risk quantification |
| **Risk-adjusted Return** | Revenue − Expected Loss | Profitability by score band |
| **Loss Given Default (LGD)** | Fraction of exposure lost if default | Combined with PD for EL |
| **Exposure at Default (EAD)** | Outstanding balance at default | Combined with PD for EL |

---

### Feature-Level Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **IV (Information Value)** | Predictive power of a feature | <0.02 useless; >0.50 suspicious |
| **WoE** | Log-odds of goods/bads per bin | Should be monotonic after supervised binning |
| **Cramér's V** | Categorical association with target | Pipeline flags < 0.02 |
| **KS per feature** | Separation of numeric feature by target class | Pipeline flags < 0.02 |
| **Null %** | Proportion of missing values | Pipeline drops > 40% |
| **PSI per feature (CSI)** | Distribution shift of a feature | <0.10 stable |

---

## 3. SQL REFRESHER

> All examples use the column vocabulary from `final_enriched_dataset.csv`.
> Assume a table named `credit_applications`.

---

### LEVEL 1 — FOUNDATIONS

#### SELECT, FROM, WHERE

```sql
-- Basic retrieval: all columns for applicants over 30
SELECT *
FROM credit_applications
WHERE edad > 30;

-- Specific columns only
SELECT cedula, edad, experian_score, target
FROM credit_applications
WHERE target = 1;  -- bad accounts only
```

#### Filtering with AND / OR / IN / BETWEEN / LIKE

```sql
-- Multiple conditions
SELECT cedula, experian_score, creditos_mora
FROM credit_applications
WHERE experian_score BETWEEN 400 AND 700
  AND creditos_mora > 0;

-- IN operator
SELECT cedula, departamento, target
FROM credit_applications
WHERE departamento IN ('BOGOTA', 'ANTIOQUIA', 'VALLE DEL CAUCA');

-- LIKE (pattern match)
SELECT cedula, clr_credit_study_result
FROM credit_applications
WHERE clr_credit_study_result LIKE 'APRO%';
```

#### ORDER BY, LIMIT

```sql
-- Top 10 riskiest applications by experian score ascending
SELECT cedula, experian_score, target
FROM credit_applications
ORDER BY experian_score ASC
LIMIT 10;
```

#### Aliases

```sql
SELECT
    cedula                        AS applicant_id,
    l_principal                   AS loan_amount,
    cuota_mensual / ingresos_smlv AS monthly_burden_ratio
FROM credit_applications;
```

---

### LEVEL 2 — AGGREGATIONS & GROUPING

#### COUNT, SUM, AVG, MIN, MAX

```sql
-- Bad rate by department
SELECT
    departamento,
    COUNT(*)                              AS total_applications,
    SUM(target)                           AS total_bads,
    ROUND(AVG(target) * 100, 2)           AS bad_rate_pct,
    ROUND(AVG(experian_score), 1)         AS avg_experian_score
FROM credit_applications
GROUP BY departamento
ORDER BY bad_rate_pct DESC;
```

#### GROUP BY with HAVING

```sql
-- Departments with > 100 applications and bad rate > 15%
SELECT
    departamento,
    COUNT(*)                        AS total,
    ROUND(AVG(target) * 100, 2)     AS bad_rate_pct
FROM credit_applications
GROUP BY departamento
HAVING COUNT(*) > 100
   AND AVG(target) > 0.15
ORDER BY bad_rate_pct DESC;
```

#### Conditional Aggregation (CASE WHEN)

```sql
-- Score band segmentation
SELECT
    CASE
        WHEN experian_score < 500  THEN 'Very High Risk'
        WHEN experian_score < 600  THEN 'High Risk'
        WHEN experian_score < 700  THEN 'Medium Risk'
        WHEN experian_score < 800  THEN 'Low Risk'
        ELSE                            'Very Low Risk'
    END                                          AS score_band,
    COUNT(*)                                     AS applications,
    SUM(target)                                  AS bads,
    ROUND(100.0 * SUM(target) / COUNT(*), 2)     AS bad_rate_pct
FROM credit_applications
GROUP BY 1
ORDER BY MIN(experian_score);
```

---

### LEVEL 3 — JOINS

#### INNER JOIN

```sql
-- Join application data with bureau enriched table
SELECT
    a.cedula,
    a.experian_score,
    b.total_debt,
    b.negative_entities,
    a.target
FROM credit_applications a
INNER JOIN bureau_data b
    ON a.cedula = b.cedula;
```

#### LEFT JOIN (keep all applicants even without bureau match)

```sql
SELECT
    a.cedula,
    a.edad,
    COALESCE(b.experian_score, -1)   AS experian_score,  -- -1 = no bureau file
    a.target
FROM credit_applications a
LEFT JOIN bureau_data b
    ON a.cedula = b.cedula;
```

#### Self-Join (e.g., compare applicant to their previous application)

```sql
SELECT
    a.cedula,
    a.application_date,
    a.experian_score                  AS current_score,
    prev.experian_score               AS prior_score,
    a.experian_score - prev.experian_score AS score_change
FROM credit_applications a
JOIN credit_applications prev
    ON a.cedula = prev.cedula
   AND prev.application_date = (
       SELECT MAX(application_date)
       FROM credit_applications
       WHERE cedula = a.cedula
         AND application_date < a.application_date
   );
```

---

### LEVEL 4 — WINDOW FUNCTIONS

Window functions are essential for credit risk analytics (vintage curves, running totals, lag features).

#### ROW_NUMBER, RANK, DENSE_RANK

```sql
-- Number applications per applicant in time order
SELECT
    cedula,
    application_date,
    experian_score,
    ROW_NUMBER() OVER (
        PARTITION BY cedula
        ORDER BY application_date
    ) AS application_sequence
FROM credit_applications;
```

#### LAG / LEAD (time series features)

```sql
-- Score at previous application (trend signal)
SELECT
    cedula,
    application_date,
    experian_score,
    LAG(experian_score, 1) OVER (
        PARTITION BY cedula
        ORDER BY application_date
    ) AS prior_experian_score,
    experian_score - LAG(experian_score, 1) OVER (
        PARTITION BY cedula
        ORDER BY application_date
    ) AS score_delta
FROM credit_applications;
```

#### Running Totals / Moving Averages

```sql
-- Cumulative bads by origination month
SELECT
    origination_month,
    COUNT(*)                                           AS apps,
    SUM(target)                                        AS bads,
    SUM(SUM(target)) OVER (ORDER BY origination_month) AS cumulative_bads
FROM credit_applications
GROUP BY origination_month
ORDER BY origination_month;
```

#### NTILE (Score Deciles)

```sql
-- Assign each application to a score decile
SELECT
    cedula,
    experian_score,
    target,
    NTILE(10) OVER (ORDER BY experian_score DESC) AS score_decile
FROM credit_applications;
```

---

### LEVEL 5 — CTEs AND SUBQUERIES

#### CTE (Common Table Expression)

```sql
-- Step 1: compute score decile; Step 2: aggregate bad rate per decile
WITH scored AS (
    SELECT
        cedula,
        experian_score,
        target,
        NTILE(10) OVER (ORDER BY experian_score DESC) AS decile
    FROM credit_applications
),
decile_stats AS (
    SELECT
        decile,
        COUNT(*)                                     AS applications,
        SUM(target)                                  AS bads,
        ROUND(100.0 * SUM(target) / COUNT(*), 2)     AS bad_rate_pct,
        ROUND(AVG(experian_score), 1)                AS avg_score
    FROM scored
    GROUP BY decile
)
SELECT *,
    SUM(applications) OVER ()                        AS total_applications,
    ROUND(100.0 * applications
          / SUM(applications) OVER (), 2)            AS pct_of_portfolio
FROM decile_stats
ORDER BY decile;
```

#### PSI Calculation in SQL

```sql
-- PSI between training period (2022) and monitoring period (2023)
WITH training AS (
    SELECT
        NTILE(10) OVER (ORDER BY experian_score) AS score_bin,
        COUNT(*) AS cnt
    FROM credit_applications
    WHERE origination_year = 2022
    GROUP BY 1
),
monitoring AS (
    SELECT
        NTILE(10) OVER (ORDER BY experian_score) AS score_bin,
        COUNT(*) AS cnt
    FROM credit_applications
    WHERE origination_year = 2023
    GROUP BY 1
),
combined AS (
    SELECT
        t.score_bin,
        t.cnt                                           AS train_cnt,
        m.cnt                                           AS monitor_cnt,
        t.cnt * 1.0 / SUM(t.cnt) OVER ()               AS train_pct,
        m.cnt * 1.0 / SUM(m.cnt) OVER ()               AS monitor_pct
    FROM training t
    JOIN monitoring m ON t.score_bin = m.score_bin
)
SELECT
    score_bin,
    ROUND(train_pct * 100, 2)                           AS train_pct,
    ROUND(monitor_pct * 100, 2)                         AS monitor_pct,
    ROUND((monitor_pct - train_pct)
          * LN(monitor_pct / train_pct), 4)             AS psi_component
FROM combined
UNION ALL
SELECT
    'TOTAL' AS score_bin,
    NULL, NULL,
    SUM((monitor_pct - train_pct) * LN(monitor_pct / train_pct))
FROM combined;
```

---

### LEVEL 6 — PROFESSIONAL / PRODUCTION SQL

#### Feature Engineering for Scorecard Training

```sql
-- Build model-ready feature set from raw tables
WITH base AS (
    SELECT
        a.cedula,
        a.application_date,
        a.target,
        -- Loan features
        a.l_principal,
        a.l_term,
        -- Bureau features
        COALESCE(b.experian_score, -999)                        AS experian_score,
        COALESCE(b.queries_6m, 0)                               AS queries_6m,
        COALESCE(b.queries_12m, 0)                              AS queries_12m,
        COALESCE(b.active_credits, 0)                           AS active_credits,
        COALESCE(b.total_debt, 0)                               AS total_debt,
        -- Derived ratios (feature engineering)
        CASE
            WHEN a.ingresos_smlv > 0
            THEN a.cuota_mensual / (a.ingresos_smlv * 1000)
            ELSE NULL
        END                                                     AS debt_to_income,
        CASE
            WHEN b.active_credits > 0
            THEN b.total_arrears_balance / NULLIF(b.active_credits, 0)
            ELSE 0
        END                                                     AS avg_arrears_per_credit,
        -- Enquiry velocity
        CASE
            WHEN COALESCE(b.queries_6m, 0) = 0 THEN 0
            ELSE COALESCE(b.queries_6m, 0) * 1.0 / NULLIF(COALESCE(b.queries_12m, 0), 0)
        END                                                     AS recent_query_ratio,
        -- Score bands
        CASE
            WHEN COALESCE(b.experian_score, -999) < 0   THEN 'no_bureau'
            WHEN b.experian_score < 500                  THEN 'very_high_risk'
            WHEN b.experian_score < 600                  THEN 'high_risk'
            WHEN b.experian_score < 700                  THEN 'medium_risk'
            ELSE                                              'low_risk'
        END                                                     AS experian_band
    FROM credit_applications a
    LEFT JOIN bureau_data b
        ON a.cedula = b.cedula
       AND b.snapshot_date = (
           SELECT MAX(snapshot_date)
           FROM bureau_data
           WHERE cedula = a.cedula
             AND snapshot_date <= a.application_date
       )
),
-- Bad rate by experian band (used for WoE calculation)
band_stats AS (
    SELECT
        experian_band,
        COUNT(*)                              AS total,
        SUM(target)                           AS bads,
        COUNT(*) - SUM(target)                AS goods
    FROM base
    GROUP BY experian_band
),
total_stats AS (
    SELECT SUM(bads) AS total_bads, SUM(goods) AS total_goods
    FROM band_stats
),
woe_calc AS (
    SELECT
        bs.experian_band,
        bs.total,
        bs.bads,
        bs.goods,
        ROUND(bs.bads * 100.0 / ts.total_bads, 4)   AS pct_bads,
        ROUND(bs.goods * 100.0 / ts.total_goods, 4) AS pct_goods,
        ROUND(
            LN(
                (bs.goods * 1.0 / ts.total_goods)
                / NULLIF(bs.bads * 1.0 / ts.total_bads, 0)
            ), 4
        )                                            AS woe,
        ROUND(
            ((bs.goods * 1.0 / ts.total_goods)
             - (bs.bads * 1.0 / ts.total_bads))
            * LN(
                (bs.goods * 1.0 / ts.total_goods)
                / NULLIF(bs.bads * 1.0 / ts.total_bads, 0)
            ), 4
        )                                            AS iv_component
    FROM band_stats bs
    CROSS JOIN total_stats ts
)
SELECT * FROM woe_calc ORDER BY woe DESC;
```

#### Vintage / Cohort Bad Rate Query

```sql
-- Vintage curve: bad rate by origination month and months-on-book (MOB)
SELECT
    DATE_TRUNC('month', a.application_date)              AS origination_month,
    EXTRACT(YEAR FROM AGE(p.snapshot_date, a.application_date)) * 12
    + EXTRACT(MONTH FROM AGE(p.snapshot_date, a.application_date)) AS mob,
    COUNT(*)                                              AS accounts,
    SUM(CASE WHEN p.dpd >= 30 THEN 1 ELSE 0 END)         AS bads_dpd30,
    ROUND(100.0 * SUM(CASE WHEN p.dpd >= 30 THEN 1 ELSE 0 END)
          / COUNT(*), 2)                                  AS bad_rate_pct
FROM credit_applications a
JOIN performance_snapshots p
    ON a.cedula = p.cedula
GROUP BY 1, 2
ORDER BY 1, 2;
```

#### Performance Window Definition

```sql
-- Label accounts: bad = ever DPD30+ within 12 months of origination
SELECT
    a.cedula,
    a.application_date,
    MAX(CASE
        WHEN p.snapshot_date BETWEEN a.application_date
                                 AND a.application_date + INTERVAL '12 months'
         AND p.dpd >= 30
        THEN 1
        ELSE 0
    END)  AS target_dpd30_12m
FROM credit_applications a
LEFT JOIN performance_snapshots p
    ON a.cedula = p.cedula
GROUP BY a.cedula, a.application_date;
```

---

## 4. TECH STACK MODULES

---

### MODULE A — PYTHON FOR CREDIT SCORING

#### A1. Foundations (Beginner)

```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("final_enriched_dataset.csv")

# Basic inspection
print(df.shape)
print(df.dtypes)
print(df.head())

# Target distribution
print(df['target'].value_counts(normalize=True))

# Null analysis
null_pct = df.isnull().mean().sort_values(ascending=False)
print(null_pct[null_pct > 0])
```

#### A2. Feature Engineering (Intermediate)

```python
# Debt-to-income ratio
df['dti'] = df['cuota_mensual'] / (df['ingresos_smlv'].replace(0, np.nan))

# Enquiry velocity (recent vs total)
df['query_recency_ratio'] = df['queries_6m'] / df['queries_12m'].replace(0, np.nan)

# Arrears intensity
df['arrears_per_credit'] = df['total_arrears_balance'] / df['active_credits'].replace(0, np.nan)

# Score band (binning with pd.cut)
df['experian_band'] = pd.cut(
    df['experian_score'],
    bins=[-np.inf, 500, 600, 700, 800, np.inf],
    labels=['very_high', 'high', 'medium', 'low', 'very_low']
)

# WoE encoding (manual)
def compute_woe(df, feature, target):
    total_bads = df[target].sum()
    total_goods = len(df) - total_bads
    stats = df.groupby(feature)[target].agg(['sum', 'count'])
    stats.columns = ['bads', 'total']
    stats['goods'] = stats['total'] - stats['bads']
    stats['pct_bads'] = stats['bads'] / total_bads
    stats['pct_goods'] = stats['goods'] / total_goods
    stats['woe'] = np.log(stats['pct_goods'] / stats['pct_bads'])
    stats['iv'] = (stats['pct_goods'] - stats['pct_bads']) * stats['woe']
    return stats
```

#### A3. Model Development (Intermediate-Advanced)

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import numpy as np

# Stratified holdout
df_dev, df_holdout = train_test_split(df, test_size=0.15, random_state=99, stratify=df['target'])

X = df_dev.drop('target', axis=1)
y = df_dev['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance
neg, pos = np.bincount(y_train)
spw = neg / pos

# XGBoost with cross-validated hyperparameter search
xgb = XGBClassifier(scale_pos_weight=spw, eval_metric='logloss', random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(xgb, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train, y_train)

print(f"Best AUC: {gs.best_score_:.4f}")
print(f"Gini: {2 * gs.best_score_ - 1:.4f}")
```

#### A4. Calibration (Advanced)

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# Isotonic regression calibration
calibrated_model = CalibratedClassifierCV(gs.best_estimator_, cv='prefit', method='isotonic')
calibrated_model.fit(X_test, y_test)

# Plot calibration curve
prob_true, prob_pred = calibration_curve(y_test, calibrated_model.predict_proba(X_test)[:, 1], n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='Calibrated XGBoost')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
plt.xlabel('Mean Predicted PD')
plt.ylabel('Fraction of Positives (Actual Bad Rate)')
plt.title('Calibration Curve')
plt.legend()
plt.show()
```

#### A5. PSI Monitoring (Professional)

```python
def compute_psi(expected, actual, bins=10):
    """Compute Population Stability Index."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _   = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected) + 1e-8
    actual_pct   = actual_counts   / len(actual)   + 1e-8

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    return psi_values.sum()

# Score distribution monitoring
train_scores = gs.best_estimator_.predict_proba(X_train)[:, 1]
holdout_scores = gs.best_estimator_.predict_proba(X_test)[:, 1]
psi = compute_psi(train_scores, holdout_scores)
print(f"Score PSI: {psi:.4f} — {'STABLE' if psi < 0.1 else 'MONITOR' if psi < 0.25 else 'REBUILD'}")
```

#### A6. SHAP Explainability (Professional)

```python
import shap

explainer = shap.TreeExplainer(gs.best_estimator_)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type='bar')

# Single prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

---

### MODULE B — SQL FOR CREDIT SCORING (see Section 3 above for detailed examples)

Key SQL skills for this role:
- **Basic**: SELECT, WHERE, GROUP BY, HAVING, CASE WHEN
- **Intermediate**: JOINs (LEFT, INNER), Subqueries, COALESCE/NULLIF
- **Advanced**: Window functions (LAG, ROW_NUMBER, NTILE), CTEs, PSI calculation
- **Professional**: Performance window labelling, vintage curve queries, WoE calculation in SQL, feature engineering from raw transaction tables

---

### MODULE C — MODEL MONITORING & GOVERNANCE

#### Monitoring Dashboard Components

```python
import pandas as pd
import matplotlib.pyplot as plt

def monthly_monitoring_report(df_train, df_monthly, score_col='predicted_pd', target_col='target'):
    report = {}

    # 1. PSI on score distribution
    report['score_psi'] = compute_psi(df_train[score_col], df_monthly[score_col])

    # 2. Bad rate by score band
    df_monthly['score_band'] = pd.cut(df_monthly[score_col], bins=10)
    report['bad_rate_by_band'] = df_monthly.groupby('score_band')[target_col].mean()

    # 3. Approval rate (assuming cut-off = 0.3)
    CUTOFF = 0.30
    report['approval_rate'] = (df_monthly[score_col] < CUTOFF).mean()

    # 4. Feature PSI for top features
    top_features = ['experian_score', 'queries_6m', 'dti', 'active_credits']
    report['feature_psi'] = {
        feat: compute_psi(df_train[feat].dropna(), df_monthly[feat].dropna())
        for feat in top_features
        if feat in df_train.columns
    }

    return report
```

---

### MODULE D — SCORECARD DEVELOPMENT WORKFLOW

The end-to-end workflow you must be able to narrate in an interview:

```
1. DEFINE THE PROBLEM
   - Performance window: 12 months post-origination
   - Bad definition: DPD30+ at any point in window
   - Population: SME applicants (this dataset)
   - Observation point: application date

2. DATA PREPARATION
   - Link applications to performance outcomes
   - Exclude exclusions (fraud, deceased, etc.)
   - Apply good/bad/indeterminate definitions

3. SAMPLE DESIGN
   - Development sample: oldest 70-80% of data
   - Out-of-time validation: most recent cohorts
   - Holdout: 15% stratified random sample (as in pipeline)

4. EXPLORATORY DATA ANALYSIS (EDA)
   - Missing rates, outliers, distributions
   - Bad rate by category for each variable
   - WoE monotonicity check after binning
   - IV ranking of all candidate features

5. FEATURE SELECTION
   - IV threshold (>0.02)
   - KS / Cramér's V (as in variation_analysis())
   - Correlation analysis (avoid redundant variables)
   - Business logic review

6. MODEL DEVELOPMENT
   - Supervised binning → WoE → Logistic Regression (traditional)
   - Or: Binning → OHE → XGBoost (as in this pipeline)
   - GridSearchCV with StratifiedKFold

7. VALIDATION
   - Dev test set (20% of dev sample)
   - Holdout (15% out-of-sample)
   - Out-of-time (most recent cohorts)
   - Metrics: Gini, KS, F1, Balanced Accuracy

8. CALIBRATION
   - Map model output PD to observed bad rate
   - Reliability diagram (predicted vs actual)
   - Calibrate by segment if needed

9. SCORECARD CONVERSION (for traditional scorecards)
   - Convert log-odds to points (PDO = Points to Double Odds)
   - Assign points per bin per variable
   - Total score maps to band maps to decision

10. IMPLEMENTATION
    - Feature specifications document
    - Artefact versioning (bin_edges.json, ohe_encoder.joblib, etc.)
    - Cut-off recommendation with approval/bad rate table

11. MONITORING
    - Monthly: PSI (score + features), bad rate by band
    - Quarterly: full performance report
    - Annual: model validation review
```

---

## 5. 2-DAY PREPARATION PLAN

---

### DAY 1 — TECHNICAL DEPTH

**Morning (4 hours): Core Credit Scoring Concepts**

| Time | Activity |
|------|----------|
| 09:00–10:00 | Read Section 1 (Keyword Glossary). Focus on: PD, calibration, WoE, IV, PSI, Gini, KS, bad definition, performance window |
| 10:00–11:00 | Read Section 2 (Metrics). Build intuition: why Gini over accuracy? When does AUC mislead? |
| 11:00–12:00 | Run `train_pipeline.py` locally (or `classification_pipeline.ipynb`). Trace each step. Note what each artefact does |
| 12:00–13:00 | Break |

**Afternoon (4 hours): Hands-on Code**

| Time | Activity |
|------|----------|
| 13:00–14:30 | Work through Section 4 Module A (Python). Run the WoE calculation, calibration curve, and PSI code on your dataset |
| 14:30–15:30 | Practice SQL: run the score band query and the WoE-in-SQL query (Section 3, Level 4–5) using SQLite or DuckDB on the CSV |
| 15:30–16:30 | Build the calibration curve from your holdout predictions using `calibration_curve` from sklearn |
| 16:30–17:30 | Write a 1-page "monitoring pack" template for the model in this pipeline. List: PSI score, PSI per feature, bad rate by band, approval rate |

---

### DAY 2 — APPLIED PRACTICE + INTERVIEW PREP

**Morning (4 hours): Applied Projects**

| Time | Activity |
|------|----------|
| 09:00–10:30 | Feature Engineering Sprint: create 5 new features from the dataset (DTI, query velocity, arrears per credit, bureau score trend, social score composite). Measure IV/KS for each |
| 10:30–11:30 | Rebuild pipeline with LightGBM. Compare Gini vs XGBoost. Tune `num_leaves`, `min_child_samples`. Document findings |
| 11:00–12:30 | PSI monitoring: split the dataset temporally (early cohorts = train distribution, later cohorts = monitoring). Compute score PSI and feature PSI. Flag any >0.25 |
| 12:30–13:30 | Break |

**Afternoon (4 hours): Interview Preparation**

| Time | Activity |
|------|----------|
| 13:30–14:30 | Practise STAR answers for: (1) a model you built end-to-end, (2) a data quality issue you solved, (3) a monitoring finding that led to action |
| 14:30–15:30 | Deep-dive on this pipeline: be ready to explain every parameter (`scale_pos_weight`, `StratifiedKFold`, `HOLDOUT_PCT`, `VARIATION_THRESHOLD`, permutation importance logic) |
| 15:30–16:30 | SQL practice: write the PSI query, vintage curve query, and WoE query from memory |
| 16:30–17:30 | Review your resume against the pointers in Section 6. Polish 3 bullet points. Prepare 2 questions to ask the interviewer |

---

### Key Questions to Prepare Answers For

1. Walk me through a scorecard you built end-to-end.
2. How do you define "bad" for a credit model? What trade-offs are involved?
3. How do you handle class imbalance in a credit dataset?
4. What is PSI and when would you trigger a model rebuild?
5. How do you calibrate a model's predicted PDs to observed bad rates?
6. What is WoE/IV and why is it used in credit scoring?
7. How do you select features? What is your process for dealing with correlated variables?
8. What's the difference between out-of-sample and out-of-time validation?
9. How would you monitor a model in production? What metrics do you track?
10. How do you explain a model decline reason to a customer or compliance team?
11. What is scale_pos_weight in XGBoost and why does it matter for credit?
12. How do you ensure training-serving consistency?

---

## 6. RESUME POINTERS

Tailor every bullet to the job description language. Use quantified outcomes wherever possible.

---

### STRONG RESUME BULLET EXAMPLES

Use this formula: **[Action verb] + [technique/tool] + [outcome / business impact]**

**Feature Engineering**
- Engineered 40+ features from bureau, transactional, and application data (debt-to-income ratios, enquiry velocity, arrears intensity), lifting model Gini from 0.38 to 0.51 on out-of-time validation
- Defined and documented production-ready feature specifications including null handling, edge-case logic, and consistency checks for 60+ variables deployed to real-time decisioning API

**Model Development**
- Developed XGBoost application scorecard for SME credit using supervised binning, OHE, and GridSearchCV with stratified 5-fold CV; achieved Gini 0.48 on 15% holdout set
- Implemented automated feature selection using gain-based and permutation importance cross-validation, reducing feature set by 35% without degrading AUC
- Built and validated end-to-end training pipeline with reproducible artefacts (binning edges, OHE encoder, feature registry, model weights) enabling one-command retraining

**Calibration**
- Produced PD calibrations using isotonic regression mapping model scores to observed bad rates by segment; validated via Hosmer-Lemeshow test and reliability diagrams
- Delivered approval-rate vs bad-rate trade-off tables at 10 cut-off thresholds, directly supporting credit strategy cut-off decisions

**Monitoring**
- Designed and maintained monthly monitoring pack tracking score PSI, feature CSI, vintage bad rate curves, and approval rate stability across 3 markets
- Identified early score drift (PSI 0.28) in new SME segment, triggering targeted recalibration that maintained bad rate within 15% of policy target

**SQL & Data**
- Authored production SQL feature pipeline extracting 50+ credit bureau and repayment signals from raw transaction tables, including performance window labelling and WoE computation
- Built vintage curve SQL framework enabling cohort-level bad rate tracking at monthly MOB intervals across multiple origination periods

**Governance & Documentation**
- Maintained feature catalogue (definitions, data source, transformation logic, missing value handling) for 60+ variables, referenced in model validation and regulatory audit
- Produced model evidence pack including methodology note, stability analysis, calibration report, and implementation specification

---

### KEYWORDS TO INCLUDE IN YOUR RESUME

Make sure these terms appear naturally:

- Application scoring / Behavioural scoring
- Credit bureau / Bureau data
- PD calibration / Probability of Default
- WoE (Weight of Evidence) / Information Value (IV)
- PSI (Population Stability Index) / CSI
- Gini / KS statistic / AUC-ROC
- Feature engineering / Feature selection / Feature specification
- XGBoost / LightGBM / Gradient Boosting
- GridSearchCV / Hyperparameter optimisation
- Stratified K-Fold / Out-of-time validation / Holdout evaluation
- Supervised binning / One-hot encoding
- Python (pandas, scikit-learn, XGBoost, SHAP)
- SQL (window functions, CTEs, aggregations)
- Model monitoring / Score distribution monitoring
- Credit decisioning / Cut-off strategy
- Class imbalance / scale_pos_weight / SMOTE
- Model governance / Model validation / Audit documentation

---

### RESUME STRUCTURE RECOMMENDATION

```
[Name] | Mumbai | LinkedIn | GitHub

SUMMARY
2-line sentence: X years in credit risk modelling, specialising in [scorecard development /
PD calibration / ML-based scoring], delivered [outcome] across [markets/products].

SKILLS
Languages: Python (pandas, scikit-learn, XGBoost, LightGBM, SHAP), SQL, R
Credit Analytics: Application Scoring, Behavioural Scoring, WoE/IV, PSI/CSI,
                  PD Calibration, Vintage Analysis, Cut-off Strategy
ML: XGBoost, LightGBM, Logistic Regression, Ensemble Methods, GridSearchCV, SHAP
Tools: Jupyter, Git, Airflow, dbt, BigQuery / Snowflake / Redshift

EXPERIENCE
[Role] | [Company] | [Dates]
• [Bullet 1 — feature engineering impact]
• [Bullet 2 — model development + Gini metric]
• [Bullet 3 — monitoring / PSI finding]
• [Bullet 4 — production / implementation]

PROJECTS
Credit Scoring Pipeline (this project): XGBoost scorecard with supervised binning,
OHE, 15% holdout evaluation. Gini: XX, KS: XX. Stack: Python, sklearn, XGBoost.

EDUCATION
```

---

### INTERVIEW MINDSET

1. **Lead with business impact** — Every technical answer should land on "...and this helped the business make better decisions / avoid loss / approve more good customers"
2. **Speak the lender's language** — Use "bads", "goods", "cut-off", "bad rate", "performance window", not just "class 1", "threshold", "precision window"
3. **Own the monitoring story** — Many candidates skip this. Knowing PSI, CSI, and vintage curves sets you apart
4. **Show production awareness** — Artefact versioning, training-serving consistency, feature specs — these show you've shipped, not just experimented
5. **Quantify everything** — "Gini improved from 0.38 to 0.51", "PSI dropped from 0.32 to 0.09 after recalibration", "15% holdout Gini within 2 points of dev test Gini"

---

*Guide built from `train_pipeline.py` and `classification_pipeline.ipynb` in this repository.*
*Dataset: SME credit scoring with bureau, application, behavioural, and alternative data features.*
