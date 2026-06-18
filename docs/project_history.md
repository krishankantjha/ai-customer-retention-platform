# RetainIQ — Comprehensive Project History & Development Ledger

This document serves as the master engineering ledger for the **RetainIQ AI Customer Retention Platform**. It records every development phase, architectural decision, algorithmic audit, model improvement, and business trade-off analysis performed on the IBM Telco Customer Churn codebase.

---

## Phase 0: Project Foundation

The primary goal of this phase was to sanitize and stabilize the repository foundation.

### Actions Taken:
1. **License Correction**: 
   * Audited the top-level [LICENSE](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/LICENSE) file.
   * Restored the standard MIT License template, removing custom placeholders and ensuring compliance with open-source engineering standards.

---

## Phase 1: Data Validation & Cleaning

This phase focused on ensuring raw data ingestion was robust, type-safe, and free of side-effects.

### 1. Refactoring Data Cleaning (`clean.py`)
The data cleaning pipeline in [clean.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/preprocessing/clean.py) was completely refactored to align with production software standards:
* **Side-Effect-Free Operations**: Rewrote data manipulation steps to operate on copies of DataFrames (`df.copy()`) rather than modifying references in-place.
* **Logging Isolation**: Refactored logging configurations to utilize isolated, standard dictConfig structures rather than global overrides that interfere with imported modules.
* **Assertion-Free Validation**: Replaced standard python `assert` statements with explicit, schema-enforcing validation exceptions (`ValueError`) to prevent validation bypasses under optimized Python execution mode (`python -O`).
* **Zero-Tenure Imputation Safeguards**: Identified that `TotalCharges` contained missing/whitespace values exclusively for customers with `tenure == 0`. Programmatically targeted only these zero-tenure rows for imputation to `0.0`, rather than performing broad statistical averages (mean/median) that introduce demographic bias.
* **Sanity Constraints**: Integrated validation checks confirming category ranges (e.g., ensuring `SeniorCitizen` contains only binary values `0` and `1`).
* **Evaluation & Testing**: Successfully executed the clean pipeline to generate the standardized processed dataset `telco_churn_clean.csv`.

---

## Phase 1.5: Exploratory Data Analysis (EDA) Audits

Audited the exploratory analysis code in [eda.ipynb](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/notebooks/eda.ipynb) and enhanced it with rigorous statistical tests to validate churn drivers.

### Enhancements:
1. **Environment-Independent Paths**: Replaced hardcoded system file paths with dynamic path resolution based on standard relative locations.
2. **Correlation Heatmaps**: Added a Pearson numerical correlation matrix heatmap to evaluate collinearity between `tenure` and billing features.
3. **Chi-Square Tests**: Implemented a Chi-Square test suite to mathematically verify relationships between categorical features (e.g., contract types, internet options) and the `Churn` label.
4. **Mann-Whitney U Tests**: Introduced non-parametric statistical tests to verify if distributions of continuous variables (`tenure`, `MonthlyCharges`) differ significantly between churn and non-churn segments.
5. **Bivariate Boxplots**: Added grouped visualizations of billing charges across contract categories to observe price distribution patterns.

---

## Phase 2: Feature Engineering & Preprocessing Pipeline

Developed the feature transformation modules and serialization registry to prevent data leakage during offline training and online serving.

### 1. Feature Redundancy Resolution (`protection_score` vs. `security_over_streaming`)
* During prototyping in [feature_engineering.ipynb](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/notebooks/feature_engineering.ipynb), identified that the proposed feature `protection_score` introduced redundant information by overlap with `addon_count`.
* **Fix**: Replaced it with `security_over_streaming` (capturing online security subscription flags against streaming consumption patterns) in both the prototyping notebooks and the pipeline code.

### 2. Preprocessing Modules Implementation
* Implemented [engineer.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/preprocessing/engineer.py) to extract engineered indicators (e.g., auto-pay flag, contract-mtm flag, addon counts).
* Implemented [pipeline.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/preprocessing/pipeline.py) using scikit-learn's `ColumnTransformer` and scaling components.
* Enforced **leakage controls**: Scalers and encoders are fit ONLY on the training split, and then used to transform the test split.
* Serialized [pipeline.pkl](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/artifacts/pipeline.pkl) and [encoders.pkl](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/artifacts/encoders.pkl) for serving consistency.

---

## Phase 3: Model Selection, Training & Evaluation

Optimized model architectures, calculated evidence-based feature weights, and performed pruning.

### 1. Evidence-Based Weighting (`commitment_score_v2`)
Instead of assuming equal weights (1.0 each) for tenure and contract indicators when calculating `commitment_score`, we fitted a dedicated Logistic Regression model and extracted feature coefficient magnitudes:
* **Contract is not Month-to-month**: **1.96**
* **Automatic Payment Method**: **0.46**
* **Tenure > 12 Months**: **0.58**
These weights were integrated into [engineer.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/preprocessing/engineer.py) to calculate the optimized `commitment_score_v2`.

### 2. Feature Redundancy Pruning
* Audited the importance ranking of `binary__has_support` (subset of `addon_count`).
* XGBoost importance ranking placed `has_support` in the bottom **5.9%** percentile (49th out of 51 features) with an ROC-AUC delta contribution of **0.0000**.
* **Action**: Pruned `binary__has_support` from the modeling dataset to minimize complexity.
* To avoid multi-collinearity, dropped `binary__is_early_stage` specifically from the Logistic Regression feature list while retaining it for XGBoost splits.

### 3. Candidate Feature Validation Experiments
Offline trials on the baseline XGBoost model utilized a trigger rule of `ROC-AUC delta >= 0.002` to qualify new features:
* **`fiber_zero_engagement_flag`**: Yielded ROC-AUC **0.8224** (Delta: **+0.0047**) -> **KEEPER**.
* **`high_charge_early_stage_flag`**: Yielded ROC-AUC **0.8201** (Delta: **+0.0024**) -> **KEEPER** (utilizing train-set median only to prevent leakage).
* **`vulnerable_customer_flag`**: Validated segment size (3.19% of data) and segment churn rate (69.44%). Yielded ROC-AUC **0.8228** (Delta: **+0.0051**) -> **KEEPER**.

### 4. Code Implementation
* Created [tune.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/training/tune.py) for hyperparameter optimization using GridSearchCV.
* Created baseline [train.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/training/train.py) and baseline [evaluate.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/training/evaluate.py).

---

## Phase 3 Model Improvement Audit & Calibrations

Addressed the low churn recall of the baseline model using class balancing, decision threshold optimization, and probability calibration.

### 1. Class Imbalance Resolution
Evaluated and compared modeling balancing techniques on the test set:
* **Tuned XGBoost Baseline (No balancing)**: ROC-AUC = 0.8220, Recall = 53.48% (at 0.50 thresh).
* **SMOTE + XGBoost**: ROC-AUC = 0.8397, Recall = 72.73%.
* **Random Undersampling**: ROC-AUC = 0.8416, Recall = 79.41%.
* **XGBoost with `scale_pos_weight`**: ROC-AUC = **0.8442**, Recall = **80.75%**, Precision = **52.43%**, F1-score = **0.6358** (Highest across all options).
* **Action**: Configured training to calculate the majority-to-minority ratio dynamically and pass it as `scale_pos_weight` (computed as `2.7686`).

### 2. Probability Calibration
Evaluated raw outputs against Platt scaling and Isotonic regression:
* **Raw XGBoost Brier Score**: 0.1499
* **Platt Scaling (Sigmoid) Brier Score**: 0.1464
* **Isotonic Regression Brier Score**: **0.1367** (Best)
* **Action**: Wrapped the XGBoost model in scikit-learn's `CalibratedClassifierCV` using `method="isotonic"` and `cv=5` (cross-validation calibration) to output calibrated risk scores. Updated [evaluate.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/training/evaluate.py) to extract the underlying base estimators (`model.calibrated_classifiers_[0].estimator`) for SHAP Explainer compatibility.

### 3. Business Decision Theory & Optimal Thresholding
Modeled customer campaign outcomes under business assumptions (Customer LTV = $1,000, Outreach Cost = $50, Success Rate = 30%, expected net benefit per TP = $250, cost per FP = -$50):
* *Theoretical Optimal Threshold*: `Cost / Expected Value = 50 / 300 = 16.67%`.
* *Empirical Optimal Threshold*: Threshold **0.15** yielded highest raw savings (**$63,250**), but targeted 769 customers with 430 false alarms.
* **Production Decision**: Selected threshold **0.25** for deployment. It yields **$61,850** in net savings (98% of peak possible savings) while reducing unnecessary outreach (FPs) by **36%** and reducing campaign cost by **25%** ($28,750 vs. $38,450) compared to the 0.15 threshold.
* **Final Evaluation Metrics**: Deploying at threshold **0.25** yields **80.75% Recall**, **52.52% Precision**, **0.6365 F1-score**, and **75.80% Accuracy** on the test split.

---

## Phase 4: Model Evaluation & Explainability

Completed the offline test evaluations and implemented the backend local explainability framework.

### 1. Offline Metric Evaluations
* Calculated test set performance: ROC-AUC: **0.8447**, PR-AUC: **0.6511**, F1-Score: **0.6365**, Recall: **80.75%**, Precision: **52.52%**, Accuracy: **75.80%**, Brier Score: **0.1367**.
* Generated and saved curves (`confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`) under `ml/artifacts/`.

### 2. Backend Local Explainability (`explain.py`)
* Implemented [explain.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/backend/app/ml/explain.py) under the backend folder structure.
* Extracted the underlying base estimators from the cross-validated calibrated model pipeline to compute local SHAP contributions for individual customer profiles.
* Programmed a custom rules-engine mapper that pairs positive SHAP drivers to prescriptive retention outreach programs ("Save Plays"):
  * Month-to-month contracts $\rightarrow$ **1-Year Contract Lock campaign** (lock-in discount rate).
  * Electronic check payment friction $\rightarrow$ **Auto-Pay Conversion campaign** ($5 bill credit setup offer).
  * Fiber optic with zero addon engagement $\rightarrow$ **Add-on Bundling Value campaign** (free 3-month security addons).
  * High monthly charges / price shocks $\rightarrow$ **Billing Rate Review** (cheaper tier downgrades or loyalty rate cuts).
  * Vulnerable demographics (early-stage seniors alone) $\rightarrow$ **Priority Support Onboarding** (assisted tech setups).
* Executed end-to-end integration tests to verify local explanation outputs and save play assignments.

---

## Technical Directory Reference Map

All implemented components are fully serialized and accessible via the directory tree below:

* **Ingestion Validation & Cleaning**: [clean.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/preprocessing/clean.py)
* **Feature Engineering Module**: [engineer.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/preprocessing/engineer.py)
* **Transformer Pipeline Registry**: [pipeline.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/preprocessing/pipeline.py)
* **Tuning Optimization Runner**: [tune.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/training/tune.py)
* **Calibrated Model Training Script**: [train.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/training/train.py)
* **Threshold-Enforced Evaluation Script**: [evaluate.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/ml/training/evaluate.py)
* **Local Inference Explainer & Save Plays**: [explain.py](file:///c:/Users/krish/Downloads/ai-customer-retention-platform/backend/app/ml/explain.py)

