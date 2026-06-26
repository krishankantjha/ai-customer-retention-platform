"""
Feature Drift Detection.
Compares baseline training distributions against incoming production inference batches
using the two-sample Kolmogorov-Smirnov (KS) test for numerical features, and the
Chi-Square Goodness-of-Fit test combined with Population Stability Index (PSI)
for categorical and binary features.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chisquare

# Add project root to path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from configs.dataset_config import config_loader

logger = logging.getLogger("ml.training.feature_drift")


def _compute_chi_square_and_psi(train_series: pd.Series, inf_series: pd.Series, epsilon: float = 1e-4) -> dict:
    """
    Computes Chi-Square Goodness-of-Fit p-value and Population Stability Index (PSI)
    between baseline (train) and target (inference) categorical distributions.
    """
    train_clean = train_series.dropna()
    inf_clean = inf_series.dropna()
    
    all_categories = sorted(list(set(train_clean.unique()) | set(inf_clean.unique())))
    
    if not all_categories:
        return {
            "chi2_statistic": 0.0,
            "p_value": 1.0,
            "psi": 0.0,
            "drifted": False,
            "note": "No valid categories found"
        }
        
    n_train = len(train_clean)
    n_inf = len(inf_clean)
    
    if n_inf < 2 or n_train < 2:
        return {
            "chi2_statistic": 0.0,
            "p_value": 1.0,
            "psi": 0.0,
            "drifted": False,
            "note": "Insufficient samples for drift test"
        }
        
    train_counts = train_clean.value_counts()
    inf_counts = inf_clean.value_counts()
    
    train_probs = {cat: (train_counts.get(cat, 0) / n_train) for cat in all_categories}
    inf_probs = {cat: (inf_counts.get(cat, 0) / n_inf) for cat in all_categories}
    
    obs_counts = [inf_counts.get(cat, 0) for cat in all_categories]
    exp_counts = [n_inf * train_probs[cat] for cat in all_categories]
    exp_counts_reg = [max(val, epsilon) for val in exp_counts]
    
    try:
        chi2_res = chisquare(f_obs=obs_counts, f_exp=exp_counts_reg)
        chi2_stat = float(chi2_res.statistic)
        p_val = float(chi2_res.pvalue)
    except Exception as e:
        logger.debug(f"Chi-square calculation failed: {e}")
        chi2_stat = 0.0
        p_val = 1.0
        
    psi = 0.0
    for cat in all_categories:
        p_t = train_probs[cat]
        p_i = inf_probs[cat]
        
        p_t_smooth = (p_t + epsilon) / (1.0 + len(all_categories) * epsilon)
        p_i_smooth = (p_i + epsilon) / (1.0 + len(all_categories) * epsilon)
        
        psi += (p_i_smooth - p_t_smooth) * np.log(p_i_smooth / p_t_smooth)
        
    # PSI >= 0.25 is the standard MLOps threshold for significant distribution drift.
    # We use PSI >= 0.25 as the primary indicator for categorical features to avoid
    # false-positive alerts caused by statistical p-value sensitivity on high-imbalance inputs.
    is_drifted = psi >= 0.25
    
    return {
        "chi2_statistic": chi2_stat,
        "p_value": p_val,
        "psi": psi,
        "drifted": is_drifted
    }


def detect_feature_drift(X_inference: pd.DataFrame, random_seed: int = 42) -> dict:
    """
    Compares the feature columns of X_inference against the baseline training set.
    Automatically categorizes columns into numerical and categorical variables,
    applying KS-test and Chi-Square/PSI tests respectively.
    """
    logger.info("Starting enhanced feature drift detection check...")
    
    # 1. Load baseline training features
    config_train_path = config_loader.training["data_paths"]["train_features"]
    train_csv = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)
    
    if not os.path.exists(train_csv):
        logger.error(f"Training features CSV not found at: {train_csv}")
        raise FileNotFoundError(f"Training features CSV not found at: {train_csv}")
        
    df_train = pd.read_csv(train_csv)
    
    # 2. Extract continuous columns list from config (for primary KS checks)
    seg_cfg = config_loader.model.get("segmentation", {})
    cont_cols = seg_cfg.get("continuous_features", [
        "numeric__tenure",
        "numeric__MonthlyCharges",
        "numeric__addon_count",
        "numeric__commitment_score",
        "numeric__Contract",
        "numeric__AvgMonthlyCharge",
        "numeric__num_services"
    ])

    # Warn when expected post-pipeline numerical columns are absent from X_inference
    missing_from_inference = [col for col in cont_cols if col not in X_inference.columns]
    if missing_from_inference:
        logger.warning(
            f"SCHEMA MISMATCH: Expected post-pipeline continuous columns missing "
            f"from X_inference: {missing_from_inference}. "
            f"Ensure preprocessed features are passed, not raw input features."
        )

    # 3. Automatically detect feature types for common columns
    exclude = {"Churn", "customerID", "customer_id", "upload_id", "predicted_at", "id"}
    common_cols = [c for c in df_train.columns if c in X_inference.columns and c not in exclude]
    
    num_cols_to_check = []
    cat_cols_to_check = []
    
    for col in common_cols:
        if col in cont_cols or col.startswith("numeric__"):
            num_cols_to_check.append(col)
        elif col.startswith("categorical__") or col.startswith("binary__") or col.startswith("ordinal__"):
            cat_cols_to_check.append(col)
        else:
            # Type and cardinality-based auto-detection
            dtype = df_train[col].dtype
            cardinality = df_train[col].nunique(dropna=True)
            if pd.api.types.is_numeric_dtype(dtype) and cardinality > 10:
                num_cols_to_check.append(col)
            else:
                cat_cols_to_check.append(col)

    if not num_cols_to_check and not cat_cols_to_check:
        logger.warning("No columns found to run drift check on. Check features schema.")
        return {
            "is_drifted": False,
            "drift_ratio": 0.0,
            "metrics": {}
        }
        
    logger.info(
        f"Running drift checks: {len(num_cols_to_check)} continuous columns (KS-test), "
        f"{len(cat_cols_to_check)} categorical columns (Chi2 & PSI)."
    )
    
    metrics = {}
    drifted_count = 0
    
    # 4. Numerical KS Tests
    for col in num_cols_to_check:
        train_vals = df_train[col].values
        inf_vals = X_inference[col].values
        
        if len(inf_vals) < 2 or len(train_vals) < 2:
            metrics[col] = {
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "drifted": False,
                "note": "Insufficient samples for drift test",
                "method": "ks_test"
            }
            continue
            
        res = ks_2samp(train_vals, inf_vals)
        p_val = float(res.pvalue)
        ks_stat = float(res.statistic)
        is_col_drifted = p_val < 0.05
        
        if is_col_drifted:
            drifted_count += 1
            logger.warning(f"Drift detected on numerical feature '{col}': KS-stat={ks_stat:.4f}, p-val={p_val:.6f}")
        else:
            logger.debug(f"No drift on numerical feature '{col}': KS-stat={ks_stat:.4f}, p-val={p_val:.6f}")
            
        metrics[col] = {
            "ks_statistic": ks_stat,
            "p_value": p_val,
            "drifted": is_col_drifted,
            "method": "ks_test"
        }
        
    # 5. Categorical Chi-Square & PSI Tests
    for col in cat_cols_to_check:
        train_series = df_train[col]
        inf_series = X_inference[col]
        
        drift_res = _compute_chi_square_and_psi(train_series, inf_series)
        is_col_drifted = drift_res["drifted"]
        
        if is_col_drifted:
            drifted_count += 1
            logger.warning(
                f"Drift detected on categorical feature '{col}': "
                f"Chi2-p-val={drift_res['p_value']:.6f}, PSI={drift_res['psi']:.4f}"
            )
        else:
            logger.debug(
                f"No drift on categorical feature '{col}': "
                f"Chi2-p-val={drift_res['p_value']:.6f}, PSI={drift_res['psi']:.4f}"
            )
            
        metrics[col] = {
            "chi2_statistic": drift_res.get("chi2_statistic", 0.0),
            "p_value": drift_res.get("p_value", 1.0),
            "psi": drift_res.get("psi", 0.0),
            "drifted": is_col_drifted,
            "method": "chi2_and_psi"
        }
        
    total_checked = len(num_cols_to_check) + len(cat_cols_to_check)
    drift_ratio = (drifted_count / total_checked) if total_checked > 0 else 0.0
    global_drift = drifted_count > 0
    
    logger.info(
        f"Drift check complete. Drifted features count: {drifted_count}/{total_checked} "
        f"({drift_ratio * 100:.2f}%)"
    )
    
    return {
        "is_drifted": global_drift,
        "drift_ratio": drift_ratio,
        "metrics": metrics
    }
