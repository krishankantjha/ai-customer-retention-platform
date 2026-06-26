"""
Statistical Validation Module.
Performs 5x2cv Paired t-Test to compare models under strict fold isolation,
and executes 1,000-trial Bootstrap Validation to calculate standard errors and confidence intervals.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader
from ml.training.ensemble import CalibratedGBDTEnsemble
from ml.preprocessing.imbalance import resample_training_data

logger = logging.getLogger("ml.training.statistical_validation")


def run_statistical_validation(train_path: str, test_path: str, model_path: str, output_dir: str) -> dict:
    """
    Performs 5x2cv paired t-tests and 1,000-trial bootstrapping to assess model validity.
    """
    logger.info("Initializing statistical validation...")

    # Validate file existences
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train features path not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test features path not found: {test_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ensemble model path not found: {model_path}")

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = config_loader.feature.get("target_column", "Churn")
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train or test datasets.")

    y_test = test_df[target_col]
    
    # Retrieve parameters from configs
    seed = config_loader.model.get("random_seed", 42)
    threshold = config_loader.model.get("decision_threshold", 0.528)
    pruned_cols = config_loader.model.get("pruned_columns", ["binary__has_support"])
    linear_exclusions = config_loader.model.get("linear_model_exclusions", ["binary__is_early_stage", "AvgMonthlyCharge"])
    k_neighbors = config_loader.model.get("smote", {}).get("k_neighbors", 5)

    # 1. Reconstruct clean, natural training features (to perform fold-isolated SMOTE)
    clean_path = config_loader.training["data_paths"]["clean_data"]
    clean_csv = clean_path if os.path.isabs(clean_path) else os.path.join(base_dir, clean_path)
    
    if not os.path.exists(clean_csv):
        raise FileNotFoundError(f"Clean data CSV not found for natural reconstruction: {clean_csv}")
        
    clean_df = pd.read_csv(clean_csv)
    
    X_clean = clean_df.drop(columns=[target_col])
    y_clean = clean_df[target_col]
    
    X_tr_raw, _, y_tr_natural, _ = train_test_split(
        X_clean, y_clean,
        test_size=0.20,
        random_state=seed,
        stratify=y_clean
    )
    
    pipeline_path = os.path.join(base_dir, config_loader.training["data_paths"]["artifacts_dir"], "pipeline.pkl")
    with open(pipeline_path, "rb") as f:
        preprocessor = pickle.load(f)
        
    from ml.preprocessing.engineer import engineer_features
    train_monthly_charges_median = float(X_tr_raw["MonthlyCharges"].median())
    train_full = X_tr_raw.assign(**{target_col: y_tr_natural.values})
    train_engineered = engineer_features(train_full, train_monthly_charges_median)
    y_train_clean = train_engineered.pop(target_col)
    
    # Transform using preprocessor
    feature_names = preprocessor.get_feature_names_out()
    X_train_transformed = pd.DataFrame(preprocessor.transform(train_engineered), columns=feature_names)

    features_all = [col for col in X_train_transformed.columns if col not in pruned_cols]
    features_linear = [col for col in features_all if col not in linear_exclusions]

    X_train_all = X_train_transformed[features_all]
    X_test_all = test_df[features_all]

    # --- 2. 5x2cv Paired t-Test ---
    logger.info("Starting 5x2cv Paired t-Test...")
    
    diffs_f1 = []
    diffs_auc = []
    vars_f1 = []
    vars_auc = []
    
    first_diff_f1 = None
    first_diff_auc = None
    
    lr_params = {
        "C": 1.0,
        "solver": "liblinear",
        "random_state": seed
    }

    for r in range(5):
        logger.info(f"Processing 5x2cv Replication {r+1}/5...")
        # Split into two equal halves (50/50)
        X_a, X_b, y_a, y_b = train_test_split(
            X_train_all, y_train_clean,
            test_size=0.5,
            random_state=seed + r,
            stratify=y_train_clean
        )
        
        # We need two folds for each replication
        # Fold 1: train on a, test on b
        # Apply SMOTE only to training fold a
        X_a_resampled, y_a_resampled = resample_training_data(
            X_a, y_a,
            random_seed=seed,
            default_k_neighbors=k_neighbors
        )
        X_a_resampled_lr = X_a_resampled[features_linear]
        X_b_lr = X_b[features_linear]
        
        # Fold 2: train on b, test on a
        # Apply SMOTE only to training fold b
        X_b_resampled, y_b_resampled = resample_training_data(
            X_b, y_b,
            random_seed=seed,
            default_k_neighbors=k_neighbors
        )
        X_b_resampled_lr = X_b_resampled[features_linear]
        X_a_lr = X_a[features_linear]
        
        # Fit models on Fold 1
        ens1 = CalibratedGBDTEnsemble(seed=seed, decision_threshold=threshold, calibration_method="isotonic", reconstruct_clean=False)
        ens1.fit(X_a_resampled, y_a_resampled)
        
        lr1 = LogisticRegression(**lr_params)
        lr1.fit(X_a_resampled_lr, y_a_resampled)
        
        # Evaluate Fold 1 on natural validation fold b
        p_ens1 = ens1.predict_proba(X_b)[:, 1]
        preds_ens1 = (p_ens1 >= threshold).astype(int)
        f1_ens1 = f1_score(y_b, preds_ens1, zero_division=0)
        auc_ens1 = roc_auc_score(y_b, p_ens1)
        
        p_lr1 = lr1.predict_proba(X_b_lr)[:, 1]
        preds_lr1 = (p_lr1 >= threshold).astype(int)
        f1_lr1 = f1_score(y_b, preds_lr1, zero_division=0)
        auc_lr1 = roc_auc_score(y_b, p_lr1)
        
        diff_f1_1 = f1_ens1 - f1_lr1
        diff_auc_1 = auc_ens1 - auc_lr1
        
        if r == 0:
            first_diff_f1 = diff_f1_1
            first_diff_auc = diff_auc_1
            
        # Fit models on Fold 2
        ens2 = CalibratedGBDTEnsemble(seed=seed, decision_threshold=threshold, calibration_method="isotonic", reconstruct_clean=False)
        ens2.fit(X_b_resampled, y_b_resampled)
        
        lr2 = LogisticRegression(**lr_params)
        lr2.fit(X_b_resampled_lr, y_b_resampled)
        
        # Evaluate Fold 2 on natural validation fold a
        p_ens2 = ens2.predict_proba(X_a)[:, 1]
        preds_ens2 = (p_ens2 >= threshold).astype(int)
        f1_ens2 = f1_score(y_a, preds_ens2, zero_division=0)
        auc_ens2 = roc_auc_score(y_a, p_ens2)
        
        p_lr2 = lr2.predict_proba(X_a_lr)[:, 1]
        preds_lr2 = (p_lr2 >= threshold).astype(int)
        f1_lr2 = f1_score(y_a, preds_lr2, zero_division=0)
        auc_lr2 = roc_auc_score(y_a, p_lr2)
        
        diff_f1_2 = f1_ens2 - f1_lr2
        diff_auc_2 = auc_ens2 - auc_lr2
        
        # Compute stats for replication
        mean_diff_f1 = (diff_f1_1 + diff_f1_2) / 2.0
        mean_diff_auc = (diff_auc_1 + diff_auc_2) / 2.0
        
        var_f1 = (diff_f1_1 - mean_diff_f1)**2 + (diff_f1_2 - mean_diff_f1)**2
        var_auc = (diff_auc_1 - mean_diff_auc)**2 + (diff_auc_2 - mean_diff_auc)**2
        
        diffs_f1.append(diff_f1_1)
        diffs_auc.append(diff_auc_1)
        
        vars_f1.append(var_f1)
        vars_auc.append(var_auc)

    # Calculate 5x2cv t-statistic
    sum_vars_f1 = sum(vars_f1)
    sum_vars_auc = sum(vars_auc)
    
    t_stat_f1 = first_diff_f1 / (np.sqrt(0.2 * sum_vars_f1)) if sum_vars_f1 > 0 else 0.0
    t_stat_auc = first_diff_auc / (np.sqrt(0.2 * sum_vars_auc)) if sum_vars_auc > 0 else 0.0
    
    p_val_f1 = stats.t.sf(np.abs(t_stat_f1), 5) * 2.0
    p_val_auc = stats.t.sf(np.abs(t_stat_auc), 5) * 2.0
    
    logger.info(f"5x2cv F1 Paired t-test -> t-statistic: {t_stat_f1:.4f}, p-value: {p_val_f1:.4f}")
    logger.info(f"5x2cv AUC Paired t-test -> t-statistic: {t_stat_auc:.4f}, p-value: {p_val_auc:.4f}")

    # --- 3. Bootstrap Validation (1,000 trials) ---
    logger.info("Running Bootstrap Validation (1,000 trials) on holdout test set...")
    try:
        with open(model_path, "rb") as f:
            ensemble_model = pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to load ensemble model: {e}")
        
    rng = np.random.default_rng(seed)
    n_bootstraps = 1000
    
    # Pre-compute test predictions to speed up bootstrapping
    test_probs = ensemble_model.predict_proba(X_test_all)[:, 1]
    
    boot_f1s = []
    boot_aucs = []
    n_samples = len(y_test)
    y_test_arr = y_test.to_numpy()

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_test_arr[indices]
        probs_boot = test_probs[indices]
        preds_boot = (probs_boot >= threshold).astype(int)

        # F1 score
        f1_b = f1_score(y_boot, preds_boot, zero_division=0)
        boot_f1s.append(f1_b)

        # ROC-AUC score
        if len(np.unique(y_boot)) > 1:
            auc_b = roc_auc_score(y_boot, probs_boot)
            boot_aucs.append(auc_b)

    boot_f1s = np.array(boot_f1s)
    boot_aucs = np.array(boot_aucs)

    bootstrap_stats = {
        "F1": {
            "Mean": float(np.mean(boot_f1s)),
            "Std": float(np.std(boot_f1s)),
            "CI_Lower": float(np.percentile(boot_f1s, 2.5)),
            "CI_Upper": float(np.percentile(boot_f1s, 97.5))
        },
        "ROC_AUC": {
            "Mean": float(np.mean(boot_aucs)),
            "Std": float(np.std(boot_aucs)),
            "CI_Lower": float(np.percentile(boot_aucs, 2.5)),
            "CI_Upper": float(np.percentile(boot_aucs, 97.5))
        }
    }

    logger.info(f"Bootstrap F1 -> Mean: {bootstrap_stats['F1']['Mean']:.4f}, 95% CI: [{bootstrap_stats['F1']['CI_Lower']:.4f}, {bootstrap_stats['F1']['CI_Upper']:.4f}]")
    logger.info(f"Bootstrap AUC -> Mean: {bootstrap_stats['ROC_AUC']['Mean']:.4f}, 95% CI: [{bootstrap_stats['ROC_AUC']['CI_Lower']:.4f}, {bootstrap_stats['ROC_AUC']['CI_Upper']:.4f}]")

    results = {
        "Threshold": threshold,
        "Methodology": "5x2cv Paired t-Test (Dietterich, 1998)",
        "Paired_T_Test": {
            "F1": {
                "T_Statistic": float(t_stat_f1) if not np.isnan(t_stat_f1) else None,
                "P_Value": float(p_val_f1) if not np.isnan(p_val_f1) else None
            },
            "ROC_AUC": {
                "T_Statistic": float(t_stat_auc) if not np.isnan(t_stat_auc) else None,
                "P_Value": float(p_val_auc) if not np.isnan(p_val_auc) else None
            }
        },
        "Bootstrap_1000_Trials": bootstrap_stats
    }

    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "statistical_results.json")
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Successfully saved statistical validation results to: {out_json}")
    except Exception as e:
        logger.error(f"Failed to write statistical validation results: {e}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # Resolve paths from configuration
    config_train_path = config_loader.training["data_paths"]["train_features"]
    config_test_path = config_loader.training["data_paths"]["test_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]

    train_csv = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)
    test_csv = config_test_path if os.path.isabs(config_test_path) else os.path.join(base_dir, config_test_path)
    artifacts_dir = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)

    model_path = os.path.join(artifacts_dir, "models", "ensemble_model.pkl")
    output_dir = os.path.join(artifacts_dir, "metrics")

    try:
        run_statistical_validation(train_csv, test_csv, model_path, output_dir)
        print("Statistical validation succeeded.")
    except Exception as e:
        logger.exception(f"Statistical validation failed: {e}")
        sys.exit(1)
