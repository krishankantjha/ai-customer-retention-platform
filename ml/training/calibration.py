"""
Probability Calibration Module.
Evaluates Platt scaling, Isotonic regression, Brier Score, and Expected Calibration Error (ECE)
using a dedicated train/validation split from the training partition to prevent test set leakage.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader
from ml.preprocessing.imbalance import resample_training_data

logger = logging.getLogger("ml.training.calibration")


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes the Expected Calibration Error (ECE) for binary classification.
    """
    try:
        if y_true is None or y_prob is None:
            logger.warning("ECE calculation failed: Input arrays cannot be None.")
            return float("nan")
            
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        n_samples = len(y_true)
        if n_samples == 0 or len(y_prob) == 0:
            logger.warning("ECE calculation failed: Empty arrays passed.")
            return float("nan")
            
        if y_true.shape != y_prob.shape:
            logger.warning(f"ECE calculation failed: Shape mismatch y_true {y_true.shape} vs y_prob {y_prob.shape}")
            return float("nan")
            
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            if i == n_bins - 1:  # Include upper bound in last bin
                in_bin = in_bin | (y_prob == bin_upper)
                
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                confidence_in_bin = np.mean(y_prob[in_bin])
                ece += (bin_size / n_samples) * np.abs(accuracy_in_bin - confidence_in_bin)
                
        return float(ece)
    except Exception as e:
        logger.warning(f"ECE calculation failed during execution: {e}")
        return float("nan")


def validate_probabilities(probs: np.ndarray, model_name: str) -> None:
    """
    Asserts that the predicted probability array is valid, finite, and within bounds [0, 1].
    """
    if probs is None or len(probs) == 0:
        raise ValueError(f"Probability output for {model_name} is empty or None.")
    if np.isnan(probs).any():
        raise ValueError(f"Probability output for {model_name} contains NaN values.")
    if np.isinf(probs).any():
        raise ValueError(f"Probability output for {model_name} contains infinite values.")
    if (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError(
            f"Probability output for {model_name} contains values outside [0, 1]. "
            f"Range: [{probs.min():.4f}, {probs.max():.4f}]"
        )


def run_calibration(train_path: str, test_path: str, calibration_dir: str) -> dict:
    """
    Performs probability calibration without test set leakage.
    Uses a 80/20 train/validation split of the training partition for model and calibrator fitting.
    """
    logger.info("Starting probability calibration process...")
    
    # 1. Load configuration and paths
    clean_path = config_loader.training["data_paths"]["clean_data"]
    clean_csv = clean_path if os.path.isabs(clean_path) else os.path.join(base_dir, clean_path)
    
    if not os.path.exists(clean_csv):
        raise FileNotFoundError(f"Clean data CSV not found for natural reconstruction: {clean_csv}")
        
    clean_df = pd.read_csv(clean_csv)
    target_col = config_loader.feature.get("target_column", "Churn")
    
    # 2. Reconstruct natural train split
    X_clean = clean_df.drop(columns=[target_col])
    y_clean = clean_df[target_col]
    
    seed = config_loader.model.get("random_seed", 42)
    
    X_tr_raw, _, y_tr_natural, _ = train_test_split(
        X_clean, y_clean,
        test_size=0.20,
        random_state=seed,
        stratify=y_clean
    )
    
    # 3. Load preprocessor and apply feature engineering
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
    
    # 4. Create dedicated calibration train/validation split (80% / 20%) from natural training features
    X_train_calib, X_val_calib, y_train_calib, y_val_calib = train_test_split(
        X_train_transformed, y_train_clean,
        test_size=0.20,
        random_state=seed,
        stratify=y_train_clean
    )
    
    pruned_cols = config_loader.model.get("pruned_columns", ["binary__has_support"])
    features_all = [col for col in X_train_transformed.columns if col not in pruned_cols]
    
    # Apply SMOTE only to the calibration training split
    X_train_calib_resampled, y_train_calib_resampled = resample_training_data(
        X_train_calib[features_all],
        y_train_calib,
        random_seed=seed,
        default_k_neighbors=config_loader.model.get("smote", {}).get("k_neighbors", 5)
    )
    
    # Train base classifier on SMOTE balanced calibration train split
    xgb_params = config_loader.model.get("champion_model", {}).copy()
    xgb_params.pop("algorithm", None)
    xgb_params["random_state"] = seed
    xgb_params["n_jobs"] = -1
    
    logger.info("Training base classifier on SMOTE-balanced train split...")
    base_model = XGBClassifier(**xgb_params)
    base_model.fit(X_train_calib_resampled, y_train_calib_resampled)
    
    # Fit calibrators on natural calibration validation split
    logger.info("Fitting Platt Scaling (Sigmoid) calibrator on validation split...")
    calib_sig = CalibratedClassifierCV(estimator=base_model, cv="prefit", method="sigmoid")
    calib_sig.fit(X_val_calib[features_all], y_val_calib)
    
    logger.info("Fitting Isotonic Regression calibrator on validation split...")
    calib_iso = CalibratedClassifierCV(estimator=base_model, cv="prefit", method="isotonic")
    calib_iso.fit(X_val_calib[features_all], y_val_calib)
    
    # 5. Evaluate final metrics on holdout test set (strictly isolated)
    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col]
    X_test_aligned = test_df[features_all]
    
    probs_base = base_model.predict_proba(X_test_aligned)[:, 1]
    probs_sigmoid = calib_sig.predict_proba(X_test_aligned)[:, 1]
    probs_isotonic = calib_iso.predict_proba(X_test_aligned)[:, 1]
    
    validate_probabilities(probs_base, "Base XGBoost")
    validate_probabilities(probs_sigmoid, "Platt Scaling (Sigmoid)")
    validate_probabilities(probs_isotonic, "Isotonic Regression")
    
    metrics = {
        "Uncalibrated": {
            "Brier_Score": float(brier_score_loss(y_test, probs_base)),
            "ECE": float(compute_ece(y_test, probs_base))
        },
        "Platt_Scaling_Sigmoid": {
            "Brier_Score": float(brier_score_loss(y_test, probs_sigmoid)),
            "ECE": float(compute_ece(y_test, probs_sigmoid))
        },
        "Isotonic_Regression": {
            "Brier_Score": float(brier_score_loss(y_test, probs_isotonic)),
            "ECE": float(compute_ece(y_test, probs_isotonic))
        }
    }
    
    logger.info(f"Uncalibrated ECE: {metrics['Uncalibrated']['ECE']:.4f}, Brier: {metrics['Uncalibrated']['Brier_Score']:.4f}")
    logger.info(f"Sigmoid ECE: {metrics['Platt_Scaling_Sigmoid']['ECE']:.4f}, Brier: {metrics['Platt_Scaling_Sigmoid']['Brier_Score']:.4f}")
    logger.info(f"Isotonic ECE: {metrics['Isotonic_Regression']['ECE']:.4f}, Brier: {metrics['Isotonic_Regression']['Brier_Score']:.4f}")
    
    # Save calibration metrics
    os.makedirs(calibration_dir, exist_ok=True)
    json_path = os.path.join(calibration_dir, "calibration_metrics.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved calibration metrics to: {json_path}")
    except Exception as e:
        logger.error(f"Failed to export calibration metrics to JSON: {e}")
        
    # Plot curves
    fig = None
    try:
        fig = plt.figure(figsize=(8, 8), dpi=150)
        
        # Diagonal reference line
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        
        # Fraction of positives and mean predicted probabilities
        for probs, label, style in [
            (probs_base, "XGBoost (Uncalibrated)", "r-"),
            (probs_sigmoid, "XGBoost + Platt Scaling", "b--"),
            (probs_isotonic, "XGBoost + Isotonic Regression", "g-.")
        ]:
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
            plt.plot(mean_predicted_value, fraction_of_positives, style, marker="o", label=label)
            
        plt.xlabel("Mean Predicted Probability", fontsize=10)
        plt.ylabel("Fraction of Positives (Actual Churn Rate)", fontsize=10)
        plt.ylim([-0.05, 1.05])
        plt.title("Reliability Curves: Probability Calibration Performance", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plot_path = os.path.join(calibration_dir, "calibration_comparison.png")
        plt.savefig(plot_path, bbox_inches="tight")
        logger.info(f"Saved reliability comparison plot to: {plot_path}")
    except Exception as e:
        logger.error(f"Plotting reliability curves failed: {e}")
    finally:
        if fig is not None:
            plt.close(fig)
            
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    
    config_train_path = config_loader.training["data_paths"]["train_features"]
    config_test_path = config_loader.training["data_paths"]["test_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    
    train_csv = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)
    test_csv = config_test_path if os.path.isabs(config_test_path) else os.path.join(base_dir, config_test_path)
    artifacts_dir = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)
    
    calibration_dir = os.path.join(artifacts_dir, "calibration")
    
    run_calibration(train_csv, test_csv, calibration_dir)
