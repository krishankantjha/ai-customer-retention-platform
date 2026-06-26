"""
Cost-Sensitive Threshold Optimization Module.
Sweeps probability thresholds over validation (out-of-fold) predictions to identify optimal decision boundaries,
and evaluates final cost-effectiveness and classification performance on the holdout test set.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader

logger = logging.getLogger("ml.training.threshold")


def optimize_threshold(test_csv: str, model_path: str, output_dir: str) -> dict:
    """
    Loads the serialized GBDT Ensemble, identifies optimal thresholds on out-of-fold validation splits,
    and evaluates metrics under optimal vs. locked thresholds on the holdout test set.
    """
    logger.info("Starting threshold optimization sweeps...")
    
    # 1. Dataset and model validation
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test features CSV not found: {test_csv}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ensemble model pickle file not found: {model_path}")
        
    try:
        test_df = pd.read_csv(test_csv)
    except Exception as e:
        raise IOError(f"Failed to read test CSV: {e}")
        
    if test_df.empty:
        raise ValueError("Test features dataset is empty.")
        
    target_col = config_loader.feature.get("target_column", "Churn")
    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test features.")
        
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    
    # Load GBDT Ensemble
    try:
        with open(model_path, "rb") as f:
            ensemble = pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to deserialize GBDT Ensemble model: {e}")
        
    # Check for validation out-of-fold data stored on the model instance
    if not hasattr(ensemble, "oof_calibrated_probs_") or not hasattr(ensemble, "oof_targets_"):
        raise ValueError("Ensemble model does not contain validation out-of-fold calibrated probabilities or targets.")
        
    y_val = ensemble.oof_targets_
    probs_val = ensemble.oof_calibrated_probs_
    
    # 2. Retrieve cost parameters from configs
    cost_cfg = config_loader.model.get("cost_coefficients", {})
    c_fn = cost_cfg.get("false_negative_cost", 5.0)
    c_fp = cost_cfg.get("false_positive_cost", 1.0)
    logger.info(f"Loaded business costs -> False Negative: {c_fn}, False Positive: {c_fp}")
    
    # 3. Sweep thresholds on out-of-fold validation split to find optimal parameters
    thresholds = np.arange(0.10, 0.91, 0.05)
    best_f1 = -1.0
    best_f1_threshold = 0.50
    min_cost = float("inf")
    best_cost_threshold = 0.50
    
    for thresh in thresholds:
        preds_val = (probs_val >= thresh).astype(int)
        
        f1 = float(f1_score(y_val, preds_val, zero_division=0))
        
        # Calculate validation cost
        tp = np.sum((preds_val == 1) & (y_val == 1))
        fp = np.sum((preds_val == 1) & (y_val == 0))
        fn = np.sum((preds_val == 0) & (y_val == 1))
        tn = np.sum((preds_val == 0) & (y_val == 0))
        
        total_cost = float(fn * c_fn + fp * c_fp)
        
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = float(thresh)
            
        if total_cost < min_cost:
            min_cost = total_cost
            best_cost_threshold = float(thresh)
            
    logger.info(f"OOF Validation -> Cost-Optimal Threshold: {best_cost_threshold:.2f} (Cost: {min_cost:.2f})")
    logger.info(f"OOF Validation -> F1-Optimal Threshold: {best_f1_threshold:.2f} (F1-Score: {best_f1:.4f})")
    
    # 4. Generate test set predictions (strictly clean holdout evaluation)
    xgb_features = ensemble.xgb_.feature_names_in_
    missing_cols = [col for col in xgb_features if col not in X_test.columns]
    if missing_cols:
        raise ValueError(f"Feature schema mismatch. Expected features: {list(xgb_features)}. Missing: {missing_cols}")
    X_test_aligned = X_test[xgb_features]
    
    probs_test = ensemble.predict_proba(X_test_aligned)[:, 1]
    
    # Function to calculate complete statistics for a given threshold on the test set
    def evaluate_threshold_metrics(probs, y_true, thresh):
        preds = (probs >= thresh).astype(int)
        acc = float(accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
        
        total_cost = float(fn * c_fn + fp * c_fp)
        
        return {
            "Threshold": float(thresh),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Total_Cost": total_cost
        }
        
    # Baseline test set cost (No outreach)
    test_n_churners = int(y_test.sum())
    test_baseline_cost = float(test_n_churners * c_fn)
    
    # Evaluate at key points
    locked_threshold = config_loader.model.get("decision_threshold", 0.528)
    
    optimal_cost_metrics = evaluate_threshold_metrics(probs_test, y_test, best_cost_threshold)
    optimal_f1_metrics = evaluate_threshold_metrics(probs_test, y_test, best_f1_threshold)
    locked_metrics = evaluate_threshold_metrics(probs_test, y_test, locked_threshold)
    
    # Calculate sweep grid for test set plotting (plotting only)
    sweep_results = []
    for thresh in thresholds:
        sweep_results.append(evaluate_threshold_metrics(probs_test, y_test, thresh))
        
    summary = {
        "Cost_Coefficients": {
            "C_FN": c_fn,
            "C_FP": c_fp
        },
        "Baseline_No_Outreach_Cost": test_baseline_cost,
        "Optimal_F1_Threshold": {
            "F1-Score": optimal_f1_metrics["F1-Score"],
            "Threshold": best_f1_threshold,
            "Test_Evaluation": optimal_f1_metrics
        },
        "Optimal_Cost_Threshold": {
            "Cost": optimal_cost_metrics["Total_Cost"],
            "Threshold": best_cost_threshold,
            "Net_Savings": test_baseline_cost - optimal_cost_metrics["Total_Cost"],
            "Test_Evaluation": optimal_cost_metrics
        },
        "Locked_Threshold_Evaluation": {
            "Threshold": locked_threshold,
            "Test_Evaluation": locked_metrics
        },
        "Threshold_Sweep_Grid": sweep_results
    }
    
    # 5. Export metrics to JSON
    try:
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "threshold_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Saved threshold metrics to: {json_path}")
    except Exception as e:
        logger.error(f"Failed to export threshold metrics to JSON: {e}")
        
    # 6. Plotting Curves
    fig = None
    try:
        fig = plt.figure(figsize=(10, 6), dpi=150)
        
        plt.plot(thresholds, [r["Precision"] for r in sweep_results], "b--", label="Precision", alpha=0.7)
        plt.plot(thresholds, [r["Recall"] for r in sweep_results], "g-.", label="Recall", alpha=0.7)
        plt.plot(thresholds, [r["F1-Score"] for r in sweep_results], "r-", label="F1-Score", linewidth=2)
        
        # Normalized Cost overlay
        costs = [r["Total_Cost"] for r in sweep_results]
        max_c = max(costs) if max(costs) > 0 else 1.0
        norm_costs = [c / max_c for c in costs]
        plt.plot(thresholds, norm_costs, "k:", label="Normalized Cost", linewidth=2)
        
        # Vertical flags
        plt.axvline(best_f1_threshold, color="red", linestyle="--", alpha=0.5, label=f"Max F1 ({best_f1_threshold:.2f})")
        plt.axvline(best_cost_threshold, color="black", linestyle="--", alpha=0.5, label=f"Min Cost ({best_cost_threshold:.2f})")
        plt.axvline(locked_threshold, color="purple", linestyle="--", alpha=0.5, label=f"Locked ({locked_threshold:.3f})")
        
        plt.xlabel("Classification Decision Threshold", fontsize=10)
        plt.ylabel("Score / Normalized Cost", fontsize=10)
        plt.title("Threshold Performance vs Business Cost Optimization (Test Evaluation)", fontsize=12)
        plt.legend(loc="lower left", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plot_path = os.path.join(output_dir, "threshold_optimization.png")
        plt.savefig(plot_path, bbox_inches="tight")
        logger.info(f"Saved threshold optimization curves plot to: {plot_path}")
    except Exception as e:
        logger.error(f"Plotting threshold curves failed: {e}")
    finally:
        if fig is not None:
            plt.close(fig)
            
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    
    config_test_path = config_loader.training["data_paths"]["test_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    
    test_csv = config_test_path if os.path.isabs(config_test_path) else os.path.join(base_dir, config_test_path)
    artifacts_dir = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)
    
    model_path = os.path.join(artifacts_dir, "models", "ensemble_model.pkl")
    output_dir = os.path.join(artifacts_dir, "metrics")
    
    optimize_threshold(test_csv, model_path, output_dir)
