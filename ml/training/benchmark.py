"""
Model Benchmarking Module.
Compiles evaluation metrics across 8 model configurations (7 families + 1 dummy baseline)
under standard and cost-optimized thresholds, computing bootstrap confidence intervals.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader

logger = logging.getLogger("ml.training.benchmark")


def run_bootstrap_validation(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float, n_bootstraps: int = 200, seed: int = 42) -> dict:
    """
    Computes bootstrap statistics (mean, std, 95% CI bounds) for F1-Score and ROC-AUC
    on the holdout test set.
    """
    rng = np.random.default_rng(seed)
    f1_scores = []
    auc_scores = []
    
    # Pre-compute predictions to speed up bootstrapping
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = model.predict(X_test).astype(float)
    except Exception as e:
        logger.warning(f"Failed to generate predictions for bootstrap: {e}")
        return {}

    n_samples = len(y_test)
    y_test_arr = y_test.to_numpy()

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_test_arr[indices]
        probs_boot = probs[indices]
        
        preds_boot = (probs_boot >= threshold).astype(int)
        
        # Safe metric calculations on bootstrap sample
        f1 = f1_score(y_boot, preds_boot, zero_division=0)
        f1_scores.append(f1)
        
        if len(np.unique(y_boot)) > 1:
            auc = roc_auc_score(y_boot, probs_boot)
            auc_scores.append(auc)
        else:
            auc_scores.append(np.nan)
            
    f1_arr = np.array(f1_scores)
    auc_arr = np.array(auc_scores)
    auc_arr_clean = auc_arr[~np.isnan(auc_arr)]
    
    stats = {
        "F1_Mean": np.mean(f1_arr),
        "F1_Std": np.std(f1_arr),
        "F1_CI_Lower": np.percentile(f1_arr, 2.5),
        "F1_CI_Upper": np.percentile(f1_arr, 97.5),
        "AUC_Mean": np.mean(auc_arr_clean) if len(auc_arr_clean) > 0 else np.nan,
        "AUC_Std": np.std(auc_arr_clean) if len(auc_arr_clean) > 0 else np.nan,
        "AUC_CI_Lower": np.percentile(auc_arr_clean, 2.5) if len(auc_arr_clean) > 0 else np.nan,
        "AUC_CI_Upper": np.percentile(auc_arr_clean, 97.5) if len(auc_arr_clean) > 0 else np.nan,
    }
    
    return stats


def run_benchmarks(train_path: str, test_path: str, metrics_dir: str) -> pd.DataFrame:
    """
    Trains 8 model configurations, evaluates them on test features,
    and returns a summary DataFrame of performance metrics and bootstrap variance.
    """
    logger.info("Starting production benchmarking loop...")
    
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    target_col = config_loader.feature.get("target_column", "Churn")
    
    # Separate features and target
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    
    # Load exclusions dynamically from configuration loader
    pruned_cols = config_loader.model.get("pruned_columns", ["binary__has_support"])
    linear_exclusions = config_loader.model.get("linear_model_exclusions", ["binary__is_early_stage", "AvgMonthlyCharge"])
    
    features = [col for col in X_train.columns if col not in pruned_cols]
    
    X_train_aligned = X_train[features]
    X_test_aligned = X_test[features]
    
    seed = config_loader.model.get("random_seed", 42)
    threshold = config_loader.model.get("decision_threshold", 0.30)
    
    # Define the model configurations with parallelization enabled where supported
    models = {
        "DummyBaseline": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=seed, eval_metric="logloss", use_label_encoder=False, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=seed, verbosity=-1, n_jobs=-1),
        "MLP": MLPClassifier(max_iter=1000, random_state=seed)
    }
    
    results = []
    
    for name, model in models.items():
        logger.info(f"Training and evaluating {name}...")
        try:
            # Custom feature subsetting for Logistic Regression to prevent multi-collinearity
            if name == "LogisticRegression":
                lr_features = [col for col in features if col not in linear_exclusions]
                X_tr = X_train_aligned[lr_features]
                X_te = X_test_aligned[lr_features]
            else:
                X_tr = X_train_aligned
                X_te = X_test_aligned
            
            # Fit model
            model.fit(X_tr, y_train)
            
            # Predict probabilities
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_te)[:, 1]
            else:
                probs = model.predict(X_te).astype(float)
            
            # Compute holdout metrics at both default (0.50) and operational thresholds
            for thresh in [0.50, threshold]:
                preds = (probs >= thresh).astype(int)
                
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)
                
                # Safe ROC-AUC calculation
                if len(np.unique(y_test)) > 1:
                    auc = roc_auc_score(y_test, probs)
                else:
                    auc = float("nan")
                
                # Bootstrap validation
                boot = run_bootstrap_validation(model, X_te, y_test, thresh, seed=seed)
                
                results.append({
                    "Model": name,
                    "Threshold": thresh,
                    "Accuracy (Holdout)": acc,
                    "Precision (Holdout)": prec,
                    "Recall (Holdout)": rec,
                    "F1-Score (Holdout)": f1,
                    "ROC-AUC (Holdout)": auc,
                    "F1-Score (Bootstrap Mean)": boot.get("F1_Mean"),
                    "F1-Score (Bootstrap Std)": boot.get("F1_Std"),
                    "F1-Score (Bootstrap 95% CI Lower)": boot.get("F1_CI_Lower"),
                    "F1-Score (Bootstrap 95% CI Upper)": boot.get("F1_CI_Upper"),
                    "ROC-AUC (Bootstrap Mean)": boot.get("AUC_Mean"),
                    "ROC-AUC (Bootstrap Std)": boot.get("AUC_Std"),
                    "ROC-AUC (Bootstrap 95% CI Lower)": boot.get("AUC_CI_Lower"),
                    "ROC-AUC (Bootstrap 95% CI Upper)": boot.get("AUC_CI_Upper"),
                })
                
            logger.info(f"{name} (threshold={threshold:.2f}) -> Holdout F1: {f1:.4f}, Holdout ROC-AUC: {auc:.4f}")
        except Exception as e:
            logger.error(f"Failed to run model {name}: {e}")
            
    # Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Ensure folder exists
    os.makedirs(metrics_dir, exist_ok=True)
    results_csv_path = os.path.join(metrics_dir, "benchmark_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved production benchmark results to: {results_csv_path}")
    
    return results_df


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
    
    metrics_dir = os.path.join(artifacts_dir, "metrics")
    
    run_benchmarks(train_csv, test_csv, metrics_dir)
