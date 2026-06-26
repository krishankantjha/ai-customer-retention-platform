"""
Confusion Matrix Export Module.
Generates the confusion matrix on the holdout test set using the calibrated GBDT ensemble
at the operational threshold (0.528) and exports counts to JSON.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader

logger = logging.getLogger("ml.training.confusion_matrix")


def export_confusion_matrix(test_path: str, model_path: str, output_dir: str) -> dict:
    """
    Loads test features, loaded ensemble model, generates predictions,
    computes confusion matrix metrics, and exports them to JSON.
    """
    logger.info("Starting confusion matrix generation...")

    # Validate inputs
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test features CSV not found: {test_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ensemble model path not found: {model_path}")

    # Load test set
    test_df = pd.read_csv(test_path)
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

    # Align columns
    xgb_features = ensemble.xgb_.feature_names_in_
    missing_cols = [col for col in xgb_features if col not in X_test.columns]
    if missing_cols:
        raise ValueError(f"Feature schema mismatch. Expected features: {list(xgb_features)}. Missing: {missing_cols}")
    X_test_aligned = X_test[xgb_features]

    # Generate predictions using decision threshold (0.528)
    threshold = config_loader.model.get("decision_threshold", 0.528)
    logger.info(f"Using operational threshold: {threshold:.3f}")

    probs = ensemble.predict_proba(X_test_aligned)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Compute confusion matrix
    if y_test.nunique() >= 2:
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    else:
        logger.warning("Test target has only 1 class. Computing custom counts.")
        tp = int(np.sum((preds == 1) & (y_test == 1)))
        fp = int(np.sum((preds == 1) & (y_test == 0)))
        fn = int(np.sum((preds == 0) & (y_test == 1)))
        tn = int(np.sum((preds == 0) & (y_test == 0)))

    metrics = {
        "decision_threshold": threshold,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,
        "Total_Samples": len(y_test)
    }

    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "confusion_matrix.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Successfully saved confusion matrix counts to: {json_path}")
    except Exception as e:
        logger.error(f"Failed to write confusion matrix results: {e}")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # Resolve paths from configuration
    config_test_path = config_loader.training["data_paths"]["test_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]

    test_csv = config_test_path if os.path.isabs(config_test_path) else os.path.join(base_dir, config_test_path)
    artifacts_dir = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)

    model_path = os.path.join(artifacts_dir, "models", "ensemble_model.pkl")
    output_dir = os.path.join(artifacts_dir, "metrics")

    try:
        export_confusion_matrix(test_csv, model_path, output_dir)
        print("Confusion matrix export succeeded.")
    except Exception as e:
        logger.exception(f"Confusion matrix generation failed: {e}")
        sys.exit(1)
