"""
Global SHAP Explanations.
Handles global importance summaries and global feature driver calculations.
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader

logger = logging.getLogger("ml.explainability.shap_global")


def compute_global_shap(train_path: str, model_path: str, output_dir: str) -> dict:
    """
    Computes global SHAP values using the champion XGBoost model from the ensemble
    and saves feature importance bar plots and beeswarm charts.
    """
    logger.info("Computing global SHAP values...")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training features CSV not found: {train_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ensemble model path not found: {model_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training features
    train_df = pd.read_csv(train_path)
    target_col = config_loader.feature.get("target_column", "Churn")
    if target_col in train_df.columns:
        X_train = train_df.drop(columns=[target_col])
    else:
        X_train = train_df
        
    # Load Ensemble
    with open(model_path, "rb") as f:
        ensemble = pickle.load(f)
        
    xgb_features = ensemble.xgb_.feature_names_in_
    X_train_aligned = X_train[xgb_features]
    
    # We explain the champion XGBoost model of the ensemble
    champion_model = ensemble.xgb_
    
    # Initialize explainer
    explainer = shap.Explainer(champion_model)
    shap_values = explainer(X_train_aligned)
    
    # Calculate global mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    global_importance = dict(zip(xgb_features, [float(v) for v in mean_abs_shap]))
    
    # 1. Generate and save summary bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train_aligned, plot_type="bar", show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)")
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    logger.info(f"Saved global SHAP summary plot to: {summary_path}")
    
    # 2. Generate and save beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("Feature Impact on Churn Prediction (Beeswarm Plot)")
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "shap_beeswarm_plot.png")
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    logger.info(f"Saved global SHAP beeswarm plot to: {beeswarm_path}")
    
    return {
        "global_importance": global_importance,
        "summary_plot_path": summary_path,
        "beeswarm_plot_path": beeswarm_path
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    config_train_path = config_loader.training["data_paths"]["train_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    
    train_csv = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)
    artifacts_dir = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)
    
    model_file = os.path.join(artifacts_dir, "models", "ensemble_model.pkl")
    
    try:
        compute_global_shap(train_csv, model_file, artifacts_dir)
        print("Global SHAP computation succeeded.")
    except Exception as e:
        logger.exception(f"Global SHAP execution failed: {e}")
        sys.exit(1)
