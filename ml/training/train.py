"""
Model training script for RetainIQ.

Trains the final optimized XGBoost Classifier on the processed training set,
dropping redundant features, and serializes the model and metadata to disk.
"""

import os
import sys
import pickle
import logging
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader

logger = logging.getLogger("ml.training.train")


def train_model(train_features_path: str, artifacts_dir: str) -> None:
    """
    Loads training data, applies feature pruning, trains the champion XGBoost model
    using optimized hyperparameters, and serializes the model and metadata.
    """
    logger.info("Starting model training process")

    # Load training features
    if not os.path.exists(train_features_path):
        raise FileNotFoundError(f"Training features not found: {train_features_path}")

    train_df = pd.read_csv(train_features_path)
    logger.info(f"Loaded training features: {train_df.shape[0]} rows, {train_df.shape[1]} columns")

    target_col = config_loader.feature.get("target_column", "Churn")
    # Separate target and features
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])

    # Define final features list (dropping redundant binary__has_support based on feature importance check)
    final_features = [col for col in X_train.columns if col != "binary__has_support"]
    X_train_final = X_train[final_features]

    logger.info(f"Feature set size: {X_train_final.shape[1]} (binary__has_support pruned)")

    # Calculate scale_pos_weight dynamically based on training class distribution
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info(f"Class imbalance ratio (scale_pos_weight): {scale_pos:.4f}")

    # Optimal hyperparameter weights found during GridSearch tuning
    best_params = config_loader.model.get("champion_model", {}).copy()
    best_params.pop("algorithm", None)
    best_params["random_state"] = config_loader.model.get("random_seed", 42)
    best_params["scale_pos_weight"] = scale_pos

    # Initialize raw XGBoost classifier and wrap it with Isotonic Calibration (cv=5)
    base_model = XGBClassifier(**best_params)
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(X_train_final, y_train)
    logger.info("Trained final calibrated XGBoost classifier model (Isotonic, cv=5)")

    # Ensure artifacts folder exists
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save trained model binary
    model_path = os.path.join(artifacts_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model binary to: {model_path}")

    # Save model metadata
    metadata = {
        "feature_names_in": final_features,
        "hyperparameters": best_params,
        "model_type": "CalibratedClassifierCV(XGBClassifier)"
    }
    metadata_path = os.path.join(artifacts_dir, "model_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved model metadata to: {metadata_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_train_path = config_loader.training["data_paths"]["train_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    train_csv = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)
    artifacts = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)

    try:
        train_model(train_csv, artifacts)
        print("Training succeeded.")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
