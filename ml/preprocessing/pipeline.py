"""
Preprocessing pipeline module for RetainIQ.

Defines the ColumnTransformer, splits the data into train and test sets,
applies the feature engineering from engineer.py, fits the preprocessing encoders,
and serializes the artifacts for training and inference.
"""

import sys
import os
import logging
import logging.config
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Add the project root to the python path to support importing from the ml package directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from ml.preprocessing.engineer import engineer_features
from configs.dataset_config import config_loader
from ml.preprocessing.loader import DataLoader
from ml.preprocessing.validator import DataValidator
from ml.preprocessing.encoding import get_categorical_encoder, get_ordinal_encoder
from ml.preprocessing.scaling import get_numeric_scaler
from ml.preprocessing.imbalance import resample_training_data


logger = logging.getLogger("ml.preprocessing.pipeline")

# Dynamically construct Column groups from config loader
_feature_cfg = config_loader.feature

TARGET_COL = _feature_cfg.get("target_column", "Churn")

# Categorical columns are loaded directly, excluding Contract as it is ordinally mapped to integer values
CATEGORICAL_COLS = [col for col in _feature_cfg.get("categorical_columns", []) if col != "Contract"]

# Numeric columns are loaded from config (excluding TotalCharges as it's dropped) and adding engineered numeric columns
NUMERIC_COLS = [col for col in _feature_cfg.get("numeric_columns", []) if col != "TotalCharges"] + ["addon_count", "commitment_score", "Contract", "AvgMonthlyCharge", "num_services"]

# Ordinal columns
ORDINAL_COLS = ["tenure_bucket"]
ORDINAL_CATEGORIES = [["0-12", "12-24", "24-48", "48+"]]

# Binary columns are loaded from config and adding engineered binary columns
_config_binary = list(_feature_cfg.get("binary_columns", []))
_engineered_binary = [
    "contract_is_mtm",
    "is_early_stage",
    "auto_pay_flag",
    "has_support",
    "contract_early_stage_flag",
    "premium_risk_flag",
    "household_stability_flag",
    "security_over_streaming",
    "fiber_zero_engagement_flag",
    "high_charge_early_stage_flag",
    "vulnerable_customer_flag"
]
BINARY_COLS = _config_binary + _engineered_binary


def run_pipeline(clean_csv_path: str, artifacts_dir: str, processed_dir: str) -> None:
    """
    Runs the full preprocessing pipeline on the cleaned dataset.
    Loads, splits, engineers features, fits/transforms scaling/encoding,
    and serializes the artifacts.
    """
    logger.info("Starting preprocessing pipeline run")

    # Ensure output directories exist
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Load clean data utilizing canonical DataLoader
    loader = DataLoader(logger)
    df = loader.load_from_csv(clean_csv_path)

    # Validate schema utilizing canonical DataValidator
    validator = DataValidator(logger)
    validator.validate_schema(df, strict=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Stratified split to preserve target class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=config_loader.model.get("random_seed", 42),
        stratify=y
    )
    logger.info(f"Split data into Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

    # Compute median MonthlyCharges on training set only to prevent data leakage
    train_monthly_charges_median = float(X_train["MonthlyCharges"].median())
    logger.info(f"Calculated training MonthlyCharges median: {train_monthly_charges_median:.2f}")

    # Apply feature engineering to both splits using the training median
    train_kwargs = {TARGET_COL: y_train.values}
    test_kwargs = {TARGET_COL: y_test.values}
    train_full = X_train.assign(**train_kwargs)
    test_full = X_test.assign(**test_kwargs)

    train_engineered = engineer_features(train_full, train_monthly_charges_median)
    test_engineered = engineer_features(test_full, train_monthly_charges_median)

    # Separate target from features again
    y_train_final = train_engineered.pop(TARGET_COL)
    y_test_final = test_engineered.pop(TARGET_COL)

    # Set up the preprocessing transformations using modular getters
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", get_numeric_scaler(), NUMERIC_COLS),
            ("ordinal", get_ordinal_encoder(categories=ORDINAL_CATEGORIES), ORDINAL_COLS),
            ("categorical", get_categorical_encoder(), CATEGORICAL_COLS),
            ("binary", "passthrough", BINARY_COLS),
        ],
        remainder="drop"
    )

    # Fit preprocessor only on the training set
    preprocessor.fit(train_engineered)
    logger.info("Fitted ColumnTransformer on training features")

    # Transform both datasets
    X_train_transformed = preprocessor.transform(train_engineered)
    X_test_transformed = preprocessor.transform(test_engineered)

    # Get output feature names
    feature_names = preprocessor.get_feature_names_out()

    # Reconstruct processed DataFrames
    train_df_raw = pd.DataFrame(X_train_transformed, columns=feature_names)
    train_df_raw[TARGET_COL] = y_train_final.values

    # Apply SMOTE oversampling to training dataset to balance target classes (no leakage risk, test split left untouched)
    X_train_resampled, y_train_resampled = resample_training_data(
        train_df_raw.drop(columns=[TARGET_COL]),
        train_df_raw[TARGET_COL],
        random_seed=config_loader.model.get("random_seed", 42),
        default_k_neighbors=config_loader.model.get("smote", {}).get("k_neighbors", 5)
    )
    
    # Reassign balanced training set
    train_df = X_train_resampled.copy()
    train_df[TARGET_COL] = y_train_resampled.values

    test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    test_df[TARGET_COL] = y_test_final.values

    # Save processed DataFrames to disk
    train_csv_path = os.path.join(processed_dir, "train_features.csv")
    test_csv_path = os.path.join(processed_dir, "test_features.csv")
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    logger.info(f"Saved processed training features to: {train_csv_path}")
    logger.info(f"Saved processed test features to: {test_csv_path}")

    # Save fitted ColumnTransformer
    pipeline_path = os.path.join(artifacts_dir, "pipeline.pkl")
    with open(pipeline_path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Saved pipeline artifact to: {pipeline_path}")

    # Save additional metadata needed for inference
    encoders_meta = {
        "train_monthly_charges_median": train_monthly_charges_median,
        "feature_names_out": list(feature_names),
        "numeric_cols": NUMERIC_COLS,
        "ordinal_cols": ORDINAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "binary_cols": BINARY_COLS,
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
    }

    encoders_path = os.path.join(artifacts_dir, "encoders.pkl")
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders_meta, f)
    logger.info(f"Saved encoder metadata to: {encoders_path}")

    # Run collinearity and correlation review report for future linear model benchmarking
    try:
        from ml.preprocessing.correlation_review import run_correlation_review
        report_path = os.path.join(artifacts_dir, "correlation_report.json")
        run_correlation_review(train_csv_path, report_path)
    except Exception as e:
        logger.warning(f"Failed to run correlation review: {e}")



def load_pipeline(artifacts_dir: str) -> tuple:
    """
    Loads and returns the fitted preprocessing pipeline and associated metadata.
    """
    pipeline_path = os.path.join(artifacts_dir, "pipeline.pkl")
    encoders_path = os.path.join(artifacts_dir, "encoders.pkl")

    if not os.path.exists(pipeline_path) or not os.path.exists(encoders_path):
        raise FileNotFoundError("Pipeline or encoder metadata files not found.")

    with open(pipeline_path, "rb") as f:
        preprocessor = pickle.load(f)

    with open(encoders_path, "rb") as f:
        metadata = pickle.load(f)

    return preprocessor, metadata


if __name__ == "__main__":
    import yaml

    # Configure logging based on local settings or defaults
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_path = os.path.join(base_dir, "configs", "logging_config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s")

    # Load paths from config loader
    config_clean_path = config_loader.training["data_paths"]["clean_data"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    config_processed_dir = config_loader.training["data_paths"]["processed_dir"]

    # Resolve paths relative to base_dir
    clean_csv = config_clean_path if os.path.isabs(config_clean_path) else os.path.join(base_dir, config_clean_path)
    artifacts = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)
    processed = config_processed_dir if os.path.isabs(config_processed_dir) else os.path.join(base_dir, config_processed_dir)

    try:
        run_pipeline(clean_csv, artifacts, processed)
        print("Pipeline execution succeeded.")
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        raise
