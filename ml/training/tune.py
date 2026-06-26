"""
Hyperparameter tuning module for RetainIQ.

Performs GridSearch cross-validation to find optimal parameters for both
Logistic Regression and XGBoost models on the training features.
"""

import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader

logger = logging.getLogger("ml.training.tune")


def tune_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Finds optimal hyperparameters for Logistic Regression.
    """
    logger.info("Tuning Logistic Regression...")

    # Drop is_early_stage for Logistic Regression to avoid collinearity
    features = [c for c in X_train.columns if c != "binary__is_early_stage"]
    X_train_lr = X_train[features]

    param_grid = config_loader.model.get("tuning", {}).get("logistic_regression", {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    })

    model = LogisticRegression(max_iter=1000, random_state=config_loader.model.get("random_seed", 42))
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train_lr, y_train)

    logger.info(f"Best LR parameters: {grid_search.best_params_}")
    logger.info(f"Best LR CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Finds optimal hyperparameters for XGBoost Classifier.
    """
    logger.info("Tuning XGBoost...")

    param_grid = config_loader.model.get("tuning", {}).get("xgboost", {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [50, 100, 150],
        "min_child_weight": [1, 3, 5]
    })

    model = XGBClassifier(
        random_state=config_loader.model.get("random_seed", 42),
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"Best XGB parameters: {grid_search.best_params_}")
    logger.info(f"Best XGB CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


if __name__ == "__main__":
    # Standard setup for running standalone
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_train_path = config_loader.training["data_paths"]["train_features"]
    train_path = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training features CSV not found: {train_path}")

    target_col = config_loader.feature.get("target_column", "Churn")
    train_df = pd.read_csv(train_path)
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])

    # Run tuning
    best_lr = tune_logistic_regression(X_train, y_train)
    best_xgb = tune_xgboost(X_train, y_train)

    print("\nTuning Results:")
    print(f"  Best Logistic Regression params : {best_lr}")
    print(f"  Best XGBoost params             : {best_xgb}")
