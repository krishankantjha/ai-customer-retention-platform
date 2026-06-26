"""
Class Imbalance Mitigation Layer.
Performs SMOTE oversampling exclusively on training partitions.
"""

import logging
import pandas as pd
from imblearn.over_sampling import SMOTE

logger = logging.getLogger("ml.preprocessing.imbalance")

def resample_training_data(X_train: pd.DataFrame, y_train: pd.Series, random_seed: int = 42, default_k_neighbors: int = 5) -> tuple:
    """
    Applies SMOTE oversampling to training dataset to balance target classes.
    Automatically detects minority class size and dynamically adjusts k_neighbors.
    Gracefully falls back to original data on invalid/extremely small inputs.
    Note: This is strictly applied only to the training split to prevent test-set leakage.
    """
    try:
        # Check if empty
        if X_train.empty or y_train.empty:
            logger.warning("Oversampling skipped: Training data is empty.")
            return X_train, y_train

        # Check unique classes
        unique_classes = y_train.nunique()
        if unique_classes < 2:
            class_list = list(y_train.unique())
            logger.warning(f"Oversampling skipped: Dataset has only {unique_classes} unique class(es): {class_list}.")
            return X_train, y_train

        # Check class counts
        class_counts = y_train.value_counts()
        min_class_count = int(class_counts.min())

        # If minority class has 1 or fewer samples, SMOTE is mathematically impossible
        if min_class_count <= 1:
            logger.warning(f"Oversampling skipped: Minority class has only {min_class_count} sample(s), SMOTE requires >= 2.")
            return X_train, y_train

        # Adjust k_neighbors dynamically if needed
        # SMOTE requires: k_neighbors <= min_class_count - 1
        k_neighbors = default_k_neighbors
        if min_class_count - 1 < default_k_neighbors:
            k_neighbors = min_class_count - 1
            logger.warning(
                f"Dynamic adjustment: k_neighbors reduced from {default_k_neighbors} to {k_neighbors} "
                f"because minority class has only {min_class_count} sample(s)."
            )

        logger.info(f"Applying SMOTE oversampling to training features (k_neighbors={k_neighbors})")
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_seed)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"Oversampling complete. Resampled shape: {X_resampled.shape[0]} rows (originally {X_train.shape[0]})")
        return X_resampled, y_resampled

    except Exception as e:
        logger.warning(f"Oversampling failed during execution: {e}. Graceful fallback to original data.")
        return X_train, y_train
