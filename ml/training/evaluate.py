"""
Model evaluation script for RetainIQ.

Loads the trained XGBoost model and evaluating feature matrices, computes performance
metrics (precision, recall, F1, AUC-ROC, PR-AUC), generates visual evaluation curves,
and computes SHAP values for model explainability.
"""

import os
import sys
import pickle
import logging
import pandas as pd

# Add project root to path to load configs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from configs.dataset_config import config_loader

# Set matplotlib backend to non-interactive 'Agg' to prevent blocking GUI windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)
import shap

logger = logging.getLogger("ml.training.evaluate")


def evaluate_model(test_features_path: str, artifacts_dir: str) -> None:
    """
    Loads test features, aligns feature columns with trained metadata,
    evaluates classification performance, saves visual evaluation plots,
    and runs SHAP explainability.
    """
    logger.info("Starting model evaluation process")

    # Load serialized model and metadata
    model_path = os.path.join(artifacts_dir, "model.pkl")
    metadata_path = os.path.join(artifacts_dir, "model_metadata.pkl")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model or metadata files not found in {artifacts_dir}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    logger.info(f"Loaded trained model: {metadata['model_type']}")

    # Load test dataset
    if not os.path.exists(test_features_path):
        raise FileNotFoundError(f"Test features CSV not found: {test_features_path}")

    target_col = config_loader.feature.get("target_column", "Churn")
    test_df = pd.read_csv(test_features_path)
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])

    # Align columns to match the features used in training
    X_test_aligned = X_test[metadata["feature_names_in"]]

    # Run prediction using the optimal business decision threshold of 0.25
    y_prob = model.predict_proba(X_test_aligned)[:, 1]
    y_pred = (y_prob >= 0.25).astype(int)

    # Calculate metrics
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)

    logger.info(f"Evaluation Metrics on Test Set:")
    logger.info(f"  ROC-AUC Score: {auc_roc:.4f}")
    logger.info(f"  PR-AUC Score : {auc_pr:.4f}")

    report = classification_report(y_test, y_pred)
    print("\n=== Model Classification Report ===")
    print(report)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Churn", "Churn"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_plot_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {cm_plot_path}")

    # 2. ROC Curve Plot
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_roc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(artifacts_dir, "roc_curve.png")
    plt.savefig(roc_plot_path)
    plt.close()
    logger.info(f"Saved ROC curve plot to: {roc_plot_path}")

    # 3. Precision-Recall Curve Plot
    plt.figure(figsize=(6, 5))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {auc_pr:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_plot_path = os.path.join(artifacts_dir, "precision_recall_curve.png")
    plt.savefig(pr_plot_path)
    plt.close()
    logger.info(f"Saved Precision-Recall curve plot to: {pr_plot_path}")

    # 4. SHAP Explanation Summary Plot
    logger.info("Computing SHAP explainer values...")
    # Extract base estimator from CalibratedClassifierCV if calibrated
    if hasattr(model, "calibrated_classifiers_"):
        shap_model = model.calibrated_classifiers_[0].estimator
    else:
        shap_model = model
    explainer = shap.Explainer(shap_model)
    shap_values = explainer(X_test_aligned)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_aligned, show=False)
    plt.title("SHAP Feature Importance Summary", fontsize=12, pad=15)
    plt.tight_layout()
    shap_plot_path = os.path.join(artifacts_dir, "shap_summary_plot.png")
    plt.savefig(shap_plot_path)
    plt.close()
    logger.info(f"Saved SHAP summary plot to: {shap_plot_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config_test_path = config_loader.training["data_paths"]["test_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    test_csv = config_test_path if os.path.isabs(config_test_path) else os.path.join(base_dir, config_test_path)
    artifacts = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)

    try:
        evaluate_model(test_csv, artifacts)
        print("Evaluation succeeded.")
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise
