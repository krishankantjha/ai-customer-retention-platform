"""
Multicollinearity and Correlation Review Utility.

Measures feature correlations in the processed training dataset, highlights highly
correlated engineered features, and provides recommendations for future linear model benchmarking.
"""

import os
import json
import logging
import pandas as pd

logger = logging.getLogger("ml.preprocessing.correlation_review")

def run_correlation_review(train_features_path: str, output_report_path: str, threshold: float = 0.70) -> dict:
    """
    Computes pairwise feature correlation on the training feature matrix,
    identifies collinear feature pairs exceeding the threshold, and outputs a JSON report.
    """
    logger.info(f"Loading training features from {train_features_path} for correlation review")
    if not os.path.exists(train_features_path):
        raise FileNotFoundError(f"Training features not found: {train_features_path}")

    df = pd.read_csv(train_features_path)
    
    # Drop Churn if it is present
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    logger.info("Computing pairwise Pearson correlation matrix")
    corr_matrix = df.corr().abs()

    # Find pairs of features exceeding threshold
    collinear_pairs = []
    features = list(corr_matrix.columns)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            val = float(corr_matrix.loc[f1, f2])
            if val >= threshold:
                collinear_pairs.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": round(val, 4)
                })

    # Sort by correlation strength descending
    collinear_pairs.sort(key=lambda x: x["correlation"], reverse=True)

    # Highlight specific engineered feature redundancies if they exist
    engineered_warnings = []
    for pair in collinear_pairs:
        f1, f2 = pair["feature_1"], pair["feature_2"]
        # Warn if AvgMonthlyCharge vs MonthlyCharges
        if ("AvgMonthlyCharge" in f1 or "AvgMonthlyCharge" in f2) and ("MonthlyCharges" in f1 or "MonthlyCharges" in f2):
            engineered_warnings.append(
                f"High collinearity between AvgMonthlyCharge and MonthlyCharges ({pair['correlation']:.4f}). "
                "For future Logistic Regression models, consider dropping AvgMonthlyCharge to prevent parameter inflation."
            )
        # Warn if num_services vs addon_count
        if ("num_services" in f1 or "num_services" in f2) and ("addon_count" in f1 or "addon_count" in f2):
            engineered_warnings.append(
                f"High collinearity between num_services and addon_count ({pair['correlation']:.4f}). "
                "For future Logistic Regression models, consider dropping num_services to reduce multicollinearity."
            )

    report = {
        "correlation_threshold": threshold,
        "collinear_pairs_count": len(collinear_pairs),
        "collinear_pairs": collinear_pairs,
        "engineered_warnings": engineered_warnings,
        "recommendations": [
            "Tree-based models (XGBoost, Random Forest, LightGBM) are highly robust to multicollinearity. No feature removal is required for the champion model.",
            "For future Logistic Regression benchmarking, multicollinearity can cause unstable coefficient estimates and high VIF.",
            "If building a linear benchmark model, drop one feature from each highly correlated pair or enable strong L1 (Lasso) or L2 (Ridge) regularization."
        ]
    }

    # Save report
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved correlation review report to {output_report_path}")
    
    # Print warnings to console
    if engineered_warnings:
        print("\n=== MULTICOLLINEARITY WARNINGS FOR FUTURE LINEAR MODELS ===")
        for warning in engineered_warnings:
            print(f" - {warning}")
        print("============================================================\n")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    from configs.dataset_config import config_loader
    
    config_train_path = config_loader.training["data_paths"]["train_features"]
    config_artifacts_dir = config_loader.training["data_paths"]["artifacts_dir"]
    train_csv = config_train_path if os.path.isabs(config_train_path) else os.path.join(base_dir, config_train_path)
    artifacts = config_artifacts_dir if os.path.isabs(config_artifacts_dir) else os.path.join(base_dir, config_artifacts_dir)
    report_path = os.path.join(artifacts, "correlation_report.json")

    run_correlation_review(train_csv, report_path)
