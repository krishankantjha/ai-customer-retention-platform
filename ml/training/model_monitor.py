"""
Model Performance and Health Monitoring.
Tracks baseline thresholds, load states, and overall system health status.
"""

import os
import sys
import pickle
import logging
import pandas as pd

# Add project root to path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from configs.dataset_config import config_loader
from ml.training.feature_drift import detect_feature_drift

logger = logging.getLogger("ml.training.model_monitor")


def get_system_health(X_inference: pd.DataFrame) -> dict:
    """
    Computes overall system health by combining model loading state, active performance
    metrics from serialized metadata, and dynamic feature drift calculations.
    """
    logger.info("Evaluating system health state...")
    
    # 1. Resolve model metadata path
    artifacts_dir_relative = config_loader.training["data_paths"].get("artifacts_dir", "ml/artifacts")
    artifacts_dir = os.path.join(base_dir, artifacts_dir_relative)
    metadata_path = os.path.join(artifacts_dir, "model_metadata.pkl")
    
    # Check if metadata exists
    if not os.path.exists(metadata_path):
        logger.error(f"Model metadata card not found at: {metadata_path}")
        return {
            "status": "Degraded",
            "message": "Model metadata artifact missing. System health check degraded.",
            "model_version": "N/A",
            "last_trained": "N/A",
            "drift_detected": False,
            "drift_ratio": 0.0,
            "metrics": {},
            "drift_details": {}
        }
        
    try:
        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)
    except Exception as e:
        logger.exception(f"Failed to deserialize model metadata card: {e}")
        return {
            "status": "Degraded",
            "message": f"Corrupt model metadata file: {e}",
            "model_version": "N/A",
            "last_trained": "N/A",
            "drift_detected": False,
            "drift_ratio": 0.0,
            "metrics": {},
            "drift_details": {}
        }
        
    # Extract metadata properties with fallbacks
    model_name = meta.get("model_name", "calibrated_ensemble")
    model_version = meta.get("version", "1.1.0")
    training_date = meta.get("training_date", "N/A")
    val_metrics = meta.get("validation_metrics", {})
    
    # 2. Run feature drift check
    try:
        drift_report = detect_feature_drift(X_inference)
        is_drifted = drift_report["is_drifted"]
        drift_ratio = drift_report["drift_ratio"]
        drift_metrics = drift_report["metrics"]
    except Exception as e:
        logger.error(f"Failed to execute feature drift detection check: {e}", exc_info=True)
        return {
            "status": "Warning",
            "message": f"Feature drift detection failed: {e}",
            "model_version": model_version,
            "last_trained": training_date,
            "drift_detected": False,
            "drift_ratio": 0.0,
            "metrics": val_metrics,
            "drift_details": {}
        }
        
    # 3. Determine system status based on drift bounds
    # Warning threshold: if any feature drifts (is_drifted is True)
    # Degraded threshold: if combined drift ratio exceeds 20%, OR numeric drift ratio exceeds 40%
    status = "Healthy"
    message = "Model is operational with stable distribution bounds."
    
    num_total = sum(1 for m in drift_metrics.values() if m.get("method") == "ks_test")
    num_drifted = sum(1 for m in drift_metrics.values() if m.get("method") == "ks_test" and m.get("drifted"))
    numeric_drift_ratio = (num_drifted / num_total) if num_total > 0 else 0.0
    
    if is_drifted:
        status = "Warning"
        message = "Feature drift detected on one or more variables."
        
    if drift_ratio >= 0.20 or numeric_drift_ratio >= 0.40:
        status = "Degraded"
        message = (
            f"Significant feature drift detected (Combined Ratio: {drift_ratio * 100:.1f}%, "
            f"Numerical Ratio: {numeric_drift_ratio * 100:.1f}%). Recalibration recommended."
        )
        
    logger.info(
        f"System Health Status evaluated: {status} "
        f"(Combined Drift Ratio: {drift_ratio * 100:.1f}%, Numeric Drift Ratio: {numeric_drift_ratio * 100:.1f}%)"
    )

    if status == "Degraded":
        logger.critical(
            f"ALERT: Model health is DEGRADED. Feature drift exceeds threshold "
            f"(combined={drift_ratio * 100:.1f}%%, numerical={numeric_drift_ratio * 100:.1f}%%). "
            f"Immediate model recalibration is required. "
            f"Drifted features: {[k for k, v in drift_metrics.items() if v.get('drifted')]}"
        )
    elif status == "Warning":
        logger.warning(
            f"ALERT: Feature drift detected on one or more variables "
            f"(combined_ratio={drift_ratio * 100:.1f}%%). Monitor closely and consider retraining."
        )
    return {
        "status": status,
        "message": message,
        "model_name": model_name,
        "model_version": model_version,
        "last_trained": training_date,
        "drift_detected": is_drifted,
        "drift_ratio": drift_ratio,
        "metrics": val_metrics,
        "drift_details": drift_metrics
    }
