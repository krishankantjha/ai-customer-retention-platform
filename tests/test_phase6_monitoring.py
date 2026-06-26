"""
Unit and Integration Tests for Phase 6: Model Drift Detection & Performance Monitoring.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force backend directory in path for FastAPI app imports
backend_dir = os.path.abspath(os.path.join(project_root, "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from ml.training.feature_drift import detect_feature_drift
from ml.training.model_monitor import get_system_health
import app.services.prediction_service as pred_service

client = TestClient(app)


@pytest.fixture
def mock_train_features_csv(tmp_path):
    """Creates a temporary train_features.csv for drift comparison."""
    # FIX HIGH-10: Fixed seed for deterministic data generation.
    # Without a seed, KS test has ~5% false-positive rate (alpha=0.05),
    # causing this test to fail ~1 in 20 CI runs.
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "numeric__tenure": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__MonthlyCharges": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__addon_count": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__commitment_score": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__Contract": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__AvgMonthlyCharge": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__num_services": rng.normal(loc=0.0, scale=1.0, size=500),
        "Churn": rng.choice([0, 1], size=500)
    })
    csv_file = tmp_path / "train_features.csv"
    df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_feature_drift_detection(mock_train_features_csv, monkeypatch):
    """Verify KS-test drift check flags drifted populations correctly."""
    from configs.dataset_config import config_loader
    monkeypatch.setitem(config_loader.training["data_paths"], "train_features", mock_train_features_csv)

    # FIX HIGH-10: Use a seeded RNG for all random data in this test.
    rng = np.random.RandomState(99)  # Different seed from fixture for independence

    # Case A: Stable (same distribution N(0,1) — should NOT be flagged as drifted)
    df_stable = pd.DataFrame({
        "numeric__tenure": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__MonthlyCharges": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__addon_count": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__commitment_score": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__Contract": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__AvgMonthlyCharge": rng.normal(loc=0.0, scale=1.0, size=500),
        "numeric__num_services": rng.normal(loc=0.0, scale=1.0, size=500)
    })

    report_stable = detect_feature_drift(df_stable)
    assert not report_stable["is_drifted"]
    assert report_stable["drift_ratio"] == 0.0

    # Case B: Drifted — MonthlyCharges shifted by +5.0 stddevs, guaranteed KS detection
    rng_b = np.random.RandomState(77)
    df_drifted = pd.DataFrame({
        "numeric__tenure": rng_b.normal(loc=0.0, scale=1.0, size=500),
        "numeric__MonthlyCharges": rng_b.normal(loc=5.0, scale=1.0, size=500),  # Large shift
        "numeric__addon_count": rng_b.normal(loc=0.0, scale=1.0, size=500),
        "numeric__commitment_score": rng_b.normal(loc=0.0, scale=1.0, size=500),
        "numeric__Contract": rng_b.normal(loc=0.0, scale=1.0, size=500),
        "numeric__AvgMonthlyCharge": rng_b.normal(loc=0.0, scale=1.0, size=500),
        "numeric__num_services": rng_b.normal(loc=0.0, scale=1.0, size=500)
    })

    report_drifted = detect_feature_drift(df_drifted)
    assert report_drifted["is_drifted"]
    assert report_drifted["drift_ratio"] > 0.0
    assert report_drifted["metrics"]["numeric__MonthlyCharges"]["drifted"]


def test_model_health_status(tmp_path, monkeypatch):
    """Verify system health maps correctly to Healthy/Warning/Degraded status."""
    # Create temp model_metadata.pkl
    meta = {
        "model_name": "xgboost_test",
        "version": "1.1.0",
        "training_date": "2026-06-24",
        "validation_metrics": {
            "roc_auc": 0.84,
            "f1_score": 0.63
        }
    }
    
    meta_path = tmp_path / "model_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
        
    from configs.dataset_config import config_loader
    monkeypatch.setitem(config_loader.training["data_paths"], "artifacts_dir", str(tmp_path))
    
    # Case A: Mock drift report as healthy (0% drift)
    mock_healthy_drift = {
        "is_drifted": False,
        "drift_ratio": 0.0,
        "metrics": {}
    }
    with patch("ml.training.model_monitor.detect_feature_drift", return_value=mock_healthy_drift):
        health = get_system_health(pd.DataFrame())
        assert health["status"] == "Healthy"
        assert health["model_version"] == "1.1.0"
        
    # Case B: Mock drift report as warning (14% drift)
    mock_warning_drift = {
        "is_drifted": True,
        "drift_ratio": 0.14,
        "metrics": {}
    }
    with patch("ml.training.model_monitor.detect_feature_drift", return_value=mock_warning_drift):
        health = get_system_health(pd.DataFrame())
        assert health["status"] == "Warning"
        
    # Case C: Mock drift report as degraded (45% drift)
    mock_degraded_drift = {
        "is_drifted": True,
        "drift_ratio": 0.45,
        "metrics": {}
    }
    with patch("ml.training.model_monitor.detect_feature_drift", return_value=mock_degraded_drift):
        health = get_system_health(pd.DataFrame())
        assert health["status"] == "Degraded"


def test_prediction_logs_appending(tmp_path, monkeypatch):
    """Verify log_prediction_events appends prediction events to JSONL trail.

    FIX: Updated to use the monthly-partitioned filename
    (prediction_logs_YYYY-MM.jsonl) introduced by the MEDIUM-4 log rotation fix.
    """
    monkeypatch.setattr(pred_service, "artifacts_dir", str(tmp_path))

    cids = ["C1", "C2"]
    probs = np.array([0.15, 0.82])
    risks = np.array([False, True])
    clusters = np.array([0, 2])

    # Trigger logging
    pred_service.log_prediction_events(cids, probs, risks, clusters)

    # Monthly-partitioned filename: prediction_logs_YYYY-MM.jsonl
    from datetime import datetime
    month_str = datetime.utcnow().strftime("%Y-%m")
    log_file = tmp_path / "metrics" / f"prediction_logs_{month_str}.jsonl"
    assert os.path.exists(log_file), (
        f"Expected log file '{log_file}' not found. "
        f"Files in metrics/: {list((tmp_path / 'metrics').iterdir()) if (tmp_path / 'metrics').exists() else 'metrics dir missing'}"
    )

    # Read and assert JSON records
    with open(log_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])

    assert rec1["customer_id"] == "C1"
    assert rec1["churn_probability"] == 0.15
    assert not rec1["is_high_risk"]
    assert rec1["cluster"] == 0

    assert rec2["customer_id"] == "C2"
    assert rec2["churn_probability"] == 0.82
    assert rec2["is_high_risk"]
    assert rec2["cluster"] == 2



def test_model_health_api_route(monkeypatch):
    """Verify GET /api/v1/analytics/model-health route returns status and metrics."""
    # 1. Log in to get token
    login_resp = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Mock DB customer query and health calls
    mock_health_payload = {
        "status": "Healthy",
        "model_version": "1.1.0",
        "drift_detected": False,
        "drift_ratio": 0.0
    }
    
    monkeypatch.setattr("app.services.prediction_service.get_preprocessed_active_customers", lambda db: pd.DataFrame())
    monkeypatch.setattr("ml.training.model_monitor.get_system_health", lambda X: mock_health_payload)
    
    # Make request
    response = client.get(
        "/api/v1/analytics/model-health",
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Healthy"
    assert data["model_version"] == "1.1.0"
    assert not data["drift_detected"]
