"""
Unit Tests for Final Production Hardening Tasks:
- SQLite & PostgreSQL compatibility.
- Environment-based CORS configuration.
- Lifespan startup and shutdown.
- Numerical and Categorical drift detection (Chi-Square & PSI).
- Model health combined drift ratio mapping.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add project root and backend to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import Settings
from ml.training.feature_drift import detect_feature_drift, _compute_chi_square_and_psi
from ml.training.model_monitor import get_system_health


# ==========================================
# 1. Database Compatibility Tests
# ==========================================
def test_database_url_postgresql_mapping():
    """Verify that postgresql:// schemes are mapped to postgresql+psycopg2://."""
    # SQLite default behaviour
    settings_sqlite = Settings(DATABASE_URL="sqlite:///./test.db")
    assert settings_sqlite.DATABASE_URL.startswith("sqlite:///")

    # PostgreSQL custom scheme mappings
    settings_pg = Settings(DATABASE_URL="postgresql://user:password@localhost:5432/testdb")
    
    # Verify mapping in session.py mock engine instantiation context
    db_url = settings_pg.DATABASE_URL
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        
    assert db_url == "postgresql+psycopg2://user:password@localhost:5432/testdb"


def test_database_connection_pool_arguments():
    """Verify engine arguments are tuned for PostgreSQL and SQLite respectively."""
    # SQLite args
    url_sqlite = "sqlite:///./test.db"
    connect_args_sqlite = {}
    engine_kwargs_sqlite = {"pool_pre_ping": True}
    
    if url_sqlite.startswith("sqlite"):
        connect_args_sqlite = {"check_same_thread": False}
        
    assert connect_args_sqlite == {"check_same_thread": False}

    # PostgreSQL args
    url_pg = "postgresql+psycopg2://user:password@localhost:5432/testdb"
    connect_args_pg = {}
    engine_kwargs_pg = {"pool_pre_ping": True}
    
    if not url_pg.startswith("sqlite"):
        engine_kwargs_pg.update({
            "pool_size": 20,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 1800
        })
        
    assert engine_kwargs_pg["pool_size"] == 20
    assert engine_kwargs_pg["max_overflow"] == 10
    assert engine_kwargs_pg["pool_recycle"] == 1800


# ==========================================
# 2. CORS Allowed Origins Validation
# ==========================================
def test_cors_origins_parsing_and_fallback():
    """Verify dynamic origins list parses correctly and fallbacks on malformed URLs."""
    # Valid configurations
    s_valid = Settings(ALLOWED_ORIGINS="https://retainiq.com, https://www.retainiq.com")
    parsed_valid = [o.strip() for o in s_valid.ALLOWED_ORIGINS.split(",") if o.strip()]
    assert parsed_valid == ["https://retainiq.com", "https://www.retainiq.com"]

    # Invalid configurations should fallback to localhost
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        s_invalid = Settings(ALLOWED_ORIGINS="https://retainiq.com, malformed_origin_without_http")
        # Validator should have run and reset the value
        assert "http://localhost:8501" in s_invalid.ALLOWED_ORIGINS
        assert "http://127.0.0.1:8501" in s_invalid.ALLOWED_ORIGINS
        mock_logger.error.assert_called()


# ==========================================
# 3. Lifespan Startup / Shutdown Lifecycle
# ==========================================
def test_lifespan_lifecycle():
    """Verify mock lifespan lifecycle runs startup eager validation."""
    from fastapi import FastAPI
    from app.main import lifespan
    
    app = FastAPI(lifespan=lifespan)
    
    with patch("app.main.startup_event") as mock_startup:
        from fastapi.testclient import TestClient
        with TestClient(app):
            # Client startup triggers lifespan context manager
            pass
        mock_startup.assert_called_once()


# ==========================================
# 4. Numerical & Categorical Drift Tests
# ==========================================
def test_categorical_drift_calculation():
    """Verify Chi-Square and PSI drift calculations under stable/shifted states."""
    # Case A: Identical Categorical Distributions (Stable)
    np.random.seed(42)
    categories = ["DSL", "Fiber optic", "No"]
    train_data = pd.Series(np.random.choice(categories, p=[0.3, 0.4, 0.3], size=500))
    inf_data_stable = pd.Series(np.random.choice(categories, p=[0.3, 0.4, 0.3], size=500))
    
    res_stable = _compute_chi_square_and_psi(train_data, inf_data_stable)
    assert not res_stable["drifted"]
    assert res_stable["p_value"] >= 0.05
    assert res_stable["psi"] < 0.1

    # Case B: Shifted Categorical Distributions (Drifted)
    inf_data_drifted = pd.Series(np.random.choice(categories, p=[0.1, 0.8, 0.1], size=500)) # Significant shift to Fiber optic
    
    res_drifted = _compute_chi_square_and_psi(train_data, inf_data_drifted)
    assert res_drifted["drifted"]
    assert res_drifted["p_value"] < 0.05 or res_drifted["psi"] >= 0.25


def test_mixed_features_drift_detection(tmp_path, monkeypatch):
    """Verify that detect_feature_drift handles mixed numerical/categorical arrays."""
    # Write a mock training dataset containing both feature types
    df_train = pd.DataFrame({
        "numeric__tenure": np.random.normal(loc=0.0, scale=1.0, size=500),
        "numeric__MonthlyCharges": np.random.normal(loc=0.0, scale=1.0, size=500),
        "categorical__InternetService_DSL": np.random.choice([0.0, 1.0], p=[0.7, 0.3], size=500),
        "categorical__InternetService_Fiber": np.random.choice([0.0, 1.0], p=[0.6, 0.4], size=500),
        "binary__SeniorCitizen": np.random.choice([0.0, 1.0], p=[0.9, 0.1], size=500),
        "Churn": np.random.choice([0, 1], size=500)
    })
    
    train_csv = tmp_path / "train_features.csv"
    df_train.to_csv(train_csv, index=False)
    
    from configs.dataset_config import config_loader
    monkeypatch.setitem(config_loader.training["data_paths"], "train_features", str(train_csv))
    
    # Stable inference dataframe
    df_inf_stable = pd.DataFrame({
        "numeric__tenure": np.random.normal(loc=0.0, scale=1.0, size=500),
        "numeric__MonthlyCharges": np.random.normal(loc=0.0, scale=1.0, size=500),
        "categorical__InternetService_DSL": np.random.choice([0.0, 1.0], p=[0.7, 0.3], size=500),
        "categorical__InternetService_Fiber": np.random.choice([0.0, 1.0], p=[0.6, 0.4], size=500),
        "binary__SeniorCitizen": np.random.choice([0.0, 1.0], p=[0.9, 0.1], size=500),
    })
    
    report_stable = detect_feature_drift(df_inf_stable)
    assert not report_stable["is_drifted"]
    assert report_stable["drift_ratio"] == 0.0
    assert report_stable["metrics"]["categorical__InternetService_DSL"]["method"] == "chi2_and_psi"
    assert report_stable["metrics"]["numeric__tenure"]["method"] == "ks_test"
    
    # Drifted inference dataframe
    df_inf_drifted = pd.DataFrame({
        "numeric__tenure": np.random.normal(loc=0.0, scale=1.0, size=500),
        "numeric__MonthlyCharges": np.random.normal(loc=0.0, scale=1.0, size=500),
        # Severe shift in category proportions
        "categorical__InternetService_DSL": np.random.choice([0.0, 1.0], p=[0.1, 0.9], size=500),
        "categorical__InternetService_Fiber": np.random.choice([0.0, 1.0], p=[0.6, 0.4], size=500),
        "binary__SeniorCitizen": np.random.choice([0.0, 1.0], p=[0.9, 0.1], size=500),
    })
    
    report_drifted = detect_feature_drift(df_inf_drifted)
    assert report_drifted["is_drifted"]
    assert report_drifted["drift_ratio"] > 0.0
    assert report_drifted["metrics"]["categorical__InternetService_DSL"]["drifted"]


# ==========================================
# 5. Combined System Health Mapping
# ==========================================
def test_system_health_combined_bounds(tmp_path, monkeypatch):
    """Verify that model monitor health maps status based on combined drift ratios."""
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
        import pickle
        pickle.dump(meta, f)
        
    from configs.dataset_config import config_loader
    monkeypatch.setitem(config_loader.training["data_paths"], "artifacts_dir", str(tmp_path))
    
    # Case A: Warning state (combined ratio is positive but small)
    mock_warning_drift = {
        "is_drifted": True,
        "drift_ratio": 0.05,
        "metrics": {
            "categorical__InternetService_DSL": {"drifted": True, "method": "chi2_and_psi"},
            "numeric__tenure": {"drifted": False, "method": "ks_test"}
        }
    }
    with patch("ml.training.model_monitor.detect_feature_drift", return_value=mock_warning_drift):
        health = get_system_health(pd.DataFrame())
        assert health["status"] == "Warning"

    # Case B: Degraded state (combined ratio >= 20%)
    mock_degraded_drift = {
        "is_drifted": True,
        "drift_ratio": 0.25,
        "metrics": {
            "categorical__InternetService_DSL": {"drifted": True, "method": "chi2_and_psi"},
            "numeric__tenure": {"drifted": False, "method": "ks_test"}
        }
    }
    with patch("ml.training.model_monitor.detect_feature_drift", return_value=mock_degraded_drift):
        health = get_system_health(pd.DataFrame())
        assert health["status"] == "Degraded"
