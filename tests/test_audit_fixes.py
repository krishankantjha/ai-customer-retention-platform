import os
import sys
import logging
import pytest
from pydantic import BaseModel

# Ensure root paths are in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
backend_path = os.path.join(project_root, "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from configs.dataset_config import deep_merge, ConfigLoader
from app.core.logging import SensitiveDataFilter, SensitiveDataFormatter, SensitiveJsonFormatter

# Test Config Loader & Deep Merge
def test_deep_merge():
    defaults = {
        "params": {
            "a": 1,
            "b": 2
        },
        "c": 3
    }
    loaded = {
        "params": {
            "b": 99
        },
        "d": 4
    }
    merged = deep_merge(defaults, loaded)
    assert merged["params"]["a"] == 1
    assert merged["params"]["b"] == 99
    assert merged["c"] == 3
    assert merged["d"] == 4

def test_config_loader_safety(tmp_path):
    # Test that ConfigLoader recovers on missing configs and uses defaults
    # Since config_loader is already initialized as a singleton, we can test ConfigLoader new instantiation behavior or _load_all_configs
    loader = ConfigLoader()
    assert loader.model is not None
    assert loader.training is not None
    assert loader.feature is not None
    assert loader.dashboard is not None

# Test Logging Redaction & Formatters
def test_sensitive_data_filter_redaction():
    filt = SensitiveDataFilter()
    
    # 1. Simple text redaction
    assert "password=[REDACTED]" in filt.redact_message("password=supersecret")
    assert "token: [REDACTED]" in filt.redact_message("token: abc-123")
    assert "monthly_charges: [REDACTED]" in filt.redact_message("monthly_charges: 75.5")
    
    # 2. Dict redaction
    d = {"password": "pass", "normal": "val", "nested": {"jwt": "token123", "x": 1}}
    redacted_d = filt.redact_dict(d)
    assert redacted_d["password"] == "[REDACTED]"
    assert redacted_d["normal"] == "val"
    assert redacted_d["nested"]["jwt"] == "[REDACTED]"
    assert redacted_d["nested"]["x"] == 1
    
    # 3. Pydantic model scrubbing
    class DummyModel(BaseModel):
        password: str
        normal_field: str
        monthly_charges: float
        
    model = DummyModel(password="admin123", normal_field="data", monthly_charges=50.25)
    scrubbed = filt.scrub_value(model)
    assert isinstance(scrubbed, dict)
    assert scrubbed["password"] == "[REDACTED]"
    assert scrubbed["normal_field"] == "data"
    assert scrubbed["monthly_charges"] == "[REDACTED]"
    
    # 4. Custom object scrubbing
    class CustomObj:
        def __init__(self):
            self.api_key = "secret_key"
            self.user = "username"
            
    obj = CustomObj()
    scrubbed_obj = filt.scrub_value(obj)
    assert isinstance(scrubbed_obj, dict)
    assert scrubbed_obj["api_key"] == "[REDACTED]"
    assert scrubbed_obj["user"] == "username"

def test_formatter_non_destructive():
    formatter = SensitiveDataFormatter("%(message)s")
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="User authentication failed for password=%s",
        args=("secretpass",),
        exc_info=None
    )
    
    formatted = formatter.format(record)
    assert "password=[REDACTED]" in formatted
    # Check that original msg and args are NOT mutated
    assert record.msg == "User authentication failed for password=%s"
    assert record.args == ("secretpass",)

def test_json_formatter_non_destructive():
    formatter = SensitiveJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Login with password=%s",
        args=("secretpass",),
        exc_info=None
    )
    
    formatted_json = formatter.format(record)
    assert "password=[REDACTED]" in formatted_json
    # Check that original record was not permanently mutated in-place
    assert record.msg == "Login with password=%s"
    assert record.args == ("secretpass",)

def test_traceback_redaction():
    formatter = SensitiveDataFormatter("%(message)s")
    try:
        raise ValueError("Database connection failed with api_key=secret123")
    except Exception as e:
        import sys as pysys
        ei = pysys.exc_info()
        tb_str = formatter.formatException(ei)
        
    assert "api_key=[REDACTED]" in tb_str
    assert "secret123" not in tb_str

def test_logging_setup_fallback_resilience():
    from app.core.logging import setup_logging
    # Call setup_logging; verify it runs without crashing and completes configuration successfully
    setup_logging()
    
    # Verify all handlers propagation settings are restored if fallback was triggered
    # (setup_logging should complete normally, but we ensure basic sanity)
    root = logging.getLogger()
    assert len(root.handlers) > 0


def test_batch_predict_threshold_override(monkeypatch):
    import numpy as np
    import pandas as pd
    from unittest.mock import MagicMock
    import app.services.prediction_service as pred_service
    
    # Mock data
    df = pd.DataFrame([{
        "customerID": "1234-AAAA",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 250.0,
        "Churn": "No"
    }, {
        "customerID": "5678-BBBB",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0,
        "TotalCharges": 900.0,
        "Churn": "No"
    }])

    # Mock artifacts
    mock_model = MagicMock()
    # Predict probabilities for 2 customers: 0.12 (low) and 0.28 (medium)
    mock_model.predict_proba.return_value = np.array([[0.88, 0.12], [0.72, 0.28]])
    mock_model.feature_names_in_ = ["MonthlyCharges", "tenure"]
    
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((2, 2))
    mock_preprocessor.get_feature_names_out.return_value = ["MonthlyCharges", "tenure"]
    
    encoders = {"train_monthly_charges_median": 50.0, "feature_names_out": ["MonthlyCharges", "tenure"]}
    metadata = {"feature_names_in": ["MonthlyCharges", "tenure"]}
    
    mock_explainer = MagicMock()
    mock_shap_values = MagicMock()
    mock_shap_values.values = np.zeros((2, 2))
    mock_explainer.return_value = mock_shap_values
    
    mock_kmeans = MagicMock()
    mock_kmeans.predict.return_value = np.array([0, 1])
    
    # Patch load_artifacts to return our mocks including mock_kmeans
    monkeypatch.setattr(pred_service, "load_artifacts", lambda: (mock_model, mock_preprocessor, encoders, metadata, mock_explainer, mock_kmeans))
    
    # Mock db session
    mock_db = MagicMock()
    
    # 1. Test config default fallback (configured to 0.15 in configs/model_config.yaml)
    # Customer 1 has prob 0.12 (under 0.15 -> low risk)
    # Customer 2 has prob 0.28 (above 0.15 -> high risk)
    pred_service.batch_predict_and_explain(df, mock_db, 1)
    
    # Check that predictions were constructed and passed to db.add_all
    added_predictions = []
    for call in mock_db.add_all.call_args_list:
        args = call[0][0]
        # Verify if they are Prediction objects
        for item in args:
            if hasattr(item, "churn_probability"):
                added_predictions.append(item)
                
    assert len(added_predictions) == 2
    # Probability 0.12 is less than 0.15 default -> False
    assert added_predictions[0].is_high_risk is False
    # Probability 0.28 is greater than 0.15 default -> True
    assert added_predictions[1].is_high_risk is True
    
    # Reset mock db
    mock_db.reset_mock()
    
    # 2. Test threshold override parameter (e.g. threshold=0.35)
    # Both Customer 1 (0.12) and Customer 2 (0.28) are under 0.35 -> both low risk
    pred_service.batch_predict_and_explain(df, mock_db, 1, threshold=0.35)
    
    added_predictions_overridden = []
    for call in mock_db.add_all.call_args_list:
        args = call[0][0]
        for item in args:
            if hasattr(item, "churn_probability"):
                added_predictions_overridden.append(item)
                
    assert len(added_predictions_overridden) == 2
    # Both must be low risk under 0.35 threshold
    assert added_predictions_overridden[0].is_high_risk is False
    assert added_predictions_overridden[1].is_high_risk is False
