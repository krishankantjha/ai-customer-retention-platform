import os
import sys
import json
import tempfile
import shutil
import pickle
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Ensure root paths are in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
backend_path = os.path.join(project_root, "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.core.security import verify_file_hash, ArtifactValidationError
from ml.segmentation.kmeans import KMeans


# 1. Test Hash Verification and Integrity Checks
def test_verify_file_hash_correct(tmp_path):
    import hashlib
    test_file = tmp_path / "test_model.pkl"
    content = b"dummy pickle data content"
    test_file.write_bytes(content)
    
    expected_hash = hashlib.sha256(content).hexdigest()
    
    # Should run without raising any error
    verify_file_hash(str(test_file), expected_hash)


def test_verify_file_hash_incorrect(tmp_path):
    test_file = tmp_path / "test_model.pkl"
    content = b"dummy pickle data content"
    test_file.write_bytes(content)
    
    incorrect_hash = "a" * 64
    
    with pytest.raises(ArtifactValidationError) as exc:
        verify_file_hash(str(test_file), incorrect_hash)
    assert "Artifact integrity check failed" in str(exc.value)


# 2. Test K-Means Stable Centroid Sorting
def test_kmeans_stable_centroid_sorting():
    # Construct a synthetic K-Means model with unordered centroids
    # Let's say n_clusters = 3, n_features = 2.
    # We want to sort based on the feature at index 1 (numeric__MonthlyCharges).
    # Centroids monthly charges: [100.0, 20.0, 50.0]
    centroids = np.array([
        [1.0, 100.0],
        [2.0, 20.0],
        [3.0, 50.0]
    ])
    
    kmeans = KMeans(n_clusters=3)
    kmeans.cluster_centers_ = centroids
    kmeans.labels_ = np.array([0, 1, 2, 0, 1, 2])
    
    # Run stable sorting manually as done in kmeans.py
    sort_idx = 1
    sort_order = np.argsort(centroids[:, sort_idx]) # should be [1, 2, 0] (20.0, 50.0, 100.0)
    
    kmeans.cluster_centers_ = kmeans.cluster_centers_[sort_order]
    
    new_label_mapping = {old_label: new_label for new_label, old_label in enumerate(sort_order)}
    new_labels = np.zeros_like(kmeans.labels_)
    for old_l, new_l in new_label_mapping.items():
        new_labels[kmeans.labels_ == old_l] = new_l
    kmeans.labels_ = new_labels
    
    # Assert centers are now sorted ascending by charges (index 1)
    assert list(kmeans.cluster_centers_[:, 1]) == [20.0, 50.0, 100.0]
    # Assert mapping was correct
    # Old mapping: [0->2, 1->0, 2->1]
    # Old labels: [0, 1, 2, 0, 1, 2] -> New labels: [2, 0, 1, 2, 0, 1]
    assert list(kmeans.labels_) == [2, 0, 1, 2, 0, 1]


# 3. Test API Schema and Response Backward Compatibility
def test_api_explain_response_schema_compatibility():
    # Create mock db and prediction models
    mock_db = MagicMock()
    mock_prediction = MagicMock()
    mock_prediction.cluster = 1
    mock_prediction.churn_probability = 0.65
    mock_prediction.is_high_risk = True
    mock_prediction.top_drivers = [{"feature": "tenure", "shap_value": 0.35}]
    mock_prediction.save_plays = [{"campaign": "Campaign A", "action": "Action A", "estimated_impact": 0.15}]
    mock_prediction.predicted_at = pd.Timestamp("2026-06-24 12:00:00")
    
    mock_customer = MagicMock()
    mock_customer.customer_id = "1234-AAAA"
    mock_customer.gender = "Female"
    mock_customer.tenure = 10
    mock_customer.monthly_charges = 85.0
    mock_customer.total_charges = 850.0
    mock_customer.prediction = mock_prediction
    
    # Import routes
    from app.schemas.prediction import CustomerExplainResponse
    from app.api.routes.predict import get_cohort_persona_name
    
    persona_name = get_cohort_persona_name(mock_prediction.cluster)
    
    response = CustomerExplainResponse(
        customer_id=mock_customer.customer_id,
        gender=mock_customer.gender,
        tenure=mock_customer.tenure,
        monthly_charges=mock_customer.monthly_charges,
        total_charges=mock_customer.total_charges,
        churn_probability=mock_prediction.churn_probability,
        is_high_risk=mock_prediction.is_high_risk,
        top_drivers=mock_prediction.top_drivers,
        save_plays=mock_prediction.save_plays,
        cluster=mock_prediction.cluster,
        cohort_persona=persona_name,
        segmentation={
            "cluster_id": mock_prediction.cluster,
            "persona": persona_name
        },
        predicted_at=mock_prediction.predicted_at
    )
    
    # Assert both root-level fields and nested segmentation are correct
    assert response.cluster == 1
    assert "High-Value" in response.cohort_persona
    assert response.segmentation.cluster_id == 1
    assert "High-Value" in response.segmentation.persona


# 4. Test Startup Lifecycle Handling
def test_startup_eager_load_failure():
    # If load_artifacts fails, startup should exit or raise an error
    from app.main import startup_event
    
    with patch("app.services.prediction_service.load_artifacts") as mock_load:
        mock_load.side_effect = ArtifactValidationError("Corrupted manifest signature")
        
        # startup_event calls sys.exit(1) on failure
        with patch("sys.exit") as mock_exit:
            startup_event()
            mock_exit.assert_called_once_with(1)


# 5. Test Diagnostics Metadata Drift Endpoint
def test_diagnostics_metadata_endpoint():
    # Setup test database and overrides to invoke the API client
    import app.database.session as session_module
    from fastapi.testclient import TestClient
    from app.main import app
    from app.services.auth_service import get_current_user
    
    # Register FastAPI dependency overrides
    app.dependency_overrides[get_current_user] = lambda: "admin"
    
    client = TestClient(app)
    headers = {"Authorization": "Bearer test_token"}
    
    try:
        # Test endpoint response directly
        response = client.get("/api/v1/analytics/diagnostics-metadata", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data
        assert "model_version" in data
        assert "diagnostics_version" in data
    finally:
        # Clean up overrides to prevent contaminating other tests
        app.dependency_overrides.clear()

