"""
Unit Tests for Phase 3: Explainability & Customer Segmentation.
"""

import os
import sys
import shutil
import pickle
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
import shap

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.explainability.shap_global import compute_global_shap
from ml.explainability.shap_local import LocalExplainer, check_business_condition
from ml.segmentation.kmeans import run_segmentation
from configs.dataset_config import config_loader


# Define picklable dummy models at the global module scope to support serialization in tests
class PicklableDummyXGB:
    def __init__(self):
        self.feature_names_in_ = ["MonthlyCharges", "tenure", "Contract_Month-to-month"]


class PicklableMockEnsemble:
    def __init__(self):
        self.xgb_ = PicklableDummyXGB()
        self.fitted_ = True
        
    def predict_proba(self, X):
        return np.array([[0.9, 0.1]] * len(X))


class PicklableMockPreprocessor:
    def __init__(self, feature_names, transformed_data):
        self.feature_names = feature_names
        self.transformed_data = transformed_data
        
    def get_feature_names_out(self):
        return np.array(self.feature_names)
        
    def transform(self, X):
        return self.transformed_data[:len(X)]


@pytest.fixture
def temp_output_dir(tmp_path):
    d = tmp_path / "artifacts"
    os.makedirs(d, exist_ok=True)
    return str(d)


@pytest.fixture
def mock_dataset():
    # 10 records with 3 features
    df = pd.DataFrame({
        "MonthlyCharges": [25.0, 85.0, 70.0, 95.0, 20.0, 110.0, 65.0, 45.0, 80.0, 55.0],
        "tenure": [12, 1, 24, 3, 48, 6, 36, 18, 2, 72],
        "Contract_Month-to-month": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        "Churn": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    })
    return df


def test_compute_global_shap(temp_output_dir, mock_dataset, monkeypatch):
    train_csv = os.path.join(temp_output_dir, "train_features.csv")
    mock_dataset.to_csv(train_csv, index=False)
    
    ensemble = PicklableMockEnsemble()
    model_pkl = os.path.join(temp_output_dir, "ensemble_model.pkl")
    with open(model_pkl, "wb") as f:
        pickle.dump(ensemble, f)
        
    # Mock SHAP Explainer internally using a real shap.Explanation object to bypass shape and type assertion checks
    shap_values = shap.Explanation(
        values=np.random.randn(10, 3),
        base_values=np.array([0.5] * 10),
        data=mock_dataset.drop(columns=["Churn"]).values,
        feature_names=["MonthlyCharges", "tenure", "Contract_Month-to-month"]
    )
    
    mock_explainer = MagicMock()
    mock_explainer.return_value = shap_values
    monkeypatch.setattr(shap, "Explainer", lambda model: mock_explainer)
    
    # Run global SHAP computation
    results = compute_global_shap(train_csv, model_pkl, temp_output_dir)
    
    assert "global_importance" in results
    assert os.path.exists(results["summary_plot_path"])
    assert os.path.exists(results["beeswarm_plot_path"])
    assert "MonthlyCharges" in results["global_importance"]


def test_local_explainer_value_aware_recommendations(mock_dataset, monkeypatch):
    # Setup mock local explanation using a real shap.Explanation object
    shap_values = shap.Explanation(
        values=np.array([[0.1, 0.35, 0.45]]), # tenure (0.35) and contract (0.45) are positive drivers
        base_values=np.array([0.5]),
        data=mock_dataset.drop(columns=["Churn"]).iloc[[1]].values,
        feature_names=["MonthlyCharges", "tenure", "Contract_Month-to-month"]
    )
    
    mock_explainer = MagicMock()
    mock_explainer.return_value = shap_values
    monkeypatch.setattr(shap, "Explainer", lambda model: mock_explainer)
    
    # Initialize LocalExplainer
    ensemble = PicklableMockEnsemble()
    explainer = LocalExplainer(
        ensemble,
        feature_names=["MonthlyCharges", "tenure", "Contract_Month-to-month"]
    )
    
    # 1. Test case: Short-term customer (tenure = 1) -> Should map tenure to welcome campaign
    customer_record_new = pd.DataFrame([{
        "MonthlyCharges": 85.0,
        "tenure": 1, # short term
        "Contract": "Month-to-month",
        "Contract_Month-to-month": 1.0
    }])
    
    explanation_new = explainer.explain_customer(customer_record_new)
    assert explanation_new["success"] is True
    plays_new = [p["play_name"] for p in explanation_new["save_plays"]]
    
    # tenure should convert to "Early Stage Loyalty Welcome"
    assert any("Early Stage" in play for play in plays_new)
    
    # 2. Test case: Long-term customer (tenure = 72) -> Should NOT map tenure to welcome campaign (should map to fallback)
    customer_record_long = pd.DataFrame([{
        "MonthlyCharges": 85.0,
        "tenure": 72, # long term
        "Contract": "Month-to-month",
        "Contract_Month-to-month": 1.0
    }])
    
    explanation_long = explainer.explain_customer(customer_record_long)
    assert explanation_long["success"] is True
    plays_long = explanation_long["save_plays"]
    
    # Check that "Early Stage Loyalty Welcome" is NOT triggered for tenure
    tenure_play = [p for p in plays_long if p["feature"] == "tenure"][0]
    assert "Early Stage" not in tenure_play["play_name"]
    assert "General Loyalty Outreach" in tenure_play["play_name"]


def test_explainer_caching(monkeypatch):
    # Mock SHAP Explainer
    mock_explainer = MagicMock()
    monkeypatch.setattr(shap, "Explainer", lambda model: mock_explainer)
    
    ensemble = PicklableMockEnsemble()
    
    # Instantiate two explainers with the same model instance
    explainer1 = LocalExplainer(ensemble, feature_names=["tenure"])
    explainer2 = LocalExplainer(ensemble, feature_names=["tenure"])
    
    # Assert they use the exact same cached Explainer object (Issue 3)
    assert explainer1.explainer is explainer2.explainer


def test_run_segmentation_continuous_only(temp_output_dir, mock_dataset, monkeypatch):
    # Setup mock clean data for natural reconstruction inside kmeans
    clean_csv = os.path.join(temp_output_dir, "telco_churn_clean.csv")
    
    # Create 10 mock records representing clean data schema
    clean_data = pd.DataFrame({
        "customerID": [f"id-{i}" for i in range(10)],
        "gender": ["Male"] * 5 + ["Female"] * 5,
        "SeniorCitizen": [0] * 10,
        "Partner": ["Yes"] * 10,
        "Dependents": ["No"] * 10,
        "tenure": [12, 1, 24, 3, 48, 6, 36, 18, 2, 72],
        "PhoneService": ["Yes"] * 10,
        "MultipleLines": ["No"] * 10,
        "InternetService": ["DSL"] * 10,
        "OnlineSecurity": ["No"] * 10,
        "OnlineBackup": ["No"] * 10,
        "DeviceProtection": ["No"] * 10,
        "TechSupport": ["No"] * 10,
        "StreamingTV": ["No"] * 10,
        "StreamingMovies": ["No"] * 10,
        "Contract": ["Month-to-month"] * 5 + ["Two year"] * 5,
        "PaperlessBilling": ["Yes"] * 10,
        "PaymentMethod": ["Electronic check"] * 10,
        "MonthlyCharges": [25.0, 85.0, 70.0, 95.0, 20.0, 110.0, 65.0, 45.0, 80.0, 55.0],
        "TotalCharges": [300.0, 85.0, 1680.0, 285.0, 960.0, 660.0, 2340.0, 810.0, 160.0, 3960.0],
        "Churn": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    })
    clean_data.to_csv(clean_csv, index=False)
    
    # Mock preprocessor pipeline.pkl using our picklable class
    feature_names = ["numeric__tenure", "numeric__MonthlyCharges", "numeric__num_services"]
    transformed_data = np.random.randn(8, 3)
    mock_preprocessor = PicklableMockPreprocessor(feature_names, transformed_data)
    
    pipeline_pkl = os.path.join(temp_output_dir, "pipeline.pkl")
    with open(pipeline_pkl, "wb") as f:
        pickle.dump(mock_preprocessor, f)
        
    # Patch config loader to return our temp paths
    monkeypatch.setitem(config_loader.training["data_paths"], "clean_data", clean_csv)
    monkeypatch.setitem(config_loader.training["data_paths"], "artifacts_dir", temp_output_dir)
    
    # Override config segmentation continuous features list
    monkeypatch.setitem(config_loader.model, "segmentation", {
        "n_clusters": 2,
        "continuous_features": ["numeric__tenure", "numeric__MonthlyCharges", "numeric__num_services"]
    })
    
    # Run K-Means segment clustering
    results = run_segmentation(clean_csv, temp_output_dir, n_clusters=2, random_seed=42)
    
    assert "metrics" in results
    assert os.path.exists(results["model_path"])
    assert os.path.exists(results["metrics_path"])
    
    # Verify persona mapping and k-selection outputs are saved (Issue 7 & 8)
    assert os.path.exists(os.path.join(temp_output_dir, "metrics", "k_selection_analysis.png"))
    assert os.path.exists(os.path.join(temp_output_dir, "metrics", "kmeans_personas.md"))
    
    # Validate metrics layout
    metrics = results["metrics"]
    assert metrics["n_clusters"] == 2
    assert "silhouette_score" in metrics
    assert "davies_bouldin_index" in metrics
    assert "cluster_0" in metrics["cluster_sizes"]
