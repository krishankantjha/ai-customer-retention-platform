"""
Unit and Integration Tests for Phase 5: Autoencoder Representation & Counterfactual Simulations.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force the backend directory into path to allow imports of app
backend_dir = os.path.abspath(os.path.join(project_root, "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ml.segmentation.autoencoder import AutoencoderWrapper
from ml.explainability.shap_local import LocalExplainer


# Define picklable mock objects at the module level
class MockClassifier:
    def __init__(self):
        self.feature_names_in_ = [
            "numeric__tenure",
            "numeric__MonthlyCharges",
            "numeric__addon_count",
            "numeric__commitment_score",
            "numeric__Contract",
            "numeric__AvgMonthlyCharge",
            "numeric__num_services"
        ]

    def predict_proba(self, X):
        # Return probability depending on Contract features to simulate risk reduction
        probs = []
        for idx in range(len(X)):
            row = X.iloc[idx]
            is_long_contract = False
            if "numeric__Contract" in X.columns and row["numeric__Contract"] > 0:
                is_long_contract = True
            
            if is_long_contract:
                probs.append([0.9, 0.1])
            else:
                probs.append([0.2, 0.8])
        return np.array(probs)


class MockPreprocessorForTest:
    def __init__(self):
        self.feature_names_out = [
            "numeric__tenure",
            "numeric__MonthlyCharges",
            "numeric__addon_count",
            "numeric__commitment_score",
            "numeric__Contract",
            "numeric__AvgMonthlyCharge",
            "numeric__num_services"
        ]

    def get_feature_names_out(self):
        return np.array(self.feature_names_out)

    def transform(self, X):
        print("\nDEBUG transform X columns:", list(X.columns))
        print("DEBUG transform X values:\n", X.to_dict(orient="records"))
        res = np.zeros((len(X), len(self.feature_names_out)))
        if "Contract" in X.columns:
            for idx, contract_val in enumerate(X["Contract"]):
                print("DEBUG idx:", idx, "contract_val:", contract_val, "type:", type(contract_val))
                if contract_val == "One year" or contract_val == 12 or contract_val == 12.0:
                    res[idx, 4] = 1.0 # numeric__Contract column
        return res


def test_autoencoder_wrapper_fit_transform():
    """
    Test 1: Verify deep MLP autoencoder fits and projects continuous data to 16 dimensions.
    The new deep architecture (64→16→64) requires more samples to converge.
    For random Gaussian data (no structure), reconstruction MSE of ~1.0 is expected
    since the autoencoder cannot compress random noise efficiently.
    The quality gate is set to 2.0 to catch catastrophic training failures.
    """
    # Use 300 samples so the deep 3-hidden-layer AE has sufficient data to fit
    np.random.seed(42)
    X = np.random.randn(300, 54).astype(np.float32)

    autoencoder = AutoencoderWrapper(latent_dim=16, random_seed=42, mse_threshold=2.0)
    mse = autoencoder.fit(X)

    # Assert MSE is a positive float (validation set MSE, not training MSE)
    assert isinstance(mse, float)
    assert mse >= 0.0, f"Validation MSE must be non-negative, got {mse:.4f}"
    assert mse < 2.0, f"Validation MSE ({mse:.4f}) should be below the 2.0 catastrophic failure threshold"

    # Transform data and check dimensions
    X_latent = autoencoder.transform(X)
    assert X_latent.shape == (300, 16), (
        f"Expected latent shape (300, 16), got {X_latent.shape}"
    )
    assert np.all(X_latent >= 0.0), "ReLU output constraint violated: latent values must be non-negative"


def test_local_explainer_counterfactual_simulations():
    """
    Test 2: Confirm simulating a contract upgrade outputs a valid predicted probability
    and reduces risk.  Artifacts are now passed directly to the LocalExplainer
    constructor (FIX CRITICAL-3: circular dependency removed — no monkeypatching needed).
    """
    # Mock dataset
    customer_dict = {
        "customerID": "1234-ABCD",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 210.0,
        "Churn": "Yes"
    }
    customer_df = pd.DataFrame([customer_dict])

    mock_model = MockClassifier()
    mock_preprocessor = MockPreprocessorForTest()

    encoders = {
        "train_monthly_charges_median": 70.0,
        "feature_names_out": mock_preprocessor.feature_names_out
    }
    metadata = {
        "feature_names_in": mock_model.feature_names_in_
    }

    # FIX CRITICAL-3: Pass artifacts directly to LocalExplainer constructor.
    # No monkeypatching of prediction_service is required — the ML module is
    # now fully self-contained and independently testable.
    explainer = LocalExplainer(
        mock_model,
        mock_model.feature_names_in_,
        explainer=MagicMock(),
        preprocessor=mock_preprocessor,
        encoders=encoders,
        metadata=metadata,
    )

    # 1. Test simulate_intervention for contract upgrade
    prob_mtm = explainer.simulate_intervention(customer_df, {"Contract": "Month-to-month"})
    prob_oneyear = explainer.simulate_intervention(customer_df, {"Contract": "One year"})

    # Ensure simulated probability is valid
    assert 0.0 <= prob_mtm <= 1.0
    assert 0.0 <= prob_oneyear <= 1.0

    # Under our MockClassifier, upgrading to One year reduces risk from 0.8 to 0.1
    assert prob_oneyear < prob_mtm
    assert prob_oneyear == 0.1
    assert prob_mtm == 0.8

    # 2. Test run_simulations
    sims = explainer.run_simulations(customer_df)
    assert len(sims) > 0

    # Assert keys exist in simulation detail
    sim = sims[0]
    assert "intervention" in sim
    assert "original_risk" in sim
    assert "simulated_risk" in sim
    assert "risk_reduction" in sim
    assert sim["risk_reduction"] > 0.0
