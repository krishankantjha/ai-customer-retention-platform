"""
Model Explainability & Actionable Save Plays Service.
Backward-compatible wrapper delegating to the modular LocalExplainer class.
"""

import os
import sys
import pandas as pd

# Add project root to path to load configs
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.explainability.shap_local import LocalExplainer


def explain_customer_churn(customer_df: pd.DataFrame, model, feature_names_in: list, top_n: int = 3) -> dict:
    """
    Backward-compatible wrapper for explain_customer_churn delegating to the modular LocalExplainer.
    """
    explainer = LocalExplainer(model, feature_names_in)
    return explainer.explain_customer(customer_df, top_n=top_n)
