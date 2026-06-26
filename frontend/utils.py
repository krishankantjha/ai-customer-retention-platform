def get_risk_cat(p: float) -> str:
    """Assigns continuous churn probabilities to business-friendly risk categories."""
    return "High Risk" if p >= 0.50 else "Medium Risk" if p >= 0.25 else "Low Risk"
