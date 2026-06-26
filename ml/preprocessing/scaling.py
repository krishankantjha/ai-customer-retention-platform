"""
Numerical Standardization Layer.
Scales continuous numeric columns using StandardScaler.
"""

from sklearn.preprocessing import StandardScaler

def get_numeric_scaler() -> StandardScaler:
    """
    Returns an unfitted StandardScaler for numerical standardization.
    Ensures zero mean and unit variance.
    """
    return StandardScaler()
