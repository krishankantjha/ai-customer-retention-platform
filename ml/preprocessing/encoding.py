"""
Encoding Preprocessing Layer.
Manages categorical transformations using Ordinal and One-Hot encoders.
"""

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def get_categorical_encoder() -> OneHotEncoder:
    """
    Returns an unfitted OneHotEncoder configured for inference resilience.
    Ignores unknown values during prediction time.
    """
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

def get_ordinal_encoder(categories: list) -> OrdinalEncoder:
    """
    Returns an unfitted OrdinalEncoder configured for handling unknown categories during inference.
    Maps unseen values to -1.
    """
    return OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)
