"""
Feature engineering module for RetainIQ.

This file defines the feature engineering logic, transforming clean data into features for training/inference.
It is imported by pipeline.py, which runs the full preprocessing. You can also run this file directly to test it.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger("ml.preprocessing.engineer")

# Constants for column lists and feature configuration.
# Keeping these here so they can be reused by pipeline.py.

# Internet add-on columns for addon_count
INTERNET_ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Protection services for security_over_streaming
PROTECTION_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection"]

# Streaming services for security_over_streaming
STREAMING_COLS = ["StreamingTV", "StreamingMovies"]

# Automatic payment options
AUTO_PAY_METHODS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

# Bins and labels to group tenure into categories
TENURE_BINS   = [0, 12, 24, 48, float("inf")]
TENURE_LABELS = ["0-12", "12-24", "24-48", "48+"]

# Columns to convert from Yes/No to 1/0 (must be done after engineering features)
BINARY_YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# Columns to drop because they are IDs or redundant
DROP_COLS = ["customerID", "TotalCharges"]


def engineer_features(df: pd.DataFrame, monthly_charges_median: float) -> pd.DataFrame:
    """
    Applies feature engineering to the clean dataset.
    
    To avoid data leakage, the monthly_charges_median should be computed 
    on the training set and passed here for both train and test splits.
    """
    if not isinstance(monthly_charges_median, (int, float)):
        raise TypeError(
            f"monthly_charges_median must be numeric, got {type(monthly_charges_median)}"
        )

    required_cols = [
        "Contract", "tenure", "PaymentMethod", "InternetService",
        "MonthlyCharges", "Partner", "Dependents",
    ] + INTERNET_ADDON_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    logger.info(f"Starting feature engineering on {df.shape[0]:,} rows.")

    # Create the engineered features (done on raw string values before mapping to binary)
    
    # 1. Contract is month-to-month
    # Month-to-month contracts are highly correlated with churn.
    df["contract_is_mtm"] = (df["Contract"] == "Month-to-month").astype(int)

    # 2. Tenure bucket
    # Categorizes tenure into buckets for lifecycle modeling.
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
    ).astype(str)

    # 3. Early stage flag
    # Flag for customers with 12 months or less of tenure.
    df["is_early_stage"] = (df["tenure"] <= 12).astype(int)

    # 4. Auto-pay flag
    # Customers using automatic payment methods.
    df["auto_pay_flag"] = (
        df["PaymentMethod"].isin(AUTO_PAY_METHODS)
    ).astype(int)

    # 5. Addon count
    # Count of internet add-ons active for the customer.
    df["addon_count"] = df[INTERNET_ADDON_COLS].eq("Yes").sum(axis=1)

    # 6. Tech support flag
    # Flag to isolate tech support usage.
    df["has_support"] = (df["TechSupport"] == "Yes").astype(int)

    # 7. Security over streaming
    # Checks if user has more security/backup add-ons than streaming ones.
    _streaming_count  = df[STREAMING_COLS].eq("Yes").sum(axis=1)
    _protection_count = df[PROTECTION_COLS].eq("Yes").sum(axis=1)
    df["security_over_streaming"] = (_protection_count > _streaming_count).astype(int)

    # 8. Month-to-month and early stage flag
    # High-risk segment combining month-to-month contracts and short tenure.
    df["contract_early_stage_flag"] = (
        (df["Contract"] == "Month-to-month") & (df["tenure"] <= 12)
    ).astype(int)

    # 9. Commitment score
    # Composite score based on long-term contract, auto-pay, and tenure > 12 months.
    # We use evidence-based weights derived from Logistic Regression coefficients.
    df["commitment_score"] = (
        1.96 * (df["Contract"] != "Month-to-month").astype(float)
        + 0.46 * df["PaymentMethod"].isin(AUTO_PAY_METHODS).astype(float)
        + 0.58 * (df["tenure"] > 12).astype(float)
    )

    # 10. Premium risk flag
    # High monthly charges for fiber optic customers.
    # Uses the training median passed in to prevent data leakage.
    df["premium_risk_flag"] = (
        (df["InternetService"] == "Fiber optic")
        & (df["MonthlyCharges"] > monthly_charges_median)
    ).astype(int)

    # 11. Household stability flag
    # Active if customer has both partner and dependents.
    # Must compute on raw strings before converting Partner/Dependents to binary.
    if pd.api.types.is_numeric_dtype(df["Partner"]) and pd.api.types.is_numeric_dtype(df["Dependents"]):
        df["household_stability_flag"] = (
            (df["Partner"] == 1) & (df["Dependents"] == 1)
        ).astype(int)
    else:
        df["household_stability_flag"] = (
            (df["Partner"] == "Yes") & (df["Dependents"] == "Yes")
        ).astype(int)

    # 12. Fiber zero engagement flag
    # Fiber optic customers with no active internet addons.
    _addons = df[INTERNET_ADDON_COLS].eq("Yes").sum(axis=1)
    df["fiber_zero_engagement_flag"] = (
        (df["InternetService"] == "Fiber optic") & (_addons == 0)
    ).astype(int)

    # 13. High charge early stage flag
    # High monthly charges and tenure <= 12 months.
    df["high_charge_early_stage_flag"] = (
        (df["MonthlyCharges"] > monthly_charges_median) & (df["tenure"] <= 12)
    ).astype(int)

    # 14. Vulnerable customer flag
    # Senior citizens living alone early in their contract.
    if pd.api.types.is_numeric_dtype(df["Partner"]) and pd.api.types.is_numeric_dtype(df["Dependents"]):
        _vuln_cond = (df["Partner"] == 0) & (df["Dependents"] == 0)
    else:
        _vuln_cond = (df["Partner"] == "No") & (df["Dependents"] == "No")

    df["vulnerable_customer_flag"] = (
        (df["SeniorCitizen"] == 1)
        & _vuln_cond
        & (df["tenure"] <= 12)
    ).astype(int)

    # 15. AvgMonthlyCharge
    # Spending intensity per month of tenure. Prevent division-by-zero using tenure + 1.
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 16. num_services
    # Sum of active client-facing subscriptions.
    if pd.api.types.is_numeric_dtype(df["PhoneService"]):
        _phone_sub = (df["PhoneService"] == 1).astype(int)
    else:
        _phone_sub = (df["PhoneService"] == "Yes").astype(int)

    if pd.api.types.is_numeric_dtype(df["InternetService"]):
        _internet_sub = (df["InternetService"] != 0).astype(int)
    else:
        _internet_sub = (df["InternetService"] != "No").astype(int)

    _backup_sub = df["OnlineBackup"].eq("Yes").astype(int)
    _support_sub = df["TechSupport"].eq("Yes").astype(int)
    _streaming_sub = df[["StreamingTV", "StreamingMovies"]].eq("Yes").sum(axis=1)

    df["num_services"] = _phone_sub + _internet_sub + _backup_sub + _support_sub + _streaming_sub

    # 17. Ordinal Contract mapping
    # Month-to-month -> 1, One year -> 12, Two year -> 24 to preserve monotonic signal.
    # Note: done after other features that use Contract raw string values.
    contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    if not pd.api.types.is_numeric_dtype(df["Contract"]):
        df["Contract"] = df["Contract"].map(contract_map)

    logger.info("All engineered features created.")

    # Map Yes/No columns to 1/0
    for col in BINARY_YES_NO_COLS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = (df[col] == "Yes").astype(int)

    logger.info(f"Binary columns mapped: {BINARY_YES_NO_COLS}")

    # Map Churn target to 1/0 if it exists in the data
    if "Churn" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["Churn"]):
            df["Churn"] = (df["Churn"] == "Yes").astype(int)
            logger.info("Churn target mapped: Yes -> 1, No -> 0")

    # Drop customer ID and TotalCharges to prevent collinearity
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped columns: {cols_to_drop}")

    logger.info(
        f"Feature engineering complete. Output shape: {df.shape[0]:,} rows, "
        f"{df.shape[1]} columns."
    )

    return df


# Smoke test running block. Run directly to test the module: python -m ml.preprocessing.engineer
if __name__ == "__main__":
    import sys
    import logging.config
    import yaml

    base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    from configs.dataset_config import config_loader

    config_path = os.path.join(base_dir, "configs", "logging_config.yaml")
    config_clean_path = config_loader.training["data_paths"]["clean_data"]
    clean_path = config_clean_path if os.path.isabs(config_clean_path) else os.path.join(base_dir, config_clean_path)

    # Set up logging
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s")

    log = logging.getLogger("ml.preprocessing.engineer")
    log.info("Starting engineer.py test run")

    # Load cleaned dataset
    df_raw = pd.read_csv(clean_path)
    log.info(f"Loaded clean data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Estimate median MonthlyCharges for testing
    mock_median = df_raw["MonthlyCharges"].median()
    log.info(f"Median monthly charges for testing: {mock_median:.2f}")

    # Run feature engineering
    df_engineered = engineer_features(df_raw, monthly_charges_median=mock_median)

    # Print validation information
    engineered_cols = [
        "contract_is_mtm", "tenure_bucket", "is_early_stage", "auto_pay_flag",
        "addon_count", "has_support", "security_over_streaming",
        "contract_early_stage_flag", "commitment_score",
        "premium_risk_flag", "household_stability_flag",
        "fiber_zero_engagement_flag", "high_charge_early_stage_flag", "vulnerable_customer_flag",
        "AvgMonthlyCharge", "num_services", "Contract"
    ]
    print("\nOutput Details:")
    print(f"  Rows    : {df_engineered.shape[0]}")
    print(f"  Columns : {df_engineered.shape[1]}")
    print("\nFeature Samples:")
    print(df_engineered[engineered_cols].head(5).to_string())
    print("\nChecking for missing values:")
    nan_count = df_engineered.isna().sum().sum()
    print(f"  Total NaNs: {nan_count}")
    assert nan_count == 0, "Found NaNs in engineered dataframe"
    print("\nTest passed. Ready for pipeline integration.")
