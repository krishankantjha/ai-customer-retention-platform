import os
import sys
import pytest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from unittest.mock import patch, MagicMock

# Add the project root to python path to support imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.preprocessing.loader import DataLoader
from ml.preprocessing.validator import DataValidator

@pytest.fixture
def sample_data():
    """Generates a valid raw mock DataFrame."""
    return pd.DataFrame({
        "customerID": ["1111-AAAA", "2222-BBBB", "3333-CCCC"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "Yes", "No"],
        "tenure": [12, 24, 6],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["Yes", "No", "Yes"],
        "OnlineBackup": ["No", "Yes", "No"],
        "DeviceProtection": ["Yes", "No", "No"],
        "TechSupport": ["No", "Yes", "Yes"],
        "StreamingTV": ["Yes", "No", "No"],
        "StreamingMovies": ["No", "Yes", "No"],
        "Contract": ["One year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [70.5, 95.0, 45.2],
        "TotalCharges": [846.0, 2280.0, 271.2],
        "Churn": ["No", "Yes", "No"]
    })

@pytest.fixture
def temp_db_path(tmp_path):
    """Fixture creating a temporary SQLite database path."""
    return str(tmp_path / "test_retention.db")

def test_load_from_csv(tmp_path, sample_data):
    # Save raw to CSV
    csv_file = tmp_path / "raw_telco.csv"
    sample_data.to_csv(csv_file, index=False)
    
    loader = DataLoader()
    df = loader.load_from_csv(str(csv_file))
    
    assert df.shape == (3, 21)
    assert df["customerID"].tolist() == ["1111-AAAA", "2222-BBBB", "3333-CCCC"]

def test_load_from_csv_file_not_found():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_from_csv("non_existent_file.csv")

def test_load_from_csv_chunked(tmp_path, sample_data):
    # Save raw to CSV
    csv_file = tmp_path / "raw_telco.csv"
    sample_data.to_csv(csv_file, index=False)
    
    loader = DataLoader()
    chunk_gen = loader.load_from_csv(str(csv_file), chunksize=2)
    
    chunks = list(chunk_gen)
    assert len(chunks) == 2
    assert chunks[0].shape == (2, 21)
    assert chunks[1].shape == (1, 21)

def test_load_from_csv_malformed(tmp_path):
    # Save a corrupt non-CSV file
    corrupt_file = tmp_path / "corrupt.csv"
    with open(corrupt_file, "w") as f:
        f.write("a,b,c\n1,2\n3,4,5,6")
        
    loader = DataLoader()
    # Pandas parser will raise an error or parse badly
    with pytest.raises(Exception):
        loader.load_from_csv(str(corrupt_file))

def test_load_from_db(temp_db_path):
    # Setup test DB tables with snake_case naming
    engine = create_engine(f"sqlite:///{temp_db_path}")
    db_df = pd.DataFrame({
        "id": [1, 2],
        "customer_id": ["1111-AAAA", "2222-BBBB"],
        "gender": ["Female", "Male"],
        "senior_citizen": [0, 1],
        "partner": ["Yes", "No"],
        "dependents": ["No", "Yes"],
        "tenure": [12, 24],
        "phone_service": ["Yes", "Yes"],
        "multiple_lines": ["No", "Yes"],
        "internet_service": ["DSL", "Fiber optic"],
        "online_security": ["Yes", "No"],
        "online_backup": ["No", "Yes"],
        "device_protection": ["Yes", "No"],
        "tech_support": ["No", "Yes"],
        "streaming_tv": ["Yes", "No"],
        "streaming_movies": ["No", "Yes"],
        "contract": ["One year", "Month-to-month"],
        "paperless_billing": ["Yes", "No"],
        "payment_method": ["Electronic check", "Mailed check"],
        "monthly_charges": [70.5, 95.0],
        "total_charges": [846.0, 2280.0],
        "churn": ["No", "Yes"],
        "upload_id": [10, 10]
    })
    db_df.to_sql("customers", engine, index=False)
    engine.dispose()
    
    loader = DataLoader()
    df = loader.load_from_db(db_url=f"sqlite:///{temp_db_path}")
    
    # Check shape (dropped 'id' and 'upload_id', so columns count should go from 23 to 21)
    assert df.shape == (2, 21)
    # Check mapping
    assert "customerID" in df.columns
    assert "SeniorCitizen" in df.columns
    assert "id" not in df.columns
    assert "upload_id" not in df.columns
    assert df["customerID"].tolist() == ["1111-AAAA", "2222-BBBB"]

def test_load_from_db_connection_leak_prevention(temp_db_path):
    # Setup test DB tables with snake_case naming
    engine = create_engine(f"sqlite:///{temp_db_path}")
    db_df = pd.DataFrame({
        "id": [1],
        "customer_id": ["1111-AAAA"],
        "gender": ["Female"],
        "senior_citizen": [0],
        "partner": ["Yes"],
        "dependents": ["No"],
        "tenure": [12],
        "phone_service": ["Yes"],
        "multiple_lines": ["No"],
        "internet_service": ["DSL"],
        "online_security": ["Yes"],
        "online_backup": ["No"],
        "device_protection": ["Yes"],
        "tech_support": ["No"],
        "streaming_tv": ["Yes"],
        "streaming_movies": ["No"],
        "contract": ["One year"],
        "paperless_billing": ["Yes"],
        "payment_method": ["Electronic check"],
        "monthly_charges": [70.5],
        "total_charges": [846.0],
        "churn": ["No"],
        "upload_id": [10]
    })
    db_df.to_sql("customers", engine, index=False)
    engine.dispose()

    loader = DataLoader()
    
    # Mock create_engine to verify dispose call
    mock_engine = MagicMock()
    with patch("ml.preprocessing.loader.create_engine", return_value=mock_engine), \
         patch("pandas.read_sql_query", side_effect=Exception("Read Sql Failed")):
        with pytest.raises(Exception, match="Read Sql Failed"):
            loader.load_from_db(db_url=f"sqlite:///{temp_db_path}")
        # Verify engine.dispose() was still called in the finally block
        mock_engine.dispose.assert_called_once()

def test_load_from_db_query_failure(temp_db_path):
    loader = DataLoader()
    with pytest.raises(Exception):
        # Querying a non-existent table should raise an exception
        loader.load_from_db(db_url=f"sqlite:///{temp_db_path}", query="SELECT * FROM non_existent_table")

def test_validator_schema(sample_data):
    validator = DataValidator()
    # Valid schema check
    assert validator.validate_schema(sample_data) is True
    
    # Missing columns check
    bad_df = sample_data.drop(columns=["customerID"])
    with pytest.raises(ValueError, match="Schema Validation Error"):
        validator.validate_schema(bad_df, strict=True)

    # Exact schema violation check
    exact_bad_df = sample_data.copy()
    exact_bad_df["extra_column"] = 1
    with pytest.raises(ValueError, match="Unexpected extra column"):
        validator.validate_schema(exact_bad_df, strict=True, exact=True)

def test_validator_missing_values(sample_data):
    validator = DataValidator()
    
    # Inject missing values
    df_missing = sample_data.copy()
    df_missing.loc[1, "TotalCharges"] = np.nan
    df_missing.loc[2, "gender"] = None
    
    report = validator.validate_missing_values(df_missing)
    assert "TotalCharges" in report
    assert "gender" in report
    assert report["TotalCharges"] == 1
    assert report["gender"] == 1

    # Test strict mode exception trigger
    with pytest.raises(ValueError, match="Null/NaN values found"):
        validator.validate_missing_values(df_missing, strict=True)

def test_validator_duplicates(sample_data):
    validator = DataValidator()
    
    # Append duplicate row
    df_dups = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
    report = validator.validate_duplicates(df_dups)
    assert report["duplicate_rows"] == 1
    assert report["duplicate_keys"] == 1

    # Test strict mode exception trigger
    with pytest.raises(ValueError, match="Duplicates detected"):
        validator.validate_duplicates(df_dups, strict=True)

def test_validator_blank_strings(sample_data):
    validator = DataValidator()
    
    # Inject blank string
    df_blanks = sample_data.copy()
    df_blanks.loc[0, "PhoneService"] = "  "
    df_blanks.loc[1, "gender"] = ""
    
    report = validator.validate_blank_strings(df_blanks)
    assert "PhoneService" in report
    assert "gender" in report
    assert report["PhoneService"] == 1
    assert report["gender"] == 1

    # Test strict mode exception trigger
    with pytest.raises(ValueError, match="Blank or whitespace-only values found"):
        validator.validate_blank_strings(df_blanks, strict=True)

def test_validator_blank_strings_ignores_nans(sample_data):
    validator = DataValidator()
    df_nans = sample_data.copy()
    df_nans.loc[0, "gender"] = np.nan
    df_nans.loc[1, "gender"] = None
    
    # NaN and None should not be evaluated as blanks (since astype(str) would cast them to "nan"/"None")
    report = validator.validate_blank_strings(df_nans)
    assert "gender" not in report

def test_validator_data_types(sample_data):
    validator = DataValidator()
    assert validator.validate_data_types(sample_data) is True
    
    # Bad type check for TotalCharges
    bad_df = sample_data.copy()
    bad_df["TotalCharges"] = bad_df["TotalCharges"].astype(str)
    with pytest.raises(ValueError, match="Expected numeric type for column 'TotalCharges'"):
        validator.validate_data_types(bad_df, strict=True)

    # Bad type check for tenure (numeric check coverage validation)
    bad_df2 = sample_data.copy()
    bad_df2["tenure"] = bad_df2["tenure"].astype(str)
    with pytest.raises(ValueError, match="Expected numeric type for column 'tenure'"):
        validator.validate_data_types(bad_df2, strict=True)

def test_validator_categorical_domains(sample_data):
    validator = DataValidator()
    assert len(validator.validate_categorical_domains(sample_data)) == 0
    
    # Inject illegal categorical domain values
    bad_df = sample_data.copy()
    bad_df.loc[0, "gender"] = "Unknown"
    bad_df.loc[1, "Contract"] = "3 Years"
    
    report = validator.validate_categorical_domains(bad_df)
    assert "gender" in report
    assert "Contract" in report
    assert report["gender"] == ["Unknown"]
    assert report["Contract"] == ["3 Years"]

    # Test strict mode exception trigger
    with pytest.raises(ValueError, match="Categorical domain boundary violation"):
        validator.validate_categorical_domains(bad_df, strict=True)

def test_validator_value_bounds(sample_data):
    validator = DataValidator()
    assert validator.validate_value_bounds(sample_data) is True
    
    # Negative bound check
    bad_df = sample_data.copy()
    bad_df.loc[0, "tenure"] = -5
    with pytest.raises(ValueError, match="Found 1 negative value"):
        validator.validate_value_bounds(bad_df, strict=True)
        
    # Bad SeniorCitizen values
    bad_df2 = sample_data.copy()
    bad_df2.loc[0, "SeniorCitizen"] = 3
    with pytest.raises(ValueError, match="SeniorCitizen contains unexpected values"):
        validator.validate_value_bounds(bad_df2, strict=True)


def test_engineer_features(sample_data):
    from ml.preprocessing.engineer import engineer_features
    
    # Run feature engineering with mock MonthlyCharges median
    df_engineered = engineer_features(sample_data, monthly_charges_median=60.0)
    
    # 1. AvgMonthlyCharge checks
    # Row 0: tenure = 12, TotalCharges = 846.0 -> AvgMonthlyCharge = 846.0 / 13
    assert np.allclose(df_engineered.loc[0, "AvgMonthlyCharge"], 846.0 / 13)
    
    # 2. num_services checks
    # Row 0: PhoneService(1) + InternetService(1) + OnlineBackup(0) + TechSupport(0) + StreamingTV(1) + StreamingMovies(0) = 3
    assert df_engineered.loc[0, "num_services"] == 3
    # Row 1: PhoneService(1) + InternetService(1) + OnlineBackup(1) + TechSupport(1) + StreamingTV(0) + StreamingMovies(1) = 5
    assert df_engineered.loc[1, "num_services"] == 5
    # Row 2: PhoneService(0) + InternetService(1) + OnlineBackup(0) + TechSupport(1) + StreamingTV(0) + StreamingMovies(0) = 2
    assert df_engineered.loc[2, "num_services"] == 2

    # 3. Ordinal Contract mapping checks
    # Month-to-month -> 1, One year -> 12, Two year -> 24
    assert df_engineered.loc[0, "Contract"] == 12
    assert df_engineered.loc[1, "Contract"] == 1
    assert df_engineered.loc[2, "Contract"] == 24


def test_engineer_features_edge_cases():
    from ml.preprocessing.engineer import engineer_features
    # Create test rows covering edge cases
    df_edge = pd.DataFrame({
        "customerID": ["9999-EDGE1", "9999-EDGE2", "9999-EDGE3"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "Yes", "No"],
        "tenure": [0, 0, 10], # tenure = 0 and tenure = 10
        "PhoneService": ["Yes", "No", "Yes"],
        "MultipleLines": ["No", "No phone service", "No"],
        "InternetService": ["DSL", "DSL", "DSL"],
        "OnlineSecurity": ["Yes", "No", "Yes"],
        "OnlineBackup": ["No", "No", "No"],
        "DeviceProtection": ["Yes", "No", "No"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["Yes", "No", "No"],
        "StreamingMovies": ["No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [70.5, 95.0, 45.2],
        "TotalCharges": [0.0, 50.0, np.nan], # TotalCharges = 0, TotalCharges > 0, TotalCharges is NaN
        "Churn": ["No", "Yes", "No"]
    })

    # Run feature engineering
    df_engineered = engineer_features(df_edge, monthly_charges_median=60.0)

    # 1. tenure = 0 and TotalCharges = 0
    # AvgMonthlyCharge = 0.0 / (0 + 1) = 0.0
    assert df_engineered.loc[0, "AvgMonthlyCharge"] == 0.0

    # 2. tenure = 0 and TotalCharges > 0
    # AvgMonthlyCharge = 50.0 / (0 + 1) = 50.0
    assert df_engineered.loc[1, "AvgMonthlyCharge"] == 50.0

    # 3. tenure = 10 and TotalCharges is NaN
    # AvgMonthlyCharge should be NaN (does not raise division-by-zero)
    assert pd.isna(df_engineered.loc[2, "AvgMonthlyCharge"])


def test_preprocessing_transformers():
    from ml.preprocessing.encoding import get_categorical_encoder, get_ordinal_encoder
    from ml.preprocessing.scaling import get_numeric_scaler
    from ml.preprocessing.imbalance import resample_training_data
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    # 1. Test scaling
    scaler = get_numeric_scaler()
    assert isinstance(scaler, StandardScaler)
    
    # 2. Test encoding
    cat_enc = get_categorical_encoder()
    assert isinstance(cat_enc, OneHotEncoder)
    assert cat_enc.handle_unknown == "ignore"
    assert cat_enc.sparse_output is False
    
    ord_categories = [["0-12", "12-24", "24-48", "48+"]]
    ord_enc = get_ordinal_encoder(ord_categories)
    assert isinstance(ord_enc, OrdinalEncoder)
    assert ord_enc.categories == ord_categories
    assert ord_enc.handle_unknown == "use_encoded_value"
    assert ord_enc.unknown_value == -1

    # 3. Test SMOTE oversampling
    # Create unbalanced mock dataset with sufficient samples for SMOTE neighbors (k_neighbors=5 by default)
    X_mock = pd.DataFrame({
        "feat1": [1.0, 2.0, 1.5, 2.1, 1.2, 1.1, 0.9, 0.8, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        "feat2": [5.0, 6.0, 5.5, 6.2, 5.1, 5.2, 4.9, 4.8, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8]
    })
    y_mock = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 8 vs 6
    
    X_res, y_res = resample_training_data(X_mock, y_mock, random_seed=42)
    
    # Assert balanced target classes
    assert len(y_res) == 16  # 8 class 0, and 8 class 1 (SMOTE oversampled the 6 class 1 to 8)
    assert (y_res == 0).sum() == 8
    assert (y_res == 1).sum() == 8
    assert X_res.shape == (16, 2)


def test_robust_smote():
    from ml.preprocessing.imbalance import resample_training_data

    # 1. Minority samples < 5 (e.g. 4) -> should adjust k_neighbors to 3
    X_mock = pd.DataFrame({"feat": [1.0, 2.0, 1.5, 2.1, 1.2, 1.1, 0.9, 0.8, 1.3, 1.4, 1.5, 1.6]})
    y_mock = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # 8 vs 4
    X_res, y_res = resample_training_data(X_mock, y_mock, random_seed=42)
    assert len(y_res) == 16  # Balanced to 8 vs 8
    assert (y_res == 1).sum() == 8

    # 2. Minority samples = 2 -> should adjust k_neighbors to 1
    X_mock2 = pd.DataFrame({"feat": [1.0, 2.0, 1.5, 2.1, 1.2, 1.1, 0.9, 0.8, 1.3, 1.4]})
    y_mock2 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8 vs 2
    X_res2, y_res2 = resample_training_data(X_mock2, y_mock2, random_seed=42)
    assert len(y_res2) == 16  # Balanced to 8 vs 8
    assert (y_res2 == 1).sum() == 8

    # 3. Minority samples = 1 -> should gracefully fallback and return original data
    X_mock3 = pd.DataFrame({"feat": [1.0, 2.0, 1.5, 2.1, 1.2, 1.1, 0.9, 0.8, 1.3]})
    y_mock3 = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1])  # 8 vs 1
    X_res3, y_res3 = resample_training_data(X_mock3, y_mock3, random_seed=42)
    assert X_res3.shape == X_mock3.shape
    assert len(y_res3) == len(y_mock3)

    # 4. Single-class dataset -> should gracefully fallback and return original data
    X_mock4 = pd.DataFrame({"feat": [1.0, 2.0, 1.5]})
    y_mock4 = pd.Series([0, 0, 0])
    X_res4, y_res4 = resample_training_data(X_mock4, y_mock4, random_seed=42)
    assert X_res4.shape == X_mock4.shape
    assert len(y_res4) == len(y_mock4)

    # 5. Empty dataset -> should gracefully fallback and return original data
    X_mock5 = pd.DataFrame(columns=["feat"])
    y_mock5 = pd.Series(dtype=int)
    X_res5, y_res5 = resample_training_data(X_mock5, y_mock5, random_seed=42)
    assert X_res5.empty
    assert y_res5.empty


def test_smote_configuration_support():
    from ml.preprocessing.imbalance import resample_training_data
    from configs.dataset_config import config_loader
    
    # Verify that the default value in the config loader matches expectation (5)
    config_k_neighbors = config_loader.model.get("smote", {}).get("k_neighbors", 5)
    assert config_k_neighbors == 5

    # Verify that resample_training_data respects a custom default_k_neighbors argument.
    X_mock = pd.DataFrame({
        "feat1": [1.0, 2.0, 1.5, 2.1, 1.2, 1.1, 0.9, 0.8, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        "feat2": [5.0, 6.0, 5.5, 6.2, 5.1, 5.2, 4.9, 4.8, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8]
    })
    y_mock = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 8 vs 6
    
    # Passing default_k_neighbors=2 (less than minority samples of 6) should not throw any error and balance classes
    X_res, y_res = resample_training_data(X_mock, y_mock, random_seed=42, default_k_neighbors=2)
    assert len(y_res) == 16
    assert (y_res == 1).sum() == 8





