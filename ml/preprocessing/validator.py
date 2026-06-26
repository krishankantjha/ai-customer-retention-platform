"""
Data Validation and Quality Checks.
Performs schema validation, duplicate checks, data type verification, and target distribution logs.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

# Add the project root to python path to support configs
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.dataset_config import config_loader

class DataValidator:
    """
    Data Validation and Quality Checks.
    Performs schema validation, duplicate checks, data type verification,
    blank/whitespace values scanning, domain audits, and bounds testing.
    """
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("ml.preprocessing.validator")

    def validate_schema(self, df: pd.DataFrame, strict: bool = False, exact: bool = False) -> bool:
        """
        Validates the columns of the input dataset against the configured feature schema.
        Handles optional target presence checks and handles strict alignment validation.

        Parameters:
            df (pd.DataFrame): The customer dataset to validate.
            strict (bool): Whether to raise a ValueError if any expected columns are missing.
            exact (bool): Whether to enforce that only expected columns are present.

        Returns:
            bool: True if the dataset matches the schema parameters, False otherwise.
        """
        feature_cfg = config_loader.feature
        expected_cols = [feature_cfg["key_column"]] + \
                        feature_cfg["categorical_columns"] + \
                        feature_cfg["binary_columns"] + \
                        feature_cfg["numeric_columns"]
        
        # Include Churn if present in dataframe (unlabeled test records might lack it)
        if feature_cfg["target_column"] in df.columns or "Churn" in df.columns:
            if feature_cfg["target_column"] not in expected_cols:
                expected_cols.append(feature_cfg["target_column"])

        self.logger.info(f"Validating schema — expecting columns: {expected_cols}")
        
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            err_msg = f"Schema Validation Error: Expected column(s) {missing} are missing."
            self.logger.error(err_msg)
            if strict:
                raise ValueError(err_msg)
            return False

        if exact:
            extra = [col for col in df.columns if col not in expected_cols]
            if extra:
                err_msg = f"Schema Validation Error: Unexpected extra column(s) {extra} found."
                self.logger.error(err_msg)
                if strict:
                    raise ValueError(err_msg)
                return False
            
        self.logger.info("Schema validation passed: all expected columns present")
        return True

    def validate_missing_values(self, df: pd.DataFrame, strict: bool = False) -> dict:
        """
        Scan and report columns with null/NaN values.
        
        Parameters:
            df (pd.DataFrame): Input dataset.
            strict (bool): If True, raises ValueError if missing values exist.
            
        Returns:
            dict: Dictionary with column names as keys and missing count as values.
        """
        self.logger.info("Scanning for null/NaN values...")
        null_counts = df.isnull().sum().to_dict()
        null_report = {col: int(count) for col, count in null_counts.items() if count > 0}
        
        if null_report:
            err_msg = f"Data Quality Error: Null/NaN values found in columns: {null_report}"
            self.logger.warning(err_msg)
            if strict:
                raise ValueError(err_msg)
        else:
            self.logger.info("No null/NaN values found")
            
        return null_report

    def validate_duplicates(self, df: pd.DataFrame, strict: bool = False) -> dict:
        """
        Locate duplicate records and duplicate customerID values.
        
        Parameters:
            df (pd.DataFrame): Input dataset.
            strict (bool): If True, raises ValueError on duplicate detection.
            
        Returns:
            dict: Report containing duplicate rows and duplicate key counts.
        """
        self.logger.info("Scanning for duplicate rows and customerID keys...")
        feature_cfg = config_loader.feature
        key_col = feature_cfg["key_column"]
        
        row_dups = int(df.duplicated().sum())
        key_dups = 0
        if key_col in df.columns:
            key_dups = int(df[key_col].duplicated().sum())

        dup_report = {
            "duplicate_rows": row_dups,
            "duplicate_keys": key_dups
        }

        if row_dups > 0 or key_dups > 0:
            err_msg = f"Data Quality Error: Duplicates detected: {dup_report}"
            self.logger.warning(err_msg)
            if strict:
                raise ValueError(err_msg)
        else:
            self.logger.info("Duplicate checks passed")

        return dup_report

    def validate_blank_strings(self, df: pd.DataFrame, strict: bool = False) -> dict:
        """
        Scan object/string columns for empty spaces or blank whitespace values,
        safely ignoring None or NaN values to avoid casting to "None" or "nan".
        
        Parameters:
            df (pd.DataFrame): Input dataset.
            strict (bool): If True, raises ValueError if blank string cells are found.
            
        Returns:
            dict: Dictionary reporting blank string counts per column.
        """
        self.logger.info("Scanning for blank/whitespace values in string columns...")
        blank_report = {}
        string_cols = df.select_dtypes(include=["object", "string"]).columns
        
        for col in string_cols:
            # Filter out true nulls/NaNs before applying string validation to avoid false positives
            non_null_series = df[col].dropna()
            blank_mask = non_null_series.astype(str).str.strip() == ""
            blank_count = int(blank_mask.sum())
            if blank_count > 0:
                blank_report[col] = blank_count
                self.logger.warning(f"Column '{col}' has {blank_count} blank/whitespace value(s)")
                
        if blank_report:
            err_msg = f"Data Quality Error: Blank or whitespace-only values found: {blank_report}"
            if strict:
                raise ValueError(err_msg)
        else:
            self.logger.info("No blank/whitespace values found in string columns")
            
        return blank_report

    def validate_data_types(self, df: pd.DataFrame, strict: bool = False) -> bool:
        """
        Verify post-cleaning data type correctness for all configured columns dynamically.
        
        Parameters:
            df (pd.DataFrame): Input dataset.
            strict (bool): If True, raises ValueError on data type mismatch.
            
        Returns:
            bool: True if data types are correct, raises ValueError on error.
        """
        self.logger.info("Verifying data types dynamically...")
        feature_cfg = config_loader.feature
        
        # Verify all configured numeric columns are numeric subdtypes
        numeric_cols = feature_cfg.get("numeric_columns", [])
        for col in numeric_cols:
            if col in df.columns:
                # Remove true nulls before check if any, or check subdtype directly
                if not np.issubdtype(df[col].dtype, np.number):
                    err_msg = f"Validation Error: Expected numeric type for column '{col}', got {df[col].dtype}"
                    self.logger.error(err_msg)
                    if strict:
                        raise ValueError(err_msg)
                    return False

        # Verify SeniorCitizen is integer/numeric specifically
        if "SeniorCitizen" in df.columns:
            if not np.issubdtype(df["SeniorCitizen"].dtype, np.number):
                err_msg = f"Validation Error: Expected SeniorCitizen to be numeric, got {df['SeniorCitizen'].dtype}"
                self.logger.error(err_msg)
                if strict:
                    raise ValueError(err_msg)
                return False

        self.logger.info("Data type check passed")
        return True

    def validate_categorical_domains(self, df: pd.DataFrame, strict: bool = False) -> dict:
        """
        Validate that string/categorical column values fall within expected category levels.
        
        Parameters:
            df (pd.DataFrame): Input dataset.
            strict (bool): If True, raises ValueError on categorical violation.
            
        Returns:
            dict: Report containing any invalid categories found per column.
        """
        self.logger.info("Verifying categorical domains...")
        feature_cfg = config_loader.feature
        domains = feature_cfg.get("categorical_domains", {})
        
        invalid_report = {}
        for col, allowed_levels in domains.items():
            if col in df.columns:
                # Drop null values for validation, as nulls are audited via validate_missing_values
                series_clean = df[col].dropna()
                
                # Check for values not in allowed list
                invalid_mask = ~series_clean.isin(allowed_levels)
                invalid_values = series_clean[invalid_mask].unique().tolist()
                
                if invalid_values:
                    invalid_report[col] = invalid_values
                    self.logger.warning(f"Column '{col}' contains invalid categorical values: {invalid_values}")
                    
        if invalid_report:
            err_msg = f"Validation Error: Categorical domain boundary violation(s): {invalid_report}"
            self.logger.error(err_msg)
            if strict:
                raise ValueError(err_msg)
        else:
            self.logger.info("Categorical domain validation passed")
            
        return invalid_report

    def validate_value_bounds(self, df: pd.DataFrame, strict: bool = False) -> bool:
        """
        Verify values do not violate physical boundaries (e.g. negative charges/tenures).
        Also checks that SeniorCitizen values are subset of {0, 1}.
        
        Parameters:
            df (pd.DataFrame): Input dataset.
            strict (bool): If True, raises ValueError on boundary check violation.
            
        Returns:
            bool: True if values are within bounds, raises ValueError on error.
        """
        self.logger.info("Verifying value bounds...")
        
        # 1. Check for negative bounds in continuous metrics
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    err_msg = f"Validation Error: Found {neg_count} negative value(s) in column '{col}'"
                    self.logger.error(err_msg)
                    if strict:
                        raise ValueError(err_msg)
                    return False

        # 2. Check that SeniorCitizen contains only 0 or 1
        if "SeniorCitizen" in df.columns:
            unique_sc = set(df["SeniorCitizen"].dropna().unique())
            if not unique_sc.issubset({0, 1}):
                err_msg = f"Validation Error: SeniorCitizen contains unexpected values {unique_sc}"
                self.logger.error(err_msg)
                if strict:
                    raise ValueError(err_msg)
                return False

        self.logger.info("Value bounds verification passed")
        return True
