import os
import sys
import logging
import logging.config
import yaml
import pandas as pd

# Add the project root to the python path to support importing configs
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from configs.dataset_config import config_loader
from ml.preprocessing.loader import DataLoader
from ml.preprocessing.validator import DataValidator

def load_raw_data(path, logger):
    """Load raw telecom churn dataset from CSV utilizing canonical DataLoader."""
    loader = DataLoader(logger)
    return loader.load_from_csv(path)

def validate_schema(df, logger):
    """Validate that all expected columns are present utilizing canonical DataValidator."""
    validator = DataValidator(logger)
    validator.validate_schema(df, strict=True)

def fix_whitespace_blanks(df, logger):
    """Scan string columns for whitespace/blank values, stripping spaces and replacing with NA."""
    logger.info("Scanning for whitespace/blank values in string columns...")
    df_copy = df.copy()
    object_cols = df_copy.select_dtypes(include=["object", "string"]).columns
    
    for col in object_cols:
        df_copy[col] = df_copy[col].astype(str).str.strip()
        blanks_mask = df_copy[col] == ""
        blanks_count = blanks_mask.sum()
        if blanks_count > 0:
            logger.warning(f"Column '{col}': {blanks_count} blank/whitespace value(s) found")
            df_copy.loc[blanks_mask, col] = pd.NA
            
    logger.info("Whitespace scan complete")
    return df_copy

def fix_total_charges(df, logger):
    """Convert TotalCharges to float64 and impute 0.0 where tenure == 0 and TotalCharges is NaN."""
    logger.info("Converting 'TotalCharges' from object to float64...")
    df_copy = df.copy()
    
    # Convert to numeric, coercing any blank strings (which are now pd.NA) to NaN
    df_copy["TotalCharges"] = pd.to_numeric(df_copy["TotalCharges"], errors="coerce")
    
    # Impute 0.0 only where tenure == 0 AND TotalCharges is NaN
    zero_tenure_nan_mask = (df_copy["tenure"] == 0) & df_copy["TotalCharges"].isna()
    impute_count = zero_tenure_nan_mask.sum()
    
    if impute_count > 0:
        df_copy.loc[zero_tenure_nan_mask, "TotalCharges"] = 0.0
        logger.info(f"Imputed TotalCharges = 0.0 for {impute_count} row(s) where tenure == 0 and TotalCharges is NaN")
    
    # Verify there are no remaining missing values in TotalCharges
    remaining_nan = df_copy["TotalCharges"].isna().sum()
    if remaining_nan > 0:
        raise ValueError(f"Data Quality Error: {remaining_nan} rows still contain NaN values in 'TotalCharges' after imputation.")
    else:
        logger.info("'TotalCharges' conversion complete — no remaining NaN values")
        
    return df_copy

def drop_duplicates(df, logger):
    """Drop duplicate records from DataFrame explicitly."""
    initial_rows = df.shape[0]
    df_clean = df.drop_duplicates(keep="first")
    dropped_count = initial_rows - df_clean.shape[0]
    
    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count} duplicate rows.")
    else:
        logger.info("Duplicate check passed: 0 duplicate rows found")
        
    return df_clean

def validate_cleaned_data(df, logger):
    """Verify data types, value bounds, and target distribution after cleaning utilizing canonical DataValidator."""
    validator = DataValidator(logger)
    validator.validate_data_types(df, strict=True)
    validator.validate_value_bounds(df, strict=True)
    validator.validate_categorical_domains(df, strict=True)
    
    # Log target class distribution
    churn_counts = df["Churn"].value_counts(dropna=False)
    churn_perc = df["Churn"].value_counts(normalize=True, dropna=False) * 100
    distribution_log = ", ".join([f"{val}: {count} ({perc:.2f}%)" for val, count, perc in zip(churn_counts.index, churn_counts.values, churn_perc.values)])
    logger.info(f"Churn Target Class Distribution: {distribution_log}")
    logger.info(f"Post-cleaning validation passed. Shape: {df.shape[0]} rows, {df.shape[1]} columns")

def save_clean_data(df, path, logger):
    """Save cleaned dataset to destination path."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        logger.error(f"Error saving clean dataset to {path}: {e}")
        raise
    logger.info(f"Clean dataset saved to: {path} ({df.shape[0]} rows, {df.shape[1]} columns)")

def clean_pipeline(raw_path=None, processed_path=None, logger=None):
    """Main orchestrator for the data cleaning pipeline."""
    if logger is None:
        logger = logging.getLogger("ml.preprocessing.clean")
        
    logger.info("RetainIQ — Data Cleaning Pipeline START")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Load paths from config loader
    config_raw_path = config_loader.training["data_paths"]["raw_data"]
    config_clean_path = config_loader.training["data_paths"]["clean_data"]
    
    # Resolve relative paths against base_dir (project root)
    if raw_path is None:
        raw_path = config_raw_path if os.path.isabs(config_raw_path) else os.path.join(base_dir, config_raw_path)
    if processed_path is None:
        processed_path = config_clean_path if os.path.isabs(config_clean_path) else os.path.join(base_dir, config_clean_path)
        
    try:
        df = load_raw_data(raw_path, logger)
        validate_schema(df, logger)
        df_clean = fix_whitespace_blanks(df, logger)
        df_clean = fix_total_charges(df_clean, logger)
        df_clean = drop_duplicates(df_clean, logger)
        validate_cleaned_data(df_clean, logger)
        save_clean_data(df_clean, processed_path, logger)
    except Exception as e:
        logger.error(f"Error during preprocessing pipeline: {e}")
        raise
        
    logger.info("RetainIQ — Data Cleaning Pipeline COMPLETE")
    return df_clean

if __name__ == "__main__":
    # Configure logging globally only when run as main script
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "logging_config.yaml"))
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
        except Exception as e:
            logging.basicConfig(level=logging.INFO)
            logging.getLogger("ml.preprocessing.clean").warning(
                f"Failed to load logging config from {config_path}, fallback to basicConfig: {e}"
            )
    else:
        logging.basicConfig(level=logging.INFO)
        
    main_logger = logging.getLogger("ml.preprocessing.clean")
    clean_pipeline(logger=main_logger)
