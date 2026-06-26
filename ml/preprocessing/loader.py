"""
Data Loader Layer.
Handles ingestion of raw customer records from CSV files and database connections.
"""

import os
import sys
import pandas as pd
import logging
from sqlalchemy import create_engine
from typing import Generator, Union

# Add the project root to the python path to support importing configs
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.dataset_config import config_loader

class DataLoader:
    """
    Handles ingestion of raw customer records from CSV files and database connections.
    """
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("ml.preprocessing.loader")

    def load_from_csv(self, csv_path: str, chunksize: int = None) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
        """
        Loads raw data from a CSV file.
        
        Parameters:
            csv_path (str): Path to the CSV file.
            chunksize (int): Optional chunk size for large CSV streaming.
            
        Returns:
            Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]: Loaded dataset or chunk generator.
        """
        self.logger.info(f"Loading raw data from CSV: {csv_path} (chunksize={chunksize})")
        if not os.path.exists(csv_path):
            self.logger.error(f"CSV file not found at {csv_path}")
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
        try:
            if chunksize:
                return pd.read_csv(csv_path, chunksize=chunksize)
            df = pd.read_csv(csv_path)
            self.logger.info(f"Successfully loaded CSV. Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            self.logger.error(f"Failed to read CSV at {csv_path}: {e}")
            raise

    def load_from_db(self, db_url: str = None, query: str = "SELECT * FROM customers") -> pd.DataFrame:
        """
        Loads customer records from the database and maps database column names
        back to original training pipeline expectations (CamelCase/PascalCase).
        
        Parameters:
            db_url (str): Database connection URL. If None, defaults to settings.DATABASE_URL.
            query (str): SQL query to run.
            
        Returns:
            pd.DataFrame: Cleaned and mapped dataset.
        """
        if db_url is None:
            try:
                from backend.app.core.config import settings
                db_url = settings.DATABASE_URL
            except ImportError:
                db_url = "sqlite:///customer_retention.db"

        # Resolve SQLite relative path to absolute path relative to project root
        if db_url.startswith("sqlite:///"):
            db_path = db_url.split("sqlite:///", 1)[1]
            if not os.path.isabs(db_path):
                if db_path.startswith("./") or db_path.startswith(".\\"):
                    db_path = db_path[2:]
                db_path = os.path.abspath(os.path.join(project_root, db_path))
                db_url = f"sqlite:///{db_path}"

        self.logger.info(f"Loading data from database: {db_url}")
        
        engine = create_engine(db_url)
        try:
            df = pd.read_sql_query(query, engine)
            self.logger.info(f"Successfully loaded database query. Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Map columns back to training expectations
            df_mapped = self._map_db_columns(df)
            return df_mapped
        except Exception as e:
            self.logger.error(f"Failed to load data from database: {e}")
            raise
        finally:
            engine.dispose()
            self.logger.info("Database connection engine successfully disposed.")

    def _map_db_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns from database snake_case back to original CamelCase schema,
        dropping internal primary/foreign keys (id, upload_id).
        """
        df_copy = df.copy()
        
        # Drop internal database keys if present
        cols_to_drop = [c for c in ["id", "upload_id"] if c in df_copy.columns]
        if cols_to_drop:
            df_copy = df_copy.drop(columns=cols_to_drop)
            self.logger.info(f"Dropped database metadata columns: {cols_to_drop}")

        # Retrieve database column mapping from config loader with a fallback map
        feature_cfg = config_loader.feature
        mapping = feature_cfg.get("database_column_mapping", {})
        
        if not mapping:
            self.logger.warning("No database column mapping found in configs. Falling back to default map.")
            mapping = {
                "customer_id": "customerID",
                "gender": "gender",
                "senior_citizen": "SeniorCitizen",
                "partner": "Partner",
                "dependents": "Dependents",
                "tenure": "tenure",
                "phone_service": "PhoneService",
                "multiple_lines": "MultipleLines",
                "internet_service": "InternetService",
                "online_security": "OnlineSecurity",
                "online_backup": "OnlineBackup",
                "device_protection": "DeviceProtection",
                "tech_support": "TechSupport",
                "streaming_tv": "StreamingTV",
                "streaming_movies": "StreamingMovies",
                "contract": "Contract",
                "paperless_billing": "PaperlessBilling",
                "payment_method": "PaymentMethod",
                "monthly_charges": "MonthlyCharges",
                "total_charges": "TotalCharges",
                "churn": "Churn"
            }
        
        # Only map columns that exist in the dataframe
        rename_map = {k: v for k, v in mapping.items() if k in df_copy.columns}
        df_copy = df_copy.rename(columns=rename_map)
        
        self.logger.info(f"Renamed {len(rename_map)} columns from database schema back to dataset names")
        return df_copy
