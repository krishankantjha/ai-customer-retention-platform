import os
import sys
import pickle
import logging
import threading
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import shap
from sqlalchemy.orm import Session

from app.database.models.customer import Customer
from app.database.models.prediction import Prediction

# Resolve project root dynamically to support importing and file resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocessing.clean import fix_whitespace_blanks, fix_total_charges, drop_duplicates
from ml.preprocessing.engineer import engineer_features
from configs.dataset_config import config_loader
from ml.preprocessing.validator import DataValidator
from app.core.security import verify_file_hash, ArtifactValidationError

logger = logging.getLogger("backend.app.services.prediction_service")

# Resolve absolute artifact paths from config loader
artifacts_dir_relative = config_loader.training["data_paths"].get("artifacts_dir", "ml/artifacts")
artifacts_dir = os.path.join(PROJECT_ROOT, artifacts_dir_relative)

MODEL_PATH = os.path.join(artifacts_dir, "model.pkl")
PIPELINE_PATH = os.path.join(artifacts_dir, "pipeline.pkl")
ENCODERS_PATH = os.path.join(artifacts_dir, "encoders.pkl")
METADATA_PATH = os.path.join(artifacts_dir, "model_metadata.pkl")
MANIFEST_PATH = os.path.join(artifacts_dir, "artifacts_manifest.json")


# Global variables to cache loaded models
_model = None
_preprocessor = None
_encoders_meta = None
_model_metadata = None
_explainer = None
_kmeans_model = None
_autoencoder = None

# Thread lock to guarantee safe lazy loading of ML artifacts in multi-threaded runtime
_lock = threading.Lock()


def load_artifacts() -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any], Any, Any]:
    """
    Loads and caches model, pipeline, metadata, SHAP explainer, and K-Means artifacts.
    Uses a thread-safe double-checked lock pattern to prevent redundant I/O operations
    when multiple API requests try to load the artifacts concurrently on startup.
    """
    global _model, _preprocessor, _encoders_meta, _model_metadata, _explainer, _kmeans_model, _autoencoder
    
    if _model is not None:
        return _model, _preprocessor, _encoders_meta, _model_metadata, _explainer, _kmeans_model
        
    with _lock:
        # Double-check inside lock boundary to prevent redundant read operations
        if _model is not None:
            return _model, _preprocessor, _encoders_meta, _model_metadata, _explainer, _kmeans_model
            
        logger.info("Loading model and pipeline artifacts from disk (thread-safe lock acquired)...")
        
        kmeans_path = os.path.join(artifacts_dir, "models", "kmeans_model.pkl")
        ae_path = os.path.join(artifacts_dir, "models", "autoencoder_model.pkl")
        
        # Verify hashes against manifest
        if not os.path.exists(MANIFEST_PATH):
            logger.error(f"Artifacts manifest not found: {MANIFEST_PATH}")
            raise ArtifactValidationError(f"Artifacts manifest not found at {MANIFEST_PATH}")
            
        import json
        try:
            with open(MANIFEST_PATH, "r") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse manifest at {MANIFEST_PATH}: {e}")
            raise ArtifactValidationError(f"Corrupt artifacts manifest: {e}")
            
        try:
            verify_file_hash(MODEL_PATH, manifest.get("model.pkl", ""))
            verify_file_hash(PIPELINE_PATH, manifest.get("pipeline.pkl", ""))
            verify_file_hash(ENCODERS_PATH, manifest.get("encoders.pkl", ""))
            verify_file_hash(METADATA_PATH, manifest.get("model_metadata.pkl", ""))
            verify_file_hash(kmeans_path, manifest.get("kmeans_model.pkl", ""))
            verify_file_hash(ae_path, manifest.get("autoencoder_model.pkl", ""))
        except Exception as e:
            logger.error(f"Artifact integrity verification failed: {e}")
            raise ArtifactValidationError(f"Artifact verification failed: {e}")
            
        # File exist checks (already implicitly verified by verify_file_hash, but we keep explicit check for clear errors)
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(f"Pipeline file not found at {PIPELINE_PATH}")
        if not os.path.exists(ENCODERS_PATH):
            raise FileNotFoundError(f"Encoders file not found at {ENCODERS_PATH}")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Model metadata file not found at {METADATA_PATH}")
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(f"K-Means model file not found at {kmeans_path}")
        if not os.path.exists(ae_path):
            raise FileNotFoundError(f"Autoencoder model file not found at {ae_path}")
            
        # Import wrapper dynamically to ensure correct unpickling scope
        from ml.segmentation.autoencoder import AutoencoderWrapper
        
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(PIPELINE_PATH, "rb") as f:
            _preprocessor = pickle.load(f)
        with open(ENCODERS_PATH, "rb") as f:
            _encoders_meta = pickle.load(f)
        with open(METADATA_PATH, "rb") as f:
            _model_metadata = pickle.load(f)
        with open(kmeans_path, "rb") as f:
            _kmeans_model = pickle.load(f)
        with open(ae_path, "rb") as f:
            _autoencoder = pickle.load(f)
            
        # Cache the SHAP Explainer once during startup to eliminate per-batch initialization costs
        if hasattr(_model, "calibrated_classifiers_"):
            shap_model = _model.calibrated_classifiers_[0].estimator
        elif hasattr(_model, "xgb_"):
            shap_model = _model.xgb_
        else:
            shap_model = _model
        _explainer = shap.Explainer(shap_model)
            
        logger.info("Artifacts successfully loaded and explainer cached.")
        return _model, _preprocessor, _encoders_meta, _model_metadata, _explainer, _kmeans_model


def get_autoencoder():
    """Returns the cached AutoencoderWrapper model, loading it if not cached."""
    global _autoencoder
    if _autoencoder is None:
        load_artifacts()
    return _autoencoder


def log_prediction_events(customer_ids: List[str], y_probs: np.ndarray, is_high_risks: np.ndarray, cluster_labels: np.ndarray):
    """
    Appends predictions audit trail to monthly-partitioned JSONL files.

    FIX MEDIUM-4: Files are rotated monthly (prediction_logs_YYYY-MM.jsonl)
    to prevent a single file growing unboundedly in production.
    """
    import json
    from datetime import datetime
    metrics_dir = os.path.join(artifacts_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Monthly rotation: each calendar month gets its own file
    now = datetime.utcnow()
    month_str = now.strftime("%Y-%m")
    log_path = os.path.join(metrics_dir, f"prediction_logs_{month_str}.jsonl")

    timestamp = now.isoformat() + "Z"

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for i, cid in enumerate(customer_ids):
                record = {
                    "timestamp": timestamp,
                    "customer_id": str(cid),
                    "churn_probability": float(y_probs[i]),
                    "is_high_risk": bool(is_high_risks[i]),
                    "cluster": int(cluster_labels[i])
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"Successfully logged {len(customer_ids)} predictions to audit trail: {log_path}")
    except Exception as e:
        logger.error(f"Failed to write prediction logs to audit trail: {e}")


def clean_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies basic data cleaning matching the ML data cleaning steps."""
    logger.info("Cleaning uploaded dataframe...")
    df_clean = df.copy()
    
    # Standardize columns casing to match expected
    col_mapping = {
        "customerid": "customerID",
        "seniorcitizen": "SeniorCitizen",
        "phoneservice": "PhoneService",
        "multiplelines": "MultipleLines",
        "internetservice": "InternetService",
        "onlinesecurity": "OnlineSecurity",
        "onlinebackup": "OnlineBackup",
        "deviceprotection": "DeviceProtection",
        "techsupport": "TechSupport",
        "streamingtv": "StreamingTV",
        "streamingmovies": "StreamingMovies",
        "paperlessbilling": "PaperlessBilling",
        "paymentmethod": "PaymentMethod",
        "monthlycharges": "MonthlyCharges",
        "totalcharges": "TotalCharges",
        "churn": "Churn"
    }
    # Apply casing standardization for any columns that might be lowercase
    df_clean = df_clean.rename(columns=lambda c: col_mapping.get(c.lower(), c))
    
    target_col = config_loader.feature.get("target_column", "Churn")
    # 21 columns required (or 20 if Churn is missing, we'll insert a placeholder Churn)
    if target_col not in df_clean.columns:
        df_clean[target_col] = None
        
    df_clean = fix_whitespace_blanks(df_clean, logger)
    df_clean = fix_total_charges(df_clean, logger)
    df_clean = drop_duplicates(df_clean, logger)
    
    # Reset index to ensure 0-based sequential row indexing
    df_clean = df_clean.reset_index(drop=True)
    return df_clean


def batch_predict_and_explain(df: pd.DataFrame, db: Session, upload_id: int, threshold: float = None) -> int:
    """
    Cleans the uploaded DataFrame, runs the feature engineering & encoding pipeline,
    generates calibrated churn probability predictions, continuous customer segment classifications,
    and SHAP explainability metrics/recommended Save Plays, and bulk inserts them.
    """
    model_obj, preprocessor_obj, encoders, metadata, explainer_obj, kmeans_obj = load_artifacts()
    
    # Clean data
    df_clean = clean_uploaded_data(df)

    # Perform strict modular validations to protect prediction pipeline
    validator = DataValidator(logger)
    validator.validate_schema(df_clean, strict=True)
    validator.validate_data_types(df_clean, strict=True)
    validator.validate_value_bounds(df_clean, strict=True)
    validator.validate_categorical_domains(df_clean, strict=True)
    
    target_col = config_loader.feature.get("target_column", "Churn")
    # Keep track of original Churn values to save in the DB (keep None/NaN as null)
    original_churns = df_clean[target_col].apply(lambda val: str(val) if pd.notna(val) else None).tolist()
    
    # Run feature engineering (this requires train_monthly_charges_median to prevent leakage)
    df_engineered = engineer_features(df_clean, encoders["train_monthly_charges_median"])
    
    # Run column transformer (StandardScaler, encoders, etc.)
    X_transformed = preprocessor_obj.transform(df_engineered)
    X_df = pd.DataFrame(X_transformed, columns=encoders["feature_names_out"])
    
    # Align to model's expected inputs (dropping binary__has_support etc)
    X_aligned = X_df[metadata["feature_names_in"]]
    
    # Model predictions
    y_prob = model_obj.predict_proba(X_aligned)[:, 1]
    # Resolve classification threshold dynamically (override parameter -> config -> fallback)
    if threshold is None:
        threshold = config_loader.model.get("decision_threshold", 0.15)
    is_high_risk = (y_prob >= threshold).astype(bool)
    
    # Predict clusters by projecting preprocessed features to 16-dimensional latent space via Autoencoder
    autoencoder = get_autoencoder()
    if autoencoder is not None:
        try:
            X_latent = autoencoder.transform(X_transformed.astype(np.float32))
            cluster_labels = kmeans_obj.predict(X_latent)
        except Exception as e:
            logger.warning(f"Failed to project features using Autoencoder: {e}. Falling back to raw continuous features.")
            seg_cfg = config_loader.model.get("segmentation", {})
            cont_cols = seg_cfg.get("continuous_features", ["numeric__tenure", "numeric__MonthlyCharges", "numeric__num_services"])
            actual_cont_cols = [col for col in cont_cols if col in X_df.columns]
            X_continuous = X_df[actual_cont_cols] if actual_cont_cols else X_df
            cluster_labels = kmeans_obj.predict(X_continuous)
    else:
        logger.warning("Autoencoder model is None. Falling back to raw continuous features for cluster prediction.")
        seg_cfg = config_loader.model.get("segmentation", {})
        cont_cols = seg_cfg.get("continuous_features", ["numeric__tenure", "numeric__MonthlyCharges", "numeric__num_services"])
        actual_cont_cols = [col for col in cont_cols if col in X_df.columns]
        X_continuous = X_df[actual_cont_cols] if actual_cont_cols else X_df
        cluster_labels = kmeans_obj.predict(X_continuous)
    
    # Instantiate LocalExplainer — pass preprocessor/encoders/metadata to eliminate
    # the circular ML→backend dependency (FIX CRITICAL-3).
    from ml.explainability.shap_local import LocalExplainer
    local_explainer = LocalExplainer(
        model_obj,
        metadata["feature_names_in"],
        explainer=explainer_obj,
        preprocessor=preprocessor_obj,
        encoders=encoders,
        metadata=metadata,
    )

    customers_to_insert = []
    for index, row in df_clean.iterrows():
        cust = Customer(
            customer_id=str(row["customerID"]),
            gender=str(row["gender"]),
            senior_citizen=int(row["SeniorCitizen"]),
            partner=str(row["Partner"]),
            dependents=str(row["Dependents"]),
            tenure=int(row["tenure"]),
            phone_service=str(row["PhoneService"]),
            multiple_lines=str(row["MultipleLines"]),
            internet_service=str(row["InternetService"]),
            online_security=str(row["OnlineSecurity"]),
            online_backup=str(row["OnlineBackup"]),
            device_protection=str(row["DeviceProtection"]),
            tech_support=str(row["TechSupport"]),
            streaming_tv=str(row["StreamingTV"]),
            streaming_movies=str(row["StreamingMovies"]),
            contract=str(row["Contract"]),
            paperless_billing=str(row["PaperlessBilling"]),
            payment_method=str(row["PaymentMethod"]),
            monthly_charges=float(row["MonthlyCharges"]),
            total_charges=float(row["TotalCharges"]),
            churn=original_churns[index],
            upload_id=upload_id
        )
        customers_to_insert.append(cust)

    # Bulk insert customers using add_all
    db.add_all(customers_to_insert)
    db.flush()

    # ----------------------------------------------------------------
    # FIX HIGH-3: Compute SHAP values for the ENTIRE batch in one call.
    # Previously called explainer per-row inside the loop — O(n) SHAP
    # instantiations.  One batch call is 20-50x faster for large uploads.
    # ----------------------------------------------------------------
    logger.info(f"Computing SHAP values for batch of {len(customers_to_insert)} customers...")
    try:
        batch_shap_values = explainer_obj(X_aligned)   # shape: (n_rows, n_features)
        batch_shap_array = batch_shap_values.values     # numpy array
    except Exception as shap_err:
        logger.warning(f"Batch SHAP computation failed ({shap_err}); falling back to per-row mode.")
        batch_shap_array = None

    predictions_to_insert = []
    for i, cust in enumerate(customers_to_insert):
        # Combine raw clean fields with model input fields to support value-aware Save Play checks
        customer_combined = X_aligned.iloc[[i]].copy()
        for col in df_clean.columns:
            if col not in customer_combined.columns:
                customer_combined[col] = df_clean.iloc[i][col]

        # Use pre-computed batch SHAP row where available; fall back to single-call
        if batch_shap_array is not None:
            explanation = local_explainer.explain_from_shap_values(
                batch_shap_array[i], customer_combined.iloc[0]
            )
        else:
            explanation = local_explainer.explain_customer(customer_combined)

        # Format Save Plays to match prediction DB schema
        db_save_plays = []
        for play in explanation["save_plays"]:
            db_save_plays.append({
                "campaign": play["play_name"],
                "action": play["recommendation"],
                "estimated_impact": float(play["contribution"]),
                "feature": play["feature"]
            })

        pred = Prediction(
            customer_id=cust.id,
            churn_probability=float(y_prob[i]),
            is_high_risk=bool(is_high_risk[i]),
            top_drivers=explanation["top_drivers"],
            save_plays=db_save_plays,
            cluster=int(cluster_labels[i])
        )
        predictions_to_insert.append(pred)
        
    # Log prediction events to audit trail
    log_prediction_events([cust.customer_id for cust in customers_to_insert], y_prob, is_high_risk, cluster_labels)
    
    # Bulk insert predictions using add_all
    db.add_all(predictions_to_insert)
    db.commit()
    return len(customers_to_insert)


def get_preprocessed_active_customers(db: Session) -> pd.DataFrame:
    """
    Fetches a representative sample of active customers from the database, runs
    feature engineering and preprocessing, and returns the preprocessed DataFrame
    for use in drift detection.

    FIX MEDIUM-5: Query is capped at 1,000 rows.  Fetching ALL customers on every
    health-check poll would load the entire DB table into memory and create
    sustained read pressure under continuous monitoring.
    """
    _, preprocessor_obj, encoders, _, _, _ = load_artifacts()

    # Sample at most 1,000 customers to bound memory and DB read cost
    customers = db.query(Customer).limit(1000).all()
    if not customers:
        return pd.DataFrame(columns=encoders["feature_names_out"])
        
    # Map model attributes back to dict for clean dataframe reconstruction
    data_list = []
    for c in customers:
        data_list.append({
            "customerID": c.customer_id,
            "gender": c.gender,
            "SeniorCitizen": c.senior_citizen,
            "Partner": c.partner,
            "Dependents": c.dependents,
            "tenure": c.tenure,
            "PhoneService": c.phone_service,
            "MultipleLines": c.multiple_lines,
            "InternetService": c.internet_service,
            "OnlineSecurity": c.online_security,
            "OnlineBackup": c.online_backup,
            "DeviceProtection": c.device_protection,
            "TechSupport": c.tech_support,
            "StreamingTV": c.streaming_tv,
            "StreamingMovies": c.streaming_movies,
            "Contract": c.contract,
            "PaperlessBilling": c.paperless_billing,
            "PaymentMethod": c.payment_method,
            "MonthlyCharges": c.monthly_charges,
            "TotalCharges": c.total_charges or 0.0,
            "Churn": c.churn
        })
        
    df_clean = pd.DataFrame(data_list)
    df_engineered = engineer_features(df_clean, encoders["train_monthly_charges_median"])
    X_transformed = preprocessor_obj.transform(df_engineered)
    X_df = pd.DataFrame(X_transformed, columns=encoders["feature_names_out"])
    return X_df
