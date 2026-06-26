from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.session import get_db
from app.database.models.customer import Customer
from app.database.models.prediction import Prediction
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
def get_analytics_overview(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Retrieve overview statistics for predicted customers.
    """
    total_customers = db.query(func.count(Prediction.id)).scalar() or 0
    if total_customers == 0:
        return {
            "total_customers": 0,
            "average_churn_probability": 0.0,
            "total_value_at_risk": 0.0,
            "risk_distribution": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }
        
    avg_probability = db.query(func.avg(Prediction.churn_probability)).scalar() or 0.0
    
    # Total monthly charges at risk (is_high_risk predictions)
    total_value_at_risk = db.query(func.sum(Customer.monthly_charges))\
        .join(Prediction, Customer.id == Prediction.customer_id)\
        .filter(Prediction.is_high_risk == True)\
        .scalar() or 0.0
        
    # High Risk: probability >= 0.50
    # Medium Risk: 0.25 <= probability < 0.50
    # Low Risk: probability < 0.25
    high_count = db.query(func.count(Prediction.id)).filter(Prediction.churn_probability >= 0.5).scalar() or 0
    medium_count = db.query(func.count(Prediction.id))\
        .filter(Prediction.churn_probability >= 0.25)\
        .filter(Prediction.churn_probability < 0.5).scalar() or 0
    low_count = db.query(func.count(Prediction.id)).filter(Prediction.churn_probability < 0.25).scalar() or 0
    
    return {
        "total_customers": total_customers,
        "average_churn_probability": float(avg_probability),
        "total_value_at_risk": float(total_value_at_risk),
        "risk_distribution": {
            "high": high_count,
            "medium": medium_count,
            "low": low_count
        }
    }


@router.get("/save-plays")
def get_save_plays_analytics(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Retrieve aggregates ofrecommended Save Play campaigns.
    """
    predictions = db.query(Prediction.save_plays).all()
    
    campaign_counts = defaultdict(int)
    campaign_impacts = defaultdict(list)
    
    for row in predictions:
        plays = row[0]
        if not plays:
            continue
        for play in plays:
            campaign = play.get("campaign")
            impact = play.get("estimated_impact", 0.0)
            if campaign:
                campaign_counts[campaign] += 1
                campaign_impacts[campaign].append(impact)
                
    results = []
    for campaign, count in campaign_counts.items():
        impacts = campaign_impacts[campaign]
        avg_impact = sum(impacts) / len(impacts) if impacts else 0.0
        results.append({
            "campaign": campaign,
            "recommendation_count": count,
            "average_estimated_impact": float(avg_impact)
        })
        
    results.sort(key=lambda x: x["recommendation_count"], reverse=True)
    return results


@router.get("/cohort-data")
def get_cohort_data(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Retrieve all customer demographic, contract, and churn prediction details for cohort analysis.
    """
    results = db.query(
        Customer.customer_id,
        Customer.gender,
        Customer.tenure,
        Customer.contract,
        Customer.internet_service,
        Customer.monthly_charges,
        Customer.total_charges,
        Customer.churn,
        Prediction.churn_probability,
        Prediction.is_high_risk,
        Prediction.cluster,
        Prediction.predicted_at
    ).join(Prediction, Customer.id == Prediction.customer_id).all()

    return [
        {
            "customer_id": r.customer_id,
            "gender": r.gender,
            "tenure": r.tenure,
            "contract": r.contract,
            "internet_service": r.internet_service,
            "monthly_charges": r.monthly_charges,
            "total_charges": r.total_charges,
            "churn": r.churn,
            "churn_probability": r.churn_probability,
            "is_high_risk": r.is_high_risk,
            "cluster": r.cluster,
            "predicted_at": r.predicted_at.isoformat() if r.predicted_at else None
        }
        for r in results
    ]


@router.get("/diagnostics-metadata")
def get_diagnostics_metadata(
    current_user: str = Depends(get_current_user)
):
    """
    Retrieve diagnostics metadata to identify version drift in user interfaces.
    """
    import os
    import json
    import hashlib
    
    # Resolve absolute path to diagnostics_metadata.json relative to backend project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    metadata_path = os.path.join(PROJECT_ROOT, "ml", "artifacts", "diagnostics_metadata.json")
    model_path = os.path.join(PROJECT_ROOT, "ml", "artifacts", "model.pkl")
    
    # Compute active model's SHA-256
    model_sha256 = ""
    if os.path.exists(model_path):
        sha256 = hashlib.sha256()
        try:
            with open(model_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            model_sha256 = sha256.hexdigest()
        except Exception:
            pass
            
    if not os.path.exists(metadata_path):
        return {
            "success": False,
            "drift_detected": True,
            "message": "Diagnostics metadata file not found.",
            "model_version": "unknown",
            "diagnostics_version": "unknown",
            "artifact_timestamp": "unknown",
            "evaluation_timestamp": "unknown",
            "model_sha256": "unknown",
            "actual_model_sha256": model_sha256
        }
        
    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)
            
        expected_sha = data.get("model_sha256", "")
        drift_detected = (expected_sha != model_sha256)
        
        return {
            "success": True,
            "drift_detected": drift_detected,
            "actual_model_sha256": model_sha256,
            **data
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read diagnostics metadata: {e}"
        )


@router.get("/model-health")
def get_model_health(
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Retrieve model health metadata, expected performance metrics,
    and Kolmogorov-Smirnov test results indicating feature drift.
    """
    try:
        from app.services.prediction_service import get_preprocessed_active_customers
        from ml.training.model_monitor import get_system_health
        
        X_active = get_preprocessed_active_customers(db)
        health_status = get_system_health(X_active)
        return health_status
    except Exception as e:
        import logging
        logging.getLogger("backend.app.api.routes.analytics").error(f"Failed to check model health: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check model health: {str(e)}"
        )


