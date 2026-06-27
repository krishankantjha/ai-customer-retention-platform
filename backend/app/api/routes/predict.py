import io
import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database.models.uploads import Upload
from app.database.models.customer import Customer
from app.services.auth_service import get_current_user
from app.services.prediction_service import batch_predict_and_explain
from app.schemas.prediction import CustomerExplainResponse

logger = logging.getLogger("backend.app.api.routes.predict")

router = APIRouter(tags=["predict"])


def process_upload_task(upload_id: int, file_bytes: bytes, threshold: float = None):
    """Background task to process the uploaded CSV file and run predictions."""
    # Obtain a fresh database session inside the background thread
    from app.database.session import SessionLocal
    db = SessionLocal()
    try:
        # Get upload record
        upload = db.query(Upload).filter(Upload.id == upload_id).first()
        if not upload:
            logger.error(f"Upload record not found: {upload_id}")
            return
            
        upload.status = "processing"
        db.commit()
        
        # Load CSV into pandas DataFrame with encoding fallbacks to prevent decode failures
        df = None
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
                break
            except (UnicodeDecodeError, ValueError):
                continue
                
        if df is None:
            raise ValueError("CSV Decoding Error: Failed to parse character encoding (tried UTF-8, Latin-1, CP1252)")
        
        # Run predictions and save
        row_count = batch_predict_and_explain(df, db, upload_id, threshold=threshold)
        
        # Update upload record on success
        upload.status = "completed"
        upload.row_count = row_count
        db.commit()
        logger.info(f"Background upload task completed successfully for upload ID {upload_id}")
    except Exception as e:
        db.rollback()
        logger.exception(f"Background upload task failed for upload ID {upload_id}: {e}")
        try:
            # Re-fetch upload to record failure
            upload = db.query(Upload).filter(Upload.id == upload_id).first()
            if upload:
                upload.status = "failed"
                # Map technical db errors to user-friendly messages
                error_str = str(e)
                if "UNIQUE constraint" in error_str or "duplicate key" in error_str:
                    upload.error_message = "Integrity Constraint Error: Dataset contains duplicate customer IDs already present in the database."
                else:
                    upload.error_message = error_str[:500]
                db.commit()
        except Exception as inner_err:
            logger.error(f"Failed to record upload failure state: {inner_err}")
    finally:
        db.close()
 
 
@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = None,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Upload a customer churn dataset CSV.
    Processing runs asynchronously in the background.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only CSV files are supported."
        )
        
    try:
        file_bytes = await file.read()
        
        # Initialize upload record in database
        upload = Upload(
            filename=file.filename,
            status="pending"
        )
        db.add(upload)
        db.commit()
        db.refresh(upload)
        
        # Trigger background processing
        background_tasks.add_task(process_upload_task, upload.id, file_bytes, threshold)
        
        return {
            "upload_id": upload.id,
            "filename": upload.filename,
            "status": upload.status,
            "message": "File upload accepted. Processing in progress."
        }
    except Exception as e:
        db.rollback()
        logger.exception(f"File upload initialization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize file upload: {str(e)}"
        )


def get_cohort_persona_name(cluster_id: int) -> str:
    if cluster_id == 0:
        return "Cluster 0: Moderate-Value, Budget-Conscious Users"
    elif cluster_id == 1:
        return "Cluster 1: High-Value Premium Cohort"
    elif cluster_id == 2:
        return "Cluster 2: New Churn-Risk Users"
    return "N/A"


@router.get("/customers/{customer_id}/explain", response_model=CustomerExplainResponse)
def get_customer_explain(
    customer_id: str,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Retrieve churn predictions, SHAP drivers, and Save Plays for a specific customer.
    """
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer with ID {customer_id} not found."
        )
        
    prediction = customer.prediction
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No churn prediction found for customer {customer_id}."
        )
        
    persona_name = get_cohort_persona_name(prediction.cluster) if prediction.cluster is not None else None
    
    # Run counterfactual simulations dynamically using LocalExplainer
    from app.services.prediction_service import load_artifacts
    from ml.explainability.shap_local import LocalExplainer
    
    try:
        model_obj, preprocessor_obj, encoders_meta, metadata, explainer_obj, _ = load_artifacts()
        local_explainer = LocalExplainer(
            model_obj,
            metadata["feature_names_in"],
            explainer=explainer_obj,
            preprocessor=preprocessor_obj,
            encoders=encoders_meta,
            metadata=metadata,
        )
        
        customer_dict = {
            "customerID": customer.customer_id,
            "gender": customer.gender,
            "SeniorCitizen": customer.senior_citizen,
            "Partner": customer.partner,
            "Dependents": customer.dependents,
            "tenure": customer.tenure,
            "PhoneService": customer.phone_service,
            "MultipleLines": customer.multiple_lines,
            "InternetService": customer.internet_service,
            "OnlineSecurity": customer.online_security,
            "OnlineBackup": customer.online_backup,
            "DeviceProtection": customer.device_protection,
            "TechSupport": customer.tech_support,
            "StreamingTV": customer.streaming_tv,
            "StreamingMovies": customer.streaming_movies,
            "Contract": customer.contract,
            "PaperlessBilling": customer.paperless_billing,
            "PaymentMethod": customer.payment_method,
            "MonthlyCharges": customer.monthly_charges,
            "TotalCharges": customer.total_charges or 0.0,
            "Churn": customer.churn
        }
        customer_df = pd.DataFrame([customer_dict])
        simulations = local_explainer.run_simulations(customer_df)
    except Exception as e:
        logger.error(f"Failed to generate counterfactual simulations for customer {customer_id}: {e}", exc_info=True)
        simulations = []

    return CustomerExplainResponse(
        customer_id=customer.customer_id,
        gender=customer.gender,
        tenure=customer.tenure,
        monthly_charges=customer.monthly_charges,
        total_charges=customer.total_charges or 0.0,
        churn_probability=prediction.churn_probability,
        is_high_risk=prediction.is_high_risk,
        top_drivers=prediction.top_drivers,
        save_plays=prediction.save_plays,
        cluster=prediction.cluster,
        cohort_persona=persona_name,
        segmentation={
            "cluster_id": prediction.cluster,
            "persona": persona_name
        } if prediction.cluster is not None else None,
        simulations=simulations,
        customer_features=customer_dict if 'customer_dict' in locals() else None,
        predicted_at=prediction.predicted_at
    )


@router.post("/predict/simulate")
def simulate_prediction(
    request: dict,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Run a single live counterfactual simulation for edited customer inputs.
    """
    from app.services.prediction_service import load_artifacts
    from ml.explainability.shap_local import LocalExplainer

    try:
        model_obj, preprocessor_obj, encoders_meta, metadata, explainer_obj, _ = load_artifacts()
        local_explainer = LocalExplainer(
            model_obj,
            metadata["feature_names_in"],
            explainer=explainer_obj,
            preprocessor=preprocessor_obj,
            encoders=encoders_meta,
            metadata=metadata,
        )

        customer_dict = {
            "customerID": request.get("customerID") or request.get("customer_id") or "SIM",
            "gender": request.get("gender") or "Male",
            "SeniorCitizen": int(request.get("SeniorCitizen") or request.get("senior_citizen") or 0),
            "Partner": request.get("Partner") or request.get("partner") or "No",
            "Dependents": request.get("Dependents") or request.get("dependents") or "No",
            "tenure": int(request.get("tenure") or 0),
            "PhoneService": request.get("PhoneService") or request.get("phone_service") or "Yes",
            "MultipleLines": request.get("MultipleLines") or request.get("multiple_lines") or "No",
            "InternetService": request.get("InternetService") or request.get("internet_service") or "No",
            "OnlineSecurity": request.get("OnlineSecurity") or request.get("online_security") or "No",
            "OnlineBackup": request.get("OnlineBackup") or request.get("online_backup") or "No",
            "DeviceProtection": request.get("DeviceProtection") or request.get("device_protection") or "No",
            "TechSupport": request.get("TechSupport") or request.get("tech_support") or "No",
            "StreamingTV": request.get("StreamingTV") or request.get("streaming_tv") or "No",
            "StreamingMovies": request.get("StreamingMovies") or request.get("streaming_movies") or "No",
            "Contract": request.get("Contract") or request.get("contract") or "Month-to-month",
            "PaperlessBilling": request.get("PaperlessBilling") or request.get("paperless_billing") or "No",
            "PaymentMethod": request.get("PaymentMethod") or request.get("payment_method") or "Mailed check",
            "MonthlyCharges": float(request.get("MonthlyCharges") or request.get("monthly_charges") or 0.0),
            "TotalCharges": float(request.get("TotalCharges") or request.get("total_charges") or 0.0),
            "Churn": request.get("Churn") or request.get("churn") or "No"
        }

        customer_df = pd.DataFrame([customer_dict])
        sim_prob = local_explainer.simulate_intervention(customer_df, {})
        return {"simulated_probability": sim_prob}
    except Exception as e:
        logger.error(f"Failed live counterfactual simulation prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation error: {str(e)}"
        )


@router.get("/uploads/{upload_id}/status")
def get_upload_status(
    upload_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Query the database uploads table to check processing state.
    """
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload with ID {upload_id} not found."
        )
    return {
        "upload_id": upload.id,
        "filename": upload.filename,
        "status": upload.status,
        "row_count": upload.row_count or 0,
        "error_message": upload.error_message,
        "uploaded_at": upload.uploaded_at
    }


@router.get("/customers/search")
def search_customers(
    q: str = "",
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """
    Search customer IDs matching a prefix query for autocomplete.
    """
    if not q:
        return []
    results = db.query(Customer.customer_id).filter(
        Customer.customer_id.like(f"{q}%")
    ).limit(15).all()
    return [r[0] for r in results]


