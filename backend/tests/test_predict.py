import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import and override session module first before importing app
import app.database.session as session_module

TEST_DATABASE_URL = "sqlite:///:memory:"

# Create an in-memory SQLite engine with StaticPool to keep connection/tables alive
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Monkeypatch SessionLocal in the database module
session_module.SessionLocal = TestingSessionLocal

from app.main import app
from app.database.base import Base
from app.database.models.uploads import Upload
from app.database.models.customer import Customer
from app.database.models.prediction import Prediction

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Initializes tables for test runtime, clean up at the end."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function", autouse=True)
def clean_tables():
    """Clears records between test runs."""
    db = TestingSessionLocal()
    db.query(Prediction).delete()
    db.query(Customer).delete()
    db.query(Upload).delete()
    db.commit()
    db.close()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unauthorized_endpoints():
    # Endpoints must reject requests lacking authentication headers with 401
    resp1 = client.post("/api/v1/upload")
    assert resp1.status_code == 401

    resp2 = client.get("/api/v1/customers/1234-ABCD/explain")
    assert resp2.status_code == 401

    resp3 = client.get("/api/v1/analytics/overview")
    assert resp3.status_code == 401

    resp4 = client.get("/api/v1/analytics/save-plays")
    assert resp4.status_code == 401


def test_login_and_token_generation():
    # Attempting to log in with incorrect credentials
    bad_login = client.post("/api/v1/auth/login", data={"username": "admin", "password": "wrongpassword"})
    assert bad_login.status_code == 400
    assert "Incorrect username or password" in bad_login.json()["detail"]

    # Successful login
    good_login = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password"})
    assert good_login.status_code == 200
    data = good_login.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_full_pipeline_and_endpoints():
    # 1. Log in to get token
    login_resp = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload CSV dataset
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,"
        "OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,"
        "PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n"
        "1234-ABCD,Male,0,No,No,5,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.0,375.0,Yes\n"
        "5678-EFGH,Female,1,Yes,No,36,Yes,Yes,Fiber optic,Yes,No,Yes,No,Yes,Yes,One year,No,Credit card (automatic),105.0,3780.0,No\n"
    )
    
    files = {"file": ("test_cohort.csv", csv_content, "text/csv")}
    upload_resp = client.post("/api/v1/upload", headers=headers, files=files)
    assert upload_resp.status_code == 202
    assert upload_resp.json()["status"] == "pending"
    upload_id = upload_resp.json()["upload_id"]

    # In FastAPI TestClient, background tasks run synchronously during/immediately after the request loop
    # So we can verify that the upload has been processed and completed in the database.
    db = TestingSessionLocal()
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    assert upload is not None
    assert upload.status == "completed"
    assert upload.row_count == 2

    # 3. Retrieve churn explanation for a specific customer
    explain_resp = client.get("/api/v1/customers/1234-ABCD/explain", headers=headers)
    assert explain_resp.status_code == 200
    explain_data = explain_resp.json()
    assert explain_data["customer_id"] == "1234-ABCD"
    assert explain_data["gender"] == "Male"
    assert explain_data["churn_probability"] >= 0.0
    assert explain_data["churn_probability"] <= 1.0
    assert isinstance(explain_data["is_high_risk"], bool)
    assert "top_drivers" in explain_data
    assert "save_plays" in explain_data
    assert len(explain_data["top_drivers"]) > 0
    assert "cluster" in explain_data
    assert "cohort_persona" in explain_data
    assert explain_data["cluster"] is not None
    assert isinstance(explain_data["cluster"], int)
    assert "Cluster" in explain_data["cohort_persona"]

    # 4. Fetch dashboard overview analytics
    overview_resp = client.get("/api/v1/analytics/overview", headers=headers)
    assert overview_resp.status_code == 200
    overview_data = overview_resp.json()
    assert overview_data["total_customers"] == 2
    assert overview_data["average_churn_probability"] >= 0.0
    assert overview_data["total_value_at_risk"] >= 0.0
    assert "risk_distribution" in overview_data

    # 5. Fetch Save Plays summary
    plays_resp = client.get("/api/v1/analytics/save-plays", headers=headers)
    assert plays_resp.status_code == 200
    plays_data = plays_resp.json()
    assert isinstance(plays_data, list)
    if len(plays_data) > 0:
        assert "campaign" in plays_data[0]
        assert "recommendation_count" in plays_data[0]
        assert "average_estimated_impact" in plays_data[0]

    # 6. Fetch cohort data and verify cluster values
    cohort_resp = client.get("/api/v1/analytics/cohort-data", headers=headers)
    assert cohort_resp.status_code == 200
    cohort_data = cohort_resp.json()
    assert isinstance(cohort_data, list)
    assert len(cohort_data) == 2
    assert "cluster" in cohort_data[0]
    assert cohort_data[0]["cluster"] is not None
    assert isinstance(cohort_data[0]["cluster"], int)


def test_upload_duplicate_customer_ids():
    # 1. Log in
    login_resp = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. CSV containing duplicate customer ID (1234-ABCD appears twice)
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,"
        "OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,"
        "PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n"
        "1234-ABCD,Male,0,No,No,5,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.0,375.0,Yes\n"
        "1234-ABCD,Female,1,Yes,No,36,Yes,Yes,Fiber optic,Yes,No,Yes,No,Yes,Yes,One year,No,Credit card (automatic),105.0,3780.0,No\n"
    )
    
    files = {"file": ("test_duplicate.csv", csv_content, "text/csv")}
    upload_resp = client.post("/api/v1/upload", headers=headers, files=files)
    assert upload_resp.status_code == 202
    upload_id = upload_resp.json()["upload_id"]

    db = TestingSessionLocal()
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    assert upload is not None
    assert upload.status == "failed"
    assert "duplicate customer IDs" in upload.error_message


def test_upload_non_utf8_encoding():
    # 1. Log in
    login_resp = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password"})
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Content with non-ASCII char (e.g. French character é encoded as Latin-1)
    csv_content_bytes = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,"
        "OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,"
        "PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n"
        "9999-\u00e9XYZ,Male,0,No,No,5,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.0,375.0,Yes\n"
    ).encode("latin-1")
    
    files = {"file": ("test_latin.csv", csv_content_bytes, "text/csv")}
    upload_resp = client.post("/api/v1/upload", headers=headers, files=files)
    assert upload_resp.status_code == 202
    upload_id = upload_resp.json()["upload_id"]

    db = TestingSessionLocal()
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    assert upload is not None
    assert upload.status == "completed"
    assert upload.row_count == 1

