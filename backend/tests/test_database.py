import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from app.database.base import Base
from app.database.models.uploads import Upload
from app.database.models.customer import Customer
from app.database.models.prediction import Prediction

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="module")
def test_engine():
    """Isolated database engine for testing, created in-memory."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(test_engine):
    """Provides a transactional database session for tests, clean up after execution."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()



def test_database_flow_and_cascade(db_session):
    # 1. Create a dummy upload log
    upload = Upload(
        filename="telco_batch_june.csv",
        status="completed",
        row_count=1
    )
    db_session.add(upload)
    db_session.commit()
    db_session.refresh(upload)

    assert upload.id is not None
    assert upload.filename == "telco_batch_june.csv"

    # 2. Create a dummy customer linked to the upload
    customer = Customer(
        customer_id="TEST-CUSTOMER-123",
        gender="Male",
        senior_citizen=0,
        partner=False,
        dependents=True,
        tenure=24,
        phone_service=True,
        multiple_lines="No",
        internet_service="Fiber optic",
        online_security="Yes",
        online_backup="No",
        device_protection="Yes",
        tech_support="Yes",
        streaming_tv="No",
        streaming_movies="Yes",
        contract="One year",
        paperless_billing=True,
        payment_method="Electronic check",
        monthly_charges=95.50,
        total_charges=2292.00,
        churn=False,
        upload_id=upload.id
    )
    db_session.add(customer)
    db_session.commit()
    db_session.refresh(customer)

    assert customer.id is not None
    assert customer.upload_id == upload.id

    # 3. Create a prediction with complex JSON structures for the customer
    top_drivers_mock = [
        {"feature": "numeric__MonthlyCharges", "shap_value": 0.32},
        {"feature": "categorical__InternetService_Fiber optic", "shap_value": 0.15}
    ]
    save_plays_mock = [
        {
            "feature": "numeric__MonthlyCharges",
            "play_name": "Billing Rate Audit",
            "recommendation": "Offer $10/month discount."
        }
    ]
    prediction = Prediction(
        customer_id=customer.id,
        churn_probability=0.38,
        is_high_risk=True,
        top_drivers=top_drivers_mock,
        save_plays=save_plays_mock
    )
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)

    assert prediction.id is not None
    assert prediction.is_high_risk is True
    assert prediction.top_drivers == top_drivers_mock
    assert prediction.save_plays == save_plays_mock

    # 4. Verify relations
    assert customer.prediction == prediction
    assert prediction.customer == customer
    assert customer.upload == upload
    assert upload.customers[0] == customer

    # 5. Verify cascading delete (deleting upload must delete customer and prediction)
    db_session.delete(upload)
    db_session.commit()

    # Query again and assert they are gone
    deleted_customer = db_session.query(Customer).filter_by(id=customer.id).first()
    deleted_prediction = db_session.query(Prediction).filter_by(id=prediction.id).first()
    deleted_upload = db_session.query(Upload).filter_by(id=upload.id).first()

    assert deleted_customer is None
    assert deleted_prediction is None
    assert deleted_upload is None


def test_row_count_constraint(db_session):
    # Verify negative row count check constraint
    upload = Upload(
        filename="corrupted_upload.csv",
        status="completed",
        row_count=-5
    )
    db_session.add(upload)
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_upload_status_constraint(db_session):
    # Verify status value check constraint
    upload = Upload(
        filename="bad_status.csv",
        status="invalid_status",
        row_count=10
    )
    db_session.add(upload)
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_prediction_probability_constraints(db_session):
    upload = Upload(filename="probability_check.csv", status="completed", row_count=1)
    db_session.add(upload)
    db_session.commit()

    customer = Customer(
        customer_id="TEST-CUSTOMER-XYZ",
        gender="Female",
        senior_citizen=0,
        partner=True,
        dependents=False,
        tenure=12,
        phone_service=True,
        multiple_lines="Yes",
        internet_service="DSL",
        online_security="No",
        online_backup="Yes",
        device_protection="No",
        tech_support="No",
        streaming_tv="Yes",
        streaming_movies="No",
        contract="Month-to-month",
        paperless_billing=False,
        payment_method="Mailed check",
        monthly_charges=50.00,
        total_charges=600.00,
        churn=None,
        upload_id=upload.id
    )
    db_session.add(customer)
    db_session.commit()

    # Verify probability check constraint (must be between 0.0 and 1.0)
    invalid_pred_over = Prediction(
        customer_id=customer.id,
        churn_probability=1.5,  # Violates check constraint
        is_high_risk=True,
        top_drivers=[],
        save_plays=[]
    )
    db_session.add(invalid_pred_over)
    with pytest.raises(IntegrityError):
        db_session.commit()

