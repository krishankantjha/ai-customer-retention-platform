from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.database.base import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, unique=True, index=True, nullable=False)
    
    # Demographics
    gender = Column(String, nullable=False)
    senior_citizen = Column(Integer, nullable=False)  # 0 or 1
    partner = Column(String, nullable=False)          # Yes or No
    dependents = Column(String, nullable=False)       # Yes or No
    
    # Services
    tenure = Column(Integer, nullable=False)
    phone_service = Column(String, nullable=False)    # Yes or No
    multiple_lines = Column(String, nullable=False)   # Yes, No, No phone service
    internet_service = Column(String, nullable=False) # DSL, Fiber optic, No
    online_security = Column(String, nullable=False)  # Yes, No, No internet service
    online_backup = Column(String, nullable=False)    # Yes, No, No internet service
    device_protection = Column(String, nullable=False) # Yes, No, No internet service
    tech_support = Column(String, nullable=False)     # Yes, No, No internet service
    streaming_tv = Column(String, nullable=False)     # Yes, No, No internet service
    streaming_movies = Column(String, nullable=False) # Yes, No, No internet service
    
    # Contract / Billing
    contract = Column(String, nullable=False)         # Month-to-month, One year, Two year
    paperless_billing = Column(String, nullable=False) # Yes or No
    payment_method = Column(String, nullable=False)    # Electronic check, Mailed check, Bank transfer, Credit card
    monthly_charges = Column(Float, nullable=False)
    total_charges = Column(Float, nullable=False)
    
    # True historical label (nullable if uploading fresh/unlabeled records)
    churn = Column(String, nullable=True)             # Yes or No
    
    # Foreign key referencing parent upload batch
    upload_id = Column(Integer, ForeignKey("uploads.id", ondelete="CASCADE"), nullable=False)

    # Relationships
    upload = relationship("Upload", back_populates="customers")
    prediction = relationship("Prediction", back_populates="customer", uselist=False, cascade="all, delete-orphan")
