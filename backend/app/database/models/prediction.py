from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, Boolean, DateTime, ForeignKey, JSON, CheckConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.database.base import Base


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("churn_probability >= 0.0 AND churn_probability <= 1.0", name="ck_prediction_probability"),
        Index("ix_predictions_sorting", "is_high_risk", "churn_probability")
    )

    id = Column(Integer, primary_key=True, index=True)
    
    # One-to-one foreign key referencing Customer
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Model scoring fields
    churn_probability = Column(Float, nullable=False, index=True)
    is_high_risk = Column(Boolean, nullable=False, default=False, index=True)
    
    # Local Explainability (SHAP values & Prescriptive Actions) stored as JSON lists (Postgres JSONB variant)
    top_drivers = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    save_plays = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    cluster = Column(Integer, nullable=True, index=True)
    
    predicted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    customer = relationship("Customer", back_populates="prediction")

