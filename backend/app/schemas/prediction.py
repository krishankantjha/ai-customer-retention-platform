from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ShapDriver(BaseModel):
    feature: str = Field(..., description="The name of the feature driving the churn probability")
    shap_value: float = Field(..., description="The SHAP value indicating feature contribution")


class SavePlay(BaseModel):
    campaign: str = Field(..., description="Name of the retention campaign play")
    action: str = Field(..., description="Action items for the account manager or automated system")
    estimated_impact: float = Field(..., description="Expected impact score of the campaign play")


class PredictionBase(BaseModel):
    churn_probability: float = Field(..., description="Model calculated probability of customer churn")
    is_high_risk: bool = Field(..., description="Flag indicating if customer is high risk (based on threshold)")
    top_drivers: List[ShapDriver] = Field(..., description="List of top features contributing to prediction")
    save_plays: List[SavePlay] = Field(..., description="List of recommended retention plays")

    @field_validator("churn_probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("churn_probability must be between 0.0 and 1.0 inclusive")
        return v


class PredictionCreate(PredictionBase):
    customer_id: int = Field(..., description="Foreign key to the customer record")


class PredictionResponse(PredictionBase):
    id: int
    customer_id: int
    predicted_at: datetime

    class Config:
        from_attributes = True


class SegmentDetail(BaseModel):
    cluster_id: Optional[int] = Field(None, description="Behavioral customer cluster segment assignment")
    persona: Optional[str] = Field(None, description="Mapped customer cohort persona characteristics")


class SimulationDetail(BaseModel):
    intervention: str = Field(..., description="Prescriptive action campaign description")
    original_risk: float = Field(..., description="Original churn probability score")
    simulated_risk: float = Field(..., description="Simulated churn probability after intervention")
    risk_reduction: float = Field(..., description="Expected reduction in churn risk")


class CustomerExplainResponse(BaseModel):
    customer_id: str
    gender: str
    tenure: int
    monthly_charges: float
    total_charges: float
    churn_probability: float
    is_high_risk: bool
    top_drivers: List[ShapDriver]
    save_plays: List[SavePlay]
    cluster: Optional[int] = None
    cohort_persona: Optional[str] = None
    segmentation: Optional[SegmentDetail] = None
    simulations: Optional[List[SimulationDetail]] = None
    predicted_at: datetime

