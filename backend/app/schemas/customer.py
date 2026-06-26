from typing import Optional
from pydantic import BaseModel, Field, field_validator


class CustomerBase(BaseModel):
    customer_id: str = Field(..., description="Unique customer ID from dataset")
    gender: str = Field(..., description="Gender of the customer (Male, Female)")
    senior_citizen: int = Field(..., description="Senior citizen status (0 or 1)")
    partner: str = Field(..., description="Whether the customer has a partner (Yes, No)")
    dependents: str = Field(..., description="Whether the customer has dependents (Yes, No)")
    tenure: int = Field(..., description="Number of months the customer has stayed with the company")
    phone_service: str = Field(..., description="Whether the customer has a phone service (Yes, No)")
    multiple_lines: str = Field(..., description="Whether the customer has multiple lines (Yes, No, No phone service)")
    internet_service: str = Field(..., description="Customer's internet service provider database value (DSL, Fiber optic, No)")
    online_security: str = Field(..., description="Whether the customer has online security (Yes, No, No internet service)")
    online_backup: str = Field(..., description="Whether the customer has online backup (Yes, No, No internet service)")
    device_protection: str = Field(..., description="Whether the customer has device protection (Yes, No, No internet service)")
    tech_support: str = Field(..., description="Whether the customer has tech support (Yes, No, No internet service)")
    streaming_tv: str = Field(..., description="Whether the customer has streaming TV (Yes, No, No internet service)")
    streaming_movies: str = Field(..., description="Whether the customer has streaming movies (Yes, No, No internet service)")
    contract: str = Field(..., description="The contract term of the customer (Month-to-month, One year, Two year)")
    paperless_billing: str = Field(..., description="Whether the customer has paperless billing (Yes, No)")
    payment_method: str = Field(..., description="The customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))")
    monthly_charges: float = Field(..., description="The amount charged to the customer monthly")
    total_charges: float = Field(..., description="The total amount charged to the customer")
    churn: Optional[str] = Field(None, description="Whether the customer churned (Yes, No, or null)")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in {"Male", "Female"}:
            raise ValueError("gender must be either 'Male' or 'Female'")
        return v

    @field_validator("senior_citizen")
    @classmethod
    def validate_senior_citizen(cls, v: int) -> int:
        if v not in {0, 1}:
            raise ValueError("senior_citizen must be 0 or 1")
        return v

    @field_validator("partner", "dependents", "phone_service", "paperless_billing")
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        if v not in {"Yes", "No"}:
            raise ValueError("value must be either 'Yes' or 'No'")
        return v

    @field_validator("multiple_lines")
    @classmethod
    def validate_multiple_lines(cls, v: str) -> str:
        if v not in {"Yes", "No", "No phone service"}:
            raise ValueError("multiple_lines must be 'Yes', 'No', or 'No phone service'")
        return v

    @field_validator("internet_service")
    @classmethod
    def validate_internet_service(cls, v: str) -> str:
        if v not in {"DSL", "Fiber optic", "No"}:
            raise ValueError("internet_service must be 'DSL', 'Fiber optic', or 'No'")
        return v

    @field_validator("online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies")
    @classmethod
    def validate_internet_addons(cls, v: str) -> str:
        if v not in {"Yes", "No", "No internet service"}:
            raise ValueError("value must be 'Yes', 'No', or 'No internet service'")
        return v

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        if v not in {"Month-to-month", "One year", "Two year"}:
            raise ValueError("contract must be 'Month-to-month', 'One year', or 'Two year'")
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment_method(cls, v: str) -> str:
        expected = {
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        }
        if v not in expected:
            raise ValueError(f"payment_method must be one of {expected}")
        return v

    @field_validator("tenure")
    @classmethod
    def validate_tenure(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tenure cannot be negative")
        return v

    @field_validator("monthly_charges")
    @classmethod
    def validate_monthly_charges(cls, v: float) -> float:
        if v < 0:
            raise ValueError("monthly_charges cannot be negative")
        return v

    @field_validator("total_charges")
    @classmethod
    def validate_total_charges(cls, v: float) -> float:
        if v < 0:
            raise ValueError("total_charges cannot be negative")
        return v

    @field_validator("churn")
    @classmethod
    def validate_churn(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"Yes", "No"}:
            raise ValueError("churn must be either 'Yes' or 'No'")
        return v


class CustomerCreate(CustomerBase):
    pass


class CustomerResponse(CustomerBase):
    id: int
    upload_id: int

    class Config:
        from_attributes = True


class CustomerBrief(BaseModel):
    id: int
    customer_id: str
    gender: str
    tenure: int
    monthly_charges: float
    churn: Optional[str] = None

    class Config:
        from_attributes = True
