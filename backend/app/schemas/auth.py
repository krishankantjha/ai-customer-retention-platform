from typing import Optional
from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str = Field(..., description="The JWT access token")
    token_type: str = Field(..., description="The type of token (e.g. bearer)")


class TokenData(BaseModel):
    username: Optional[str] = Field(None, description="Username encoded in the JWT token subject")


class UserLogin(BaseModel):
    username: str = Field(..., description="Username used for authentication")
    password: str = Field(..., description="Password used for authentication")
