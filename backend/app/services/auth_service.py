from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import ValidationError

from app.core.config import settings
from app.core.security import verify_password
from app.schemas.auth import TokenData

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login"
)


from sqlalchemy.orm import Session
from app.database.session import get_db

def authenticate_user(db: Session, username: str, password: str) -> Optional[str]:
    """Authenticate a user using settings configurations or database profiles."""
    if username == settings.ADMIN_USERNAME and verify_password(password, settings.ADMIN_PASSWORD_HASH):
        return username
    
    from app.database.models.user import User
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.hashed_password):
        return username
        
    return None


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> str:
    """FastAPI dependency to extract and validate the JWT token, returning the username."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except (JWTError, ValidationError):
        raise credentials_exception
    
    if token_data.username == settings.ADMIN_USERNAME:
        return token_data.username
        
    from app.database.models.user import User
    user = db.query(User).filter(User.username == token_data.username).first()
    if not user:
        raise credentials_exception
        
    return token_data.username
