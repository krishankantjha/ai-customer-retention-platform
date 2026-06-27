from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.auth import Token
from app.services.auth_service import authenticate_user
from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    OAuth2 compatible token login, retrieve a JWT access token for future requests.
    This endpoint supports interactive login in Swagger UI.
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(subject=user)
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


from app.schemas.auth import UserCreate, UserResponse
from app.database.models.user import User
from app.core.security import get_password_hash
from app.core.config import settings

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account with secure Bcrypt password hashing."""
    # Check if username is settings.ADMIN_USERNAME
    if user_in.username.lower() == settings.ADMIN_USERNAME.lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
        
    # Check if user already exists in db
    existing_user = db.query(User).filter(User.username == user_in.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
        
    # Hash password and save
    hashed = get_password_hash(user_in.password)
    new_user = User(username=user_in.username, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
