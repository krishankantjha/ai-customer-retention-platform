"""
Application configuration module.
Loads settings from environment variables or a local .env file.
"""

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core API Configurations
    APP_NAME: str = "AI Customer Retention Platform API"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api"

    # Security & Authentication
    JWT_SECRET: str = "super-secret-key-change-in-production-1234567890"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ADMIN_USERNAME: str = "admin"
    # Default bcrypt hash for "password"
    ADMIN_PASSWORD_HASH: str = "$2b$12$w5uwmvOZ7LEYex5dC7L3/uzqj0jXnSsOJuwwj3zAXWDjE5jRi8LUG"

    # Database Persistence
    DATABASE_URL: str = "sqlite:///./customer_retention.db"

    # CORS Allowed Origins
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501"

    # Machine Learning Model Paths
    MODEL_PATH: str = "ml/artifacts/model.pkl"
    PIPELINE_PATH: str = "ml/artifacts/pipeline.pkl"

    _DEFAULT_JWT_SECRET: str = "super-secret-key-change-in-production-1234567890"

    @model_validator(mode="after")
    def validate_allowed_origins(self) -> "Settings":
        if self.ALLOWED_ORIGINS:
            origins = [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]
            for origin in origins:
                if not (origin.startswith("http://") or origin.startswith("https://")):
                    import logging
                    logger = logging.getLogger("backend.app.core.config")
                    logger.error(
                        f"CORS CONFIGURATION ERROR: Invalid origin '{origin}' in ALLOWED_ORIGINS. "
                        f"Origins must start with http:// or https://. "
                        f"Falling back to default localhost origins."
                    )
                    self.ALLOWED_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"
                    break
        return self


    @model_validator(mode="after")
    def enforce_secret_rotation_in_production(self) -> "Settings":
        """Raise at startup if the default JWT secret is still set in production."""
        if self.APP_ENV == "production" and self.JWT_SECRET == self._DEFAULT_JWT_SECRET:
            raise ValueError(
                "SECURITY ERROR: JWT_SECRET must be changed from the default value "
                "before deploying to production. Set the JWT_SECRET environment "
                "variable to a cryptographically secure random string."
            )
        return self

    @model_validator(mode="after")
    def resolve_sqlite_db_path(self) -> "Settings":
        if self.DATABASE_URL.startswith("sqlite:///"):
            db_path = self.DATABASE_URL.split("sqlite:///", 1)[1]
            if not os.path.isabs(db_path):
                backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if db_path.startswith("./") or db_path.startswith(".\\"):
                    db_path = db_path[2:]
                abs_db_path = os.path.abspath(os.path.join(backend_dir, db_path))
                self.DATABASE_URL = f"sqlite:///{abs_db_path}"
        return self


settings = Settings()
