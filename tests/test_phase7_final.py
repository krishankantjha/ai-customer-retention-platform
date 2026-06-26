"""
Unit and Integration Tests for Phase 7 (Final Production Upgrades):
Rate limiting, database health verification, search autocompletion, and upload status endpoints.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force backend directory in path
backend_dir = os.path.abspath(os.path.join(project_root, "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.database.models.uploads import Upload
from app.database.models.customer import Customer
from app.core.rate_limiter import RateLimiterMiddleware

client = TestClient(app)


@pytest.fixture
def auth_headers():
    """Authenticates the test client and yields bearer headers."""
    login_resp = client.post("/api/v1/auth/login", data={"username": "admin", "password": "password"})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_db():
    """Generates a mock DB session and clears dependency overrides afterward."""
    db = MagicMock()
    yield db
    app.dependency_overrides.clear()


def test_rate_limiter_middleware_logic():
    """
    Asserts rate limiter middleware correctly counts requests and returns 429
    when thresholds are breached.
    """
    test_app = FastAPI()
    # Configure with low limits (2 requests per window) to test rate limit triggers
    test_app.add_middleware(RateLimiterMiddleware, limit=2, window_seconds=10)
    
    @test_app.get("/api/v1/upload")
    def dummy_upload():
        return {"ok": True}
        
    @test_app.get("/api/v1/customers/123/explain")
    def dummy_explain():
        return {"ok": True}
        
    test_client = TestClient(test_app)
    
    # Request 1: Allowed
    r1 = test_client.get("/api/v1/upload")
    assert r1.status_code == 200
    
    # Request 2: Allowed
    r2 = test_client.get("/api/v1/upload")
    assert r2.status_code == 200
    
    # Request 3: Blocked (429)
    r3 = test_client.get("/api/v1/upload")
    assert r3.status_code == 429
    assert r3.json()["detail"] == "Too many requests. Please try again later."


def test_database_health_check_endpoint_healthy(mock_db):
    """Asserts GET /health returns 200 when database executes successfully."""
    from app.database.session import get_db
    app.dependency_overrides[get_db] = lambda: mock_db
    
    # Mock simple successful query execution
    mock_db.execute.return_value = None
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}



def test_database_health_check_endpoint_degraded(mock_db):
    """Asserts GET /health returns 500 when SQL execution fails."""
    from app.database.session import get_db
    app.dependency_overrides[get_db] = lambda: mock_db
    
    # Simulate DB error
    mock_db.execute.side_effect = Exception("DB Connection Timeout")
    
    response = client.get("/health")
    assert response.status_code == 500
    assert "Database connection degraded" in response.json()["detail"]


def test_upload_status_tracking_endpoint(auth_headers, mock_db):
    """Asserts GET /uploads/{upload_id}/status fetches correct status from DB."""
    from app.database.session import get_db
    app.dependency_overrides[get_db] = lambda: mock_db
    
    # Build fake upload record
    fake_upload = Upload(
        id=999,
        filename="telecom_users.csv",
        status="processing",
        row_count=100,
        error_message=None
    )
    
    mock_db.query.return_value.filter.return_value.first.return_value = fake_upload
    
    response = client.get("/api/v1/uploads/999/status", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["upload_id"] == 999
    assert data["status"] == "processing"
    assert data["filename"] == "telecom_users.csv"
    assert data["row_count"] == 100


def test_upload_status_endpoint_not_found(auth_headers, mock_db):
    """Asserts GET /uploads/{upload_id}/status returns 404 for missing IDs."""
    from app.database.session import get_db
    app.dependency_overrides[get_db] = lambda: mock_db
    
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    response = client.get("/api/v1/uploads/888/status", headers=auth_headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_customer_autocomplete_search_endpoint(auth_headers, mock_db):
    """Asserts GET /customers/search returns matched prefix IDs."""
    from app.database.session import get_db
    app.dependency_overrides[get_db] = lambda: mock_db
    
    # Mock database returning prefix hits
    mock_db.query.return_value.filter.return_value.limit.return_value.all.return_value = [
        ("7590-VHVEG",),
        ("7590-XXXXX",)
    ]
    
    response = client.get("/api/v1/customers/search?q=7590", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == ["7590-VHVEG", "7590-XXXXX"]


def test_customer_autocomplete_search_empty(auth_headers, mock_db):
    """Asserts autocomplete search returns empty list when prefix is empty."""
    from app.database.session import get_db
    app.dependency_overrides[get_db] = lambda: mock_db
    
    response = client.get("/api/v1/customers/search?q=", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []
