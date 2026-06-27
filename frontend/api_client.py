import os
import requests
import streamlit as st

class RetainIQAPIClient:
    """
    Unified class-based HTTP client for all communications between the Streamlit frontend
    and the FastAPI backend service.
    """
    def __init__(self, base_url: str = None):
        if not base_url:
            try:
                base_url = st.secrets.get("API_BASE_URL")
            except Exception:
                pass
            if not base_url:
                base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
        self.base_url = base_url
        
    def get_headers(self) -> dict:
        """Helper to retrieve JWT authentication headers from st.session_state."""
        token = st.session_state.get("jwt_token")
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}
        
    def login(self, username, password):
        """Sends user login credentials and returns token response."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/auth/login",
                data={"username": username, "password": password},
                timeout=5
            )
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"detail": f"Failed to connect to API server: {str(e)}"}

    def register(self, username, password):
        """Registers a new user account with custom credentials."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/auth/register",
                json={"username": username, "password": password},
                timeout=5
            )
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"detail": f"Failed to connect to API server: {str(e)}"}

    def get_overview(self):
        """Fetches high-level cohort overview metrics."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/analytics/overview",
                headers=self.get_headers(),
                timeout=10
            )
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"detail": f"Failed to fetch overview: {str(e)}"}

    def get_save_plays(self):
        """Fetches list of targeted campaign recommendations (Save Plays)."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/analytics/save-plays",
                headers=self.get_headers(),
                timeout=10
            )
            return response.status_code, response.json()
        except Exception:
            return 500, []

    def get_cohort_data(self):
        """Fetches detailed customer rows for demographic segmentation charts."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/analytics/cohort-data",
                headers=self.get_headers(),
                timeout=15
            )
            return response.status_code, response.json()
        except Exception:
            return 500, []

    def get_diagnostics_metadata(self):
        """Fetches baseline classifier diagnostics and SHA checks."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/analytics/diagnostics-metadata",
                headers=self.get_headers(),
                timeout=10
            )
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"success": False, "message": str(e)}

    def get_model_health(self):
        """Fetches KS test feature drift statistics and baseline scoring metrics."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/analytics/model-health",
                headers=self.get_headers(),
                timeout=15
            )
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"detail": f"Failed to fetch model health: {str(e)}"}

    def upload_cohort(self, files):
        """Uploads raw CSV files for batch processing."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/upload",
                headers=self.get_headers(),
                files=files,
                timeout=30
            )
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"detail": f"Network error during upload: {str(e)}"}

    def get_upload_status(self, upload_id: str):
        """Queries async state of database upload tasks (pending, processing, completed, failed)."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/uploads/{upload_id}/status",
                headers=self.get_headers(),
                timeout=5
            )
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"detail": f"Failed to fetch upload status: {str(e)}"}

    def get_customer_explanation(self, cust_id: str):
        """Fetches SHAP force plots, demographics, recommendations and counterfactuals for an ID."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/customers/{cust_id}/explain",
                headers=self.get_headers(),
                timeout=10
            )
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"detail": f"Failed to query customer explanation: {str(e)}"}

    def search_customers(self, query: str):
        """Fetches a list of matching customer IDs for predictive search autocomplete."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/customers/search",
                headers=self.get_headers(),
                params={"q": query},
                timeout=5
            )
            return response.status_code, response.json()
        except Exception:
            return 500, []

    def simulate_intervention(self, customer_data: dict):
        """Sends modified customer demographics/services to the backend to get a live simulated churn score."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/predict/simulate",
                headers=self.get_headers(),
                json=customer_data,
                timeout=10
            )
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"detail": f"Failed to run simulation: {str(e)}"}

    def get_health(self):
        """Calls database validation health checks on backend service."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code, response.json()
        except Exception as e:
            return 500, {"status": "degraded", "detail": str(e)}
