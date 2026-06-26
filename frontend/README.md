# RetainIQ Streamlit Web Portal

This directory contains the Streamlit dashboard client for the RetainIQ platform.

---

## Directory Structure & Views

The portal interface is structured into modular tab views under the `views/` directory:

* **`app.py`**: The application entrypoint. Configures global CSS styling, navigation sidebar, and route dispatching.
* **`api_client.py`**: Handles API connections using `RetainIQAPIClient`.
* **`views/auth_view.py`**: Provides the login form and handles session authentication token handshakes.
* **`views/dashboard_view.py`**: Displays executive-level metrics, risk distributions, and at-risk queues.
* **`views/explorer_view.py`**: Shows demographic summaries, churn drivers, local SHAP breakdowns, and save play recommendations.
* **`views/counterfactual_view.py`**: Houses interactive sliders to model how telemetry modifications affect customer churn probability risk.
* **`views/executive_view.py`**: Plots contract splits, tenure scatters, and OLS regression analysis lines.
* **`views/explainability_view.py`**: Displays global model metrics, SHAP summary plots, and Beeswarm charts.
* **`views/segments_view.py`**: Outlines customer segments built via K-Means++ clustering.
* **`views/ingestion_view.py`**: Handles batch cohort file uploads and monitors preprocessing pipelines.
* **`views/drift_view.py`**: Displays feature distributions, statistical tests, and PSI drift diagnostics.
* **`views/settings_view.py`**: Contains profile settings, alert threshold adjustments, and webhook definitions.

---

## Local Setup

To run the analytical web portal locally:

1. **Activate the Virtual Environment**:
   ```bash
   # Run from the project root
   source venv/Scripts/activate
   ```
2. **Start the Streamlit Application**:
   ```bash
   # Navigate to the frontend folder
   cd frontend/
   
   # Start the server
   streamlit run app.py
   ```
   The portal should automatically launch in your default web browser at: http://localhost:8501
