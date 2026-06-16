# RetainIQ — AI-Powered Customer Retention Intelligence Platform
RetainIQ is an end-to-end, production-grade machine learning platform designed to predict, analyze, and mitigate customer churn. The platform uses classification models trained on subscription customer datasets (modeled on the IBM Telco Churn dataset) and translates findings into actionable "Save Plays" to protect Monthly Recurring Revenue (MRR).
---
## 🚀 Key Features
*   **Real-time Inference API:** Single-customer churn scoring returning risk percentages, risk levels (High/Medium/Low), and top churn drivers.
*   **Asynchronous Batch Inference:** Drag-and-drop CSV file uploader in the React UI that processes thousands of rows concurrently via background workers.
*   **Explainable AI (XAI):** Real-time feature contribution explanations (such as SHAP values) showing why a customer is marked at risk.
*   **Prescriptive Save Plays:** Automated action recommendations matched to customer risk factors (e.g. auto-pay upgrades for payment friction).
*   **Interactive Executive Dashboard:** Recharts-based KPI visualizer displaying total accounts at risk, average churn rates, and revenue impact.
*   **Multi-Container Architecture:** Containerized using Docker and orchestrated via Docker Compose with Nginx handling routing proxies.
---
## 📁 Repository Structure
```text
ai-customer-retention-platform/
├── backend/                  # FastAPI web server
│   ├── app/
│   │   ├── api/              # Routers and auth middleware
│   │   ├── core/             # Configuration, logging, and security
│   │   ├── database/         # SQLAlchemy models and migrations
│   │   ├── ml/               # Inference engine (model loaders, preprocessors)
│   │   └── services/         # Core business logic services
│   ├── tests/                # Pytest integration tests
│   └── requirements.txt      # Python dependencies list
├── frontend/                 # React SPA (Vite + Vanilla CSS)
│   ├── src/
│   │   ├── components/       # Visual elements, charts, and forms
│   │   ├── pages/            # Page layouts (Dashboard, CustomerRisk, Uploads)
│   │   └── services/         # Axios API connection layers
│   └── package.json          # Node dependencies list
├── ml/                       # Machine Learning pipelines
│   ├── notebooks/            # EDA, engineering, and modeling notebooks
│   ├── preprocessing/        # Data validation and feature pipelines
│   ├── training/             # Fitting, hyperparameter tuning, and evaluation
│   └── artifacts/            # Serialized models and scaling pipelines
├── configs/                  # Global YAML logging and model parameters
├── docker/                   # Compose files and Nginx reverse proxy configurations
├── docs/                     # Architecture manuals and domain translations
├── data/                     # Raw and processed datasets (ignored by git)
└── scripts/                  # Automated setup and seeding scripts
```
---
## 🛠️ Getting Started
### Prerequisites
*   Python 3.9+
*   Node.js 18+
*   Git Bash, WSL, or macOS/Linux terminal (for executing `.sh` scripts)
### 1. Initialize Local Environment
Run the setup script from the repository root to initialize your Python virtual environment, upgrade package managers, and install dependencies:
```bash
./scripts/setup.sh
```
### 2. Configure Environment Variables
Open the generated local configuration file `backend/.env` and replace the placeholder value for `JWT_SECRET` with a secure random key:
```env
JWT_SECRET="your-secure-random-token-here"
```
### 3. Run the Applications
#### Run Backend API
```bash
cd backend
source ../venv/Scripts/activate # Windows Git Bash
# source ../venv/bin/activate   # Linux/macOS
uvicorn app.main:app --reload
```
The backend API documentation will be available at: http://localhost:8000/docs
#### Run Frontend Client
```bash
cd frontend
npm run dev
```
The frontend application will be available at: http://localhost:5173
---
## 📄 License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
# ai-customer-retention-