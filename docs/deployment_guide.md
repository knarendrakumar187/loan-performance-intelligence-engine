# Loan Performance Intelligence Engine (LPIE) — Deployment Guide

This guide provides end-to-end instructions for running and deploying the **Loan Performance Intelligence Engine** locally, in Docker containers, and to free/production cloud platforms.

---

## 1. Local Execution Options

### Option A: Command Line Pipeline (Fastest Batch Execution)

```bash
# 1. Clone repository
git clone https://github.com/knarendrakumar187/loan-performance-intelligence-engine.git
cd loan-performance-intelligence-engine

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run entire pipeline (Phases 0 through 8 in ~65s)
python run_pipeline.py
```

### Option B: Interactive Streamlit Web Dashboard

```bash
# Launch interactive web application
streamlit run app.py
```
- Open browser at **`http://localhost:8501`**
- Provides live single-loan scoring, TreeSHAP waterfall explanations, scenario stress testing sliders, anomaly case study explorer, and copilot review notes generator.

### Option C: Production FastAPI REST API Server

```bash
# Launch REST API server
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```
- Open Swagger API Documentation at **`http://localhost:8000/docs`**
- Test endpoints:
  - `POST /api/v1/predict/loan`: Real-time loan scoring
  - `POST /api/v1/copilot/review-note`: Generate SHAP-grounded reviewer notes
  - `GET /health`: Health status

---

## 2. Docker Container Deployment (1-Command)

### Using Docker Directly

```bash
# 1. Build Docker image
docker build -t lpie-engine:latest .

# 2. Run container (Streamlit Dashboard on port 8501)
docker run -d -p 8501:8501 -p 8000:8000 --name lpie-app lpie-engine:latest

# 3. Access Dashboard
# Visit http://localhost:8501 in your browser
```

### Using Docker Compose (Runs Dashboard + API together)

```bash
# Start both Web Dashboard and REST API services
docker-compose up -d

# View running services
docker-compose ps

# Stop services
docker-compose down
```

---

## 3. Free Cloud Deployment Options

### Option 1: Deploy to Streamlit Community Cloud (Free & Instant)
1. Fork or push this repository to your GitHub account: `https://github.com/knarendrakumar187/loan-performance-intelligence-engine`.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Sign in with GitHub and click **"New app"**.
4. Select repository: `loan-performance-intelligence-engine`, Branch: `master`, Main file path: `app.py`.
5. Click **"Deploy!"**.
6. Your live web dashboard will be accessible via a public URL (e.g. `https://loan-performance-intelligence.streamlit.app`).

### Option 2: Deploy to Hugging Face Spaces (Free Docker/Streamlit)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Select **Streamlit** or **Docker** SDK.
3. Link your GitHub repository or push the repository files.
4. Hugging Face will automatically build and host the interactive app with hardware acceleration.

### Option 3: Deploy to Render / Railway / Fly.io (Free Tier Cloud Container)
1. Connect your GitHub repository to [Render.com](https://render.com) or [Railway.app](https://railway.app).
2. Create a **New Web Service**.
3. Set Build Command: `pip install -r requirements.txt && python run_pipeline.py`
4. Set Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. Click **Deploy**.

---

## 4. REST API Integration Example (Python & cURL)

### Single-Loan Scoring via cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict/loan" \
     -H "Content-Type: application/json" \
     -d '{
       "loan_id": "LN000123",
       "month_index": 18,
       "loan_age_months": 18,
       "remaining_term_months": 342,
       "original_balance": 350000,
       "current_balance": 330000,
       "interest_rate": 6.75,
       "credit_score_band": "660-699",
       "ltv_band": "70-80",
       "dti_band": "30-40",
       "state": "CA",
       "loan_purpose": "Purchase",
       "occupancy_type": "Primary",
       "property_type": "Single_Family",
       "servicer_name": "ServicerA",
       "current_status": "Current",
       "days_past_due": 0,
       "modification_flag": 0
     }'
```

### Response Payload

```json
{
  "loan_id": "LN000123",
  "prob_3m_delinquency": 0.2415,
  "prob_6m_delinquency": 0.3120,
  "prob_12m_default": 0.1142,
  "prob_12m_prepayment": 0.4850,
  "pred_next_state": "Current",
  "anomaly_score": 0.0850,
  "exception_type": "none",
  "recommended_action": "APPROVE",
  "confidence": 0.886
}
```
