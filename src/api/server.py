"""
FastAPI REST API Service for the Loan Performance Intelligence Engine (LPIE).
Exposes real-time scoring, anomaly detection, stress simulation, and copilot endpoints.
"""

from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src import config
from src.anomaly.ml_scorer import compute_composite_anomaly_scores
from src.copilot.reviewer import generate_reviewer_note_local, load_data_dictionary
from src.features.engineer import engineer_features

# ──────────────────────────────────────────────
# App Initialization
# ──────────────────────────────────────────────
app = FastAPI(
    title="Loan Performance Intelligence Engine (LPIE) API",
    description="Production REST API for multi-horizon loan delinquency prediction, default hazard estimation, anomaly detection, and grounded copilot generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODELS = {}
FEATURE_COLS = []


@app.on_event("startup")
def load_models():
    """Load model checkpoints into memory at startup."""
    global MODELS, FEATURE_COLS
    model_dir = config.PROJECT_ROOT / "checkpoints" / "improved"
    MODELS = joblib.load(model_dir / "improved_models.joblib")
    FEATURE_COLS = joblib.load(model_dir / "feature_columns.joblib")
    print(f"[✓] REST API initialized: loaded {len(MODELS)} models with {len(FEATURE_COLS)} feature columns")


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────
class LoanInput(BaseModel):
    loan_id: str = Field(default="LN000001", description="Unique loan identifier")
    month_index: int = Field(default=18, description="Month index on book")
    loan_age_months: int = Field(default=18, description="Age of loan in months")
    remaining_term_months: int = Field(default=342, description="Remaining contract term")
    original_balance: float = Field(default=300000.0, description="Original loan amount")
    current_balance: float = Field(default=285000.0, description="Current unpaid principal balance")
    interest_rate: float = Field(default=6.5, description="Note interest rate (%)")
    credit_score_band: str = Field(default="700-739", description="Credit score band")
    ltv_band: str = Field(default="70-80", description="Loan-to-value band")
    dti_band: str = Field(default="30-40", description="Debt-to-income band")
    state: str = Field(default="CA", description="US State code")
    loan_purpose: str = Field(default="Purchase", description="Loan purpose")
    occupancy_type: str = Field(default="Primary", description="Occupancy type")
    property_type: str = Field(default="Single_Family", description="Property type")
    servicer_name: str = Field(default="ServicerA", description="Servicer name")
    current_status: str = Field(default="Current", description="Current delinquency status")
    days_past_due: int = Field(default=0, description="Days past due")
    modification_flag: int = Field(default=0, description="Loan modification indicator")


class PredictionOutput(BaseModel):
    loan_id: str
    prob_3m_delinquency: float
    prob_6m_delinquency: float
    prob_12m_default: float
    prob_12m_prepayment: float
    pred_next_state: str
    anomaly_score: float
    exception_type: str
    recommended_action: str
    confidence: float


# ──────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Loan Performance Intelligence Engine API",
        "version": "1.0.0",
        "models_loaded": len(MODELS) > 0,
    }


@app.post("/api/v1/predict/loan", response_model=PredictionOutput, tags=["Prediction"])
def predict_single_loan(loan: LoanInput):
    """Generate multi-horizon delinquency, default, prepayment, and anomaly predictions for a single loan."""
    if not MODELS:
        load_models()

    row_dict = loan.model_dump()
    row_dict["reporting_month"] = "2024-04"
    row_dict["origination_month"] = "2022-10"

    df_single = pd.DataFrame([row_dict])
    df_proc = engineer_features(df_single, is_train=False)

    for col in FEATURE_COLS:
        if col not in df_proc.columns:
            df_proc[col] = 0.0

    X = df_proc[FEATURE_COLS].fillna(0).values

    p_3m = float(MODELS["next_3m_delinquency_flag"].predict_proba(X)[:, 1][0])
    p_6m = float(MODELS["next_6m_delinquency_flag"].predict_proba(X)[:, 1][0])
    p_def = float(MODELS["next_12m_default_flag"].predict_proba(X)[:, 1][0])
    p_prep = float(MODELS["next_12m_prepayment_flag"].predict_proba(X)[:, 1][0])
    next_st = str(MODELS["next_state"].predict(X)[0])

    anom_scores, exc_types, _ = compute_composite_anomaly_scores(df_single)
    anom = float(anom_scores.iloc[0])
    exc_type = str(exc_types.iloc[0])

    if anom >= 0.65 or p_def >= 0.60:
        act = "IMMEDIATE_REVIEW"
    elif p_def >= 0.35 or anom >= 0.40:
        act = "ESCALATE"
    elif p_def >= 0.15:
        act = "WATCH_LIST"
    else:
        act = "APPROVE"

    conf = round(float(0.50 + abs(p_def - 0.50)), 3)

    return PredictionOutput(
        loan_id=loan.loan_id,
        prob_3m_delinquency=round(p_3m, 4),
        prob_6m_delinquency=round(p_6m, 4),
        prob_12m_default=round(p_def, 4),
        prob_12m_prepayment=round(p_prep, 4),
        pred_next_state=next_st,
        anomaly_score=round(anom, 4),
        exception_type=exc_type,
        recommended_action=act,
        confidence=conf,
    )


@app.post("/api/v1/copilot/review-note", tags=["Copilot"])
def generate_review_note(loan: LoanInput):
    """Generate a grounded, audit-ready markdown reviewer note citing TreeSHAP values."""
    if not MODELS:
        load_models()

    row_dict = loan.model_dump()
    row_dict["reporting_month"] = "2024-04"
    row_dict["origination_month"] = "2022-10"

    df_single = pd.DataFrame([row_dict])
    df_proc = engineer_features(df_single, is_train=False)

    for col in FEATURE_COLS:
        if col not in df_proc.columns:
            df_proc[col] = 0.0

    X = df_proc[FEATURE_COLS].fillna(0).values
    p_def = float(MODELS["next_12m_default_flag"].predict_proba(X)[:, 1][0])
    p_prep = float(MODELS["next_12m_prepayment_flag"].predict_proba(X)[:, 1][0])

    base_est = MODELS["next_12m_default_flag"]
    if hasattr(base_est, "calibrated_classifiers_"):
        base_est = base_est.calibrated_classifiers_[0].estimator

    explainer = shap.TreeExplainer(base_est)
    shap_vals = explainer.shap_values(X)[0]
    shap_dict = {FEATURE_COLS[i]: float(shap_vals[i]) for i in range(len(FEATURE_COLS))}

    prediction = {"default_prob": p_def, "prepay_prob": p_prep, "confidence": "High" if p_def < 0.1 or p_def > 0.8 else "Medium"}
    data_dict = load_data_dictionary()
    note = generate_reviewer_note_local(row_dict, shap_dict, prediction, data_dict)

    return {"loan_id": loan.loan_id, "reviewer_note": note}
