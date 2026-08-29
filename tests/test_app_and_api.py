"""
End-to-end integration and smoke tests for LPIE app, API, and core modules.
Verifies zero runtime exceptions across all features and endpoints.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from fastapi.testclient import TestClient

from src import config
from src.api.server import app, load_models
from src.features.engineer import engineer_features
from src.anomaly.ml_scorer import compute_composite_anomaly_scores
from src.copilot.reviewer import generate_reviewer_note_local, load_data_dictionary


@pytest.fixture(scope="module")
def api_client():
    load_models()
    return TestClient(app)


def test_api_health(api_client):
    """Test health endpoint."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True


def test_api_predict_single_loan(api_client):
    """Test single loan prediction endpoint."""
    payload = {
        "loan_id": "TEST_001",
        "month_index": 18,
        "loan_age_months": 18,
        "remaining_term_months": 342,
        "original_balance": 300000.0,
        "current_balance": 285000.0,
        "interest_rate": 6.5,
        "credit_score_band": "700-739",
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
    }
    response = api_client.post("/api/v1/predict/loan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prob_12m_default" in data
    assert 0.0 <= data["prob_12m_default"] <= 1.0
    assert "prob_12m_prepayment" in data
    assert 0.0 <= data["prob_12m_prepayment"] <= 1.0
    assert data["recommended_action"] in ["APPROVE", "WATCH_LIST", "ESCALATE", "IMMEDIATE_REVIEW"]


def test_api_copilot_review_note(api_client):
    """Test grounded copilot reviewer note endpoint."""
    payload = {
        "loan_id": "TEST_002",
        "month_index": 12,
        "loan_age_months": 12,
        "remaining_term_months": 348,
        "original_balance": 250000.0,
        "current_balance": 245000.0,
        "interest_rate": 7.25,
        "credit_score_band": "620-659",
        "ltv_band": "80-90",
        "dti_band": "40-50",
        "state": "FL",
        "loan_purpose": "Purchase",
        "occupancy_type": "Primary",
        "property_type": "Single_Family",
        "servicer_name": "ServicerB",
        "current_status": "30DPD",
        "days_past_due": 30,
        "modification_flag": 0
    }
    response = api_client.post("/api/v1/copilot/review-note", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reviewer_note" in data
    assert "Reviewer Note" in data["reviewer_note"]
    assert "SHAP" in data["reviewer_note"]


def test_models_exist_and_predict():
    """Test that all improved models predict valid probability ranges."""
    model_dir = config.PROJECT_ROOT / "checkpoints" / "improved"
    models = joblib.load(model_dir / "improved_models.joblib")
    feature_cols = joblib.load(model_dir / "feature_columns.joblib")

    dummy_X = np.zeros((5, len(feature_cols)))
    for target, model in models.items():
        if target == "next_state":
            preds = model.predict(dummy_X)
            assert len(preds) == 5
        else:
            probs = model.predict_proba(dummy_X)[:, 1]
            assert len(probs) == 5
            assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_anomaly_scoring_pipeline():
    """Test anomaly detection scoring pipeline."""
    sample_df = pd.DataFrame([{
        "loan_id": "LN_ANOM_01",
        "month_index": 5,
        "current_balance": -50000.0,  # Negative balance anomaly
        "original_balance": 200000.0,
        "days_past_due": 0,
        "current_status": "Current",
        "interest_rate": 6.0,
    }])
    scores, types, rules_df = compute_composite_anomaly_scores(sample_df)
    assert len(scores) == 1
    assert scores.iloc[0] > 0.10
    assert types.iloc[0] == "data_entry_error"
