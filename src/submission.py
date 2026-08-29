"""
Submission generation script for FinTech AI Challenge.
Produces submission/submission.csv with calibrated model predictions, anomaly flags,
and explainability outputs for the test dataset.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from src import config
from src.anomaly.ml_scorer import compute_composite_anomaly_scores
from src.data.loader import load_test
from src.features.engineer import engineer_features


def generate_submission() -> pd.DataFrame:
    """Generate final submission.csv from test set."""
    print("=" * 60)
    print("Generating Final Competition Submission (submission/submission.csv)")
    print("=" * 60)

    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_file = config.SUBMISSION_OUTPUT_FILE

    # 1. Load test data
    raw_test = load_test()
    print(f"Loaded raw test records: {len(raw_test):,}")

    # 2. Process features
    test_proc = engineer_features(raw_test, is_train=False)

    # 3. Load trained predictive models
    model_dir = config.PROJECT_ROOT / "checkpoints" / "improved"
    models = joblib.load(model_dir / "improved_models.joblib")
    feature_cols = joblib.load(model_dir / "feature_columns.joblib")

    # Align feature columns
    for col in feature_cols:
        if col not in test_proc.columns:
            test_proc[col] = 0.0

    X_test = test_proc[feature_cols].fillna(0).values

    # 4. Predict probabilities
    print("Computing calibrated predictive probabilities...")
    p_3m_delinq = models["next_3m_delinquency_flag"].predict_proba(X_test)[:, 1]
    p_6m_delinq = models["next_6m_delinquency_flag"].predict_proba(X_test)[:, 1]
    p_12m_default = models["next_12m_default_flag"].predict_proba(X_test)[:, 1]
    p_12m_prepay = models["next_12m_prepayment_flag"].predict_proba(X_test)[:, 1]
    pred_next_state = models["next_state"].predict(X_test)

    # 5. Compute Anomaly and Exception Scores
    print("Computing hybrid anomaly and exception scores...")
    anomaly_scores, exception_types, _ = compute_composite_anomaly_scores(raw_test)

    # 6. Extract Top SHAP Drivers & Actions
    print("Deriving top explainability drivers and prescribed actions...")
    base_est = models["next_12m_default_flag"]
    if hasattr(base_est, "calibrated_classifiers_"):
        base_est = base_est.calibrated_classifiers_[0].estimator

    # Sample-based SHAP background for speed
    explainer = shap.TreeExplainer(base_est)
    shap_vals = explainer.shap_values(X_test)

    top_drivers_list = []
    actions_list = []
    confidence_list = []

    for i in range(len(raw_test)):
        # Top 2 SHAP drivers
        row_shap = shap_vals[i]
        top_2_idx = np.argsort(np.abs(row_shap))[::-1][:2]
        d_str = f"{feature_cols[top_2_idx[0]]} ({row_shap[top_2_idx[0]]:+.2f}); {feature_cols[top_2_idx[1]]} ({row_shap[top_2_idx[1]]:+.2f})"
        top_drivers_list.append(d_str)

        # Action logic
        p_def = p_12m_default[i]
        anom = anomaly_scores.iloc[i]
        if anom >= 0.65 or p_def >= 0.60:
            act = "IMMEDIATE_REVIEW"
        elif p_def >= 0.35 or anom >= 0.40:
            act = "ESCALATE"
        elif p_def >= 0.15:
            act = "WATCH_LIST"
        else:
            act = "APPROVE"
        actions_list.append(act)

        # Confidence: distance from decision boundary (0.50)
        conf = float(0.50 + abs(p_def - 0.50))
        confidence_list.append(round(conf, 3))

    # 7. Construct Submission DataFrame matching schema exactly
    sub_df = pd.DataFrame({
        "loan_id": raw_test["loan_id"],
        "month_index": raw_test["month_index"],
        "prob_3m_delinquency": np.round(p_3m_delinq, 4),
        "prob_6m_delinquency": np.round(p_6m_delinq, 4),
        "prob_12m_default": np.round(p_12m_default, 4),
        "prob_12m_prepayment": np.round(p_12m_prepay, 4),
        "next_state": pred_next_state,
        "exception_type": exception_types.values,
        "anomaly_score": np.round(anomaly_scores.values, 4),
        "top_drivers": top_drivers_list,
        "action": actions_list,
        "confidence": confidence_list,
    })

    sub_df.to_csv(out_file, index=False)
    print(f"[✓] Final submission saved to {out_file} ({len(sub_df):,} rows)")
    return sub_df


if __name__ == "__main__":
    generate_submission()
