"""
Machine learning anomaly scorer using Isolation Forest and supervised Exception Classifier.
Combines deterministic rule signals with unsupervised spatial anomaly detection.
"""

from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from src import config
from src.anomaly.rules import evaluate_business_rules


def train_anomaly_models(train_df: pd.DataFrame) -> Tuple[IsolationForest, RandomForestClassifier]:
    """Train unsupervised Isolation Forest and supervised Exception Type Classifier."""
    print("Training Hybrid Anomaly Detection Models (Isolation Forest + RF Classifier)...")

    # Select numerical modeling features
    feat_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if not c.startswith("next_") and not c.startswith("prob_") and c not in ("exception_required",)
    ]
    X_train = train_df[feat_cols].fillna(0).values

    # 1. Unsupervised Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    iso_forest.fit(X_train)

    # 2. Supervised Exception Type Classifier
    y_exception = train_df.get("exception_type", pd.Series("none", index=train_df.index)).fillna("none").values
    clf_exception = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    clf_exception.fit(X_train, y_exception)

    # Save models
    save_dir = config.PROJECT_ROOT / "checkpoints" / "anomaly"
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso_forest, save_dir / "isolation_forest.joblib")
    joblib.dump(clf_exception, save_dir / "exception_classifier.joblib")
    joblib.dump(feat_cols, save_dir / "anomaly_features.joblib")
    print(f"  [✓] Anomaly models saved to {save_dir}")

    return iso_forest, clf_exception


def compute_composite_anomaly_scores(
    df: pd.DataFrame,
    iso_forest: IsolationForest = None,
    clf_exception: RandomForestClassifier = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Compute hybrid anomaly score (0 to 1) and predict exception_type for each record."""
    save_dir = config.PROJECT_ROOT / "checkpoints" / "anomaly"
    if iso_forest is None or clf_exception is None:
        iso_forest = joblib.load(save_dir / "isolation_forest.joblib")
        clf_exception = joblib.load(save_dir / "exception_classifier.joblib")
        feat_cols = joblib.load(save_dir / "anomaly_features.joblib")
    else:
        feat_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if not c.startswith("next_") and not c.startswith("prob_") and c not in ("exception_required",)
        ]

    # 1. Rule-based scores
    rules_df, rule_scores = evaluate_business_rules(df)

    # 2. ML Isolation Forest score (normalized to [0, 1])
    df_eval = df.copy()
    for col in feat_cols:
        if col not in df_eval.columns:
            df_eval[col] = 0.0
    X = df_eval[feat_cols].fillna(0).values
    # Invert decision function: lower score = more anomalous
    raw_decision = iso_forest.decision_function(X)
    ml_scores = 1.0 - (raw_decision - raw_decision.min()) / (raw_decision.max() - raw_decision.min() + 1e-8)
    ml_scores = np.clip(ml_scores, 0.0, 1.0).round(4)

    # 3. Hybrid Composite Score
    composite_scores = (0.45 * rule_scores.values + 0.55 * ml_scores).round(4)

    # 4. Predict Exception Type
    pred_exception_types = clf_exception.predict(X)

    # If rules strongly fired, override with rule exception type
    rule_fired_mask = rule_scores.values >= 0.20
    rule_exceptions = []
    for idx, row in rules_df.iterrows():
        if row.get("R001_balance_exceeds_original", 0) or row.get("R002_negative_balance", 0) or row.get("R004_negative_remaining_term", 0):
            rule_exceptions.append("data_entry_error")
        elif row.get("R009_source_conflict", 0):
            rule_exceptions.append("source_conflict")
        elif row.get("R007_stale_record", 0) or row.get("R008_document_missing", 0):
            rule_exceptions.append("stale_record")
        elif row.get("R003_dpd_status_mismatch", 0) or row.get("R010_impossible_status", 0):
            rule_exceptions.append("suspicious_transition")
        else:
            rule_exceptions.append(None)

    final_exception_types = []
    for i in range(len(df)):
        if rule_exceptions[i] is not None:
            final_exception_types.append(rule_exceptions[i])
        elif composite_scores[i] >= 0.65 and pred_exception_types[i] != "none":
            final_exception_types.append(pred_exception_types[i])
        elif composite_scores[i] >= 0.70:
            final_exception_types.append("suspicious_transition")
        else:
            final_exception_types.append("none")

    return (
        pd.Series(composite_scores, index=df.index, name="anomaly_score"),
        pd.Series(final_exception_types, index=df.index, name="exception_type"),
        rules_df,
    )
