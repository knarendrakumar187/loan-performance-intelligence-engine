"""
Baseline performance prediction models (Logistic Regression / Dummy).
Evaluated strictly using time-aware chronological validation.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from src import config
from src.models.splitter import time_aware_train_val_split


TARGETS_BINARY = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
]
TARGET_MULTICLASS = "next_state"


def evaluate_binary_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, target_name: str
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for binary classification."""
    # Drop NaNs if any in evaluation targets
    valid_mask = ~np.isnan(y_true)
    y_t = y_true[valid_mask].astype(int)
    y_p = y_prob[valid_mask]

    if len(np.unique(y_t)) < 2:
        return {
            "roc_auc": 0.5,
            "pr_auc": float(np.mean(y_t)),
            "f1_score": 0.0,
            "brier_score": float(np.mean((y_p - y_t) ** 2)),
            "recall_at_80_precision": 0.0,
        }

    roc_auc = roc_auc_score(y_t, y_p)
    pr_auc = average_precision_score(y_t, y_p)
    y_pred = (y_p >= 0.5).astype(int)
    f1 = f1_score(y_t, y_pred, zero_division=0)
    brier = brier_score_loss(y_t, y_p)

    # Recall at fixed precision (0.80)
    precision, recall, thresholds = precision_recall_curve(y_t, y_p)
    high_prec_mask = precision >= 0.80
    recall_at_80 = float(np.max(recall[high_prec_mask])) if np.any(high_prec_mask) else 0.0

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "f1_score": round(float(f1), 4),
        "brier_score": round(float(brier), 4),
        "recall_at_80_precision": round(recall_at_80, 4),
    }


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline_models() -> Dict[str, Any]:
    """Train baseline Logistic Regression models for all 5 targets and evaluate on time-aware validation split."""
    print("=" * 60)
    print("Training Baseline Models (Logistic Regression)")
    print("=" * 60)

    train_feat_file = config.PROCESSED_DATA_DIR / "train_features.csv"
    if not train_feat_file.exists():
        from src.features.engineer import run_feature_engineering
        run_feature_engineering()

    df = pd.read_csv(train_feat_file)
    train_split, val_split, cutoff_month = time_aware_train_val_split(df)

    non_feature_cols = set(config.TARGET_COLUMNS + ["loan_id", "reporting_month", "origination_month"])
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    X_train = train_split[feature_cols].fillna(0).values
    X_val = val_split[feature_cols].fillna(0).values

    models = {}
    metrics_summary = {}

    # 1. Train Binary Targets
    for target in TARGETS_BINARY:
        print(f"\n▶ Training Baseline for `{target}`...")
        y_train = train_split[target].values
        y_val = val_split[target].values

        train_mask = ~np.isnan(y_train)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=500, random_state=config.RANDOM_SEED)
        )
        clf.fit(X_train[train_mask], y_train[train_mask].astype(int))

        val_probs = clf.predict_proba(X_val)[:, 1]
        metrics = evaluate_binary_predictions(y_val, val_probs, target)
        metrics_summary[target] = metrics
        models[target] = clf

        print(f"  [Val Metrics] ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f} | Brier: {metrics['brier_score']:.4f}")

    # 2. Train Multiclass Target (`next_state`)
    if TARGET_MULTICLASS in df.columns:
        print(f"\n▶ Training Baseline for `{TARGET_MULTICLASS}` (Multiclass)...")
        y_train_mc = train_split[TARGET_MULTICLASS].fillna("Current").values
        y_val_mc = val_split[TARGET_MULTICLASS].fillna("Current").values

        clf_mc = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=500, random_state=config.RANDOM_SEED)
        )
        clf_mc.fit(X_train, y_train_mc)

        y_pred_mc = clf_mc.predict(X_val)
        macro_f1 = f1_score(y_val_mc, y_pred_mc, average="macro", zero_division=0)
        metrics_summary[TARGET_MULTICLASS] = {
            "macro_f1": round(float(macro_f1), 4),
            "classes": list(clf_mc.classes_),
        }
        models[TARGET_MULTICLASS] = clf_mc
        print(f"  [Val Metrics] Macro-F1: {macro_f1:.4f}")

    # Save baseline models
    model_dir = config.PROJECT_ROOT / "checkpoints" / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_dir / "baseline_models.joblib")
    print(f"\n[✓] Baseline models saved to {model_dir}")

    return metrics_summary


if __name__ == "__main__":
    train_baseline_models()
