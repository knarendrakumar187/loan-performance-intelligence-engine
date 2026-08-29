"""
Improved predictive performance models using tuned XGBoost and LightGBM with
class imbalance mitigation and probability calibration (Platt scaling & Isotonic).
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
import xgboost as xgb
import lightgbm as lgb

from src import config
from src.models.baseline import TARGET_MULTICLASS, TARGETS_BINARY, evaluate_binary_predictions
from src.models.splitter import time_aware_train_val_split

plt.switch_backend("Agg")


def train_improved_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train improved Gradient Boosted models (XGBoost/LightGBM) with class imbalance weighting and probability calibration."""
    print("=" * 60)
    print("Training Improved Gradient Boosted Models (XGBoost / LightGBM)")
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1. Train Binary Targets with XGBoost + Class Weighting + Calibration
    for i, target in enumerate(TARGETS_BINARY):
        print(f"\n▶ Training Improved Model for `{target}`...")
        y_train = train_split[target].values
        y_val = val_split[target].values

        train_mask = ~np.isnan(y_train)
        y_tr_clean = y_train[train_mask].astype(int)
        X_tr_clean = X_train[train_mask]

        val_mask = ~np.isnan(y_val)
        y_val_clean = y_val[val_mask].astype(int)
        X_val_clean = X_val[val_mask]

        # Calculate scale_pos_weight
        pos_count = np.sum(y_tr_clean == 1)
        neg_count = np.sum(y_tr_clean == 0)
        scale_weight = float(neg_count / max(1, pos_count))
        print(f"  Class balance: Positive = {pos_count:,} ({pos_count/len(y_tr_clean)*100:.2f}%) | scale_pos_weight = {scale_weight:.2f}")

        # Base XGBoost model
        base_xgb = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_weight,
            random_state=config.RANDOM_SEED,
            eval_metric="logloss",
            n_jobs=-1,
        )
        base_xgb.fit(X_tr_clean, y_tr_clean)

        # Probability Calibration via Platt Scaling (Sigmoid)
        calibrated_clf = CalibratedClassifierCV(estimator=base_xgb, method="sigmoid", cv=3)
        calibrated_clf.fit(X_tr_clean, y_tr_clean)

        val_probs = calibrated_clf.predict_proba(X_val_clean)[:, 1]
        metrics = evaluate_binary_predictions(y_val_clean, val_probs, target)
        metrics_summary[target] = metrics
        models[target] = calibrated_clf

        print(f"  [Improved Val Metrics] ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f} | Brier: {metrics['brier_score']:.4f}")

        # Plot Reliability Curve
        prob_true, prob_pred = calibration_curve(y_val_clean, val_probs, n_bins=10)
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(prob_pred, prob_true, marker="o", color="#1f77b4", label="Calibrated XGBoost")
        ax.set_title(f"Reliability Curve: {target}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()

    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "calibration_reliability_curves.png", dpi=150)
    plt.close()

    # 2. Train Multiclass Target (`next_state`) via LightGBM
    if TARGET_MULTICLASS in df.columns:
        print(f"\n▶ Training Improved Multiclass Model for `{TARGET_MULTICLASS}`...")
        y_train_mc = train_split[TARGET_MULTICLASS].fillna("Current").values
        y_val_mc = val_split[TARGET_MULTICLASS].fillna("Current").values

        lgb_mc = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_mc.fit(X_train, y_train_mc)

        y_pred_mc = lgb_mc.predict(X_val)
        macro_f1 = f1_score(y_val_mc, y_pred_mc, average="macro", zero_division=0)
        metrics_summary[TARGET_MULTICLASS] = {
            "macro_f1": round(float(macro_f1), 4),
            "classes": list(lgb_mc.classes_),
        }
        models[TARGET_MULTICLASS] = lgb_mc
        print(f"  [Improved Val Metrics] Multiclass Macro-F1: {macro_f1:.4f}")

    # Save improved models
    model_dir = config.PROJECT_ROOT / "checkpoints" / "improved"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, model_dir / "improved_models.joblib")
    joblib.dump(feature_cols, model_dir / "feature_columns.joblib")
    print(f"\n[✓] Improved models saved to {model_dir}")

    return models, metrics_summary


if __name__ == "__main__":
    train_improved_models()
