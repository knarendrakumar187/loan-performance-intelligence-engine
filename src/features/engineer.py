"""
Leakage-safe feature engineering pipeline for loan performance prediction.
Transforms raw loan panel datasets into modeling-ready numeric feature sets.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from src import config
from src.data.loader import load_test, load_train


# Ordinal risk tier mappings for loan attributes
CREDIT_TIER_MAP = {
    "<620": 6, "620-659": 5, "660-699": 4, "700-739": 3, "740-779": 2, "780+": 1, "Missing": 4
}
LTV_TIER_MAP = {
    "<60": 1, "60-70": 2, "70-80": 3, "80-90": 4, "90-95": 5, ">95": 6, "Missing": 3
}
DTI_TIER_MAP = {
    "<20": 1, "20-30": 2, "30-40": 3, "40-50": 4, ">50": 5, "Missing": 3
}


def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Build domain features, encode categoricals, impute missing values, and remove leakage fields."""
    df = df.copy()

    # 1. Ratios and Financial Engineering
    df["balance_to_orig_ratio"] = (df["current_balance"] / (df["original_balance"] + 1e-5)).clip(0, 3.0)
    df["seasoning_pct"] = (df["loan_age_months"] / (df["loan_age_months"] + df["remaining_term_months"] + 1e-5)).clip(0, 1.0)
    df["monthly_interest_accrual"] = df["current_balance"] * (df["interest_rate"] / 100.0 / 12.0)

    # 2. Risk Tier Encodings
    for col in ["credit_score_band", "ltv_band", "dti_band"]:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")

    df["credit_risk_tier"] = df["credit_score_band"].map(CREDIT_TIER_MAP).fillna(4).astype(int)
    df["ltv_risk_tier"] = df["ltv_band"].map(LTV_TIER_MAP).fillna(3).astype(int)
    df["dti_risk_tier"] = df["dti_band"].map(DTI_TIER_MAP).fillna(3).astype(int)
    df["combined_risk_index"] = df["ltv_risk_tier"] + df["dti_risk_tier"] + df["credit_risk_tier"]
    df = df.drop(columns=["credit_score_band", "ltv_band", "dti_band"], errors="ignore")

    # 3. Delinquency Indicators
    if "days_past_due" in df.columns:
        df["is_currently_delinquent"] = (df["days_past_due"] > 0).astype(int)
        df["is_severely_delinquent"] = (df["days_past_due"] >= 60).astype(int)
        df["dpd_to_age_ratio"] = df["days_past_due"] / (df["loan_age_months"] * 30 + 1e-5)
    else:
        df["is_currently_delinquent"] = 0
        df["is_severely_delinquent"] = 0
        df["dpd_to_age_ratio"] = 0.0

    if "modification_flag" in df.columns:
        df["is_modified"] = df["modification_flag"].fillna(0).astype(int)
    else:
        df["is_modified"] = 0

    # 4. Categorical One-Hot Encoding
    cat_cols = ["state", "loan_purpose", "occupancy_type", "property_type", "servicer_name", "current_status"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Missing").astype(str)

    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=True, dtype=int)

    # 5. Missing Value Imputation (Median + Indicator for Numerics)
    numeric_impute_cols = ["current_balance", "interest_rate", "original_balance", "loan_age_months", "remaining_term_months"]
    for col in numeric_impute_cols:
        if col in df.columns:
            if df[col].isnull().any():
                df[f"{col}_is_missing"] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(df[col].median())

    # 6. Explicit Drop of Leakage and Non-Feature Metadata Columns
    drop_cols = [c for c in config.LEAKAGE_DROP_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def run_feature_engineering() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute feature engineering pipeline on raw datasets and save processed outputs."""
    print("=" * 60)
    print("Running Feature Engineering Pipeline")
    print("=" * 60)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = load_train()
    test_raw = load_test()

    print(f"Loaded raw train: {train_raw.shape} | raw test: {test_raw.shape}")

    train_proc = engineer_features(train_raw, is_train=True)
    test_proc = engineer_features(test_raw, is_train=False)

    # Align columns between train and test (excluding target columns)
    target_and_meta = [c for c in config.TARGET_COLUMNS if c in train_proc.columns]
    feat_cols_train = [c for c in train_proc.columns if c not in target_and_meta and c not in ("loan_id", "reporting_month", "origination_month")]

    # Ensure test has all feature columns with 0 fill for any missing dummies
    for col in feat_cols_train:
        if col not in test_proc.columns:
            test_proc[col] = 0

    train_out = config.PROCESSED_DATA_DIR / "train_features.csv"
    test_out = config.PROCESSED_DATA_DIR / "test_features.csv"

    train_proc.to_csv(train_out, index=False)
    test_proc.to_csv(test_out, index=False)

    print(f"[✓] Engineered train features saved to {train_out} ({train_proc.shape})")
    print(f"[✓] Engineered test features saved to {test_out} ({test_proc.shape})")
    print(f"[✓] Total active predictive features: {len(feat_cols_train)}")

    return train_proc, test_proc


if __name__ == "__main__":
    run_feature_engineering()
