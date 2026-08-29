"""
Outlier and domain-rule anomaly detection for loan performance data.
"""

import numpy as np
import pandas as pd
from src import config


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect statistical outliers (IQR, Z-score) and domain rule violations.

    Returns:
        pd.DataFrame with binary indicator flags for each anomaly check and
        summary count columns.
    """
    print("Detecting outliers and domain rule violations...")
    outliers_df = pd.DataFrame(index=df.index)

    # ──────────────────────────────────────────────
    # 1. Statistical Outliers (IQR and Z-score)
    # ──────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_clean = df[col].dropna()
        if len(col_clean) == 0:
            continue

        # IQR method
        q1 = col_clean.quantile(0.25)
        q3 = col_clean.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_df[f"{col}_iqr_outlier"] = (
                (df[col] < lower_bound) | (df[col] > upper_bound)
            ).fillna(False).astype(int)
        else:
            outliers_df[f"{col}_iqr_outlier"] = 0

        # Z-score method (|z| > 3)
        std = col_clean.std()
        if std > 0:
            z_scores = np.abs((df[col] - col_clean.mean()) / std)
            outliers_df[f"{col}_z_outlier"] = (z_scores > 3).fillna(False).astype(int)
        else:
            outliers_df[f"{col}_z_outlier"] = 0

    # ──────────────────────────────────────────────
    # 2. Domain & Business Rule Violations
    # ──────────────────────────────────────────────
    # R001: Current balance exceeds original balance (unless modification)
    if "current_balance" in df.columns and "original_balance" in df.columns:
        mod_flag = df["modification_flag"] if "modification_flag" in df.columns else 0
        outliers_df["rule_balance_gt_original"] = (
            (df["current_balance"] > df["original_balance"]) & (mod_flag == 0)
        ).fillna(False).astype(int)

    # R002: Negative balance
    if "current_balance" in df.columns:
        outliers_df["rule_negative_balance"] = (
            df["current_balance"] < 0
        ).fillna(False).astype(int)

    # R003: DPD vs Status mismatch
    if "days_past_due" in df.columns and "current_status" in df.columns:
        outliers_df["rule_dpd_status_mismatch"] = (
            ((df["current_status"] == "Current") & (df["days_past_due"] > 0)) |
            ((df["current_status"] == "30DPD") & (df["days_past_due"] != 30)) |
            ((df["current_status"] == "60DPD") & (df["days_past_due"] != 60)) |
            ((df["current_status"] == "90DPD") & (df["days_past_due"] != 90))
        ).fillna(False).astype(int)

    # R004: Negative remaining term
    if "remaining_term_months" in df.columns:
        outliers_df["rule_negative_remaining_term"] = (
            df["remaining_term_months"] < 0
        ).fillna(False).astype(int)

    # R005: Future origination (origination > reporting)
    if "origination_month" in df.columns and "reporting_month" in df.columns:
        outliers_df["rule_future_origination"] = (
            df["origination_month"] > df["reporting_month"]
        ).fillna(False).astype(int)

    # R006: Closed/prepaid with substantial positive balance
    if "current_status" in df.columns and "current_balance" in df.columns:
        outliers_df["rule_closed_with_balance"] = (
            (df["current_status"] == "Prepaid") & (df["current_balance"] > 1000)
        ).fillna(False).astype(int)

    # R007: Stale update (>6 months old)
    if "last_updated_at" in df.columns and "reporting_month" in df.columns:
        outliers_df["rule_stale_record"] = (
            df["last_updated_at"].str.slice(0, 4) < "2022"
        ).fillna(False).astype(int)

    # R008: Missing document status
    if "document_status" in df.columns:
        outliers_df["rule_document_missing"] = (
            df["document_status"] == "Missing"
        ).fillna(False).astype(int)

    # Summary columns
    rule_cols = [c for c in outliers_df.columns if c.startswith("rule_")]
    stat_cols = [c for c in outliers_df.columns if c.endswith("_outlier")]

    outliers_df["total_rule_violations"] = outliers_df[rule_cols].sum(axis=1) if rule_cols else 0
    outliers_df["total_stat_outliers"] = outliers_df[stat_cols].sum(axis=1) if stat_cols else 0
    outliers_df["total_outlier_flags"] = (
        outliers_df["total_rule_violations"] + outliers_df["total_stat_outliers"]
    )

    return outliers_df
