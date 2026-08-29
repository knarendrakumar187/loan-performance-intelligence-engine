"""
Record-level and batch-level data quality scoring engine.
"""

import numpy as np
import pandas as pd
from src import config


def compute_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute record-level data quality scores across Completeness, Validity, and Consistency.

    Scoring Methodology:
    - Completeness (0-100): Proportion of non-null required attributes.
    - Validity (0-100): Penalties for negative balances, negative terms, out-of-range rates.
    - Consistency (0-100): Penalties for DPD/status mismatch, balance > original, date breaks.
    - Overall Quality Score (0-100): 0.35 * Completeness + 0.35 * Validity + 0.30 * Consistency.

    Returns:
        pd.DataFrame with completeness_score, validity_score, consistency_score,
        overall_quality_score, and quality_grade (A/B/C/D/F).
    """
    print("Computing record-level and batch-level quality scores...")
    scores = pd.DataFrame(index=df.index)

    # 1. Completeness Score (0-100)
    check_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    completeness = df[check_cols].notna().mean(axis=1) * 100.0
    scores["completeness_score"] = completeness.round(2)

    # 2. Validity Score (0-100)
    validity = pd.Series(100.0, index=df.index)
    if "current_balance" in df.columns:
        validity[df["current_balance"] < 0] -= 40.0
    if "remaining_term_months" in df.columns:
        validity[df["remaining_term_months"] < 0] -= 30.0
    if "interest_rate" in df.columns:
        validity[(df["interest_rate"] < 0) | (df["interest_rate"] > 25.0)] -= 30.0
    if "days_past_due" in df.columns:
        validity[df["days_past_due"] < 0] -= 40.0

    scores["validity_score"] = np.clip(validity, 0.0, 100.0).round(2)

    # 3. Consistency Score (0-100)
    consistency = pd.Series(100.0, index=df.index)
    if "current_balance" in df.columns and "original_balance" in df.columns:
        mod_flag = df["modification_flag"] if "modification_flag" in df.columns else 0
        consistency[(df["current_balance"] > df["original_balance"]) & (mod_flag == 0)] -= 35.0

    if "current_status" in df.columns and "days_past_due" in df.columns:
        dpd_mismatch = (
            ((df["current_status"] == "Current") & (df["days_past_due"] > 0)) |
            ((df["current_status"] == "30DPD") & (df["days_past_due"] != 30))
        )
        consistency[dpd_mismatch] -= 30.0

    if "origination_month" in df.columns and "reporting_month" in df.columns:
        consistency[df["origination_month"] > df["reporting_month"]] -= 50.0

    if "current_status" in df.columns and "current_balance" in df.columns:
        consistency[(df["current_status"] == "Prepaid") & (df["current_balance"] > 1000)] -= 30.0

    scores["consistency_score"] = np.clip(consistency, 0.0, 100.0).round(2)

    # 4. Overall Composite Quality Score (0-100)
    scores["overall_quality_score"] = (
        0.35 * scores["completeness_score"] +
        0.35 * scores["validity_score"] +
        0.30 * scores["consistency_score"]
    ).round(2)

    # 5. Quality Grade
    conditions = [
        scores["overall_quality_score"] >= 90.0,
        scores["overall_quality_score"] >= 80.0,
        scores["overall_quality_score"] >= 70.0,
        scores["overall_quality_score"] >= 60.0,
    ]
    choices = ["A", "B", "C", "D"]
    scores["quality_grade"] = np.select(conditions, choices, default="F")

    return scores


def get_batch_quality_summary(scores_df: pd.DataFrame) -> dict:
    """Generate batch-level aggregated data quality metrics."""
    return {
        "mean_quality_score": round(float(scores_df["overall_quality_score"].mean()), 2),
        "median_quality_score": round(float(scores_df["overall_quality_score"].median()), 2),
        "mean_completeness": round(float(scores_df["completeness_score"].mean()), 2),
        "mean_validity": round(float(scores_df["validity_score"].mean()), 2),
        "mean_consistency": round(float(scores_df["consistency_score"].mean()), 2),
        "grade_distribution": scores_df["quality_grade"].value_counts(normalize=True).round(4).to_dict(),
        "clean_records_pct": round(float((scores_df["overall_quality_score"] >= 95.0).mean() * 100), 2),
    }
