"""
Deterministic rule-based anomaly detection engine.
Executes business validation rules and detects servicer reconciliation conflicts.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from src import config
from src.data.loader import load_servicer_updates


def evaluate_business_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Evaluate 10 deterministic validation rules against loan records.

    Returns:
        rules_df: Binary indicators for each rule.
        rule_score: Normalized composite severity score in [0.0, 1.0].
    """
    rules_df = pd.DataFrame(index=df.index)

    # R001: Balance exceeds original without modification
    if "current_balance" in df.columns and "original_balance" in df.columns:
        mod_flag = df["modification_flag"] if "modification_flag" in df.columns else 0
        rules_df["R001_balance_exceeds_original"] = (
            (df["current_balance"] > df["original_balance"]) & (mod_flag == 0)
        ).fillna(False).astype(int)
    else:
        rules_df["R001_balance_exceeds_original"] = 0

    # R002: Negative balance
    if "current_balance" in df.columns:
        rules_df["R002_negative_balance"] = (df["current_balance"] < 0).fillna(False).astype(int)
    else:
        rules_df["R002_negative_balance"] = 0

    # R003: DPD vs Status mismatch
    if "days_past_due" in df.columns and "current_status" in df.columns:
        rules_df["R003_dpd_status_mismatch"] = (
            ((df["current_status"] == "Current") & (df["days_past_due"] > 0)) |
            ((df["current_status"] == "30DPD") & (df["days_past_due"] != 30)) |
            ((df["current_status"] == "60DPD") & (df["days_past_due"] != 60))
        ).fillna(False).astype(int)
    else:
        rules_df["R003_dpd_status_mismatch"] = 0

    # R004: Negative remaining term
    if "remaining_term_months" in df.columns:
        rules_df["R004_negative_remaining_term"] = (df["remaining_term_months"] < 0).fillna(False).astype(int)
    else:
        rules_df["R004_negative_remaining_term"] = 0

    # R005: Future origination
    if "origination_month" in df.columns and "reporting_month" in df.columns:
        rules_df["R005_future_origination"] = (df["origination_month"] > df["reporting_month"]).fillna(False).astype(int)
    else:
        rules_df["R005_future_origination"] = 0

    # R006: Closed/prepaid with positive balance
    if "current_status" in df.columns and "current_balance" in df.columns:
        rules_df["R006_closed_with_balance"] = (
            (df["current_status"] == "Prepaid") & (df["current_balance"] > 1000)
        ).fillna(False).astype(int)
    else:
        rules_df["R006_closed_with_balance"] = 0

    # R007: Stale record
    if "last_updated_at" in df.columns:
        rules_df["R007_stale_record"] = (df["last_updated_at"].str.slice(0, 4) < "2022").fillna(False).astype(int)
    else:
        rules_df["R007_stale_record"] = 0

    # R008: Missing documentation
    if "document_status" in df.columns:
        rules_df["R008_document_missing"] = (df["document_status"] == "Missing").fillna(False).astype(int)
    else:
        rules_df["R008_document_missing"] = 0

    # R009: Servicer conflict check
    try:
        servicer_df = load_servicer_updates()
        conflict_keys = set(servicer_df.loc[servicer_df["is_conflict"] == 1, ["loan_id", "month_index"]].itertuples(index=False, name=None))
        current_keys = list(zip(df["loan_id"], df["month_index"]))
        rules_df["R009_source_conflict"] = [1 if k in conflict_keys else 0 for k in current_keys]
    except Exception:
        rules_df["R009_source_conflict"] = 0

    # R010: Impossible transition / status reversion
    if "current_status" in df.columns:
        rules_df["R010_impossible_status"] = (
            df["current_status"].isin(["Default"]) & (df.get("days_past_due", 0) == 0)
        ).fillna(False).astype(int)
    else:
        rules_df["R010_impossible_status"] = 0

    # Severity-weighted composite rule score [0, 1]
    weights = {
        "R001_balance_exceeds_original": 0.25,
        "R002_negative_balance": 0.35,
        "R003_dpd_status_mismatch": 0.25,
        "R004_negative_remaining_term": 0.20,
        "R005_future_origination": 0.35,
        "R006_closed_with_balance": 0.20,
        "R007_stale_record": 0.15,
        "R008_document_missing": 0.10,
        "R009_source_conflict": 0.30,
        "R010_impossible_status": 0.30,
    }

    weighted_sum = sum(rules_df[col] * weights[col] for col in rules_df.columns if col in weights)
    rule_score = np.clip(weighted_sum, 0.0, 1.0).round(4)

    return rules_df, pd.Series(rule_score, index=df.index, name="rule_anomaly_score")
