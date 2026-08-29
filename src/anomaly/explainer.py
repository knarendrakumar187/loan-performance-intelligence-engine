"""
Anomaly driver explainer and reviewer-ready exception generator.
Produces human-interpretable plain-English diagnostic reasoning for flagged loans.
"""

from typing import List
import pandas as pd
import numpy as np


def generate_anomaly_explanations(
    df: pd.DataFrame,
    anomaly_scores: pd.Series,
    exception_types: pd.Series,
    rules_df: pd.DataFrame,
    min_score_threshold: float = 0.50,
    top_n: int = 25,
) -> pd.DataFrame:
    """Generate plain-English diagnostic explanations and reviewer actions for flagged records."""
    flagged_mask = (anomaly_scores >= min_score_threshold) | (exception_types != "none")
    flagged_indices = df[flagged_mask].index

    # Sort by anomaly score descending
    sorted_indices = anomaly_scores.loc[flagged_indices].sort_values(ascending=False).index[:top_n]

    records = []
    for idx in sorted_indices:
        loan_id = df.loc[idx, "loan_id"] if "loan_id" in df.columns else f"LN_{idx}"
        month_idx = df.loc[idx, "month_index"] if "month_index" in df.columns else 1
        rep_month = df.loc[idx, "reporting_month"] if "reporting_month" in df.columns else "N/A"
        score = float(anomaly_scores.loc[idx])
        exc_type = str(exception_types.loc[idx])

        # 1. Identify triggered rules
        fired_rules = []
        rule_row = rules_df.loc[idx]
        for r_col in rules_df.columns:
            if rule_row[r_col] == 1:
                fired_rules.append(r_col.replace("R00", "R").replace("R0", "R"))

        # 2. Derive Plain-English Reasoning
        reasons = []
        curr_bal = df.loc[idx, "current_balance"] if "current_balance" in df.columns else 0
        orig_bal = df.loc[idx, "original_balance"] if "original_balance" in df.columns else 0
        dpd = df.loc[idx, "days_past_due"] if "days_past_due" in df.columns else 0
        status = df.loc[idx, "current_status"] if "current_status" in df.columns else "N/A"

        if "R1_balance_exceeds_original" in fired_rules or curr_bal > orig_bal:
            reasons.append(f"Outstanding balance (${curr_bal:,.2f}) exceeds original balance (${orig_bal:,.2f}) without modification flag.")

        if "R2_negative_balance" in fired_rules or curr_bal < 0:
            reasons.append(f"Negative balance detected (${curr_bal:,.2f}), indicating potential reversal or ledger accounting error.")

        if "R3_dpd_status_mismatch" in fired_rules or (status == "Current" and dpd > 0):
            reasons.append(f"Contract state '{status}' conflicts with recorded {dpd} Days Past Due.")

        if "R9_source_conflict" in fired_rules:
            reasons.append("Servicer secondary feed reports conflicting delinquency and balance state.")

        if "R7_stale_record" in fired_rules or "R8_document_missing" in fired_rules:
            reasons.append("Missing critical loan origination documentation or stale multi-quarter reporting update.")

        if not reasons:
            reasons.append(f"Multivariate outlier: feature vector significantly deviates in spatial density (Isolation Forest score {score:.2f}).")

        plain_reasoning = " ".join(reasons)

        # 3. Prescribe Reviewer Action
        if "negative" in plain_reasoning or "exceeds" in plain_reasoning:
            action = "Escalate to Servicing Accounting for balance correction"
        elif "conflicts" in plain_reasoning or "secondary" in plain_reasoning:
            action = "Request dual-source data tape reconciliation from primary servicer"
        elif "documentation" in plain_reasoning or "stale" in plain_reasoning:
            action = "Issue custodian document deficiency notice"
        else:
            action = "Queue for secondary credit underwriting audit"

        records.append({
            "loan_id": loan_id,
            "month_index": month_idx,
            "reporting_month": rep_month,
            "anomaly_score": score,
            "exception_type": exc_type,
            "triggered_rules": ", ".join(fired_rules) if fired_rules else "None (ML Outlier)",
            "plain_english_reasoning": plain_reasoning,
            "recommended_action": action,
        })

    return pd.DataFrame(records)
