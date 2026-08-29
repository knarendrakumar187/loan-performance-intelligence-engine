"""
Synthetic Data Generator for the Loan Performance Intelligence Engine.

Generates development-scale datasets matching the exact schemas from the
Intain Campus FinTech Challenge 2026 problem statement (Section 7).

The generator produces realistic loan-level monthly performance data with:
- Correlated features (e.g., higher LTV → higher default risk)
- Realistic state transitions (Current → 30DPD → 60DPD → ... → Default/Prepaid)
- Deliberately injected data quality issues for anomaly detection testing
- Forward-looking target variables computed without leakage

Usage:
    python -m src.data.synthesize
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import config


# ──────────────────────────────────────────────
# Transition probabilities for loan states
# ──────────────────────────────────────────────
# Rows = current state, Cols = next state
# Order: Current, 30DPD, 60DPD, 90DPD, Default, Prepaid
BASE_TRANSITION_MATRIX = np.array([
    # Current  30DPD   60DPD   90DPD   Default Prepaid
    [0.88,     0.05,   0.00,   0.00,   0.00,   0.07],   # Current
    [0.30,     0.40,   0.20,   0.02,   0.03,   0.05],   # 30DPD
    [0.10,     0.15,   0.35,   0.25,   0.10,   0.05],   # 60DPD
    [0.05,     0.05,   0.10,   0.35,   0.40,   0.05],   # 90DPD
    [0.00,     0.00,   0.00,   0.00,   1.00,   0.00],   # Default (absorbing)
    [0.00,     0.00,   0.00,   0.00,   0.00,   1.00],   # Prepaid (absorbing)
])

STATUS_INDEX = {s: i for i, s in enumerate(config.LOAN_STATUSES)}
DPD_MAP = {"Current": 0, "30DPD": 30, "60DPD": 60, "90DPD": 90,
           "Default": 120, "Prepaid": 0}


def _risk_adjusted_transitions(credit_band: str, ltv_band: str) -> np.ndarray:
    """Adjust transition matrix based on credit quality and LTV.

    Higher risk (lower credit, higher LTV) → more transitions to delinquency.
    """
    mat = BASE_TRANSITION_MATRIX.copy()

    # Credit score adjustment
    credit_risk = {
        "<620": 0.06, "620-659": 0.03, "660-699": 0.01,
        "700-739": 0.0, "740-779": -0.01, "780+": -0.02,
    }
    risk_shift = credit_risk.get(credit_band, 0.0)

    # LTV adjustment
    ltv_risk = {
        "<60": -0.02, "60-70": -0.01, "70-80": 0.0,
        "80-90": 0.01, "90-95": 0.03, ">95": 0.05,
    }
    risk_shift += ltv_risk.get(ltv_band, 0.0)

    # Apply: increase probability of worsening, decrease staying/improving
    for i in range(4):  # Only non-absorbing states
        if risk_shift > 0:
            # Increase transitions to worse states
            worse_cols = list(range(i + 1, 6))
            if worse_cols:
                mat[i, worse_cols] += risk_shift / len(worse_cols)
                mat[i, i] -= risk_shift
        else:
            # Decrease transitions to worse states (negative shift = improvement)
            better_cols = list(range(0, i + 1))
            mat[i, better_cols] -= risk_shift / len(better_cols)
            worse_cols = list(range(i + 1, 6))
            if worse_cols:
                mat[i, worse_cols] += risk_shift / len(worse_cols)

    # Ensure valid probabilities
    mat = np.clip(mat, 0.001, 1.0)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def _generate_loan_static(rng: np.random.Generator, n_loans: int) -> pd.DataFrame:
    """Generate origination-level static attributes for each loan."""

    # Credit score distribution (skewed toward prime)
    credit_weights = [0.08, 0.12, 0.20, 0.25, 0.20, 0.15]
    ltv_weights = [0.10, 0.15, 0.25, 0.25, 0.15, 0.10]
    dti_weights = [0.15, 0.30, 0.30, 0.15, 0.10]

    records = []
    for i in range(n_loans):
        loan_id = f"LN{i:06d}"
        orig_month_offset = rng.integers(0, 24)  # Origination spread over 24 months (2022-01 to 2023-12)
        orig_year = 2022 + orig_month_offset // 12
        orig_m = (orig_month_offset % 12) + 1
        origination_month = f"{orig_year}-{orig_m:02d}"

        credit_band = rng.choice(config.CREDIT_SCORE_BANDS, p=credit_weights)
        ltv_band = rng.choice(config.LTV_BANDS, p=ltv_weights)
        dti_band = rng.choice(config.DTI_BANDS, p=dti_weights)

        # Correlated: lower credit → higher interest rate
        credit_idx = config.CREDIT_SCORE_BANDS.index(credit_band)
        base_rate = 6.5 - credit_idx * 0.4 + rng.normal(0, 0.3)
        interest_rate = round(max(3.0, min(10.0, base_rate)), 3)

        # Original balance: log-normal distribution
        original_balance = int(rng.lognormal(mean=12.2, sigma=0.5))
        original_balance = max(50_000, min(800_000, original_balance))

        records.append({
            "loan_id": loan_id,
            "origination_month": origination_month,
            "original_balance": original_balance,
            "interest_rate": interest_rate,
            "credit_score_band": credit_band,
            "ltv_band": ltv_band,
            "dti_band": dti_band,
            "state": rng.choice(config.STATES),
            "loan_purpose": rng.choice(config.LOAN_PURPOSES, p=[0.45, 0.30, 0.25]),
            "occupancy_type": rng.choice(config.OCCUPANCY_TYPES, p=[0.75, 0.10, 0.15]),
            "property_type": rng.choice(config.PROPERTY_TYPES, p=[0.55, 0.20, 0.15, 0.10]),
            "servicer_name": rng.choice(config.SERVICER_NAMES),
            "loan_term_months": rng.choice([180, 240, 360], p=[0.15, 0.15, 0.70]),
        })

    return pd.DataFrame(records)


def _simulate_loan_history(
    rng: np.random.Generator,
    loan: pd.Series,
    n_months: int,
) -> list[dict]:
    """Simulate monthly performance history for a single loan."""

    trans_mat = _risk_adjusted_transitions(loan["credit_score_band"], loan["ltv_band"])
    current_status_idx = 0  # Start as Current
    current_balance = loan["original_balance"]
    monthly_payment = current_balance * (loan["interest_rate"] / 100 / 12)

    records = []
    orig_year, orig_month_num = map(int, loan["origination_month"].split("-"))

    for m in range(n_months):
        # Calculate reporting month
        total_months = orig_month_num - 1 + m
        year = orig_year + total_months // 12
        month_num = (total_months % 12) + 1
        reporting_month = f"{year}-{month_num:02d}"

        status = config.LOAN_STATUSES[current_status_idx]
        dpd = DPD_MAP[status]

        # Balance amortization (simplified)
        if status not in ("Default", "Prepaid"):
            principal_payment = monthly_payment * 0.3  # Simplified
            current_balance = max(0, current_balance - principal_payment)

        # Determine modification flag (small chance for delinquent loans)
        modification_flag = 0
        if status in ("30DPD", "60DPD", "90DPD") and rng.random() < 0.05:
            modification_flag = 1

        # Loss severity (only for defaults)
        loss_severity = "None"
        if status == "Default":
            loss_severity = rng.choice(
                config.LOSS_SEVERITY_BANDS[1:], p=[0.4, 0.4, 0.2]
            )

        # Source system and document status
        source_system = rng.choice(config.SOURCE_SYSTEMS, p=[0.7, 0.3])
        doc_status = rng.choice(
            config.DOCUMENT_STATUSES,
            p=[0.70, 0.15, 0.10, 0.05]
        )

        record = {
            "loan_id": loan["loan_id"],
            "month_index": m + 1,
            "reporting_month": reporting_month,
            "origination_month": loan["origination_month"],
            "loan_age_months": m + 1,
            "remaining_term_months": max(0, loan["loan_term_months"] - m - 1),
            "original_balance": loan["original_balance"],
            "current_balance": round(current_balance, 2),
            "interest_rate": loan["interest_rate"],
            "credit_score_band": loan["credit_score_band"],
            "ltv_band": loan["ltv_band"],
            "dti_band": loan["dti_band"],
            "state": loan["state"],
            "loan_purpose": loan["loan_purpose"],
            "occupancy_type": loan["occupancy_type"],
            "property_type": loan["property_type"],
            "servicer_name": loan["servicer_name"],
            "current_status": status,
            "days_past_due": dpd,
            "modification_flag": modification_flag,
            "prepayment_flag": 1 if status == "Prepaid" else 0,
            "default_flag": 1 if status == "Default" else 0,
            "loss_severity_band": loss_severity,
            "last_updated_at": f"{reporting_month}-15",
            "source_system": source_system,
            "document_status": doc_status,
        }
        records.append(record)

        # Absorbing states — stop generating
        if status in ("Default", "Prepaid"):
            break

        # Transition to next state
        current_status_idx = rng.choice(6, p=trans_mat[current_status_idx])

    # Compute forward-looking targets directly on records (leakage-free by construction)
    n_rec = len(records)
    for i in range(n_rec):
        future_statuses = [records[j]["current_status"] for j in range(i + 1, min(i + 13, n_rec))]
        
        # next_3m_delinquency_flag
        f3 = future_statuses[:3]
        records[i]["next_3m_delinquency_flag"] = (
            int(any(s in ("30DPD", "60DPD", "90DPD", "Default") for s in f3))
            if len(f3) > 0 else np.nan
        )

        # next_6m_delinquency_flag
        f6 = future_statuses[:6]
        records[i]["next_6m_delinquency_flag"] = (
            int(any(s in ("30DPD", "60DPD", "90DPD", "Default") for s in f6))
            if len(f6) > 0 else np.nan
        )

        # next_12m_default_flag
        f12 = future_statuses[:12]
        records[i]["next_12m_default_flag"] = (
            int(any(s == "Default" for s in f12))
            if len(f12) > 0 else np.nan
        )

        # next_12m_prepayment_flag
        records[i]["next_12m_prepayment_flag"] = (
            int(any(s == "Prepaid" for s in f12))
            if len(f12) > 0 else np.nan
        )

        # next_state
        records[i]["next_state"] = records[i + 1]["current_status"] if i + 1 < n_rec else None

        # exception flags (default)
        records[i]["exception_required"] = 0
        records[i]["exception_type"] = "none"

    return records


def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through since targets are computed during simulation."""
    return df


def _inject_data_quality_issues(
    df: pd.DataFrame, rng: np.random.Generator, rate: float = 0.05
) -> pd.DataFrame:
    """Inject deliberate data quality issues for anomaly detection testing."""
    n = len(df)
    n_issues = int(n * rate)

    # 1. Missing values in selected columns
    missing_cols = ["credit_score_band", "ltv_band", "dti_band", "interest_rate", "current_balance"]
    for col in missing_cols:
        missing_idx = rng.choice(n, size=int(n * rate * 0.5), replace=False)
        df.iloc[missing_idx, df.columns.get_loc(col)] = np.nan

    # 2. Outlier balances (balance > original or negative)
    outlier_idx = rng.choice(n, size=n_issues // 3, replace=False)
    split_pt = len(outlier_idx) // 2
    idx_high = outlier_idx[:split_pt]
    idx_neg = outlier_idx[split_pt:]

    orig_bal = df["original_balance"].iloc[idx_high].values
    curr_bal_col = df.columns.get_loc("current_balance")
    df.iloc[idx_high, curr_bal_col] = orig_bal * rng.uniform(1.1, 2.5, size=len(idx_high))

    curr_bal = df["current_balance"].iloc[idx_neg].values
    df.iloc[idx_neg, curr_bal_col] = -np.abs(curr_bal)

    df.iloc[outlier_idx, df.columns.get_loc("exception_required")] = 1
    df.iloc[outlier_idx, df.columns.get_loc("exception_type")] = "data_entry_error"

    # 3. Status/DPD inconsistency (e.g., Current with DPD=60)
    current_mask = np.where(df["current_status"].values == "Current")[0]
    if len(current_mask) > 0:
        inconsist_size = min(len(current_mask), n_issues // 4)
        inconsist_idx = rng.choice(current_mask, size=inconsist_size, replace=False)
        dpd_col = df.columns.get_loc("days_past_due")
        df.iloc[inconsist_idx, dpd_col] = rng.choice([30, 60, 90], size=inconsist_size)
        df.iloc[inconsist_idx, df.columns.get_loc("exception_required")] = 1
        df.iloc[inconsist_idx, df.columns.get_loc("exception_type")] = "suspicious_transition"

    # 4. Stale records
    stale_idx = rng.choice(n, size=n_issues // 5, replace=False)
    df.iloc[stale_idx, df.columns.get_loc("last_updated_at")] = "2020-01-15"
    df.iloc[stale_idx, df.columns.get_loc("document_status")] = "Stale"
    df.iloc[stale_idx, df.columns.get_loc("exception_required")] = 1
    df.iloc[stale_idx, df.columns.get_loc("exception_type")] = "stale_record"

    return df


def _generate_servicer_updates(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate partial/conflicting servicer updates for source-conflict detection."""
    n_updates = min(10_000, len(df) // 5)
    sample_idx = rng.choice(len(df), size=n_updates, replace=False)
    updates = df.iloc[sample_idx][
        ["loan_id", "month_index", "current_balance", "current_status",
         "days_past_due", "servicer_name"]
    ].copy().reset_index(drop=True)

    n_conflicts = int(n_updates * 0.20)
    conflict_idx = rng.choice(n_updates, size=n_conflicts, replace=False)

    # Balance conflicts
    c_bal = conflict_idx[:n_conflicts // 3]
    bal_vals = updates["current_balance"].iloc[c_bal].values
    updates.iloc[c_bal, updates.columns.get_loc("current_balance")] = bal_vals * rng.uniform(0.8, 1.2, size=len(c_bal))

    # Status conflicts
    c_stat = conflict_idx[n_conflicts // 3: 2 * n_conflicts // 3]
    updates.iloc[c_stat, updates.columns.get_loc("current_status")] = rng.choice(config.LOAN_STATUSES[:4], size=len(c_stat))

    # DPD conflicts
    c_dpd = conflict_idx[2 * n_conflicts // 3:]
    updates.iloc[c_dpd, updates.columns.get_loc("days_past_due")] = rng.choice([0, 30, 60, 90], size=len(c_dpd))

    updates["source_system"] = "SystemB"
    updates["update_timestamp"] = "2024-06-15"
    updates["is_conflict"] = 0
    updates.iloc[conflict_idx, updates.columns.get_loc("is_conflict")] = 1

    return updates


def _generate_macro_scenarios() -> pd.DataFrame:
    """Generate macro scenario assumptions (base, adverse, high-prepay)."""
    scenarios = []
    for name, params in config.SCENARIOS.items():
        scenarios.append({
            "scenario_name": name,
            "description": params["description"],
            "credit_score_shift": params["credit_score_shift"],
            "interest_rate_shift": params["interest_rate_shift"],
            "default_rate_multiplier": params["default_rate_multiplier"],
            "prepayment_rate_multiplier": params["prepayment_rate_multiplier"],
        })
    return pd.DataFrame(scenarios)


def _generate_data_dictionary() -> str:
    """Generate plain-English data dictionary for LLM grounding."""
    return """# Data Dictionary — Loan Performance Intelligence Engine

## Monthly Performance Fields

| Field | Type | Description |
|-------|------|-------------|
| loan_id | string | Unique identifier for each loan (format: LN######) |
| month_index | int | Sequential month number within loan's observation window (1-based) |
| reporting_month | string | Calendar month of observation (YYYY-MM format) |
| origination_month | string | Month when the loan was originated (YYYY-MM format) |
| loan_age_months | int | Number of months since origination |
| remaining_term_months | int | Remaining months until maturity |
| original_balance | float | Loan balance at origination (USD) |
| current_balance | float | Outstanding principal balance as of reporting month (USD) |
| interest_rate | float | Current annual interest rate (%) |
| credit_score_band | string | Borrower credit score range at origination (<620, 620-659, 660-699, 700-739, 740-779, 780+) |
| ltv_band | string | Loan-to-value ratio range at origination (<60, 60-70, 70-80, 80-90, 90-95, >95) |
| dti_band | string | Debt-to-income ratio range (<20, 20-30, 30-40, 40-50, >50) |
| state | string | US state where the property is located (2-letter code) |
| loan_purpose | string | Purpose of the loan (Purchase, Refinance_Cashout, Refinance_NoCashout) |
| occupancy_type | string | How the property is occupied (Primary, Second_Home, Investment) |
| property_type | string | Type of property (Single_Family, Condo, Townhouse, Multi_Family) |
| servicer_name | string | Current loan servicer (ServicerA through ServicerE) |
| current_status | string | Loan performance status (Current, 30DPD, 60DPD, 90DPD, Default, Prepaid) |
| days_past_due | int | Number of days the loan payment is overdue |
| modification_flag | int | Whether the loan has been modified (0=No, 1=Yes) |
| prepayment_flag | int | Whether the loan has been prepaid in full (0=No, 1=Yes) |
| default_flag | int | Whether the loan has defaulted (0=No, 1=Yes) |
| loss_severity_band | string | Loss severity category if defaulted (None, Low, Medium, High) |
| last_updated_at | string | Date when this record was last updated |
| source_system | string | System from which this data was sourced (SystemA, SystemB) |
| document_status | string | Status of loan documentation (Complete, Partial, Missing, Stale) |

## Target Variables

| Field | Type | Description |
|-------|------|-------------|
| next_3m_delinquency_flag | int | Will the loan become delinquent (30+ DPD) in the next 3 months? (0/1) |
| next_6m_delinquency_flag | int | Will the loan become delinquent (30+ DPD) in the next 6 months? (0/1) |
| next_12m_default_flag | int | Will the loan default in the next 12 months? (0/1) |
| next_12m_prepayment_flag | int | Will the loan prepay in full in the next 12 months? (0/1) |
| next_state | string | What will the loan status be next month? (Current/30DPD/60DPD/90DPD/Default/Prepaid) |
| exception_required | int | Does this record require exception review? (0/1) |
| exception_type | string | Type of exception if flagged (none, data_entry_error, stale_record, source_conflict, suspicious_transition) |

## Servicer Updates Fields

| Field | Type | Description |
|-------|------|-------------|
| loan_id | string | Loan identifier matching the main performance file |
| month_index | int | Month index matching the main performance file |
| current_balance | float | Balance reported by servicer (may differ from main file) |
| current_status | string | Status reported by servicer (may differ from main file) |
| days_past_due | int | DPD reported by servicer (may differ from main file) |
| servicer_name | string | Reporting servicer |
| source_system | string | Always SystemB for servicer updates |
| update_timestamp | string | When the servicer submitted this update |
| is_conflict | int | Whether this record conflicts with the main performance file (0/1) |

## Macro Scenario Fields

| Field | Type | Description |
|-------|------|-------------|
| scenario_name | string | Scenario identifier (base, adverse_credit, high_prepayment) |
| description | string | Plain-English description of the scenario |
| credit_score_shift | int | Number of credit bands to shift (negative = downgrade) |
| interest_rate_shift | float | Basis point shift to interest rates |
| default_rate_multiplier | float | Multiplier applied to baseline default probability |
| prepayment_rate_multiplier | float | Multiplier applied to baseline prepayment probability |
"""


def _generate_validation_rules() -> dict:
    """Generate deterministic validation rules for data quality checks."""
    return {
        "version": "1.0",
        "description": "Deterministic validation rules for loan performance data",
        "rules": [
            {
                "id": "R001",
                "name": "balance_exceeds_original",
                "description": "Current balance should not exceed original balance (unless modification)",
                "condition": "current_balance > original_balance AND modification_flag == 0",
                "severity": "high",
                "exception_type": "data_entry_error"
            },
            {
                "id": "R002",
                "name": "negative_balance",
                "description": "Current balance should not be negative",
                "condition": "current_balance < 0",
                "severity": "critical",
                "exception_type": "data_entry_error"
            },
            {
                "id": "R003",
                "name": "dpd_status_mismatch",
                "description": "Days past due should be consistent with current status",
                "condition": "(current_status == 'Current' AND days_past_due > 0) OR (current_status == '30DPD' AND days_past_due != 30)",
                "severity": "high",
                "exception_type": "suspicious_transition"
            },
            {
                "id": "R004",
                "name": "negative_remaining_term",
                "description": "Remaining term should not be negative",
                "condition": "remaining_term_months < 0",
                "severity": "medium",
                "exception_type": "data_entry_error"
            },
            {
                "id": "R005",
                "name": "future_origination",
                "description": "Origination date should not be after reporting date",
                "condition": "origination_month > reporting_month",
                "severity": "critical",
                "exception_type": "data_entry_error"
            },
            {
                "id": "R006",
                "name": "closed_with_balance",
                "description": "Prepaid or defaulted loans should have zero or near-zero balance",
                "condition": "(current_status IN ('Prepaid') AND current_balance > 1000)",
                "severity": "high",
                "exception_type": "data_entry_error"
            },
            {
                "id": "R007",
                "name": "stale_update",
                "description": "Last updated date is more than 6 months before reporting month",
                "condition": "months_between(last_updated_at, reporting_month) > 6",
                "severity": "medium",
                "exception_type": "stale_record"
            },
            {
                "id": "R008",
                "name": "missing_documentation",
                "description": "Active loans should not have missing documentation",
                "condition": "current_status NOT IN ('Default', 'Prepaid') AND document_status == 'Missing'",
                "severity": "medium",
                "exception_type": "stale_record"
            },
            {
                "id": "R009",
                "name": "source_conflict",
                "description": "Balance or status differs between primary and servicer systems",
                "condition": "abs(primary_balance - servicer_balance) / primary_balance > 0.05",
                "severity": "high",
                "exception_type": "source_conflict"
            },
            {
                "id": "R010",
                "name": "impossible_transition",
                "description": "Loan should not transition from Default/Prepaid back to Current",
                "condition": "previous_status IN ('Default', 'Prepaid') AND current_status == 'Current'",
                "severity": "critical",
                "exception_type": "suspicious_transition"
            }
        ]
    }


def _generate_submission_template(df: pd.DataFrame) -> pd.DataFrame:
    """Generate submission template with required output columns."""
    sample = df[["loan_id", "month_index"]].head(10).copy()
    sample["prob_3m_delinquency"] = 0.0
    sample["prob_6m_delinquency"] = 0.0
    sample["prob_12m_default"] = 0.0
    sample["prob_12m_prepayment"] = 0.0
    sample["next_state"] = ""
    sample["exception_type"] = ""
    sample["anomaly_score"] = 0.0
    sample["top_drivers"] = ""
    sample["action"] = ""
    sample["confidence"] = 0.0
    return sample


def generate_all():
    """Generate all synthetic datasets and supporting files."""
    print("Generating synthetic data...")
    rng = np.random.default_rng(config.RANDOM_SEED)

    # Ensure output directory exists
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate static loan attributes
    print(f"  Generating {config.SYNTH_NUM_LOANS} loan static attributes...")
    static_df = _generate_loan_static(rng, config.SYNTH_NUM_LOANS)

    # 2. Simulate monthly histories
    print("  Simulating monthly loan histories...")
    all_records = []
    max_months = 36  # Max observation window
    for _, loan in static_df.iterrows():
        # Random observation length (18-36 months)
        n_months = rng.integers(18, max_months + 1)
        history = _simulate_loan_history(rng, loan, n_months)
        all_records.extend(history)

    panel_df = pd.DataFrame(all_records)
    print(f"  Generated {len(panel_df)} monthly records for {config.SYNTH_NUM_LOANS} loans")

    # 3. Compute forward-looking targets
    print("  Computing target variables (leakage-free)...")
    panel_df = _compute_targets(panel_df)

    # 4. Inject data quality issues
    print("  Injecting data quality issues for anomaly detection...")
    panel_df = _inject_data_quality_issues(panel_df, rng)

    # 5. Split into train / test (Time-aware split)
    # Train: has target labels; Test: targets removed
    # Cutoff where ~80% of records fall chronologically
    month_counts = panel_df["reporting_month"].value_counts().sort_index()
    cum_counts = month_counts.cumsum() / len(panel_df)
    test_cutoff = cum_counts[cum_counts >= 0.80].index[0]
    train_mask = panel_df["reporting_month"] < test_cutoff
    test_mask = ~train_mask

    train_df = panel_df[train_mask].copy()
    test_df = panel_df[test_mask].drop(columns=config.TARGET_COLUMNS).copy()

    print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    # 6. Save all files
    print("  Saving files...")
    train_df.to_csv(config.TRAIN_FILE, index=False)
    test_df.to_csv(config.TEST_FILE, index=False)
    static_df.to_csv(config.STATIC_FILE, index=False)

    # Servicer updates
    servicer_df = _generate_servicer_updates(panel_df, rng)
    servicer_df.to_csv(config.SERVICER_FILE, index=False)

    # Macro scenarios
    macro_df = _generate_macro_scenarios()
    macro_df.to_csv(config.MACRO_FILE, index=False)

    # Data dictionary
    with open(config.DATA_DICT_FILE, "w", encoding="utf-8") as f:
        f.write(_generate_data_dictionary())

    # Validation rules
    with open(config.VALIDATION_RULES_FILE, "w", encoding="utf-8") as f:
        json.dump(_generate_validation_rules(), f, indent=2)

    # Submission template
    template_df = _generate_submission_template(train_df)
    template_df.to_csv(config.SUBMISSION_TEMPLATE_FILE, index=False)

    print(f"\nFiles saved to {config.RAW_DATA_DIR}:")
    for f in sorted(config.RAW_DATA_DIR.glob("*")):
        if f.name != ".gitkeep":
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.2f} MB")

    print("\n[✓] Synthetic data generation complete")
    return train_df, test_df, static_df


if __name__ == "__main__":
    generate_all()
