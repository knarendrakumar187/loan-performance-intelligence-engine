"""
Central configuration for the Loan Performance Intelligence Engine.
All paths, seeds, and key parameters in one place for reproducibility.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
DOCS_DIR = PROJECT_ROOT / "docs"

# ──────────────────────────────────────────────
# Data files
# ──────────────────────────────────────────────
TRAIN_FILE = RAW_DATA_DIR / "loan_monthly_performance_train.csv"
TEST_FILE = RAW_DATA_DIR / "loan_monthly_performance_test.csv"
STATIC_FILE = RAW_DATA_DIR / "loan_static_attributes.csv"
SERVICER_FILE = RAW_DATA_DIR / "servicer_updates.csv"
MACRO_FILE = RAW_DATA_DIR / "macro_scenarios.csv"
DATA_DICT_FILE = RAW_DATA_DIR / "data_dictionary.md"
VALIDATION_RULES_FILE = RAW_DATA_DIR / "validation_rules.json"
SUBMISSION_TEMPLATE_FILE = RAW_DATA_DIR / "submission_template.csv"
SUBMISSION_OUTPUT_FILE = SUBMISSION_DIR / "submission.csv"

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Synthetic data parameters
# ──────────────────────────────────────────────
SYNTH_NUM_LOANS = 3000
SYNTH_AVG_MONTHS = 24
SYNTH_TARGET_ROWS = 50_000

# ──────────────────────────────────────────────
# Schema definitions (from problem statement §7)
# ──────────────────────────────────────────────
FEATURE_COLUMNS = [
    "loan_id", "month_index", "reporting_month", "origination_month",
    "loan_age_months", "remaining_term_months", "original_balance",
    "current_balance", "interest_rate", "credit_score_band", "ltv_band",
    "dti_band", "state", "loan_purpose", "occupancy_type", "property_type",
    "servicer_name", "current_status", "days_past_due", "modification_flag",
    "prepayment_flag", "default_flag", "loss_severity_band",
    "last_updated_at", "source_system", "document_status",
]

TARGET_COLUMNS = [
    "next_3m_delinquency_flag",
    "next_6m_delinquency_flag",
    "next_12m_default_flag",
    "next_12m_prepayment_flag",
    "next_state",
    "exception_required",
    "exception_type",
]

# Columns that MUST be dropped before model training (leakage risk)
LEAKAGE_DROP_COLUMNS = [
    "default_flag",          # Direct target leakage
    "prepayment_flag",       # Direct target leakage
    "loss_severity_band",    # Post-event information
    "last_updated_at",       # Administrative timestamp
    "source_system",         # Administrative metadata
    "document_status",       # Post-event information
]

# ──────────────────────────────────────────────
# Categorical value domains
# ──────────────────────────────────────────────
CREDIT_SCORE_BANDS = ["<620", "620-659", "660-699", "700-739", "740-779", "780+"]
LTV_BANDS = ["<60", "60-70", "70-80", "80-90", "90-95", ">95"]
DTI_BANDS = ["<20", "20-30", "30-40", "40-50", ">50"]
STATES = [
    "CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
    "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "UT",
]
LOAN_PURPOSES = ["Purchase", "Refinance_Cashout", "Refinance_NoCashout"]
OCCUPANCY_TYPES = ["Primary", "Second_Home", "Investment"]
PROPERTY_TYPES = ["Single_Family", "Condo", "Townhouse", "Multi_Family"]
SERVICER_NAMES = ["ServicerA", "ServicerB", "ServicerC", "ServicerD", "ServicerE"]
LOAN_STATUSES = ["Current", "30DPD", "60DPD", "90DPD", "Default", "Prepaid"]
LOSS_SEVERITY_BANDS = ["None", "Low", "Medium", "High"]
DOCUMENT_STATUSES = ["Complete", "Partial", "Missing", "Stale"]
SOURCE_SYSTEMS = ["SystemA", "SystemB"]
EXCEPTION_TYPES = [
    "none", "data_entry_error", "stale_record",
    "source_conflict", "suspicious_transition",
]

# ──────────────────────────────────────────────
# Time-aware split boundaries
# ──────────────────────────────────────────────
# Reporting months are 1..30 (synthetic)
TRAIN_MONTHS_END = 18       # Train: months 1-18
VAL_MONTHS_END = 22         # Val:   months 19-22
# Test: months 23+

# ──────────────────────────────────────────────
# Model parameters
# ──────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

# ──────────────────────────────────────────────
# Scenario definitions
# ──────────────────────────────────────────────
SCENARIOS = {
    "base": {
        "credit_score_shift": 0,
        "interest_rate_shift": 0.0,
        "default_rate_multiplier": 1.0,
        "prepayment_rate_multiplier": 1.0,
        "description": "Current economic conditions maintained",
    },
    "adverse_credit": {
        "credit_score_shift": -1,   # Shift one band down
        "interest_rate_shift": 1.5,  # +150bps
        "default_rate_multiplier": 2.0,
        "prepayment_rate_multiplier": 0.6,
        "description": "Credit tightening with higher defaults",
    },
    "high_prepayment": {
        "credit_score_shift": 0,
        "interest_rate_shift": -1.0,  # -100bps (rate drop triggers refis)
        "default_rate_multiplier": 0.8,
        "prepayment_rate_multiplier": 2.5,
        "description": "Rate drop driving heavy refinancing activity",
    },
}
