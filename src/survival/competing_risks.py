"""
Time-to-Event and Competing Risks Survival Modeling for Loan Portfolios.
Implements Kaplan-Meier, Cox Proportional Hazards, and Cumulative Incidence Functions (CIF).
"""

from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from src import config
from src.data.loader import load_train

plt.switch_backend("Agg")


def prepare_survival_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Transform monthly panel data into loan-level time-to-event survival format with competing risk flags.

    Event Coding:
    - Event = 0: Right-censored (loan active at observation end)
    - Event = 1: Default (absorbing credit failure)
    - Event = 2: Prepayment (absorbing voluntary payoff)
    """
    print("Aggregating loan panel data into survival time-to-event format...")

    loan_records = []
    for loan_id, group in df.groupby("loan_id"):
        duration = int(group["loan_age_months"].max())
        orig_vintage = str(group["origination_month"].iloc[0])[:4]
        credit_band = str(group["credit_score_band"].iloc[0])
        interest_rate = float(group["interest_rate"].median())
        orig_balance = float(group["original_balance"].iloc[0])

        # Check terminal state
        has_defaulted = int(any(group["default_flag"] == 1) or any(group["current_status"] == "Default"))
        has_prepaid = int(any(group["prepayment_flag"] == 1) or any(group["current_status"] == "Prepaid"))

        if has_defaulted:
            event_type = 1  # Default
        elif has_prepaid:
            event_type = 2  # Prepayment
        else:
            event_type = 0  # Right-censored

        loan_records.append({
            "loan_id": loan_id,
            "duration": max(1, duration),
            "event_type": event_type,
            "is_default": 1 if event_type == 1 else 0,
            "is_prepaid": 1 if event_type == 2 else 0,
            "vintage": orig_vintage,
            "credit_band": credit_band,
            "is_prime": 1 if credit_band in ("740-779", "780+", "700-739") else 0,
            "interest_rate": interest_rate,
            "original_balance": orig_balance,
        })

    surv_df = pd.DataFrame(loan_records)
    print(f"  Total survival cohorts: {len(surv_df):,} loans")
    print(f"  Defaults: {surv_df['is_default'].sum():,} ({surv_df['is_default'].mean()*100:.1f}%)")
    print(f"  Prepayments: {surv_df['is_prepaid'].sum():,} ({surv_df['is_prepaid'].mean()*100:.1f}%)")
    print(f"  Right-censored (Active): {(surv_df['event_type']==0).sum():,} ({(surv_df['event_type']==0).mean()*100:.1f}%)")

    return surv_df


def run_survival_analysis() -> Dict[str, any]:
    """Execute full survival modeling suite and save plots."""
    print("=" * 60)
    print("Running Task 3: Time-to-Event / Survival Modeling Engine")
    print("=" * 60)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw_train = load_train()
    surv_df = prepare_survival_dataset(raw_train)

    # ──────────────────────────────────────────────
    # 1. Competing Risks: Cumulative Incidence Function (CIF)
    # ──────────────────────────────────────────────
    kmf_default = KaplanMeierFitter()
    kmf_prepay = KaplanMeierFitter()

    kmf_default.fit(surv_df["duration"], event_observed=surv_df["is_default"], label="Default Risk")
    kmf_prepay.fit(surv_df["duration"], event_observed=surv_df["is_prepaid"], label="Prepayment Risk")

    plt.figure(figsize=(9, 6))
    plt.plot(kmf_default.timeline, 1.0 - kmf_default.survival_function_, label="Cumulative Default Probability", color="#d62728", lw=2)
    plt.plot(kmf_prepay.timeline, 1.0 - kmf_prepay.survival_function_, label="Cumulative Prepayment Probability", color="#2ca02c", lw=2)
    plt.title("Competing Risks: Cumulative Incidence Curves (Default vs Prepayment)")
    plt.xlabel("Loan Age (Months on Book)")
    plt.ylabel("Cumulative Incidence Probability")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    cif_plot = config.FIGURES_DIR / "survival_cif_competing_risks.png"
    plt.savefig(cif_plot, dpi=150)
    plt.close()

    # ──────────────────────────────────────────────
    # 2. Stratified Survival Curves by Credit Quality
    # ──────────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    kmf_prime = KaplanMeierFitter()
    kmf_subprime = KaplanMeierFitter()

    prime_mask = surv_df["is_prime"] == 1
    kmf_prime.fit(surv_df.loc[prime_mask, "duration"], event_observed=surv_df.loc[prime_mask, "is_default"], label="Prime (Score >= 700)")
    kmf_subprime.fit(surv_df.loc[~prime_mask, "duration"], event_observed=surv_df.loc[~prime_mask, "is_default"], label="Subprime / Near-Prime (< 700)")

    kmf_prime.plot_survival_function(ci_show=True, color="#1f77b4", lw=2)
    kmf_subprime.plot_survival_function(ci_show=True, color="#ff7f0e", lw=2)
    plt.title("Default-Free Survival Curves by Credit Quality Segment")
    plt.xlabel("Months Since Origination")
    plt.ylabel("Default-Free Survival Probability S(t)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    credit_plot = config.FIGURES_DIR / "survival_by_credit_segment.png"
    plt.savefig(credit_plot, dpi=150)
    plt.close()

    # Log-rank test
    lr_res = logrank_test(
        surv_df.loc[prime_mask, "duration"],
        surv_df.loc[~prime_mask, "duration"],
        event_observed_A=surv_df.loc[prime_mask, "is_default"],
        event_observed_B=surv_df.loc[~prime_mask, "is_default"],
    )
    print(f"  Log-Rank Test p-value (Credit Segments): {lr_res.p_value:.4e}")

    # ──────────────────────────────────────────────
    # 3. Stratified Survival Curves by Vintage
    # ──────────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    for vintage in sorted(surv_df["vintage"].unique()):
        v_mask = surv_df["vintage"] == vintage
        kmf_v = KaplanMeierFitter()
        kmf_v.fit(surv_df.loc[v_mask, "duration"], event_observed=surv_df.loc[v_mask, "is_default"], label=f"{vintage} Vintage")
        kmf_v.plot_survival_function(ci_show=False, lw=2)

    plt.title("Default-Free Survival Curves by Origination Vintage")
    plt.xlabel("Months Since Origination")
    plt.ylabel("Survival Probability S(t)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    vintage_plot = config.FIGURES_DIR / "survival_by_vintage.png"
    plt.savefig(vintage_plot, dpi=150)
    plt.close()

    # ──────────────────────────────────────────────
    # 4. Cox Proportional Hazards Model
    # ──────────────────────────────────────────────
    print("Fitting Cox Proportional Hazards model...")
    cph_df = surv_df[["duration", "is_default", "is_prime", "interest_rate", "original_balance"]].copy()
    cph_df["orig_bal_100k"] = cph_df["original_balance"] / 100_000.0
    cph_df = cph_df.drop(columns=["original_balance"])

    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col="duration", event_col="is_default")
    c_index = cph.concordance_index_
    hazard_ratios = cph.hazard_ratios_.to_dict()

    print(f"  Cox PH Concordance Index (C-index): {c_index:.4f}")
    print(f"  Hazard Ratios: {hazard_ratios}")

    return {
        "c_index": round(float(c_index), 4),
        "hazard_ratios": hazard_ratios,
        "logrank_p_value": float(lr_res.p_value),
        "sample_size": len(surv_df),
    }


if __name__ == "__main__":
    run_survival_analysis()
