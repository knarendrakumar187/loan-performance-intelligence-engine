"""
Anomaly detection orchestrator for Task 4.
Runs hybrid rule + ML scoring, generates reviewer-ready examples, and outputs markdown report.
"""

from pathlib import Path
import pandas as pd
from src import config
from src.anomaly.explainer import generate_anomaly_explanations
from src.anomaly.ml_scorer import compute_composite_anomaly_scores, train_anomaly_models
from src.data.loader import load_train


def run_anomaly_detection() -> pd.DataFrame:
    """Execute complete anomaly and exception detection pipeline."""
    print("=" * 60)
    print("Running Task 4: Anomaly & Exception Intelligence Engine")
    print("=" * 60)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / "anomaly_report.md"

    # 1. Load train dataset
    train_df = load_train()

    # 2. Train models
    iso_forest, clf_exception = train_anomaly_models(train_df)

    # 3. Score records
    scores, exception_types, rules_df = compute_composite_anomaly_scores(train_df, iso_forest, clf_exception)

    # 4. Generate Top Reviewer-Ready Examples (25 examples >= 20 required)
    examples_df = generate_anomaly_explanations(train_df, scores, exception_types, rules_df, top_n=25)

    # 5. Write Markdown Report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Task 4: Anomaly & Exception Detection Report\n\n")
        f.write("**Challenge:** Intain Campus FinTech Challenge 2026 | AI Track  \n")
        f.write(f"**Portfolio Analyzed:** {len(train_df):,} monthly loan performance records  \n")
        f.write(f"**Detection Framework:** Hybrid Rule-Based (10 deterministic checks) + Unsupervised Spatial ML (Isolation Forest) + Supervised Exception Classification\n\n")

        f.write("---\n\n")

        # Section 1: Summary Statistics
        f.write("## 1. Exception & Anomaly Distribution Overview\n\n")
        exc_counts = exception_types.value_counts()
        f.write("| Exception Class | Total Flagged | Proportion of Portfolio | Primary Risk Driver |\n")
        f.write("|-----------------|---------------|-------------------------|---------------------|\n")
        for exc, count in exc_counts.items():
            desc = "Normal performing records" if exc == "none" else "Accounting balance/ledger errors" if exc == "data_entry_error" else "Servicer dual-tape mismatches" if exc == "source_conflict" else "Contract vs delinquency mismatch" if exc == "suspicious_transition" else "Custodian document deficiencies"
            f.write(f"| `{exc}` | {count:,} | {count/len(train_df)*100:.2f}% | {desc} |\n")

        f.write(f"\n- **High-Risk Flagged Records (Score >= 0.60):** `{(scores >= 0.60).sum():,}` records (`{(scores >= 0.60).mean()*100:.2f}%`)\n")
        f.write(f"- **Mean Anomaly Score:** `{scores.mean():.4f}` | **Median Anomaly Score:** `{scores.median():.4f}`\n\n")

        f.write("---\n\n")

        # Section 2: Reviewer-Ready Examples Table
        f.write("## 2. Reviewer-Ready Anomaly Examples (25 Case Studies)\n\n")
        f.write("The table below presents high-priority flagged records requiring human underwriting/servicing review, complete with anomaly drivers and prescribed remediations:\n\n")

        f.write("| Loan ID | Reporting Month | Anomaly Score | Exception Class | Triggered Rules | Diagnostic Reasoning | Recommended Action |\n")
        f.write("|---------|-----------------|---------------|-----------------|-----------------|----------------------|--------------------|\n")

        for _, row in examples_df.iterrows():
            f.write(f"| `{row['loan_id']}` | {row['reporting_month']} | **{row['anomaly_score']:.3f}** | `{row['exception_type']}` | {row['triggered_rules']} | {row['plain_english_reasoning']} | *{row['recommended_action']}* |\n")

        f.write("\n---\n\n")

        # Section 3: Governance & Review Protocol
        f.write("## 3. Operational Governance & Servicer Remediation Workflow\n\n")
        f.write("1. **Data Entry Errors (`data_entry_error`):** Automatically routed to loan servicing accounting queues with negative balance or over-balance alerts.\n")
        f.write("2. **Source Conflicts (`source_conflict`):** Flagged for dual-source tape cross-reconciliation with primary servicer API.\n")
        f.write("3. **Suspicious State Transitions (`suspicious_transition`):** Pauses automated default/foreclosure processing pending manual servicer status audit.\n")
        f.write("4. **Stale Records (`stale_record`):** Issues automated custodian document deficiency tickets.\n")

    print(f"[✓] Anomaly Detection Report generated: {report_path}")
    return examples_df


if __name__ == "__main__":
    run_anomaly_detection()
