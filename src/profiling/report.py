"""
Data intelligence and profiling orchestrator.
Generates comprehensive data_intelligence_report.md from raw datasets.
"""

from pathlib import Path
import pandas as pd
from src import config
from src.data.loader import load_test, load_train
from src.profiling.distributions import analyze_distributions
from src.profiling.drift import analyze_drift
from src.profiling.missingness import analyze_missingness
from src.profiling.outliers import detect_outliers
from src.profiling.quality_score import compute_quality_scores, get_batch_quality_summary
from src.profiling.relationships import analyze_relationships


def generate_report():
    """Run full data profiling suite and output markdown report."""
    print("=" * 60)
    print("Running Task 1: Data Intelligence & Profiling Engine")
    print("=" * 60)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.REPORTS_DIR / "data_intelligence_report.md"

    # 1. Load Data
    train_df = load_train()
    test_df = load_test()

    # 2. Run analyses
    dist_stats = analyze_distributions(train_df)
    missing_stats = analyze_missingness(train_df)
    outliers_df = detect_outliers(train_df)
    rel_stats = analyze_relationships(train_df)
    drift_stats = analyze_drift(train_df, test_df)
    quality_scores = compute_quality_scores(train_df)
    batch_quality = get_batch_quality_summary(quality_scores)

    # 3. Construct Markdown Report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Task 1: Data Intelligence & Profiling Report\n\n")
        f.write("**Challenge:** FinTech AI Challenge | AI Track  \n")
        f.write(f"**Dataset Dimensions:** Train = {len(train_df):,} rows × {train_df.shape[1]} columns | Test = {len(test_df):,} rows × {test_df.shape[1]} columns  \n")
        f.write(f"**Observation Period:** {train_df['reporting_month'].min()} to {train_df['reporting_month'].max()} (Train) vs {test_df['reporting_month'].min()} to {test_df['reporting_month'].max()} (Test)\n\n")

        f.write("---\n\n")

        # Section 1: Data Quality Index
        f.write("## 1. Data Quality Index (DQI) & Summary Scores\n\n")
        f.write("The composite Data Quality Index is computed from three weighted pillars: **Completeness (35%)**, **Validity (35%)**, and **Consistency (30%)**.\n\n")
        f.write("| Quality Metric | Score / Value | Status / Description |\n")
        f.write("|----------------|---------------|----------------------|\n")
        f.write(f"| **Overall Mean DQI** | **{batch_quality['mean_quality_score']:.2f} / 100** | Weighted overall portfolio quality |\n")
        f.write(f"| **Median DQI** | {batch_quality['median_quality_score']:.2f} / 100 | Robust central tendency |\n")
        f.write(f"| **Completeness Pillar** | {batch_quality['mean_completeness']:.2f}% | Non-null feature rate across schema |\n")
        f.write(f"| **Validity Pillar** | {batch_quality['mean_validity']:.2f} / 100 | Adherence to value ranges & positive balances |\n")
        f.write(f"| **Consistency Pillar** | {batch_quality['mean_consistency']:.2f} / 100 | Cross-field logic (DPD vs status, balance <= orig) |\n")
        f.write(f"| **Clean Records (DQI >= 95)** | {batch_quality['clean_records_pct']:.2f}% | Zero or negligible anomaly flags |\n\n")

        f.write("### Quality Grade Breakdown\n\n")
        f.write("| Grade | Threshold | Proportion of Portfolio |\n")
        f.write("|-------|-----------|-------------------------|\n")
        for grade, prop in batch_quality["grade_distribution"].items():
            f.write(f"| Grade {grade} | {'>= 90' if grade=='A' else '>= 80' if grade=='B' else '>= 70' if grade=='C' else '>= 60' if grade=='D' else '< 60'} | {prop * 100:.2f}% |\n")
        f.write("\n---\n\n")

        # Section 2: Missingness
        f.write("## 2. Missingness Profiling & Pattern Analysis\n\n")
        f.write("| Column Name | Missing Count | Missing Percentage | Imputation Strategy |\n")
        f.write("|-------------|---------------|--------------------|---------------------|\n")
        if missing_stats["column_stats"]:
            for col, stats in missing_stats["column_stats"].items():
                strategy = "Median + Indicator" if col in ("interest_rate", "current_balance") else "Missing Category / Mode"
                f.write(f"| `{col}` | {stats['count']:,} | {stats['percentage']:.2f}% | {strategy} |\n")
        else:
            f.write("| *No missing values detected* | 0 | 0.00% | N/A |\n")
        f.write("\n---\n\n")

        # Section 3: Outliers & Rule Violations
        f.write("## 3. Outlier Detection & Business Rule Violations\n\n")
        rule_cols = [c for c in outliers_df.columns if c.startswith("rule_")]
        f.write("| Rule ID & Name | Violation Count | Portfolio % | Severity | Exception Class |\n")
        f.write("|----------------|-----------------|-------------|----------|-----------------|\n")
        for r_col in rule_cols:
            v_count = int(outliers_df[r_col].sum())
            v_pct = (v_count / len(train_df)) * 100
            sev = "High" if "negative" in r_col or "future" in r_col else "Medium"
            f.write(f"| `{r_col}` | {v_count:,} | {v_pct:.2f}% | {sev} | Data Quality / Rule Exception |\n")
        f.write(f"\n**Total Records Flagged for Review:** `{(outliers_df['total_outlier_flags'] > 0).sum():,}` records (`{((outliers_df['total_outlier_flags'] > 0).sum() / len(train_df)) * 100:.2f}%` of train).\n\n")

        f.write("---\n\n")

        # Section 4: Cross-Column Relationships
        f.write("## 4. Feature Correlations & Categorical Associations\n\n")
        f.write("- **Pearson Correlation Heatmap:** Saved to `reports/figures/correlation_matrix.png`\n")
        f.write("- **Cramér's V Association Matrix:** Saved to `reports/figures/cramers_v_matrix.png`\n")
        f.write(f"- **DPD vs Status Relationship Breaks:** `{rel_stats['relationship_breaks'].get('dpd_vs_status_mismatches', 0):,}` occurrences\n")
        f.write(f"- **Balance Exceeding Original (Unmodified):** `{rel_stats['relationship_breaks'].get('balance_exceeds_original_count', 0):,}` occurrences\n\n")

        f.write("---\n\n")

        # Section 5: Train vs Test Drift
        f.write("## 5. Train vs. Test Drift Assessment\n\n")
        f.write("Drift evaluated using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) tests** for numeric features, and **Chi-Squared tests** for categorical features.\n\n")
        f.write("| Feature | Metric Type | Drift Metric (PSI / Stat) | p-value | Drift Classification |\n")
        f.write("|---------|-------------|---------------------------|---------|----------------------|\n")
        for col, psi in drift_stats["numeric_psi"].items():
            ks_p = drift_stats["numeric_ks"].get(col, {}).get("p_value", "N/A")
            tag = "🟢 Stable (PSI < 0.10)" if psi < 0.10 else "🟡 Moderate (0.10 <= PSI < 0.25)" if psi < 0.25 else "🔴 Severe (PSI >= 0.25)"
            f.write(f"| `{col}` | Numeric PSI | {psi:.4f} | {ks_p} | {tag} |\n")

        for col, chi in drift_stats["categorical_chi2"].items():
            if chi.get("statistic") is not None:
                f.write(f"| `{col}` | Categorical Chi2 | {chi['statistic']:.2f} | {chi['p_value']:.4e} | {'🟢 Stable' if chi['p_value'] > 0.01 else '🟡 Statistically Significant Shift'} |\n")

        f.write("\n---\n\n")

        # Section 6: Recommendations
        f.write("## 6. Recommendations for Downstream ML Pipeline (Task 2)\n\n")
        f.write("1. **Time-Aware Split:** Enforce chronological train/validation/test split by `reporting_month`.\n")
        f.write("2. **Leakage Controls:** Explicitly drop `default_flag`, `prepayment_flag`, `loss_severity_band`, and `last_updated_at` before model training.\n")
        f.write("3. **Missing Value Imputation:** Apply median imputation with missing indicators for numeric features; create an explicit `'Missing'` category for categorical bands.\n")
        f.write("4. **Target Seasoning:** Compute historical delinquency momentum and balance decay rate to assist tree ensembles in capturing early credit deterioration.\n")

    print(f"[✓] Data Intelligence Report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_report()
