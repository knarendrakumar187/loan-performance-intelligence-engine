"""
Correlation, association, and cross-column relationship profiling.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from src import config

plt.switch_backend("Agg")


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Compute Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty or confusion_matrix.shape[0] <= 1 or confusion_matrix.shape[1] <= 1:
        return 0.0
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n <= 1:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    min_dim = min((kcorr - 1), (rcorr - 1))
    if min_dim <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / min_dim))


def analyze_relationships(df: pd.DataFrame) -> dict:
    """Analyze correlations, categorical associations, and relationship breaks."""
    print("Analyzing feature correlations and relationships...")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pearson correlation for numeric columns
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if not c.startswith("next_") and not c.startswith("prob_")
    ]
    numeric_df = df[numeric_cols].dropna()
    pearson_corr = numeric_df.corr().round(4).to_dict()

    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title("Pearson Correlation Matrix (Numeric Features)")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "correlation_matrix.png", dpi=150)
        plt.close()

    # 2. Cramér's V for categorical features
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in ("loan_id", "last_updated_at", "reporting_month", "origination_month")
    ]
    cramers_matrix = {}
    for c1 in cat_cols:
        cramers_matrix[c1] = {}
        for c2 in cat_cols:
            if c1 == c2:
                cramers_matrix[c1][c2] = 1.0
            else:
                mask = df[c1].notna() & df[c2].notna()
                if mask.sum() > 10:
                    cramers_matrix[c1][c2] = round(cramers_v(df.loc[mask, c1], df.loc[mask, c2]), 4)
                else:
                    cramers_matrix[c1][c2] = 0.0

    if len(cat_cols) > 1:
        plt.figure(figsize=(9, 7))
        cramers_df = pd.DataFrame(cramers_matrix)
        sns.heatmap(cramers_df, annot=True, cmap="Blues", fmt=".2f", vmin=0, vmax=1)
        plt.title("Cramér's V Association Matrix (Categorical Features)")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "cramers_v_matrix.png", dpi=150)
        plt.close()

    # 3. Cross-column business relationship checks
    breaks = {}
    if "days_past_due" in df.columns and "current_status" in df.columns:
        dpd_stat_mismatch = (
            ((df["current_status"] == "Current") & (df["days_past_due"] > 0)) |
            ((df["current_status"] == "30DPD") & (df["days_past_due"] != 30))
        ).sum()
        breaks["dpd_vs_status_mismatches"] = int(dpd_stat_mismatch)

    if "current_balance" in df.columns and "original_balance" in df.columns:
        bal_gt_orig = (df["current_balance"] > df["original_balance"]).sum()
        breaks["balance_exceeds_original_count"] = int(bal_gt_orig)

    if "origination_month" in df.columns and "reporting_month" in df.columns:
        future_orig = (df["origination_month"] > df["reporting_month"]).sum()
        breaks["future_origination_date_count"] = int(future_orig)

    return {
        "pearson_correlation": pearson_corr,
        "cramers_v": cramers_matrix,
        "relationship_breaks": breaks,
    }
