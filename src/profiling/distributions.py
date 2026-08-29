"""
Column distribution analysis and visualization for loan performance data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, kurtosis, skew
from src import config

plt.switch_backend("Agg")


def analyze_distributions(df: pd.DataFrame) -> dict:
    """Analyze statistical distributions of all columns and save distribution plots."""
    print("Analyzing feature distributions...")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}

    for col in df.columns:
        if col.startswith("next_") or col.startswith("prob_"):
            continue

        col_stats = {}
        if pd.api.types.is_numeric_dtype(df[col]):
            col_dropna = df[col].dropna()
            if len(col_dropna) > 0:
                col_stats = {
                    "type": "numeric",
                    "count": int(len(col_dropna)),
                    "missing_pct": round(float(df[col].isna().mean() * 100), 2),
                    "mean": round(float(col_dropna.mean()), 4),
                    "median": round(float(col_dropna.median()), 4),
                    "std": round(float(col_dropna.std()), 4),
                    "skewness": round(float(skew(col_dropna)), 4),
                    "kurtosis": round(float(kurtosis(col_dropna)), 4),
                    "percentiles": {
                        "p1": round(float(col_dropna.quantile(0.01)), 2),
                        "p25": round(float(col_dropna.quantile(0.25)), 2),
                        "p50": round(float(col_dropna.quantile(0.50)), 2),
                        "p75": round(float(col_dropna.quantile(0.75)), 2),
                        "p99": round(float(col_dropna.quantile(0.99)), 2),
                    },
                }

                # Save plot for key numeric features
                if col in ("current_balance", "original_balance", "interest_rate", "loan_age_months", "days_past_due"):
                    plt.figure(figsize=(8, 5))
                    sns.histplot(col_dropna, kde=True, bins=30, color="#1f77b4")
                    plt.title(f"Distribution of {col}")
                    plt.xlabel(col)
                    plt.ylabel("Frequency")
                    plt.tight_layout()
                    plt.savefig(config.FIGURES_DIR / f"dist_{col}.png", dpi=150)
                    plt.close()

        elif pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            val_counts = df[col].value_counts(dropna=False)
            probs = val_counts / len(df)
            col_stats = {
                "type": "categorical",
                "count": int(df[col].notna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 2),
                "cardinality": int(df[col].nunique(dropna=True)),
                "entropy": round(float(entropy(probs)), 4),
                "top_categories": val_counts.head(5).to_dict(),
            }

            # Save bar chart for key categoricals
            if col in ("current_status", "credit_score_band", "ltv_band", "dti_band", "loan_purpose"):
                plt.figure(figsize=(8, 4.5))
                val_counts.head(10).plot(kind="bar", color="#2ca02c")
                plt.title(f"Category Distribution: {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                plt.savefig(config.FIGURES_DIR / f"dist_{col}.png", dpi=150)
                plt.close()

        stats[col] = col_stats

    return stats
