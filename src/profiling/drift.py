"""
Train vs Test drift analysis (PSI, Kolmogorov-Smirnov, Chi-Squared).
"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp
from src import config

plt.switch_backend("Agg")


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between reference and target distributions."""
    expected_clean = expected.dropna().values
    actual_clean = actual.dropna().values

    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0

    # Determine quantile bins from reference (expected)
    percentiles = np.linspace(0, 100, buckets + 1)
    try:
        bins = np.percentile(expected_clean, percentiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        bins[0] = -np.inf
        bins[-1] = np.inf
    except Exception:
        return 0.0

    # Bin frequencies
    expected_counts, _ = np.histogram(expected_clean, bins=bins)
    actual_counts, _ = np.histogram(actual_clean, bins=bins)

    # Convert to proportions with smoothing for zero counts
    expected_pct = np.maximum(expected_counts / len(expected_clean), 1e-4)
    actual_pct = np.maximum(actual_counts / len(actual_clean), 1e-4)

    # Re-normalize
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_val, 4))


def analyze_drift(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Analyze feature distribution drift between training and test sets."""
    print("Analyzing train vs test drift...")
    warnings.filterwarnings("ignore")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if c in test_df.columns and not c.startswith("next_") and not c.startswith("prob_")
    ]
    cat_cols = [
        c for c in train_df.select_dtypes(include=["object", "category"]).columns
        if c in test_df.columns and c not in ("loan_id", "last_updated_at")
    ]

    drift_stats = {
        "numeric_psi": {},
        "numeric_ks": {},
        "categorical_chi2": {},
        "drift_summary": {"high_drift_cols": [], "moderate_drift_cols": [], "stable_cols": []},
    }

    # Numeric drift: PSI and KS test
    for col in numeric_cols:
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()

        if len(train_vals) > 0 and len(test_vals) > 0:
            psi_val = calculate_psi(train_vals, test_vals)
            drift_stats["numeric_psi"][col] = psi_val

            stat, p_val = ks_2samp(train_vals, test_vals)
            drift_stats["numeric_ks"][col] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
            }

            if psi_val >= 0.25:
                drift_stats["drift_summary"]["high_drift_cols"].append(col)
            elif psi_val >= 0.10:
                drift_stats["drift_summary"]["moderate_drift_cols"].append(col)
            else:
                drift_stats["drift_summary"]["stable_cols"].append(col)

    # Categorical drift: Chi-squared test
    for col in cat_cols:
        train_counts = train_df[col].value_counts(normalize=True).dropna()
        test_counts = test_df[col].value_counts(normalize=True).dropna()

        all_cats = list(set(train_counts.index).union(set(test_counts.index)))
        train_freq = np.array([train_counts.get(c, 1e-4) for c in all_cats]) * len(train_df)
        test_freq = np.array([test_counts.get(c, 1e-4) for c in all_cats]) * len(test_df)
        test_freq = test_freq * (train_freq.sum() / test_freq.sum())

        try:
            stat, p_val = chisquare(f_obs=test_freq, f_exp=train_freq)
            drift_stats["categorical_chi2"][col] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_val), 4),
            }
        except Exception:
            drift_stats["categorical_chi2"][col] = {"statistic": None, "p_value": None}

    # Generate PSI Summary plot
    if drift_stats["numeric_psi"]:
        psi_series = pd.Series(drift_stats["numeric_psi"]).sort_values(ascending=True)
        plt.figure(figsize=(9, 6))
        colors = ["green" if v < 0.10 else "orange" if v < 0.25 else "red" for v in psi_series.values]
        psi_series.plot(kind="barh", color=colors)
        plt.axvline(0.10, color="orange", linestyle="--", label="Moderate Drift (0.10)")
        plt.axvline(0.25, color="red", linestyle="--", label="Severe Drift (0.25)")
        plt.title("Population Stability Index (PSI) by Numeric Feature")
        plt.xlabel("PSI Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "drift_psi_summary.png", dpi=150)
        plt.close()

    return drift_stats
