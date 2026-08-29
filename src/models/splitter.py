"""
Time-aware data splitting engine for loan performance models.
Ensures zero future-to-past data leakage and strict chronological validation.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from src import config


def time_aware_train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.25,
    time_col: str = "reporting_month",
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Perform a strict chronological time-aware train/validation split.

    This ensures models are trained on historical months and validated on future months,
    preventing any lookahead bias or multi-record loan leakage across folds.

    Args:
        df: Input training dataframe with features and targets.
        val_ratio: Approximate fraction of latest chronological observations for validation.
        time_col: Column indicating the observation timestamp/month.

    Returns:
        train_split: Historical training split.
        val_split: Forward out-of-time validation split.
        cutoff_month: The chronological boundary month string.
    """
    print(f"Executing time-aware chronological split on `{time_col}` (val_ratio={val_ratio})...")

    # Get chronological unique months
    sorted_months = sorted(df[time_col].dropna().unique())
    month_counts = df[time_col].value_counts().sort_index()
    cum_pct = month_counts.cumsum() / len(df)

    # Find cutoff where (1 - val_ratio) proportion of records fall
    train_target_pct = 1.0 - val_ratio
    cutoff_idx = cum_pct[cum_pct >= train_target_pct].index[0]

    train_mask = df[time_col] < cutoff_idx
    val_mask = df[time_col] >= cutoff_idx

    train_split = df[train_mask].copy()
    val_split = df[val_mask].copy()

    print(f"  Chronological cutoff: `{cutoff_idx}`")
    print(f"  Train split: {len(train_split):,} records ({len(train_split)/len(df)*100:.1f}%) | Period: {train_split[time_col].min()} to {train_split[time_col].max()}")
    print(f"  Validation split: {len(val_split):,} records ({len(val_split)/len(df)*100:.1f}%) | Period: {val_split[time_col].min()} to {val_split[time_col].max()}")

    return train_split, val_split, cutoff_idx
