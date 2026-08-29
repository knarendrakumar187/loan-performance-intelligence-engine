"""
Data loader utilities for loading raw and processed datasets.
"""

import pandas as pd
from src import config


def load_train() -> pd.DataFrame:
    """Load the training dataset (monthly performance panel)."""
    return pd.read_csv(config.TRAIN_FILE)


def load_test() -> pd.DataFrame:
    """Load the test dataset (unlabeled)."""
    return pd.read_csv(config.TEST_FILE)


def load_static() -> pd.DataFrame:
    """Load loan static attributes (origination-level)."""
    return pd.read_csv(config.STATIC_FILE)


def load_servicer_updates() -> pd.DataFrame:
    """Load servicer updates (partial/conflicting records)."""
    return pd.read_csv(config.SERVICER_FILE)


def load_macro_scenarios() -> pd.DataFrame:
    """Load macro scenario assumptions."""
    return pd.read_csv(config.MACRO_FILE)
