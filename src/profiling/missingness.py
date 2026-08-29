import pandas as pd
import numpy as np
from src import config

def analyze_missingness(df: pd.DataFrame) -> dict:
    """Analyze missing values in the dataset."""
    print("Analyzing missingness...")
    
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    col_missing = {}
    for col in df.columns:
        if missing_counts[col] > 0:
            col_missing[col] = {
                'count': int(missing_counts[col]),
                'percentage': float(missing_pct[col])
            }
            
    missing_indicator_df = df.isnull().astype(int)
    cols_with_missing = missing_indicator_df.columns[missing_indicator_df.sum() > 0]
    
    if len(cols_with_missing) > 1:
        missing_corr = missing_indicator_df[cols_with_missing].corr().to_dict()
    else:
        missing_corr = {}
        
    mcar_test = {'status': 'not_implemented', 'note': 'Proxy implementation'}
    
    return {
        'column_stats': col_missing,
        'missing_patterns_corr': missing_corr,
        'mcar_test_proxy': mcar_test
    }
