import pandas as pd
import numpy as np
from src import config

def audit_feature_leakage(features_df: pd.DataFrame, targets_df: pd.DataFrame) -> dict:
    """
    Computes maximum Pearson and Spearman correlation between all engineered features and each target variable.
    Flags any feature with correlation > 0.90 as potential target leakage.
    Asserts that zero post-event administrative columns exist in the feature set.
    
    Args:
        features_df: Dataframe of engineered features.
        targets_df: Dataframe of target variables.
        
    Returns:
        Dictionary containing audit results and flagged features.
    """
    leakage_cols = getattr(config, 'LEAKAGE_DROP_COLUMNS', ['default_flag', 'prepayment_flag', 'loss_severity_band', 'last_updated_at', 'source_system', 'document_status'])
    
    # Assert zero post-event administrative columns exist
    found_leakage_cols = [c for c in leakage_cols if c in features_df.columns]
    assert len(found_leakage_cols) == 0, f"Leakage columns found in features: {found_leakage_cols}"
    
    results = {}
    flagged_features = []
    
    print(f"{'Feature':<40} | {'Target':<20} | {'Pearson':<10} | {'Spearman':<10} | {'Flagged':<10}")
    print("-" * 100)
    
    # Compute correlations
    for target_col in targets_df.columns:
        if target_col not in targets_df:
            continue
            
        target_series = targets_df[target_col].astype(float)
        
        for feature_col in features_df.columns:
            if not pd.api.types.is_numeric_dtype(features_df[feature_col]):
                continue
                
            feature_series = features_df[feature_col].astype(float)
            
            # Drop NaNs for correlation
            valid_idx = ~(feature_series.isna() | target_series.isna())
            if valid_idx.sum() < 2:
                continue
                
            pearson_corr = feature_series[valid_idx].corr(target_series[valid_idx], method='pearson')
            spearman_corr = feature_series[valid_idx].corr(target_series[valid_idx], method='spearman')
            
            # Handle possible NaNs in correlation results
            pearson_corr = 0.0 if np.isnan(pearson_corr) else pearson_corr
            spearman_corr = 0.0 if np.isnan(spearman_corr) else spearman_corr
            
            max_corr = max(abs(pearson_corr), abs(spearman_corr))
            is_flagged = max_corr > 0.90
            
            if is_flagged:
                flagged_features.append({
                    'feature': feature_col,
                    'target': target_col,
                    'pearson': pearson_corr,
                    'spearman': spearman_corr
                })
                
            print(f"{feature_col:<40} | {target_col:<20} | {pearson_corr:>10.4f} | {spearman_corr:>10.4f} | {str(is_flagged):<10}")
            
            results[f"{feature_col}_{target_col}"] = {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'flagged': is_flagged
            }
            
    print("-" * 100)
    print(f"Total features checked: {features_df.shape[1]}")
    print(f"Potential leakage features flagged: {len(flagged_features)}")
    
    if flagged_features:
        print("\nWARNING: Potential leakage detected in the following features:")
        for item in flagged_features:
            print(f" - {item['feature']} (Target: {item['target']}, Max Corr: {max(abs(item['pearson']), abs(item['spearman'])):.4f})")
    else:
        print("\nSUCCESS: No significant target leakage detected.")
        
    return {
        'audit_results': results,
        'flagged_features': flagged_features
    }
