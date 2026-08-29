"""
Naive Baseline Survival Models (Constant Hazard / Exponential Model).
Provides reference log-likelihood, AIC, and C-index baselines for survival comparisons.
"""

from typing import Dict
import numpy as np
import pandas as pd
from lifelines import ExponentialFitter, WeibullFitter
from src.survival.competing_risks import prepare_survival_dataset
from src.data.loader import load_train


def fit_baseline_hazard_models() -> Dict[str, any]:
    """Fit naive constant hazard (Exponential) and Weibull survival baselines."""
    print("Fitting baseline parametric survival models (Constant Hazard vs Weibull)...")
    raw_train = load_train()
    surv_df = prepare_survival_dataset(raw_train)

    durations = surv_df["duration"]
    events = surv_df["is_default"]

    # 1. Constant Hazard (Exponential Model)
    exp_fitter = ExponentialFitter()
    exp_fitter.fit(durations, event_observed=events)
    exp_lambda = float(exp_fitter.lambda_)
    exp_aic = float(exp_fitter.AIC_)
    exp_log_likelihood = float(exp_fitter.log_likelihood_)

    # 2. Weibull Accelerated Failure Time (AFT) Baseline
    weibull_fitter = WeibullFitter()
    weibull_fitter.fit(durations, event_observed=events)
    weibull_aic = float(weibull_fitter.AIC_)
    weibull_log_likelihood = float(weibull_fitter.log_likelihood_)

    print(f"  [Baseline: Constant Hazard] Lambda: {exp_lambda:.4f} | Log-Likelihood: {exp_log_likelihood:.2f} | AIC: {exp_aic:.2f}")
    print(f"  [Parametric: Weibull] Log-Likelihood: {weibull_log_likelihood:.2f} | AIC: {weibull_aic:.2f}")

    return {
        "exponential_hazard_rate": round(exp_lambda, 6),
        "exponential_aic": round(exp_aic, 2),
        "exponential_log_likelihood": round(exp_log_likelihood, 2),
        "weibull_aic": round(weibull_aic, 2),
        "weibull_log_likelihood": round(weibull_log_likelihood, 2),
    }


if __name__ == "__main__":
    fit_baseline_hazard_models()
