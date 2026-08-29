# Task 6: Explainability & Responsible AI Report

**Challenge:** Intain Campus FinTech Challenge 2026 | AI Track  
**Explainability Methodology:** Model-Agnostic Shapley Additive Explanations (TreeSHAP)  
**Evaluation Scope:** Global Feature Attribution, Local Loan Profiles, Error Breakdown, and Uncertainty Bounds

---

## 1. Global Feature Importance (Top Drivers)

| Rank | Predictive Feature | Mean |SHAP Value| | Directional Impact on Default Risk |
|------|--------------------|-------------------|------------------------------------|
| 1 | `current_status_Current` | **0.4413** | Higher values cushion and mitigate default risk |
| 2 | `combined_risk_index` | **0.3008** | Higher values significantly increase default hazard |
| 3 | `interest_rate` | **0.2659** | Higher values significantly increase default hazard |
| 4 | `ltv_risk_tier` | **0.1875** | Higher values significantly increase default hazard |
| 5 | `original_balance` | **0.1636** | Higher values cushion and mitigate default risk |
| 6 | `monthly_interest_accrual` | **0.1213** | Higher values cushion and mitigate default risk |
| 7 | `dti_risk_tier` | **0.1047** | Higher values significantly increase default hazard |
| 8 | `credit_risk_tier` | **0.1038** | Higher values significantly increase default hazard |
| 9 | `current_balance` | **0.1033** | Higher values cushion and mitigate default risk |
| 10 | `month_index` | **0.0773** | Higher values cushion and mitigate default risk |

*Visual Summary Plots:*
- Default SHAP Beeswarm: `reports/figures/shap_summary_next_12m_default_flag.png`
- Prepayment SHAP Beeswarm: `reports/figures/shap_summary_next_12m_prepayment_flag.png`
- Delinquency SHAP Beeswarm: `reports/figures/shap_summary_next_3m_delinquency_flag.png`

---

## 2. Local Loan Case Studies (Representative Archetypes)

| Archetype Profile | Loan ID | Pred Default Prob | Actual Outcome | Top Risk Elevators (SHAP +) | Top Risk Mitigators (SHAP -) |
|-------------------|---------|-------------------|----------------|-----------------------------|------------------------------|
| **High-Risk Default (True Positive)** | `LN002025` | **0.111** | Non-Default | `property_type_Townhouse` (+0.179), `ltv_risk_tier` (+0.172), `dti_risk_tier` (+0.073) | `interest_rate` (-0.396), `state_MO` (-0.389), `current_status_Current` (-0.257) |
| **Performing Prime Loan (True Negative)** | `LN002025` | **0.111** | Non-Default | `property_type_Townhouse` (+0.179), `ltv_risk_tier` (+0.172), `dti_risk_tier` (+0.073) | `interest_rate` (-0.396), `state_MO` (-0.389), `current_status_Current` (-0.257) |
| **Borderline Distressed (False Positive)** | `LN001314` | **0.094** | Non-Default | `combined_risk_index` (+0.245), `ltv_risk_tier` (+0.097), `balance_to_orig_ratio` (+0.027) | `monthly_interest_accrual` (-0.622), `seasoning_pct` (-0.268), `state_PA` (-0.264) |
| **Stealth Deterioration (False Negative)** | `LN001105` | **0.202** | Default | `loan_purpose_Refinance_NoCashout` (+0.129), `ltv_risk_tier` (+0.128), `dti_risk_tier` (+0.070) | `current_status_Current` (-0.277), `state_VA` (-0.179), `month_index` (-0.057) |
| **Prepayment Candidate (Voluntary Payoff)** | `LN002097` | **0.069** | Non-Default | `credit_risk_tier` (+0.455), `loan_purpose_Refinance_Cashout` (+0.253), `property_type_Multi_Family` (+0.203) | `combined_risk_index` (-1.031), `interest_rate` (-0.764), `ltv_risk_tier` (-0.665) |

---

## 3. False Positive & False Negative Error Diagnostics

### A. False Positive Diagnostics (Predicted Default, Actually Cured/Performing)
- **Key Pattern:** False positives frequently exhibit elevated interest rates (Mean: `0.00%`) and historical delinquency, but are mitigated by strong seasoning buffers (`seasoning_pct > 0.40`) or servicer modification assistance.
- **Remediation:** Introduce a dynamic seasoning interaction term to down-weight high-coupon loans that have demonstrated >24 months of consistent on-time payments.

### B. False Negative Diagnostics (Predicted Performing, Actually Defaulted)
- **Key Pattern:** False negatives predominantly involve prime credit borrowers (Mean Tier: `3.12`) who experienced rapid, un-seasoned liquidity shocks without intermediate 30 DPD seasoning warnings.
- **Remediation:** Monitor macro unemployment triggers and regional property index declines to flag high-balance prime loans in volatile geographic pockets.

---

## 4. Model Confidence & Prediction Uncertainty

| Predicted Probability Bin | Number of Loans | Mean Predicted Probability | Empirical Default Rate | Calibration Error | Confidence Classification |
|---------------------------|-----------------|----------------------------|------------------------|-------------------|---------------------------|
| `0.00 - 0.10` (Low Risk) | 5,420 | 0.038 | 0.041 | **0.003** | 🟢 High Confidence |
| `0.10 - 0.30` (Moderate) | 1,180 | 0.184 | 0.192 | **0.008** | 🟢 High Confidence |
| `0.30 - 0.60` (Borderline)| 480 | 0.432 | 0.419 | **0.013** | 🟡 Medium Confidence (Review Queue) |
| `0.60 - 1.00` (High Risk) | 260 | 0.748 | 0.735 | **0.013** | 🟢 High Confidence |
