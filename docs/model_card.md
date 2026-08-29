# Model Card: Loan Performance Intelligence Engine

**Model Name:** Loan Performance Intelligence Engine (LPIE-v1.0)  
**Challenge Track:** FinTech AI Challenge — AI Track  
**Date:** August 2026  
**License:** Apache 2.0  
**Authors:** Hackathon Engineering Team  

---

## 1. Model Details

### 1.1 Overview
The **Loan Performance Intelligence Engine (LPIE)** is an enterprise-grade multi-task machine learning system designed to predict credit deterioration, early delinquency transitions, default hazards, and voluntary prepayments across multi-vintage mortgage portfolios.

### 1.2 Model Architecture & Components

| Task / Subsystem | Primary Architecture | Calibration / Loss Strategy | Primary Output |
|------------------|----------------------|-----------------------------|----------------|
| **Next 3M Delinquency** | Calibrated XGBoost | Platt Scaling (Sigmoid CV) + `scale_pos_weight` | Continuous Probability $[0, 1]$ |
| **Next 6M Delinquency** | Calibrated XGBoost | Platt Scaling (Sigmoid CV) + `scale_pos_weight` | Continuous Probability $[0, 1]$ |
| **Next 12M Default** | Calibrated XGBoost | Platt Scaling (Sigmoid CV) + `scale_pos_weight=6.12` | Continuous Probability $[0, 1]$ |
| **Next 12M Prepayment** | Calibrated XGBoost | Platt Scaling (Sigmoid CV) | Continuous Probability $[0, 1]$ |
| **State Transition** | LightGBM Multiclass | Multi-logloss with class balanced priors | Discrete State Category |
| **Time-to-Event / Survival** | Cox Proportional Hazards + CIF | Cause-Specific Competing Risks | Hazard Ratios & Cumulative Incidence |
| **Anomaly & Exception Engine** | Isolation Forest + Deterministic Rules | Hybrid $0.45 \times \text{Rule} + 0.55 \times \text{ML}$ | Anomaly Score $[0, 1]$ + Exception Type |
| **Explainability Layer** | TreeSHAP Explainer | Local Additive Attribution Values | Feature Contribution $\phi_i$ |
| **Reviewer Copilot** | Grounded Deterministic Template | Data Dictionary & SHAP Constrained | Audit-logged Markdown Reviewer Note |

---

## 2. Intended Use

### 2.1 Primary Intended Uses
- **Portfolio Surveillance:** Monthly automated risk screening across secondary mortgage portfolios.
- **Servicing Triage & Loss Mitigation:** Identifying at-risk borrowers early (3-6 months prior to severe default) to initiate proactive modification or forbearance programs.
- **Prepayment Cash Flow Forecasting:** Estimating prepayment speeds for structured finance cash flow waterfalls.
- **Exception & Reconciliation Auditing:** Flagging accounting data entry errors and servicer dual-tape reporting discrepancies.

### 2.2 Out-of-Scope & Prohibited Uses
- **Automated Point-of-Sale Denials:** This model is designed for portfolio surveillance and secondary market analytics; it must NOT be used as an automated adverse action engine without independent human credit underwriting.
- **Unverified Fair Lending Proxy Ingestion:** Features containing demographic proxies outside documented loan attributes must not be ingested.

---

## 3. Training & Validation Data

### 3.1 Dataset Summary
- **Training Set (Panel):** 27,355 monthly performance records across 3,000 unique loans.
- **Validation Split (Strict Out-of-Time):**
  - **Train Partition:** 20,015 records (`2022-01` to `2023-10`, 73.2%)
  - **Validation Partition:** 7,340 records (`2023-11` to `2024-04`, 26.8%)
- **Test Set:** 6,930 out-of-time monthly records.

### 3.2 Target Leakage Prevention Protocol
The following post-event and administrative columns are strictly quarantined and dropped prior to feature engineering:
- `default_flag` (Direct target leakage)
- `prepayment_flag` (Direct target leakage)
- `loss_severity_band` (Post-event loss resolution indicator)
- `last_updated_at` (Administrative ingestion timestamp)
- `source_system` (Infrastructure metadata)
- `document_status` (Administrative tracking flag)

Automated correlation audit confirmed maximum feature-to-target Pearson correlation $< 0.35$ with 0 leakage violations.

---

## 4. Quantitative Performance Evaluation

### 4.1 Predictive Modeling Performance (Out-of-Time Validation)

| Target Metric | Baseline (LogReg) | Improved (Calibrated XGBoost / LightGBM) | Relative Improvement |
|---------------|-------------------|------------------------------------------|----------------------|
| **Next 12M Default ROC-AUC** | 0.7255 | **0.8168** | **+12.6% (+0.0913)** |
| **Next 12M Default PR-AUC** | 0.3768 | **0.5318** | **+41.1% (+0.1550)** |
| **Next 12M Default Brier Score** | 0.1872 | **0.1063** | **-43.2% (Better Calibration)** |
| **Next 12M Prepayment ROC-AUC** | 0.5750 | **0.7119** | **+23.8% (+0.1369)** |
| **Next 12M Prepayment PR-AUC** | 0.5585 | **0.6970** | **+24.8% (+0.1385)** |
| **Next 3M Delinquency ROC-AUC** | 0.7238 | **0.7350** | **+1.5%** |
| **Next 6M Delinquency ROC-AUC** | 0.6708 | **0.7082** | **+5.6%** |
| **Multiclass Transition Macro-F1** | 0.2872 | **0.3086** | **+7.4%** |

### 4.2 Survival & Hazard Modeling
- **Competing Risks Separation:** Default incidence (12.6%) vs Prepayment incidence (55.7%).
- **Log-Rank Statistical Separation:** $p = 2.8271 \times 10^{-13}$ across credit segments.
- **Cox PH C-Index:** `0.6214` (Prime hazard ratio $HR = 0.731$, coupon rate $HR = 1.578$).
- **Parametric AIC Comparison:** Weibull AIC (`3,845.22`) outperforms Constant Hazard (`3,995.01`) by 149.8 points.

---

## 5. Explainability, Uncertainty & Governance

### 5.1 Top Feature Drivers (Global TreeSHAP)
1. `days_past_due` (Mean |SHAP| = 0.412): Recent delinquency momentum is the strongest single predictor of short-term default.
2. `balance_to_orig_ratio` (Mean |SHAP| = 0.287): High negative amortization or un-amortized balance signals distress.
3. `combined_risk_index` (Mean |SHAP| = 0.215): Interaction of LTV, DTI, and credit risk tiers.
4. `monthly_interest_accrual` (Mean |SHAP| = 0.183): Absolute monthly carrying cost burden.

### 5.2 Prediction Uncertainty Calibration

| Predicted Bin | Count | Empirical Event Rate | Max Calibration Error | Confidence Level |
|---------------|-------|----------------------|-----------------------|------------------|
| `0.00 - 0.10` | 5,420 | 4.1% | 0.3% | High Confidence (Automated Approval) |
| `0.10 - 0.30` | 1,180 | 19.2% | 0.8% | High Confidence (Watchlist) |
| `0.30 - 0.60` | 480 | 41.9% | 1.3% | Medium Confidence (Human Review Queue) |
| `0.60 - 1.00` | 260 | 73.5% | 1.3% | High Confidence (Immediate Intervention) |

---

## 6. Ethical Considerations & Fair Lending

### 6.1 Compliance with Fair Lending Regulations
- **ECOA & Regulation B Alignment:** The model excludes prohibited demographic characteristics (race, gender, marital status, age, national origin).
- **Disparate Impact Monitoring:** State-level and credit band performance metrics are audited continuously to avoid unwarranted regional bias.
- **Adverse Action Explainability:** Every prediction produces localized TreeSHAP feature attributions that map directly to compliant adverse action reason codes.

### 6.2 Limitations & Operational Guidance
- **Extreme Macroeconomic Disruption:** In the event of unprecedented macroeconomic shocks (>300 bps rate hikes or severe economic shutdown), stress scenario simulations (`src/scenario/simulator.py`) should be re-calibrated.
- **Servicer Tape Reconciliation:** Records flagged with `source_conflict` or `anomaly_score > 0.60` must be resolved with primary servicing tapes before downstream risk provisioning.
