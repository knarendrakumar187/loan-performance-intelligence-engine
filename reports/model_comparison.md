# Task 2: Loan Performance Predictive Modeling Report

**Challenge:** Intain Campus FinTech Challenge 2026 | AI Track  
**Evaluation Methodology:** Strict Time-Aware Chronological Out-of-Time Validation  
**Train Period:** 2022-01 to 2023-10 (`20,015` records, 73.2%)  
**Validation Period:** 2023-11 to 2024-04 (`7,340` records, 26.8%)  

---

## 1. Executive Summary & Model Performance Comparison

We evaluated baseline linear models (Standardized Logistic Regression with balanced weights) against improved tree ensembles (tuned XGBoost with dynamic `scale_pos_weight` and 3-fold Platt probability calibration; LightGBM for multiclass state transition).

### Comprehensive Metrics Table (Out-of-Time Validation)

| Target Variable | Model Family | ROC-AUC | PR-AUC | F1-Score | Brier Score | Recall @ 80% Precision |
|-----------------|--------------|---------|--------|----------|-------------|-------------------------|
| **Next 3M Delinquency** | Baseline (LogReg) | 0.7238 | 0.5699 | 0.5322 | 0.1857 | 0.1420 |
| | **Improved (Calibrated XGBoost)** | **0.7350** | **0.5781** | **0.5320** | **0.1447** | **0.2185** |
| **Next 6M Delinquency** | Baseline (LogReg) | 0.6708 | 0.5810 | 0.5015 | 0.2125 | 0.0864 |
| | **Improved (Calibrated XGBoost)** | **0.7082** | **0.6051** | **0.5620** | **0.1930** | **0.1650** |
| **Next 12M Default** | Baseline (LogReg) | 0.7255 | 0.3768 | 0.3901 | 0.1872 | 0.0450 |
| | **Improved (Calibrated XGBoost)** | **0.8168** | **0.5318** | **0.5244** | **0.1063** | **0.2870** |
| **Next 12M Prepayment** | Baseline (LogReg) | 0.5750 | 0.5585 | 0.5370 | 0.2457 | 0.0210 |
| | **Improved (Calibrated XGBoost)** | **0.7119** | **0.6970** | **0.7093** | **0.2392** | **0.3412** |
| **Next State (Multiclass)** | Baseline (Multinomial LogReg) | — | — | 0.2872 (Macro) | — | — |
| | **Improved (LightGBM Multiclass)** | — | — | **0.3086 (Macro)** | — | — |

---

## 2. Key Takeaways & Model Improvements

1. **Massive Default Discrimination Gains (+0.0913 ROC-AUC):**
   - 12-month default prediction improved from **0.7255 to 0.8168 ROC-AUC** and **0.3768 to 0.5318 PR-AUC**, with Brier score reducing from 0.1872 to **0.1063** (43% error reduction).
   - Tree depth and non-linear interactions between `balance_to_orig_ratio`, `combined_risk_index`, and `dpd_to_age_ratio` proved crucial in identifying distressed loans early.

2. **Prepayment Modeling Breakthrough (+0.1369 ROC-AUC):**
   - 12-month prepayment prediction jumped from **0.5750 to 0.7119 ROC-AUC** and **0.5585 to 0.6970 PR-AUC**.
   - Seasoning curve position (`seasoning_pct`) and interest rate spread are primary drivers for voluntary payoffs.

3. **Probability Calibration Excellence:**
   - Platt Scaling (Sigmoid calibration via 3-fold cross-validation) produced monotonic, well-calibrated reliability curves saved to `reports/figures/calibration_reliability_curves.png`.

---

## 3. Class Imbalance Strategy & Justification

Financial default datasets naturally exhibit strong class imbalance (e.g. 14.04% positive rate for 12-month default).

### Method Chosen: Algorithmic Cost-Sensitive `scale_pos_weight`
$$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{17,414}{2,601} \approx 6.12$$

### Why Over-Sampling (SMOTE) Was Rejected:
- **Synthetic Sample Artifacts:** Synthetic interpolation in high-dimensional mixed tabular data (e.g. binary indicator flags + continuous balances) generates physically impossible financial combinations (e.g. negative amortization without modification).
- **Preserved Risk Calibration:** `scale_pos_weight` directly scales the loss gradient without distorting the empirical feature space distribution, allowing Platt scaling post-processing to recover true posterior probabilities $P(Y=1|X)$.

---

## 4. Leakage Prevention & Audit Confirmation

- **Zero Future Leakage:** Target variables were computed strictly forward in time from simulated loan history.
- **Administrative Columns Dropped:** All post-event flags (`default_flag`, `prepayment_flag`, `loss_severity_band`, `last_updated_at`, `source_system`, `document_status`) were dropped prior to model ingestion.
- **Automated Audit Verified:** Maximum feature-to-target Pearson/Spearman correlation was `< 0.35` across all 69 active features, with zero leakage violations.
