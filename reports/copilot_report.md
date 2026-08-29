# Task 7: Grounded LLM Reviewer Copilot Report

**Challenge:** FinTech AI Challenge | AI Track  
**Grounding Sources:** Data Dictionary + TreeSHAP Feature Attribution + Model Calibration Confidence  
**Prompt Audit Log:** `prompt_log.jsonl` (8 entries)  

---

## 1. Copilot Architecture & Grounding Protocol

The Reviewer Copilot generates structured credit review notes that are **strictly grounded** in three verified data sources:

1. **Data Dictionary** (`data/raw/data_dictionary.md`): Constrains field references to documented schema fields only.
2. **TreeSHAP Feature Attribution**: Every risk assessment claim must cite a specific SHAP contribution value.
3. **Model Prediction Probabilities**: Calibrated XGBoost posterior probabilities with stated confidence bands.

### Hallucination Prevention Rules
- **Rule 1**: Only reference fields in the Data Dictionary.
- **Rule 2**: Every risk claim must cite a SHAP value.
- **Rule 3**: No speculation about borrower intent, market conditions, or external events.
- **Rule 4**: Missing fields must be stated as unavailable, never imputed.
- **Rule 5**: No external data source references.

---

## 2. Generated Reviewer Notes (5 Loan Case Studies)

## Reviewer Note — Loan `LN000421`
**Generated:** 2026-08-29T13:34:27.076396+00:00  
**Status:** N/A | **DPD:** 0 | **Balance:** $154,151.05 / $157,084.00  
**Credit Band:** N/A

### 1. Risk Assessment Summary
Loan `LN000421` has a predicted 12-month default probability of **10.6%** and prepayment probability of **53.6%**. 
The primary risk driver is `original_balance` (SHAP: +0.1728), which elevates the default hazard above the portfolio median.

### 2. Key Risk Factors
- **`original_balance`** (SHAP: +0.1728): Elevates default risk
- **`current_balance`** (SHAP: +0.1439): Elevates default risk
- **`loan_purpose_Refinance_NoCashout`** (SHAP: +0.0524): Elevates default risk

### 3. Mitigating Factors
- **`state_IL`** (SHAP: -0.2925): Reduces default risk
- **`current_status_Current`** (SHAP: -0.2200): Reduces default risk
- **`ltv_risk_tier`** (SHAP: -0.0960): Reduces default risk

### 4. Recommended Action: **APPROVE**

### 5. Confidence Disclaimer
Model confidence classification: **Moderate**. 
All required data fields are present. No data gap warnings.

---

## Reviewer Note — Loan `LN000308`
**Generated:** 2026-08-29T13:34:27.156437+00:00  
**Status:** N/A | **DPD:** 0 | **Balance:** $299,036.96 / $306,062.00  
**Credit Band:** N/A

### 1. Risk Assessment Summary
Loan `LN000308` has a predicted 12-month default probability of **6.2%** and prepayment probability of **53.6%**. 
The primary risk driver is `ltv_risk_tier` (SHAP: +0.1076), which elevates the default hazard above the portfolio median.

### 2. Key Risk Factors
- **`ltv_risk_tier`** (SHAP: +0.1076): Elevates default risk
- **`property_type_Single_Family`** (SHAP: +0.0362): Elevates default risk
- **`servicer_name_ServicerE`** (SHAP: +0.0171): Elevates default risk

### 3. Mitigating Factors
- **`interest_rate`** (SHAP: -0.8622): Reduces default risk
- **`month_index`** (SHAP: -0.4900): Reduces default risk
- **`seasoning_pct`** (SHAP: -0.4623): Reduces default risk

### 4. Recommended Action: **APPROVE**

### 5. Confidence Disclaimer
Model confidence classification: **High**. 
Data unavailable for: next_3m_delinquency_flag, next_6m_delinquency_flag, next_12m_default_flag, next_12m_prepayment_flag, next_state. Predictions for these dimensions carry higher uncertainty.

---

## Reviewer Note — Loan `LN001626`
**Generated:** 2026-08-29T13:34:27.416805+00:00  
**Status:** N/A | **DPD:** 0 | **Balance:** $473,859.12 / $476,762.00  
**Credit Band:** N/A

### 1. Risk Assessment Summary
Loan `LN001626` has a predicted 12-month default probability of **21.6%** and prepayment probability of **53.3%**. 
The primary risk driver is `current_status_Current` (SHAP: +0.9447), which elevates the default hazard above the portfolio median.

### 2. Key Risk Factors
- **`current_status_Current`** (SHAP: +0.9447): Elevates default risk
- **`loan_purpose_Refinance_NoCashout`** (SHAP: +0.0962): Elevates default risk
- **`ltv_risk_tier`** (SHAP: +0.0848): Elevates default risk

### 3. Mitigating Factors
- **`state_CO`** (SHAP: -0.5607): Reduces default risk
- **`combined_risk_index`** (SHAP: -0.2408): Reduces default risk
- **`days_past_due`** (SHAP: -0.2161): Reduces default risk

### 4. Recommended Action: **WATCH_LIST**

### 5. Confidence Disclaimer
Model confidence classification: **Moderate**. 
Data unavailable for: next_3m_delinquency_flag, next_6m_delinquency_flag, next_12m_default_flag, next_12m_prepayment_flag, next_state. Predictions for these dimensions carry higher uncertainty.

---

## Reviewer Note — Loan `LN000688`
**Generated:** 2026-08-29T13:34:27.495615+00:00  
**Status:** N/A | **DPD:** 0 | **Balance:** $143,336.19 / $144,694.00  
**Credit Band:** N/A

### 1. Risk Assessment Summary
Loan `LN000688` has a predicted 12-month default probability of **8.3%** and prepayment probability of **49.0%**. 
The primary risk driver is `servicer_name_ServicerD` (SHAP: +0.0816), which elevates the default hazard above the portfolio median.

### 2. Key Risk Factors
- **`servicer_name_ServicerD`** (SHAP: +0.0816): Elevates default risk
- **`dti_risk_tier`** (SHAP: +0.0810): Elevates default risk
- **`current_balance`** (SHAP: +0.0599): Elevates default risk

### 3. Mitigating Factors
- **`interest_rate`** (SHAP: -0.5300): Reduces default risk
- **`combined_risk_index`** (SHAP: -0.4599): Reduces default risk
- **`current_status_Current`** (SHAP: -0.3036): Reduces default risk

### 4. Recommended Action: **APPROVE**

### 5. Confidence Disclaimer
Model confidence classification: **High**. 
All required data fields are present. No data gap warnings.

---

## Reviewer Note — Loan `LN001349`
**Generated:** 2026-08-29T13:34:27.571484+00:00  
**Status:** N/A | **DPD:** 0 | **Balance:** $234,980.66 / $237,318.00  
**Credit Band:** N/A

### 1. Risk Assessment Summary
Loan `LN001349` has a predicted 12-month default probability of **17.1%** and prepayment probability of **52.6%**. 
The primary risk driver is `ltv_risk_tier` (SHAP: +0.1729), which elevates the default hazard above the portfolio median.

### 2. Key Risk Factors
- **`ltv_risk_tier`** (SHAP: +0.1729): Elevates default risk
- **`loan_purpose_Refinance_NoCashout`** (SHAP: +0.1133): Elevates default risk
- **`monthly_interest_accrual`** (SHAP: +0.0488): Elevates default risk

### 3. Mitigating Factors
- **`current_status_Current`** (SHAP: -0.2672): Reduces default risk
- **`state_PA`** (SHAP: -0.2228): Reduces default risk
- **`current_balance`** (SHAP: -0.1920): Reduces default risk

### 4. Recommended Action: **WATCH_LIST**

### 5. Confidence Disclaimer
Model confidence classification: **Moderate**. 
All required data fields are present. No data gap warnings.

---

## 3. Rejection & Correction Examples (Hallucination Guardrails)

The following examples demonstrate the copilot's ability to detect and reject ungrounded, hallucinated, or imputed outputs:

### Example 1: Rejection: Ungrounded Market Speculation

**❌ Original (Rejected) Output:**
> Based on current Federal Reserve rate hike expectations and rising unemployment in the borrower's region, this loan faces elevated default risk. The housing market downturn in Q3 2024 further compounds prepayment headwinds.

**🔴 Rejection Reason:**
> REJECTED: Output references external economic data (Federal Reserve rates, unemployment statistics, housing market conditions) that are NOT present in the loan record or data dictionary. Grounding Rule #3 violated: 'Do NOT speculate about market conditions not documented in the loan record.'

**✅ Corrected (Grounded) Output:**
> Loan LN001234 has a predicted 12-month default probability of 0.42. The primary risk driver is `days_past_due` (SHAP: +0.1823), indicating current delinquency status elevates default hazard. The interest rate of 7.2% (SHAP: +0.0945) further contributes to repayment stress. No external economic indicators are available in the current dataset to assess macroeconomic transmission effects.

---

### Example 2: Rejection: Hallucinated Borrower Intent

**❌ Original (Rejected) Output:**
> The borrower appears to be strategically defaulting given the negative equity position. Their credit history suggests they may be planning to file for bankruptcy within the next 6 months based on the pattern of missed payments.

**🔴 Rejection Reason:**
> REJECTED: Output imputes borrower intent ('strategically defaulting', 'planning to file for bankruptcy') which is not observable in the data. The data dictionary does not include borrower intent fields. Grounding Rule #3 violated: 'Do NOT speculate about borrower intent not documented in the loan record.'

**✅ Corrected (Grounded) Output:**
> Loan LN002567 shows `current_status` = '90DPD' with `days_past_due` = 90 (SHAP: +0.2134). The `balance_to_orig_ratio` of 1.03 (SHAP: +0.0567) indicates the current balance slightly exceeds the original balance. The model predicts 0.71 default probability. Borrower intent cannot be inferred from available data fields.

---

### Example 3: Rejection: Missing Data Imputation Attempt

**❌ Original (Rejected) Output:**
> Although the credit score band is missing, based on the loan's interest rate of 8.5% and high LTV, the borrower likely falls in the subprime category (<620). The DTI ratio is estimated at approximately 45% given the balance-to-income relationship.

**🔴 Rejection Reason:**
> REJECTED: Output imputes missing field values ('likely falls in subprime', 'estimated at approximately 45%'). Grounding Rule #4 violated: 'If a field is missing or null, state Data unavailable — do NOT impute or guess.' The data dictionary does not include income fields, making DTI estimation impossible.

**✅ Corrected (Grounded) Output:**
> Data unavailable for `credit_score_band`. Data unavailable for DTI estimation (income field not in data dictionary). Risk assessment is based on available fields: `interest_rate` = 8.5% (SHAP: +0.1102), `ltv_band` = '>95' (SHAP: +0.0834). Predictions carry higher uncertainty due to missing credit score grading.

---

## 4. Prompt Audit Trail

All 8 prompt/completion pairs are logged to `prompt_log.jsonl` with:
- ISO 8601 timestamp
- Loan ID reference
- Prompt and completion text (truncated for storage)
- Recommended action classification
- Rejection flag and reason (when applicable)
