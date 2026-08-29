# Data Dictionary — Loan Performance Intelligence Engine

## Monthly Performance Fields

| Field | Type | Description |
|-------|------|-------------|
| loan_id | string | Unique identifier for each loan (format: LN######) |
| month_index | int | Sequential month number within loan's observation window (1-based) |
| reporting_month | string | Calendar month of observation (YYYY-MM format) |
| origination_month | string | Month when the loan was originated (YYYY-MM format) |
| loan_age_months | int | Number of months since origination |
| remaining_term_months | int | Remaining months until maturity |
| original_balance | float | Loan balance at origination (USD) |
| current_balance | float | Outstanding principal balance as of reporting month (USD) |
| interest_rate | float | Current annual interest rate (%) |
| credit_score_band | string | Borrower credit score range at origination (<620, 620-659, 660-699, 700-739, 740-779, 780+) |
| ltv_band | string | Loan-to-value ratio range at origination (<60, 60-70, 70-80, 80-90, 90-95, >95) |
| dti_band | string | Debt-to-income ratio range (<20, 20-30, 30-40, 40-50, >50) |
| state | string | US state where the property is located (2-letter code) |
| loan_purpose | string | Purpose of the loan (Purchase, Refinance_Cashout, Refinance_NoCashout) |
| occupancy_type | string | How the property is occupied (Primary, Second_Home, Investment) |
| property_type | string | Type of property (Single_Family, Condo, Townhouse, Multi_Family) |
| servicer_name | string | Current loan servicer (ServicerA through ServicerE) |
| current_status | string | Loan performance status (Current, 30DPD, 60DPD, 90DPD, Default, Prepaid) |
| days_past_due | int | Number of days the loan payment is overdue |
| modification_flag | int | Whether the loan has been modified (0=No, 1=Yes) |
| prepayment_flag | int | Whether the loan has been prepaid in full (0=No, 1=Yes) |
| default_flag | int | Whether the loan has defaulted (0=No, 1=Yes) |
| loss_severity_band | string | Loss severity category if defaulted (None, Low, Medium, High) |
| last_updated_at | string | Date when this record was last updated |
| source_system | string | System from which this data was sourced (SystemA, SystemB) |
| document_status | string | Status of loan documentation (Complete, Partial, Missing, Stale) |

## Target Variables

| Field | Type | Description |
|-------|------|-------------|
| next_3m_delinquency_flag | int | Will the loan become delinquent (30+ DPD) in the next 3 months? (0/1) |
| next_6m_delinquency_flag | int | Will the loan become delinquent (30+ DPD) in the next 6 months? (0/1) |
| next_12m_default_flag | int | Will the loan default in the next 12 months? (0/1) |
| next_12m_prepayment_flag | int | Will the loan prepay in full in the next 12 months? (0/1) |
| next_state | string | What will the loan status be next month? (Current/30DPD/60DPD/90DPD/Default/Prepaid) |
| exception_required | int | Does this record require exception review? (0/1) |
| exception_type | string | Type of exception if flagged (none, data_entry_error, stale_record, source_conflict, suspicious_transition) |

## Servicer Updates Fields

| Field | Type | Description |
|-------|------|-------------|
| loan_id | string | Loan identifier matching the main performance file |
| month_index | int | Month index matching the main performance file |
| current_balance | float | Balance reported by servicer (may differ from main file) |
| current_status | string | Status reported by servicer (may differ from main file) |
| days_past_due | int | DPD reported by servicer (may differ from main file) |
| servicer_name | string | Reporting servicer |
| source_system | string | Always SystemB for servicer updates |
| update_timestamp | string | When the servicer submitted this update |
| is_conflict | int | Whether this record conflicts with the main performance file (0/1) |

## Macro Scenario Fields

| Field | Type | Description |
|-------|------|-------------|
| scenario_name | string | Scenario identifier (base, adverse_credit, high_prepayment) |
| description | string | Plain-English description of the scenario |
| credit_score_shift | int | Number of credit bands to shift (negative = downgrade) |
| interest_rate_shift | float | Basis point shift to interest rates |
| default_rate_multiplier | float | Multiplier applied to baseline default probability |
| prepayment_rate_multiplier | float | Multiplier applied to baseline prepayment probability |
