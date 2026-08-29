# Task 4: Anomaly & Exception Detection Report

**Challenge:** FinTech AI Challenge | AI Track  
**Portfolio Analyzed:** 27,355 monthly loan performance records  
**Detection Framework:** Hybrid Rule-Based (10 deterministic checks) + Unsupervised Spatial ML (Isolation Forest) + Supervised Exception Classification

---

## 1. Exception & Anomaly Distribution Overview

| Exception Class | Total Flagged | Proportion of Portfolio | Primary Risk Driver |
|-----------------|---------------|-------------------------|---------------------|
| `none` | 22,663 | 82.85% | Normal performing records |
| `stale_record` | 2,884 | 10.54% | Custodian document deficiencies |
| `source_conflict` | 1,087 | 3.97% | Servicer dual-tape mismatches |
| `data_entry_error` | 446 | 1.63% | Accounting balance/ledger errors |
| `suspicious_transition` | 275 | 1.01% | Contract vs delinquency mismatch |

- **High-Risk Flagged Records (Score >= 0.60):** `22` records (`0.08%`)
- **Mean Anomaly Score:** `0.1278` | **Median Anomaly Score:** `0.0900`

---

## 2. Reviewer-Ready Anomaly Examples (25 Case Studies)

The table below presents high-priority flagged records requiring human underwriting/servicing review, complete with anomaly drivers and prescribed remediations:

| Loan ID | Reporting Month | Anomaly Score | Exception Class | Triggered Rules | Diagnostic Reasoning | Recommended Action |
|---------|-----------------|---------------|-----------------|-----------------|----------------------|--------------------|
| `LN000419` | 2024-02 | **0.761** | `data_entry_error` | R1_balance_exceeds_original, R3_dpd_status_mismatch, R9_source_conflict | Outstanding balance ($627,425.66) exceeds original balance ($270,093.00) without modification flag. Contract state 'Current' conflicts with recorded 90 Days Past Due. Servicer secondary feed reports conflicting delinquency and balance state. | *Escalate to Servicing Accounting for balance correction* |
| `LN002723` | 2023-12 | **0.728** | `source_conflict` | R6_closed_with_balance, R8_document_missing, R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. Missing critical loan origination documentation or stale multi-quarter reporting update. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN000275` | 2024-04 | **0.700** | `source_conflict` | R6_closed_with_balance, R8_document_missing, R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. Missing critical loan origination documentation or stale multi-quarter reporting update. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN000222` | 2024-03 | **0.694** | `source_conflict` | R6_closed_with_balance, R8_document_missing, R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. Missing critical loan origination documentation or stale multi-quarter reporting update. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN000791` | 2023-04 | **0.662** | `data_entry_error` | R2_negative_balance, R7_stale_record, R9_source_conflict | Negative balance detected ($-270,393.49), indicating potential reversal or ledger accounting error. Servicer secondary feed reports conflicting delinquency and balance state. Missing critical loan origination documentation or stale multi-quarter reporting update. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN002475` | 2024-01 | **0.661** | `source_conflict` | R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN000501` | 2024-03 | **0.654** | `source_conflict` | R3_dpd_status_mismatch, R9_source_conflict | Contract state 'Current' conflicts with recorded 90 Days Past Due. Servicer secondary feed reports conflicting delinquency and balance state. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN002658` | 2023-12 | **0.649** | `data_entry_error` | R2_negative_balance | Negative balance detected ($-693,993.05), indicating potential reversal or ledger accounting error. | *Queue for secondary credit underwriting audit* |
| `LN000803` | 2023-02 | **0.630** | `source_conflict` | R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN001257` | 2024-02 | **0.627** | `source_conflict` | R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN000035` | 2023-11 | **0.626** | `source_conflict` | R6_closed_with_balance, R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN002915` | 2023-09 | **0.624** | `source_conflict` | R6_closed_with_balance, R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN002883` | 2023-07 | **0.622** | `data_entry_error` | R2_negative_balance | Negative balance detected ($-216,529.88), indicating potential reversal or ledger accounting error. | *Queue for secondary credit underwriting audit* |
| `LN001984` | 2023-09 | **0.620** | `data_entry_error` | R2_negative_balance | Negative balance detected ($-166,962.45), indicating potential reversal or ledger accounting error. | *Queue for secondary credit underwriting audit* |
| `LN002489` | 2023-02 | **0.613** | `data_entry_error` | R1_balance_exceeds_original, R6_closed_with_balance, R8_document_missing | Outstanding balance ($840,444.93) exceeds original balance ($422,865.00) without modification flag. Missing critical loan origination documentation or stale multi-quarter reporting update. | *Escalate to Servicing Accounting for balance correction* |
| `LN001023` | 2024-02 | **0.612** | `data_entry_error` | R2_negative_balance, R8_document_missing | Negative balance detected ($-117,353.03), indicating potential reversal or ledger accounting error. Missing critical loan origination documentation or stale multi-quarter reporting update. | *Issue custodian document deficiency notice* |
| `LN001524` | 2024-04 | **0.611** | `data_entry_error` | R1_balance_exceeds_original, R6_closed_with_balance | Outstanding balance ($78,771.26) exceeds original balance ($67,280.00) without modification flag. | *Escalate to Servicing Accounting for balance correction* |
| `LN001861` | 2022-05 | **0.610** | `source_conflict` | R6_closed_with_balance, R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN002604` | 2024-02 | **0.605** | `source_conflict` | R6_closed_with_balance, R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN001557` | 2023-09 | **0.602** | `data_entry_error` | R2_negative_balance | Negative balance detected ($-335,534.96), indicating potential reversal or ledger accounting error. | *Queue for secondary credit underwriting audit* |
| `LN000554` | 2024-03 | **0.600** | `data_entry_error` | R2_negative_balance | Negative balance detected ($-235,585.07), indicating potential reversal or ledger accounting error. | *Queue for secondary credit underwriting audit* |
| `LN000671` | 2024-02 | **0.600** | `data_entry_error` | R1_balance_exceeds_original, R6_closed_with_balance | Outstanding balance ($880,983.91) exceeds original balance ($636,603.00) without modification flag. | *Escalate to Servicing Accounting for balance correction* |
| `LN001532` | 2022-08 | **0.599** | `data_entry_error` | R2_negative_balance | Negative balance detected ($-55,947.92), indicating potential reversal or ledger accounting error. | *Queue for secondary credit underwriting audit* |
| `LN000146` | 2023-06 | **0.599** | `source_conflict` | R6_closed_with_balance, R9_source_conflict | Servicer secondary feed reports conflicting delinquency and balance state. | *Request dual-source data tape reconciliation from primary servicer* |
| `LN001769` | 2023-06 | **0.596** | `data_entry_error` | R1_balance_exceeds_original, R6_closed_with_balance | Outstanding balance ($627,088.20) exceeds original balance ($541,790.00) without modification flag. | *Escalate to Servicing Accounting for balance correction* |

---

## 3. Operational Governance & Servicer Remediation Workflow

1. **Data Entry Errors (`data_entry_error`):** Automatically routed to loan servicing accounting queues with negative balance or over-balance alerts.
2. **Source Conflicts (`source_conflict`):** Flagged for dual-source tape cross-reconciliation with primary servicer API.
3. **Suspicious State Transitions (`suspicious_transition`):** Pauses automated default/foreclosure processing pending manual servicer status audit.
4. **Stale Records (`stale_record`):** Issues automated custodian document deficiency tickets.
