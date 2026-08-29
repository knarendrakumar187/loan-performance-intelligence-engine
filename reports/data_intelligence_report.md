# Task 1: Data Intelligence & Profiling Report

**Challenge:** FinTech AI Challenge | AI Track  
**Dataset Dimensions:** Train = 27,355 rows × 33 columns | Test = 6,930 rows × 26 columns  
**Observation Period:** 2022-01 to 2024-04 (Train) vs 2024-05 to 2026-11 (Test)

---

## 1. Data Quality Index (DQI) & Summary Scores

The composite Data Quality Index is computed from three weighted pillars: **Completeness (35%)**, **Validity (35%)**, and **Consistency (30%)**.

| Quality Metric | Score / Value | Status / Description |
|----------------|---------------|----------------------|
| **Overall Mean DQI** | **97.66 / 100** | Weighted overall portfolio quality |
| **Median DQI** | 98.65 / 100 | Robust central tendency |
| **Completeness Pillar** | 95.72% | Non-null feature rate across schema |
| **Validity Pillar** | 99.66 / 100 | Adherence to value ranges & positive balances |
| **Consistency Pillar** | 97.59 / 100 | Cross-field logic (DPD vs status, balance <= orig) |
| **Clean Records (DQI >= 95)** | 91.30% | Zero or negligible anomaly flags |

### Quality Grade Breakdown

| Grade | Threshold | Proportion of Portfolio |
|-------|-----------|-------------------------|
| Grade A | >= 90 | 91.31% |
| Grade B | >= 80 | 8.62% |
| Grade C | >= 70 | 0.07% |

---

## 2. Missingness Profiling & Pattern Analysis

| Column Name | Missing Count | Missing Percentage | Imputation Strategy |
|-------------|---------------|--------------------|---------------------|
| `current_balance` | 684 | 2.50% | Median + Indicator |
| `interest_rate` | 690 | 2.52% | Median + Indicator |
| `credit_score_band` | 681 | 2.49% | Missing Category / Mode |
| `ltv_band` | 679 | 2.48% | Missing Category / Mode |
| `dti_band` | 675 | 2.47% | Missing Category / Mode |
| `loss_severity_band` | 26,977 | 98.62% | Missing Category / Mode |
| `next_3m_delinquency_flag` | 2,117 | 7.74% | Missing Category / Mode |
| `next_6m_delinquency_flag` | 2,117 | 7.74% | Missing Category / Mode |
| `next_12m_default_flag` | 2,117 | 7.74% | Missing Category / Mode |
| `next_12m_prepayment_flag` | 2,117 | 7.74% | Missing Category / Mode |
| `next_state` | 2,117 | 7.74% | Missing Category / Mode |

---

## 3. Outlier Detection & Business Rule Violations

| Rule ID & Name | Violation Count | Portfolio % | Severity | Exception Class |
|----------------|-----------------|-------------|----------|-----------------|
| `rule_balance_gt_original` | 214 | 0.78% | Medium | Data Quality / Rule Exception |
| `rule_negative_balance` | 232 | 0.85% | High | Data Quality / Rule Exception |
| `rule_dpd_status_mismatch` | 340 | 1.24% | Medium | Data Quality / Rule Exception |
| `rule_negative_remaining_term` | 0 | 0.00% | High | Data Quality / Rule Exception |
| `rule_future_origination` | 0 | 0.00% | High | Data Quality / Rule Exception |
| `rule_closed_with_balance` | 1,609 | 5.88% | Medium | Data Quality / Rule Exception |
| `rule_stale_record` | 270 | 0.99% | Medium | Data Quality / Rule Exception |
| `rule_document_missing` | 2,782 | 10.17% | Medium | Data Quality / Rule Exception |

**Total Records Flagged for Review:** `7,346` records (`26.85%` of train).

---

## 4. Feature Correlations & Categorical Associations

- **Pearson Correlation Heatmap:** Saved to `reports/figures/correlation_matrix.png`
- **Cramér's V Association Matrix:** Saved to `reports/figures/cramers_v_matrix.png`
- **DPD vs Status Relationship Breaks:** `340` occurrences
- **Balance Exceeding Original (Unmodified):** `216` occurrences

---

## 5. Train vs. Test Drift Assessment

Drift evaluated using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) tests** for numeric features, and **Chi-Squared tests** for categorical features.

| Feature | Metric Type | Drift Metric (PSI / Stat) | p-value | Drift Classification |
|---------|-------------|---------------------------|---------|----------------------|
| `month_index` | Numeric PSI | 4.3146 | 0.0 | 🔴 Severe (PSI >= 0.25) |
| `loan_age_months` | Numeric PSI | 4.3146 | 0.0 | 🔴 Severe (PSI >= 0.25) |
| `remaining_term_months` | Numeric PSI | 3.0098 | 0.0 | 🔴 Severe (PSI >= 0.25) |
| `original_balance` | Numeric PSI | 0.0023 | 0.0667 | 🟢 Stable (PSI < 0.10) |
| `current_balance` | Numeric PSI | 0.0051 | 0.0038 | 🟢 Stable (PSI < 0.10) |
| `interest_rate` | Numeric PSI | 0.0436 | 0.0 | 🟢 Stable (PSI < 0.10) |
| `days_past_due` | Numeric PSI | 0.0182 | 0.0 | 🟢 Stable (PSI < 0.10) |
| `modification_flag` | Numeric PSI | 0.0000 | 1.0 | 🟢 Stable (PSI < 0.10) |
| `prepayment_flag` | Numeric PSI | 0.0000 | 0.6863 | 🟢 Stable (PSI < 0.10) |
| `default_flag` | Numeric PSI | 0.0000 | 0.7775 | 🟢 Stable (PSI < 0.10) |
| `reporting_month` | Categorical Chi2 | 20487708.36 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `origination_month` | Categorical Chi2 | 45411.86 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `credit_score_band` | Categorical Chi2 | 655.53 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `ltv_band` | Categorical Chi2 | 364.83 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `dti_band` | Categorical Chi2 | 183.10 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `state` | Categorical Chi2 | 1405.07 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `loan_purpose` | Categorical Chi2 | 101.75 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `occupancy_type` | Categorical Chi2 | 94.90 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `property_type` | Categorical Chi2 | 15.52 | 1.4000e-03 | 🟡 Statistically Significant Shift |
| `servicer_name` | Categorical Chi2 | 187.73 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `current_status` | Categorical Chi2 | 730.24 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `loss_severity_band` | Categorical Chi2 | 96.09 | 0.0000e+00 | 🟡 Statistically Significant Shift |
| `source_system` | Categorical Chi2 | 6.67 | 9.8000e-03 | 🟡 Statistically Significant Shift |
| `document_status` | Categorical Chi2 | 15.61 | 1.4000e-03 | 🟡 Statistically Significant Shift |

---

## 6. Recommendations for Downstream ML Pipeline (Task 2)

1. **Time-Aware Split:** Enforce chronological train/validation/test split by `reporting_month`.
2. **Leakage Controls:** Explicitly drop `default_flag`, `prepayment_flag`, `loss_severity_band`, and `last_updated_at` before model training.
3. **Missing Value Imputation:** Apply median imputation with missing indicators for numeric features; create an explicit `'Missing'` category for categorical bands.
4. **Target Seasoning:** Compute historical delinquency momentum and balance decay rate to assist tree ensembles in capturing early credit deterioration.
