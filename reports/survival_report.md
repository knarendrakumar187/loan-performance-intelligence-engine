# Task 3: Time-to-Event & Competing Risks Survival Modeling Report

**Challenge:** Intain Campus FinTech Challenge 2026 | AI Track  
**Dataset Scope:** 3,000 Unique Origination Loan Cohorts  
**Observation Window:** 1 to 36 Months on Book (Seasoning Duration)  

---

## 1. Executive Summary & Competing Risks Framework

In loan-level credit analytics, standard binary classification models ask *"will a loan default within 12 months?"*, whereas survival models answer *"**when** will the event occur, and what is the trajectory of cumulative risk over the loan lifecycle?"*.

A loan terminates via one of two mutually exclusive competing absorbing states:
1. **Default (Credit Event):** Borrower fails to service debt (`378` loans, **12.6%**).
2. **Prepayment (Voluntary Payoff):** Borrower refinances or pays off balance (`1,672` loans, **55.7%**).
3. **Right-Censored (Active):** Loan remains performing and open at observation cutoff (`950` loans, **31.7%**).

Treating prepayment as ordinary non-event censoring in a naive single-risk survival model severely overestimates cumulative default probabilities. We explicitly implemented a **Competing Risks Framework** via Cumulative Incidence Functions (CIF) and Cause-Specific Hazards.

---

## 2. Competing Risks Cumulative Incidence Curves

The figure below (saved to `reports/figures/survival_cif_competing_risks.png`) depicts the empirical cumulative incidence trajectories $F_k(t) = P(T \le t, \text{Event} = k)$:

- **Prepayment Velocity:** Rises sharply after month 12 as loans season and borrowers capture refinancing opportunities, reaching `55.7%` cumulative incidence by month 36.
- **Default Incidence:** Exhibits classic seasoning ramp peaking between months 12 and 24, leveling off at `12.6%` cumulative incidence.

---

## 3. Segment-Level Stratified Survival Curves

### A. Credit Quality Stratification (Prime vs Subprime)
- **Prime ($Score \ge 700$):** Default-free survival probability remains above `91.4%` through 36 months.
- **Subprime ($Score < 700$):** Experiences accelerated attrition with default-free survival dropping to `78.2%`.
- **Statistical Significance:** Log-Rank Test $\chi^2$ statistic yields **$p = 2.8271 \times 10^{-13}$**, confirming extreme statistical separation between credit quality tiers.
- *Plot reference:* `reports/figures/survival_by_credit_segment.png`

### B. Vintage Cohort Stratification (2022 vs 2023)
- Cohorts originated in 2022 demonstrate higher cumulative seasoning compared to 2023 cohorts, reflecting macroeconomic rate environment shifts.
- *Plot reference:* `reports/figures/survival_by_vintage.png`

---

## 4. Cox Proportional Hazards Model & Covariate Effects

We fitted a semi-parametric Cox Proportional Hazards model:
$$\lambda(t | X) = \lambda_0(t) \exp\left(\beta_1 X_{\text{is\_prime}} + \beta_2 X_{\text{interest\_rate}} + \beta_3 X_{\text{orig\_balance}}\right)$$

### Estimated Hazard Ratios (HR) & Coefficients

| Covariate | Coefficient ($\beta$) | Hazard Ratio ($\exp(\beta)$) | 95% Confidence Interval | Interpretation |
|-----------|-----------------------|------------------------------|-------------------------|----------------|
| `is_prime` | -0.3138 | **0.7306** | [0.612, 0.872] | **26.94% reduction** in default hazard for Prime borrowers ($p < 0.001$). |
| `interest_rate` | +0.4559 | **1.5777** | [1.385, 1.797] | **+57.77% increase** in default hazard for every +1.0% increase in coupon rate. |
| `orig_bal_100k` | +0.0187 | **1.0189** | [0.974, 1.066] | +1.89% default hazard per $100k balance increase (modest effect). |

**Model Concordance Index (C-index):** `0.6214` (measures ranking concordance across censorship times).

---

## 5. Comparison Against Naive Survival Baselines

| Model Architecture | Model Family | Log-Likelihood | AIC | Degrees of Freedom | Key Limitation / Strength |
|--------------------|--------------|----------------|-----|--------------------|---------------------------|
| **Constant Hazard (Exponential)** | Naive Baseline | -1,996.51 | 3,995.01 | 1 | Assumes memoryless constant failure rate $\lambda = 72.37$, which violates empirical mortgage seasoning. |
| **Weibull Accelerated Failure Time** | Parametric | -1,920.61 | 3,845.22 | 2 | Captures monotonic hazard shape; **149.8 AIC point improvement** over constant hazard. |
| **Cox Proportional Hazards** | Semi-Parametric | -1,884.30 | 3,774.60 | 3 | Fully non-parametric baseline hazard with covariate conditioning; best fit. |

---

## 6. Censoring Treatment Summary

- **Administrative Censoring:** Handled transparently by right-censoring indicator $\delta_i = 0$ at the observation boundary (2024-04).
- **Competing Event Handling:** Prepayment events are treated as informative competing termination states rather than uninformative censoring, preserving unbiased default hazard estimates.
