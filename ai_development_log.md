# AI Development Log

**Project:** Loan Performance Intelligence Engine
**Challenge:** Intain Campus FinTech Challenge 2026 — AI Track

This log documents all AI tool usage throughout the development of this project, as required by Task 8 of the problem statement.

---

## Session 1 — Project Scaffold (2026-08-29)

### AI Tool Used
- **Model:** Google Gemini (Antigravity IDE agent)
- **Purpose:** Project scaffolding, directory structure, README, configuration

### Representative Prompt
```
Build a non-LLM-first ML system for the Loan Performance Intelligence Engine.
Start with Phase 0: Repo structure, README, requirements.txt, synthetic-data generator
matching the exact schemas from the problem statement.
```

### What Was Accepted
- Directory structure following modular ML project conventions
- .gitignore covering Python, data files, model checkpoints
- README template with setup instructions and reproducibility notes
- CHANGELOG format
- Configuration module with centralized paths and random seeds

### What Was Rejected / Modified
- (None yet — first commit is standard scaffold)

### Human Review Process
- Reviewed implementation plan before approving code generation
- Verified directory structure covers all 7 required tasks
- Confirmed .gitignore won't accidentally include large data files

### Approximate AI-Generated Code Share
- **This commit:** ~95% AI-generated (scaffold is boilerplate)
- Human contribution: problem statement analysis, architectural decisions, approval

### Lessons Learned
- Starting with a detailed implementation plan before any code prevents wasted commits
- Pinning dependency versions from the start avoids reproducibility issues later

---

## Session 2 — Synthetic Data & Environment Setup (2026-08-29)

### AI Tool Used
- **Model:** Gemini 3.7 Flash
- **Purpose:** Dependency resolution, synthetic data generator optimization, schema validation

### Representative Prompt
```
Generate synthetic datasets matching the exact problem statement schemas:
- loan_monthly_performance_train.csv
- loan_monthly_performance_test.csv
- loan_static_attributes.csv
- servicer_updates.csv
- macro_scenarios.csv
- data_dictionary.md
- validation_rules.json
```

### What Was Accepted
- Vectorized target computation and deliberate anomaly injection (sub-10s generation for 34k+ records)
- Time-aware 80/20 train/test cohort split by cumulative reporting month
- Comprehensive data dictionary and validation rules covering 10 business logic checks

### What Was Rejected / Modified
- Rejected row-by-row `df.loc` loops in favor of vectorized numpy indexing for instant dataset generation
- Adjusted calendar observation window to 2022-2024 to create realistic seasoning curves across 24 origination cohorts

### Approximate AI-Generated Code Share
- ~90% AI-generated, 10% human architectural and schema oversight

### Lessons Learned
- Computing forward-looking target flags directly during per-loan history simulation is mathematically leakage-free and orders of magnitude faster than post-hoc dataframe grouping.

---

## Session 3 — Task 1: Data Intelligence & Profiling (2026-08-29)

### AI Tool Used
- **Model:** Gemini 3.7 Flash
- **Subagents:** `codewriter` (Pro model)
- **Purpose:** Full statistical profiling, outlier detection, PSI drift computation, composite DQI scoring engine, and automated markdown report generation.

### Representative Prompt
```
Build the complete data profiling module covering:
- Column distributions (skewness, kurtosis, percentiles)
- Missingness pattern analysis & MCAR approximation
- Domain-specific outlier & rule violation detection
- Pearson correlation & Cramér's V associations
- Train vs Test drift (PSI, KS test, Chi-squared)
- Record-level & batch-level Data Quality Index (DQI 0-100)
- reports/data_intelligence_report.md
```

### What Was Accepted
- Weighted 3-pillar Data Quality Index (Completeness 35%, Validity 35%, Consistency 30%) with individual loan grades (A/B/C/D/F).
- Population Stability Index (PSI) calculation with quantile bins and categorical Chi-Squared stability tests.
- Generation of 14 visual distribution, correlation, and drift summary plots in `reports/figures/`.

### What Was Rejected / Modified
- Corrected placeholder column names (`balance`, `dpd`) to exact schema names (`current_balance`, `original_balance`, `days_past_due`).
- Vectorized outlier and rule checks with numpy/pandas boolean masks to support fast scaling.

### Approximate AI-Generated Code Share
- ~90% AI-generated, 10% human review & domain constraint tuning

### Lessons Learned
- Visualizing distribution drift (e.g. `loan_age_months` naturally drifting as cohorts mature) provides vital context: demographic features remain stable (PSI < 0.05) while age features reflect temporal progression.

---

## Session 4 — Task 2: Loan Performance Predictive Modeling (2026-08-29)

### AI Tool Used
- **Model:** Gemini 3.7 Flash
- **Subagents:** `codewriter` (Pro model)
- **Purpose:** Leakage-safe feature engineering, automated correlation audit, time-aware chronological validation, baseline Logistic Regression, calibrated XGBoost/LightGBM model training, and performance comparison reporting.

### Representative Prompt
```
Build the complete predictive modeling pipeline for Task 2 (20 pts):
- Engineer domain features (ratios, delinquency indicators, risk index, seasoning)
- Run automated target leakage audit (correlation threshold > 0.90 check)
- Strict chronological time-aware split (2022-01 to 2023-10 train, 2023-11 to 2024-04 val)
- Train baseline Logistic Regression with StandardScaler
- Train improved Calibrated XGBoost with scale_pos_weight for all binary targets
- Train LightGBM multiclass classifier for next_state
- Output reports/model_comparison.md and calibration reliability curves
```

### What Was Accepted
- Strict time-aware splitting preventing cross-cohort multi-record loan leakage across folds.
- Cost-sensitive `scale_pos_weight` weighting coupled with 3-fold Platt probability calibration.
- Reaching **0.8168 ROC-AUC on 12M Default** and **0.7119 ROC-AUC on Prepayment**.
- Side-by-side metric comparison table covering ROC-AUC, PR-AUC, F1, Brier Score, and Recall@80% precision.

### What Was Rejected / Modified
- Dropped raw categorical strings (`credit_score_band`, etc.) in favor of ordinal numeric risk tiers (`credit_risk_tier`, etc.) to prevent string-to-float conversion errors in scikit-learn models.
- Standardized linear features via `StandardScaler` in baseline pipeline to prevent L-BFGS convergence failures.
- Computed precision-recall curve threshold optimization alongside standard 0.50 cutoff for heavily imbalanced default labels.

### Approximate AI-Generated Code Share
- ~92% AI-generated, 8% human tuning of hyperparameter bounds and threshold evaluation

### Lessons Learned
- Default prediction in financial panels benefits disproportionately from non-linear interaction terms and Platt calibration, reducing Brier calibration error by over 43% relative to uncalibrated baselines.

---

## Session 5 — Task 3: Time-to-Event / Survival Modeling (2026-08-29)

### AI Tool Used
- **Model:** Gemini 3.7 Flash
- **Purpose:** Time-to-event aggregation, competing risks Cumulative Incidence Functions (CIF), Kaplan-Meier curve stratification, Cox Proportional Hazards regression, and baseline parametric hazard benchmarking.

### Representative Prompt
```
Implement the complete survival modeling suite for Task 3 (15 pts):
- Competing risks model (Default vs Prepayment vs Right-Censoring)
- Kaplan-Meier curves stratified by credit band (Prime vs Subprime) and origination vintage
- Statistical significance testing via Log-Rank test
- Cox PH hazard ratios for key covariates (is_prime, interest_rate, original_balance)
- Naive baseline comparison (Constant Hazard exponential vs Weibull vs Cox PH)
- Save figures and generate reports/survival_report.md
```

### What Was Accepted
- Competing risks formulation modeling voluntary prepayment (55.7%) alongside involuntary default (12.6%) to prevent biased single-risk default overestimation.
- Stratification revealing extreme divergence between Prime and Subprime default-free survival ($p = 2.8271 \times 10^{-13}$).
- Parametric comparisons showing 149.8 AIC point superiority for Weibull over memoryless constant hazard.

### What Was Rejected / Modified
- Rejected single-risk Kaplan-Meier modeling where prepayments are treated as non-informative censoring, replacing it with dual Cumulative Incidence Functions.

### Approximate AI-Generated Code Share
- ~90% AI-generated, 10% human statistical verification

### Lessons Learned
- Prepayment represents a major competing absorbing event in mortgage panels; failing to model competing risks leads to statistically biased risk estimates.

---

## Session 6 — Task 4: Anomaly & Exception Detection (2026-08-29)

### AI Tool Used
- **Model:** Gemini 3.7 Flash
- **Purpose:** Deterministic business rule validation, servicer secondary tape cross-matching, unsupervised Isolation Forest spatial density modeling, supervised Random Forest exception classification, plain-English diagnostic driver generation, and markdown reporting.

### Representative Prompt
```
Build the complete anomaly and exception detection engine for Task 4 (10 pts):
- Combine 10 deterministic validation rules with an unsupervised Isolation Forest score
- Predict exception_type (data_entry_error, stale_record, source_conflict, suspicious_transition, none)
- Produce >= 20 reviewer-ready anomaly case studies with plain-English diagnostic reasoning and remediation actions
- Output reports/anomaly_report.md
```

### What Was Accepted
- Hybrid composite score formulation: $0.45 \times \text{rule\_score} + 0.55 \times \text{isolation\_forest\_score}$.
- Generation of 25 detailed reviewer-ready anomaly case studies with individual loan IDs, reporting months, anomaly scores, triggered rules, plain-English diagnostic explanations, and prescribed servicer remediations.
- Automated servicer dual-source reconciliation against `servicer_updates.csv`.

### What Was Rejected / Modified
- Rejected pure rule-based thresholding in favor of hybrid rule + ML scoring to capture subtle multivariate balance anomalies that individual threshold rules miss.

### Approximate AI-Generated Code Share
- ~92% AI-generated, 8% human underwriting policy calibration

### Lessons Learned
- Rule-based systems excel at catching explicit accounting violations (e.g. negative balances), while Isolation Forests detect multivariate distributional outliers; combining them provides superior coverage.

---

## Session 7 — Task 5: Scenario & Stress Simulation (2026-08-29)

### AI Tool Used
- **Model:** Gemini 3.7 Flash
- **Purpose:** Macroeconomic stress scenario shock transmission, portfolio and segment-level (credit tier, vintage cohort, geographic state) projection engine, comparative bar chart visualization, and markdown reporting.

### Representative Prompt
```
Build the scenario and stress simulation engine for Task 5 (10 pts):
- Apply Base, Adverse Credit (+150 bps, 1-band downgrade), and High Prepayment (-100 bps) shocks
- Re-score portfolio through calibrated XGBoost models
- Output segment-level projected delinquency, default, and prepayment rates
- Generate comparative visualization reports/figures/scenario_stress_comparison.png
- Output reports/scenario_report.md
```

### What Was Accepted
- Non-linear transmission showing 12M default rate jumping from 14.83% (Base) to 33.91% (Adverse Credit).
- Segment-level granular projections breaking down stress vulnerabilities by credit score band, origination vintage, and top 5 property states.
- High prepayment scenario capturing 100% voluntary payoff acceleration across prime cohorts under lower rate regimes.

### What Was Rejected / Modified
- Rejected simple proportional scaling of overall rates in favor of feeding shocked borrower-level feature vectors through trained non-linear tree models and then applying macroeconomic risk multipliers.

### Approximate AI-Generated Code Share
- ~92% AI-generated, 8% human stress test calibration

### Lessons Learned
- Feeding feature-level economic shocks through calibrated non-linear models uncovers segment-specific vulnerabilities (e.g., subprime cohorts suffering disproportionately higher default spikes) compared to naive linear portfolio multipliers.






