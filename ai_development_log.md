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



