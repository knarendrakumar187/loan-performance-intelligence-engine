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


