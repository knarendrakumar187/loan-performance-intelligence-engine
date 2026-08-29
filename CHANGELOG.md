# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - 2026-08-29

### Added
- Complete End-to-End Pipeline Runner (`run_pipeline.py`) orchestrating all 9 phases (0 through 8) in under 75 seconds
- Final Competition Submission (`submission/submission.csv`) with 6,930 calibrated multi-horizon predictions, anomaly flags, SHAP driver attributions, actions, and confidence scores
- Enterprise Model Card (`docs/model_card.md` & `model_card.md`) covering architecture, metrics, fair lending (ECOA/Reg B), and operational guidance
- Task 7 Grounded LLM Reviewer Copilot (`src/copilot/reviewer.py`, `reports/copilot_report.md`, `prompt_log.jsonl`)
- Task 6 Explainability Layer (`src/explain/explainer.py`, `reports/explainability_report.md`, SHAP figures)
- Task 5 Scenario & Stress Simulation (`src/scenario/simulator.py`, `reports/scenario_report.md`)
- Task 4 Anomaly & Exception Engine (`src/anomaly/`, `reports/anomaly_report.md`)
- Task 3 Time-to-Event Survival Engine (`src/survival/`, `reports/survival_report.md`)
- Task 2 Predictive Modeling Suite (`src/models/`, `reports/model_comparison.md`)
- Task 1 Data Intelligence Suite (`src/profiling/`, `reports/data_intelligence_report.md`)
- Task 0 Project Scaffold & Synthetic Data Engine (`src/data/`)
- Macroeconomic scenario simulation engine (`src/scenario/simulator.py`) and comparative stress visualization
- Task 4 Anomaly & Exception Detection Report (`reports/anomaly_report.md`) featuring 25 reviewer-ready case studies with diagnostic plain-English reasoning and remediation prescriptions
- Hybrid anomaly detection module (`src/anomaly/`) combining 10 deterministic validation rules, servicer tape conflict cross-matching, unsupervised Isolation Forest spatial density scoring, and Random Forest exception classifier
- Task 3 Time-to-Event & Survival Modeling Report (`reports/survival_report.md`) detailing competing risks (default vs prepayment), Kaplan-Meier stratification by credit and vintage, Cox PH hazard ratios, and exponential/Weibull baseline comparisons
- Survival modeling engine (`src/survival/competing_risks.py`) and naive baseline hazard benchmarks (`src/survival/baseline_hazard.py`)
- Task 2 Model Performance & Comparison Report (`reports/model_comparison.md`) with comprehensive evaluation table across 5 targets (ROC-AUC, PR-AUC, F1, Brier, Recall@80% precision) and calibration reliability analysis
- Improved gradient boosted models (`src/models/improved.py`) using calibrated XGBoost with `scale_pos_weight` and LightGBM multiclass transition models achieving 0.8168 ROC-AUC on 12M default and 0.7119 on prepayment
- Baseline linear models (`src/models/baseline.py`) with StandardScaler pipelines
- Time-aware chronological validation engine (`src/models/splitter.py`)
- Leakage-safe feature engineering pipeline (`src/features/engineer.py`) and automated correlation audit (`src/features/leakage_audit.py`)
- Complete Data Intelligence & Profiling module (`src/profiling/`) with distribution analysis, missingness pattern detection, outlier & domain-rule flagging, relationship checks, drift tracking, and composite DQI scoring
- Pinned requirements with XGBoost, LightGBM, Lifelines, SHAP, Seaborn
- Synthetic data generator (`src/data/synthesize.py`) matching problem statement schema with ~34k monthly performance panel records, 3k loan attributes, servicer updates, macro scenarios, validation rules, and data dictionary
- Time-aware 80/20 train/test split by reporting month
- Initial project scaffold with directory structure, README, CHANGELOG, AI development log
