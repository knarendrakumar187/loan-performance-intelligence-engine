# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
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
