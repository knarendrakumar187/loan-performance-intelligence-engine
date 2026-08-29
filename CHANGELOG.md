# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Task 1 Data Intelligence Report (`reports/data_intelligence_report.md`) covering DQI score (97.66/100), missingness breakdown, business rule violations, correlation/Cramér's V matrices, and PSI/KS drift analysis
- Complete Data Intelligence & Profiling module (`src/profiling/`) with distribution analysis, missingness pattern detection, outlier & domain-rule flagging, relationship checks, drift tracking, and composite DQI scoring
- Pinned requirements with XGBoost, LightGBM, Lifelines, SHAP, Seaborn
- Synthetic data generator (`src/data/synthesize.py`) matching problem statement schema with ~34k monthly performance panel records, 3k loan attributes, servicer updates, macro scenarios, validation rules, and data dictionary
- Time-aware 80/20 train/test split by reporting month
- Initial project scaffold with directory structure, README, CHANGELOG, AI development log
