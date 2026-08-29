# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Pinned requirements with XGBoost, LightGBM, Lifelines, SHAP, Seaborn
- Synthetic data generator (`src/data/synthesize.py`) matching problem statement schema with ~34k monthly performance panel records, 3k loan attributes, servicer updates, macro scenarios, validation rules, and data dictionary
- Time-aware 80/20 train/test split by reporting month
- Initial project scaffold with directory structure, README, CHANGELOG, AI development log
