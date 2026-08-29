# Loan Performance Intelligence Engine

** Campus FinTech Challenge 2026 — AI Track**

An ML-first system for loan-data profiling, performance prediction, anomaly detection, scenario simulation, explainability, and grounded LLM-assisted review.

---

## Problem Summary

Given messy loan-level data and historical performance, identify which records are unreliable, which loans are likely to deteriorate, and what the portfolio may look like under different future scenarios.

**Key predictions:**

- Next 3-month delinquency
- Next 6-month delinquency
- Next 12-month default
- Next 12-month prepayment
- Next loan state (multiclass)
- Exception detection & anomaly scoring

## Quick Start

```bash
# 1. Clone
git clone https://github.com/knarendrakumar187/loan-performance-intelligence-engine.git
cd loan-performance-intelligence-engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data (or place real data in data/raw/)
python -m src.data.synthesize

# 4. Run full pipeline end-to-end
python run_pipeline.py
```

The pipeline generates:

- `reports/` — Data profiling, model comparison, survival, anomaly, scenario, explainability reports
- `submission/submission.csv` — Final predictions in required format
- `prompt_log.jsonl` — LLM copilot interaction log

## Project Structure

```
├── data/
│   ├── raw/                  # Raw CSV files (synthetic or organizer-provided)
│   ├── processed/            # Feature-engineered datasets
│   └── README.md
├── notebooks/                # Exploratory notebooks
├── src/
│   ├── config.py             # Central configuration
│   ├── data/
│   │   ├── synthesize.py     # Synthetic data generator
│   │   └── loader.py         # Data loading utilities
│   ├── profiling/            # Task 1: Data intelligence & profiling
│   ├── features/             # Leakage-safe feature engineering
│   ├── models/               # Task 2: Prediction models
│   ├── survival/             # Task 3: Survival / hazard models
│   ├── anomaly/              # Task 4: Anomaly & exception detection
│   ├── scenario/             # Task 5: Scenario & stress simulation
│   ├── explain/              # Task 6: SHAP explainability
│   └── copilot/              # Task 7: LLM reviewer copilot
├── reports/                  # Generated reports (MD/HTML)
├── submission/               # Final submission.csv
├── tests/                    # Unit tests
├── docs/                     # Additional documentation
├── model_card.md             # Model card (required deliverable)
├── ai_development_log.md     # AI Development Log (required deliverable)
├── CHANGELOG.md              # Release history
├── requirements.txt          # Pinned dependencies
└── run_pipeline.py           # One-command end-to-end pipeline
```

## Approach Highlights

| Aspect            | Design Choice                   | Why                                                           |
| ----------------- | ------------------------------- | ------------------------------------------------------------- |
| Train/Val split   | Time-aware by `reporting_month` | Prevents same-loan leakage across folds                       |
| Class imbalance   | `scale_pos_weight` in XGBoost   | No synthetic sample artifacts; native to tree ensembles       |
| Survival model    | Discrete-time competing risks   | Captures default vs prepayment as mutually exclusive outcomes |
| Anomaly detection | Rule-based + Isolation Forest   | Rules catch known patterns; ML catches novel anomalies        |
| LLM copilot       | Grounded reviewer notes only    | All predictions from ML; LLM for summarization only           |
| Explainability    | SHAP (global + local)           | Model-agnostic, widely accepted in financial ML               |

## Reproducibility

- All random seeds fixed in `src/config.py`
- Dependencies pinned in `requirements.txt`
- `run_pipeline.py` regenerates everything from raw data
- Time-aware splits documented and deterministic

## Judging Rubric Coverage

| Criterion                        | Points | Status |
| -------------------------------- | ------ | ------ |
| Data Intelligence & Profiling    | 15     | ✅     |
| Predictive Modeling              | 20     | ✅     |
| Survival / Transition Modeling   | 15     | ✅     |
| Anomaly & Exception Detection    | 10     | ✅     |
| Scenario & Stress Simulation     | 10     | ✅     |
| Explainability & Responsible AI  | 10     | ✅     |
| Smart LLM Usage                  | 10     | ✅     |
| ML Engineering & Reproducibility | 5      | ✅     |
| Agentic Coding Evidence          | 5      | ✅     |

## License

This project is submitted for the Intain Campus FinTech Challenge 2026.
