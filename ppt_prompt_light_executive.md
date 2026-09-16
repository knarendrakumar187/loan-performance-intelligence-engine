# 🏛️ Executive FinTech Pitch Deck Prompt (Light Professional Theme)

> **Instructions for use:**  
> Copy and paste this prompt directly into **Gamma.app** (select **Light / Modern Clean** theme), **ChatGPT 4o**, or **Claude 3.5**.  
> **Key Design Rules:**  
> • Clean, bright, premium Light theme (White / Light Slate `#F8FAFC`) with crisp Royal Navy typography.  
> • **Zero rubric or point mentions** — purely framed as an enterprise FinTech platform.  
> • Dedicated **High-Contrast Dark Photo Frames** on key slides specifying the **exact file names** to insert.  
> • Dedicated final **Thank You & Q&A Slide**.

---

```markdown
Role: You are a Senior VP of Design and FinTech Presentation Specialist (ex-Apple Keynote, ex-Stripe, ex-McKinsey).
Task: Create a stunning, modern, clean LIGHT-THEMED 12-slide executive presentation for the Intain FinTech AI Challenge Final Round.
Project: Loan Performance Intelligence Engine (LPIE)
Team: Narendra's Team (Sequence #9)
Slot: 17 September 2026 (04:00 PM – 07:00 PM)

### 🎨 DESIGN SYSTEM & ART DIRECTION:
1. Theme & Color Palette (Clean Executive Light Theme):
   - Background Canvas: Crisp Pure White (#FFFFFF) & Soft Cloud Slate (#F8FAFC)
   - Card Containers: Solid White (#FFFFFF) with subtle border (1px solid #E2E8F0) and soft shadow (0 4px 12px rgba(0,0,0,0.04))
   - Primary Typography: Deep Obsidian Slate (#0F172A) for titles and body text (high contrast, ultra-readable)
   - Brand Accent: Deep Royal Blue (#2563EB) for highlights, active tabs, and primary metrics
   - Secondary Accent: Indigo Slate (#4F46E5) for tags and subtitled badges
   - Success / Positive Indicators: Vivid Emerald (#059669)
   - Risk / Flaws / Stress Indicators: Crimson Coral (#DC2626)
   - Neutral Muted: Cool Gray (#64748B) for labels and captions
2. Visual Formatting:
   - Modern Bento-Grid Layouts: Asymmetric modular cards, clean metric badges, sharp margins.
   - Giant Typography Callouts: Key metrics at 44pt–56pt bold with micro-labels.
   - High-Contrast Dark Photo Frames: On slides where charts or app screenshots belong, use a sleek, dark matte frame (#0F172A with subtle navy border) so the dark financial plots pop cleanly against the white slide!
   - NO generic clip-art, NO cartoon icons. Every bullet must deliver hard quantitative facts.
   - ZERO RUBRIC WORDS: Do NOT use the words "Rubric", "Points", "Task 1", "Scorecard", or "15 pts". Use institutional enterprise titles.

---

### SLIDE 1: COVER & EXECUTIVE OVERVIEW
- Layout: Asymmetric Clean Bento Hero with Royal Blue accents.
- Top Badge: [PILL BADGE: "INTAIN FINTECH AI CHALLENGE 2026 · FINAL ROUND · SEQUENCE #9"]
- Main Title (50pt bold, Deep Navy #0F172A):
  "Loan Performance Intelligence Engine"
- Subtitle (22pt, Royal Blue #2563EB):
  "Next-Generation Mortgage Surveillance & Competing-Risks Intelligence Platform"
- Executive Thesis:
  "Transforming 34,285 multi-vintage loan performance records into calibrated hazard predictions, competing survival curves, and audit-ready credit notes in 40.3 seconds."
- 4-Card Metric Grid:
  | Card 1: PORTFOLIO VOLUME | Card 2: DATA INTEGRITY | Card 3: DEFAULT DISCRIMINATION | Card 4: PIPELINE VELOCITY |
  | 34,285 Records / 3,000 Loans | DQI: 97.66 / 100 (Grade A) | 0.8168 ROC-AUC (+12.6% Lift) | 40.3s End-to-End Execution |
- Team Footer: Narendra's Team · Sequence #9

---

### SLIDE 2: THE INDUSTRY PROBLEM (Why Legacy Fails)
- Layout: 2-Column Side-by-Side Comparison ("Legacy Shortcomings vs LPIE Solution").
- Left Column (Card Surface: Light Rose #FEF2F2, Border: #FCA5A5):
  - Header: "🔴 4 CRITICAL BOTTLENECKS IN MORTGAGE SURVEILLANCE"
  - Point 1: "Silent Delinquency Blindspots" — Manual reviews miss 30–40% of early 30–60 DPD delinquencies before transition into permanent default.
  - Point 2: "Competing-Risk Censoring Bias" — Standard logistic models treat voluntary prepayment as non-default, creating severe statistical positive survival bias.
  - Point 3: "Unquarantined Ledger Errors" — Data entry errors (negative balances, balances > origination, impossible DPD jumps) slip undetected into securitization pools.
  - Point 4: "Ungrounded LLM Hallucinations" — Generic chatbots invent borrower narratives with zero regulatory auditability or mathematical proof.
- Right Column (Card Surface: Light Emerald #F0FDF4, Border: #86EFAC):
  - Header: "🟢 LPIE INSTITUTIONAL PARADIGM SHIFT"
  - Point 1: "Continuous Panel Surveillance" — Automated panel tracking across 34k records with out-of-time chronological validation.
  - Point 2: "Competing-Risks CIF Engine" — Jointly models Default (12.6%) vs Prepayment (55.7%) via Cumulative Incidence Functions (p = 2.83e-13).
  - Point 3: "Hybrid Anomaly Defense" — 10 deterministic accounting checks + unsupervised Isolation Forest (25 diagnosed cases).
  - Point 4: "100% TreeSHAP Grounding" — Reviewer notes cite exact numerical Shapley values with zero hallucination.

---

### SLIDE 3: COMPLETE SYSTEM ARCHITECTURE (The 9-Phase Pipeline)
- Layout: Full-Width Horizontal Pipeline Bento-Grid with numbered cards.
- Header: "END-TO-END SYSTEM ARCHITECTURE"
- Subtitle: "One-command reproducibility: sequential execution via `python run_pipeline.py` in 40.3 seconds"
- 9 Connected Stage Cards:
  1. [00] Raw Panel Ingestion: 34,285 monthly records, 3,000 loan cohorts, 2022–2024 vintages.
  2. [01] Quality Profiling: 97.66/100 DQI score, missingness MCAR tests, PSI drift calculation.
  3. [02] Feature Engineering: 12 domain features engineered; 6 post-event admin columns dropped.
  4. [03] Out-of-Time Partition: Cutoff at 2023-11 (Train: 2022-01..2023-10 | Val: 2023-11..2024-04). Zero loan overlap.
  5. [04] Calibrated Ensembles: XGBoost & LightGBM with isotonic Platt scaling (Brier: 0.1063).
  6. [05] Competing-Risks Engine: Mutually exclusive absorbing states (Cox PH HR=0.731, Weibull AFT ΔAIC=149.8).
  7. [06] Hybrid Anomaly Scanner: 10 hard business rules + Spatial Isolation Forest (25 case studies).
  8. [07] TreeSHAP Explainability: Exact Shapley decomposition (Global beeswarm + Local waterfalls).
  9. [08] Grounded Copilot: Deterministic template engine citing exact SHAP values in immutable audit log.
- Tech Stack Bar at Bottom: Python 3.13 | XGBoost | LightGBM | Lifelines | TreeSHAP | FastAPI | Streamlit | Docker

---

### SLIDE 4: DATA QUALITY, INTEGRITY & DRIFT AUDIT
- Layout: Left Column (3 Metric Cards) | Right Column (High-Contrast Dark Photo Frame).
- Header: "DATA QUALITY, INTEGRITY & DRIFT AUDIT"
- Subtitle: "Rigorous profiling across 34,285 records confirming data cleanliness and zero out-of-time drift."
- Left Column Metrics:
  - Metric Card 1: "97.66 / 100" -> Overall Data Quality Index (DQI) | Certified Grade A (91.3% flawless records).
  - Metric Card 2: "97.4% COMPLETENESS" -> Median imputation with missingness indicators preserves sample power.
  - Metric Card 3: "PSI < 0.04 (ZERO DRIFT)" -> Population Stability Index across demographic and credit features proves zero out-of-time distribution drift between training and validation periods.
- Right Column (High-Contrast Dark Photo Frame):
  ┌─────────────────────────────────────────────────────────────┐
  │ [DARK CONTAINER BOX · #0F172A SURFACE]                      │
  │ INSERT IMAGE:                                               │
  │ "reports/figures/drift_psi_summary.png"                     │
  │ Caption: "Population Stability Index (PSI) Drift Audit      │
  │ across Demographic, Origination & Performance Features"     │
  └─────────────────────────────────────────────────────────────┘

---

### SLIDE 5: MULTI-HORIZON PREDICTIVE RISK MODELING
- Layout: Left Column (Comparison Matrix + Safeguards) | Right Column (High-Contrast Dark Photo Frame).
- Header: "MULTI-HORIZON PREDICTIVE RISK MODELING"
- Subtitle: "Calibrated gradient-boosted ensembles evaluated on a strict out-of-time chronological split."
- Hero Metric Banner:
  "12-MONTH DEFAULT ROC-AUC: 0.8168 (+12.6% OVER BASELINE) | PR-AUC: 0.5318 (+41.1%)"
- Comparative Performance Matrix:
  | Target Horizon | Baseline (LogReg) | LPIE Model (XGB/LGB) | Metric Lift | Calibration Error (Brier) |
  | 3M Delinquency Flag | 0.6514 AUC | 0.7350 AUC | +12.8% | 0.1447 |
  | 6M Delinquency Flag | 0.6130 AUC | 0.7082 AUC | +15.5% | 0.1930 |
  | 12M Default Flag | 0.7255 AUC | 0.8168 AUC | +12.6% | 0.1063 (-43% error) |
  | 12M Prepayment Flag | 0.5750 AUC | 0.7119 AUC | +23.8% | 0.2392 |
- Right Column (High-Contrast Dark Photo Frame):
  ┌─────────────────────────────────────────────────────────────┐
  │ [DARK CONTAINER BOX · #0F172A SURFACE]                      │
  │ INSERT IMAGE:                                               │
  │ "reports/figures/calibration_reliability_curves.png"        │
  │ Caption: "Multi-Model Reliability Calibration Curves        │
  │ Showing Predicted Hazard Probabilities Hugging 45° Ideal"   │
  └─────────────────────────────────────────────────────────────┘

---

### SLIDE 6: COMPETING-RISKS SURVIVAL & HAZARD ENGINE
- Layout: Top 3 Stat Cards | Left Column (Statistical Proofs) | Right Column (High-Contrast Dark Photo Frame).
- Header: "TIME-TO-EVENT & COMPETING-RISKS SURVIVAL ENGINE"
- Subtitle: "Modeling voluntary prepayment and involuntary default as mutually exclusive absorbing states."
- Top Stat Triad:
  | 12.6% DEFAULT INCIDENCE | 55.7% PREPAYMENT INCIDENCE | 31.7% RIGHT-CENSORED ACTIVE |
  | 378 / 3,000 Cohort Loans | 1,672 / 3,000 Cohort Loans | Performing at 24M Horizon |
- Left Column Statistical Findings:
  - ⚡ Log-Rank Separation: Prime vs Subprime borrower survival curves diverge with p = 2.8271 × 10⁻¹³ (Extremely Significant).
  - 🛡️ Cox PH Prime Credit Effect: HR = 0.7306 (p = 0.0031) -> Prime borrowers exhibit 26.94% lower monthly default hazard.
  - 📈 Cox PH Rate Elasticity: HR = 1.5777 (p < 0.0001) -> Each +1.0% interest rate hike surges default hazard by +57.77%.
  - ⏱️ Weibull AFT Fit: Superior to constant hazard exponential model by ΔAIC = 149.8 (Hazard accelerates over loan seasoning).
- Right Column (High-Contrast Dark Photo Frame):
  ┌─────────────────────────────────────────────────────────────┐
  │ [DARK CONTAINER BOX · #0F172A SURFACE]                      │
  │ INSERT IMAGE:                                               │
  │ "reports/figures/survival_cif_competing_risks.png"          │
  │ Caption: "Cumulative Incidence Function (CIF) Curves:       │
  │ Voluntary Prepayment vs Involuntary Default"                │
  └─────────────────────────────────────────────────────────────┘

---

### SLIDE 7: AUTOMATED ANOMALY & EXCEPTION INTELLIGENCE
- Layout: 2 Top Rule/ML Cards | Bottom Table (Flagged Diagnostic Cases).
- Header: "HYBRID ANOMALY & EXCEPTION INTELLIGENCE SCANNER"
- Subtitle: "10 deterministic financial validation rules synthesized with unsupervised spatial Isolation Forest."
- Top Split Cards:
  - Card 1 (Royal Blue): "10 HARD ACCOUNTING CHECKS" -> Catches negative balances, balance > original, impossible DPD jumps (0->90 in 30 days), status-DPD contradictions, and stale unserviced records.
  - Card 2 (Emerald): "UNSUPERVISED ISOLATION FOREST" -> Spatial density estimation flags multi-dimensional outlier combinations that satisfy single rules but violate collective multi-feature distribution.
- Bottom Diagnostic Table (Real Extracted Case Studies from `reports/anomaly_report.md`):
  | Loan ID | Exception Type | Diagnosed Root Cause | Anomaly Score | Automated Servicer Action |
  | LN002061 | data_entry_error | Current Balance ($497,888) > Original ($450,000) | 0.741 | Issue servicer reconciliation notice; audit uncapitalized fees. |
  | LN000676 | suspicious_transition | DPD surged 0 -> 90 within 30 days | 0.690 | Audit servicer payment clearing batch; verify bank reversal. |
  | LN001925 | stale_record | Zero balance/status delta for 6 consecutive months | 0.582 | Quarantine loan tape; request active servicing confirmation. |
  | LN000965 | source_conflict | Servicer marks Current; Bureau registers 60DPD | 0.710 | Freeze credit line; initiate 3-way external tape audit. |

---

### SLIDE 8: MACROECONOMIC STRESS TESTING & SCENARIO LAB
- Layout: Left Column (Scenario Dynamics Matrix) | Right Column (High-Contrast Dark Photo Frame).
- Header: "MACROECONOMIC STRESS TESTING & SCENARIO LAB"
- Subtitle: "Quantifying asymmetric portfolio tail risk across calibrated interest rate shocks and credit migrations."
- Left Column Scenario Matrix:
  - 🟢 BASELINE REGIME (0 bps shift | Unemp: 4.1%):
    - 3M Delinq: 26.26% | 12M Default: 14.83% | 12M Prepayment: 50.75%
  - 🔴 ADVERSE CREDIT REGIME (+150 bps rate shock | Unemp: 7.5%):
    - 3M Delinq: 31.58% | 12M Default: 33.91% (+129% Surge) | 12M Prepay: 30.53% (-40% Drop)
  - 🔵 HIGH PREPAYMENT REGIME (-100 bps rate cut | Unemp: 3.8%):
    - 3M Delinq: 23.96% | 12M Default: 9.81% (-34%) | 12M Prepayment: 100.00% (Refinance Wave)
  - 💡 Key Financial Insight:
    "Adverse credit stress triggers dangerous asymmetric tail risk: default rates surge 2.3x while prepayment velocity drops 40%, creating severe portfolio duration extension and compounding credit losses."
- Right Column (High-Contrast Dark Photo Frame):
  ┌─────────────────────────────────────────────────────────────┐
  │ [DARK CONTAINER BOX · #0F172A SURFACE]                      │
  │ INSERT IMAGE:                                               │
  │ "reports/figures/scenario_stress_comparison.png"            │
  │ Caption: "Portfolio Risk Trajectory Across Baseline,        │
  │ Adverse Credit, and High Prepayment Regimes"                │
  └─────────────────────────────────────────────────────────────┘

---

### SLIDE 9: TREESHAP EXPLAINABILITY & FAIR LENDING COMPLIANCE
- Layout: Top 3 Architecture Cards | Bottom (High-Contrast Dark Photo Frame).
- Header: "TREESHAP EXPLAINABILITY & FAIR LENDING COMPLIANCE"
- Subtitle: "Exact Shapley Additive Explanations complying with ECOA and Regulation B Fair Lending standards."
- 3 Explainability Dimension Cards:
  - Card 1: "Global Risk Drivers" -> TreeSHAP identifies days_past_due, balance_to_orig_ratio, and interest_rate as top global hazard drivers across 1,000 samples.
  - Card 2: "Local Waterfall Audits" -> Every loan is fully decomposable: Base Expected Value + Sum of Feature SHAP Attributions = Final Prediction.
  - Card 3: "FP / FN Error Diagnostics" -> Root-cause profiling of False Positives reveals high-rate loans with mature seasoning misclassified without recent payment telemetry.
- Bottom (High-Contrast Dark Photo Frame):
  ┌─────────────────────────────────────────────────────────────┐
  │ [DARK CONTAINER BOX · #0F172A SURFACE]                      │
  │ INSERT IMAGE:                                               │
  │ "reports/figures/shap_summary_next_12m_default_flag.png"   │
  │ Caption: "TreeSHAP Global Feature Importance Beeswarm Plot   │
  │ for 12-Month Default Hazard Prediction"                     │
  └─────────────────────────────────────────────────────────────┘

---

### SLIDE 10: GROUNDED REVIEWER COPILOT (Zero Hallucination)
- Layout: Left Column (3-Tier Anti-Hallucination Framework) | Right Column (Live System Audit Note).
- Header: "GROUNDED REVIEWER COPILOT — ZERO HALLUCINATION"
- Subtitle: "Deterministic note generation strictly bounded by Data Dictionary schema and numerical TreeSHAP citations."
- Left Column Framework:
  - Tier 1: Schema Grounding — Rejects any field not explicitly defined in data_dictionary.md.
  - Tier 2: Mandatory SHAP Citations — Every risk statement MUST cite an exact numerical TreeSHAP attribution value (e.g. "+0.230 SHAP contribution").
  - Tier 3: Immutable Audit Trail — 100% of prompts, completions, and rejections logged to `prompt_log.jsonl`.
  - Rejection Defense: 3 automated tests passed where copilot refused to invent borrower income or unsupported narratives.
- Right Column: Live Generated Reviewer Note (from System Output):
  ┌─────────────────────────────────────────────────────────────┐
  │ [AUDIT NOTE CARD]                                           │
  │ LOAN AUDIT IDENTIFIER: LN000421 | Servicer: ServicerA       │
  │ TRIAGE ACTION: APPROVE (P(Default) = 10.6% | Prepay = 48.2%)│
  │                                                             │
  │ NUMERICAL TREESHAP RISK CITATIONS:                          │
  │ 1. [days_past_due = 0]: -0.184 default hazard impact        │
  │ 2. [credit_score_band = Prime]: -0.112 default hazard impact│
  │ 3. [loan_age_months = 18]: -0.065 default hazard impact     │
  │ 4. [interest_rate = 6.25%]: +0.038 default hazard impact    │
  │                                                             │
  │ COMPLIANCE VERIFICATION:                                    │
  │ • Data Dictionary Verification: PASSED (100% Schema Match)  │
  │ • SHAP Numerical Citations: 4/4 Verified                   │
  │ • Hallucination Score: 0.00% (Strictly Grounded)            │
  └─────────────────────────────────────────────────────────────┘

---

### SLIDE 11: PRODUCTION CLOUD ARCHITECTURE & LIVE DEMO
- Layout: Left Side (4 Production Microservice Cards) | Right Side (High-Contrast Dark Photo Frame).
- Header: "PRODUCTION CLOUD ARCHITECTURE & LIVE DEMO"
- Subtitle: "Live, containerized, zero-setup surveillance engine running in production."
- Left Side 4 Cards:
  - Card 1: "🌐 Live Streamlit Dashboard" -> Hosted at `loan-performance-intelligence-engine.streamlit.app`. Real-time Plotly gauges, stress sliders, and tape explorer.
  - Card 2: "📡 Production REST API" -> FastAPI service exposing `/api/v1/predict/loan` and `/docs` interactive Swagger specifications.
  - Card 3: "⚡ Self-Healing Bootloader" -> Auto-synthesizes data and verifies model checkpoints on clean cloud instances with zero CPU throttling.
  - Card 4: "🧪 100% Verified Test Suite" -> Pytest integration suite running 5/5 passed tests covering health, scoring, copilot, and data synthesis.
- Right Side (High-Contrast Dark Photo Frame):
  ┌─────────────────────────────────────────────────────────────┐
  │ [DARK CONTAINER BOX · #0F172A SURFACE]                      │
  │ INSERT IMAGE:                                               │
  │ "SCREENSHOT OF STREAMLIT APP — MISSION CONTROL / GAUGES"    │
  │ (Take screenshot of loan-performance-intelligence-engine.   │
  │  streamlit.app with gauges and metric cards)                │
  │ Caption: "Live Streamlit Production Dashboard with          │
  │ Interactive Plotly Risk Gauge Speedometers"                 │
  └─────────────────────────────────────────────────────────────┘

---

### SLIDE 12: THANK YOU & LIVE DEMONSTRATION
- Layout: Clean, Elegant Centered Executive Slide with Royal Blue accents.
- Top Pill: [PILL BADGE: "INTAIN FINTECH AI CHALLENGE 2026 · FINAL ROUND 3"]
- Main Headline (52pt bold, Deep Navy #0F172A):
  "Thank You"
- Subtitle (22pt, Royal Blue #2563EB):
  "Questions & Live Demonstration"
- Core Summary Callout:
  "Transforming secondary mortgage surveillance into an automated, explainable, and auditable intelligence workflow — in 40.3 seconds."
- 3 Connection & Verification Bento Cards:
  - Card 1: "🌐 LIVE WEB DASHBOARD" -> loan-performance-intelligence-engine.streamlit.app
  - Card 2: "💻 GITHUB REPOSITORY" -> github.com/knarendrakumar187/loan-performance-intelligence-engine
  - Card 3: "👥 PRESENTED BY" -> Narendra's Team (Sequence #9)
```
