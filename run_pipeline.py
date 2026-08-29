#!/usr/bin/env python
"""
run_pipeline.py — One-command end-to-end pipeline for the
Loan Performance Intelligence Engine.

Usage:
    python run_pipeline.py              # Full pipeline
    python run_pipeline.py --dry-run    # Verify imports and config only
    python run_pipeline.py --phase 0    # Run specific phase only
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config


def ensure_directories():
    """Create all required output directories."""
    for d in [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR,
        config.SUBMISSION_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    print("[✓] Directories verified")


def phase_0_data():
    """Phase 0: Generate or verify synthetic data."""
    from src.data.synthesize import generate_all
    generate_all()
    print("[✓] Phase 0: Synthetic data generated")


def phase_1_profiling():
    """Phase 1: Data intelligence and profiling."""
    from src.profiling.report import generate_report
    generate_report()
    print("[✓] Phase 1: Data profiling complete")


def phase_2_prediction():
    """Phase 2: Feature engineering and model training."""
    from src.features.engineer import run_feature_engineering
    from src.models.baseline import train_baseline_models
    from src.models.improved import train_improved_models
    run_feature_engineering()
    train_baseline_models()
    train_improved_models()
    print("[✓] Phase 2: Prediction models trained")


def phase_3_survival():
    """Phase 3: Survival / hazard modeling."""
    from src.survival.competing_risks import run_survival_analysis
    run_survival_analysis()
    print("[✓] Phase 3: Survival analysis complete")


def phase_4_anomaly():
    """Phase 4: Anomaly and exception detection."""
    from src.anomaly.detector import run_anomaly_detection
    run_anomaly_detection()
    print("[✓] Phase 4: Anomaly detection complete")


def phase_5_scenario():
    """Phase 5: Scenario and stress simulation."""
    from src.scenario.simulator import run_scenario_simulation
    run_scenario_simulation()
    print("[✓] Phase 5: Scenario simulation complete")


def phase_6_explain():
    """Phase 6: Explainability layer."""
    from src.explain.explainer import run_explainability
    run_explainability()
    print("[✓] Phase 6: Explainability complete")


def phase_7_copilot():
    """Phase 7: LLM-assisted reviewer copilot."""
    from src.copilot.reviewer import run_copilot_demo
    run_copilot_demo()
    print("[✓] Phase 7: Copilot demo complete")


def phase_8_submission():
    """Phase 8: Generate final submission."""
    from src.models.submission import generate_submission
    generate_submission()
    print("[✓] Phase 8: Submission generated")


PHASES = {
    0: phase_0_data,
    1: phase_1_profiling,
    2: phase_2_prediction,
    3: phase_3_survival,
    4: phase_4_anomaly,
    5: phase_5_scenario,
    6: phase_6_explain,
    7: phase_7_copilot,
    8: phase_8_submission,
}


def main():
    parser = argparse.ArgumentParser(
        description="Loan Performance Intelligence Engine — End-to-End Pipeline"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verify imports and config only, don't run phases"
    )
    parser.add_argument(
        "--phase", type=int, choices=range(9),
        help="Run a specific phase only (0-8)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Loan Performance Intelligence Engine")
    print("Intain Campus FinTech Challenge 2026 — AI Track")
    print("=" * 60)

    ensure_directories()

    if args.dry_run:
        print("\n[DRY RUN] Config and directories verified. Exiting.")
        return

    start = time.time()

    if args.phase is not None:
        print(f"\n▶ Running Phase {args.phase} only")
        PHASES[args.phase]()
    else:
        print("\n▶ Running full pipeline (Phases 0-8)")
        for phase_num, phase_fn in PHASES.items():
            print(f"\n{'─' * 40}")
            print(f"Phase {phase_num}")
            print(f"{'─' * 40}")
            phase_fn()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Submission: {config.SUBMISSION_OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
