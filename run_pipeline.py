#!/usr/bin/env python3
"""
run_pipeline.py  –  Unified Single-Command Reproducible Entry Point
===================================================================
ISRO SIH AI-ML Dead Reckoning System — Master Reproduction Script.

Executes the entire navigation stack end-to-end from raw data to final benchmarks:
  [Raw Data Ingest] → [Allan Dev Calibration] → [Body-Vehicle Alignment]
  → [Windowed Dataset Prep] → [VelocityNet Training / Export]
  → [Artifact Classifier Training] → [Hybrid Dead Reckoning]
  → [ESEKF Fusion with NHC & ZUPT] → [HMM Road Map Matching]
  → [Multi-Phase Audit & Report Generation]

Usage:
    python3 run_pipeline.py
"""

import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'phase1' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'phase2' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'phase3' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'phase4' / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('run_pipeline')


def main():
    t0 = time.time()
    log.info('=' * 80)
    log.info('ISRO SIH INTELLIGENT DEAD RECKONING SYSTEM — MASTER REPRODUCTION PIPELINE')
    log.info('=' * 80)

    # 1. Phase 1: Ingestion & Baseline
    log.info('\n[1/4] Executing Phase 1: Ingestion, Calibration, Alignment & Baseline...')
    from step1_ingest import build_manifest
    from step3_calibration import run_calibration
    from step4_alignment import run_alignment
    from step6_evaluation import run_evaluation as p1_eval

    manifest, loaded_dfs = build_manifest()
    calib = run_calibration(loaded_dfs, manifest)
    aln_res = run_alignment(loaded_dfs, manifest, calib)
    p1_results, _ = p1_eval(loaded_dfs, manifest, aln_res)

    # 2. Phase 2: AI Velocity Model & EKF
    log.info('\n[2/4] Executing Phase 2: VelocityNet Training, Classifier & Hybrid DR...')
    from phase2.src.run_phase2 import main as run_p2
    # Ensure models and ONNX are present
    models_dir = PROJECT_ROOT / 'phase2' / 'outputs' / 'models'
    if not (models_dir / 'velocity_net.onnx').exists():
        run_p2()

    # 3. Phase 3 & 4: NHC ESEKF, Map Matching, Stress Tests & Audit
    log.info('\n[3/4] Executing Phase 3 & 4: NHC ESEKF, HMM Map Snapping, Stress Tests & Audit...')
    from phase4.src.run_phase4 import main as run_p4
    run_p4()

    elapsed = time.time() - t0
    log.info('\n' + '=' * 80)
    log.info(f'MASTER PIPELINE COMPLETE in {elapsed:.1f}s')
    log.info(f'Final Audit Report: {PROJECT_ROOT}/phase4/FINAL_AUDIT_REPORT.md')
    log.info('=' * 80)


if __name__ == '__main__':
    main()
