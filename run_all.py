#!/usr/bin/env python3
"""
run_all.py
=============================================================================
ACXIS Master Execution Orchestrator
Executes all experiments in sequential order:
Step 0: Verify cache integrity and data provenance (or retrain from scratch)
Step 1: 100-step validation cost minimization (derives optimal tau*)
Step 2: Algorithm 1 engine with amount-tiered safety guard
Step 3: Comparative baselines benchmark (Table 4)
Step 4: Ablation benchmark panels (Table 6)
Step 5: 1,000-iteration stratified bootstrap resampling (Tables 5 & 7)
Step 6: Latency and throughput hardware benchmark (Table 3)
Step 7: Synchronized publication figures generation

Strictly adheres to Rule 0 (100% Genuine Execution, Zero Derived Numbers).
=============================================================================
"""

import os
import sys
import argparse
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="ACXIS Full Replication Master Runner")
    parser.add_argument("--from-scratch", action="store_true", help="Retrain Stage 1 XGBoost and regenerate SHAP from raw CSV files")
    args = parser.parse_args()

    step0_args = ["--force"] if args.from_scratch else []

    scripts = [
        ("Step 0: Stage 1 Training & TreeSHAP Attribution Extraction", "scripts/step0_train_stage1_and_shap.py", step0_args, "logs/00_stage1_and_shap.log"),
        ("Step 1: Stage 1 Validation Optimization Grid", "scripts/step1_data_and_stage1_coopt.py", [], "logs/01_stage1_coopt.log"),
        ("Step 2: Algorithm 1 Execution Engine", "scripts/step2_algorithm1_engine.py", [], "logs/02_algorithm1.log"),
        ("Step 3: Comparative Baselines Benchmark (Table 4)", "scripts/step3_table4_master_baselines.py", [], "logs/03_table4_baselines.log"),
        ("Step 4: Ablation Benchmark Panels (Table 6)", "scripts/step4_table6_ablation_panels.py", [], "logs/04_table6_ablation.log"),
        ("Step 5: 1,000-Iteration Bootstrap Resampling (Tables 5 & 7)", "scripts/step5_bootstrap_resampling_1000.py", [], "logs/05_bootstrap.log"),
        ("Step 6: Latency & Hardware Throughput Benchmark (Table 3)", "scripts/step6_latency_timing_loop.py", [], "logs/06_latency.log"),
        ("Step 7: Synchronized Publication Figures Generation", "scripts/step7_generate_publication_figures.py", [], "logs/07_figures.log")
    ]

    print("=" * 80)
    print("ACXIS FULL REPLICATION PIPELINE")
    if args.from_scratch:
        print("MODE: FULL RE-EXECUTION FROM RAW CSV DATASETS (FROM SCRATCH)")
    else:
        print("MODE: VERIFIED CACHE EXECUTION")
    print("Zero hardcoded numbers; all operating thresholds derived on validation splits")
    print("=" * 80)
    print(f"Working Directory: {SCRIPT_DIR}\n")
    
    t_start = time.time()
    for name, script_rel, extra_args, log_rel in scripts:
        script_path = os.path.join(SCRIPT_DIR, script_rel)
        log_path = os.path.join(SCRIPT_DIR, log_rel)
        print(f"\n>>> Running: {name}")
        print(f"    Script: {script_path}")
        print(f"    Log:    {log_path}")
        
        t0 = time.time()
        cmd = [sys.executable, script_path] + extra_args
        with open(log_path, "w") as log_file:
            res = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            
        elapsed = time.time() - t0
        if res.returncode == 0:
            print(f"    [✓] SUCCESS ({elapsed:.1f}s)")
        else:
            print(f"    [✗] FAILED with returncode {res.returncode} ({elapsed:.1f}s). Check log: {log_path}")
            sys.exit(1)
            
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_time/60.0:.1f} MINUTES!")
    print("=" * 80)

if __name__ == '__main__':
    main()
