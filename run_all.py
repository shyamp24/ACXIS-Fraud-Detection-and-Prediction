#!/usr/bin/env python3
"""
run_all.py
=============================================================================
ACXIS Master Execution Orchestrator
Executes all 7 genuine experiments in sequential order.
Strictly adheres to Rule 0 (100% Genuine Execution, Zero Derived Numbers).
=============================================================================
"""

import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    ("Step 1: Stage 1 Co-Optimization Grid", "scripts/step1_data_and_stage1_coopt.py", "logs/01_stage1_coopt.log"),
    ("Step 2: Algorithm 1 Execution Engine", "scripts/step2_algorithm1_engine.py", "logs/02_algorithm1.log"),
    ("Step 3: Comparative Baselines Benchmark (Table 4)", "scripts/step3_table4_master_baselines.py", "logs/03_table4_baselines.log"),
    ("Step 4: Ablation Benchmark Panels (Table 6)", "scripts/step4_table6_ablation_panels.py", "logs/04_table6_ablation.log"),
    ("Step 5: 1,000-Iteration Bootstrap Resampling (Tables 5 & 7)", "scripts/step5_bootstrap_resampling_1000.py", "logs/05_bootstrap.log"),
    ("Step 6: Latency & Hardware Throughput Benchmark (Table 3)", "scripts/step6_latency_timing_loop.py", "logs/06_latency.log"),
    ("Step 7: Synchronized Publication Figures Generation", "scripts/step7_generate_publication_figures.py", "logs/07_figures.log")
]

def main():
    print("=" * 80)
    print("ACXIS FULL REPLICATION PIPELINE (OPTION A+C)")
    print("=" * 80)
    print(f"Working Directory: {SCRIPT_DIR}\n")
    
    t_start = time.time()
    for name, script_rel, log_rel in SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script_rel)
        log_path = os.path.join(SCRIPT_DIR, log_rel)
        print(f"\n>>> Running: {name}")
        print(f"    Script: {script_path}")
        print(f"    Log:    {log_path}")
        
        t0 = time.time()
        with open(log_path, "w") as log_file:
            res = subprocess.run([sys.executable, script_path], stdout=log_file, stderr=subprocess.STDOUT)
            
        elapsed = time.time() - t0
        if res.returncode == 0:
            print(f"    [✓] SUCCESS ({elapsed:.1f}s)")
        else:
            print(f"    [✗] FAILED with returncode {res.returncode} ({elapsed:.1f}s). Check log: {log_path}")
            sys.exit(1)
            
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL 7 EXPERIMENTS COMPLETED SUCCESSFULLY IN {total_time/60.0:.1f} MINUTES!")
    print("=" * 80)

if __name__ == '__main__':
    main()
