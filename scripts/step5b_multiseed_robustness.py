#!/usr/bin/env python3
"""
STEP 5b: MULTI-SEED META-CLASSIFIER ROBUSTNESS EVALUATION (10 RANDOM SEEDS)
=============================================================================
Evaluates the downstream Random Forest meta-classifier across 10 independent
random seeds (seeds 42 through 51) using full Algorithm 1 OOF safety guard
calibration to quantify bagging variance on candidate alerts across both
PaySim and IEEE-CIS.

Discloses the sensitivity of downstream ensemble bagging on fixed thresholds:
- PaySim: 5 of 10 seeds positive (0 leaks), 5 seeds with 1-2 leaks (mean -₹131,061).
- IEEE-CIS: 10 of 10 seeds positive (0 leaks), mean +$209.50 ± $91.15 (+0.22%).
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from step2_algorithm1_engine import run_dataset_algorithm1

REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_JSON_DIR = os.path.join(REPO_ROOT, "results_json")
RESULTS_CSV_DIR = os.path.join(REPO_ROOT, "results_csv")

SEEDS = list(range(42, 52))

def evaluate_dataset(dataset_name, curr_symbol):
    print("\n" + "=" * 80)
    print(f"STEP 5b: EVALUATING {dataset_name.upper()} ACROSS 10 SEEDS (42..51)")
    print("=" * 80)

    records = []
    for s in SEEDS:
        print(f"\n--- {dataset_name.upper()} Seed {s} ---")
        res = run_dataset_algorithm1(dataset_name, n_tiers=3, rf_depth=5, seed=s)
        records.append({
            "dataset": dataset_name,
            "seed": s,
            "fps_suppressed": res["fps_suppressed"],
            "fns_leaked": res["fns_leaked"],
            "savings": res["savings"],
            "savings_pct": res["savings_pct"],
            "acxis_recall": res["acxis_recall"],
            "acxis_precision": res["acxis_precision"]
        })

    df = pd.DataFrame(records)
    summary = {
        "dataset": dataset_name,
        "seeds_evaluated": len(SEEDS),
        "mean_savings": float(df["savings"].mean()),
        "std_savings": float(df["savings"].std()),
        "mean_savings_pct": float(df["savings_pct"].mean()),
        "std_savings_pct": float(df["savings_pct"].std()),
        "mean_fps_suppressed": float(df["fps_suppressed"].mean()),
        "std_fps_suppressed": float(df["fps_suppressed"].std()),
        "max_fns_leaked": int(df["fns_leaked"].max()),
        "min_fns_leaked": int(df["fns_leaked"].min()),
        "detailed_runs": records
    }

    n_pos = sum(r["savings"] > 0 for r in records)
    print("\n" + "=" * 80)
    print(f"Summary over {len(SEEDS)} seeds ({dataset_name.upper()}):")
    print(f"  Mean Savings  : {curr_symbol}{summary['mean_savings']:,.2f} ± {curr_symbol}{summary['std_savings']:,.2f} ({summary['mean_savings_pct']:.2f}% ± {summary['std_savings_pct']:.2f}%)")
    print(f"  FPs Suppressed: {summary['mean_fps_suppressed']:.1f} ± {summary['std_fps_suppressed']:.1f}")
    print(f"  FN Leaks Range: [{summary['min_fns_leaked']} .. {summary['max_fns_leaked']}] (Max leaks: {summary['max_fns_leaked']})")
    print(f"  Positive Seeds: {n_pos}/{len(SEEDS)} seeds achieve positive net savings (0 leaks)")
    print("=" * 80)

    return summary, df

def main():
    print("=" * 80)
    print("STEP 5b: DUAL-DATASET MULTI-SEED ROBUSTNESS EVALUATION (10 SEEDS: 42..51)")
    print("=" * 80)

    summary_paysim, df_paysim = evaluate_dataset("paysim", "₹")
    summary_ieeecis, df_ieeecis = evaluate_dataset("ieeecis", "$")

    combined_json = {
        "paysim": summary_paysim,
        "ieeecis": summary_ieeecis
    }

    out_json = os.path.join(RESULTS_JSON_DIR, "exp5b_multiseed_robustness.json")
    with open(out_json, "w") as f:
        json.dump(combined_json, f, indent=2)

    df_combined = pd.concat([df_paysim, df_ieeecis], ignore_index=True)
    out_csv = os.path.join(RESULTS_CSV_DIR, "table5b_multiseed_robustness.csv")
    df_combined.to_csv(out_csv, index=False)

    print(f"\n[✓] Saved Dual-Dataset Multi-Seed JSON to: {out_json}")
    print(f"[✓] Saved Dual-Dataset Multi-Seed CSV  to: {out_csv}")

if __name__ == "__main__":
    main()
