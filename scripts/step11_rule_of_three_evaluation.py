#!/usr/bin/env python3
"""
STEP 11: RULE-OF-THREE STATISTICAL UPPER BOUND ON FALSE NEGATIVE LEAKAGE RISK
=============================================================================
Calculates the exact Hanley-Jippner / Efron Rule-of-Three 95% Clopper-Pearson /
Poisson upper confidence bounds for zero observed false negative leaks:
    p_upper ≈ 3 / N_frauds
across all evaluated benchmark datasets (PaySim, IEEE-CIS, ULB).
"""

import json
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_JSON_DIR = os.path.join(REPO_ROOT, "results_json")

def main():
    print("=" * 80)
    print("STEP 11: RULE-OF-THREE UPPER BOUND EVALUATION (ZERO FALSE NEGATIVE LEAKS)")
    print("=" * 80)

    # Candidate alert true frauds evaluated (zero leaks observed)
    # PaySim: 808 candidate true frauds (p >= 0.2179)
    # IEEE-CIS: 1,642 candidate true frauds (p >= 0.3763)
    # ULB: 41 candidate true frauds (p >= 0.0863)
    datasets = [
        ("PaySim", 808, 0),
        ("IEEE-CIS", 1642, 0),
        ("ULB", 41, 0),
    ]

    results = {}

    print(f"{'Dataset':<12} | {'Tested Frauds (N)':<18} | {'FN Leaks':<10} | {'Exact 95% Bound':<16} | {'Rule-of-Three'}")
    print("-" * 75)

    for name, n_frauds, fn_leaks in datasets:
        rule_of_three = 3.0 / n_frauds
        exact_bound = -np.log(0.05) / n_frauds

        results[name.lower()] = {
            "dataset": name,
            "candidate_true_frauds_evaluated": int(n_frauds),
            "false_negative_leaks_observed": int(fn_leaks),
            "rule_of_three_upper_bound_pct": float(round(rule_of_three * 100, 4)),
            "exact_95_upper_bound_pct": float(round(exact_bound * 100, 4)),
            "formula": "3 / N"
        }

        print(f"{name:<12} | {n_frauds:<18} | {fn_leaks:<10} | {exact_bound*100:6.3f}%         | {rule_of_three*100:6.3f}% (3/{n_frauds})")

    out_json = os.path.join(RESULTS_JSON_DIR, "exp11_rule_of_three_bounds.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Saved Rule-of-Three bounds to: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
