#!/usr/bin/env python3
"""
scripts/step9_cost_sensitivity_and_breakeven.py
=============================================================================
MULTI-DIMENSIONAL COST SENSITIVITY & BREAK-EVEN ANALYSIS (TABLE 7 & FIG. 10)
Evaluates operational cost robustness across:
1. Tiered Multiplier Variations: 0.5x (-50%), 1.0x (Baseline), 1.5x (+50%), 2.0x (+100%)
2. Flat Penalty Regimes:
   - PaySim: Flat ₹1,000 and ₹1,500 per alert
   - IEEE-CIS: Flat $10 and $15 per alert
   - ULB: Flat €10 and €15 per alert
3. Proportional Investigation Costs:
   - alpha in {0.5%, 1.0%, 2.0%, 5.0%} of transaction amount
4. Break-Even Investigation Cost Ratio Sweep:
   - Sweeps CFP / mean(Amount) in [0.001, 0.50] (ratios from 2:1 to 1000:1)

Complies strictly with RULE 0: 100% genuine execution on real dataset arrays.

Outputs:
- results_json/exp6_cost_sensitivity_results.json
- results_csv/table7_cost_sensitivity_master.csv
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PKG_DIR, "cache")
RESULTS_JSON_DIR = os.path.join(PKG_DIR, "results_json")
RESULTS_CSV_DIR = os.path.join(PKG_DIR, "results_csv")
FIGURES_DIR = os.path.join(PKG_DIR, "figures")

os.makedirs(RESULTS_JSON_DIR, exist_ok=True)
os.makedirs(RESULTS_CSV_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_data():
    with open(os.path.join(RESULTS_JSON_DIR, "exp1_stage1_coopt_grid.json")) as f:
        grid_d = json.load(f)

    # PaySim
    d_ps = np.load(os.path.join(CACHE_DIR, "paysim_stage1_and_shap.npz"))
    supp_ps = np.load(os.path.join(CACHE_DIR, "paysim_suppressed_test_alerts.npz"))["supp_indices"]

    # IEEE-CIS
    d_ie = np.load(os.path.join(CACHE_DIR, "ieeecis_stage1_and_shap_test.npz"))
    supp_ie = np.load(os.path.join(CACHE_DIR, "ieeecis_suppressed_test_alerts.npz"))["supp_indices"]

    # ULB
    d_ulb = np.load(os.path.join(CACHE_DIR, "ulb_stage1_and_shap.npz"))
    supp_ulb = np.load(os.path.join(CACHE_DIR, "ulb_suppressed_test_alerts.npz"))["supp_indices"]

    # Exact operational cost schedules matching step2_algorithm1_engine.py
    brackets_paysim = [
        (50_000, 500), (100_000, 1000), (150_000, 1500), (200_000, 2000),
        (500_000, 3000), (1_000_000, 5000), (float('inf'), 10000)
    ]
    cutpoints_ieee = [68.5, 125.0, 276.85, 554.0, 1651.0, 2755.95]
    limits_ieee = list(cutpoints_ieee) + [float('inf')]
    costs_ieee = [5.0 * e for e in [1, 2, 3, 4, 6, 10, 20]]
    brackets_ieeecis = list(zip(limits_ieee, costs_ieee))

    brackets_ulb = [
        (20.0, 2.0), (50.0, 5.0), (100.0, 10.0), (250.0, 15.0),
        (500.0, 25.0), (1000.0, 50.0), (float('inf'), 100.0)
    ]

    datasets = {
        "paysim": {
            "name": "PaySim Mobile Money",
            "curr": "₹",
            "tau": float(grid_d["paysim"]["optimal_tau_validation"]),
            "y": d_ps["y_test"],
            "amt": d_ps["amt_test"],
            "p": d_ps["p_test"],
            "supp_idx": set(supp_ps),
            "brackets": brackets_paysim,
            "flats": [1000, 1500]
        },
        "ieeecis": {
            "name": "IEEE-CIS E-Commerce",
            "curr": "$",
            "tau": float(grid_d["ieeecis"]["optimal_tau_validation"]),
            "y": d_ie["y_test"],
            "amt": d_ie["amt_test"],
            "p": d_ie["p_test"],
            "supp_idx": set(supp_ie),
            "brackets": brackets_ieeecis,
            "flats": [10, 15]
        },
        "ulb": {
            "name": "ULB Credit Card",
            "curr": "€",
            "tau": float(grid_d["ulb"]["optimal_tau_validation"]),
            "y": d_ulb["y_test"],
            "amt": d_ulb["amt_test"],
            "p": d_ulb["p_test"],
            "supp_idx": set(supp_ulb),
            "brackets": brackets_ulb,
            "flats": [10, 15]
        }
    }
    return datasets

def main():
    print("=" * 80)
    print("STEP 9: MULTI-DIMENSIONAL COST SENSITIVITY & BREAK-EVEN ANALYSIS")
    print("=" * 80)

    datasets = load_data()
    table_rows = []
    json_out = {}

    for ds_key, cfg in datasets.items():
        ds_name = cfg["name"]
        curr = cfg["curr"]
        tau = cfg["tau"]
        y = cfg["y"]
        amt = cfg["amt"]
        p = cfg["p"]
        supp_idx = cfg["supp_idx"]
        brackets = cfg["brackets"]

        cand_mask = (p >= tau)
        fn_mask = (y == 1) & (~cand_mask)
        fn_cost = float(np.sum(amt[fn_mask]))

        fp_indices = np.where((y == 0) & cand_mask)[0]
        n_fp = len(fp_indices)
        fp_amts = amt[fp_indices]

        # Identify which FPs are suppressed
        is_suppressed = np.array([idx in supp_idx for idx in fp_indices], dtype=bool)
        n_supp = int(np.sum(is_suppressed))

        print(f"\n[{ds_name}] Candidate FPs: {n_fp:,} | Suppressed FPs: {n_supp:,} | FN Cost: {curr}{fn_cost:,.2f}")
        ds_json = {"scenarios": [], "breakeven_sweep": []}

        # 1. Tiered Multiplier Variations: 0.5x, 1.0x, 1.5x, 2.0x
        for mult, label in [
            (0.5, "Low Cost (-50%)"),
            (1.0, "Baseline Tiered (1.0x)"),
            (1.5, "High Cost (+50%)"),
            (2.0, "Very High Cost (+100%)")
        ]:
            def tiered_fp_cost(a):
                for thresh, c in brackets:
                    if a <= thresh:
                        return mult * c
                return mult * brackets[-1][1]

            all_fp_costs = np.array([tiered_fp_cost(a) for a in fp_amts])
            base_fp_total = float(np.sum(all_fp_costs))
            base_total = fn_cost + base_fp_total

            sav = float(np.sum(all_fp_costs[is_suppressed]))
            acxis_total = base_total - sav
            pct = (sav / base_total * 100) if base_total > 0 else 0.0

            table_rows.append({
                "dataset": ds_name,
                "category": "Tiered Multiplier",
                "scenario": label,
                "cost_without": round(base_total, 2),
                "cost_with": round(acxis_total, 2),
                "savings": round(sav, 2),
                "savings_pct": round(pct, 2),
                "currency": curr
            })
            ds_json["scenarios"].append({
                "category": "Tiered Multiplier", "scenario": label,
                "cost_without": base_total, "cost_with": acxis_total, "savings": sav, "pct": pct
            })

        # 2. Flat Penalties
        for flat in cfg["flats"]:
            base_fp_total = float(n_fp * flat)
            base_total = fn_cost + base_fp_total
            sav = float(n_supp * flat)
            acxis_total = base_total - sav
            pct = (sav / base_total * 100) if base_total > 0 else 0.0

            table_rows.append({
                "dataset": ds_name,
                "category": "Flat Penalty",
                "scenario": f"Flat {curr}{flat:,} per False Alarm",
                "cost_without": round(base_total, 2),
                "cost_with": round(acxis_total, 2),
                "savings": round(sav, 2),
                "savings_pct": round(pct, 2),
                "currency": curr
            })
            ds_json["scenarios"].append({
                "category": "Flat Penalty", "scenario": f"Flat {curr}{flat:,}",
                "cost_without": base_total, "cost_with": acxis_total, "savings": sav, "pct": pct
            })

        # 3. Proportional-to-Amount Investigation Cost
        for prop in [0.005, 0.01, 0.02, 0.05]:
            all_fp_costs = fp_amts * prop
            base_fp_total = float(np.sum(all_fp_costs))
            base_total = fn_cost + base_fp_total
            sav = float(np.sum(all_fp_costs[is_suppressed]))
            acxis_total = base_total - sav
            pct = (sav / base_total * 100) if base_total > 0 else 0.0

            table_rows.append({
                "dataset": ds_name,
                "category": "Proportional Cost",
                "scenario": f"FP Cost = {prop*100:.1f}% of Amount",
                "cost_without": round(base_total, 2),
                "cost_with": round(acxis_total, 2),
                "savings": round(sav, 2),
                "savings_pct": round(pct, 2),
                "currency": curr
            })
            ds_json["scenarios"].append({
                "category": "Proportional Cost", "scenario": f"FP Cost = {prop*100:.1f}%",
                "cost_without": base_total, "cost_with": acxis_total, "savings": sav, "pct": pct
            })

        # 4. Break-Even Sweep across 20 log-spaced ratios
        mean_amt = float(np.mean(amt))
        ratios = np.logspace(np.log10(0.001), np.log10(0.50), 20)
        for r in ratios:
            fixed_fp_cost = r * mean_amt
            base_fp_total = float(n_fp * fixed_fp_cost)
            base_total = fn_cost + base_fp_total
            sav = float(n_supp * fixed_fp_cost)
            acxis_total = base_total - sav
            pct = (sav / base_total * 100) if base_total > 0 else 0.0
            ds_json["breakeven_sweep"].append({
                "ratio_cfp_to_mean_amt": float(r),
                "fixed_cfp": float(fixed_fp_cost),
                "cost_without": float(base_total),
                "cost_with": float(acxis_total),
                "savings": float(sav),
                "savings_pct": float(pct)
            })

        json_out[ds_key] = ds_json

    # Save CSV
    df = pd.DataFrame(table_rows)
    out_csv = os.path.join(RESULTS_CSV_DIR, "table7_cost_sensitivity_master.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[✓] Saved Table 7 Master CSV to: {out_csv}")

    # Save JSON
    out_json = os.path.join(RESULTS_JSON_DIR, "exp6_cost_sensitivity_results.json")
    with open(out_json, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"[✓] Saved Cost Sensitivity JSON to: {out_json}")

    print("\n--- TABLE 7 EXECUTION SUMMARY ---")
    for r in table_rows[:12]:
        print(f"{r['dataset']} | {r['category']:17s} | {r['scenario']:30s} | {r['currency']}{r['cost_without']:>12,.2f} | {r['currency']}{r['cost_with']:>12,.2f} | {r['currency']}{r['savings']:>10,.2f} | +{r['savings_pct']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
