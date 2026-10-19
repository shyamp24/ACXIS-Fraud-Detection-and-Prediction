#!/usr/bin/env python3
"""
scripts/step1_data_and_stage1_coopt.py
=============================================================================
EXPERIMENT 1: 100-STEP EXHAUSTIVE THRESHOLD SEARCH (0.01 TO 0.99)
Calibrates Stage 1 operational cost curves, evaluates boundary conditions,
and determines optimal operating points (tau*) on validation splits.

Outputs:
- results_json/exp1_stage1_coopt_grid.json
- results_csv/table2_experimental_settings.csv
- figures/fig4_cost_vs_threshold_curves.png
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directory Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PKG_DIR, "cache")
RESULTS_CSV_DIR = os.path.join(PKG_DIR, "results_csv")
RESULTS_JSON_DIR = os.path.join(PKG_DIR, "results_json")
FIGURES_DIR = os.path.join(PKG_DIR, "figures")
LOGS_DIR = os.path.join(PKG_DIR, "logs")

for d in [RESULTS_CSV_DIR, RESULTS_JSON_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Dataset Cost Schedules
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

def evaluate_cost_full(y_true, probs, amounts, tau, brackets):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    preds = (probs >= tau).astype(int)
    
    fn_mask = (y_true == 1) & (preds == 0)
    fn_cost = float(np.sum(amounts[fn_mask]))
    
    fp_mask = (y_true == 0) & (preds == 1)
    fp_amounts = amounts[fp_mask]
    fp_cost = 0.0
    if len(fp_amounts) > 0:
        prev = 0.0
        for limit, cost in brackets:
            in_b = (fp_amounts > prev) & (fp_amounts <= limit)
            fp_cost += np.sum(in_b) * cost
            prev = limit
            
    total_cost = fn_cost + fp_cost
    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum(fp_mask))
    fn = int(np.sum(fn_mask))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {
        'tau': float(tau), 'total_cost': total_cost, 'fn_cost': fn_cost, 'fp_cost': fp_cost,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'recall': rec, 'precision': prec, 'f1': f1
    }

def main():
    print("=" * 80)
    print("STEP 1: 100-STEP EXHAUSTIVE THRESHOLD SEARCH (0.01 TO 0.99)")
    print("=" * 80)
    
    thresholds = np.linspace(0.01, 0.99, 100)
    datasets = ['paysim', 'ieeecis', 'ulb']
    
    grid_results = {}
    table2_records = []
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    for idx, ds_name in enumerate(datasets):
        print(f"\n>>> Processing {ds_name.upper()}...")
        npz_path = os.path.join(CACHE_DIR, f"{ds_name}_stage1_and_shap.npz")
        if not os.path.exists(npz_path):
            print(f"Error: {npz_path} not found!")
            continue
            
        data = np.load(npz_path)
        p_val, p_test = data['p_val'], data['p_test']
        y_val, y_test = data['y_val'], data['y_test']
        amt_val, amt_test = data['amt_val'], data['amt_test']
        
        brackets = brackets_paysim if ds_name == 'paysim' else (brackets_ieeecis if ds_name == 'ieeecis' else brackets_ulb)
        curr_sym = '₹' if ds_name == 'paysim' else ('$' if ds_name == 'ieeecis' else '€')
        
        # 100-step sweep on validation split
        val_curve = []
        for t in thresholds:
            res = evaluate_cost_full(y_val, p_val, amt_val, t, brackets)
            val_curve.append(res)
            
        val_costs = [r['total_cost'] for r in val_curve]
        opt_idx = int(np.argmin(val_costs))
        opt_tau_val = float(thresholds[opt_idx])
        min_cost_val = float(val_costs[opt_idx])
        
        print(f"Validation 100-Step Optimum: tau* = {opt_tau_val:.4f} | Min Cost: {curr_sym}{min_cost_val:,.2f}")
        
        # Evaluate on Test split at Default (0.50) and Tuned (tau*)
        test_default = evaluate_cost_full(y_test, p_test, amt_test, 0.50, brackets)
        test_tuned = evaluate_cost_full(y_test, p_test, amt_test, opt_tau_val, brackets)
        
        # Co-optimized Stage 1 point (Co-Optimized Framework: tau_1 = 0.16838384 on PaySim, 0.2800 on IEEE-CIS)
        coopt_tau = 0.16838384 if ds_name == 'paysim' else (0.2800 if ds_name == 'ieeecis' else opt_tau_val)
        test_coopt = evaluate_cost_full(y_test, p_test, amt_test, coopt_tau, brackets)
        
        # Amount Quantiles for Candidate Alerts at operating point
        cand_mask_val = p_val >= coopt_tau
        amt_cand_val = amt_val[cand_mask_val]
        
        n_tiers = 5 if ds_name == 'ieeecis' else 3
        quantiles = np.linspace(0, 1, n_tiers + 1)[1:-1]
        tier_cuts = [float(np.quantile(amt_cand_val, q)) for q in quantiles]
        
        print(f"Test Set at tau={opt_tau_val:.4f} : Cost = {curr_sym}{test_tuned['total_cost']:,.2f} | Recall = {test_tuned['recall']:.4f} | Precision = {test_tuned['precision']:.4f}")
        if ds_name == 'ieeecis':
            print(f"Test Set at Co-Opt tau=0.2800: Cost = {curr_sym}{test_coopt['total_cost']:,.2f} | Recall = {test_coopt['recall']:.4f} | Precision = {test_coopt['precision']:.4f}")
            
        grid_results[ds_name] = {
            'validation_curve_100_steps': val_curve,
            'optimal_tau_validation': opt_tau_val,
            'coopt_tau_stage1': coopt_tau,
            'amount_tier_cutpoints': tier_cuts,
            'test_default_tau_0_5': test_default,
            'test_tuned_tau': test_tuned,
            'test_coopt_tau': test_coopt
        }
        
        table2_records.append({
            'dataset': ds_name.upper(),
            'val_tx_count': len(y_val),
            'test_tx_count': len(y_test),
            'currency': curr_sym,
            'tau_default': 0.50,
            'tau_tuned_validation': opt_tau_val,
            'tau_coopt_stage1': coopt_tau,
            'n_amount_tiers': n_tiers,
            'tier_cutoffs': tier_cuts,
            'test_cost_default': test_default['total_cost'],
            'test_recall_default': test_default['recall'],
            'test_cost_tuned': test_tuned['total_cost'],
            'test_recall_tuned': test_tuned['recall'],
            'test_cost_coopt': test_coopt['total_cost'],
            'test_recall_coopt': test_coopt['recall']
        })
        
        # Plot cost curves
        ax = axes[idx]
        taus_plot = [r['tau'] for r in val_curve]
        tot_costs_plot = [r['total_cost'] / 1e3 for r in val_curve]
        fn_costs_plot = [r['fn_cost'] / 1e3 for r in val_curve]
        fp_costs_plot = [r['fp_cost'] / 1e3 for r in val_curve]
        
        ax.plot(taus_plot, tot_costs_plot, label='Total Operational Cost', color='#1A237E', linewidth=2.5)
        ax.plot(taus_plot, fn_costs_plot, label='FN Stolen Loss', color='#B71C1C', linestyle='--', linewidth=1.8)
        ax.plot(taus_plot, fp_costs_plot, label='FP Review Cost', color='#E65100', linestyle=':', linewidth=1.8)
        
        ax.axvline(opt_tau_val, color='#2E7D32', linestyle='-.', linewidth=1.5, label=f'Tuned $\\tau^*$ = {opt_tau_val:.2f}')
        if ds_name == 'ieeecis':
            ax.axvline(coopt_tau, color='#6A1B9A', linestyle='-', linewidth=1.5, label=f'Co-Opt $\\tau_1$ = {coopt_tau:.2f}')
            
        ax.set_title(f"{ds_name.upper()} Operational Cost Curve", fontsize=12, fontweight='bold')
        ax.set_xlabel("Classification Threshold ($\\tau$)", fontsize=10)
        ax.set_ylabel(f"Cost ({curr_sym} Thousands)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, loc='upper center')
        
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig4_cost_vs_threshold_curves.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\n[✓] Saved Cost Curve Figure to: {fig_path}")
    
    # Save CSV and JSON
    json_path = os.path.join(RESULTS_JSON_DIR, "exp1_stage1_coopt_grid.json")
    with open(json_path, "w") as f:
        json.dump(grid_results, f, indent=2)
    print(f"[✓] Saved 100-step Grid JSON to: {json_path}")
    
    df_t2 = pd.DataFrame(table2_records)
    csv_path = os.path.join(RESULTS_CSV_DIR, "table2_experimental_settings.csv")
    df_t2.to_csv(csv_path, index=False)
    print(f"[✓] Saved Table 2 Settings CSV to: {csv_path}")
    print("\n" + "=" * 80)
    print("STEP 1 COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
