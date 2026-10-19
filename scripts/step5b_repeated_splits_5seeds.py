#!/usr/bin/env python3
"""
scripts/step5b_repeated_splits_5seeds.py
=============================================================================
MULTI-SEED / MULTI-SPLIT ROBUSTNESS EVALUATION (5 INDEPENDENT SEEDS)
Evaluates ACXIS across 5 independent random seeds: [42, 101, 202, 303, 404]
to quantify true empirical variance across random initializations and cross-fitting splits.

Performs:
1. 5 distinct fold splits and model initializations per dataset.
2. Measures Base Cost, ACXIS Cost, Net Savings, Savings %, Suppressed FPs, Leaked FNs.
3. Computes sample Mean ± SD across the 5 independent runs.
4. Outputs:
   - results_csv/table5b_multiseed_robustness.csv
   - results_json/exp5b_multiseed_robustness.json
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from cache_utils import load_stage1_cache

# Directory Setup (Portable relative paths)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PKG_DIR, "cache")
RESULTS_CSV_DIR = os.path.join(PKG_DIR, "results_csv")
RESULTS_JSON_DIR = os.path.join(PKG_DIR, "results_json")
LOGS_DIR = os.path.join(PKG_DIR, "logs")

for d in [RESULTS_CSV_DIR, RESULTS_JSON_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

SEEDS = [42, 101, 202, 303, 404]

brackets_paysim = [
    (50_000, 500), (100_000, 1000), (150_000, 1500), (200_000, 2000),
    (500_000, 3000), (1_000_000, 5000), (float('inf'), 10000)
]
brackets_ieeecis = [
    (100, 5.0), (500, 15.0), (1000, 30.0), (float('inf'), 50.0)
]
brackets_ulb = [(float('inf'), 10.0)]

def evaluate_cost(y_true, y_pred, amounts, brackets):
    fps = np.where((y_pred == 1) & (y_true == 0))[0]
    fns = np.where((y_pred == 0) & (y_true == 1))[0]
    tps = np.where((y_pred == 1) & (y_true == 1))[0]
    tns = np.where((y_pred == 0) & (y_true == 0))[0]
    
    fp_cost = 0.0
    for idx in fps:
        a = amounts[idx]
        c = brackets[-1][1]
        for limit, cost in brackets:
            if a <= limit:
                c = cost
                break
        fp_cost += c
        
    fn_cost = float(np.sum(amounts[fns]))
    total_cost = fp_cost + fn_cost
    
    p = len(tps) / (len(tps) + len(fps)) if (len(tps) + len(fps)) > 0 else 0.0
    r = len(tps) / (len(tps) + len(fns)) if (len(tps) + len(fns)) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    return {
        'total_cost': total_cost, 'fp_cost': fp_cost, 'fn_cost': fn_cost,
        'fps': len(fps), 'fns': len(fns), 'tps': len(tps), 'tns': len(tns),
        'precision': p, 'recall': r, 'f1': f1
    }

def run_seed_evaluation(ds_name, seed, data, tau_1, n_tiers=3, rf_depth=5):
    brackets = brackets_paysim if ds_name == 'paysim' else (brackets_ieeecis if ds_name == 'ieeecis' else brackets_ulb)
    
    p_val, p_test = data['p_val'], data['p_test']
    y_val, y_test = data['y_val'], data['y_test']
    amt_val, amt_test = data['amt_val'], data['amt_test']
    shap_val, shap_test = data['shap_val'], data['shap_test']
    
    cand_val_mask = p_val >= tau_1
    cand_test_mask = p_test >= tau_1
    
    if ds_name == 'paysim':
        full_splits = np.load(os.path.join(CACHE_DIR, "paysim_full_raw_splits.npz"))
        X_val = np.hstack([full_splits['X_val'][cand_val_mask], shap_val[cand_val_mask]])
        X_test = np.hstack([full_splits['X_test'][cand_test_mask], shap_test[cand_test_mask]])
    else:
        X_val = shap_val[cand_val_mask]
        X_test = shap_test[cand_test_mask]
        
    meta_y_val = (y_val[cand_val_mask] == 0).astype(int)
    amt_val_c = amt_val[cand_val_mask]
    amt_test_c = amt_test[cand_test_mask]
    
    # 5-fold cross-fitting with the specific seed
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_q = np.zeros(len(meta_y_val), dtype=float)
    oof_models = []
    
    for fold, (tr, va) in enumerate(skf.split(X_val, meta_y_val), 1):
        clf = RandomForestClassifier(n_estimators=100, max_depth=rf_depth, random_state=seed + fold, n_jobs=-1)
        clf.fit(X_val[tr], meta_y_val[tr])
        oof_q[va] = clf.predict_proba(X_val[va])[:, 1]
        oof_models.append(clf)
        
    # Amount-tiered safety guard calibration
    quantiles = np.linspace(0, 1, n_tiers + 1)[1:-1]
    tier_cutpoints = [float(np.quantile(amt_val_c, q)) for q in quantiles]
    cuts = [-np.inf] + tier_cutpoints + [np.inf]
    tiers_v = np.digitize(amt_val_c, cuts) - 1
    tiers_t = np.digitize(amt_test_c, cuts) - 1
    
    val_fraud_mask = (meta_y_val == 0)
    tau_tiers = []
    
    for k in range(n_tiers):
        m_k = val_fraud_mask & (tiers_v == k)
        max_k = float(np.max(oof_q[m_k])) if np.sum(m_k) > 0 else 1.0
        t_k = max_k + 1e-4
        oof_supp_k = (oof_q[tiers_v == k] >= t_k).astype(int)
        oof_fp_supp_k = int(np.sum((oof_supp_k == 1) & (meta_y_val[tiers_v == k] == 1)))
        
        # Safe abstention if threshold >= 1.0 or zero FPs safely suppressed
        if t_k >= 1.0 or oof_fp_supp_k == 0:
            t_k = float('inf')
        tau_tiers.append(t_k)
        
    # Predict on test candidates via 5-fold ensemble
    q_test = np.mean([clf.predict_proba(X_test)[:, 1] for clf in oof_models], axis=0)
    
    supp_test = np.zeros(len(q_test), dtype=int)
    for k in range(n_tiers):
        if tau_tiers[k] != float('inf'):
            in_k = tiers_t == k
            supp_test[in_k] = (q_test[in_k] >= tau_tiers[k]).astype(int)
        
    # Evaluate Base and ACXIS
    base_preds = (p_test >= tau_1).astype(int)
    base_cost = evaluate_cost(y_test, base_preds, amt_test, brackets)
    
    acxis_preds = base_preds.copy()
    cand_test_indices = np.where(cand_test_mask)[0]
    supp_test_indices = cand_test_indices[supp_test == 1]
    acxis_preds[supp_test_indices] = 0
    acxis_cost = evaluate_cost(y_test, acxis_preds, amt_test, brackets)
    
    sav = base_cost['total_cost'] - acxis_cost['total_cost']
    sav_pct = (sav / base_cost['total_cost']) * 100
    fps_supp = int(base_cost['fps'] - acxis_cost['fps'])
    fns_leak = int(acxis_cost['fns'] - base_cost['fns'])
    
    return {
        'seed': seed,
        'base_cost': base_cost['total_cost'],
        'acxis_cost': acxis_cost['total_cost'],
        'savings': sav,
        'savings_pct': sav_pct,
        'fps_suppressed': fps_supp,
        'fns_leaked': fns_leak
    }

def main():
    print("=" * 80)
    print("MULTI-SEED EVALUATION ACROSS 5 INDEPENDENT SEEDS (42, 101, 202, 303, 404)")
    print("=" * 80)
    
    grid_path = os.path.join(RESULTS_JSON_DIR, "exp1_stage1_coopt_grid.json")
    with open(grid_path) as f:
        grid_data = json.load(f)
        
    master_results = {}
    summary_rows = []
    
    for ds_name in ['paysim', 'ieeecis', 'ulb']:
        print(f"\n>>> Running 5-Seed Evaluation on {ds_name.upper()}...")
        tau_1 = float(grid_data[ds_name]['optimal_tau_validation'])
        data = load_stage1_cache(ds_name, CACHE_DIR)
        curr = "₹" if ds_name == 'paysim' else ("$" if ds_name == 'ieeecis' else "€")
        
        seed_runs = []
        for s in SEEDS:
            res = run_seed_evaluation(ds_name, s, data, tau_1)
            seed_runs.append(res)
            print(f"  Seed {s:3d}: Base={curr}{res['base_cost']:,.2f} | ACXIS={curr}{res['acxis_cost']:,.2f} | Sav={curr}{res['savings']:,.2f} ({res['savings_pct']:+.2f}%) | Supp={res['fps_suppressed']:2d} | Leak={res['fns_leaked']}")
            
        savings_arr = np.array([r['savings'] for r in seed_runs])
        pct_arr = np.array([r['savings_pct'] for r in seed_runs])
        fps_arr = np.array([r['fps_suppressed'] for r in seed_runs])
        fns_arr = np.array([r['fns_leaked'] for r in seed_runs])
        
        master_results[ds_name] = {
            'seeds': seed_runs,
            'mean_savings': float(np.mean(savings_arr)),
            'std_savings': float(np.std(savings_arr, ddof=1)),
            'mean_savings_pct': float(np.mean(pct_arr)),
            'std_savings_pct': float(np.std(pct_arr, ddof=1)),
            'mean_fps_suppressed': float(np.mean(fps_arr)),
            'std_fps_suppressed': float(np.std(fps_arr, ddof=1)),
            'mean_fns_leaked': float(np.mean(fns_arr))
        }
        
        summary_rows.append({
            'Dataset': ds_name.upper(),
            'Operating tau*': f"{tau_1:.4f}",
            'Mean Savings': f"{curr}{np.mean(savings_arr):,.2f} ± {curr}{np.std(savings_arr, ddof=1):,.2f}",
            'Mean Savings %': f"{np.mean(pct_arr):.2f}% ± {np.std(pct_arr, ddof=1):.2f}%",
            'Suppressed FPs': f"{np.mean(fps_arr):.1f} ± {np.std(fps_arr, ddof=1):.1f}",
            'Leaked FNs': f"{np.mean(fns_arr):.2f}"
        })
        
    # Save artifacts
    csv_out = os.path.join(RESULTS_CSV_DIR, "table5b_multiseed_robustness.csv")
    pd.DataFrame(summary_rows).to_csv(csv_out, index=False)
    print(f"\n[✓] Saved Table 5b Multi-Seed Robustness CSV to: {csv_out}")
    
    json_out = os.path.join(RESULTS_JSON_DIR, "exp5b_multiseed_robustness.json")
    with open(json_out, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"[✓] Saved Multi-Seed Master JSON to: {json_out}")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
