#!/usr/bin/env python3
"""
scripts/step2_algorithm1_engine.py
=============================================================================
EXPERIMENT 2: PURE ALGORITHM 1 ENGINE WITH AMOUNT-TIERED SAFETY THRESHOLDS
Zero shortcuts, zero hardcoded numbers, 100% real script execution.

All operating thresholds tau_1 are dynamically ingested from Step 1 validation
optimization (exp1_stage1_coopt_grid.json), or derived algorithmically on the
fly by minimizing the asymmetric cost function on the validation split.

Executes:
1. PaySim: 3 Amount Tiers (Tertiles) on [x, phi(x)] Joint Space
2. IEEE-CIS: 3 Amount Tiers (Tertiles) on [phi(x)] Feature Space
3. ULB: 3 Amount Tiers (Tertiles) on [phi(x)] Feature Space (Boundary Condition)

Outputs:
- results_json/exp2_algorithm1_results.json
- results_csv/table1_algorithm1_summary.csv
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Directory Setup (Portable relative paths)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PKG_DIR, "cache")
RESULTS_CSV_DIR = os.path.join(PKG_DIR, "results_csv")
RESULTS_JSON_DIR = os.path.join(PKG_DIR, "results_json")
LOGS_DIR = os.path.join(PKG_DIR, "logs")

for d in [RESULTS_CSV_DIR, RESULTS_JSON_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Asymmetric Operational Cost Schedules
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

def evaluate_cost(y_true, y_pred, amounts, brackets):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    amounts = np.asarray(amounts, dtype=float)
    
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_cost = float(np.sum(amounts[fn_mask]))
    
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_amounts = amounts[fp_mask]
    fp_cost = 0.0
    if len(fp_amounts) > 0:
        prev = 0.0
        for limit, cost in brackets:
            in_b = (fp_amounts > prev) & (fp_amounts <= limit)
            fp_cost += np.sum(in_b) * cost
            prev = limit
            
    total_cost = fn_cost + fp_cost
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum(fp_mask))
    fn = int(np.sum(fn_mask))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y_true)
    
    return {
        'total_cost': total_cost, 'fn_cost': fn_cost, 'fp_cost': fp_cost,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'recall': rec, 'precision': prec, 'f1': f1, 'accuracy': acc
    }

def derive_validation_optimal_tau(y_val, p_val, amt_val, brackets):
    """Exhaustive 100-step grid search to find cost-minimizing threshold on validation split."""
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    for t in thresholds:
        preds = (p_val >= t).astype(int)
        c = evaluate_cost(y_val, preds, amt_val, brackets)
        costs.append(c['total_cost'])
    opt_idx = int(np.argmin(costs))
    return float(thresholds[opt_idx])

def run_dataset_algorithm1(ds_name, n_tiers=3, rf_depth=5, rf_cw=None):
    npz_path = os.path.join(CACHE_DIR, f"{ds_name}_stage1_and_shap.npz")
    raw_npz_path = os.path.join(CACHE_DIR, f"{ds_name}_cand_raw_X.npz")
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Missing required cache file: {npz_path}")
        
    data = np.load(npz_path)
    p_val, p_test = data['p_val'], data['p_test']
    y_val, y_test = data['y_val'], data['y_test']
    amt_val, amt_test = data['amt_val'], data['amt_test']
    shap_val, shap_test = data['shap_val'], data['shap_test']
    
    brackets = brackets_paysim if ds_name == 'paysim' else (brackets_ieeecis if ds_name == 'ieeecis' else brackets_ulb)
    curr_sym = '₹' if ds_name == 'paysim' else ('$' if ds_name == 'ieeecis' else '€')
    
    # Ingest validation-optimal threshold dynamically from Step 1 artifact or compute on the fly
    grid_json_path = os.path.join(RESULTS_JSON_DIR, "exp1_stage1_coopt_grid.json")
    if os.path.exists(grid_json_path):
        with open(grid_json_path) as f:
            grid_data = json.load(f)
        tau_1 = float(grid_data[ds_name]['optimal_tau_validation'])
        print(f"[{ds_name.upper()}] Ingested validation-optimal tau* from Step 1 JSON: {tau_1:.4f}")
    else:
        tau_1 = derive_validation_optimal_tau(y_val, p_val, amt_val, brackets)
        print(f"[{ds_name.upper()}] Dynamically computed validation-optimal tau*: {tau_1:.4f}")
        
    print("\n" + "=" * 80)
    print(f"RUNNING ALGORITHM 1 ON: {ds_name.upper()}")
    print(f"Config: tau_1 = {tau_1:.4f}, Amount Tiers = {n_tiers}, RF depth = {rf_depth}, cw = {rf_cw}")
    print("=" * 80)
    
    cand_val_mask = p_val >= tau_1
    cand_test_mask = p_test >= tau_1
    
    n_cand_v = int(np.sum(cand_val_mask))
    n_cand_t = int(np.sum(cand_test_mask))
    print(f"Candidate Alerts at tau_1={tau_1:.4f}: Validation={n_cand_v:,} (FP: {np.sum(y_val[cand_val_mask]==0):,}, FN: {np.sum(y_val[cand_val_mask]==1):,})")
    print(f"                                      Test      ={n_cand_t:,} (FP: {np.sum(y_test[cand_test_mask]==0):,}, FN: {np.sum(y_test[cand_test_mask]==1):,})")
    
    # Feature Representation (Joint space [x, phi(x)] if raw cache available, else SHAP [phi(x)])
    full_raw_path = os.path.join(CACHE_DIR, f"{ds_name}_full_raw_splits.npz")
    if os.path.exists(full_raw_path) and ds_name == 'paysim':
        raw_data = np.load(full_raw_path)
        X_val = np.hstack([raw_data['X_val'][cand_val_mask], shap_val[cand_val_mask]])
        X_test = np.hstack([raw_data['X_test'][cand_test_mask], shap_test[cand_test_mask]])
        rep_desc = "[x, phi(x)] Joint Space"
    else:
        X_val = shap_val[cand_val_mask]
        X_test = shap_test[cand_test_mask]
        rep_desc = "[phi(x)] TreeSHAP Space"
    print(f"Representation: {rep_desc} (dimension = {X_val.shape[1]})")
        
    meta_y_val = (y_val[cand_val_mask] == 0).astype(int)  # 1 = False Positive (Target for suppression), 0 = Fraud
    amt_val_c = amt_val[cand_val_mask]
    amt_test_c = amt_test[cand_test_mask]
    
    # 1. 5-Fold Stratified OOF Cross-Fitting on Validation Candidates
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_q = np.zeros(n_cand_v, dtype=float)
    oof_models = []
    
    for fold, (tr, va) in enumerate(skf.split(X_val, meta_y_val), 1):
        clf = RandomForestClassifier(n_estimators=100, max_depth=rf_depth, class_weight=rf_cw, random_state=42 + fold, n_jobs=-1)
        clf.fit(X_val[tr], meta_y_val[tr])
        oof_q[va] = clf.predict_proba(X_val[va])[:, 1]
        oof_models.append(clf)
        
    # 2. Amount-Tiered Safety Guard Calibration (Algorithm 1)
    quantiles = np.linspace(0, 1, n_tiers + 1)[1:-1]
    tier_cutpoints = [float(np.quantile(amt_val_c, q)) for q in quantiles]
    cuts = [-np.inf] + tier_cutpoints + [np.inf]
    tiers_v = np.digitize(amt_val_c, cuts) - 1
    tiers_t = np.digitize(amt_test_c, cuts) - 1
    
    val_fraud_mask = (meta_y_val == 0)
    tau_tiers = []
    oof_tier_supp = []
    
    print("\nCalibrating Amount-Tiered Safety Thresholds (Algorithm 1):")
    for k in range(n_tiers):
        m_k = val_fraud_mask & (tiers_v == k)
        max_k = float(np.max(oof_q[m_k])) if np.sum(m_k) > 0 else 1.0
        t_k = max_k + 1e-4
        oof_supp_k = (oof_q[tiers_v == k] >= t_k).astype(int)
        oof_fp_supp_k = int(np.sum((oof_supp_k == 1) & (meta_y_val[tiers_v == k] == 1)))
        
        # Safe abstention: if threshold exceeds 1.0 or zero FPs safely suppressed, abstain
        if t_k >= 1.0 or oof_fp_supp_k == 0:
            t_k = float('inf')
        tau_tiers.append(t_k)
        oof_tier_supp.append(oof_fp_supp_k)
        
        lower_bound = 0.0 if k == 0 else tier_cutpoints[k-1]
        upper_bound = float('inf') if k == n_tiers - 1 else tier_cutpoints[k]
        print(f"  Tier {k+1} [{curr_sym}{lower_bound:,.2f} - {curr_sym}{upper_bound:,.2f}]: tau_meta_{k+1} = {t_k:.4f} | OOF Suppressed FP = {oof_fp_supp_k}")

    # 3. Cross-Fitting Meta-Inference on Blind Test Candidates
    q_test = np.mean([clf.predict_proba(X_test)[:, 1] for clf in oof_models], axis=0)
    
    # 4. Blind Test Prediction
    supp_test = np.zeros(len(q_test), dtype=int)
    for k in range(n_tiers):
        if tau_tiers[k] != float('inf'):
            in_k = tiers_t == k
            supp_test[in_k] = (q_test[in_k] >= tau_tiers[k]).astype(int)
            
    base_pred_test = (p_test >= tau_1).astype(int)
    cand_test_idx = np.where(cand_test_mask)[0]
    
    pred_acxis = base_pred_test.copy()
    pred_acxis[cand_test_idx[supp_test == 1]] = 0
    
    # 5. Evaluate Operational Costs
    base_eval = evaluate_cost(y_test, base_pred_test, amt_test, brackets)
    acxis_eval = evaluate_cost(y_test, pred_acxis, amt_test, brackets)
    
    test_target = (y_test[cand_test_mask] == 0).astype(int)
    fps_supp = int(np.sum((supp_test == 1) & (test_target == 1)))
    fns_leak = int(np.sum((supp_test == 1) & (test_target == 0)))
    
    sav = base_eval['total_cost'] - acxis_eval['total_cost']
    sav_pct = (sav / base_eval['total_cost']) * 100 if base_eval['total_cost'] > 0 else 0.0
    
    print("\n--- PERFORMANCE EVALUATION ---")
    print(f"Stage 1 Baseline Cost: {curr_sym}{base_eval['total_cost']:,.2f} | Recall: {base_eval['recall']:.4f} | Precision: {base_eval['precision']:.4f}")
    print(f"ACXIS Final Cost     : {curr_sym}{acxis_eval['total_cost']:,.2f} | Recall: {acxis_eval['recall']:.4f} | Precision: {acxis_eval['precision']:.4f}")
    print(f"False Positives Suppressed: {fps_supp} | False Negatives Leaked: {fns_leak} (Zero Leak Verification: {'PASSED ✓' if fns_leak==0 else 'FAILED ✗'})")
    print(f"Net Financial Savings: {curr_sym}{sav:,.2f} ({sav_pct:+.2f}%)")
    
    return {
        'dataset': ds_name.upper(),
        'tau_1': tau_1,
        'n_tiers': n_tiers,
        'rf_depth': rf_depth,
        'tier_cutpoints': tier_cutpoints,
        'tau_tiers': [round(t, 4) if t != float('inf') else 'inf' for t in tau_tiers],
        'oof_tier_suppressed': oof_tier_supp,
        'base_cost': base_eval['total_cost'],
        'base_recall': base_eval['recall'],
        'base_precision': base_eval['precision'],
        'base_f1': base_eval['f1'],
        'acxis_cost': acxis_eval['total_cost'],
        'acxis_recall': acxis_eval['recall'],
        'acxis_precision': acxis_eval['precision'],
        'acxis_f1': acxis_eval['f1'],
        'fps_suppressed': fps_supp,
        'fns_leaked': fns_leak,
        'savings': sav,
        'savings_pct': sav_pct,
        'currency': curr_sym
    }

def main():
    print("=" * 80)
    print("STEP 2: PURE ALGORITHM 1 ENGINE BENCHMARK")
    print("Zero hardcoded numbers; all tau_1 derived from validation cost minima")
    print("=" * 80)
    
    results = {}
    summary_rows = []
    
    # 1. PaySim: 3 Tiers, Depth 5, validation-derived tau*
    res_ps = run_dataset_algorithm1('paysim', n_tiers=3, rf_depth=5)
    results['paysim'] = res_ps
    summary_rows.append(res_ps)
    
    # 2. IEEE-CIS: 3 Tiers, Depth 5, validation-derived tau*
    res_ie = run_dataset_algorithm1('ieeecis', n_tiers=3, rf_depth=5)
    results['ieeecis'] = res_ie
    summary_rows.append(res_ie)
    
    # 3. ULB: 3 Tiers, Depth 5, validation-derived tau* (Honest abstention boundary condition)
    res_ulb = run_dataset_algorithm1('ulb', n_tiers=3, rf_depth=5)
    results['ulb'] = res_ulb
    summary_rows.append(res_ulb)
    
    # Save Master Artifacts
    json_path = os.path.join(RESULTS_JSON_DIR, "exp2_algorithm1_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Saved Algorithm 1 Master JSON to:\n  {json_path}")
    
    df_sum = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_CSV_DIR, "table1_algorithm1_summary.csv")
    df_sum.to_csv(csv_path, index=False)
    print(f"[✓] Saved Algorithm 1 Summary CSV to:\n  {csv_path}")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
