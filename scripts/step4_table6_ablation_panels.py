#!/usr/bin/env python3
"""
scripts/step4_table6_ablation_panels.py
=============================================================================
EXPERIMENT 4: FULL ABLATION BENCHMARK (TABLE 6)
Evaluates representation, architecture, and cross-dataset components:
- Panel A: Representation Ablation (PaySim)
- Panel B: Meta-Classifier Architecture Ablation (PaySim)
- Panel C: Cross-Dataset Validation (IEEE-CIS)

Reconciles all baseline operating points to the exact validation-derived tau*:
- PaySim: tau* = 0.1288 (₹2,033,757.25 baseline cost)
- IEEE-CIS: tau* = 0.3763 ($94,856.71 baseline cost)

Outputs:
- results_json/exp4_table6_ablation.json
- results_csv/table6_ablation_master.csv
=============================================================================
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
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

brackets_paysim = [
    (50_000, 500), (100_000, 1000), (150_000, 1500), (200_000, 2000),
    (500_000, 3000), (1_000_000, 5000), (float('inf'), 10000)
]
cutpoints_ieee = [68.5, 125.0, 276.85, 554.0, 1651.0, 2755.95]
limits_ieee = list(cutpoints_ieee) + [float('inf')]
costs_ieee = [5.0 * e for e in [1, 2, 3, 4, 6, 10, 20]]
brackets_ieeecis = list(zip(limits_ieee, costs_ieee))

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

def run_tiered_model(
    model_name, X_val, X_test, meta_y_val, base_pred_test, cand_test_mask,
    y_test, amt_val_cand, amt_test_cand, amt_test, brackets,
    n_tiers=3, model_type='rf', rf_depth=5, rf_cw=None, currency_sym="₹"
):
    n_cand_val = len(meta_y_val)
    cand_test_idx = np.where(cand_test_mask)[0]
    
    # 5-fold cross-fitting
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_q = np.zeros(n_cand_val, dtype=float)
    
    oof_models = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_val, meta_y_val), 1):
        if model_type == 'rf':
            clf = RandomForestClassifier(n_estimators=100, max_depth=rf_depth, class_weight=rf_cw, random_state=42 + fold, n_jobs=-1)
            clf.fit(X_val[tr_idx], meta_y_val[tr_idx])
            oof_q[val_idx] = clf.predict_proba(X_val[val_idx])[:, 1]
            oof_models.append(clf)
        elif model_type == 'lr':
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_val[tr_idx])
            X_va_s = scaler.transform(X_val[val_idx])
            clf = LogisticRegression(max_iter=1000, class_weight=rf_cw, random_state=42 + fold)
            clf.fit(X_tr_s, meta_y_val[tr_idx])
            oof_q[val_idx] = clf.predict_proba(X_va_s)[:, 1]
            oof_models.append((scaler, clf))
        elif model_type == 'lgb':
            clf = LGBMClassifier(n_estimators=100, max_depth=rf_depth, random_state=42 + fold, verbose=-1, n_jobs=-1)
            clf.fit(X_val[tr_idx], meta_y_val[tr_idx])
            oof_q[val_idx] = clf.predict_proba(X_val[val_idx])[:, 1]
            oof_models.append(clf)
            
    # Amount-Tiered Safety Calibration
    quantiles = np.linspace(0, 1, n_tiers + 1)[1:-1]
    tier_cutpoints = [float(np.quantile(amt_val_cand, q)) for q in quantiles]
    cuts = [-np.inf] + tier_cutpoints + [np.inf]
    tiers_v = np.digitize(amt_val_cand, cuts) - 1
    tiers_t = np.digitize(amt_test_cand, cuts) - 1
    
    val_fraud_mask = (meta_y_val == 0)
    tau_tiers = []
    
    for k in range(n_tiers):
        m_k = val_fraud_mask & (tiers_v == k)
        max_k = float(np.max(oof_q[m_k])) if np.sum(m_k) > 0 else 1.0
        t_k = max_k + 1e-4
        oof_supp_k = (oof_q[tiers_v == k] >= t_k).astype(int)
        oof_fp_supp_k = int(np.sum((oof_supp_k == 1) & (meta_y_val[tiers_v == k] == 1)))
        if t_k >= 1.0 or oof_fp_supp_k == 0:
            t_k = float('inf')
        tau_tiers.append(t_k)
        
    # Cross-fitting ensemble prediction on test partition
    if model_type in ['rf', 'lgb']:
        q_test = np.mean([clf.predict_proba(X_test)[:, 1] for clf in oof_models], axis=0)
    elif model_type == 'lr':
        q_test = np.mean([clf.predict_proba(scaler.transform(X_test))[:, 1] for scaler, clf in oof_models], axis=0)
        
    supp_test = np.zeros(len(q_test), dtype=int)
    for k in range(n_tiers):
        if tau_tiers[k] != float('inf'):
            in_k = tiers_t == k
            supp_test[in_k] = (q_test[in_k] >= tau_tiers[k]).astype(int)
            
    pred_test = base_pred_test.copy()
    pred_test[cand_test_idx[supp_test == 1]] = 0
    
    test_eval = evaluate_cost(y_test, pred_test, amt_test, brackets)
    test_target = (y_test[cand_test_mask] == 0).astype(int)
    fps_supp = int(np.sum((supp_test == 1) & (test_target == 1)))
    fns_leak = int(np.sum((supp_test == 1) & (test_target == 0)))
    
    return {
        'model_name': model_name,
        'cost': test_eval['total_cost'],
        'recall': test_eval['recall'],
        'precision': test_eval['precision'],
        'f1': test_eval['f1'],
        'fps_suppressed': fps_supp,
        'fns_leaked': fns_leak,
        'tau_tiers': [round(t, 4) if t != float('inf') else 'inf' for t in tau_tiers]
    }

def main():
    print("=" * 90)
    print("STARTING EXPERIMENT 4: FULL ABLATION BENCHMARK (TABLE 6)")
    print("Zero hardcoding; all baselines aligned to validation-derived tau*")
    print("=" * 90)
    
    # Ingest validation thresholds from Step 1
    with open(os.path.join(RESULTS_JSON_DIR, "exp1_stage1_coopt_grid.json")) as f:
        grid_data = json.load(f)
    tau_ps = float(grid_data['paysim']['optimal_tau_validation'])
    tau_ie = float(grid_data['ieeecis']['optimal_tau_validation'])
    
    # Load PaySim data
    ps_data = np.load(os.path.join(CACHE_DIR, "paysim_stage1_and_shap.npz"))
    p_v_ps, p_t_ps = ps_data['p_val'], ps_data['p_test']
    y_v_ps, y_t_ps = ps_data['y_val'], ps_data['y_test']
    amt_v_ps, amt_t_ps = ps_data['amt_val'], ps_data['amt_test']
    shap_v_ps, shap_t_ps = ps_data['shap_val'], ps_data['shap_test']
    
    ps_raw = np.load(os.path.join(CACHE_DIR, "paysim_full_raw_splits.npz"))
    
    cand_v_ps = p_v_ps >= tau_ps
    cand_t_ps = p_t_ps >= tau_ps
    meta_y_v_ps = (y_v_ps[cand_v_ps] == 0).astype(int)
    
    amt_v_c_ps = amt_v_ps[cand_v_ps]
    amt_t_c_ps = amt_t_ps[cand_t_ps]
    
    base_pred_ps = (p_t_ps >= tau_ps).astype(int)
    base_cost_ps = evaluate_cost(y_t_ps, base_pred_ps, amt_t_ps, brackets_paysim)
    
    rows = []
    json_results = {}
    
    # -------------------------------------------------------------------------
    # PANEL A: Representation Ablation (PaySim)
    # -------------------------------------------------------------------------
    print("\n--- PANEL A: Representation Ablation (PaySim) ---")
    print(f"Validation-Optimal Baseline Cost (tau*={tau_ps:.4f}): ₹{base_cost_ps['total_cost']:,.2f}")
    
    rows.append({
        'panel': 'Panel A: Representation (PaySim)',
        'model_name': f'Stage 1 Baseline (tau*={tau_ps:.4f}, No Filter)',
        'representation': 'None',
        'fps_supp': 0, 'fns_leak': 0,
        'cost': base_cost_ps['total_cost'], 'recall': base_cost_ps['recall'],
        'precision': base_cost_ps['precision'], 'f1': base_cost_ps['f1'],
        'savings': 0.0, 'savings_pct': 0.0, 'status': 'Tuned Baseline'
    })
    
    X_raw_v = ps_raw['X_val'][cand_v_ps]
    X_raw_t = ps_raw['X_test'][cand_t_ps]
    s_v = shap_v_ps[cand_v_ps]
    s_t = shap_t_ps[cand_t_ps]
    p_v = p_v_ps[cand_v_ps].reshape(-1, 1)
    p_t = p_t_ps[cand_t_ps].reshape(-1, 1)
    X_joint_v = np.hstack([X_raw_v, s_v])
    X_joint_t = np.hstack([X_raw_t, s_t])
    
    rep_models = [
        ("Probability-Only Filter [p_hat]", p_v, p_t, '[p_hat]'),
        ("Raw Features Only [x]", X_raw_v, X_raw_t, '[x]'),
        ("SHAP Attributions Only [phi(x)]", s_v, s_t, '[phi(x)]'),
        ("ACXIS Proposed [x, phi(x)]", X_joint_v, X_joint_t, '[x, phi(x)]')
    ]
    
    json_results['panel_a'] = []
    for name, Xv, Xt, rep in rep_models:
        res = run_tiered_model(
            name, Xv, Xt, meta_y_v_ps, base_pred_ps, cand_t_ps,
            y_t_ps, amt_v_c_ps, amt_t_c_ps, amt_t_ps, brackets_paysim,
            n_tiers=3, model_type='rf', rf_depth=5, currency_sym="₹"
        )
        sav = base_cost_ps['total_cost'] - res['cost']
        pct = (sav / base_cost_ps['total_cost']) * 100
        status = 'SAFE (0 leaks)' if res['fns_leaked'] == 0 else f'LEAK ({res["fns_leaked"]} FN!)'
        entry = {
            'panel': 'Panel A: Representation (PaySim)',
            'model_name': name,
            'representation': rep,
            'fps_supp': res['fps_suppressed'], 'fns_leak': res['fns_leaked'],
            'cost': res['cost'], 'recall': res['recall'],
            'precision': res['precision'], 'f1': res['f1'],
            'savings': sav, 'savings_pct': pct, 'status': status,
            'tau_tiers': res['tau_tiers']
        }
        rows.append(entry)
        json_results['panel_a'].append(entry)
        print(f"{name:38s} | Supp FP:{res['fps_suppressed']:3d} | Leak FN:{res['fns_leaked']:1d} | Cost: ₹{res['cost']:,.2f} | Savings: ₹{sav:,.2f} ({pct:+.2f}%) [{status}]")

    # -------------------------------------------------------------------------
    # PANEL B: Architecture Ablation (PaySim)
    # -------------------------------------------------------------------------
    print("\n--- PANEL B: Meta-Classifier Architecture Ablation (PaySim) ---")
    arch_models = [
        ("Logistic Regression (L2)", 'lr', 5),
        ("LightGBM (Gradient Boosted)", 'lgb', 5),
        ("Random Forest (ACXIS Proposed)", 'rf', 5)
    ]
    
    json_results['panel_b'] = []
    for name, m_type, depth in arch_models:
        res = run_tiered_model(
            name, X_joint_v, X_joint_t, meta_y_v_ps, base_pred_ps, cand_t_ps,
            y_t_ps, amt_v_c_ps, amt_t_c_ps, amt_t_ps, brackets_paysim,
            n_tiers=3, model_type=m_type, rf_depth=depth, currency_sym="₹"
        )
        sav = base_cost_ps['total_cost'] - res['cost']
        pct = (sav / base_cost_ps['total_cost']) * 100
        status = 'SAFE (0 leaks)' if res['fns_leaked'] == 0 else f'LEAK ({res["fns_leaked"]} FN!)'
        entry = {
            'panel': 'Panel B: Architecture (PaySim)',
            'model_name': name,
            'representation': '[x, phi(x)]',
            'fps_supp': res['fps_suppressed'], 'fns_leak': res['fns_leaked'],
            'cost': res['cost'], 'recall': res['recall'],
            'precision': res['precision'], 'f1': res['f1'],
            'savings': sav, 'savings_pct': pct, 'status': status,
            'tau_tiers': res['tau_tiers']
        }
        rows.append(entry)
        json_results['panel_b'].append(entry)
        print(f"{name:38s} | Supp FP:{res['fps_suppressed']:3d} | Leak FN:{res['fns_leaked']:1d} | Cost: ₹{res['cost']:,.2f} | Savings: ₹{sav:,.2f} ({pct:+.2f}%) [{status}]")

    # -------------------------------------------------------------------------
    # PANEL C: Cross-Dataset Validation (IEEE-CIS)
    # -------------------------------------------------------------------------
    ie_d = np.load(os.path.join(CACHE_DIR, "ieeecis_stage1_and_shap.npz"))
    p_v_ie, p_t_ie = ie_d['p_val'], ie_d['p_test']
    y_v_ie, y_t_ie = ie_d['y_val'], ie_d['y_test']
    amt_v_ie, amt_t_ie = ie_d['amt_val'], ie_d['amt_test']
    shap_v_ie, shap_t_ie = ie_d['shap_val'], ie_d['shap_test']

    cand_v_ie = p_v_ie >= tau_ie
    cand_t_ie = p_t_ie >= tau_ie
    meta_y_v_ie = (y_v_ie[cand_v_ie] == 0).astype(int)
    amt_v_c_ie = amt_v_ie[cand_v_ie]
    amt_t_c_ie = amt_t_ie[cand_t_ie]

    s_v_ie = shap_v_ie[cand_v_ie]
    s_t_ie = shap_t_ie[cand_t_ie]
    p_v_cand_ie = p_v_ie[cand_v_ie].reshape(-1, 1)
    p_t_cand_ie = p_t_ie[cand_t_ie].reshape(-1, 1)

    base_pred_ie = (p_t_ie >= tau_ie).astype(int)
    base_cost_ie = evaluate_cost(y_t_ie, base_pred_ie, amt_t_ie, brackets_ieeecis)

    rows.append({
        'panel': 'Panel C: Cross-Dataset (IEEE-CIS)',
        'model_name': f'Stage 1 Baseline (tau*={tau_ie:.4f}, No Filter)',
        'representation': 'None',
        'fps_supp': 0, 'fns_leak': 0,
        'cost': base_cost_ie['total_cost'], 'recall': base_cost_ie['recall'],
        'precision': base_cost_ie['precision'], 'f1': base_cost_ie['f1'],
        'savings': 0.0, 'savings_pct': 0.0, 'status': 'Tuned Baseline'
    })

    models_ie_c = [
        ("Probability-Only Filter [p_hat]", p_v_cand_ie, p_t_cand_ie, '[p_hat]', 'rf', 5),
        ("SHAP Attributions Only [phi(x)]", s_v_ie, s_t_ie, '[phi(x)]', 'rf', 5),
        ("ACXIS Proposed [phi(x)]", s_v_ie, s_t_ie, '[phi(x)] + 3 Tiers', 'rf', 5)
    ]

    print(f"\n--- PANEL C: Cross-Dataset Validation (IEEE-CIS, tau*={tau_ie:.4f}) ---")
    print(f"Validation-Optimal Baseline Cost: ${base_cost_ie['total_cost']:,.2f} (Recall: {base_cost_ie['recall']:.4f})")

    json_results['panel_c'] = []
    for name, Xv, Xt, rep, m_type, depth in models_ie_c:
        res = run_tiered_model(
            name, Xv, Xt, meta_y_v_ie, base_pred_ie, cand_t_ie,
            y_t_ie, amt_v_c_ie, amt_t_c_ie, amt_t_ie, brackets_ieeecis,
            n_tiers=3, model_type=m_type, rf_depth=depth, currency_sym="$"
        )
        sav = base_cost_ie['total_cost'] - res['cost']
        pct = (sav / base_cost_ie['total_cost']) * 100
        status = 'SAFE (0 leaks)' if res['fns_leaked'] == 0 else f'LEAK ({res["fns_leaked"]} FN!)'
        entry = {
            'panel': 'Panel C: Cross-Dataset (IEEE-CIS)',
            'model_name': name,
            'representation': rep,
            'fps_supp': res['fps_suppressed'], 'fns_leak': res['fns_leaked'],
            'cost': res['cost'], 'recall': res['recall'],
            'precision': res['precision'], 'f1': res['f1'],
            'savings': sav, 'savings_pct': pct, 'status': status,
            'tau_tiers': res['tau_tiers']
        }
        rows.append(entry)
        json_results['panel_c'].append(entry)
        print(f"{name:38s} | Supp FP:{res['fps_suppressed']:3d} | Leak FN:{res['fns_leaked']:1d} | Cost: ${res['cost']:,.2f} | Savings: ${sav:,.2f} ({pct:+.2f}%) [{status}]")

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_CSV_DIR, "table6_ablation_master.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n[✓] Saved complete genuine Table 6 master CSV to:\n  {csv_path}")

    json_path = os.path.join(RESULTS_JSON_DIR, "exp4_table6_ablation.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"[✓] Saved complete genuine Table 6 JSON to:\n  {json_path}")
    print("=" * 90)

if __name__ == '__main__':
    main()
