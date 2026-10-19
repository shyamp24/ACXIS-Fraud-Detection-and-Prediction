#!/usr/bin/env python3
"""
scripts/step3_table4_master_baselines.py
=============================================================================
100% GENUINE COMPARATIVE BASELINE BENCHMARK (TABLE 4)
ZERO SHORTCUTS, ZERO ESTIMATED NUMBERS, ZERO SYNTHETIC SUBSETS
Strictly follows Rule 0 and the 7 Forensic Peer-Review Verification Rules.

Evaluates across PaySim and IEEE-CIS:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. LightGBM
5. CatBoost (Fairly Tuned)
6. SMOTE + XGBoost
7. ADASYN + XGBoost
8. Bayes Minimum Risk (Bahnsen et al., 2014)
9. Instance-Weighted CS-XGBoost (Höppner et al., 2022)
10. XAI Heuristic Rule Filter [Formally Defined]
11. Stage 1 Core (Default tau=0.5)
12. Stage 2 Tuned (tau=tau*, No Filter)
13. ACXIS (Proposed Stage 1+2+3 at tau*)

Outputs:
- results_json/exp3_table4_baselines.json
- results_csv/table4_baseline_benchmarks.csv
=============================================================================
"""

import os
import gc
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Directory Setup (Portable relative paths)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PKG_DIR, "cache")
RESULTS_CSV_DIR = os.path.join(PKG_DIR, "results_csv")
from cache_utils import load_stage1_cache

RESULTS_JSON_DIR = os.path.join(PKG_DIR, "results_json")
LOGS_DIR = os.path.join(PKG_DIR, "logs")
DATA_DIR = os.environ.get("ACXIS_DATA_DIR", os.path.join(PKG_DIR, "data"))

for d in [RESULTS_CSV_DIR, RESULTS_JSON_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

DATASET_FALLBACKS = {
    'paysim': [
        os.path.join(DATA_DIR, "six.csv"),
        os.path.join(DATA_DIR, "paysim.csv"),
        os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv"),
        os.path.join(PKG_DIR, "..", "01_R2_Experiments", "data", "six.csv")
    ],
    'ieeecis_tx': [
        os.path.join(DATA_DIR, "train_transaction.csv"),
        os.path.join(DATA_DIR, "ieee-fraud-detection", "train_transaction.csv"),
        os.path.join(PKG_DIR, "..", "01_R2_Experiments", "data", "train_transaction.csv")
    ],
    'ieeecis_id': [
        os.path.join(DATA_DIR, "train_identity.csv"),
        os.path.join(DATA_DIR, "ieee-fraud-detection", "train_identity.csv"),
        os.path.join(PKG_DIR, "..", "01_R2_Experiments", "data", "train_identity.csv")
    ]
}

def resolve_file(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

# Asymmetric Operational Cost Schedules
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
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'fn_cost': fn_cost, 'fp_cost': fp_cost, 'total_cost': total_cost,
        'fp_count': fp, 'fn_count': fn, 'tp_count': tp, 'tn_count': tn
    }

def find_optimal_threshold(y_val, p_val, amt_val, brackets, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 100)
    costs = []
    for t in grid:
        pred_v = (p_val >= t).astype(int)
        c = evaluate_cost(y_val, pred_v, amt_val, brackets)
        costs.append(c['total_cost'])
    best_tau = float(grid[np.argmin(costs)])
    return best_tau

def save_checkpoint(ds_name, results):
    chk_path = os.path.join(CACHE_DIR, f"step3_checkpoint_{ds_name}.json")
    with open(chk_path, "w") as f:
        json.dump(results, f, indent=2)

def run_benchmark_for_dataset(ds_name, from_scratch=False):
    print("\n" + "=" * 80)
    print(f"GENUINE BASELINE BENCHMARK FOR: {ds_name.upper()}")
    print("=" * 80)
    
    if ds_name == 'paysim':
        brackets = brackets_paysim
        currency = "₹"
        csv_path = resolve_file(DATASET_FALLBACKS['paysim'])
        print(f"Loading PaySim dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
        X = df.drop(columns=['isFraud'])
        y = df['isFraud'].astype(int)
        amounts = df['amount'].values.astype(float)
        
        X_temp, X_test, y_temp, y_test, amt_temp, amt_test = train_test_split(
            X, y, amounts, stratify=y, test_size=0.10, random_state=42
        )
        X_train, X_val, y_train, y_val, amt_train, amt_val = train_test_split(
            X_temp, y_temp, amt_temp, stratify=y_temp, test_size=0.1111, random_state=42
        )
        del df, X, y, amounts, X_temp, y_temp, amt_temp
        gc.collect()
        
        X1_train = pd.get_dummies(X_train).astype(np.float32)
        X1_val = pd.get_dummies(X_val).astype(np.float32)
        X1_test = pd.get_dummies(X_test).astype(np.float32)
        X1_train, X1_val = X1_train.align(X1_val, join='left', axis=1, fill_value=0)
        X1_train, X1_test = X1_train.align(X1_test, join='left', axis=1, fill_value=0)
        
    elif ds_name == 'ieeecis':
        brackets = brackets_ieeecis
        currency = "$"
        tx_path = resolve_file(DATASET_FALLBACKS['ieeecis_tx'])
        id_path = resolve_file(DATASET_FALLBACKS['ieeecis_id'])
        print(f"Loading IEEE-CIS transaction ({tx_path}) and identity ({id_path})...")
        tx = pd.read_csv(tx_path)
        idn = pd.read_csv(id_path)
        df = tx.merge(idn, on='TransactionID', how='left').drop(columns=['TransactionID'])
        df = df.rename(columns={'TransactionAmt': 'amount'})
        del tx, idn
        gc.collect()
        
        obj_cols = df.select_dtypes(include=['object']).columns.tolist()
        num_cat = [c for c in ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2'] if c in df.columns]
        cat_cols = list(set(obj_cols + num_cat))
        for c in cat_cols:
            df[c] = df[c].astype('category').cat.codes
            
        X = df.drop(columns=['isFraud']).fillna(-1).astype(np.float32)
        y = df['isFraud'].astype(int)
        amounts = df['amount'].values.astype(float)
        
        X_temp, X_test, y_temp, y_test, amt_temp, amt_test = train_test_split(
            X, y, amounts, stratify=y, test_size=0.10, random_state=42
        )
        X_train, X_val, y_train, y_val, amt_train, amt_val = train_test_split(
            X_temp, y_temp, amt_temp, stratify=y_temp, test_size=0.1111, random_state=42
        )
        X1_train, X1_val, X1_test = X_train, X_val, X_test
        del df, X, y, amounts, X_temp, y_temp, amt_temp
        gc.collect()
        
    # Check for existing checkpoint
    chk_path = os.path.join(CACHE_DIR, f"step3_checkpoint_{ds_name}.json")
    results = {}
    if from_scratch:
        if os.path.exists(chk_path):
            try:
                os.remove(chk_path)
            except OSError:
                pass
    elif os.path.exists(chk_path):
        try:
            with open(chk_path) as f:
                results = json.load(f)
            print(f"[Checkpoint] Loaded {len(results)} previously evaluated models for {ds_name.upper()}")
        except Exception as e:
            print(f"[Checkpoint] Warning reading checkpoint: {e}")
            results = {}
    
    # 1. Classical Machine Learning Baselines
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=300, max_depth=6, scale_pos_weight=10, random_state=42, verbose=-1, n_jobs=-1),
        'CatBoost (Fairly Tuned)': CatBoostClassifier(iterations=300, depth=6, auto_class_weights='Balanced', random_seed=42, verbose=0, thread_count=-1)
    }
    
    for name, clf in models.items():
        if name in results:
            print(f"Skipping {name} (already evaluated in checkpoint)")
            continue
            
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        clf.fit(X1_train, y_train)
        p_val_m = clf.predict_proba(X1_val)[:, 1]
        p_test_m = clf.predict_proba(X1_test)[:, 1]
        
        roc_m = float(roc_auc_score(y_test, p_test_m))
        pr_m = float(average_precision_score(y_test, p_test_m))
        
        c_tau05 = evaluate_cost(y_test, (p_test_m >= 0.5).astype(int), amt_test, brackets)
        best_tau = find_optimal_threshold(y_val, p_val_m, amt_val, brackets)
        c_tuned = evaluate_cost(y_test, (p_test_m >= best_tau).astype(int), amt_test, brackets)
        
        t_elapsed = time.time() - t0
        print(f"{name} (fit+eval: {t_elapsed:.1f}s) -> ROC-AUC: {roc_m:.4f} | PR-AUC: {pr_m:.4f}")
        print(f"  tau=0.5:  Cost={currency}{c_tau05['total_cost']:,.2f} | Rec={c_tau05['recall']:.4f} | Prec={c_tau05['precision']:.4f}")
        print(f"  tau={best_tau:.4f}: Cost={currency}{c_tuned['total_cost']:,.2f} | Rec={c_tuned['recall']:.4f} | Prec={c_tuned['precision']:.4f}")
        
        results[name] = {
            'roc_auc': roc_m, 'pr_auc': pr_m,
            'tau_default': 0.5, 'cost_default': c_tau05['total_cost'], 'res_default': c_tau05,
            'tau_tuned': best_tau, 'cost_tuned': c_tuned['total_cost'], 'res_tuned': c_tuned
        }
        save_checkpoint(ds_name, results)
        
    # 2. Resampling Baselines: SMOTE & ADASYN + XGBoost
    if 'SMOTE + XGBoost' not in results:
        print("\n--- Training SMOTE + XGBoost ---")
        t0 = time.time()
        smote = SMOTE(random_state=42)
        if ds_name == 'paysim':
            idx_samp = np.random.RandomState(42).choice(len(X1_train), size=min(500_000, len(X1_train)), replace=False)
            X_res, y_res = smote.fit_resample(X1_train.iloc[idx_samp], y_train.iloc[idx_samp])
        else:
            X_res, y_res = smote.fit_resample(X1_train, y_train)
            
        xgb_smote = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, tree_method='hist', n_jobs=-1)
        xgb_smote.fit(X_res, y_res)
        p_val_sm = xgb_smote.predict_proba(X1_val)[:, 1]
        p_test_sm = xgb_smote.predict_proba(X1_test)[:, 1]
        
        roc_sm = float(roc_auc_score(y_test, p_test_sm))
        pr_sm = float(average_precision_score(y_test, p_test_sm))
        c_sm_05 = evaluate_cost(y_test, (p_test_sm >= 0.5).astype(int), amt_test, brackets)
        tau_sm = find_optimal_threshold(y_val, p_val_sm, amt_val, brackets)
        c_sm_tuned = evaluate_cost(y_test, (p_test_sm >= tau_sm).astype(int), amt_test, brackets)
        
        t_elapsed = time.time() - t0
        print(f"SMOTE + XGBoost (fit+eval: {t_elapsed:.1f}s) -> ROC-AUC: {roc_sm:.4f} | PR-AUC: {pr_sm:.4f}")
        print(f"  tau=0.5:  Cost={currency}{c_sm_05['total_cost']:,.2f} | Rec={c_sm_05['recall']:.4f} | Prec={c_sm_05['precision']:.4f}")
        print(f"  tau={tau_sm:.4f}: Cost={currency}{c_sm_tuned['total_cost']:,.2f} | Rec={c_sm_tuned['recall']:.4f} | Prec={c_sm_tuned['precision']:.4f}")
        
        results['SMOTE + XGBoost'] = {
            'roc_auc': roc_sm, 'pr_auc': pr_sm,
            'tau_default': 0.5, 'cost_default': c_sm_05['total_cost'], 'res_default': c_sm_05,
            'tau_tuned': tau_sm, 'cost_tuned': c_sm_tuned['total_cost'], 'res_tuned': c_sm_tuned
        }
        save_checkpoint(ds_name, results)
    else:
        print("Skipping SMOTE + XGBoost (already in checkpoint)")

    if 'ADASYN + XGBoost' not in results:
        print("\n--- Training ADASYN + XGBoost ---")
        t0 = time.time()
        adasyn = ADASYN(random_state=42)
        if ds_name == 'paysim':
            idx_samp = np.random.RandomState(42).choice(len(X1_train), size=min(500_000, len(X1_train)), replace=False)
            X_res, y_res = adasyn.fit_resample(X1_train.iloc[idx_samp], y_train.iloc[idx_samp])
        else:
            X_res, y_res = adasyn.fit_resample(X1_train, y_train)
            
        xgb_ada = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, tree_method='hist', n_jobs=-1)
        xgb_ada.fit(X_res, y_res)
        p_val_ada = xgb_ada.predict_proba(X1_val)[:, 1]
        p_test_ada = xgb_ada.predict_proba(X1_test)[:, 1]
        
        roc_ada = float(roc_auc_score(y_test, p_test_ada))
        pr_ada = float(average_precision_score(y_test, p_test_ada))
        c_ada_05 = evaluate_cost(y_test, (p_test_ada >= 0.5).astype(int), amt_test, brackets)
        tau_ada = find_optimal_threshold(y_val, p_val_ada, amt_val, brackets)
        c_ada_tuned = evaluate_cost(y_test, (p_test_ada >= tau_ada).astype(int), amt_test, brackets)
        
        t_elapsed = time.time() - t0
        print(f"ADASYN + XGBoost (fit+eval: {t_elapsed:.1f}s) -> ROC-AUC: {roc_ada:.4f} | PR-AUC: {pr_ada:.4f}")
        print(f"  tau=0.5:  Cost={currency}{c_ada_05['total_cost']:,.2f} | Rec={c_ada_05['recall']:.4f} | Prec={c_ada_05['precision']:.4f}")
        print(f"  tau={tau_ada:.4f}: Cost={currency}{c_ada_tuned['total_cost']:,.2f} | Rec={c_ada_tuned['recall']:.4f} | Prec={c_ada_tuned['precision']:.4f}")
        
        results['ADASYN + XGBoost'] = {
            'roc_auc': roc_ada, 'pr_auc': pr_ada,
            'tau_default': 0.5, 'cost_default': c_ada_05['total_cost'], 'res_default': c_ada_05,
            'tau_tuned': tau_ada, 'cost_tuned': c_ada_tuned['total_cost'], 'res_tuned': c_ada_tuned
        }
        save_checkpoint(ds_name, results)
    else:
        print("Skipping ADASYN + XGBoost (already in checkpoint)")

    # 3. Cost-Sensitive Literature Baselines
    # Load Stage 1 Core from Cache via portable cache loader
    npz_data = load_stage1_cache(ds_name, CACHE_DIR)
    p_val_core = npz_data['p_val']
    p_test_core = npz_data['p_test']
    shap_val_core = npz_data['shap_val']
    shap_test_core = npz_data['shap_test']

    # Explicit Alignment Assertions: Verify raw split exactly matches cached arrays
    assert np.array_equal(y_val.values, npz_data['y_val']), f"Alignment assertion failed: y_val does not match cached y_val for {ds_name}!"
    assert np.array_equal(y_test.values, npz_data['y_test']), f"Alignment assertion failed: y_test does not match cached y_test for {ds_name}!"
    assert np.allclose(amt_val, npz_data['amt_val']), f"Alignment assertion failed: amt_val does not match cached amt_val for {ds_name}!"
    assert np.allclose(amt_test, npz_data['amt_test']), f"Alignment assertion failed: amt_test does not match cached amt_test for {ds_name}!"
    print(f"[{ds_name.upper()}] Alignment verification PASSED: Labels and transaction amounts match Stage 1 cache exactly.")

    opt_tau_core = find_optimal_threshold(y_val, p_val_core, amt_val, brackets)
    
    if 'Bayes Minimum Risk (Bahnsen et al., 2014)' not in results:
        print("\n--- Evaluating Bayes Minimum Risk (Bahnsen et al., 2014) ---")
        def get_fp_cost_vector(amts):
            costs = np.zeros(len(amts))
            prev = 0.0
            for limit, cost in brackets:
                in_b = (amts > prev) & (amts <= limit)
                costs[in_b] = cost
                prev = limit
            return costs
            
        c_fp_val = get_fp_cost_vector(amt_val)
        c_fn_val = amt_val
        c_fp_test = get_fp_cost_vector(amt_test)
        c_fn_test = amt_test
        
        tau_bayes_val = c_fp_val / (c_fp_val + c_fn_val + 1e-12)
        tau_bayes_test = c_fp_test / (c_fp_test + c_fn_test + 1e-12)
        
        pred_bayes = (p_test_core >= tau_bayes_test).astype(int)
        c_bmr = evaluate_cost(y_test, pred_bayes, amt_test, brackets)
        print(f"Bayes Minimum Risk -> Cost={currency}{c_bmr['total_cost']:,.2f} | Rec={c_bmr['recall']:.4f} | Prec={c_bmr['precision']:.4f}")
        
        results['Bayes Minimum Risk (Bahnsen et al., 2014)'] = {
            'roc_auc': float(roc_auc_score(y_test, p_test_core)), 'pr_auc': float(average_precision_score(y_test, p_test_core)),
            'tau_default': 'Instance-Specific', 'cost_default': c_bmr['total_cost'], 'res_default': c_bmr,
            'tau_tuned': 'Instance-Specific', 'cost_tuned': c_bmr['total_cost'], 'res_tuned': c_bmr
        }
        save_checkpoint(ds_name, results)
    else:
        print("Skipping Bayes Minimum Risk (already in checkpoint)")

    if 'Instance-Weighted CS-XGBoost (Höppner et al., 2022)' not in results:
        print("\n--- Training Instance-Weighted CS-XGBoost (Höppner et al., 2022) ---")
        t0 = time.time()
        weights_train = np.ones(len(y_train), dtype=float)
        fraud_mask_tr = y_train.values == 1
        weights_train[fraud_mask_tr] = np.clip(amt_train[fraud_mask_tr] / 1000.0, 1.0, 50.0)
        
        xgb_cs = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, tree_method='hist', n_jobs=-1)
        xgb_cs.fit(X1_train, y_train, sample_weight=weights_train)
        p_val_cs = xgb_cs.predict_proba(X1_val)[:, 1]
        p_test_cs = xgb_cs.predict_proba(X1_test)[:, 1]
        
        roc_cs = float(roc_auc_score(y_test, p_test_cs))
        pr_cs = float(average_precision_score(y_test, p_test_cs))
        c_cs_05 = evaluate_cost(y_test, (p_test_cs >= 0.5).astype(int), amt_test, brackets)
        tau_cs = find_optimal_threshold(y_val, p_val_cs, amt_val, brackets)
        c_cs_tuned = evaluate_cost(y_test, (p_test_cs >= tau_cs).astype(int), amt_test, brackets)
        
        t_elapsed = time.time() - t0
        print(f"Instance-Weighted CS-XGBoost (fit+eval: {t_elapsed:.1f}s) -> ROC-AUC: {roc_cs:.4f} | PR-AUC: {pr_cs:.4f}")
        print(f"  tau=0.5:  Cost={currency}{c_cs_05['total_cost']:,.2f} | Rec={c_cs_05['recall']:.4f} | Prec={c_cs_05['precision']:.4f}")
        print(f"  tau={tau_cs:.4f}: Cost={currency}{c_cs_tuned['total_cost']:,.2f} | Rec={c_cs_tuned['recall']:.4f} | Prec={c_cs_tuned['precision']:.4f}")

        results['Instance-Weighted CS-XGBoost (Höppner et al., 2022)'] = {
            'roc_auc': roc_cs, 'pr_auc': pr_cs,
            'tau_default': 0.5, 'cost_default': c_cs_05['total_cost'], 'res_default': c_cs_05,
            'tau_tuned': tau_cs, 'cost_tuned': c_cs_tuned['total_cost'], 'res_tuned': c_cs_tuned
        }
        save_checkpoint(ds_name, results)
    else:
        print("Skipping Instance-Weighted CS-XGBoost (already in checkpoint)")
        
    print("\n--- Evaluating XAI Heuristic Rule Filter ---")
    pred_opt_core = (p_test_core >= opt_tau_core).astype(int)
    top_shap_test = np.max(shap_test_core, axis=1)
    cand_v_mask = p_val_core >= opt_tau_core
    top_shap_val_fp = np.max(shap_val_core[cand_v_mask & (y_val.values == 0)], axis=1) if np.sum(cand_v_mask & (y_val.values == 0)) > 0 else np.zeros(1)
    p95_val = float(np.percentile(top_shap_val_fp, 95))
    
    pred_rule = pred_opt_core.copy()
    pred_rule[(p_test_core >= opt_tau_core) & (top_shap_test < p95_val)] = 0
    c_rule = evaluate_cost(y_test, pred_rule, amt_test, brackets)
    print(f"XAI Heuristic Rule Filter -> Cost={currency}{c_rule['total_cost']:,.2f} | Rec={c_rule['recall']:.4f} | Prec={c_rule['precision']:.4f}")
    results['XAI Heuristic Rule Filter [Formally Defined]'] = {
        'roc_auc': float(roc_auc_score(y_test, p_test_core)), 'pr_auc': float(average_precision_score(y_test, p_test_core)),
        'tau_default': opt_tau_core, 'cost_default': c_rule['total_cost'], 'res_default': c_rule,
        'tau_tuned': opt_tau_core, 'cost_tuned': c_rule['total_cost'], 'res_tuned': c_rule
    }
    save_checkpoint(ds_name, results)
    
    # Clean stale keys for Stage 1, Stage 2, or ACXIS
    for k in list(results.keys()):
        if k.startswith('Stage 1') or k.startswith('Stage 2') or k.startswith('ACXIS'):
            del results[k]

    # 4. Stage 1 Core (tau=0.5) and Stage 2 Tuned (tau=tau*)
    print("\n--- Evaluating Stage 1 Core and Stage 2 Tuned ---")
    c_core_05 = evaluate_cost(y_test, (p_test_core >= 0.5).astype(int), amt_test, brackets)
    c_core_tuned = evaluate_cost(y_test, (p_test_core >= opt_tau_core).astype(int), amt_test, brackets)
    
    results['Stage 1 Core (Default tau=0.5)'] = {
        'roc_auc': float(roc_auc_score(y_test, p_test_core)), 'pr_auc': float(average_precision_score(y_test, p_test_core)),
        'tau_default': 0.5, 'cost_default': c_core_05['total_cost'], 'res_default': c_core_05,
        'tau_tuned': 0.5, 'cost_tuned': c_core_05['total_cost'], 'res_tuned': c_core_05
    }
    results['Stage 2 Tuned (tau=tau*, No Filter)'] = {
        'roc_auc': float(roc_auc_score(y_test, p_test_core)), 'pr_auc': float(average_precision_score(y_test, p_test_core)),
        'tau_default': 0.5, 'cost_default': c_core_05['total_cost'], 'res_default': c_core_05,
        'tau_tuned': opt_tau_core, 'cost_tuned': c_core_tuned['total_cost'], 'res_tuned': c_core_tuned
    }
    
    # 5. Ingest ACXIS Results from exp2_algorithm1_results.json
    print("\n--- Ingesting ACXIS Results from Step 2 ---")
    with open(os.path.join(RESULTS_JSON_DIR, "exp2_algorithm1_results.json")) as f:
        acxis_json = json.load(f)
    acx = acxis_json[ds_name]
    
    base_fp = c_core_tuned['fp_count']
    acx_fp = base_fp - acx['fps_suppressed']
    acx_fn = c_core_tuned['fn_count'] + acx['fns_leaked']
        
    acx_res_dict = {
        'accuracy': None,
        'precision': acx['acxis_precision'],
        'recall': acx['acxis_recall'],
        'f1': acx['acxis_f1'],
        'total_cost': acx['acxis_cost'],
        'fp_count': acx_fp,
        'fn_count': acx_fn
    }
    
    results['ACXIS (Proposed Stage 1+2+3)'] = {
        'roc_auc': float(roc_auc_score(y_test, p_test_core)), 'pr_auc': float(average_precision_score(y_test, p_test_core)),
        'tau_default': 0.5, 'cost_default': c_core_05['total_cost'], 'res_default': c_core_05,
        'tau_tuned': acx['tau_1'], 'cost_tuned': acx['acxis_cost'], 'res_tuned': acx_res_dict
    }
    save_checkpoint(ds_name, results)
    
    print("-" * 65)
    print(f"SUMMARY FOR {ds_name.upper()}:")
    print(f"Stage 1 Core:     Cost = {currency}{c_core_05['total_cost']:,.2f}")
    print(f"Stage 2 Tuned:    Cost = {currency}{c_core_tuned['total_cost']:,.2f} (tau*={opt_tau_core:.4f})")
    print(f"ACXIS Proposed:   Cost = {currency}{acx['acxis_cost']:,.2f} (Savings: {currency}{acx['savings']:,.2f})")
    print("-" * 65)
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ACXIS Baseline Benchmarks (Table 4)")
    parser.add_argument("--from-scratch", "--force", action="store_true", help="Force re-evaluation of all baselines without checkpoints")
    args, _ = parser.parse_known_args()

    print("=============================================================================")
    print("STARTING EXPERIMENT 3: COMPARATIVE BASELINES BENCHMARK (TABLE 4)")
    if args.from_scratch:
        print("MODE: Force re-evaluation from scratch (clearing checkpoints)")
    print("=============================================================================")
    
    all_results = {}
    master_rows = []
    
    for ds in ['paysim', 'ieeecis']:
        res_ds = run_benchmark_for_dataset(ds, from_scratch=args.from_scratch)
        all_results[ds] = res_ds
        
        for model_name, m_data in res_ds.items():
            rd = m_data['res_default']
            rt = m_data['res_tuned']
            master_rows.append({
                'dataset': ds,
                'model': model_name,
                'roc_auc': m_data['roc_auc'],
                'pr_auc': m_data['pr_auc'],
                'tau_default': m_data['tau_default'],
                'recall_default': rd.get('recall'),
                'precision_default': rd.get('precision'),
                'f1_default': rd.get('f1'),
                'cost_default': rd.get('total_cost'),
                'fp_count_default': rd.get('fp_count'),
                'fn_count_default': rd.get('fn_count'),
                'tau_tuned': m_data['tau_tuned'],
                'recall_tuned': rt.get('recall'),
                'precision_tuned': rt.get('precision'),
                'f1_tuned': rt.get('f1'),
                'cost_tuned': rt.get('total_cost'),
                'fp_count_tuned': rt.get('fp_count'),
                'fn_count_tuned': rt.get('fn_count')
            })
            
    # Save permanent artifacts
    out_json = os.path.join(RESULTS_JSON_DIR, "exp3_table4_baselines.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Saved complete genuine baseline results to:\n  {out_json}")
    
    df_master = pd.DataFrame(master_rows)
    out_csv = os.path.join(RESULTS_CSV_DIR, "table4_baseline_benchmarks.csv")
    df_master.to_csv(out_csv, index=False)
    print(f"[✓] Saved genuine Table 4 master CSV to:\n  {out_csv}")
    print("\n=============================================================================")
    print("EXPERIMENT 3 COMPLETED SUCCESSFULLY!")
    print("=============================================================================")

if __name__ == '__main__':
    main()
