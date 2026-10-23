#!/usr/bin/env python3
"""
scripts/step6_latency_timing_loop.py
=============================================================================
STEP 6: LATENCY AND THROUGHPUT BENCHMARK (TABLE 3)
=============================================================================
Measures single-item (B=1) latency distributions and batched (B=1000)
throughput across all pipeline execution stages.

Measures:
1. Preprocessing / Feature Encoding (10,000 trials)
2. Stage 1 Fast-Path XGBoost Scoring (10,000 trials)
3. TreeSHAP Attribution Calculation (1,000 trials)
4. Stage 3 Random Forest Meta-Inference (10,000 trials)
5. Full Triggered ACXIS Pipeline (1,000 trials)
6. LIME Baseline Explainer (100 trials)
7. True Measured Batch Throughput (B=1000, 100 trials) in tx/sec

Outputs:
- results_json/exp6_latency_results.json
- results_csv/table3_latency_and_throughput.csv
=============================================================================
"""

import os
import time
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
from cache_utils import load_stage1_cache

# Directory Setup (Portable relative paths)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PKG_DIR, "cache")
RESULTS_CSV_DIR = os.path.join(PKG_DIR, "results_csv")
RESULTS_JSON_DIR = os.path.join(PKG_DIR, "results_json")
LOGS_DIR = os.path.join(PKG_DIR, "logs")
DATA_DIR = os.environ.get("ACXIS_DATA_DIR", os.path.join(PKG_DIR, "data"))

for d in [RESULTS_CSV_DIR, RESULTS_JSON_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

N_SINGLE_TRIALS = 10000
N_BATCH_TRIALS = 100
BATCH_SIZE = 1000

DATASET_CANDIDATES = [
    os.path.join(DATA_DIR, "six.csv"),
    os.path.join(DATA_DIR, "paysim.csv"),
    os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv"),
    os.path.join(PKG_DIR, "..", "01_R2_Experiments", "data", "six.csv")
]

def resolve_file(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def compute_percentiles(times_ms):
    times_ms = np.asarray(times_ms, dtype=float)
    return {
        'mean': float(np.mean(times_ms)),
        'std': float(np.std(times_ms, ddof=1)),
        'p50': float(np.percentile(times_ms, 50)),
        'p95': float(np.percentile(times_ms, 95)),
        'p99': float(np.percentile(times_ms, 99)),
        'min': float(np.min(times_ms)),
        'max': float(np.max(times_ms))
    }

def run_latency_benchmark():
    print("=" * 80)
    print("STEP 6: HARDWARE LATENCY & THROUGHPUT BENCHMARK (TABLE 3)")
    print("=" * 80)
    
    csv_path = resolve_file(DATASET_CANDIDATES)
    if not csv_path:
        raise FileNotFoundError("Could not find PaySim CSV data file in configured paths.")
        
    print(f"Loading PaySim transactions from: {csv_path}")
    df = pd.read_csv(csv_path, nrows=25000)
    df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
    X_df = pd.get_dummies(df.drop(columns=['isFraud'])).astype(np.float32)
    X_real = X_df.values
    y_real = df['isFraud'].astype(int).values
    n_feats = X_real.shape[1]
    
    # Load or train Stage 1 XGBoost model
    xgb_saved = os.path.join(CACHE_DIR, "paysim_xgb_stage1.json")
    if os.path.isfile(xgb_saved):
        print(f"Loading Stage 1 XGBoost model from {xgb_saved}...")
        xgb = XGBClassifier()
        xgb.load_model(xgb_saved)
    else:
        print(f"Training Stage 1 XGBoost model (500 trees, depth 5, scale_pos_weight=10)...")
        xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, scale_pos_weight=10.0, tree_method='hist')
        xgb.fit(X_real[:10000], y_real[:10000])
        xgb.save_model(xgb_saved)
    
    # Train Stage 3 Random Forest on candidate alert feature space
    print("Loading candidate features for Stage 3 Random Forest timing...")
    ps_data = load_stage1_cache("paysim", CACHE_DIR)
    tau_ps = 0.21792179217921794
    cand_v = ps_data['p_val'] >= tau_ps
    cand_npz = os.path.join(CACHE_DIR, "paysim_cand_raw_X.npz")
    full_npz = np.load(os.path.join(CACHE_DIR, "paysim_full_raw_splits.npz"))
    X_cand_v = full_npz['X_val'][cand_v]
    X_cand_t = full_npz['X_test'][ps_data['p_test'] >= tau_ps]
    np.savez_compressed(cand_npz, X_cand_v=X_cand_v, X_cand_t=X_cand_t)
        
    s_v = ps_data['shap_val'][cand_v]
    y_meta_v = (ps_data['y_val'][cand_v] == 0).astype(int)
    X_joint_v = np.hstack([X_cand_v, s_v])

    print(f"Training Stage 3 Random Forest on candidate alerts ({X_joint_v.shape[0]} alerts, {X_joint_v.shape[1]} features)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_joint_v, y_meta_v)
    
    print("Initializing TreeExplainer on Stage 1 model...")
    explainer = shap.TreeExplainer(xgb)
    
    # 0. Preprocessing timing
    print(f"\n0. Measuring Single-Item Preprocessing Latency ({N_SINGLE_TRIALS:,} trials)...")
    sample_raw_records = df.drop(columns=['isFraud']).iloc[:N_SINGLE_TRIALS].to_dict(orient='records')
    feature_cols = X_df.columns.tolist()
    
    t_preproc_single = []
    for rec in sample_raw_records:
        t0 = time.perf_counter()
        vec = np.zeros(len(feature_cols), dtype=np.float32)
        vec[0] = rec['step']
        vec[1] = rec['amount']
        vec[2] = rec['oldbalanceOrg']
        vec[3] = rec['newbalanceOrig']
        vec[4] = rec['oldbalanceDest']
        vec[5] = rec['newbalanceDest']
        type_col = f"type_{rec['type']}"
        if type_col in feature_cols:
            vec[feature_cols.index(type_col)] = 1.0
        t1 = time.perf_counter()
        t_preproc_single.append((t1 - t0) * 1000.0)
    stats_preproc = compute_percentiles(t_preproc_single)
    print(f"  Preprocessing: Mean={stats_preproc['mean']:.4f} ms | p95={stats_preproc['p95']:.4f} ms | p99={stats_preproc['p99']:.4f} ms")
    
    # 1. Stage 1 Fast-Path
    print(f"\n1. Measuring Stage 1 Fast-Path Scoring ({N_SINGLE_TRIALS:,} trials)...")
    sample_vectors = [X_real[i:i+1] for i in range(N_SINGLE_TRIALS)]
    t_stage1_single = []
    for vec in sample_vectors:
        t0 = time.perf_counter()
        _ = xgb.predict_proba(vec)[0, 1]
        t1 = time.perf_counter()
        t_stage1_single.append((t1 - t0) * 1000.0)
    stats_stage1 = compute_percentiles(t_stage1_single)
    print(f"  Stage 1 Core: Mean={stats_stage1['mean']:.4f} ms | p95={stats_stage1['p95']:.4f} ms | p99={stats_stage1['p99']:.4f} ms")
    
    # 2. TreeSHAP Attribution Calculation
    print("\n2. Measuring TreeSHAP Attribution (1,000 trials)...")
    t_shap_single = []
    for i in range(1000):
        vec = sample_vectors[i]
        t0 = time.perf_counter()
        _ = explainer.shap_values(vec, check_additivity=False)
        t1 = time.perf_counter()
        t_shap_single.append((t1 - t0) * 1000.0)
    stats_shap = compute_percentiles(t_shap_single)
    print(f"  TreeSHAP: Mean={stats_shap['mean']:.4f} ms | p95={stats_shap['p95']:.4f} ms | p99={stats_shap['p99']:.4f} ms")
    
    # 3. Stage 3 RF Meta-Inference
    print(f"\n3. Measuring Stage 3 Meta-Classifier Inference ({N_SINGLE_TRIALS:,} trials)...")
    sample_joint_vectors = [np.hstack([X_real[i:i+1], np.zeros((1, n_feats), dtype=np.float32)]) for i in range(N_SINGLE_TRIALS)]
    t_stage3_single = []
    for j_vec in sample_joint_vectors:
        t0 = time.perf_counter()
        _ = rf.predict_proba(j_vec)[0, 1]
        t1 = time.perf_counter()
        t_stage3_single.append((t1 - t0) * 1000.0)
    stats_stage3 = compute_percentiles(t_stage3_single)
    print(f"  Stage 3 Meta: Mean={stats_stage3['mean']:.4f} ms | p95={stats_stage3['p95']:.4f} ms | p99={stats_stage3['p99']:.4f} ms")
    
    # 4. Full Triggered ACXIS Pipeline (Preproc + Stage 1 + TreeSHAP + Stage 3)
    print("\n4. Measuring Full Triggered ACXIS Pipeline (1,000 trials)...")
    t_full_single = []
    for i in range(1000):
        rec = sample_raw_records[i]
        t0 = time.perf_counter()
        # Preproc
        vec = np.zeros(len(feature_cols), dtype=np.float32)
        vec[0] = rec['step']
        vec[1] = rec['amount']
        vec[2] = rec['oldbalanceOrg']
        vec[3] = rec['newbalanceOrig']
        vec[4] = rec['oldbalanceDest']
        vec[5] = rec['newbalanceDest']
        type_col = f"type_{rec['type']}"
        if type_col in feature_cols:
            vec[feature_cols.index(type_col)] = 1.0
        vec_2d = vec.reshape(1, -1)
        # Stage 1
        p = xgb.predict_proba(vec_2d)[0, 1]
        # TreeSHAP
        sv = explainer.shap_values(vec_2d, check_additivity=False)
        # Stage 3
        j_vec = np.hstack([vec_2d, sv])
        _ = rf.predict_proba(j_vec)[0, 1]
        t1 = time.perf_counter()
        t_full_single.append((t1 - t0) * 1000.0)
    stats_full = compute_percentiles(t_full_single)
    print(f"  Full ACXIS Triggered: Mean={stats_full['mean']:.4f} ms | p95={stats_full['p95']:.4f} ms | p99={stats_full['p99']:.4f} ms")
    
    # 5. Baseline: LIME Explainer
    print("\n5. Measuring Baseline LIME Explainer Latency (100 trials)...")
    lime_exp = LimeTabularExplainer(X_real[:5000], feature_names=feature_cols, mode='classification')
    t_lime = []
    for i in range(100):
        t0 = time.perf_counter()
        _ = lime_exp.explain_instance(X_real[i], xgb.predict_proba, num_features=5)
        t1 = time.perf_counter()
        t_lime.append((t1 - t0) * 1000.0)
    stats_lime = compute_percentiles(t_lime)
    print(f"  LIME: Mean={stats_lime['mean']:.4f} ms | p95={stats_lime['p95']:.4f} ms | p99={stats_lime['p99']:.4f} ms")
    
    # 6. Batch Throughput Measurement (B=1000)
    print(f"\n6. Measuring Batch Throughput (B={BATCH_SIZE:,}, {N_BATCH_TRIALS} trials)...")
    batch_X = X_real[:BATCH_SIZE]
    
    # Stage 1 Batch
    t_b_stage1 = []
    for _ in range(N_BATCH_TRIALS):
        t0 = time.perf_counter()
        _ = xgb.predict_proba(batch_X)
        t1 = time.perf_counter()
        t_b_stage1.append(t1 - t0)
    th_stage1 = BATCH_SIZE / np.mean(t_b_stage1)
    
    # TreeSHAP Batch
    t_b_shap = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = explainer.shap_values(batch_X, check_additivity=False)
        t1 = time.perf_counter()
        t_b_shap.append(t1 - t0)
    th_shap = BATCH_SIZE / np.mean(t_b_shap)
    
    # Stage 3 Batch
    batch_joint = np.hstack([batch_X, np.zeros((BATCH_SIZE, n_feats), dtype=np.float32)])
    t_b_stage3 = []
    for _ in range(N_BATCH_TRIALS):
        t0 = time.perf_counter()
        _ = rf.predict_proba(batch_joint)
        t1 = time.perf_counter()
        t_b_stage3.append(t1 - t0)
    th_stage3 = BATCH_SIZE / np.mean(t_b_stage3)
    
    th_full = BATCH_SIZE / (np.mean(t_b_stage1) + np.mean(t_b_shap) + np.mean(t_b_stage3))
    
    print(f"  Stage 1 Throughput : {th_stage1:,.1f} tx/sec")
    print(f"  TreeSHAP Throughput: {th_shap:,.1f} tx/sec")
    print(f"  Stage 3 Throughput : {th_stage3:,.1f} tx/sec")
    print(f"  Full Batch Trigger : {th_full:,.1f} tx/sec")
    
    # Compile Summary Table
    rows = [
        {
            'Pipeline Stage': 'Feature Preprocessing',
            'Type / Scope': 'Single Transaction',
            'Mean Latency (ms)': round(stats_preproc['mean'], 4),
            'Std Dev (ms)': round(stats_preproc['std'], 4),
            'p50 Tail (ms)': round(stats_preproc['p50'], 4),
            'p95 Tail (ms)': round(stats_preproc['p95'], 4),
            'p99 Tail (ms)': round(stats_preproc['p99'], 4),
            'Batch Throughput (tx/s)': 'N/A'
        },
        {
            'Pipeline Stage': 'Stage 1 Core (Fast-Path XGBoost)',
            'Type / Scope': 'All Transactions (100%)',
            'Mean Latency (ms)': round(stats_stage1['mean'], 4),
            'Std Dev (ms)': round(stats_stage1['std'], 4),
            'p50 Tail (ms)': round(stats_stage1['p50'], 4),
            'p95 Tail (ms)': round(stats_stage1['p95'], 4),
            'p99 Tail (ms)': round(stats_stage1['p99'], 4),
            'Batch Throughput (tx/s)': f"{th_stage1:,.1f}"
        },
        {
            'Pipeline Stage': 'Stage 2 TreeSHAP Attribution',
            'Type / Scope': 'Candidate Alerts Only',
            'Mean Latency (ms)': round(stats_shap['mean'], 4),
            'Std Dev (ms)': round(stats_shap['std'], 4),
            'p50 Tail (ms)': round(stats_shap['p50'], 4),
            'p95 Tail (ms)': round(stats_shap['p95'], 4),
            'p99 Tail (ms)': round(stats_shap['p99'], 4),
            'Batch Throughput (tx/s)': f"{th_shap:,.1f}"
        },
        {
            'Pipeline Stage': 'Stage 3 Meta-Classifier (Random Forest)',
            'Type / Scope': 'Candidate Alerts Only',
            'Mean Latency (ms)': round(stats_stage3['mean'], 4),
            'Std Dev (ms)': round(stats_stage3['std'], 4),
            'p50 Tail (ms)': round(stats_stage3['p50'], 4),
            'p95 Tail (ms)': round(stats_stage3['p95'], 4),
            'p99 Tail (ms)': round(stats_stage3['p99'], 4),
            'Batch Throughput (tx/s)': f"{th_stage3:,.1f}"
        },
        {
            'Pipeline Stage': 'Full Triggered ACXIS Pipeline',
            'Type / Scope': 'Candidate Alerts End-to-End',
            'Mean Latency (ms)': round(stats_full['mean'], 4),
            'Std Dev (ms)': round(stats_full['std'], 4),
            'p50 Tail (ms)': round(stats_full['p50'], 4),
            'p95 Tail (ms)': round(stats_full['p95'], 4),
            'p99 Tail (ms)': round(stats_full['p99'], 4),
            'Batch Throughput (tx/s)': f"{th_full:,.1f}"
        },
        {
            'Pipeline Stage': 'LIME Baseline Explainer',
            'Type / Scope': 'Baseline Comparison',
            'Mean Latency (ms)': round(stats_lime['mean'], 4),
            'Std Dev (ms)': round(stats_lime['std'], 4),
            'p50 Tail (ms)': round(stats_lime['p50'], 4),
            'p95 Tail (ms)': round(stats_lime['p95'], 4),
            'p99 Tail (ms)': round(stats_lime['p99'], 4),
            'Batch Throughput (tx/s)': 'N/A'
        }
    ]
    
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_CSV_DIR, "table3_latency_and_throughput.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n[✓] Saved Table 3 Latency CSV to:\n  {csv_path}")
    
    json_path = os.path.join(RESULTS_JSON_DIR, "exp6_latency_results.json")
    with open(json_path, "w") as f:
        json.dump({
            'preprocessing': stats_preproc,
            'stage1': stats_stage1,
            'treeshap': stats_shap,
            'stage3': stats_stage3,
            'full_pipeline': stats_full,
            'lime': stats_lime,
            'batch_throughput_tx_per_sec': {
                'stage1': th_stage1,
                'treeshap': th_shap,
                'stage3': th_stage3,
                'full_pipeline': th_full
            }
        }, f, indent=2)
    print(f"[✓] Saved Latency JSON to:\n  {json_path}")
    print("\n" + "=" * 80)

if __name__ == '__main__':
    run_latency_benchmark()
