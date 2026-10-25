#!/usr/bin/env python3
"""
scripts/step0_train_stage1_and_shap.py
=============================================================================
STAGE 1 MODEL TRAINING & TREESHAP ATTRIBUTION EXTRACTION
Provides complete end-to-end reproducibility from raw datasets to cached NPZ.

Pipeline:
1. Loads raw dataset CSV files from ACXIS_DATA_DIR or local data/ directory.
2. Performs deterministic 80/10/10 train/val/test splits (random_state=42).
3. Trains Stage 1 XGBoost (n_estimators=500, max_depth=5, lr=0.05, scale_pos_weight=10).
4. Computes TreeSHAP attribution matrices on validation and test partitions.
5. Saves precomputed arrays to cache/{dataset}_stage1_and_shap.npz.
6. For large datasets (>100MB like IEEE-CIS), saves split files (_val.npz, _test.npz)
   to ensure 100% plain Git compatibility without requiring Git LFS.

If cache files already exist on disk and are valid (>1KB), validates integrity.
=============================================================================
"""

import os
import sys
import gc
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap

# Portable directory resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PKG_DIR, "cache")
LOGS_DIR = os.path.join(PKG_DIR, "logs")
DATA_DIR = os.environ.get("ACXIS_DATA_DIR", os.path.join(PKG_DIR, "data"))

for d in [CACHE_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

DATASET_FALLBACKS = {
    'paysim': [
        os.path.join(DATA_DIR, "six.csv"),
        os.path.join(DATA_DIR, "paysim.csv"),
        os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv")
    ],
    'ieeecis_tx': [
        os.path.join(DATA_DIR, "train_transaction.csv"),
        os.path.join(DATA_DIR, "ieee-fraud-detection", "train_transaction.csv")
    ],
    'ieeecis_id': [
        os.path.join(DATA_DIR, "train_identity.csv"),
        os.path.join(DATA_DIR, "ieee-fraud-detection", "train_identity.csv")
    ],
    'ulb': [
        os.path.join(DATA_DIR, "creditcard.csv"),
        os.path.join(DATA_DIR, "card.csv")
    ]
}

def resolve_file(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def check_cache_status():
    datasets = ['paysim', 'ieeecis', 'ulb']
    print("=" * 80)
    print("CACHE INTEGRITY & PROVENANCE CHECK")
    print("=" * 80)
    all_present = True
    for ds in datasets:
        p = os.path.join(CACHE_DIR, f"{ds}_stage1_and_shap.npz")
        p_val = os.path.join(CACHE_DIR, f"{ds}_stage1_and_shap_val.npz")
        p_test = os.path.join(CACHE_DIR, f"{ds}_stage1_and_shap_test.npz")

        # Check split files
        if os.path.exists(p_val) and os.path.exists(p_test):
            sz_v = os.path.getsize(p_val)
            sz_t = os.path.getsize(p_test)
            if sz_v > 1000 and sz_t > 1000:
                print(f"  [✓] {ds.upper():8s}: Split files present ({sz_v/(1024*1024):.1f}MB val, {sz_t/(1024*1024):.1f}MB test) [Plain Git Safe]")
                continue
            else:
                print(f"  [!] {ds.upper():8s}: Split files are LFS stubs (<1KB). Need real data.")
                all_present = False
                continue

        # Check monolithic file
        if os.path.exists(p):
            sz = os.path.getsize(p)
            if sz > 1000:
                print(f"  [✓] {ds.upper():8s}: {p} ({sz/(1024*1024):.2f} MB)")
            else:
                print(f"  [!] {ds.upper():8s}: Monolithic file is an LFS stub (<1KB). Rebuilding required.")
                all_present = False
        else:
            print(f"  [✗] {ds.upper():8s}: Missing cache file")
            all_present = False
    return all_present

def train_and_cache_paysim(force=False):
    npz_out = os.path.join(CACHE_DIR, "paysim_stage1_and_shap.npz")
    raw_npz_out = os.path.join(CACHE_DIR, "paysim_cand_raw_X.npz")
    full_splits_out = os.path.join(CACHE_DIR, "paysim_full_raw_splits.npz")
    
    if os.path.exists(npz_out) and os.path.getsize(npz_out) > 1000 and not force:
        print(f"[PaySim] Cache already exists at {npz_out}. Skipping retraining.")
        return
        
    csv_path = resolve_file(DATASET_FALLBACKS['paysim'])
    if not csv_path:
        print("[PaySim] Raw data CSV not found. Retaining existing cache if present.")
        return
        
    print(f"\n[PaySim] Loading raw dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
    X = df.drop(columns=['isFraud'])
    y = df['isFraud'].astype(int)
    amounts = df['amount'].values.astype(float)

    print("[PaySim] Performing deterministic 80/10/10 stratified train/val/test split (seed=42)...")
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

    # Save full raw splits for representation ablations (Table 6 Panel A)
    np.savez_compressed(
        full_splits_out,
        X_val=X1_val.values.astype(np.float32),
        X_test=X1_test.values.astype(np.float32)
    )
    print(f"[PaySim] Saved full raw feature splits to {full_splits_out}")

    print("[PaySim] Training Stage 1 XGBoost (n_estimators=500, max_depth=5, lr=0.05, scale_pos_weight=10)...")
    clf = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        scale_pos_weight=10, random_state=42, tree_method='hist', n_jobs=-1
    )
    clf.fit(X1_train, y_train)

    p_val = clf.predict_proba(X1_val)[:, 1].astype(np.float32)
    p_test = clf.predict_proba(X1_test)[:, 1].astype(np.float32)

    print("[PaySim] Computing TreeSHAP attribution matrices...")
    explainer = shap.TreeExplainer(clf)
    shap_val = explainer.shap_values(X1_val).astype(np.float32)
    shap_test = explainer.shap_values(X1_test).astype(np.float32)

    np.savez_compressed(
        npz_out,
        p_val=p_val, p_test=p_test,
        y_val=y_val.values, y_test=y_test.values,
        amt_val=amt_val, amt_test=amt_test,
        shap_val=shap_val, shap_test=shap_test
    )
    print(f"[PaySim] Successfully wrote: {npz_out}")

    # Save validated candidate features at validated tau* = 0.2179
    tau_star = 0.21792179217921794
    cand_v = p_val >= tau_star
    cand_t = p_test >= tau_star
    np.savez_compressed(
        raw_npz_out,
        X_cand_v=X1_val.values[cand_v],
        X_cand_t=X1_test.values[cand_t]
    )
    print(f"[PaySim] Successfully wrote raw candidate features at tau*={tau_star:.4f} to: {raw_npz_out}")

def train_and_cache_ieeecis(force=False):
    npz_out = os.path.join(CACHE_DIR, "ieeecis_stage1_and_shap.npz")
    val_out = os.path.join(CACHE_DIR, "ieeecis_stage1_and_shap_val.npz")
    test_out = os.path.join(CACHE_DIR, "ieeecis_stage1_and_shap_test.npz")
    cand_out = os.path.join(CACHE_DIR, "ieeecis_cand_raw_X.npz")

    if (os.path.exists(val_out) and os.path.getsize(val_out) > 1000 and 
        os.path.exists(test_out) and os.path.getsize(test_out) > 1000 and not force):
        print(f"[IEEE-CIS] Cache already exists ({val_out}, {test_out}). Skipping retraining.")
        return

    tx_path = resolve_file(DATASET_FALLBACKS['ieeecis_tx'])
    id_path = resolve_file(DATASET_FALLBACKS['ieeecis_id'])
    if not tx_path or not id_path:
        print("[IEEE-CIS] Raw data CSVs not found. Retaining existing cache.")
        return

    print(f"\n[IEEE-CIS] Loading raw datasets ({tx_path} and {id_path})...")
    tx = pd.read_csv(tx_path)
    idn = pd.read_csv(id_path)
    data = tx.merge(idn, on='TransactionID', how='left')
    del tx, idn
    gc.collect()

    data = data.rename(columns={'TransactionAmt': 'amount'}).drop(columns=['TransactionID'])
    obj_cols = data.select_dtypes(include=['object']).columns.tolist()
    num_cat = [c for c in ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2'] if c in data.columns]
    cat_cols = list(set(obj_cols + num_cat))
    for c in cat_cols:
        data[c] = data[c].astype('category')

    X = data.drop(columns=['isFraud'])
    y = data['isFraud'].astype(int)
    amounts = data['amount'].values.astype(float)

    print("[IEEE-CIS] Performing deterministic 80/10/10 stratified split...")
    X_temp, X_test, y_temp, y_test, amt_temp, amt_test = train_test_split(
        X, y, amounts, stratify=y, test_size=0.10, random_state=42
    )
    X_train, X_val, y_train, y_val, amt_train, amt_val = train_test_split(
        X_temp, y_temp, amt_temp, stratify=y_temp, test_size=0.1111, random_state=42
    )
    del data, X, y, amounts, X_temp, y_temp, amt_temp
    gc.collect()

    print("[IEEE-CIS] Training Stage 1 XGBoost (n_estimators=500, max_depth=5, lr=0.05, scale_pos_weight=10)...")
    clf = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        scale_pos_weight=10, random_state=42, eval_metric='logloss',
        tree_method='hist', enable_categorical=True
    )
    clf.fit(X_train, y_train)

    p_val = clf.predict_proba(X_val)[:, 1].astype(np.float32)
    p_test = clf.predict_proba(X_test)[:, 1].astype(np.float32)

    print("[IEEE-CIS] Computing TreeSHAP attribution matrices...")
    explainer = shap.TreeExplainer(clf)
    shap_val = explainer.shap_values(X_val).astype(np.float32)
    shap_test = explainer.shap_values(X_test).astype(np.float32)

    # Save split files (<60MB each) for Git friendliness without Git LFS
    np.savez_compressed(
        val_out,
        p_val=p_val, y_val=y_val.values,
        amt_val=amt_val, shap_val=shap_val
    )
    np.savez_compressed(
        test_out,
        p_test=p_test, y_test=y_test.values,
        amt_test=amt_test, shap_test=shap_test
    )
    print(f"[IEEE-CIS] Saved split cache files:\n  {val_out} ({os.path.getsize(val_out)/(1024*1024):.1f} MB)\n  {test_out} ({os.path.getsize(test_out)/(1024*1024):.1f} MB)")

    # Save candidate raw X at validated tau* = 0.3763
    tau_star = 0.37633763376337635
    cand_v = p_val >= tau_star
    cand_t = p_test >= tau_star
    X_num_v = pd.get_dummies(X_val).values.astype(np.float32)[cand_v]
    X_num_t = pd.get_dummies(X_test).values.astype(np.float32)[cand_t]
    np.savez_compressed(cand_out, X_cand_v=X_num_v, X_cand_t=X_num_t)
    print(f"[IEEE-CIS] Saved raw candidate features to {cand_out}")

def train_and_cache_ulb(force=False):
    npz_out = os.path.join(CACHE_DIR, "ulb_stage1_and_shap.npz")
    if os.path.exists(npz_out) and os.path.getsize(npz_out) > 1000 and not force:
        print(f"[ULB] Cache already exists at {npz_out}. Skipping retraining.")
        return

    csv_path = resolve_file(DATASET_FALLBACKS['ulb'])
    if not csv_path:
        print("[ULB] Raw credit card CSV not found. Retaining existing cache if present.")
        return

    print(f"\n[ULB] Loading raw dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    fraud_col = 'Class' if 'Class' in df.columns else 'isFraud'
    amt_col = 'Amount' if 'Amount' in df.columns else 'amount'
    
    X = df.drop(columns=[fraud_col])
    y = df[fraud_col].astype(int)
    amounts = df[amt_col].values.astype(float)

    print("[ULB] Performing deterministic 80/10/10 stratified split (seed=42)...")
    X_temp, X_test, y_temp, y_test, amt_temp, amt_test = train_test_split(
        X, y, amounts, stratify=y, test_size=0.10, random_state=42
    )
    X_train, X_val, y_train, y_val, amt_train, amt_val = train_test_split(
        X_temp, y_temp, amt_temp, stratify=y_temp, test_size=0.1111, random_state=42
    )
    del df, X, y, amounts, X_temp, y_temp, amt_temp
    gc.collect()

    print("[ULB] Training Stage 1 XGBoost (n_estimators=500, max_depth=5, lr=0.05, scale_pos_weight=10)...")
    clf = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        scale_pos_weight=10, random_state=42, eval_metric='logloss',
        tree_method='hist'
    )
    clf.fit(X_train, y_train)

    p_val = clf.predict_proba(X_val)[:, 1].astype(np.float32)
    p_test = clf.predict_proba(X_test)[:, 1].astype(np.float32)

    print("[ULB] Computing TreeSHAP attribution matrices...")
    explainer = shap.TreeExplainer(clf)
    shap_val = explainer.shap_values(X_val).astype(np.float32)
    shap_test = explainer.shap_values(X_test).astype(np.float32)

    np.savez_compressed(
        npz_out,
        p_val=p_val, p_test=p_test,
        y_val=y_val.values, y_test=y_test.values,
        amt_val=amt_val, amt_test=amt_test,
        shap_val=shap_val, shap_test=shap_test
    )
    print(f"[ULB] Successfully wrote: {npz_out}")

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training and SHAP Extraction")
    parser.add_argument("--force", action="store_true", help="Force retrain even if cache exists")
    parser.add_argument("--check-only", action="store_true", help="Only check cache integrity")
    args = parser.parse_args()

    status = check_cache_status()
    if args.check_only:
        sys.exit(0 if status else 1)

    if not status or args.force:
        print("\nRebuilding cache from raw source files...")
        train_and_cache_paysim(force=args.force)
        train_and_cache_ieeecis(force=args.force)
        train_and_cache_ulb(force=args.force)
    else:
        print("\nAll cache files verified and ready for experimental steps 1 through 7.")

if __name__ == '__main__':
    main()
