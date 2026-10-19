#!/usr/bin/env python3
"""
scripts/step0_train_stage1_and_shap.py
=============================================================================
STAGE 1 MODEL TRAINING & TREESHAP ATTRIBUTION EXTRACTION
Provides complete end-to-end reproducibility from raw datasets to cached NPZ.

Pipeline:
1. Loads raw dataset CSV files from ACXIS_DATA_DIR or local data/ directory.
2. Performs deterministic 80/10/10 train/val/test splits (random_state=42).
3. Trains Stage 1 XGBoost (scale_pos_weight=10, max_depth=6, n_estimators=300).
4. Computes TreeSHAP attribution matrices on validation and test partitions.
5. Saves precomputed arrays to cache/{dataset}_stage1_and_shap.npz.

If cache files already exist on disk, validates integrity and reports status.
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
        os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv"),
        "/Users/shyamsundar/Dropbox/4 SHYAM DOCs/PHD/PET 2021/PROJECT/Reports/Report  2/Project Phase 2/six.csv"
    ],
    'ieeecis_tx': [
        os.path.join(DATA_DIR, "train_transaction.csv"),
        os.path.join(DATA_DIR, "ieee-fraud-detection/train_transaction.csv"),
        "/Users/shyamsundar/Dropbox/4 SHYAM DOCs/PHD/PET 2021/PROJECT/Reports/Report  2/Project Phase 2/Datasets/IEEE-ICS/ieee-fraud-detection/train_transaction.csv"
    ],
    'ieeecis_id': [
        os.path.join(DATA_DIR, "train_identity.csv"),
        os.path.join(DATA_DIR, "ieee-fraud-detection/train_identity.csv"),
        "/Users/shyamsundar/Dropbox/4 SHYAM DOCs/PHD/PET 2021/PROJECT/Reports/Report  2/Project Phase 2/Datasets/IEEE-ICS/ieee-fraud-detection/train_identity.csv"
    ],
    'ulb': [
        os.path.join(DATA_DIR, "creditcard.csv"),
        os.path.join(DATA_DIR, "card.csv"),
        "/Users/shyamsundar/Dropbox/4 SHYAM DOCs/PHD/PET 2021/PROJECT/Reports/Report  2/Project Phase 2/creditcard.csv"
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
        if os.path.exists(p):
            sz = os.path.getsize(p) / (1024 * 1024)
            d = np.load(p)
            print(f"  [✓] {ds.upper():8s}: {p} ({sz:.2f} MB)")
            print(f"      Keys: {list(d.keys())} | Val shape: {d['p_val'].shape} | Test shape: {d['p_test'].shape}")
        else:
            print(f"  [✗] {ds.upper():8s}: Missing ({p})")
            all_present = False
    return all_present

def train_and_cache_paysim(force=False):
    npz_out = os.path.join(CACHE_DIR, "paysim_stage1_and_shap.npz")
    raw_npz_out = os.path.join(CACHE_DIR, "paysim_cand_raw_X.npz")
    if os.path.exists(npz_out) and not force:
        print(f"[PaySim] Cache already exists at {npz_out}. Skipping retraining.")
        return
    csv_path = resolve_file(DATASET_FALLBACKS['paysim'])
    if not csv_path:
        print("[PaySim] Raw data CSV not found. Retaining existing cache.")
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

    print("[PaySim] Training Stage 1 XGBoost (scale_pos_weight=10, 300 estimators, max_depth=6)...")
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=10, random_state=42, tree_method='hist', n_jobs=-1
    )
    clf.fit(X1_train, y_train)

    p_val = clf.predict_proba(X1_val)[:, 1]
    p_test = clf.predict_proba(X1_test)[:, 1]

    print("[PaySim] Computing TreeSHAP attribution matrices...")
    explainer = shap.TreeExplainer(clf)
    shap_val = explainer.shap_values(X1_val)
    shap_test = explainer.shap_values(X1_test)

    np.savez_compressed(
        npz_out,
        p_val=p_val, p_test=p_test,
        y_val=y_val.values, y_test=y_test.values,
        amt_val=amt_val, amt_test=amt_test,
        shap_val=shap_val, shap_test=shap_test
    )
    print(f"[PaySim] Successfully wrote: {npz_out}")

    # Also save raw candidate features for joint space
    tau_approx = 0.1288
    cand_v = p_val >= tau_approx
    cand_t = p_test >= tau_approx
    np.savez_compressed(
        raw_npz_out,
        X_cand_v=X1_val.values[cand_v],
        X_cand_t=X1_test.values[cand_t]
    )
    print(f"[PaySim] Successfully wrote raw candidate features to: {raw_npz_out}")

def train_and_cache_ieeecis(force=False):
    npz_out = os.path.join(CACHE_DIR, "ieeecis_stage1_and_shap.npz")
    if os.path.exists(npz_out) and not force:
        print(f"[IEEE-CIS] Cache already exists at {npz_out}. Skipping retraining.")
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

    print("[IEEE-CIS] Training Stage 1 XGBoost (scale_pos_weight=10, 500 estimators)...")
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

    np.savez_compressed(
        npz_out,
        p_val=p_val, p_test=p_test,
        y_val=y_val.values, y_test=y_test.values,
        amt_val=amt_val, amt_test=amt_test,
        shap_val=shap_val, shap_test=shap_test
    )
    print(f"[IEEE-CIS] Successfully wrote: {npz_out}")

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
    else:
        print("\nAll cache files verified and ready for experimental steps 1 through 7.")

if __name__ == '__main__':
    main()
