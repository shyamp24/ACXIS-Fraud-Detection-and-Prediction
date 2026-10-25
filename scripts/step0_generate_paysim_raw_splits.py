#!/usr/bin/env python3
"""
scripts/step0_generate_paysim_raw_splits.py
=============================================================================
PAY-SIM RAW SPLIT GENERATOR FOR ABLATION EXPERIMENTS (TABLE 6)
Deterministic, reproducible extraction of raw preprocessed validation and test
splits directly from raw PaySim transaction data.

Outputs:
- cache/paysim_full_raw_splits.npz (X_val: float32, X_test: float32)
- cache/paysim_cand_raw_X.npz (X_cand_v, X_cand_t filtered at tau* = 0.2179)
=============================================================================
"""

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PKG_DIR, "cache")
DATA_DIR = os.environ.get("ACXIS_DATA_DIR", os.path.join(PKG_DIR, "data"))

CANDIDATES = [
    os.path.join(DATA_DIR, "six.csv"),
    os.path.join(DATA_DIR, "paysim.csv"),
    os.path.join(DATA_DIR, "PS_20174392719_1491204439457_log.csv")
    ]

def resolve_file():
    for p in CANDIDATES:
        if os.path.isfile(p):
            return p
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate PaySim Raw Splits for Table 6 Ablation")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing cache")
    args = parser.parse_args()

    out_file = os.path.join(CACHE_DIR, "paysim_full_raw_splits.npz")
    cand_out = os.path.join(CACHE_DIR, "paysim_cand_raw_X.npz")

    if os.path.isfile(out_file) and not args.force:
        print(f"[PaySim Splits] Cache file already exists at {out_file} ({os.path.getsize(out_file)/(1024*1024):.2f} MB). Skipping.")
        return

    csv_path = resolve_file()
    if not csv_path:
        print(f"[PaySim Splits] Error: PaySim CSV not found in search paths: {CANDIDATES}")
        print("Please place six.csv or paysim.csv in the data/ directory or set ACXIS_DATA_DIR.")
        sys.exit(1)

    print(f"[PaySim Splits] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
    X = df.drop(columns=['isFraud'])
    y = df['isFraud'].astype(int)
    amounts = df['amount'].values.astype(float)

    print("[PaySim Splits] Performing deterministic 80/10/10 stratified split (seed=42)...")
    X_temp, X_test, y_temp, y_test, amt_temp, amt_test = train_test_split(
        X, y, amounts, stratify=y, test_size=0.10, random_state=42
    )
    X_train, X_val, y_train, y_val, amt_train, amt_val = train_test_split(
        X_temp, y_temp, amt_temp, stratify=y_temp, test_size=0.1111, random_state=42
    )
    del df, X, y, amounts, X_temp, y_temp, amt_temp
    gc.collect()

    print("[PaySim Splits] One-hot encoding categorical attributes...")
    X1_train = pd.get_dummies(X_train).astype(np.float32)
    X1_val = pd.get_dummies(X_val).astype(np.float32)
    X1_test = pd.get_dummies(X_test).astype(np.float32)
    X1_train, X1_val = X1_train.align(X1_val, join='left', axis=1, fill_value=0)
    X1_train, X1_test = X1_train.align(X1_test, join='left', axis=1, fill_value=0)

    X_val_mat = X1_val.values.astype(np.float32)
    X_test_mat = X1_test.values.astype(np.float32)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(out_file, X_val=X_val_mat, X_test=X_test_mat)
    print(f"[PaySim Splits] [✓] Successfully saved full splits to {out_file} (Val: {X_val_mat.shape}, Test: {X_test_mat.shape})")

    # If stage1 cache exists, also update candidate features at validated tau* = 0.2179
    stage1_npz = os.path.join(CACHE_DIR, "paysim_stage1_and_shap.npz")
    if os.path.isfile(stage1_npz) and os.path.getsize(stage1_npz) > 1000:
        d = np.load(stage1_npz)
        tau_star = 0.21792179217921794
        cand_v = d['p_val'] >= tau_star
        cand_t = d['p_test'] >= tau_star
        np.savez_compressed(cand_out, X_cand_v=X_val_mat[cand_v], X_cand_t=X_test_mat[cand_t])
        print(f"[PaySim Splits] [✓] Saved candidate features at tau*={tau_star:.4f} to {cand_out} ({cand_v.sum()} val, {cand_t.sum()} test)")

if __name__ == '__main__':
    main()
