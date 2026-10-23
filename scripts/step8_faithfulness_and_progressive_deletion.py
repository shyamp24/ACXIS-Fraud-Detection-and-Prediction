#!/usr/bin/env python3
"""
scripts/step8_faithfulness_and_progressive_deletion.py
=============================================================================
PROGRESSIVE FEATURE DELETION FAITHFULNESS BENCHMARK (FIGURE 9)
Evaluates standard Most-Relevant-First (MoRF) vs. Least-Relevant-First (LeRF)
Progressive Feature Deletion Protocols (Samek et al., 2017; Petsiuk et al., 2018)
alongside Global XGBoost Gain and 50-replicate Random baseline across the
candidate alert pool (p >= tau* = 0.2179, N = 1,528) on PaySim.

Evaluates progressive feature deletion across PaySim candidate alerts.

Outputs:
- results_json/exp5_faithfulness_benchmarks.json
- results_csv/table9_faithfulness_curves.csv
- figures/Fig_SHAP_Faithfulness_Curves.png (Fig. 9)
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(PKG_DIR, "cache")
RESULTS_JSON_DIR = os.path.join(PKG_DIR, "results_json")
RESULTS_CSV_DIR = os.path.join(PKG_DIR, "results_csv")
FIGURES_DIR = os.path.join(PKG_DIR, "figures")

os.makedirs(RESULTS_JSON_DIR, exist_ok=True)
os.makedirs(RESULTS_CSV_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def calc_trapz(y, dx=1.0):
    """Numpy 1.x and 2.x compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=dx))
    return float(np.trapz(y, dx=dx))


def main():
    print("=" * 80)
    print("STEP 8: PROGRESSIVE FEATURE DELETION FAITHFULNESS BENCHMARK (FIG. 9)")
    print("=" * 80)

    # 1. Load cached splits and predictions
    splits = np.load(os.path.join(CACHE_DIR, "paysim_full_raw_splits.npz"))
    X_test = splits["X_test"]

    d = np.load(os.path.join(CACHE_DIR, "paysim_stage1_and_shap.npz"))
    p_test = d["p_test"]
    shap_test = d["shap_test"]
    y_test = d["y_test"]

    with open(os.path.join(RESULTS_JSON_DIR, "exp1_stage1_coopt_grid.json")) as f:
        grid_d = json.load(f)
    tau_star = float(grid_d["paysim"]["optimal_tau_validation"])
    print(f"Loaded validation-optimal threshold tau* = {tau_star:.4f}")

    cand_mask = (p_test >= tau_star)
    X_cand = X_test[cand_mask].copy()
    shap_cand = shap_test[cand_mask].copy()
    y_cand = y_test[cand_mask]
    n_cand = len(X_cand)
    n_frauds = int(np.sum(y_cand == 1))
    n_fps = int(np.sum(y_cand == 0))

    print(f"Candidate Alerts on Test: N = {n_cand:,} (Frauds: {n_frauds:,}, False Alarms: {n_fps:,})")

    xgb = XGBClassifier()
    xgb.load_model(os.path.join(CACHE_DIR, "paysim_xgb_stage1.json"))

    feature_names = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
    ]

    # Baseline predicted probability
    p_cand_orig = xgb.predict_proba(X_cand)[:, 1].astype(np.float32)
    baseline_p_mean = float(np.mean(p_cand_orig))
    print(f"Baseline mean predicted fraud probability (k=0): {baseline_p_mean:.4f}")

    # Attribution Mass Shares & Top Driver Frequencies
    top_feat_idx = np.argmax(np.abs(shap_cand), axis=1)
    top_freq = {}
    for idx, name in enumerate(feature_names):
        cnt = int(np.sum(top_feat_idx == idx))
        top_freq[name] = {"count": cnt, "pct": float(cnt / n_cand * 100)}

    mean_abs_shap = np.mean(np.abs(shap_cand), axis=0)
    total_shap = np.sum(mean_abs_shap)
    mass_share = {}
    for idx, name in enumerate(feature_names):
        mass_share[name] = {
            "mean_abs_shap": float(mean_abs_shap[idx]),
            "mass_share_pct": float(mean_abs_shap[idx] / total_shap * 100)
        }

    # 2. Deletion Curves (Protocol A: Median, Protocol B: Mean)
    baselines_A = np.median(X_test, axis=0)
    baselines_B = np.mean(X_test, axis=0)

    # TreeSHAP MoRF and LeRF
    morf_rank = np.argsort(-shap_cand, axis=1)
    lerf_rank = np.argsort(shap_cand, axis=1)

    morf_A, lerf_A = [baseline_p_mean], [baseline_p_mean]
    morf_B, lerf_B = [baseline_p_mean], [baseline_p_mean]

    for k in range(1, len(feature_names) + 1):
        X_mA, X_lA = X_cand.copy(), X_cand.copy()
        X_mB, X_lB = X_cand.copy(), X_cand.copy()
        for i in range(n_cand):
            del_m = morf_rank[i, :k]
            del_l = lerf_rank[i, :k]
            X_mA[i, del_m] = baselines_A[del_m]
            X_lA[i, del_l] = baselines_A[del_l]
            X_mB[i, del_m] = baselines_B[del_m]
            X_lB[i, del_l] = baselines_B[del_l]

        pmA = float(np.mean(xgb.predict_proba(X_mA)[:, 1]))
        plA = float(np.mean(xgb.predict_proba(X_lA)[:, 1]))
        pmB = float(np.mean(xgb.predict_proba(X_mB)[:, 1]))
        plB = float(np.mean(xgb.predict_proba(X_lB)[:, 1]))

        morf_A.append(pmA)
        lerf_A.append(plA)
        morf_B.append(pmB)
        lerf_B.append(plB)
        print(f"k={k:2d}: MoRF_A={pmA:.4f}, LeRF_A={plA:.4f} | MoRF_B={pmB:.4f}, LeRF_B={plB:.4f}")

    # Global Gain Feature Ordering
    gain = xgb.feature_importances_
    gain_order = np.argsort(-gain)
    gain_A, gain_B = [baseline_p_mean], [baseline_p_mean]
    for k in range(1, len(feature_names) + 1):
        del_k = gain_order[:k]
        X_gA, X_gB = X_cand.copy(), X_cand.copy()
        X_gA[:, del_k] = baselines_A[del_k]
        X_gB[:, del_k] = baselines_B[del_k]
        gain_A.append(float(np.mean(xgb.predict_proba(X_gA)[:, 1])))
        gain_B.append(float(np.mean(xgb.predict_proba(X_gB)[:, 1])))

    # 50-Replicate Random Baseline
    np.random.seed(42)
    n_reps = 50
    rand_A_reps, rand_B_reps = [], []
    for rep in range(n_reps):
        r_order = np.random.permutation(len(feature_names))
        rA_curve, rB_curve = [baseline_p_mean], [baseline_p_mean]
        for k in range(1, len(feature_names) + 1):
            del_k = r_order[:k]
            X_rA, X_rB = X_cand.copy(), X_cand.copy()
            X_rA[:, del_k] = baselines_A[del_k]
            X_rB[:, del_k] = baselines_B[del_k]
            rA_curve.append(float(np.mean(xgb.predict_proba(X_rA)[:, 1])))
            rB_curve.append(float(np.mean(xgb.predict_proba(X_rB)[:, 1])))
        rand_A_reps.append(rA_curve)
        rand_B_reps.append(rB_curve)

    rand_A = list(np.mean(rand_A_reps, axis=0))
    rand_B = list(np.mean(rand_B_reps, axis=0))

    audc_morf_A = float(calc_trapz(morf_A, dx=1.0) / len(feature_names))
    audc_lerf_A = float(calc_trapz(lerf_A, dx=1.0) / len(feature_names))
    audc_gain_A = float(calc_trapz(gain_A, dx=1.0) / len(feature_names))
    audc_rand_A = float(calc_trapz(rand_A, dx=1.0) / len(feature_names))

    audc_morf_B = float(calc_trapz(morf_B, dx=1.0) / len(feature_names))
    audc_lerf_B = float(calc_trapz(lerf_B, dx=1.0) / len(feature_names))
    audc_gain_B = float(calc_trapz(gain_B, dx=1.0) / len(feature_names))
    audc_rand_B = float(calc_trapz(rand_B, dx=1.0) / len(feature_names))

    print(f"\nAUDC Protocol A (Median): MoRF={audc_morf_A:.4f}, Gain={audc_gain_A:.4f}, Random={audc_rand_A:.4f}, LeRF={audc_lerf_A:.4f}")
    print(f"AUDC Protocol B (Mean)  : MoRF={audc_morf_B:.4f}, Gain={audc_gain_B:.4f}, Random={audc_rand_B:.4f}, LeRF={audc_lerf_B:.4f}")

    # 3. Single-feature permutation reliance
    base_pr_auc = float(average_precision_score(y_test, p_test))
    np.random.seed(42)
    X_perm = X_test.copy()
    X_perm[:, 2] = np.random.permutation(X_perm[:, 2])  # oldbalanceOrg
    p_perm = xgb.predict_proba(X_perm)[:, 1]
    perm_pr_auc = float(average_precision_score(y_test, p_perm))
    perm_drop_pct = float((base_pr_auc - perm_pr_auc) / base_pr_auc * 100)
    print(f"\nPermutation reliance on oldbalanceOrg: Base PR-AUC = {base_pr_auc:.4f} -> Permuted = {perm_pr_auc:.4f} (-{perm_drop_pct:.2f}%)")

    # 4. Save results to JSON
    results = {
        "candidate_alerts": {
            "total_alerts": n_cand,
            "true_frauds": n_frauds,
            "false_alarms": n_fps,
            "baseline_mean_p": baseline_p_mean
        },
        "top_attribution_frequency": top_freq,
        "overall_attribution_mass_share": mass_share,
        "protocol_a_median": {
            "morf": [round(v, 4) for v in morf_A],
            "gain": [round(v, 4) for v in gain_A],
            "random": [round(v, 4) for v in rand_A],
            "lerf": [round(v, 4) for v in lerf_A],
            "audc_morf": round(audc_morf_A, 4),
            "audc_gain": round(audc_gain_A, 4),
            "audc_random": round(audc_rand_A, 4),
            "audc_lerf": round(audc_lerf_A, 4)
        },
        "protocol_b_mean": {
            "morf": [round(v, 4) for v in morf_B],
            "gain": [round(v, 4) for v in gain_B],
            "random": [round(v, 4) for v in rand_B],
            "lerf": [round(v, 4) for v in lerf_B],
            "audc_morf": round(audc_morf_B, 4),
            "audc_gain": round(audc_gain_B, 4),
            "audc_random": round(audc_rand_B, 4),
            "audc_lerf": round(audc_lerf_B, 4)
        },
        "single_feature_reliance": {
            "feature": "oldbalanceOrg",
            "base_pr_auc": round(base_pr_auc, 4),
            "permuted_pr_auc": round(perm_pr_auc, 4),
            "drop_pct": round(perm_drop_pct, 2)
        }
    }

    out_json = os.path.join(RESULTS_JSON_DIR, "exp5_faithfulness_benchmarks.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Saved JSON to: {out_json}")

    # 5. Save CSV
    df_curves = pd.DataFrame({
        "features_deleted_k": list(range(len(feature_names) + 1)),
        "morf_protocol_a": [round(v, 4) for v in morf_A],
        "gain_protocol_a": [round(v, 4) for v in gain_A],
        "random_protocol_a": [round(v, 4) for v in rand_A],
        "lerf_protocol_a": [round(v, 4) for v in lerf_A],
        "morf_protocol_b": [round(v, 4) for v in morf_B],
        "gain_protocol_b": [round(v, 4) for v in gain_B],
        "random_protocol_b": [round(v, 4) for v in rand_B],
        "lerf_protocol_b": [round(v, 4) for v in lerf_B]
    })
    out_csv = os.path.join(RESULTS_CSV_DIR, "table9_faithfulness_curves.csv")
    df_curves.to_csv(out_csv, index=False)
    print(f"[✓] Saved CSV to: {out_csv}")

    # 6. Plot Figure 9 with all 4 benchmarks
    plt.rcParams["font.family"] = "Times New Roman"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)
    k_steps = np.arange(len(feature_names) + 1)

    # Panel A: Protocol A (Median)
    ax1.plot(k_steps, morf_A, 'o-', color='#B22222', lw=2.2, label=f'TreeSHAP MoRF (AUDC={audc_morf_A:.4f})')
    ax1.plot(k_steps, gain_A, '^-', color='#FF8C00', lw=1.8, label=f'Global XGBoost Gain (AUDC={audc_gain_A:.4f})')
    ax1.plot(k_steps, rand_A, 'd:', color='#708090', lw=1.8, label=f'Random Baseline (AUDC={audc_rand_A:.4f})')
    ax1.plot(k_steps, lerf_A, 's--', color='#1E90FF', lw=2.0, label=f'TreeSHAP LeRF (AUDC={audc_lerf_A:.4f})')
    ax1.set_title('(a) Protocol A: Training Median Replacement', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel('Number of Deleted Features (k)', fontsize=10)
    ax1.set_ylabel('Mean Predicted Fraud Probability (p̄)', fontsize=10)
    ax1.set_xticks(k_steps)
    ax1.grid(True, ls=':', alpha=0.6)
    ax1.legend(loc='center right', fontsize=8.2, framealpha=0.95)
    ax1.set_ylim(-0.02, 0.95)

    # Panel B: Protocol B (Marginal Mean)
    ax2.plot(k_steps, morf_B, 'o-', color='#B22222', lw=2.2, label=f'TreeSHAP MoRF (AUDC={audc_morf_B:.4f})')
    ax2.plot(k_steps, gain_B, '^-', color='#FF8C00', lw=1.8, label=f'Global XGBoost Gain (AUDC={audc_gain_B:.4f})')
    ax2.plot(k_steps, rand_B, 'd:', color='#708090', lw=1.8, label=f'Random Baseline (AUDC={audc_rand_B:.4f})')
    ax2.plot(k_steps, lerf_B, 's--', color='#1E90FF', lw=2.0, label=f'TreeSHAP LeRF (AUDC={audc_lerf_B:.4f})')
    ax2.set_title('(b) Protocol B: Marginal Mean Replacement', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel('Number of Deleted Features (k)', fontsize=10)
    ax2.set_ylabel('Mean Predicted Fraud Probability (p̄)', fontsize=10)
    ax2.set_xticks(k_steps)
    ax2.grid(True, ls=':', alpha=0.6)
    ax2.legend(loc='center right', fontsize=8.2, framealpha=0.95)
    ax2.set_ylim(-0.02, 0.95)

    plt.tight_layout()
    out_fig = os.path.join(FIGURES_DIR, "Fig_SHAP_Faithfulness_Curves.png")
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved Figure 9 to: {out_fig}")
    print("=" * 80)

if __name__ == "__main__":
    main()
