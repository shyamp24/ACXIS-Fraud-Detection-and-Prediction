# ACXIS Precomputed Cache Directory

This directory contains verified, deterministic precomputed Stage 1 model predictions and TreeSHAP attribution matrices to enable immediate replication without requiring multi-gigabyte raw data downloads.

---

## 1. Cached Artifact Inventory

| Artifact Filename | Size | Description | Partition / Role |
| :--- | :---: | :--- | :--- |
| `paysim_stage1_and_shap.npz` | 57.20 MB | PaySim Stage 1 test probabilities and TreeSHAP matrix ($N=636,262$) | Held-Out Test |
| `paysim_full_raw_splits.npz` | 18.95 MB | PaySim partitioned raw train/val/test feature matrices | Train / Val / Test |
| `paysim_cand_raw_X.npz` | 47,333 bytes | PaySim candidate alerts raw feature matrix ($\hat{p} \ge \tau^*$, $N=1,528$) | Held-Out Test |
| `paysim_xgb_stage1.json` | 1.37 MB | PaySim trained Stage 1 XGBoost model architecture & booster weights | Model Booster |
| `ieeecis_stage1_and_shap_val.npz` | 58.39 MB | IEEE-CIS validation split probabilities and TreeSHAP attributions | Validation Tuning |
| `ieeecis_stage1_and_shap_test.npz` | 58.34 MB | IEEE-CIS blind test split probabilities and TreeSHAP attributions | Blind Test |
| `ieeecis_cand_raw_X.npz` | 1,014,446 bytes | IEEE-CIS candidate alerts raw feature matrix ($\hat{p} \ge \tau^*$, $N=3,446$) | Held-Out Test |
| `ulb_stage1_and_shap.npz` | 6.39 MB | ULB Credit Card Stage 1 test probabilities and TreeSHAP matrix | Held-Out Test |
| `ulb_suppressed_test_alerts.npz` | 864 bytes | ULB suppressed test alerts mask (864 bytes; 0 alerts suppressed) | Boundary Condition |

---

## 2. Technical Cache Architecture Notes

### A. IEEE-CIS Partitioning Strategy (`_val.npz` and `_test.npz`)
The monolithic IEEE-CIS TreeSHAP attribution matrix exceeds 116 MB due to the 432-feature tabular representation. To strictly adhere to GitHub's 100 MB per-file repository limit without introducing external Git-LFS dependencies, the IEEE-CIS cache is cleanly partitioned into:
- `ieeecis_stage1_and_shap_val.npz` (58.39 MB)
- `ieeecis_stage1_and_shap_test.npz` (58.34 MB)

The pipeline script `scripts/cache_utils.py` transparently resolves and merges these split NPZ files on-the-fly.

### B. ULB Suppressed Alerts Cache Size (864 bytes)
The file `cache/ulb_suppressed_test_alerts.npz` is intentionally compact (864 bytes) because it contains:
- `supp_indices`: an empty array `shape (0,)`
- `supp_mask`: binary array of zeros `shape (46,)`
- `q_test`: meta-classifier probabilities `shape (46,)`

This reflects the mathematically expected operational abstention of the ACXIS amount-tiered safety guard on the ULB benchmark: because the safety threshold converges to infinity ($\tau_{\text{meta}}^* \to \infty$) to prevent false negative leakage, zero false alarms are suppressed ($\Delta\mathrm{FN} = 0$, $\Delta\mathrm{FP} = 0$). The 864-byte file is valid, complete, and uncorrupted.
