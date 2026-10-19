# Dataset Ingestion & Configuration Guide

This directory documents the public financial fraud datasets utilized in evaluating the **ACXIS** framework.

---

## 1. Dataset Provenance & Download Specifications

All three benchmark datasets are publicly accessible open-source benchmarks:

### Dataset 1: PaySim Mobile Money Fraud Simulation
- **Reference:** Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. In 28th European Modeling and Simulation Symposium (EMSS).
- **Public Repository:** [Kaggle PaySim1 Benchmark](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Scale:** 6,362,620 financial transactions | 11 attributes | 8,213 fraud instances (0.129% base rate).
- **Supported File Names:**
  - `six.csv` *(Internal project name denoting the 6.36M transaction log)*
  - `paysim.csv`
  - `PS_20174392719_1491204439457_log.csv` *(Original Kaggle extraction filename)*
- **Required Columns:** `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.

### Dataset 2: IEEE-CIS Financial Fraud Detection
- **Reference:** IEEE Computational Intelligence Society & Vesta Corporation (2019).
- **Public Repository:** [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data)
- **Scale:** 590,540 real-world e-commerce transactions | 434 total features | 20,663 frauds (3.50% base rate).
- **Required Files:**
  - `train_transaction.csv` (Transaction amounts, cards, addresses, P/R email domains, V-features)
  - `train_identity.csv` (Device identity, network features, browser metadata)
- **Merge Key:** `TransactionID` (left join: `tx.merge(idn, on='TransactionID', how='left')`).

### Dataset 3: ULB European Credit Card Fraud Benchmark
- **Reference:** Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating probability with undersampling for unbalanced classification*. In IEEE SSCI.
- **Public Repository:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Scale:** 284,807 transactions | 30 PCA features + Amount + Class | 492 frauds (0.173% base rate).
- **Supported File Names:**
  - `creditcard.csv`
  - `card.csv`

---

## 2. Setting Up Data for End-to-End Retraining

If you wish to re-train the models from scratch (`python run_all.py --from-scratch`), place the raw CSV files either:
1. Directly in this `data/` directory:
   ```
   data/
   ├── six.csv                        (or paysim.csv)
   ├── train_transaction.csv
   ├── train_identity.csv
   └── creditcard.csv                 (or card.csv)
   ```
2. Or set the environment variable pointing to your local storage location:
   ```bash
   export ACXIS_DATA_DIR="/path/to/your/raw/csv/directory"
   ```

---

## 3. Precomputed Cache Mode (Zero Download Required)

For immediate replication without downloading multi-gigabyte raw datasets, the `cache/` directory contains complete, verified precomputed Stage 1 model predictions and TreeSHAP attribution matrices:
- `cache/paysim_stage1_and_shap.npz` (57.2 MB)
- `cache/ieeecis_stage1_and_shap_val.npz` (58.4 MB)
- `cache/ieeecis_stage1_and_shap_test.npz` (58.3 MB)
- `cache/ulb_stage1_and_shap.npz` (6.4 MB)
- `cache/paysim_full_raw_splits.npz` (18.9 MB)
- `cache/paysim_cand_raw_X.npz` (68 KB)
- `cache/ieeecis_cand_raw_X.npz` (987 KB)
- `cache/paysim_xgb_stage1.json` (1.4 MB)

All files are strictly $< 60$ MB and are directly version-controlled as plain Git objects. No Git LFS installation or external download bandwidth is required.
