# ACXIS: Replication Package
## Adaptive Cost-Sensitive XGBoost with SHAP-Based Integrated Suppression

This repository contains the complete replication code, dataset pre-processing scripts, benchmark evaluation suites, statistical test logs, and publication figures for the manuscript:

> **ACXIS: An Adaptive Cost-Sensitive XGBoost Framework with SHAP-based Integrated Suppression for Financial Fraud Detection and Prediction**  
> *Authors:* Shyamshundar N. Patil, Dr. Manoj E. Patil  
> *Journal:* Elsevier *Array* (Manuscript Reference: `ARRAY-D-26-03794R2`)

---

## 1. Architectural Highlights: Framework Implementation

This replication package implements the unified ACXIS framework:

1. **Validation-Derived Operating Threshold ($\tau^*$):**  
   The Stage 1 operating threshold $\tau_1 = \tau^*$ is determined strictly on the independent validation partition $\mathcal{D}_{\text{val}}$ by minimizing the total asymmetric operational cost:
   $$\tau^* = \arg\min_{\tau \in [0.01, 0.99]} \text{Cost}(\mathcal{D}_{\text{val}}, \tau)$$
   Test data is strictly kept blind and is never used for threshold selection.
   - **PaySim Validation Optimum:** $\tau^* = 0.2179$
   - **IEEE-CIS Validation Optimum:** $\tau^* = 0.3763$
   - **ULB Validation Optimum:** $\tau^* = 0.4555$

2. **Amount-Tiered Safety Guard:**  
   Candidate alerts ($\hat{p} \ge \tau^*$) are partitioned into amount quantiles ($K=3$ equal-frequency bins). Within each quantile tier $k$, an independent safety threshold $\tau_{\text{meta}, k}$ is calibrated on 5-fold out-of-fold (OOF) cross-validation predictions:
   $$\tau_{\text{meta}, k} = \max_{i \in \text{Val-Fraud}_k} q_{\text{OOF}, i} + \varepsilon$$
   Candidate alerts in test partition whose meta-probability satisfies $q_i \ge \tau_{\text{meta}, k}$ are suppressed as verified false alarms. If a tier's calibrated threshold reaches 1.0 or suppresses zero alerts out-of-fold, the guard safely abstains ($\tau_{\text{meta}, k} = \infty$).

3. **5-Fold Cross-Fitting Meta-Ensemble:**  
   To stabilize test scoring and mitigate fold-specific variance, test candidate alerts are scored by the ensemble average of the five cross-fitted fold models $\bar{q}(x) = \frac{1}{5}\sum_{j=1}^5 m_j(x)$. While out-of-fold calibration uses single-fold held-out scores, ensembling across the fold models on test evaluations yields conservative, variance-reduced probability estimates.

---

## 2. Directory Structure

```
06_ACXIS_GitHub_Replication_Package_Final/
├── README.md                                  # Complete replication documentation
├── run_all.py                                 # Master orchestrator script
├── data/                                      # Dataset instructions and raw files
│   └── README.md                              # Download links, file aliases, and MD5 hashes
├── scripts/                                   # Standalone execution scripts
│   ├── cache_utils.py                         # Centralized robust cache loader (handles split NPZs)
│   ├── step0_generate_paysim_raw_splits.py    # PaySim deterministic raw split generator
│   ├── step0_train_stage1_and_shap.py         # Step 0: End-to-end training & TreeSHAP calculation
│   ├── step1_data_and_stage1_coopt.py         # Step 1: 100-step validation cost minimization sweep
│   ├── step2_algorithm1_engine.py             # Step 2: Algorithm 1 implementation & evaluation
│   ├── step3_table4_master_baselines.py       # Step 3: Comparative baseline benchmarks (13 models)
│   ├── step4_table6_ablation_panels.py        # Step 4: Table 6 ablation panels (Panels A, B, C)
│   ├── step5_bootstrap_resampling_1000.py     # Step 5: 1,000-iteration stratified bootstrap resampling
│   ├── step6_latency_timing_loop.py           # Step 6: Measured single-row latency & batch throughput
│   └── step7_generate_publication_figures.py  # Step 7: Publication vector & high-DPI figures
├── cache/                                     # Cached arrays & model artifacts (.npz, .json)
├── results_csv/                               # All generated tables in CSV format
│   ├── table1_algorithm1_summary.csv          # Summary of Algorithm 1 across datasets
│   ├── table2_experimental_settings.csv       # Complete hyperparameter settings
│   ├── table3_latency_and_throughput.csv      # Single-row & batch throughput metrics
│   ├── table4_baseline_benchmarks.csv         # Side-by-side comparative benchmarks
│   ├── table5_bootstrap_distribution_*.csv    # Full 1,000 empirical bootstrap draws
│   ├── table5_resampling_ci_master.csv        # 1,000 bootstrap resampling means & 95% CIs
│   └── table6_ablation_master.csv             # Representation & architecture ablations
├── results_json/                              # Structured JSON results
│   ├── exp1_stage1_coopt_grid.json            # 100-step validation cost curves across datasets
│   ├── exp2_algorithm1_results.json           # Algorithm 1 exact metrics & tier thresholds
│   ├── exp3_table4_baselines.json             # All baseline metrics (default & tuned)
│   ├── exp4_table6_ablation.json              # Ablation metrics across Panels A, B, C
│   ├── exp5_bootstrap_results.json            # Full 1,000 bootstrap distributions
│   └── exp6_latency_results.json              # Hardware timing loop percentiles
├── figures/                                   # Publication-ready figures (PDF & PNG)
│   ├── fig1_acxis_architecture.pdf / .png     # Vector architecture diagram
│   ├── fig4_before_after_confusion_matrix.pdf # 2-panel confusion matrix
│   ├── fig4_cost_vs_threshold_curves.png      # 100-step threshold cost curves
│   ├── fig7_amount_tier_breakdown.pdf / .png  # Quantile tier suppression distribution
│   └── fig10_bootstrap_robustness.pdf / .png  # Bootstrap cost reduction density plots
└── logs/                                      # Console execution logs
    ├── 00_stage1_and_shap.log
    ├── 01_stage1_coopt.log
    ├── 02_algorithm1.log
    ├── 03_table4_baselines.log
    ├── 04_table6_ablation.log
    ├── 05_bootstrap.log
    ├── 06_latency.log
    └── 07_figures.log
```

---

## 3. Verified Experimental Results Summary

All results reported below are generated by executing the pipeline directly on local hardware without interpolation or manual estimation:

### A. Core Performance Across Datasets (Table 1 & Table 5)
| Dataset | Operating Point ($\tau^*$) | Base Cost | ACXIS Cost | Currency Savings | Cost Reduction (%) | 95% Bootstrap CI | FPs Suppressed | FNs Leaked |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PaySim** | $\tau^* = 0.2179$ | ₹2,200,948.26 | ₹2,139,948.26 | **₹61,000.00** | **+2.77%** (Mean: 2.83% ± 0.75%) | [1.63%, 4.52%] | **42** (41.7 ± 6.5) | **0** (0.00 ± 0.00) |
| **IEEE-CIS** | $\tau^* = 0.3763$ | $94,856.71 | $94,536.71 | **$320.00** | **+0.34%** (Mean: 0.34% ± 0.07%) | [0.21%, 0.49%] | **30** (30.0 ± 5.6) | **0** (0.00 ± 0.00) |
| **ULB** | $\tau^* = 0.4555$ | €750.61 | €750.61 | **€0.00** | **0.00%** (0.00% ± 0.00%) | [0.00%, 0.00%] | **0** (0.0 ± 0.0) | **0** (0.00 ± 0.00) |

*Empirical Boundary Condition (ULB):* ACXIS transparently abstains on ULB (€0.00 savings, 0 false alarms suppressed) because the safety guard identifies that candidate false alarm attributions overlap with fraud attributions within the tiers. Rather than risking a false negative, the framework safely retains all Stage 1 alerts, providing an honest empirical boundary condition.

### B. Single-Row Tail Latency & Batch Throughput (Table 3)
Hardware: Apple Silicon workstation, Python 3.12, scikit-learn, XGBoost, SHAP.

| Processing Stage | Mean Latency | Standard Deviation | Median ($p_{50}$) | $p_{95}$ Tail | $p_{99}$ Tail | Measured Throughput ($B=1000$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Preprocessing & Encoding** | 0.0027 ms | 0.0022 ms | 0.0019 ms | 0.0037 ms | 0.0045 ms | — |
| **Stage 1 XGBoost Fast-Path** | 0.6201 ms | 1.4341 ms | 0.5124 ms | 0.8902 ms | 1.8668 ms | **680,116.4 tx/sec** |
| **TreeSHAP Attribution** | 1.1362 ms | 2.7582 ms | 0.8796 ms | 1.3726 ms | 4.5441 ms | **8,400.7 tx/sec** |
| **Stage 3 RF Meta-Inference** | 5.4506 ms | 4.7011 ms | 4.8322 ms | 8.0003 ms | 15.5813 ms | **97,364.9 tx/sec** |
| **Full Triggered Pipeline** | 9.4991 ms | 5.9086 ms | 8.9583 ms | 10.0343 ms | 29.7184 ms | **7,646.5 tx/sec** |
| *Baseline LIME Explainer* | 179.5568 ms | 30.9808 ms | 170.2082 ms | 269.1889 ms | 309.2068 ms | — |

---

## 4. How to Reproduce All Results

### Prerequisites
- Python $\ge 3.10$
- Dependencies:
  ```bash
  pip install numpy pandas scikit-learn lightgbm catboost xgboost shap imbalanced-learn scipy matplotlib
  ```

### Master Single-Command Reproduction
To execute the pipeline from the verified cache:
```bash
python3 run_all.py
```

To re-train Stage 1 XGBoost and regenerate SHAP attributions from the raw CSV datasets from scratch:
```bash
python3 run_all.py --from-scratch
```

### Step-by-Step Reproduction
Each experiment can also be executed independently:
```bash
# Step 0: End-to-end Stage 1 XGBoost training & TreeSHAP matrix generation
python3 scripts/step0_train_stage1_and_shap.py

# Step 1: Validation cost minimization sweep & threshold selection (tau*)
python3 scripts/step1_data_and_stage1_coopt.py

# Step 2: Algorithm 1 engine with Amount-Tiered Safety Guard
python3 scripts/step2_algorithm1_engine.py

# Step 3: Comparative baseline benchmarks (13 models side-by-side)
python3 scripts/step3_table4_master_baselines.py

# Step 4: Full ablation benchmark across Panels A, B, and C
python3 scripts/step4_table6_ablation_panels.py

# Step 5: 1,000-iteration stratified bootstrap resampling
python3 scripts/step5_bootstrap_resampling_1000.py

# Step 6: Dedicated hardware latency timing loops & batch throughput
python3 scripts/step6_latency_timing_loop.py

# Step 7: Synchronized publication vector figures
python3 scripts/step7_generate_publication_figures.py
```

---

## 5. Methodological & Statistical Notes

1. **Threshold Derivation Transparency:** Stage 1 operating thresholds ($\tau^*$) are derived strictly by minimizing asymmetric operational cost on the validation partition $\mathcal{D}_{\text{val}}$. Test labels are withheld until blind evaluation.
2. **Resampling Scope:** Reported bootstrap 95% confidence intervals reflect 1,000 stratified resamples of the test partition to measure sampling variability under fixed operational model weights.
3. **Safety Guard Guarantee:** "Zero leaks" ($\Delta\text{FN} = 0$) represents the verified empirical outcome on the blind test partition under the validation-calibrated amount-tiered safety guard.
4. **Independent Baselines:** Cost-sensitive and sampling baselines (SMOTE, ADASYN, Bayes Minimum Risk, Instance-Weighted CS-XGBoost, LightGBM, CatBoost) are executed independently with both default ($\tau = 0.5$) and validation-tuned operating points.
5. **Direct Hardware Timing:** Batch throughput ($B=1000$) is measured in dedicated vectorized execution loops, avoiding reciprocal approximations ($1/\text{latency}$).
