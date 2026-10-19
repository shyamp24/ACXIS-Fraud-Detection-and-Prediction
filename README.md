# ACXIS: Replication Package
## Adaptive Cost-Sensitive XGBoost with SHAP-Based Integrated Suppression

This repository contains the complete replication code, dataset pre-processing scripts, benchmark evaluation suites, multi-seed statistical logs, and publication figures for the manuscript:

> **ACXIS: An Adaptive Cost-Sensitive XGBoost Framework with SHAP-based Integrated Suppression for Financial Fraud Detection and Prediction**  
> *Authors:* Shyamshundar N. Patil, Dr. Manoj E. Patil  
> *Journal:* Elsevier *Array* (Manuscript Reference: `ARRAY-D-26-03794R2`)  
> *Public Repository:* [https://github.com/shyamp24/ACXIS-Fraud-Detection-and-Prediction](https://github.com/shyamp24/ACXIS-Fraud-Detection-and-Prediction)

---

## Table of Contents
1. [Framework Architecture & Mechanics](#1-framework-architecture--mechanics)
2. [Data Provenance, Fixed Seeds & Exact Split Protocols](#2-data-provenance-fixed-seeds--exact-split-protocols)
3. [Hyperparameter Settings & Preprocessing Pipelines (Table 2)](#3-hyperparameter-settings--preprocessing-pipelines-table-2)
4. [Validation Threshold Sweep & Extended Boundary Curve](#4-validation-threshold-sweep--extended-boundary-curve)
5. [Operational Cost Schedules, Flat Penalties & Break-Even Analysis](#5-operational-cost-schedules-flat-penalties--break-even-analysis)
6. [Comparative Baseline Benchmarks: 13 Literature Models (Table 4)](#6-comparative-baseline-benchmarks-13-literature-models-table-4)
7. [Comprehensive Ablation Studies: Panels A, B, and C (Table 6)](#7-comprehensive-ablation-studies-panels-a-b-and-c-table-6)
8. [Data Resampling & Statistical Significance (Tables 5 & 8)](#8-data-resampling--statistical-significance-tables-5--8)
9. [Hardware Timing Loops, Tail Latencies & Exact Function Calls (Table 3)](#9-hardware-timing-loops-tail-latencies--exact-function-calls-table-3)
10. [Repository Structure & Quickstart Reproduction](#10-repository-structure--quickstart-reproduction)

---

## 1. Framework Architecture & Mechanics

ACXIS addresses the operational false-alarm dilemma in high-throughput transaction screening through a three-stage decoupled architecture:

```
Raw Transaction x
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Fast-Path XGBoost Classifier                  │
│ Compute fraud probability: p̂ = f_xgb(x)               │
└────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────── p̂ < τ* ──────────────► Direct Approval (Bulk Traffic)
       │
       ▼ (Candidate Alerts: p̂ ≥ τ*)
┌────────────────────────────────────────────────────────┐
│ Stage 2: Validation-Derived Operating Threshold (τ*)   │
│ τ* = argmin Cost(D_val, τ)                             │
│ PaySim: τ* = 0.2179 | IEEE-CIS: τ* = 0.3763 | ULB: 0.4555
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ TreeSHAP Attribution Extraction                        │
│ Compute local feature attribution vector: φ(x)         │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Meta-Classifier (Random Forest Ensemble)      │
│ 5-fold cross-fitting ensemble: q̄(x) = (1/5) ∑ m_j(x)   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Amount-Tiered Operational Safety Guard                 │
│ Quantile bins K=3; τ_meta,k = max_{Val-Fraud_k} q_OOF + ε
└────────────────────────────────────────────────────────┘
       │
       ├── q̄ ≥ τ_meta,k ──► Verified False Alarm ──► Suppress Alert (Approved)
       └── q̄ < τ_meta,k ──► Unresolved Alert     ──► Escalated to Human Review
```

### Key Operational Guarantees:
1. **Validation-Derived Operating Point ($\tau^*$):** The Stage 1 threshold $\tau^*$ is determined strictly on the independent validation partition $\mathcal{D}_{\text{val}}$ by minimizing the total asymmetric operational cost. The test partition remains strictly blind during parameter tuning.
2. **Amount-Tiered Safety Guard:** Candidate alerts ($\hat{p} \ge \tau^*$) are stratified into $K=3$ transaction amount quantile tiers. Within each tier $k$, an independent suppression threshold is calibrated against 5-fold out-of-fold (OOF) cross-validation predictions:
   $$\tau_{\text{meta}, k} = \max_{i \in \text{Val-Fraud}_k} q_{\text{OOF}, i} + \varepsilon$$
   If an amount tier suppresses zero false positives out-of-fold or reaches $\tau_{\text{meta}, k} \ge 1.0$, it automatically abstains ($\tau_{\text{meta}, k} = \infty$).
3. **Recall Preservation ($\Delta\mathrm{FN} = 0$):** Under the safety guard, only high-confidence false positives are suppressed; no additional false negatives are introduced on test evaluations across all evaluated random seeds.

---

## 2. Data Provenance, Fixed Seeds & Exact Split Protocols

### Deterministic Pseudo-Random Seeds
All experiments are initialized with fixed, documented random seeds for strict determinism:
- Primary pipeline seed: `random_state = 42` (Stage 1 split, model training, TreeSHAP, and Stage 3 cross-fitting).
- Data resampling protocol: 1,000 stratified bootstrap test resamples drawn under `random_state = 42` to quantify partition variance.

### Exact Partitioning Protocol per Dataset
| Dataset | Total Transactions | Total Frauds | Fraud Rate | Train Partition (80%) | Validation Partition (10%) | Test Partition (10%) | Split Strategy & Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PaySim** | 6,362,620 | 8,213 | 0.129% | 5,089,928 tx | 636,241 tx | 636,243 tx | **Stratified Split (`random_state=42`):** PaySim is an agent-based synthetic simulation where simulation steps do not model seasonal macro-drift. Stratified splitting preserves the extreme 0.129% class ratio across validation and test partitions. |
| **IEEE-CIS** | 590,540 | 20,663 | 3.50% | 472,432 tx | 59,054 tx | 59,054 tx | **Stratified Split (`random_state=42`):** Merged transaction and identity tables (432 features). Stratified splitting ensures equal representation of complex device/identity multi-attribute fraud patterns across calibration and test sets. |
| **ULB Credit** | 284,807 | 492 | 0.173% | 227,845 tx | 28,481 tx | 28,481 tx | **Stratified Split (`random_state=42`):** Extreme imbalance benchmark (492 frauds across 284k records). Stratification guarantees sufficient fraud samples in validation (49) and test (49). |

### PaySim Typology Restriction Analysis (TRANSFER / CASH_OUT)
In PaySim, all 8,213 fraudulent transactions occur strictly within `TRANSFER` and `CASH_OUT` transaction types (Lopez-Rojas et al., 2016). We evaluated ACXIS across three distinct ingestion settings:
1. **Full Operational Stream (Primary Benchmark, 6.36M tx):** Evaluates all transaction types with one-hot encoded `type` to model operational screening where non-fraudulent transaction types (`PAYMENT`, `CASH_IN`, `DEBIT`) pass through Stage 1. Net savings: **₹61,000.00 (+2.77%)**, $\Delta\mathrm{FN} = 0$.
2. **Dedicated TRANSFER + CASH_OUT Subset (2.77M tx):** Filtering exclusively to the 2,770,409 candidate transactions yields: Stage 1 baseline cost ₹5,336,347; Stage 2 tuned cost ($\tau^*=0.1879$) ₹1,763,733; Stage 3 ACXIS cost ₹1,710,233 (Recall: 98.05%, Precision: 48.9%). Incremental net savings: **₹53,500.00 (+3.03%)**, $\Delta\mathrm{FN} = 0$.
3. **Forward-Chaining Chronological Temporal Split (Full 6.36M tx):** Sorted by simulation timestamp `step` (first 80% train, next 10% val, last 10% test): Stage 1 baseline cost at $\tau=0.5$ incurs ₹132,349,950; Stage 2 tuned ($\tau^*=0.0370$) reduces cost to ₹4,808,912 (Recall: 98.49%); Stage 3 ACXIS further suppresses 135 false alarms, reducing cost to ₹4,554,412. Incremental net savings: **₹254,500.00 (+5.29%)**, $\Delta\mathrm{FN} = 0$.

---

## 3. Hyperparameter Settings & Preprocessing Pipelines (Table 2)

All model architectures, hyperparameter grids, and preprocessing steps are fully synchronized with manuscript Table 2:

### Feature Preprocessing
- **PaySim:** Drop identifier columns (`nameOrig`, `nameDest`, `isFlaggedFraud`). One-hot encode categorical transaction `type`. Retain `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`.
- **IEEE-CIS:** Left merge `train_transaction.csv` and `train_identity.csv` on `TransactionID`. Categorical features (`card1`–`card6`, `addr1`–`addr2`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`, `id_12`–`id_38`) encoded via categorical codes; numerical features imputed with $-1.0$.
- **ULB:** Features `V1`–`V28` (pre-standardized PCA components) and `Amount`. Robust scaling applied to `Amount`.

### Model Hyperparameters
| Component | Architecture | Hyperparameters & Configuration | Search Grid / Tuning Policy |
| :--- | :--- | :--- | :--- |
| **Stage 1 Primary** | XGBoost (`XGBClassifier`) | `n_estimators=500`, `max_depth=5`, `learning_rate=0.05`, `scale_pos_weight=10.0`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method='hist'`, `random_state=42` | Grid search over `max_depth` $\in \{4, 5, 6, 8\}$, `scale_pos_weight` $\in \{1, 5, 10, 20\}$ on $\mathcal{D}_{\text{val}}$. |
| **Stage 3 Meta** | Random Forest (`RandomForestClassifier`) | `n_estimators=100`, `max_depth=5`, `class_weight='balanced'`, `max_features='sqrt'`, `n_jobs=-1`, `random_state=42` | Regularized tree depth ($\le 5$) prevents memorization of rare fraud attributions. |
| **Meta Ensemble** | 5-Fold Cross-Fitting | Out-of-fold calibration on single-fold predictions; test predictions scored by full 5-model average: $\bar{q}(x) = \frac{1}{5}\sum_{j=1}^5 m_j(x)$ | Mitigates fold-specific variance and stabilizes test scoring. |
| **CatBoost Baseline** | CatBoost (`CatBoostClassifier`) | `iterations=300`, `depth=6`, `learning_rate=0.05`, `auto_class_weights='Balanced'`, `random_seed=42` | Tuned depth and iteration budget matched to XGBoost. |
| **LightGBM Baseline** | LightGBM (`LGBMClassifier`) | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=10`, `random_state=42` | Standard GBDT baseline. |
| **CS-XGBoost** | Instance-Weighted XGBoost | `sample_weight = Amount` for positive class, $C_{\text{FP}}(\text{Amount})$ for negative class | Implements Höppner et al. (2022) cost-weighting. |
| **Bayes Min. Risk** | Decision-Theoretic BMR | $\tau^*_{\text{BMR}}(i) = C_{\text{FP}}(i) / (C_{\text{FP}}(i) + C_{\text{FN}}(i))$ | Implements Bahnsen et al. (2014) / Elkan (2001). |

---

## 4. Validation Threshold Sweep & Extended Boundary Curve

### 100-Step Validation Cost Optimization Grid
The Stage 1 operating threshold $\tau^*$ is selected by an exhaustive 100-step grid search over $\tau \in [0.01, 0.99]$ on the validation split $\mathcal{D}_{\text{val}}$:
$$\tau^* = \arg\min_{\tau \in [0.01, 0.99]} \left( \sum_{i \in \text{FN}(\tau)} \text{Amount}_i + \sum_{j \in \text{FP}(\tau)} C_{\text{FP}}(\text{Amount}_j) \right)$$
- **PaySim:** $\tau^* = 0.2179$ (Validation cost: ₹2,391,599.76; Test cost: ₹2,200,948.26).
- **IEEE-CIS:** $\tau^* = 0.3763$ (Validation cost: $93,700.71; Test cost: $94,856.71).
- **ULB:** $\tau^* = 0.4555$ (Validation cost: €1,202.81; Test cost: €750.61).

### Extended Boundary Exploration [0.001, 0.990]
To address Reviewer 5's question regarding ultra-low thresholds ($\tau \le 0.001$), we extended the sweep to $\tau \in [0.001, 0.990]$:
| Dataset | Operating $\tau^*$ | Cost at $\tau^*$ | Cost at $\tau = 0.01$ | Cost at $\tau = 0.001$ | Cost Ratio ($\tau=0.001$ vs $\tau^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PaySim** | $0.2179$ | ₹2,391,599.76 | ₹11,046,851.26 | ₹31,593,351.26 | **13.21× higher cost** |
| **IEEE-CIS** | $0.3763$ | $93,700.71 | $409,625.56 | $537,895.00 | **5.74× higher cost** |
| **ULB** | $0.4555$ | €1,202.81 | €1,744.81 | €4,460.14 | **3.71× higher cost** |

*Methodological Conclusion:* Ultra-low thresholds flood operational investigation queues with massive volumes of false positives, driving investigation overhead to unmanageable levels. The cost function is strictly convex, confirming that the global cost minimum is bounded within the interior interval $[0.20, 0.50]$.

---

## 5. Operational Cost Schedules, Flat Penalties & Break-Even Analysis

### Framing of Operational Cost Schedules
In accordance with cost-sensitive learning literature (Elkan, 2001; Hand et al., 2000), false negative costs represent direct financial chargeback losses equal to the transaction amount ($C_{\text{FN}} = \text{Amount}$). False positive costs represent administrative review overheads. As stated in the manuscript, tiered operational cost schedules represent **illustrative operational assumptions**:
- **PaySim (₹ INR):** Amount $\le 50\text{k} \to ₹500$; $\le 100\text{k} \to ₹1,000$; $\le 150\text{k} \to ₹1,500$; $\le 200\text{k} \to ₹2,000$; $\le 500\text{k} \to ₹3,000$; $\le 1\text{M} \to ₹5,000$; $> 1\text{M} \to ₹10,000$.
- **IEEE-CIS (\$ USD):** Amount $\le \$68.5 \to \$5$; $\le \$125 \to \$10$; $\le \$276.85 \to \$15$; $\le \$554 \to \$20$; $\le \$1,651 \to \$30$; $\le \$2,755.95 \to \$50$; $> \$2,755.95 \to \$100$.
- **ULB (€ EUR):** Amount $\le €20 \to €2$; $\le €50 \to €5$; $\le €100 \to €10$; $\le €250 \to €15$; $\le €500 \to €25$; $\le €1,000 \to €50$; $> €1,000 \to €100$.

### Flat Penalty & Proportional Cost Evaluation
Alongside tiered schedules, ACXIS was evaluated under flat administrative fees and proportional review percentages:
- **Flat Penalties:**
  - PaySim flat ₹1,000: Net savings = **₹42,000.00**; flat ₹1,500: Net savings = **₹63,000.00** ($\Delta\mathrm{FN} = 0$).
  - IEEE-CIS flat \$10: Net savings = **\$300.00**; flat \$15: Net savings = **\$450.00** ($\Delta\mathrm{FN} = 0$).
  - ULB flat €10 / €15: Net savings = **€0.00** (safely abstains).
- **Proportional Investigation Costs (1.0% and 5.0% of Amount):**
  - PaySim 1.0%: Net savings = **₹31,452.18**; 5.0%: Net savings = **₹157,260.90** ($\Delta\mathrm{FN} = 0$).
  - IEEE-CIS 1.0%: Net savings = **$41.20**; 5.0%: Net savings = **$206.00** ($\Delta\mathrm{FN} = 0$).

### Break-Even Sensitivity Analysis over $C_{\text{FN}} / C_{\text{FP}}$ Ratios
Because ACXIS operates with strictly zero false negative leakage ($\Delta\mathrm{FN} = 0$), every suppressed alert is a pure false positive ($42$ on PaySim, $30$ on IEEE-CIS). Consequently, net financial savings:
$$\Delta\text{Cost} = \sum_{j \in \text{Suppressed FPs}} C_{\text{FP}}(j)$$
are **mathematically guaranteed to remain strictly positive for any non-zero investigation cost ($C_{\text{FP}} > 0$)**. Sweeping $C_{\text{FN}} / C_{\text{FP}}$ ratios across $2:1$ to $1,000:1$ confirms that net financial savings remain strictly positive across all evaluated regimes (+0.97% to +2.63% on PaySim; +0.07% to +13.32% on IEEE-CIS).

---

## 6. Comparative Baseline Benchmarks: 13 Literature Models (Table 4)

All 13 baseline models were executed on identical dataset splits at both Default ($\tau = 0.5$) and Validation-Tuned ($\tau = \tau^*$) thresholds:

| Dataset | Model Category | Model Name | ROC-AUC | PR-AUC | Cost (Default $\tau=0.5$) | Cost (Tuned $\tau=\tau^*$) | Recall (Tuned) | Precision (Tuned) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PaySim** | Classical ML | Logistic Regression | 0.9887 | 0.5682 | ₹56,100,610 | ₹29,132,570 | 0.9329 | 0.0521 |
| | Classical ML | Decision Tree | 0.9917 | 0.7722 | ₹8,598,279 | ₹7,552,479 | 0.8841 | 0.8240 |
| | Classical ML | Random Forest | 0.9991 | 0.8870 | ₹19,255,390 | ₹9,666,656 | 0.8951 | 0.9234 |
| | Modern GBDT | LightGBM | 0.9492 | 0.3900 | ₹47,311,170 | ₹44,031,810 | 0.7927 | 0.1245 |
| | Modern GBDT | CatBoost (Fairly Tuned) | 0.9986 | 0.9540 | ₹2,737,506 | ₹3,113,489 | 0.9793 | 0.9125 |
| | Sampling | SMOTE + XGBoost | 0.9985 | 0.9250 | ₹7,894,599 | ₹8,705,982 | 0.9512 | 0.8456 |
| | Sampling | ADASYN + XGBoost | 0.9985 | 0.9085 | ₹9,250,484 | ₹9,351,167 | 0.9488 | 0.8321 |
| | Cost-Sensitive | Bayes Minimum Risk (Bahnsen 2014) | 0.9997 | 0.9610 | ₹10,226,160 | ₹10,226,160 | 0.9634 | 0.9012 |
| | Cost-Sensitive | CS-XGBoost (Höppner 2022) | 0.9996 | 0.9461 | ₹2,667,929 | ₹2,367,511 | 0.9829 | 0.7845 |
| | XAI Heuristic | XAI Heuristic Rule Filter | 0.9997 | 0.9610 | ₹181,066,000 | ₹181,066,000 | 0.8245 | 0.4512 |
| | **ACXIS Stages** | Stage 1 Core ($\tau=0.5$) | 0.9997 | 0.9610 | ₹7,433,385 | ₹7,433,385 | 0.9245 | 0.8161 |
| | **ACXIS Stages** | Stage 2 Tuned ($\tau^*=0.2179$) | 0.9997 | 0.9610 | ₹7,433,385 | ₹2,200,948 | 0.9842 | 0.5288 |
| | **ACXIS Stages** | **ACXIS Proposed (Stage 1+2+3)** | **0.9997** | **0.9610** | **₹7,433,385** | **₹2,139,948** | **0.9842** | **0.5437** |
| **IEEE-CIS** | Classical ML | Logistic Regression | 0.7722 | 0.1500 | $254,796 | $227,382 | 0.6512 | 0.0845 |
| | Classical ML | Decision Tree | 0.8576 | 0.4092 | $170,971 | $166,845 | 0.7123 | 0.3214 |
| | Classical ML | Random Forest | 0.8806 | 0.5053 | $168,816 | $169,944 | 0.7245 | 0.3845 |
| | Modern GBDT | LightGBM | 0.9395 | 0.6819 | $117,312 | $108,691 | 0.7654 | 0.4512 |
| | Modern GBDT | CatBoost (Fairly Tuned) | 0.9499 | 0.7041 | $101,753 | $98,960 | 0.7812 | 0.4689 |
| | Sampling | SMOTE + XGBoost | 0.9039 | 0.5959 | $230,444 | $133,057 | 0.7412 | 0.3912 |
| | Sampling | ADASYN + XGBoost | 0.9038 | 0.6014 | $224,695 | $135,782 | 0.7389 | 0.3876 |
| | Cost-Sensitive | Bayes Minimum Risk (Bahnsen 2014) | 0.9596 | 0.7713 | $181,625 | $181,625 | 0.7512 | 0.5123 |
| | Cost-Sensitive | CS-XGBoost (Höppner 2022) | 0.9114 | 0.6285 | $224,494 | $123,592 | 0.7545 | 0.4125 |
| | XAI Heuristic | XAI Heuristic Rule Filter | 0.9596 | 0.7713 | $249,246 | $249,246 | 0.6845 | 0.3125 |
| | **ACXIS Stages** | Stage 1 Core ($\tau=0.5$) | 0.9596 | 0.7713 | $107,018 | $107,018 | 0.7507 | 0.6267 |
| | **ACXIS Stages** | Stage 2 Tuned ($\tau^*=0.3763$) | 0.9596 | 0.7713 | $107,018 | $94,857 | 0.7948 | 0.4765 |
| | **ACXIS Stages** | **ACXIS Proposed (Stage 1+2+3)** | **0.9596** | **0.7713** | **$107,018** | **$94,537** | **0.7948** | **0.4807** |

*Key Findings:*
1. ACXIS Stage 2 threshold optimization captures the majority of savings, while Stage 3 safely suppresses remaining false alarms (+₹61,000 on PaySim, +$320 on IEEE-CIS) without leaking a single fraud.
2. CatBoost achieves $98,960 on IEEE-CIS (within 4.7% of ACXIS). ACXIS is framed as a safe operational refinement rather than an exaggerated breakthrough.

---

## 7. Comprehensive Ablation Studies: Panels A, B, and C (Table 6)

Table 6 evaluates representations, architectures, and cross-dataset generalizability:

### Panel A: Meta-Feature Representation Ablation (PaySim)
- **Probability-Only Filter $[\hat{p}]$:** Suppresses 0 FPs (₹0 savings, 0% reduction). Fails because probability alone cannot distinguish borderline frauds from true false alarms.
- **Raw Features Only $[x]$:** Suppresses 92 FPs, 0 FNs leaked (₹97,000 savings, +4.41%).
- **SHAP Attributions Only $[\phi(x)]$:** Suppresses 37 FPs, but **leaks 1 FN** (-₹266,452.42, -12.11% loss).
- **ACXIS Combined $[x, \phi(x)]$:** Suppresses 42 FPs, **0 FNs leaked** (+₹61,000.00, +2.77% net savings). Combining raw values with attributions stabilizes decision boundaries.

### Panel B: Meta-Classifier Architecture Ablation (PaySim)
- **Logistic Regression (L2):** Suppresses 74 FPs, but **leaks 4 FNs** (-₹205,607.30, -9.34% loss).
- **LightGBM:** Suppresses 126 FPs, but **leaks 1 FN** (+₹189,807.32, +8.62%; unacceptable fraud leakage).
- **Random Forest (ACXIS Proposed):** Suppresses 42 FPs, **0 FNs leaked** (+₹61,000.00, +2.77% net savings). Random Forest's bagged variance reduction prevents the boundary over-fitting that causes false negatives.

### Panel C: Cross-Dataset Validation & Raw vs SHAP Ablation (IEEE-CIS)
To address Reviewer 5's request to evaluate raw features vs SHAP on IEEE-CIS:
- **Baseline Stage 1 ($\tau^* = 0.3763$):** Cost = $94,856.71 (Recall: 0.7948).
- **Probability-Only Filter $[\hat{p}]$:** Suppresses 0 FPs ($0 savings, 0% reduction).
- **Raw Features Only $[x]$:** Suppresses 20 FPs, but **leaks 3 FNs** (Cost: $94,957.71; Net loss: **-$101.00**, -0.11%).
- **SHAP Attributions Only $[\phi(x)]$:** Suppresses 30 FPs, **0 FNs leaked** (Cost: $94,536.71; Net savings: **+$320.00**, +0.34%).
- **ACXIS Proposed $[\phi(x)] + 3 \text{ Tiers}$:** Suppresses 30 FPs, **0 FNs leaked** (Cost: $94,536.71; Net savings: **+$320.00**, +0.34%).

*Crucial Scientific Finding:* In high-dimensional spaces (IEEE-CIS, 432 features), raw features suffer from collinear noise that misclassifies true frauds as false alarms (3 FNs leaked). SHAP attributions act as an essential noise filter, isolating model reasoning to achieve zero false negative leakage.

---

## 8. Data Resampling & Statistical Significance (Tables 5 & 8)

To address Reviewer 1 and Reviewer 5's request for empirical data resampling beyond fixed-split seed variance, ACXIS evaluates transaction partition uncertainty via a **1,000-iteration stratified test bootstrap** (`scripts/step5_bootstrap_resampling_1000.py`). This protocol directly reflects operational banking deployment—where compliance engines are calibrated and frozen for fixed audit cycles—and quantifies streaming transaction volume uncertainty and class imbalance fluctuations while holding the compliance engine constant:

- **PaySim:** Mean Cost Reduction = **+2.83% ± 0.75%** [95% CI: 1.63%, 4.52%]; Suppressed FPs = 41.7 ± 6.5; Leaked FNs = 0.00 ± 0.00 (**0 leaks across all 1,000 draws**; $p_{95} \le 0.37\%$). On the full unresampled test partition, point savings are **₹61,000.00** (+2.77%).
- **IEEE-CIS:** Mean Cost Reduction = **+0.34% ± 0.07%** [95% CI: 0.21%, 0.49%]; Suppressed FPs = 30.0 ± 5.6; Leaked FNs = 0.00 ± 0.00 (**0 leaks across all 1,000 draws**; $p_{95} \le 0.37\%$). On the full unresampled test partition, point savings are **$320.00** (+0.34%).
- **ULB:** Mean Cost Reduction = **0.00% ± 0.00%** [95% CI: 0.00%, 0.00%]; Suppressed FPs = 0.0 ± 0.0; Leaked FNs = 0.00 ± 0.00 (Safely abstains with zero false negative leakage).

---

## 9. Hardware Timing Loops, Tail Latencies & Exact Function Calls (Table 3)

### Exact Function Calls Timed in `step6_latency_timing_loop.py`
All timing measurements are executed on genuine models using high-resolution monotonic clocks (`time.perf_counter_ns()`):
1. **Feature Preprocessing:** `pipe.transform(sample_row)`
2. **Stage 1 Fast-Path Inference:** `xgb_model.predict_proba(sample_row)[:, 1]`
3. **Stage 2 TreeSHAP Attribution:** `shap_explainer.shap_values(sample_row)`
4. **Stage 3 Meta-Classifier Inference:** `meta_rf.predict_proba(sample_attributions)[:, 1]`
5. **Full Triggered Pipeline:** End-to-end sequential execution of Preprocessing $\to$ Stage 1 $\to$ TreeSHAP $\to$ Stage 3 $\to$ Amount-Tiered Safety Guard.
6. **LIME Baseline Explainer:** `lime_explainer.explain_instance(sample_row[0], xgb_model.predict_proba, num_features=10)`
7. **Batch Throughput ($B=1000$):** Measured via vectorized batch execution: `xgb_model.predict_proba(batch_1000)[:, 1]` and `meta_rf.predict_proba(batch_1000)[:, 1]`.

### Measured Latency & Throughput Benchmark (Table 3)
Hardware: Apple Silicon workstation, Python 3.12, scikit-learn, XGBoost, SHAP.
| Pipeline Stage | Mean Latency | Standard Deviation | Median ($p_{50}$) | $p_{95}$ Tail | $p_{99}$ Tail | Measured Throughput ($B=1000$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Feature Preprocessing** | 0.0019 ms | 0.0020 ms | 0.0018 ms | 0.0019 ms | 0.0027 ms | — |
| **Stage 1 Core (Fast-Path XGBoost)** | 0.5467 ms | 0.3444 ms | 0.4624 ms | 0.8344 ms | 1.7158 ms | **227,967.4 tx/s** |
| **Stage 2 TreeSHAP Attribution** | 3.4912 ms | 0.8676 ms | 3.2926 ms | 4.3655 ms | 5.9078 ms | **1,301.7 tx/s** |
| **Stage 3 Meta-Classifier (Random Forest)** | 4.9470 ms | 1.7017 ms | 4.6395 ms | 6.0970 ms | 10.3198 ms | **137,259.3 tx/s** |
| **Full Triggered ACXIS Pipeline** | 11.6947 ms | 2.0084 ms | 11.0611 ms | 14.7062 ms | 19.2839 ms | **1,282.2 tx/s** |
| *Baseline LIME Explainer* | 175.6956 ms | 24.4376 ms | 167.9571 ms | 212.8126 ms | 262.7964 ms | — |

---

## 10. Repository Structure & Quickstart Reproduction

### Directory Map
```
06_ACXIS_GitHub_Replication_Package_Final/
├── README.md                                  # Complete replication documentation
├── run_all.py                                 # Master orchestrator script
├── data/
│   └── README.md                              # Download URLs, MD5 checksums & column schemas
├── scripts/
│   ├── cache_utils.py                         # Robust cache loader (transparently handles split NPZs)
│   ├── step0_generate_paysim_raw_splits.py    # Standalone PaySim split generator
│   ├── step0_train_stage1_and_shap.py         # Stage 1 training & TreeSHAP calculation
│   ├── step1_data_and_stage1_coopt.py         # 100-step validation cost minimization sweep
│   ├── step2_algorithm1_engine.py             # Algorithm 1 implementation & evaluation
│   ├── step3_table4_master_baselines.py       # Comparative baseline benchmarks (13 models)
│   ├── step4_table6_ablation_panels.py        # Ablation benchmark (Panels A, B, C)
│   ├── step5_bootstrap_resampling_1000.py     # 1,000-iteration stratified bootstrap resampling
│   ├── step6_latency_timing_loop.py           # Single-row tail latency & batch throughput
│   └── step7_generate_publication_figures.py  # Publication-ready vector figures
├── cache/                                     # Versioned plain binary caches (<59MB each, No Git LFS)
├── results_csv/                               # Generated tables in CSV format (Tables 1-6)
├── results_json/                              # Complete structured JSON benchmark logs
├── figures/                                   # High-DPI and vector publication figures
└── logs/                                      # Full console execution logs for all steps
```

### Quickstart Execution
To execute all experiments from the verified cache (takes ~25 minutes on standard hardware):
```bash
python3 run_all.py
```

To re-train all models from the raw CSV datasets from scratch:
```bash
python3 run_all.py --from-scratch
```
*(Ensure raw CSVs are placed in `data/` as documented in `data/README.md`).*

---

## Citation & Authorship
If using this code or benchmark data, please cite:
```bibtex
@article{patil2026acxis,
  title={ACXIS: An Adaptive Cost-Sensitive XGBoost Framework with SHAP-based Integrated Suppression for Financial Fraud Detection and Prediction},
  author={Patil, Shyamshundar N. and Patil, Manoj E.},
  journal={Array},
  volume={26},
  pages={100379},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.array.2026.100379}
}
```
