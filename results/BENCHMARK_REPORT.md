# Statistical Benchmark Report: Tabular ML, Deep Learning & Foundation Models

**Scope:** Empirical evaluation of classical, gradient boosted, deep tabular, and zero-shot transformer architectures for customer churn prediction.

---

## 1. Executive Summary

A common question in applied tabular machine learning is whether emerging Tabular Deep Learning architectures (e.g., **FT-Transformer**, Gorishniy et al., NeurIPS 2021) or zero-shot Tabular Foundation Models (e.g., **TabPFN**, Hollmann et al., Nature 2025 / ICLR 2023) outperform tuned Gradient Boosted Decision Trees (**LightGBM**, **CatBoost**, **XGBoost**).

### Key Empirical Findings
1. **Tree Supremacy Maintained**: Calibrated **LightGBM** achieved the highest cross-validated ROC-AUC (**0.8510**) and lowest Brier score (**0.1284**), outperforming FT-Transformer (0.8342) and TabPFN (0.8291).
2. **Statistical Significance Verified**: Using the **Nadeau-Bengio (2003) corrected repeated k-fold t-test**, the performance advantage of LightGBM over FT-Transformer is statistically significant ($p = 0.024$, Cohen's $d = 0.84$).
3. **Serving Latency Advantage**: LightGBM achieves **0.82 ms/sample** CPU serving latency — **55x faster than FT-Transformer** (45.3 ms) and **146x faster than TabPFN** (120.4 ms), critical for real-time CRM lookup SLAs.

---

## 2. Experimental Setup & Methodology

### Data Preprocessing & Leakage Control
- **Dataset**: IBM Telco Customer Churn (7,043 instances, 19 raw features).
- **Validation Scheme**: 5-Fold Stratified Cross-Validation ($K=5$).
- **Strict Pipeline Encapsulation**: All transformations (imputation, one-hot encoding, feature scaling, 28-feature engineering, and feature selection) were fitted exclusively on training folds to prevent data leakage.
- **Hardware Consistency**: Evaluated on identical x86_64 CPU compute instances to ensure fair latency benchmarking.

### Algorithms Evaluated
1. **Baselines**: Logistic Regression (L1/L2), Random Forest, Extra Trees.
2. **Modern GBDTs**: LightGBM, CatBoost, XGBoost.
3. **Tabular Deep Learning**: FT-Transformer (Feature Tokenizer + Transformer with multi-head self-attention).
4. **Prior-Data Fitted Networks**: TabPFN (zero-shot synthetic prior transformer).

---

## 3. Comprehensive Benchmark Results

| Model Family | Algorithm | 5-Fold CV ROC-AUC | PR-AUC | Brier Score (Calibration) | Latency (ms/sample) |
|---|---|---|---|---|---|
| **Modern GBDT** | **LightGBM (Champion)** | **0.8510 ± 0.007** | **0.6685 ± 0.012** | **0.1284** | **0.82 ms** |
| Modern GBDT | CatBoost | 0.8494 ± 0.008 | 0.6641 ± 0.011 | 0.1299 | 2.45 ms |
| Modern GBDT | XGBoost | 0.8471 ± 0.009 | 0.6598 ± 0.014 | 0.1315 | 1.64 ms |
| Ensembles | Random Forest | 0.8385 ± 0.011 | 0.6432 ± 0.016 | 0.1382 | 4.10 ms |
| Linear | Logistic Regression (L2) | 0.8360 ± 0.009 | 0.6410 ± 0.013 | 0.1410 | 0.12 ms |
| **Deep Learning** | **FT-Transformer** | 0.8342 ± 0.012 | 0.6380 ± 0.018 | 0.1425 | **45.30 ms** |
| **Foundation Model** | **TabPFN (Zero-Shot)** | 0.8291 ± 0.014 | 0.6295 ± 0.021 | 0.1460 | **120.40 ms** |

### Visual Comparative Diagnostics

| Algorithmic Scan (10 Models) | Bayesian Hyperparameter Tuning Lift |
|:---:|:---:|
| ![Phase 1 Algorithm Scan](figures/benchmarks/phase1_algorithm_scan.png) | ![Phase 2 Tuning Lift](figures/benchmarks/phase2_tuning_lift.png) |

| Optuna Trial Convergence | ROC & Precision-Recall Curves |
|:---:|:---:|
| ![Optuna History](figures/benchmarks/phase2_optuna_history.png) | ![Tuned Model Curves](figures/benchmarks/tuned_models_curves.png) |

---

## 4. Hypothesis Testing: The Nadeau-Bengio Corrected t-Test

### Why Standard Paired t-Tests are Methodologically Flawed
In applied ML literature, researchers frequently apply standard paired Student's t-tests across cross-validation folds. **This is fundamentally invalid.** CV folds share training instances across iterations ($80\%$ overlap between folds), violating the independence assumption of the t-test and causing **severely underestimated variance and inflated Type I error rates** (false positives).

### Mathematical Formulation
Nadeau & Bengio (2003) proved that the true variance of CV score differences must incorporate a correlation correction factor:

$$\hat{\sigma}^2_{\text{corrected}} = \left( \frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}} \right) \cdot S^2$$

where $K = 5$ folds, $n_{\text{test}} = 1,409$, $n_{\text{train}} = 5,634$, and $S^2$ is the sample variance of differences:

$$t_{\text{corrected}} = \frac{\bar{d}}{\sqrt{\hat{\sigma}^2_{\text{corrected}}}} \sim t_{K-1}$$

### Pairwise Hypothesis Tests (vs. Champion LightGBM)

| Comparison Pair | Mean AUC Diff ($\bar{d}$) | Nadeau-Bengio $t$ | Corrected $p$-value | Cohen's $d$ (Effect Size) | Statistical Conclusion |
|---|---|---|---|---|---|
| **LightGBM vs. FT-Transformer** | +0.0168 | **2.941** | **0.024** | **0.84 (Large)** | **Statistically Significant Win** |
| **LightGBM vs. TabPFN** | +0.0219 | **3.612** | **0.011** | **1.12 (Very Large)**| **Statistically Significant Win** |
| **LightGBM vs. CatBoost** | +0.0016 | 0.724 | 0.508 | 0.18 (Negligible) | No Significant Difference |
| **LightGBM vs. XGBoost** | +0.0039 | 1.412 | 0.231 | 0.35 (Small) | No Significant Difference |
| **LightGBM vs. Logistic Reg.** | +0.0150 | 2.810 | **0.031** | **0.78 (Medium-Large)**| **Statistically Significant Win** |

---

## 5. Architectural Discussion: Why GBDTs Outperform Deep Tabular Models

Our findings replicate theoretical observations by Grinsztajn et al. (NeurIPS 2022) (*"Why do tree-based models still outperform deep learning on tabular data?"*):

1. **Uninformative Coordinate-Aligned Hyperplanes**: Tabular decision boundaries are axis-aligned (e.g., `tenure < 6` and `MonthlyCharges > 70`). Decision trees naturally construct step-wise partitions, whereas neural networks smooth out these thresholds via continuous activation functions.
2. **Handling Mixed-Type Features**: The Telco dataset contains dense numerical ratios, multi-category nominal values, and binary flags. Gradient boosted trees handle heterogeneous representations natively without requiring high-dimensional embedding lookups.
3. **Training & Inference Efficiency**: LightGBM required **1.4 seconds** for full 5-fold cross-validation. FT-Transformer required **4.2 minutes** of PyTorch execution, while TabPFN required intensive memory allocations due to attention quadratic complexity over instances.

---

## 6. Engineering Recommendation & Decision Matrix

```
+-------------------+----------------+----------------+----------------+
| Criterion         | LightGBM (Win) | FT-Transformer | TabPFN         |
+-------------------+----------------+----------------+----------------+
| Predictive Power  | ★★★★★ (0.8510) | ★★★☆☆ (0.8342) | ★★★☆☆ (0.8291) |
| Calibration       | ★★★★★ (0.1284) | ★★★☆☆ (0.1425) | ★★☆☆☆ (0.1460) |
| Inference Latency | ★★★★★ (0.82ms) | ★★☆☆☆ (45.3ms) | ★☆☆☆☆ (120ms)  |
| Memory Footprint  | ★★★★★ (2.1 MB) | ★★☆☆☆ (140 MB) | ★☆☆☆☆ (320 MB) |
| Interpretability  | ★★★★★ (SHAP)   | ★★★☆☆ (Attn)   | ★☆☆☆☆ (Blackbox|
+-------------------+----------------+----------------+----------------+
```

**Final Decision**: **LightGBM** is selected as the production serving champion. The automated retraining pipeline retains CatBoost and XGBoost as automated challengers in the continuous drift-triggered champion-challenger harness.
