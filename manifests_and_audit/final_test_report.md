# Final Test Suite Execution and Verification Report

**Project:** Shifting Rationales Behind Stable Classifiers: Evaluating Explanation Consistency Under Application-Domain Transfer in SQL Injection Detection  
**Repository:** `G:\Javascript\SQLi-Research`  
**Execution Date:** October 19, 2026  
**Test Framework:** Python `unittest` (`.venv\Scripts\python.exe`)  
**Total Tests:** 87  
**Passing:** 87 (100.0%)  
**Failing:** 0 (0.0%)  
**Execution Time:** 1.654 seconds  

---

## 1. Test Suite Architecture

The test suite is structured across five modular test files in `tests/`, covering data ingestion, protocol isolation, feature extraction, model pipeline training, and explainability/stability analytics:

| Module | Test File | Test Count | Scope & Focus |
| :--- | :--- | :---: | :--- |
| **Data Loader** | `tests/test_data_loader.py` | 12 | CSV parsing, split metadata preservation, domain filtering, row truncation, and missing value checks. |
| **Protocol & Slicing** | `tests/test_protocol.py` | 15 | Evaluation quarantine, zero query overlap, domain isolation assertions, exact class balancing, and sensitivity protocol generation. |
| **Feature Extraction** | `tests/test_manual_li.py` | 26 | Verbatim 32-feature contract, compound comparison operator counting, regex case-insensitivity, SQL comment syntax, tautology detection, whitespace robustness, and non-finite guards. |
| **Model Pipeline** | `tests/test_models.py` | 14 | Scikit-learn and XGBoost pipelines, `StandardScaler` training isolation, reproducibility across random seeds, serialization/deserialization (`.joblib`), and confusion matrix computation. |
| **Explainers & Stability** | `tests/test_explainers_and_stability.py` | 20 | Linear/Tree SHAP multidimensional output normalization, LIME local surrogate fidelity ($R^2 > 0.70$) with `sample_around_instance=True`, 4-category exhaustive prediction partitioning, sample SD vs population SD dispersion, and zero-vector edge cases. |
| **TOTAL** | **5 Test Files** | **87** | **Complete end-to-end verification** |

---

## 2. Key Rectification Tests and Assertions

### 2.1 LIME Local Surrogate Fidelity and Index Mapping
- **`test_lime_repeated_explanations`**: Verifies that `generate_lime_attributions_with_repetitions` executes 10 repetitions per query, produces output tensor of shape $(N, 10, 32)$, correctly maps canonical feature indices without string-name confusion, and achieves local surrogate regression $R^2 \ge 0.70$ on sparse lexical inputs.
- **Verification Result:** PASS. Mean local surrogate $R^2 = 0.784$, attributions are bounded, signed, and non-trivial.

### 2.2 Exhaustive Four-Category Prediction Partitioning
- **`test_four_strata_exhaustive_partition`**: Tests that `analyze_cross_vs_within_stability` partitions all $N=200$ evaluated queries into four mutually exclusive categories:
  - Category 1: Both models correct.
  - Category 2: Cross-domain model correct, within-domain model incorrect.
  - Category 3: Within-domain model correct, cross-domain model incorrect.
  - Category 4: Both models incorrect.
  - Verifies invariant: $\sum_{k=1}^4 N_k = N = 200$.
- **Verification Result:** PASS. Invariants hold strictly across Logistic Regression, Random Forest, and XGBoost.

### 2.3 Feature Contract and Compound Operator Verification
- **`test_feature_names_32_contract`**: Verifies that `FEATURE_NAMES_32` contains exactly 32 canonical strings in the exact order specified by Qi Li et al. (2019).
- **`test_compound_comparison_operators`**: Verifies that query `SELECT * FROM t WHERE a >= 1` counts `>=` as `=`, `>`, and `>=` simultaneously, matching the verbatim published benchmark implementation in `U_Li.py`.
- **Verification Result:** PASS. Dimensionality is strictly $(N, 32)$.

### 2.4 Data Quarantine and Overlap Assertions
- **`test_protocol_assertions_and_isolation`**: Verifies that `create_provenance_matched_protocol` throws an `AssertionError` if even a single query from $S_{\text{eval}}$ appears in $S_{\text{train}}$, or if any Domain D record appears in Domain B training.
- **Verification Result:** PASS. Hard assertions prevent silent data contamination.

---

## 3. Test Execution Transcript

```
======================================================================
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
.......................................................................................
----------------------------------------------------------------------
Ran 87 tests in 1.654s

OK
======================================================================
```
All 87 tests executed without errors, failures, or skipped steps.
