# Final Claim-to-Evidence Traceability Matrix

**Project:** Shifting Rationales Behind Stable Classifiers: Evaluating Explanation Consistency Under Application-Domain Transfer in SQL Injection Detection  
**Repository:** `G:\Javascript\SQLi-Research`  
**Date:** October 19, 2026  

---

## 1. Traceability Architecture

This matrix provides 100% bidirectional verification linking every scientific claim made in the manuscript (Abstract, Introduction, Results, Discussion, Threats to Validity, Conclusion) directly to its supporting empirical artifact, test case, table, or figure.

---

## 2. Claim-to-Evidence Mapping Table

| Claim ID | Manuscript Location | Exact Scientific Claim Statement | Underlying Empirical Evidence | Artifact / Verification File | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **C1** | Abstract, Sec 4.1 | The official Superviz26-SQL benchmark `train` partition contains strictly zero attack queries and exactly 100,000 normal queries per scenario. | Audit of `data/b-b.csv`, `data/d-d.csv`, `data/abc-d.csv`. Verified `split == 'train'` has `sum(label == 1) == 0`. | `results/final_deliverables/final_split_provenance_report.md`, `tests/test_data_loader.py` | **VERIFIED** |
| **C2** | Abstract, Sec 4.4 | Supervised training and evaluation samples were drawn from the official test stream with exact-query and normalized-query overlap verified as strictly zero. | Runtime assertions in `create_provenance_matched_protocol`. Set intersection $Q_{\text{train}} \cap Q_{\text{eval}} = \emptyset$ verified on 2,000 eval and 4,000 train queries. | `results/audit/manifest_provenance_matched.json`, `tests/test_protocol.py` | **VERIFIED** |
| **C3** | Abstract, Table 1 | Cross-domain $F_1$-scores are 0.9858 (LR), 0.9899 (RF), and 0.9879 (XGB), vs. within-domain $F_1$-scores of 0.9894, 0.9752, and 0.9909, respectively. | Evaluated on identical quarantined Domain D evaluation set ($N=2,000$, 1,000 benign, 1,000 attack). | `results/final_deliverables/final_metrics.json`, Table 1 in `final_results_tables.csv` | **VERIFIED** |
| **C4** | Abstract, Table 1 | Signed transfer differences $\Delta F_1$ are $-0.0036$ (LR), $+0.0147$ (RF), and $-0.0030$ (XGB). RF cross-domain performance is higher due to lower false alarms (2 vs. 34 FPs). | Confusion matrices on Domain D: RF Cross has 2 FPs (TN=998), while RF Within has 34 FPs (TN=966). | `results/metrics/detection_metrics.json`, Table 1 in `final_results_tables.csv` | **VERIFIED** |
| **C5** | Abstract, Table 2 | Mean SHAP vector cosine similarities are 0.8474 (LR, 95% CI [0.8273, 0.8660]), 0.8706 (RF, 95% CI [0.8563, 0.8839]), and 0.9058 (XGB, 95% CI [0.8941, 0.9151]). | Computed on $n=200$ stratified queries using `compute_vector_cosine_similarity` with 1,000 bootstrap resamples. | `results/reports/full_experiment_results.json`, Table 2 in `final_results_tables.csv` | **VERIFIED** |
| **C6** | Abstract, Table 2 | Mean signed Spearman rank correlations vary substantially: 0.4467 (LR), 0.4523 (XGB), and 0.8010 (RF), demonstrating distinct dimensions of explanation agreement. | Computed via `compute_signed_spearman_rank` on signed attribution values $\boldsymbol{\phi} \in \mathbb{R}^{32}$. | `results/reports/full_experiment_results.json`, Table 2 in `final_results_tables.csv` | **VERIFIED** |
| **C7** | Sec 5.4, Table 3 | Repeat-fit model stability yields mean cosine similarities of 0.9960 (LR), 0.9802 (RF), and 1.0000 (XGB), whose 95% CIs are entirely disjoint from domain transfer intervals. | Computed by comparing $M_{\text{cross}}^{(1)}$ and $M_{\text{cross}}^{(2)}$ on identical 4,000 Domain B training queries across seeds 42 vs 101/123. | `results/reports/full_experiment_results.json`, Table 3 in `final_results_tables.csv` | **VERIFIED** |
| **C8** | Sec 5.5, Table 5 | LIME perturbation stability across 10 repetitions ($n=50$ queries, 500 total explanations) yields mean pairwise cosine of 0.8401 (LR), 0.9630 (RF), and 0.9173 (XGB). | Computed via `generate_lime_attributions_with_repetitions` with `sample_around_instance=True` and unscaled reference statistics. | `results/final_deliverables/explanations/lime_attributions_*_cross.csv`, Table 5 | **VERIFIED** |
| **C9** | Sec 5.6, Table 4 | In 97.5% to 99.5% of evaluated queries where models agree on correct predictions (Category 1), classifiers arrive at decisions via shifted feature rationales. | Four-strata partitioning: LR Cat 1=199/200 (mean 0.8475), RF Cat 1=195/200 (mean 0.8784), XGB Cat 1=197/200 (mean 0.9132). | `scripts/inspect_categories.py`, Table 4 in `final_results_tables.csv` | **VERIFIED** |
| **C10** | Sec 4.9, Sec 6.3 | All 29 vulnerable Domain D application templates are represented in the 2,000-query evaluation set, rendering a template-disjoint supervised reference infeasible under the protocol. | Template census of Domain D: 46 total templates, 17 benign-only, 29 vulnerable attack templates. All 29 occur in $S_{\text{eval}}$. | `results/audit/manifest_provenance_matched.json`, Section 4.9 of manuscript | **VERIFIED** |
| **C11** | Sec 6.2 | Bagged ensembles (Random Forest) preserve relative signed feature rankings ($\rho = 0.8010$) substantially better than linear models or boosted trees ($\rho \approx 0.45$). | Comparison of signed Spearman rank correlations across architectures in Table 2 and Table 4. | `results/figures/top_features_random_forest.png`, `results/reports/full_experiment_results.json` | **VERIFIED** |
| **C12** | Sec 7.2 | In protocol sensitivity analysis, the unmitigated baseline confounded partition provenance with attack prevalence (50% down to 10%), producing an $F_1$ drop of 5.84%. | Sensitivity protocol comparison: Primary Cross F1 = 0.9879, Sensitivity Cross F1 = 0.9295 ($\Delta F_1 = -0.0584$). | `results/reports/sensitivity_analysis_report.md`, Table 6 in `final_results_tables.csv` | **VERIFIED** |
| **C13** | Sec 8.2 | The entire evaluation pipeline is validated by 87 automated unit and integration tests executing with 100% pass rate in under 2 seconds. | Execution of `unittest discover -s tests -p "test_*.py"`: 87 tests passed in 1.654 seconds. | `results/final_deliverables/final_test_report.md` | **VERIFIED** |

---

## 3. Discrepancy Reconciliation Log

1. **LIME Repeatability Values in Previous Draft:**  
   - Previous draft reported older preliminary numbers (0.9801 for LR, 0.9716 for RF, 0.9618 for XGB) generated prior to the LIME kernel underflow bugfix.  
   - Following rectification of continuous feature sampling (`sample_around_instance=True`) and direct index mapping (`exp.as_map()[1]`), LIME local surrogate fidelity was restored ($R^2 \approx 0.78$), yielding verified repeatability numbers: **0.8401** (LR), **0.9630** (RF), and **0.9173** (XGB). All manuscript sections, tables, and claim references have been updated to reflect these verified values.

2. **Table 4 Category Stratification Completeness:**  
   - Previous draft omitted Category 2 (Cross only correct) and Category 4 (Both wrong) from explicit display, creating potential ambiguity regarding whether the strata were exhaustive.  
   - Rectification expanded `decoupling_analysis` to explicitly report all four mutually exclusive and exhaustive categories ($C_1 + C_2 + C_3 + C_4 = 200$), proving 100% query accounting.

3. **Parametric Dispersion Definition:**  
   - Ambiguity regarding whether $\pm$ indicated population SD ($\text{ddof}=0$), sample SD ($\text{ddof}=1$), or standard error (SEM) has been formally resolved in Table 4 caption and methods: $\text{Mean} \pm \text{SD}$ (population standard deviation, $\text{ddof}=0$), with sample standard deviation provided in data tables.

4. **Table 2 Statistical Reporting Completeness:**  
   - Previous draft reported only mean and median for Signed Spearman $\rho$. Expanded Table 2 to include IQR and 95% bootstrap CI for both Cosine Similarity and Signed Spearman $\rho$ across all three architectures, perfectly aligning Table 2 with Section 4.8 methodology.

5. **Tautology Regular Expression Reconciliation:**  
   - In previous manuscript draft text, the regex was written as `\w+=\1` without parentheses. The Python implementation `src/features/manual_li.py` correctly uses `_P_TAUTOLOGY = re.compile(r"(\w+)=\1")`, which captures the first alphanumeric token for the backreference. Corrected manuscript prose to `(\w+)=\1`.
