# Final Comprehensive Project Change Log

**Project:** Shifting Rationales Behind Stable Classifiers: Evaluating Explanation Consistency Under Application-Domain Transfer in SQL Injection Detection  
**Repository:** `G:\Javascript\SQLi-Research`  
**Date:** October 19, 2026  
**Git Branch:** `audit-and-rectification`  

---

## 1. Overview of Audit and Rectification Workflow

This log records every modification made across the repository during the comprehensive research audit, rectification, experiment validation, and paper finalization process. Prior to modification, all pre-audit evidence was archived in `results_baseline_pre_audit/`, and a dedicated Git branch `audit-and-rectification` was established to ensure complete version control and rollback capability.

---

## 2. Chronological Record of Meaningful Modifications

### Change 1: Snapshot Preservation of Baseline Evidence
- **Files Modified / Created:** Created backup directory `results_baseline_pre_audit/` containing full copies of legacy results, models, explanations, and reports.
- **Rationale:** Rule 2 of the research audit mandates preserving original evidence so that previous and corrected results can be compared without destroying pre-audit artifacts.
- **Impact:** Guaranteed preservation of historical preliminary findings.

### Change 2: LIME Continuous Sampling and Index Mapping Rectification
- **Files Modified:** `src/explainers/lime_explainer.py`
- **Problem:** Default LIME tabular explainer with `discretize_continuous=False` and `sample_around_instance=False` perturbed features around global training mean $\mu$. Instances located $30\sigma$ away in sparse lexical space suffered exponential kernel underflow ($e^{-d^2/\sigma^2} \approx 10^{-26}$). Ridge regression slopes collapsed to $10^{-11}$ ($R^2 \approx 10^{-8}$), causing feature name mapping failures and collapsing attributions to near-zero noise.
- **Rectification:** Set `sample_around_instance=True` and passed unscaled reference statistics. Replaced string lookup with exact index mapping via `exp.as_map()[1]`.
- **Impact:** Restored local surrogate regression fidelity ($R^2$ jumped to $0.70–0.85$). Produced authentic, non-trivial, signed feature attributions.

### Change 3: Exhaustive Four-Category Prediction Stratification
- **Files Modified:** `src/analysis/stability.py`, `src/visualization/plots.py`
- **Problem:** Legacy stability reporting only included Category 1 (Both correct) and Category 3 (Within only correct), omitting Category 2 (Cross only correct) and Category 4 (Both wrong).
- **Rectification:** Expanded `decoupling_analysis` to compute all four mutually exclusive and exhaustive categories ($C_1 + C_2 + C_3 + C_4 = 200$). Updated `plot_decoupling_strata` to dynamically plot all active categories with sample sizes.
- **Impact:** Achieved 100% mathematical accounting of all 200 evaluation queries across Logistic Regression (199+1+0+0=200), Random Forest (195+4+1+0=200), and XGBoost (197+0+3+0=200).

### Change 4: Hard Protocol Assertions Against Leakage and Imbalance
- **Files Modified:** `src/data/protocol.py`
- **Problem:** Absence of automated assertions in protocol generation allowed silent data contamination or prevalence skew to go undetected.
- **Rectification:** Added hard assertions verifying $|S_{\text{eval}}| = 2,000$, $\text{prev} = 0.500$, $|S_{\text{cross}}| = 4,000$, $|S_{\text{within}}| = 4,000$, zero exact string overlap ($Q_{\text{train}} \cap Q_{\text{eval}} = \emptyset$), zero normalized string overlap, and zero Domain D records in Domain B training.
- **Impact:** Guarantees that any experiment execution halts immediately if data leakage or class imbalance occurs.

### Change 5: Compound Comparison Operator Semantics & Unit Testing
- **Files Modified:** `src/features/manual_li.py`, `tests/test_manual_li.py`
- **Problem:** Audit revealed that `>=` and `<=` are counted multiple times (by `=`, `>`, and `>=`).
- **Rectification:** Verified against original published benchmark implementation in `U_Li.py` (Qi Li et al., IEEE Access 2019) that this is the verbatim published behavior. Preserved semantics to ensure benchmark fidelity, added documentation, and expanded unit tests in `test_manual_li.py` to 26 comprehensive tests.
- **Impact:** Retained exact compatibility with published research while documenting feature counting mechanics.

### Change 6: Matplotlib Non-Interactive Agg Backend Configuration
- **Files Modified:** `src/visualization/plots.py`, `run_experiment.py`
- **Problem:** Default Matplotlib backend on Windows (`TkAgg`) crashed during multi-threaded Random Forest training with `Tcl_AsyncDelete: async handler deleted by the wrong thread`.
- **Rectification:** Added `import matplotlib; matplotlib.use("Agg")` at the very top of `plots.py` and `run_experiment.py` before any pyplot imports.
- **Impact:** Eliminated GUI thread crashes, enabling uninterrupted execution of the full 2,000-query / 1,500-explanation pipeline.

### Change 7: Parametric Dispersion Metrics Expansion
- **Files Modified:** `src/analysis/stability.py`
- **Problem:** Ambiguity in reporting $\text{Mean} \pm \text{SD}$ where population standard deviation ($\text{ddof}=0$) was not clearly distinguished from sample standard deviation ($\text{ddof}=1$) or standard error (SEM).
- **Rectification:** Updated `compute_distribution_summary` to provide `std` ($\text{ddof}=0$), `sample_std` ($\text{ddof}=1$), and `sem` ($s/\sqrt{n}$). Explicitly defined table captions in all reports and the manuscript.
- **Impact:** Complete statistical clarity and transparency for peer review.

### Change 8: Test Suite Expansion
- **Files Modified:** `tests/test_explainers_and_stability.py`, `tests/test_manual_li.py`
- **Problem:** Test suite contained 78 tests and lacked coverage for LIME surrogate $R^2$, 4-category partition invariants, and compound operator semantics.
- **Rectification:** Added 9 new unit and integration tests, expanding the suite to 87 tests.
- **Impact:** 100% test pass rate in 1.65 seconds (`Ran 87 tests in 1.654s — OK`).

### Change 9: Full Experiment Re-execution and Artifact Serialization
- **Files Modified / Generated:** `run_experiment.py`, `results/`
- **Problem:** Previous CSVs, JSON reports, and figures contained preliminary or unrectified values.
- **Rectification:** Executed full end-to-end experiment from clean state using verified pipeline (`--eval-size 2000 --train-size 4000 --shap-size 200 --lime-size 50 --seed 42`).
- **Impact:** Regenerated all 6 detection models, 6 SHAP attribution CSVs, 3 LIME attribution CSVs (1,500 explanations), 9 publication figures, and unified JSON results.

### Change 10: Manuscript Synchronization & DOCX Compilation
- **Files Modified:** `results/reports/research_paper_draft.md`, `final_manuscript.md`, `final_manuscript.docx`
- **Problem:** Manuscript draft contained preliminary LIME repeatability numbers (0.9801) and lacked documentation of `sample_around_instance=True` and 87 unit tests.
- **Rectification:** Synchronized all numbers with verified outputs (LIME repeatability: 0.8401 for LR, 0.9630 for RF, 0.9173 for XGB; 87 tests; explicit dispersion definitions). Rebuilt DOCX manuscript using `python-docx`.
- **Impact:** Perfect alignment between empirical data and scientific manuscript.


### Change 11: Pre-Submission Reviewer Audit & Methodological Rectification
- **Files Modified / Regenerated:** `results/final_deliverables/final_manuscript.md`, `results/reports/research_paper_draft.md`, `final_manuscript.docx`, `final_manuscript.html`, `final_manuscript.pdf`, `final_claim_evidence_matrix.md`, `final_change_log.md`
- **Audit Source:** External peer reviewer audit evaluating authenticity, statistical terminology, SHAP comparability, benchmark versioning, and claim defensibility.
- **Problems Identified & Rectifications:**
  1. *Statistical Terminology (ddof Inconsistency):* Section 4.8 and Table 4 note previously described standard deviation as "sample standard deviation with degrees of freedom ddof = 0". Mathematically, `ddof = 0` calculates the population standard deviation, whereas sample standard deviation uses `ddof = 1`. Corrected text and captions to: `population standard deviation (SD, calculated with ddof = 0)`.
  2. *Tautology Regex Omission in Manuscript Prose:* Section 4.3 omitted the capturing group parentheses in text (`\w+=\1`), which would be an invalid regular expression. Corrected to `(\w+)=\1`, matching the verbatim compiled expression `_P_TAUTOLOGY = re.compile(r"(\w+)=\1")` in `src/features/manual_li.py`.
  3. *Table 2 Statistical Completeness for Signed Spearman $\rho$:* Section 4.8 claimed mean, median, IQR, and 95% bootstrap CI are reported for all metric distributions, but Table 2 previously omitted IQR and CI for Spearman $\rho$. Expanded Table 2 to include complete reporting for both Cosine Similarity and Signed Spearman $\rho$ (LR: $\rho=0.4467$, IQR=0.1059, 95% CI [0.4385, 0.4547]; RF: $\rho=0.8010$, IQR=0.1743, 95% CI [0.7826, 0.8183]; XGB: $\rho=0.4523$, IQR=0.2089, 95% CI [0.4282, 0.4775]).
  4. *Causal Language in Results:* Section 5.1 claimed within-domain Random Forest "overfit to specific numeric formatting in Domain D records". Replaced with descriptive observation that within-domain RF produced more false alarms (34 vs. 2), noting that while consistent with sensitivity to domain lexical patterns, further controlled analyses are needed. Tempered Section 5.3 bagging claims into observed architectural patterns.
  5. *Sample Size Scope Qualification for Concordant Queries:* Ensured the 97.5%–99.5% concordant-correct rate is explicitly qualified as evaluated on the **200-query explanation subset ($n=200$)** across Abstract, Contributions, Table 4, and Conclusion.
  6. *Removal of Repetitive Research Questions:* Streamlined Section 1.2 to eliminate redundant word-for-word repetition of RQ1–RQ4 that already appear formally in Section 3.
  7. *SHAP vs. LIME Functional Roles:* Clarified that SHAP evaluates cross-domain vs within-domain attribution consistency, while LIME assesses local perturbation repeatability across seeds.
  8. *Dataset Versioning Clarity:* Explicitly acknowledged Superviz26-SQL Version 1.0 (DOI: `10.5281/zenodo.19627322`) and noted the subsequent Zenodo record (`21068333`), confirming that all files and counts correspond strictly to Version 1.0.
  9. *Perturbation vs. Domain Transfer Distinction:* In Section 2.4, clarified that adversarial perturbation studies alter input queries to test explainer fragility, whereas domain transfer keeps target queries intact to evaluate whether classifiers trained on different application distributions produce concordant explanations.
- **Impact:** Complete alignment across code, data tables, mathematical definitions, reviewer guidelines, and manuscript deliverables. Generated publication-ready DOCX and PDF documents.
