# SQL Injection Explainability Transfer Benchmark: Complete Technical Package

**Paper Title:** *Shifting Rationales Behind Stable Classifiers: Evaluating Explanation Consistency Under Application-Domain Transfer in SQL Injection Detection*  
**Research Benchmark:** Superviz26-SQL (Version 1.0, Zenodo DOI: `10.5281/zenodo.19627322`)  
**Package Scope:** Self-contained code, unit test suite, paper drafts, and empirical verification artifacts.

---

## 1. Executive Summary & Research Question

Machine learning Web Application Firewalls (WAFs) are typically evaluated on single application databases. In enterprise settings, however, classifiers trained on one database schema (e.g., Domain B: Sakila DVD rental) are deployed to inspect queries against different backend database schemas (e.g., Domain D: Oracle Human Resources).

While traditional benchmarks report accuracy or $F_1$-score alone, this research investigates the fundamental question:
> **When an SQL injection classifier transfers across application domains with high detection accuracy, do its underlying feature attributions remain consistent with an in-domain detector, or do models arrive at correct predictions via shifted feature weightings?**

### Key Findings
1. **High Detection Performance Coexists with Shifted Explanations:**
   - All three evaluated architectures maintain strong detection performance ($F_1 \ge 0.985$) under cross-domain transfer ($B \to D$), with performance shifts within $|\Delta F_1| \le 0.015$.
   - Concurrently, feature attributions exhibit measurable shifts (mean signed SHAP cosine similarity $0.847–0.906$, and Spearman rank correlation $\rho$ dropping to $0.447$ in Logistic Regression and $0.452$ in XGBoost).
2. **Concordant Decisions on Shifted Rationales:**
   - In 97.5%–99.5% of queries in a stratified 200-query explanation sample, cross-domain and within-domain models make identical, correct predictions despite relying on divergent feature weightings.
3. **Architectural Differences in Feature Ranking:**
   - Bagged ensembles (Random Forest) preserve relative signed feature rankings substantially better ($\rho = 0.8010$) than linear models ($\rho = 0.4467$) or boosted trees ($\rho = 0.4523$).

---

## 2. Package Organization

```
complete_project_package/
│
├── README_FOR_SUPERVISOR.md           <-- You are here (Technical overview for faculty/supervisor)
├── HOW_TO_EXPLAIN_TO_MAM.md           <-- Quick presentation script & defense Q&A for meetings
│
├── run_experiment.py                  <-- Master pipeline script (data loading -> train -> XAI -> eval)
├── requirements.txt                   <-- Python package dependencies (scikit-learn, shap, lime, xgboost)
│
├── src/                               <-- Clean modular implementation code
│   ├── data/
│   │   ├── loader.py                  <-- Robust chunked loading, provenance tracking, label sanitization
│   │   ├── protocol.py                <-- Quarantined evaluation sampling (zero query/normalized leakage)
│   │   ├── audit.py                   <-- Verifies exact partition query counts and template ecology
│   │   └── download.py                <-- Automated dataset fetcher from Zenodo
│   ├── features/
│   │   └── manual_li.py               <-- Verbatim port of 32 manual lexical features (Qi Li et al., 2019)
│   ├── models/
│   │   ├── train.py                   <-- Pipeline wrapper ensuring train-only StandardScaler fitting
│   │   └── evaluate.py                <-- Metric calculation (Accuracy, Prec, Rec, F1, ROC-AUC, PR-AUC)
│   ├── explainers/
│   │   ├── shap_explainer.py          <-- TreeExplainer (RF, XGB) & LinearExplainer (LR) aligned to class 1
│   │   └── lime_explainer.py          <-- Instance-centered perturbation sampling with direct index mapping
│   ├── analysis/
│   │   ├── stability.py               <-- Cosine similarity, Spearman rho, 4-strata decoupling analysis
│   │   └── sensitivity.py             <-- Evaluates protocol variation and split-provenance effects
│   └── visualization/
│       └── plots.py                   <-- Generates publication-ready figures (metrics, top features, strata)
│
├── tests/                             <-- 87 automated unit & integration tests (100% pass rate in <2s)
│   ├── test_manual_li.py              <-- Validates all 32 lexical features, tautology regex, edge cases
│   ├── test_data_loader.py            <-- Validates CSV chunking, label types, split provenance
│   ├── test_protocol_and_provenance.py<-- Asserts strict zero query leakage (train ∩ eval = ∅)
│   ├── test_explainers_and_stability.py<-- Validates SHAP/LIME mappings, cosine similarity edge cases
│   ├── test_models_and_evaluation.py  <-- Tests model training, prediction pipelines, metrics
│   └── test_template_disjoint_feasibility.py <-- Proves Domain D template saturation (29 templates)
│
├── paper_and_manuscripts/             <-- Final scientific paper ready for submission
│   ├── final_manuscript.pdf           <-- Formatted publication PDF (500 KB)
│   ├── final_manuscript.docx          <-- Word format with native tables and equations
│   └── final_manuscript.md            <-- Standard academic Markdown draft
│
├── results_and_tables/                <-- Verified empirical results
│   ├── final_results_tables.csv       <-- All 6 publication tables in CSV format
│   ├── final_metrics.json             <-- Full serialized metric dictionary
│   └── final_claim_evidence_matrix.md <-- Bidirectional traceability linking all claims to artifacts
│
├── figures/                           <-- High-resolution publication plots
│   ├── detection_metrics_*.png        <-- Detection comparisons across cross, within, and repeat fits
│   ├── top_features_*.png             <-- Top 10 signed feature attributions across models
│   └── decoupling_strata_*.png        <-- 4-category prediction strata breakdown
│
└── manifests_and_audit/               <-- Reproducibility and governance logs
    ├── final_data_manifest.json       <-- Exact query IDs, split provenance, and scenario tags
    ├── final_split_provenance_report.md <-- Detailed breakdown of official Superviz26-SQL partitions
    ├── final_change_log.md            <-- Step-by-step record of all audit fixes and rectifications
    └── final_test_report.md           <-- Execution log of the 87 automated test suite
```

---

## 3. High-Level Architectural Flow

```
[Raw Superviz26-SQL Benchmark]
  Domain B: Sakila (b-b.csv)       Domain D: Oracle HR (d-d.csv)
  44,876 Benign / 5,088 Attack     44,867 Benign / 5,133 Attack
               │                                │
               │                                ▼ STEP 1: Lock Evaluation Set First (Seed 42)
               │                     ┌──────────────────────────────────────────────┐
               │                     │ Quarantined S_eval: 2,000 queries (prevalence 50%)
               │                     │ (1,000 Benign / 1,000 Attack)                │
               │                     └──────────────────────────────────────────────┘
               │                                │
               ▼ STEP 2: Cross Training         ▼ STEP 3: Within Training
┌──────────────────────────────┐     ┌──────────────────────────────────────────────┐
│ S_cross: 4,000 queries       │     │ S_within: 4,000 queries                      │
│ (2,000 Benign / 2,000 Attack)│     │ (2,000 Benign / 2,000 Attack from remainder) │
│ Zero Domain D contamination  │     │ Strict quarantine: Q_train ∩ S_eval = ∅      │
└──────────────────────────────┘     └──────────────────────────────────────────────┘
               │                                │
               ▼ STEP 4: Feature Extraction (32 Lexical Features - manual_li.py)
               │                                │
               ▼ STEP 5: Model Fitting & Standalone Preprocessing (StandardScaler)
       M_cross (LR, RF, XGB)             M_within (LR, RF, XGB)
               │                                │
               └────────────────┬───────────────┘
                                ▼ STEP 6: Quarantined Evaluation on S_eval
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
[Predictive Performance]                    [Attribution Consistency]
Accuracy, F1, ROC-AUC, PR-AUC               Signed SHAP Attributions (n=200)
(Table 1: F1 >= 0.985)                      Vector Cosine Similarity & Spearman rho
                                            (Table 2: Cosine 0.847–0.906, rho 0.447–0.801)
```

---

## 4. How to Verify & Execute Live

### Prerequisites
Python 3.10+ with standard machine learning dependencies:
```bash
pip install -r requirements.txt
```

### 1. Run Complete Test Suite (Takes ~2 seconds)
```bash
python -m unittest discover tests
```
*Expected output: `Ran 87 tests in ~1.8s ... OK` (verifies zero leakage, feature logic, and XAI mappings).*

### 2. Run Full Experiment Pipeline
```bash
python run_experiment.py --eval-size 2000 --train-size 4000 --shap-size 200 --lime-size 50 --seed 42
```
*Executes data loading, 32-feature extraction, model training (LR, RF, XGBoost), SHAP explanation extraction, LIME perturbation repeats, statistical aggregation, and figure generation.*

