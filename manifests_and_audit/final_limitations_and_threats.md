# Methodological Limitations and Threats to Validity

**Project:** Shifting Rationales Behind Stable Classifiers: Evaluating Explanation Consistency Under Application-Domain Transfer in SQL Injection Detection  
**Repository:** `G:\Javascript\SQLi-Research`  
**Date:** October 19, 2026  

---

## 1. Overview and Scientific Integrity Statement

In adherence to scientific rigor, this document details the methodological boundaries, structural constraints, and potential threats to validity governing this research. A lower but methodologically valid result is superior to an impressive but invalid result. Claims in the research manuscript are strictly scoped to what the empirical evidence directly supports.

---

## 2. Threats to External Validity

### 2.1 Single Transfer Scenario ($N = 1$ Pair)
- **Constraint:** The experimental evaluation is conducted on a single source-to-target transfer pair: Domain B (Sakila DVD rental) $\to$ Domain D (Oracle Human Resources).
- **Impact:** While Sakila and Oracle HR feature fundamentally distinct relational schemas (media catalog vs. enterprise personnel), column types, and query semantics, our findings represent a descriptive case study.
- **Scientific Boundary:** We do not claim that explanation drift occurs universally across all possible database schemas, nor that Random Forest will always preserve signed feature rankings on arbitrary enterprise transfers. Broader generalization requires multi-target transfers across additional DBMS engines (PostgreSQL, SQLite, Microsoft SQL Server).

### 2.2 Synthetic Attack Payloads (SQLMap Tooling)
- **Constraint:** All attack payloads in the Superviz26-SQL benchmark were generated using the automated exploitation tool SQLMap.
- **Impact:** While SQLMap generates realistic, evasive payloads across eight distinct injection techniques (boolean blind, error-based, time blind, union, stacked, inline, tautology, piggyback), automated tools follow algorithmic perturbation templates.
- **Scientific Boundary:** Real-world human adversaries who construct bespoke multi-stage payloads, polyglots, or context-aware syntax evasion may introduce linguistic distribution shifts not captured in automated benchmarks.

---

## 3. Threats to Internal Validity

### 3.1 Benchmark Partition Anomaly and Custom Test-Partition Sampling
- **Constraint:** Superviz26-SQL was authored as an unsupervised anomaly detection benchmark. Consequently, its official `train` partition contains strictly zero attack queries and exactly 100,000 normal queries per scenario. Attack queries exist exclusively in the official `test` partition stream.
- **Impact:** Fitting supervised binary classifiers by taking normal queries from the official `train` partition and attack queries from the official `test` partition introduces severe partition confounding (classifiers latch onto partition generation artifacts and timing differences).
- **Resolution and Remaining Limitation:** To ensure environmental homogeneity, we sampled both benign and attack queries from the official `test` partition stream under strict query-level isolation ($Q_{\text{train}} \cap Q_{\text{eval}} = \emptyset$). Although exact and normalized query overlap is verified to be strictly zero, training and evaluation records originate from the same high-level data generation campaign. This protocol deviation is explicitly disclosed in all reports and the manuscript.

### 3.2 Template Exposure Asymmetry (Interpolation vs. Extrapolation)
- **Constraint:** Domain D contains 46 total query templates, of which only 29 support vulnerable attack endpoints. Our 2,000-query evaluation set saturated all 29 attack templates (100% coverage).
- **Impact:** Excluding the 29 evaluation templates leaves zero attack queries in the candidate Domain D training pool. Therefore, constructing a template-disjoint supervised within-domain reference is mathematically infeasible under the locked evaluation set.
- **Scientific Consequence:** 
  - $M_{\text{within}}$ operates in a **template-interpolating** regime: it observed all 29 query template structures during training and evaluates on unseen literal constants and payload parameters.
  - $M_{\text{cross}}$ operates in a **template-extrapolating** regime: it evaluates on entirely novel schema entities and query structures.
- **Scientific Boundary:** Observed attribution differences reflect the joint effect of application schema shift and template interpolation vs. extrapolation. They cannot be causally attributed to schema shift alone.

### 3.3 Protocol Sensitivity Confounding
- **Constraint:** In our protocol sensitivity analysis, the unmitigated baseline altered both partition provenance (benign from `train`, attack from `test`) and attack prevalence (50% down to 10%).
- **Impact:** The resulting 5.84% drop in $F_1$ reflects their joint interaction rather than partition confounding alone. This interaction is explicitly documented in Section 7.2 of the manuscript.

---

## 4. Threats to Construct Validity

### 4.1 Manual Lexical Feature Space (`benchmark_32`)
- **Constraint:** Classifiers are trained on the 32 manual lexical features formulated by Qi Li et al. (IEEE Access 2019).
- **Impact:** While manual lexical features remain widely deployed in production WAFs due to sub-millisecond latency and low computational overhead, they represent surface syntactic properties (character lengths, special character counts, keyword flags) rather than semantic parse trees or graph structures.
- **Scientific Boundary:** Conclusions regarding explanation drift apply specifically to manual lexical feature representations. Deep contextual representations (e.g., CodeBERT token embeddings) or AST-based graph neural networks may exhibit different stability dynamics.

### 4.2 Attribution Consistency as an Explanation Proxy
- **Constraint:** Explanation consistency is quantified via vector cosine similarity and signed Spearman rank correlation between post-hoc attribution vectors.
- **Impact:** High cosine similarity indicates that two models assign similar relative weights to the same feature directions; high Spearman rank correlation indicates that features are ordered similarly in signed importance.
- **Scientific Boundary:** Neither metric implies that either model's explanation is "ground-truth faithful," causally correct, or optimal for human security triage. A classifier may produce consistent explanations that rely on spurious heuristics, or inconsistent explanations where both sets of rationales are defensible.

### 4.3 Signed vs. Unsigned Rank Correlation
- **Constraint:** Spearman rank correlation is calculated directly on signed feature attribution values ($\boldsymbol{\phi} \in \mathbb{R}^{32}$), where positive values increase attack probability and negative values decrease it.
- **Impact:** This measures the ranking of signed directional contributions, not the ranking of unsigned feature importance magnitude ($|\boldsymbol{\phi}|$). This distinction is explicitly defined in Section 4.8 of the paper.

---

## 5. Threats to Conclusion Validity

### 5.1 Sample Size Constraints in Subgroup Analyses
- **Constraint:** In the 200-query explanation sample, models achieve $>98.5\%$ accuracy, placing 195 to 199 queries into Category 1 (concordant correct).
- **Impact:** Consequently, error categories (Categories 2, 3, and 4) contain very small sample sizes ($n = 1, 3,$ or $4$).
- **Scientific Boundary:** Numerical averages in error strata (e.g., $S_{\text{cos}} = 0.4964$ for RF Cat 2, $S_{\text{cos}} = 0.4212$ for XGB Cat 3) serve strictly as descriptive observations of individual query behaviors. They cannot support generalized inferential claims regarding attribution behavior on misclassified instances.

### 5.2 Descriptive vs. Causal Interpretation
- **Scientific Boundary:** The empirical findings provide descriptive evidence that high classification performance coexists with measurable attribution differences in this benchmark setting. The data does not establish statistical independence or a causal decoupling mechanism across general populations of ML classifiers.
