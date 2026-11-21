# Study plan (exported from the plan file docs/study_plan.json)

- Plan created: 2026-11-15T09:20:24.353Z (UTC), before the first benchmark results were produced.
- SHA-256 of docs/study_plan.json: `e0e83c6bd1076939681485f9f5f7a3571f7e9f74235f6e3d398a8f3db6bc4039`
- The timestamp is internal to the project; for a verifiable timestamp, deposit docs/study_plan.json on OSF or Zenodo.
- Planned versus exploratory analyses are listed in Table 'Status of the analyses' of the manuscript (tab:analysis_status).

## Title

Matching-theory undersampling: methods, KEEL benchmark study, LaTeX manuscript

## Phase: Setup

### Setup

- **Install dependencies and freeze versions.** Install the PyPI extras into the imbml environment (KEEL data package, matching library for reference solvers, scikit-posthocs, baycomp, complexity-measure package) and save a frozen version list for the reproducibility statement.
- **Verify KEEL loader and inventory datasets.** Load every bundled KEEL binary dataset and record n, d, imbalance ratio, minority count, nominal features and missing values. If the package fails, request network access to the KEEL website instead.

## Phase: Foundations

### Literature

- **Search prior matching-based resampling work.** Use the literature-review skill with CrossRef, arXiv and web search to find prior uses of stable matching, assignment, roommates, optimal transport and statistical matching in undersampling or instance selection, and set evidence-backed novelty claims.
- **Build verified reference base.** Collect and DOI-verify core references: undersampling methods, matching theory (deferred acceptance, rural hospitals theorem, stable roommates, Kuhn-Munkres, Edmonds, uniqueness results), optimal matching in observational studies, evaluation methodology, KEEL, imbalanced-learn, complexity measures; export references.bib.
- **Write related-work synthesis and positioning.** Draft structured notes for the Introduction and Related Work sections, including a positioning table that compares the closest methods (NearMiss, cluster-based, StatMatch, OT undersampling) with the proposed family.

### Data

- **Curate KEEL benchmark datasets.** Select ~50-60 binary datasets with at least 10 minority instances using documented inclusion rules; encode nominal features, handle missing values, and label each dataset's base-data family for the independence robustness check.
- **Compute dataset characteristics and complexity.** Build the dataset table (n, d, IR, minority size) and compute data complexity measures (e.g. F1, N1, N3) plus the safe/borderline/rare/outlier minority typology for the later meta-analysis.
- **Define fixed cross-validation folds.** Generate 5x5 repeated stratified folds per dataset with recorded seeds and save them, so every sampler and classifier sees identical splits.

### Methods

- **Implement four matching undersamplers.** Write imbalanced-learn-compatible samplers: boundary-stable (hospitals-residents deferred acceptance with minority capacities), boundary-optimal (rectangular min-cost assignment), coverage-stable (stable-roommates pairing, keep one per pair, iterate) and coverage-optimal (Edmonds min-weight matching on a kNN graph, iterate), with exact target counts and seeds.
- **Prove and test theoretical properties.** Write proofs of greedy equivalence and uniqueness under distance-symmetric preferences, proposer invariance of the selected subset via the rural hospitals theorem, the ball characterization of the boundary-stable subset, and Wasserstein interpretations of the optimal variants. Unit-test each against reference solvers and brute force on tiny instances.
- **Profile runtime and illustrate selections.** Measure runtime scaling with majority size, and draft Figure 1: a 2D synthetic panel showing which majority points each matching method, RUS, NearMiss and cluster centroids retain (figure-style skill).

## Phase: Benchmark experiments

### Experiments

- **Build leakage-safe experiment runner.** Pipeline with scaling and resampling inside training folds only. Samplers: none, RUS, NearMiss-1/2/3, Tomek, ENN, NCR, OSS, CNN, IHT, ClusterCentroids, the 4 proposed methods, plus class weighting and SMOTE as references. Classifiers: CART, kNN, SVM-RBF, LR, RF with default settings. Logs per-fold metrics, selection diagnostics and runtimes, with checkpointing.
- **Run main benchmark.** Run all datasets x 25 folds x samplers x classifiers in parallel on 32 cores, check completeness, and save the raw per-fold results as a parquet artifact. Primary metrics G-mean and ROC-AUC; secondary balanced accuracy, F1, MCC, PR-AUC, sensitivity, specificity.
- **Run sensitivity and ablation experiments.** Vary the target ratio (1:1, 1:1.5, 1:2), the coverage keep-rule (random vs nearest-to-minority), the preference structure (Euclidean vs class-specific asymmetric) and the kNN candidate size, on all datasets with a reduced classifier set.

## Phase: Statistical analysis

### Statistics

- **Run rank-based omnibus and post-hoc tests.** For each classifier and primary metric: average ranks, Friedman/Iman-Davenport, Wilcoxon signed-rank with Holm correction for each proposed method vs each baseline, CD diagrams, and win/tie/loss counts.
- **Run Bayesian and robustness analyses.** Bayesian signed-rank tests with a ROPE (proposed vs RUS and vs the best baseline); robustness checks with one dataset per family, IR groups (<9 vs >=9) and secondary metrics; export LaTeX tables and publication figures.

### Mechanisms

- **Compute selection diagnostics.** For each sampler, quantify boundary proximity of the retained majority, Wasserstein/energy distance to the minority and to the full majority, and the share of noisy/borderline majority retained, mapped onto the boundary-vs-coverage desiderata.
- **Analyse when matching helps.** Meta-regression of performance gains (proposed minus RUS and minus best baseline) on IR, size, dimensionality and complexity measures, with interpretable plots and a runtime comparison figure.

## Phase: Manuscript

### Manuscript

- **Derive paper narrative and figure plan.** Run the paper-narrative skill on a draft abstract and the figure deck to fix the storyline and Figure 1, then finalize every figure with the figure-style skill.
- **Draft LaTeX manuscript.** Write the elsarticle manuscript: Introduction, Related Work, Preliminaries, Methods (algorithms, propositions, proofs), Experimental Design, Results, Discussion with limitations, Conclusion, and supplementary material, with every number generated from the results files.
- **Compile and verify manuscript.** Compile with tectonic in the latex environment, then cross-check every reported number, table and citation against the result artifacts and references.bib.
- **Referee-style self-review and revision.** Write an internal referee report in your usual format (summary, assessment, numbered major and minor comments), revise the manuscript to address it, and deliver the final PDF and source bundle.

## Planned outputs

- Manuscript PDF plus LaTeX source bundle (elsarticle .tex, references.bib, figures)
- Python module of the four matching undersamplers with unit tests
- Curated KEEL dataset registry and dataset-characteristics table
- Raw per-fold benchmark results (parquet) and summary tables
- Statistical analysis tables and figures (CD diagrams, Bayesian tests, mechanism and runtime plots)
- Internal referee-style review report of the draft

## Feasibility note at planning time

Running the full study and drafting the paper is feasible on this machine; whether matching-based undersampling beats random undersampling is the open empirical question, and the manuscript will report the outcome either way. Exact Edmonds matching on the largest majority classes will run on a kNN-sparsified graph, which the paper will state.
