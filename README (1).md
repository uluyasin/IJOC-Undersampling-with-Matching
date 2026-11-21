[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Boundary or Coverage, Stable or Optimal? A Computational Study of Matching-Based Undersampling for Imbalanced Classification

This archive is distributed in association with the [INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper [Boundary or Coverage, Stable or Optimal? A Computational Study of Matching-Based Undersampling for Imbalanced Classification](https://doi.org/10.1287/ijoc.2026.XXXX) by [author names withheld for double-anonymous review].

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2026.XXXX

https://doi.org/10.1287/ijoc.2026.XXXX.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{MatchingUndersampling2026,
  author =        {Anonymous},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Boundary or Coverage, Stable or Optimal? A Computational Study of Matching-Based Undersampling for Imbalanced Classification}},
  year =          {2026},
  doi =           {10.1287/ijoc.2026.XXXX.cd},
  url =           {https://github.com/INFORMSJoC/2026.XXXX},
  note =          {Available for download at https://github.com/INFORMSJoC/2026.XXXX},
}
```

## Description

This repository contains the code, the input data and the instructions needed to reproduce a computational study of matching-based undersampling for imbalanced binary classification. Four matching undersamplers form a 2x2 design. The *boundary arm* matches minority to majority instances and keeps the matched majority instances; the *coverage arm* pairs majority instances and keeps one member of each pair, round after round, until the target size is reached. Each arm is solved either with a *stable* matching (deferred acceptance or stable roommates) or with an *optimal* matching (minimum-cost assignment or minimum-weight perfect matching). The four methods are MUS-BS, MUS-BO, MUS-CS and MUS-CO (boundary/coverage arm, stable/optimal concept). They are compared with no resampling and eleven standard undersamplers of imbalanced-learn, with SMOTE and class weighting as non-undersampling references, on 55 binary KEEL datasets with five classifiers and five-times repeated stratified five-fold cross-validation.

Result files, tables and figures are not stored in the repository: the scripts write them to `results/`, `figs/` and `figs_ijoc/` (section [Replicating](#replicating)). The paper and its Online Supplement summarise the experiments; the complete experimental design and all detailed result tables and figures are in the extended-results document, whose LaTeX sources are in `docs/extended_results/`.

### Repository contents

| Folder | File | Purpose |
|---|---|---|
| (root) | `README.md`, `LICENSE`, `AUTHORS`, `requirements.txt`, `SHA256SUMS.txt`, `.gitignore` | This file; MIT License; author placeholder (double-anonymous review); pinned Python dependencies; SHA-256 checksums of all other files; files and folders excluded from version control (the generated outputs). |
| `src/` | `matchsampling.py` | The four matching undersamplers as one imbalanced-learn compatible estimator, `MatchingUnderSampler(arm, concept, ...)`, with the aliases `MUS_BS`, `MUS_BO`, `MUS_CS` and `MUS_CO`. |
| `tests/` | `test_matchsampling.py` | Test suite (289 tests: stability, uniqueness, rural-hospitals invariance, ball characterisation, Wasserstein optimality against brute force, agreement with reference solvers, exact output sizes). |
| `data/` | `benchmark_datasets.joblib`, `cv_folds.joblib`, `dataset_registry.csv`, `dataset_complexity.csv` | Curated datasets, the 5x5 cross-validation folds, the registry of all bundled KEEL datasets (inclusion flags, exclusion reasons, seeds) and the complexity measures. |
| | `dedup_datasets.joblib`, `dedup_folds.joblib` | Datasets without exact duplicate rows and their folds (duplicates experiment). |
| | `mus_co_complete_graph_optima.csv` | Test instances of the candidate-graph study of MUS-CO and their certified complete-graph optima (see Notes). |
| `pipeline/` | `curate_keel_benchmark.py` | Builds the 55-dataset KEEL benchmark, the dataset registry, the complexity measures and the 5x5 cross-validation folds. |
| | `make_dedup_inputs.py` | Inputs of the duplicates experiment: deduplicated datasets, their folds and the duplicate counts per dataset. |
| | `bench_runner.py`, `merge_runs.py` | Leakage-safe experiment runner (imputation and scaling fitted on the training fold; one resampling per dataset, fold and sampler); merging of its shard outputs. |
| | `stats_pipeline.py` | Friedman/Iman-Davenport, Nemenyi, Wilcoxon-Holm, Bayesian signed-rank, 2x2 factorial, robustness and ablation analyses. |
| | `mechanisms_pipeline.py` | Selection diagnostics (boundary proximity, coverage error, energy distances, unsafe-instance enrichment, overlaps), dataset-level gain analysis and run times under load. |
| | `figures_pipeline.py` | Figures in the original layout (used in the extended results); every plotted number is recomputed, checked against the analysis tables and listed in `figure_numbers.csv`. |
| | `figures_ijoc.py`, `split_fig3_ijoc.py` | Figures in the IJOC format (font sizes scaled for the 6.5-in text width), written to `figs_ijoc/`; splitting of the mechanism figure into panels (a)-(b) (Figure 4 of the paper) and (c)-(d) (Figure S1 of the Online Supplement). |
| | `mus_co_k_sweep.py` | Candidate-graph study of MUS-CO (Table S1 of the Online Supplement). |
| | `extra_samplers.py` | Samplers of the additional experiments: local pivotal method (LPM2), k-means-stratified random undersampling, row-permutation wrapper, and the two intervention variants of MUS-BS (safecap, nounsafe). |
| | `make_sel_small.py`, `referee_runner.py`, `referee_analysis.py`, `lpm_validation.py` | Runner and analysis of the additional experiments (threshold adjustment and tuning, row order, intervention, spatially balanced baselines, ensembles, duplicates); synthetic validation of LPM2. |
| | `revision_analyses/` (`rev_common.py`, `rev_heavy.py`, `rev_bayes.py`, `rev_analyses.py`, `paths.json`) | Additional analyses: family-clustered bootstrap, pooled confusion matrices, ROPE sensitivity and Bayesian tests against ClusterCentroids, primary family of 40 tests, mechanism correlations without near-duplicates, per-dataset corrected standard errors, exact Wasserstein distortions and coverage bounds, greedy versus optimal matching costs, candidate-graph gap. |
| | `timing_isolated.py`, `summarize_timing_isolated.py`, `fig5_isolated.py` | Isolated re-timing (single process, one thread) on one fold per dataset and synthetic scaling up to 100,000 majority instances; per-sampler summary; the cost figure in the original layout. |
| | `selection_illustration.py` | Selections of all methods on the two-dimensional example (figure of the extended results). |
| `configs/` | `cfg_*.json`, `figures_inputs.json` | Run configurations for `bench_runner.py` and `referee_runner.py`; input map of the figure scripts. |
| `docs/` | `extended_results/` | LaTeX sources of the extended-results document (complete experimental design and all detailed results; see `docs/extended_results/README.md`). |
| | `study_plan.json`, `study_plan.md` | The study plan, created before the benchmark was run (timestamp and SHA-256 in `study_plan.md`). |
| | `data_curation_notes.md`, `environment.txt` | Curation decisions and loader repairs; complete frozen Python environment. |

### Using the undersamplers

```python
import sys; sys.path.insert(0, "src")
from matchsampling import MatchingUnderSampler, MUS_CS

sampler = MatchingUnderSampler(arm="coverage", concept="stable", random_state=0)  # same as MUS_CS(random_state=0)
X_res, y_res = sampler.fit_resample(X_train, y_train)

# inside an imbalanced-learn pipeline
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
model = make_pipeline(MUS_CS(random_state=0), RandomForestClassifier(random_state=0))
```

Features should be scaled before resampling; the experiments use min-max scaling fitted on the training fold. The boundary arm (`arm="boundary"`) matches minority to majority instances and keeps the matched majority instances; the coverage arm (`arm="coverage"`) pairs majority instances and keeps one member of each pair, round after round, until the target size is reached. `concept="stable"` uses deferred acceptance or stable roommates (under distance preferences both reduce to greedy nearest-available matching); `concept="optimal"` uses a minimum-cost assignment (boundary arm) or a minimum-weight perfect matching that is exact on a candidate graph of `n_candidates=10` nearest neighbours plus the stable matching (coverage arm). Other options: `sampling_strategy` (imbalanced-learn semantics), `keep_rule` (`"random"` or `"boundary"`, coverage arm) and `preferences` (`"symmetric"` or `"asymmetric"` Mahalanobis preferences, boundary-stable method).

### Data

The datasets come from the KEEL imbalanced data repository (Alcala-Fdez et al., 2011), obtained through the imbalanced-databases package; please cite the original sources when using them. The benchmark contains 55 binary datasets from 24 base-data families, selected from the files bundled with imbalanced-databases 0.1.1 by documented rules: KEEL origin, binary, at least ten minority instances, at most four datasets per base-data family. The curation decisions are documented in `docs/data_curation_notes.md` and in Section E1 of the extended results; `data/dataset_registry.csv` lists all bundled datasets with inclusion flags and exclusion reasons, and `data/dataset_complexity.csv` the complexity measures and the minority-instance typology. The MIT License of this repository covers its code and documentation; the datasets remain subject to the terms of their original sources.

### Experimental protocol and seeds

- Five-times repeated stratified five-fold cross-validation (25 train-test splits per dataset), identical for all methods: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=cv_seed)` with `cv_seed = 20261115 + dataset_index` (column `cv_seed` of `data/dataset_registry.csv`; the folds are stored in `data/cv_folds.joblib`).
- Within each training fold, median imputation and min-max scaling to [0, 1] are fitted on the training data and applied to the test fold; the sampler is applied once per dataset, fold and sampler to the scaled training fold; every classifier is trained on the resampled data and evaluated on the untouched test fold.
- Seed of the sampler and of the classifiers on fold `k` (`k = 0, ..., 24`): `cv_seed * 100 + k` (`pipeline/bench_runner.py`; reused by `pipeline/referee_runner.py`).
- Classifiers (scikit-learn defaults otherwise): CART; k-NN with k = 5; SVM with RBF kernel, C = 1, gamma = scale; L2 logistic regression with C = 1 (max_iter = 2000); random forest with 100 trees. Primary metrics: G-mean and ROC-AUC; secondary: balanced accuracy, F1, MCC, PR-AUC.
- Complexity measures: `np.random.seed(20261115)` before every measure. Bayesian signed-rank tests: prior 0.5, 50,000 posterior samples, `np.random.seed(20261115)` before every test. Family-clustered bootstrap: B = 5000, seed 20261120.
- The full protocol, the compared methods and the statistical analysis are described in Section E2 of the extended results.

## Building

The code is pure Python; nothing needs to be compiled. All experiments were run with CPython 3.11.18 on 64-bit Windows in a conda environment named `imbml`. To recreate it:

```
conda create -n imbml python=3.11
conda activate imbml
pip install -r requirements.txt
pip install --no-deps imbalanced-databases==0.1.1
```

Versions used for all experiments (pinned in `requirements.txt`; the complete frozen environment, including transitive dependencies, is `docs/environment.txt`):

| Package | Version | Use |
|---|---|---|
| numpy | 2.4.6 | |
| scipy | 1.17.1 | |
| pandas | 3.1.1 | |
| pyarrow | 26.0.1 | parquet input/output |
| scikit-learn | 1.9.1 | classifiers, preprocessing, folds |
| imbalanced-learn | 0.14.2 | standard undersamplers, SMOTE, sampler interface |
| networkx | 3.6.1 | Edmonds fallback (MUS-CO), Gomory-Hu trees |
| statsmodels | 0.15.0 | LOWESS |
| baycomp | 1.0.2 | Bayesian signed-rank tests |
| problexity | 0.5.11 | data-complexity measures |
| matplotlib | 3.11.2 | figures |
| joblib | 1.6.0 | data and selection files |
| scikit-posthocs | 0.18.0 | part of the experiment environment; not imported by the scripts |
| imbalanced-databases | 0.1.1 | KEEL dataset loader (install with `--no-deps`) |
| matching | 1.4.3 | reference solver in the tests |
| pytest | 9.2.0 | tests |

Notes:

- imbalanced-databases is installed with `--no-deps` because its metadata declares the deprecated PyPI name `sklearn` (scikit-learn itself is installed by `requirements.txt`). Its `get_data_loaders()` and `summary()` are incompatible with scikit-learn >= 1.2 (`OneHotEncoder` keyword); `pipeline/curate_keel_benchmark.py` calls the `load_*` functions with `encode=False`.
- pytest is not part of the experiment environment; the tests were run in a separate environment with the same pinned core versions.
- Hardware of all runs and timings: Intel Core i9-13900HX (32 logical processors), 32 GB of memory, Windows.
- The extended-results document needs a LaTeX distribution with pdflatex and BibTeX (packages listed in `docs/extended_results/README.md`).

## Results

The repository holds no result files; every result of the paper, the Online Supplement and the extended results is regenerated by the steps in [Replicating](#replicating). The per-fold results are written to `results/`, the analysis tables to `results/analysis/`, `results/revision/` and `results/revision/analyses/`, the figures to `figs/` (original layout) and `figs_ijoc/` (IJOC format). `figs/figure_numbers.csv` lists every number shown in the figures and their captions.

### Output files

| File | Content | Written by (step) |
|---|---|---|
| `results/results_main_final.parquet` | Per-fold results of the main experiment: 55 datasets x 25 folds x 18 configurations x 5 classifiers (class weighting is undefined for k-NN), 122,375 rows | `bench_runner.py`, `merge_runs.py` (2) |
| `results/selections_main_final.joblib` | Retained majority indices of the 15 undersamplers for every training fold (20,625 records) | `bench_runner.py`, `merge_runs.py` (2) |
| `results/results_ablation.parquet` | Per-fold results of the ablations (14 variants; CART, k-NN, SVM), 57,750 rows | `bench_runner.py`, `merge_runs.py` (2) |
| `results/analysis/stats_*.csv`, `.tex` | Ranks, Friedman/Iman-Davenport, Wilcoxon-Holm, Bayesian, factorial, robustness, reference and ablation analyses | `stats_pipeline.py` (3) |
| `results/analysis/diag_*.csv`, `meta_*.csv`, `runtime_*.csv`, `selection_overlap*.csv` | Selection diagnostics, dataset-level gains, run times under load, selection overlaps | `mechanisms_pipeline.py` (4) |
| `figs/` | Figures in the original layout, `figure_numbers.csv` | `figures_pipeline.py` (5) |
| `results/mus_co_k_sweep.csv` | Candidate-graph study of MUS-CO | `mus_co_k_sweep.py` (6) |
| `results/revision/x1_*` ... `x6_*`, `x6_duplicate_counts.csv` | Additional experiments X1-X6 (per fold and summaries); duplicate counts | `referee_runner.py`, `referee_analysis.py` (7); `make_dedup_inputs.py` (1) |
| `results/revision/analyses/A1_*` ... `A9_*`, `analyses_numbers.json` | Additional analyses A1-A9 | `revision_analyses/rev_heavy.py`, `rev_bayes.py`, `rev_analyses.py` (8) |
| `results/revision/timing_isolated_benchmark.csv`, `timing_isolated_scaling.csv`, `timing_isolated_summary.csv` | Isolated timings (one fold per dataset; synthetic scaling; per-sampler summary) | `timing_isolated.py`, `summarize_timing_isolated.py` (9) |
| `figs_ijoc/` | Figures in the IJOC format | `figures_ijoc.py`, `split_fig3_ijoc.py` (10) |
| `docs/extended_results/figures/figS_selection_illustration.pdf` | Selections on the two-dimensional example | `selection_illustration.py` (11) |
| `pipeline/lpm_validation.csv` | Synthetic validation of LPM2 and KSRUS | `lpm_validation.py` (12) |

### Tables and figures of the paper and the Online Supplement

| Paper | Content | Source data | Script (step) |
|---|---|---|---|
| Figure 1 | What is matched matters, how it is matched does not | `results/analysis/stats_factorial_2x2.csv`, `stats_per_dataset_means.csv`; toy data generated in the script | `figures_ijoc.py` (10): `figs_ijoc/fig1_hook.pdf` |
| Figure 2 | Comparison with random undersampling and the standard undersamplers | `results/analysis/stats_bayes.csv`, `stats_ranks_friedman.csv`, `stats_wilcoxon_holm.csv` | `figures_ijoc.py` (10): `figs_ijoc/fig2_evidence.pdf` |
| Figure 3 | Computing time and accuracy of the fifteen undersamplers | `results/revision/timing_isolated_summary.csv`, `results/analysis/stats_ranks_friedman.csv` | `figures_ijoc.py` (10): `figs_ijoc/fig5_cost.pdf` |
| Figure 4 | Why the arms differ, panels (a)-(b) | `results/analysis/diag_by_dataset.csv`, `stats_per_dataset_means.csv` | `figures_ijoc.py`, `split_fig3_ijoc.py` (10): `figs_ijoc/fig3_mechanism_ab.pdf` |
| Table 1 | The 2x2 family of undersamplers | (no results) | - |
| Table 2 | Does optimality pay? Solution quality, accuracy and computing time | `results/revision/analyses/A7_*.csv`, `A8_*.csv`; `results/analysis/selection_overlap.csv`, `stats_factorial_2x2.csv`; `results/revision/timing_isolated_summary.csv`, `timing_isolated_scaling.csv` | steps 3, 4, 8, 9 |
| Table S1 | Gap between the minimum-weight matching on the candidate graph and on the complete graph | `results/mus_co_k_sweep.csv` | `mus_co_k_sweep.py` (6) |
| Table S2 | Nested threshold adjustment and tuning | `results/revision/x1_summary.csv` | `referee_runner.py`, `referee_analysis.py` (7) |
| Table S3 | Interventions on the boundary arm | `results/revision/x3_summary.csv` | `referee_runner.py`, `referee_analysis.py` (7) |
| Table S4 | Bayesian signed-rank tests in G-mean for four ROPE widths | `results/revision/analyses/A3_bayes_rope.csv` | `rev_bayes.py`, `rev_analyses.py` (8) |
| Figure S1 | Selection diagnostics and accuracy, panels (c)-(d) of Figure 4 | as Figure 4 | `split_fig3_ijoc.py` (10): `figs_ijoc/fig3_mechanism_cd.pdf` |
| Figure S2 | Dataset-level G-mean difference to RUS against the share of safe minority examples | `results/analysis/stats_per_dataset_means.csv`, `data/dataset_complexity.csv` | `figures_ijoc.py` (10): `figs_ijoc/fig4_when.pdf` |

### Tables and figures of the extended results

Numbers refer to the compiled extended-results document (`docs/extended_results/`), which numbers its sections, tables, figures and algorithms with the prefix E, as cited in the paper and the Online Supplement; labels are the LaTeX labels. The figure files in `docs/extended_results/figures/` are copies of the figures in the original layout written by `figures_pipeline.py` (to `figs/`) and `stats_pipeline.py` (`fig_ablation_ratio`, to `results/analysis/`), and of the output of `selection_illustration.py`.

| Extended results | Label | Content | Source data | Script (step) |
|---|---|---|---|---|
| Table E1 | `tab:datasets` | Benchmark datasets | `data/dataset_registry.csv` | `curate_keel_benchmark.py` (1; also writes `dataset_table.tex`) |
| Table E2 | `tab:analysis_status` | Status of the analyses (planned/exploratory) | `docs/study_plan.md`, `docs/study_plan.json` | - |
| Table E3 | `tab:factorial` | 2x2 factorial contrasts | `results/analysis/stats_factorial_2x2.csv` | `stats_pipeline.py` (3) |
| Table E4 | `tab:ranks_friedman` | Average ranks, Friedman/Iman-Davenport, Nemenyi CD | `results/analysis/stats_ranks_friedman.csv` | `stats_pipeline.py` (3) |
| Table E5 | `tab:wilcoxon_gmean` | Wilcoxon-Holm tests, G-mean | `results/analysis/stats_wilcoxon_holm.csv` | `stats_pipeline.py` (3) |
| Table E6 | `tab:x1` | Threshold adjustment and tuning (X1) | `results/revision/x1_summary.csv` | `referee_runner.py`, `referee_analysis.py` (7) |
| Table E7 | `tab:x3` | Intervention on the boundary arm (X3) | `results/revision/x3_summary.csv` | `referee_runner.py`, `referee_analysis.py` (7) |
| Table E8 | `tab:diag_summary` | Selection diagnostics | `results/analysis/diag_summary.csv`, `diag_by_dataset.csv` | `mechanisms_pipeline.py` (4) |
| Figure E1 | `fig:mechanism` | Why the arms differ (panels (a)-(b) also Figure 4 of the paper) | `results/analysis/diag_by_dataset.csv`, `stats_per_dataset_means.csv` | `figures_pipeline.py` (5) |
| Figure E2 | `fig:when` | When does the boundary arm lose less? | `results/analysis/stats_per_dataset_means.csv`, `data/dataset_complexity.csv` | `figures_pipeline.py` (5) |
| Table E9 | `tab:x4` | Spatially balanced baselines (X4) | `results/revision/x4_summary.csv` | `referee_runner.py`, `referee_analysis.py` (7) |
| Table E10 | `tab:x5` | Ensembles and seed variability (X5) | `results/revision/x5_summary.csv` | `referee_runner.py`, `referee_analysis.py` (7) |
| Figure E3 | `fig:illustration` | Selections on a two-dimensional example | generated in the script | `selection_illustration.py` (11) |
| Table E11 | `tab:wilcoxon_auc` | Wilcoxon-Holm tests, ROC-AUC | `results/analysis/stats_wilcoxon_holm.csv` | `stats_pipeline.py` (3) |
| Tables E12, E13 | `tab:bayes_gmean`, `tab:bayes_auc` | Bayesian signed-rank tests (ROPE 0.01) | `results/analysis/stats_bayes.csv` | `stats_pipeline.py` (3) |
| Table E14 | `tab:a3_bayes_rope` | ROPE sensitivity, Bayesian tests against ClusterCentroids (A3) | `results/revision/analyses/A3_bayes_rope.csv` | `rev_bayes.py`, `rev_analyses.py` (8) |
| Table E15 | `tab:robustness` | Robustness of the pairwise conclusions | `results/analysis/stats_robustness.csv` | `stats_pipeline.py` (3) |
| Table E16 | `tab:x6` | Exact duplicates (X6) | `results/revision/x6_summary.csv`, `x6_duplicate_counts.csv` | `make_dedup_inputs.py` (1), `referee_runner.py`, `referee_analysis.py` (7) |
| Table E17 | `tab:a1_family_bootstrap` | Family-clustered bootstrap (A1) | `results/revision/analyses/A1_family_bootstrap.csv` | `rev_analyses.py` (8) |
| Table E18 | `tab:a2_pooled` | Pooled confusion matrices (A2) | `results/revision/analyses/A2_pooled_confusion.csv` | `rev_analyses.py` (8) |
| Tables E19, E20 | `tab:reference_gmean`, `tab:reference_auc` | Comparison with SMOTE and class weighting | `results/analysis/stats_reference_smote_cw.csv` | `stats_pipeline.py` (3) |
| Table E21 | `tab:meta_gain_corr` | Gains vs dataset characteristics | `results/analysis/meta_gain_correlations.csv`, `meta_gains_per_dataset.csv` | `mechanisms_pipeline.py` (4) |
| Table E22 | `tab:a6_per_dataset` | Per-dataset differences with corrected standard errors (A6) | `results/revision/analyses/A6_per_dataset_nadeau_bengio.csv` | `rev_analyses.py` (8) |
| Table E23 | `tab:overlap` | Selection overlap (Jaccard) | `results/analysis/selection_overlap.csv` | `mechanisms_pipeline.py` (4) |
| Table E24 | `tab:a7a8` | Wasserstein distortion, greedy vs optimal matching cost (A7, A8) | `results/revision/analyses/A7_wasserstein_by_dataset.csv`, `A7_wasserstein_folds.csv`, `A8_coverage_rounds.csv`, `A8_complete_graph.csv` | `rev_heavy.py`, `rev_analyses.py` (8) |
| Table E25 | `tab:ablation` | Ablations | `results/analysis/stats_ablation.csv` | `stats_pipeline.py` (3) |
| Figures E4, E5 | `fig:cd_gmean`, `fig:cd_auc` | Critical-difference diagrams (G-mean, ROC-AUC) | `results/analysis/stats_ranks_friedman.csv` | `figures_pipeline.py` (5) |
| Figure E6 | `fig:ablation` | Target-ratio ablation | `results/results_ablation.parquet` | `stats_pipeline.py` (3) |
| Figure E7 | `fig:map_full` | Boundary-coverage map of all methods | `results/analysis/diag_summary.csv`, `diag_by_dataset.csv` | `figures_pipeline.py` (5) |
| Figure E8 | `fig:runtime_grid` | Resampling time vs majority size (under load) | `results/analysis/runtime_by_dataset.csv`, `runtime_benchmark.csv` | `figures_pipeline.py` (5) |
| Table E26 | `tab:runtime` | Resampling time (under load and in isolation) | `results/analysis/runtime_benchmark.csv`; `results/revision/timing_isolated_summary.csv` | `mechanisms_pipeline.py` (4); `timing_isolated.py`, `summarize_timing_isolated.py` (9) |
| Table E27 | `tab:scaling` | Synthetic scaling in isolation | `results/revision/timing_isolated_scaling.csv` | `timing_isolated.py` (9) |
| Algorithms E1, E2 | `alg:boundary`, `alg:coverage` | The two arms (pseudocode) | - | implemented in `src/matchsampling.py` |
| Table E28 | `tab:ksweep` | Candidate degree of MUS-CO | `results/mus_co_k_sweep.csv` | `mus_co_k_sweep.py` (6) |

## Replicating

Run all commands from the root of this repository, in the order given; each step reads the outputs of the earlier steps.

1. Data. The curated inputs are included in `data/`; to rebuild them from imbalanced-databases 0.1.1, run `python pipeline/curate_keel_benchmark.py --out data`. Then run `python pipeline/make_dedup_inputs.py` (always needed: it also writes `results/revision/x6_duplicate_counts.csv`, read in step 7).
2. Experiments: `python pipeline/bench_runner.py configs/<cfg>.json <shard> <n_shards>` for every shard `0 ... n_shards-1` of each configuration (`python pipeline/bench_runner.py configs/<cfg>.json` runs all units in one process). Shards are independent processes; finished units are skipped on restart. Main experiment: `cfg_main_baselines.json`, `cfg_main_proposed_bsc.json`, `cfg_main_musco.json` (output in `results/main/`); ablations: `cfg_ablation_baselines.json`, `cfg_ablation_bsc.json` (output in `results/ablation/`). Then merge the shards: `python pipeline/merge_runs.py`. The resampling and fitting times recorded in the original runs add up to about 30 hours for the main experiment (20 of them for MUS-CO; measured with about 20 processes running in parallel) and about 1 hour for the ablations.
3. Statistics: `python pipeline/stats_pipeline.py --main results/results_main_final.parquet --ablation results/results_ablation.parquet --registry data/dataset_registry.csv --complexity data/dataset_complexity.csv --out results/analysis`, then `mv results/analysis/run_manifest.json results/run_manifest_stats.json`.
4. Selection diagnostics: `python pipeline/mechanisms_pipeline.py --results results/results_main_final.parquet --selections results/selections_main_final.joblib --data data/benchmark_datasets.joblib --folds data/cv_folds.joblib --registry data/dataset_registry.csv --complexity data/dataset_complexity.csv --out results/analysis --ms-path src/matchsampling.py`, then `mv results/analysis/run_manifest.json results/run_manifest_mechanisms.json`.
5. Figures in the original layout: `python pipeline/figures_pipeline.py --inputs configs/figures_inputs.json --out figs` (reads the analysis tables of steps 3 and 4 and stops if a recomputed number differs from them).
6. Candidate-graph study of MUS-CO: `python pipeline/mus_co_k_sweep.py` (writes `results/mus_co_k_sweep.csv`; about 2 minutes).
7. Additional experiments: `python pipeline/make_sel_small.py results/selections_main_final.joblib results/revision/sel_small.joblib`, then `python pipeline/referee_runner.py configs/cfg_referee.json <shard_id>` for as many shard processes as desired (tasks are claimed through files, finished tasks are skipped), then `python pipeline/referee_analysis.py configs/cfg_referee.json`. The threshold-adjustment and tuning experiment uses the first three repetitions (15 folds per dataset); MUS-CO is excluded from all additional experiments.
8. Additional analyses (paths in `pipeline/revision_analyses/paths.json`, relative to the repository root): `python pipeline/revision_analyses/rev_heavy.py plan`, `python pipeline/revision_analyses/rev_heavy.py run <task> <shard> <n_shards>` for the tasks `w1`, `rounds` and `complete`, then `python pipeline/revision_analyses/rev_bayes.py` and `python pipeline/revision_analyses/rev_analyses.py`.
9. Isolated timing: `python pipeline/timing_isolated.py AB` (about 1.5 hours on the machine described under Building), then `python pipeline/summarize_timing_isolated.py`.
10. Figures in the IJOC format: `python pipeline/figures_ijoc.py configs/figures_inputs.json figs_ijoc` (all nine figures; `fig5_cost` from the isolated timings), then `python pipeline/split_fig3_ijoc.py` (writes `figs_ijoc/fig3_mechanism_ab` and `fig3_mechanism_cd`). The cost figure in the original layout: `python pipeline/fig5_isolated.py configs/figures_inputs.json figs`.
11. Selection illustration: `python pipeline/selection_illustration.py` (writes `docs/extended_results/figures/figS_selection_illustration.pdf` and `.png`).
12. Synthetic validation of the local pivotal method and of k-means-stratified undersampling: `python pipeline/lpm_validation.py` (writes `pipeline/lpm_validation.csv`).
13. Tests: `python -m pytest tests -q` with `src` on the Python path, for example `PYTHONPATH=src python -m pytest tests -q -p no:cacheprovider` (POSIX shell) or `$env:PYTHONPATH="src"; python -m pytest tests -q -p no:cacheprovider` (PowerShell); 289 tests.
14. Extended-results document: in `docs/extended_results/`, run `pdflatex extended_results`, `bibtex extended_results` and `pdflatex extended_results` twice (or `latexmk -pdf extended_results.tex`).

### Notes on provenance and reproducibility

- The MUS-BS, MUS-BO and MUS-CS results (main experiment and ablations) were produced with an earlier revision of `matchsampling.py` whose code for these three samplers is identical to the released version (verified by diff); the MUS-CO results were produced with the released version. MUS-CO results of the earlier revision, which used a different exact solver, were discarded. The released file differs from the file used for the runs only in one corrected docstring (the description of `n_candidates`); the executable code is identical.
- The MUS-CO ablations were not run because of their cost; the effect of the candidate-graph size was studied instead (step 6).
- Candidate-graph study: the complete-graph optima in `data/mus_co_complete_graph_optima.csv` were certified during development by linear programming over all pairs (odd-set cuts and column generation; every reduced cost >= -1e-7 at the optimum). That development solver is not part of the repository, so the optima are provided as data; `pipeline/mus_co_k_sweep.py` regenerates the test instances and computes the candidate-graph optima with the released module. The candidate-graph costs are exact optima on the candidate graph and do not depend on the solver; the run times do.
- Run times in `results/analysis/runtime_*.csv` are measured while several benchmark processes run in parallel; `results/revision/timing_isolated_*.csv` repeats the measurement in isolation (single process, one thread).
- ClusterCentroids with hard voting is not bit-reproducible on some tie-heavy discrete datasets (car_good, car_vgood, kr_vs_k_three_vs_eleven, kr_vs_k_zero_one_vs_draw): a rerun reproduced the original per-fold results in all but 14 of the 825 rerun ClusterCentroids units (first three repetitions).
- The seeded keep coin of the coverage arm is attached to the pair members in row order, so permuting the training rows acts like a change of seed for MUS-CS; the boundary-arm selections depend on the row order only through ties.
- MUS-CO is optimal on its candidate graph (10 nearest neighbours plus the stable matching), not on the complete graph; the gap was a median 0.03% (at most 1.6%) on the folds that could be checked.
- The exact Wasserstein distances of the additional analyses are computed with a HiGHS transportation LP; 17 of them were cross-checked with network simplex.
- Two KEEL loaders of imbalanced-databases 0.1.1 need repairs, which `curate_keel_benchmark.py` applies: `ecoli_0_vs_1` keeps the original class column as a feature and labels the majority class as positive, and `abalone19` fails to parse; see `docs/data_curation_notes.md`.
- Naming: the folder `results/revision/`, the configuration `configs/cfg_referee.json` and the scripts `pipeline/referee_*.py` and `pipeline/revision_analyses/` hold the additional experiments and analyses (Section 4.3 of the paper). They were added after an internal pre-submission review of a first draft; the names were kept so that all paths used by the scripts remain valid.

## Ongoing Development

This repository is a snapshot of the code and data used for the paper and is not developed further here. Information on any later development version is withheld for double-anonymous review.

## Support

Contact information withheld for review.
