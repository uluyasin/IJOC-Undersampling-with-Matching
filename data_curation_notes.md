# KEEL benchmark curation notes

Benchmark for the matching-based undersampling study: 55 binary KEEL datasets from 24 base-data families, their characteristics and complexity measures, and fixed 5x5 cross-validation folds. All numbers below were computed by the code that produced the artifacts (reproduction script in Section 10).

## 1. Source and loading

- Data come from the PyPI package `imbalanced-databases` 0.1.1, which bundles the files. Every loader was called as `load_<name>(return_X_y=False, encode=False)`; `encode=True`, `get_data_loaders()` and `summary()` were not used (they crash on scikit-learn >= 1.2 and label-encode nominal attributes as arbitrary ordinal codes).
- The package module defines 114 `load_*` functions: the 110 exported loaders (the input inventory, exactly the package's `__all__`), 3 unexported dataset loaders (`load_new_thyroid2`, `load_glass`, `load_vehicle`) and `load_artificial_data` (synthetic generator, ignored). The registry covers all 113 dataset loaders.
- Source classification uses the `citation` field each loader returns. It equals either the package's KEEL BibTeX (Alcala-Fdez et al. 2011, J. Mult.-Valued Logic Soft Comput. 17:255-287) or the KRNN BibTeX (Zhang, Tari & Cheriet 2017, Pattern Recognit. 62:33-44). This gives 100 KEEL loaders (99 exported + new_thyroid2) and 13 KRNN-collection loaders (11 exported + glass, vehicle).
- Every KEEL `.dat` file was also parsed directly from the package (`pkgutil.get_data`): header types (`real`/`integer` vs `{...}` nominal), comma-separated values, whitespace stripped, `?`/`<null>` treated as missing (none occur). For the 97 KEEL loaders that load correctly, the loader output is identical to the direct parse (values, column names, labels).
- Loader problems found and handled:
  1. `load_abalone19` raises ValueError because its 32 positive labels are written `' positive'` (leading space). The data were taken from the bundled file with whitespace stripped (n = 4174, n_min = 32, IR = 129.44).
  2. `load_ecoli_0_vs_1` has a target leak. The loader writes the label into a new lowercase `class` column, and the original `Class` column (the label itself, as bytes) stays in `data` as an 8th "feature". The leaked column is not used. In this file `positive` marks the *majority* class (143 vs 77), so y = 1 is the `negative` label here (minority = 77).
  3. `new_thyroid2` (unexported) is an exact row permutation of `new_thyroid1`: the two files have identical (x, y) multisets. It is excluded as a duplicate. No other pair of KEEL files is duplicated.
  4. The `.dat.bak` files in the package differ from the `.dat` files only by the KEEL `@inputs`/`@outputs` header lines.

## 2. Inclusion rules and counts

| Stage | Rule | Removed | Remaining |
|---|---|---|---|
| start | all bundled dataset loaders | - | 113 |
| (i) | KEEL origin (citation field); non-KEEL = KRNN collection: ada, cm1, german, glass, hepatitis, hiva, hypothyroid, kc1, pc1, satimage, spectf, sylva, vehicle | 13 | 100 |
| (i) | bundled dataset-level duplicate (new_thyroid2 = permutation of new_thyroid1) | 1 | 99 |
| (ii) | binary target (all are binary) | 0 | 99 |
| (iii) | at least 10 minority instances; removed: ecoli_0_1_3_7_vs_2_6 (7), glass5 (9), glass_0_1_6_vs_5 (9), glass_0_4_vs_5 (9), glass_0_6_vs_5 (9), lymphography_normal_fibrosis (6), poker_9_vs_7 (8), shuttle_c2_vs_c4 (6), winequality_white_9_vs_4 (5), zoo_3 (5) | 10 | 89 (24 families) |
| (iv) | at most 4 datasets per base-data family | 34 | **55** |

**Cap choice.** Eligible datasets per family after (i)-(iii): ecoli 18, yeast 14, glass 9, abalone 7, kddcup 5, kr-vs-k 5, vehicle 4, winequality-red 4, poker 3, shuttle 3, car 2, page-blocks 2, winequality-white 2, cleveland 1, dermatology 1, flare 1, haberman 1, iris 1, led7digit 1, new-thyroid 1, pima 1, segment 1, vowel 1, wisconsin 1. Total retained as a function of the cap: 3 -> 47, 4 -> 55, 5 -> 61. A cap of 4 is the only cap whose total falls in [50, 60].

**IR-span rule (deterministic) for families above the cap.** For a family with m > 4 eligible datasets, take c = 4 targets equally spaced in log(IR) between the family's minimum and maximum IR. For each target in increasing order, keep the not-yet-kept dataset with the smallest |log IR - target| (ties broken alphabetically). The rule always keeps the family's minimum- and maximum-IR datasets.

| Family | Eligible | Kept (IR) | log-spaced IR targets |
|---|---|---|---|
| ecoli | 18 | ecoli_0_vs_1 (1.86), ecoli1 (3.36), ecoli3 (8.6), ecoli4 (15.8) | 1.86, 3.79, 7.74, 15.8 |
| yeast | 14 | yeast1 (2.46), yeast3 (8.1), yeast_1_vs_7 (14.3), yeast6 (41.4) | 2.46, 6.3, 16.15, 41.4 |
| glass | 9 | glass1 (1.82), glass_0_1_2_3_vs_4_5_6 (3.2), glass6 (6.38), glass4 (15.46) | 1.82, 3.71, 7.57, 15.46 |
| abalone | 7 | abalone9_18 (16.4), abalone_3_vs_11 (32.47), abalone_20_vs_8_9_10 (72.69), abalone19 (129.44) | 16.4, 32.66, 65.02, 129.44 |
| kddcup | 5 | kddcup_guess_passwd_vs_satan (29.98), kddcup_land_vs_portsweep (49.52), kddcup_buffer_overflow_vs_back (73.43), kddcup_rootkit_imap_vs_back (100.14) | 29.98, 44.82, 66.99, 100.14 |
| kr-vs-k | 5 | kr_vs_k_zero_one_vs_draw (26.63), kr_vs_k_three_vs_eleven (35.23), kr_vs_k_zero_vs_eight (53.07), kr_vs_k_zero_vs_fifteen (80.22) | 26.63, 38.46, 55.55, 80.22 |

Families with 4 or fewer eligible datasets are kept whole (vehicle 4, winequality-red 4, poker 3, shuttle 3, car 2, page-blocks 2, winequality-white 2, and 11 singletons).

## 3. Families

- Family = the original base dataset. Selected counts: abalone (4), car (2), cleveland (1), dermatology (1), ecoli (4), flare (1), glass (4), haberman (1), iris (1), kddcup (4), kr-vs-k (4), led7digit (1), new-thyroid (1), page-blocks (2), pima (1), poker (3), segment (1), shuttle (3), vehicle (4), vowel (1), winequality-red (4), winequality-white (2), wisconsin (1), yeast (4).
- winequality-red and winequality-white are separate families (different base files). new-thyroid (Thyroid gland data, 215 instances) and the excluded KRNN hypothyroid set come from different base data. All shuttle variants (c0-vs-c4, 2_vs_5, 6_vs_2-3) belong to one family. cleveland is the Cleveland heart-disease data.
- **One-dataset-per-family robustness subset (pre-specified, 24 datasets).** Within each family, the selected dataset with the median IR (for an even count, the lower of the two middle IRs) is flagged `family_representative` in the registry and the complexity table: abalone_3_vs_11, car_good, cleveland_0_vs_4, dermatology_6, ecoli1, flare_f, glass_0_1_2_3_vs_4_5_6, haberman, iris0, kddcup_land_vs_portsweep, kr_vs_k_three_vs_eleven, led7digit_0_2_4_5_6_7_8_9_vs_1, new_thyroid1, page_blocks0, pima, poker_8_9_vs_5, segment0, shuttle_6_vs_2_3, vehicle1, vowel0, winequality_red_8_vs_6, winequality_white_3_vs_7, wisconsin, yeast3.

## 4. Encoding (no scaling)

- Numeric attributes (`real`/`integer` in the KEEL header) are float64.
- Nominal attributes (header `{...}`) are one-hot encoded over the categories observed in the dataset, in header order. A nominal attribute with exactly 2 observed categories becomes one 0/1 indicator column for the later-declared category. An attribute with a single observed category is dropped.
- Missing values would be kept as NaN for in-fold imputation; none of the 100 KEEL files contains missing values.
- Constant columns (including nominal attributes with a single observed category) are dropped. Selected datasets affected:
  - ecoli_0_vs_1: 1 removed: Chg
  - kddcup_buffer_overflow_vs_back: 14 removed: Atr-1, Atr-6, Atr-7, Atr-8, Atr-10, Atr-11, Atr-14, Atr-17, Atr-18, Atr-19, Atr-20, Atr-21, Atr-33, Atr-34
  - kddcup_guess_passwd_vs_satan: 12 removed: Atr-6, Atr-7, Atr-8, Atr-13, Atr-14, Atr-17, Atr-18, Atr-19, Atr-20, Atr-12, Atr-15, Atr-16
  - kddcup_land_vs_portsweep: 15 removed: Atr-7, Atr-8, Atr-10, Atr-11, Atr-13, Atr-14, Atr-17, Atr-18, Atr-19, Atr-20, Atr-21, Atr-5, Atr-12, Atr-15, Atr-16
  - kddcup_rootkit_imap_vs_back: 8 removed: Atr-6, Atr-7, Atr-14, Atr-17, Atr-18, Atr-19, Atr-20, Atr-21
  - segment0: 1 removed: Region-pixel-count
- `feature_types` records 'numeric' or 'onehot' per column. The target is y in {0,1} (int8), with 1 = minority class.
- Caveat: KEEL declares some integer-valued count or code attributes as nominal (kddcup Atr-7, Atr-8, Atr-10, Atr-14, Atr-17, Atr-18, i.e. KDD Cup 99 counts such as wrong_fragment and num_access_files; flare-F activity/evolution codes and flare counts; kr-vs-k rank attributes). Following the header, they were one-hot encoded and not treated as ordinal. Affected selected datasets: flare_f, kddcup_buffer_overflow_vs_back, kddcup_guess_passwd_vs_satan, kddcup_land_vs_portsweep, kddcup_rootkit_imap_vs_back, kr_vs_k_three_vs_eleven, kr_vs_k_zero_one_vs_draw, kr_vs_k_zero_vs_eight, kr_vs_k_zero_vs_fifteen.

## 5. Dataset characteristics

- Selected datasets: n = 150-5472 (median 855); d = 3-66 after encoding; n_min = 10-559; IR = 1.82-129.44 (median 16.40).
- IR groups: low (IR < 9) 19 datasets, median IR 2.99; high (IR >= 9) 36 datasets, median IR 33.85.
- 15 datasets have nominal attributes (hence one-hot columns); 0 have missing values.
- `duplicate_rows` = n minus the number of distinct encoded feature vectors. 29 of 55 datasets contain exact duplicates; the highest shares are led7digit_0_2_4_5_6_7_8_9_vs_1 81%, flare_f 73%, kddcup_land_vs_portsweep 59%, kddcup_rootkit_imap_vs_back 56%, kddcup_buffer_overflow_vs_back 55%, kddcup_guess_passwd_vs_satan 42%.
- Conflicting duplicates (same x, different y) occur in 4 datasets: flare_f: 93 rows in 19 groups (duplicates 779/1066); haberman: 12 rows in 6 groups (duplicates 23/306); led7digit_0_2_4_5_6_7_8_9_vs_1: 74 rows in 7 groups (duplicates 360/443); page_blocks0: 35 rows in 10 groups (duplicates 79/5472).

## 6. Complexity measures (`dataset_complexity.csv`)

- Data: the full dataset, median-imputed (a no-op here) and MinMax-scaled to [0, 1] on the full data; Euclidean geometry.
- From problexity 0.5.11: F1, F1v, F2, F3, F4 (feature overlap), L1, L2, L3 (linearity, LinearSVC), N3, N4, LSC (neighbourhood), T2, T3, T4 (dimensionality), C1 (entropy of class proportions) and C2 (multi-class imbalance ratio, 1 - 1/IR_mc). problexity orients every measure as larger = more complex (its default score weights are all +1); for C1/C2, larger = more imbalanced. `np.random.seed(20261115)` is set before every measure, which makes L1-L3 (LinearSVC shuffling) and L3/N4 (random interpolation) reproducible.
- **N1 and N2 are not taken from problexity.** Its source deviates from the standard definitions (Lorena et al. 2019, ACM Comput. Surv. 52(5):107):
  - problexity `n1` counts cross-class MST *edges* divided by 2n, instead of the fraction of *vertices* incident to a cross-class MST edge. It also builds the graph with `csr_matrix(dist)`, which silently drops zero-distance (duplicate) edges. On a 4-point toy problem (line 0,1,2,3; y = 0,0,1,1) it returns 0.125 where the definition gives 0.5.
  - problexity `n2` uses `argsort(...)[:,1]` on the between-class distance matrix, i.e. the *second*-nearest enemy. On a toy problem with a known answer of 0.909 it returns 0.5.
  - Columns `N1` and `N2` hold reference implementations of the published definitions. N1 keeps exact duplicates as near-zero-length MST edges. N1 matches an independent networkx Kruskal MST, and N2 matches a naive loop. The raw package values are kept as `N1_problexity` and `N2_problexity`. Across the 55 datasets, Spearman rho(N1, N1_problexity) = 0.994 and rho(N2, N2_problexity) = 0.993, and N1_problexity/N1 ranges from 0.25 to 0.43. Rankings are therefore nearly unchanged, but absolute values differ; use `N1`/`N2`.
  - problexity's LSC and N3 were checked against naive implementations and agree exactly.
- No subsampling was needed. Every measure ran on the full data, including the largest set (page_blocks0, n = 5472, about 106 s for all measures). `complexity_subsampled` is False everywhere.
- Not computed: T1, Density, ClsCoef, Hubs (graph and hypersphere measures, not required, slow O(n^2) Python loops).
- `single_feature_separable` flags datasets where one feature alone separates the classes perfectly (range test on unscaled data): abalone_3_vs_11, iris0, kddcup_buffer_overflow_vs_back, kddcup_guess_passwd_vs_satan, kddcup_land_vs_portsweep, kddcup_rootkit_imap_vs_back, shuttle_2_vs_5, shuttle_6_vs_2_3, shuttle_c0_vs_c4. Expect ceiling effects and tied ranks on these.

## 7. Minority typology (Napierala & Stefanowski 2016)

- For every minority instance, count minority instances among its k = 5 nearest neighbours (Euclidean, same scaled data, self excluded by index): 4-5 safe, 2-3 borderline, 1 rare, 0 outlier. Proportions and counts are reported per dataset. `maj_overlap_share` is the share of majority instances with at most 2 majority instances among their 5 neighbours.
- Means across the 55 datasets: safe 0.486, borderline 0.189, rare 0.112, outlier 0.214; majority overlap share 0.021. Dominant type: safe 30, outlier 13, borderline 10, rare 2.
- Differences from the original: Napierala & Stefanowski used HVDM on raw attributes. Here the distance is Euclidean on MinMax-scaled data with one-hot nominal attributes, as fixed for the whole study. Exact duplicates count as neighbours, and ties at equal distance follow scikit-learn's brute-force ordering.

## 8. Fixed cross-validation folds (`cv_folds.joblib`)

- `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=20261115 + dataset_index)` (scikit-learn 1.9.1). dataset_index is the 0-based position in the Python-sorted list of the 55 selected names, so seeds run from 20261115 (abalone19) to 20261169 (yeast_1_vs_7). The seed is stored per dataset and also appears in the registry (`cv_seed`).
- Structure: `name -> {'seed', 'dataset_index', 'n_splits', 'n_repeats', 'splitter', 'folds'}`. `folds` is a list of 25 dicts with `repeat` (0-4), `fold` (0-4), `train_idx` and `test_idx` (int32 row indices into `benchmark_datasets.joblib[name]['X']`).
- Checked: every training fold has >= 8 minority instances and every test fold >= 2 (minimum over all 1375 folds: 8 and 2); train and test are disjoint; the test folds of each repeat partition the data; folds regenerated from the stored seed are identical.

## 9. Caveats and recommendations for the analysis

1. Family dependence: 24 independent base datasets, with ecoli, yeast, glass, abalone, kddcup, kr-vs-k, vehicle and winequality-red contributing 4 datasets each (32 of 55). Report the pre-specified one-per-family subset (Section 3) alongside the full benchmark.
2. Exact duplicates can fall on both sides of a train/test split, which favours memorising classifiers (kNN, RF) on flare-F, led7digit, kddcup, wisconsin and segment0. Zero distances also create ties for the matching algorithms, so implementations should break ties deterministically (for example by index).
3. Small minority classes: shuttle_6_vs_2_3 and winequality_red_3_vs_5 have n_min = 10 (2 minority instances per test fold); 8 datasets have n_min < 20. Per-fold AUC/F1/G-mean will be noisy, so aggregate per repeat or over the 5x5 folds before ranking.
4. The 9 single-feature-separable datasets will likely give near-perfect scores for all methods, producing ties in Friedman/Nemenyi ranks.
5. The IR groups are unbalanced (19 low vs 36 high); group-wise analyses have less power for the low group.
6. The bundled files were not re-checked against the KEEL website (the sandbox has no access to it). They are the package's copies; for example, the KEEL web version of new-thyroid2 may differ from the bundled duplicate.

## 10. Reproduction

[curate_keel_benchmark.py](../pipeline/curate_keel_benchmark.py) (`python curate_keel_benchmark.py --out OUTDIR`) rebuilds all five data artifacts from the installed package; `--skip-complexity` skips Sections 6-7. It asserts that the loader cross-check finds exactly the two known data problems (abalone19, ecoli_0_vs_1) and the three unexported loaders. A fresh run in a separate process reproduced the saved artifacts: `dataset_registry.csv` and `dataset_table.tex` byte-identical; `benchmark_datasets.joblib` arrays and metadata identical; all 1375 folds identical; `dataset_complexity.csv` identical except the wall-clock `complexity_seconds` column. Full run time is about 8 min single-process (complexity dominates).

## Artifacts

- [benchmark_datasets.joblib](../data/benchmark_datasets.joblib): dict name -> X (float64, NaN allowed, unscaled), y (int8, 1 = minority), feature_names, feature_types, family, source, keel_name, loader, keel_file, minority_label.
- [dataset_registry.csv](../data/dataset_registry.csv): all 113 bundled dataset loaders with source, family, included flag, exclusion reason, characteristics, dataset_index/cv_seed and loader issues.
- [dataset_complexity.csv](../data/dataset_complexity.csv): complexity measures, minority typology, flags and timings for the 55 selected datasets.
- [cv_folds.joblib](../data/cv_folds.joblib): fixed folds with seeds.
- [curate_keel_benchmark.py](../pipeline/curate_keel_benchmark.py): standalone pipeline that regenerates all of the above.
- [dataset_table.tex](extended_results/tables/dataset_table.tex): booktabs table (name, family, n, d, n_min, IR), sorted by IR.
