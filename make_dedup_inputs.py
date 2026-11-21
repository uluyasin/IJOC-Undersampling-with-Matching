"""Inputs of the exact-duplicates experiment (X6 in configs/cfg_referee.json).

Exact duplicates are rows with identical (encoded) features and identical label. For every benchmark dataset this script
counts them, and for the datasets that contain any, it removes the repeated rows (the first occurrence is kept; rows
whose feature vectors occur with both labels are therefore kept once per label) and draws new 5 x 5 stratified folds on
the remaining rows with the dataset's own seed from data/cv_folds.joblib.

Inputs : data/benchmark_datasets.joblib, data/cv_folds.joblib
Outputs: data/dedup_datasets.joblib   name -> {"X", "y", "orig_rows" (row numbers in the original dataset)}
         data/dedup_folds.joblib      name -> {"seed", "n_splits", "n_repeats", "folds", "splitter"}, folds as in
                                      data/cv_folds.joblib (entries "repeat", "fold", "train_idx", "test_idx")
         results/revision/x6_duplicate_counts.csv  one row per dataset:
             n, n_min                           rows and minority rows;
             n_exact_dup_rows (_min, _maj)      rows removed (repeated occurrences of an identical row), by class;
             n_rows_in_exact_groups             rows that belong to a group of identical rows (all occurrences);
             n_cross_class_groups               feature vectors that occur with both labels;
             n_rows_in_cross_class_groups       rows with such a feature vector;
             n_rows_with_any_feature_duplicate  rows whose feature vector occurs more than once (labels ignored);
             has_nan                            whether the feature matrix contains missing values.

Usage (from the repository root):
    python pipeline/make_dedup_inputs.py           # write the three files
    python pipeline/make_dedup_inputs.py --check   # compare with existing files instead of writing
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

DATA = "data/benchmark_datasets.joblib"
FOLDS = "data/cv_folds.joblib"
OUT_DATA = "data/dedup_datasets.joblib"
OUT_FOLDS = "data/dedup_folds.joblib"
OUT_COUNTS = "results/revision/x6_duplicate_counts.csv"
N_SPLITS, N_REPEATS = 5, 5


def counts_and_keep(X, y):
    F = pd.DataFrame(X)
    Fy = F.assign(__label__=y)
    dup_first = Fy.duplicated(keep="first").to_numpy()
    in_exact = Fy.duplicated(keep=False).to_numpy()
    key = pd.util.hash_pandas_object(F, index=False).to_numpy()
    n_labels = pd.Series(y).groupby(key).nunique()
    cross = n_labels[n_labels > 1].index
    row = dict(n=int(len(y)), n_min=int((y == 1).sum()), n_exact_dup_rows=int(dup_first.sum()),
               n_exact_dup_min=int((dup_first & (y == 1)).sum()), n_exact_dup_maj=int((dup_first & (y == 0)).sum()),
               n_rows_in_exact_groups=int(in_exact.sum()), n_cross_class_groups=int(len(cross)),
               n_rows_in_cross_class_groups=int(np.isin(key, cross).sum()),
               n_rows_with_any_feature_duplicate=int(F.duplicated(keep=False).sum()),
               has_nan=bool(np.isnan(np.asarray(X, dtype=float)).any()))
    return row, ~dup_first


def build(data, folds):
    rows, dd, df = [], {}, {}
    for name, ds in data.items():
        X, y = ds["X"], np.asarray(ds["y"])
        row, keep = counts_and_keep(X, y)
        rows.append(dict(dataset=name, **row))
        if row["n_exact_dup_rows"] == 0:
            continue
        Xk, yk = X[keep], y[keep]
        dd[name] = {"X": Xk, "y": yk.astype(np.int64), "orig_rows": np.flatnonzero(keep).astype(np.int64)}
        seed = folds[name]["seed"]
        rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
        fl = [{"repeat": i // N_SPLITS, "fold": i % N_SPLITS, "train_idx": tr.astype(np.int32), "test_idx": te.astype(np.int32)}
              for i, (tr, te) in enumerate(rskf.split(np.zeros((len(yk), 1)), yk))]
        df[name] = {"seed": seed, "n_splits": N_SPLITS, "n_repeats": N_REPEATS, "folds": fl,
                    "splitter": f"RepeatedStratifiedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS}, "
                                f"random_state={seed}) on deduplicated rows"}
    return pd.DataFrame(rows), dd, df


def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(u, v) for u, v in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return a == b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the existing output files")
    a = ap.parse_args()
    counts, dd, df = build(joblib.load(DATA), joblib.load(FOLDS))
    if not a.check:
        os.makedirs(os.path.dirname(OUT_COUNTS), exist_ok=True)
        counts.to_csv(OUT_COUNTS, index=False)
        joblib.dump(dd, OUT_DATA, compress=3)
        joblib.dump(df, OUT_FOLDS, compress=3)
        print(f"{int((counts.n_exact_dup_rows > 0).sum())} datasets with exact duplicates, "
              f"{int(counts.n_exact_dup_rows.sum())} rows removed; wrote {OUT_DATA}, {OUT_FOLDS}, {OUT_COUNTS}")
        return
    old = pd.read_csv(OUT_COUNTS)
    ok_counts = old.equals(counts.astype(old.dtypes.to_dict()))
    ok_data, ok_folds = same(dd, joblib.load(OUT_DATA)), same(df, joblib.load(OUT_FOLDS))
    print(f"counts identical: {ok_counts}; deduplicated data identical: {ok_data}; folds identical: {ok_folds}")
    sys.exit(0 if (ok_counts and ok_data and ok_folds) else 1)


if __name__ == "__main__":
    main()
