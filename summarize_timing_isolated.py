"""Per-sampler summary of the isolated run-time measurements.

Input : results/revision/timing_isolated_benchmark.csv, written by `python pipeline/timing_isolated.py A` (or AB):
        one row per dataset and sampler with the median time `t_median` (seconds) over `reps` repetitions,
        measured in a single process with one thread.
Output: results/revision/timing_isolated_summary.csv, one row per sampler with
        n_datasets               number of datasets;
        median, p95, max         median, 95th percentile (linear interpolation) and maximum of t_median over the datasets;
        slope, lo, hi            least-squares slope of log10(t_median) on log10(n_maj) and its 95% t-interval
                                 (slope -/+ t_{0.975, n-2} * standard error);
        reps_min, reps_median    minimum and median number of repetitions.
        Rows are sorted by the median time.

The summary is read by pipeline/fig5_isolated.py and pipeline/figures_ijoc.py (Figure 3 of the paper) and provides the
run-time columns of Table 2 of the paper and the scaling table of the extended results.

Usage (from the repository root):
    python pipeline/summarize_timing_isolated.py           # write the summary
    python pipeline/summarize_timing_isolated.py --check   # compare with an existing summary instead of writing
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

BENCH = "results/revision/timing_isolated_benchmark.csv"
SUMMARY = "results/revision/timing_isolated_summary.csv"


def summarize(bench: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sampler, g in bench.groupby("sampler", sort=True):
        t = g["t_median"].to_numpy(dtype=float)
        fit = stats.linregress(np.log10(g["n_maj"].to_numpy(dtype=float)), np.log10(t))
        half = stats.t.ppf(0.975, len(g) - 2) * fit.stderr
        rows.append(dict(sampler=sampler, n_datasets=int(len(g)), median=float(np.median(t)),
                         p95=float(np.percentile(t, 95)), max=float(t.max()), slope=float(fit.slope),
                         lo=float(fit.slope - half), hi=float(fit.slope + half),
                         reps_min=int(g["reps"].min()), reps_median=float(g["reps"].median())))
    return pd.DataFrame(rows).sort_values("median", kind="mergesort").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--out", default=SUMMARY)
    ap.add_argument("--check", action="store_true", help="compare with the existing file at --out")
    a = ap.parse_args()
    new = summarize(pd.read_csv(a.bench))
    if not a.check:
        new.to_csv(a.out, index=False)
        print(f"wrote {a.out} ({len(new)} samplers)")
        return
    old = pd.read_csv(a.out)
    assert list(old.columns) == list(new.columns), (list(old.columns), list(new.columns))
    assert old["sampler"].tolist() == new["sampler"].tolist(), "sampler order differs"
    num = [c for c in new.columns if c != "sampler"]
    dev = float(np.nanmax(np.abs(old[num].to_numpy(float) - new[num].to_numpy(float))))
    print(f"maximum absolute difference to {a.out}: {dev:.3g}")
    sys.exit(0 if dev < 1e-9 else 1)


if __name__ == "__main__":
    main()
