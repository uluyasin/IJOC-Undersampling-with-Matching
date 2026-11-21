"""Validation of the LPM2 and KSRUS undersamplers on synthetic data.

(1) exact sample size for many (n, m), with and without duplicated points;
(2) empirical inclusion probabilities ~ m/n over many seeds (uniform and clustered populations);
(3) coverage error (CE, mechanisms-pipeline definition) and Voronoi spatial balance (SB) of LPM2
    versus simple random sampling (RUS), KSRUS and MUS-CS on uniform and clustered populations.
Writes lpm_validation.csv (one row per check) and prints a summary.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import chisquare, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extra_samplers as es  # noqa: E402


def population(kind, n, d, rng):
    if kind == "uniform":
        return rng.random((n, d))
    centres = rng.random((5, d))
    lab = rng.integers(0, 5, n)
    return np.clip(centres[lab] + 0.05 * rng.standard_normal((n, d)), 0, 1)


def main(out="lpm_validation.csv"):
    rows = []
    rng = np.random.default_rng(20261120)
    # (1) exact size
    n_ok = n_tot = 0
    for n in (2, 3, 7, 50, 101, 400):
        for m in sorted({1, max(1, n // 7), n // 2, n - 1}):
            for dup in (False, True):
                X = population("uniform", n, 3, rng)
                if dup:
                    X[: n // 2] = X[0]
                for s in range(20):
                    n_tot += 1
                    n_ok += len(es.lpm2_select(X, m, random_state=s)) == m
    rows.append(dict(check="exact_size_LPM2", value=n_ok / n_tot, detail=f"{n_ok}/{n_tot} runs returned exactly m"))
    # (2) inclusion probabilities
    for kind in ("uniform", "clustered"):
        n, m, R = 60, 12, 4000
        X = population(kind, n, 2, rng)
        cnt = np.zeros(n)
        for s in range(R):
            cnt[es.lpm2_select(X, m, random_state=s)] += 1
        emp = cnt / R
        se = np.sqrt(m / n * (1 - m / n) / R)
        rows.append(dict(check=f"inclusion_prob_{kind}", value=float(np.max(np.abs(emp - m / n))),
                         detail=f"n={n}, m={m}, R={R}: target {m / n:.4f}, empirical min {emp.min():.4f} max {emp.max():.4f}, "
                                f"max|dev| {np.max(np.abs(emp - m / n)):.4f} (binomial SE {se:.4f}), "
                                f"chi-square p = {chisquare(cnt, np.full(n, R * m / n)).pvalue:.3f}"))
    # (3) coverage error / spatial balance vs RUS, KSRUS, MUS-CS
    import matchsampling as ms
    for kind in ("uniform", "clustered"):
        for d in (2, 8):
            n, m, R = 400, 40, 100
            res = {k: {"CE": [], "SB": []} for k in ("LPM2", "RUS", "KSRUS", "MUS-CS")}
            for s in range(R):
                X = population(kind, n, d, np.random.default_rng(1000 + s))
                y = np.r_[np.zeros(n, int), np.ones(m, int)]
                Xa = np.vstack([X, np.random.default_rng(5000 + s).random((m, d))])
                sel = {"LPM2": es.lpm2_select(X, m, random_state=s),
                       "RUS": np.sort(np.random.RandomState(s).choice(n, m, replace=False)),
                       "KSRUS": es.ksrus_select(X, m, random_state=s)[0]}
                cs = ms.MatchingUnderSampler(arm="coverage", concept="stable", random_state=s)
                cs.fit_resample(Xa, y)
                sel["MUS-CS"] = cs.sample_indices_[cs.sample_indices_ < n]
                for k, S in sel.items():
                    res[k]["CE"].append(es.coverage_error(X, S))
                    res[k]["SB"].append(es.spatial_balance(X, S))
            for k in ("RUS", "KSRUS", "MUS-CS"):
                for met in ("CE", "SB"):
                    a, b = np.array(res["LPM2"][met]), np.array(res[k][met])
                    rows.append(dict(check=f"{met}_LPM2_vs_{k}_{kind}_d{d}", value=float(np.mean(a < b)),
                                     detail=f"n={n}, m={m}, {R} populations: mean {met} LPM2 {a.mean():.4f} vs {k} {b.mean():.4f}; "
                                            f"LPM2 lower in {np.mean(a < b) * 100:.0f}% (Wilcoxon p = {wilcoxon(a, b).pvalue:.2g})"))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, out), index=False)
    return df


if __name__ == "__main__":
    import bench_runner as br
    br._load_matchsampling(os.path.join(HERE, "..", "src", "matchsampling.py"))
    t0 = time.time()
    df = main()
    pd.set_option("display.max_colwidth", 200)
    print(df.to_string(index=False))
    print(f"{time.time() - t0:.0f} s")
