"""A3: Bayesian signed-rank tests (baycomp) of each MUS method vs RUS and vs ClusterCentroids, every classifier,
G-mean and ROC-AUC, ROPE in {0.0025, 0.005, 0.01, 0.02}. Settings as stats_pipeline.py: per-dataset means over the
25 folds rounded to 1e-12, prior 0.5, 50,000 posterior samples, np.random.seed(20261115) before every test;
probabilities = baycomp Posterior.probs(True) (share of samples in which each region is the most probable).
Resumable: completed rows are kept in rev/out/A3_bayes_partial.csv.

    python rev_bayes.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_common as rc  # noqa: E402
from baycomp.multiple import SignedRankTest, Posterior  # noqa: E402

MUS = ["MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
CLFS = ["CART", "kNN", "SVM", "LR", "RF"]
ROPES = [0.0025, 0.005, 0.01, 0.02]
SEED, PRIOR, NSAMPLES = 20261115, 0.5, 50000


def verdict(pl, pe, pr, thr=0.95):
    if pl >= thr:
        return "MUS better (P>=0.95)"
    if pr >= thr:
        return "MUS worse (P>=0.95)"
    if pe >= thr:
        return "practically equivalent (P>=0.95)"
    lab = ["MUS better", "equivalent", "MUS worse"][int(np.argmax([pl, pe, pr]))]
    return f"inconclusive (most probable: {lab})"


def main():
    P = rc.load_paths()
    res = pd.read_parquet(P["results_main_final.parquet"], columns=["dataset", "repeat", "fold", "sampler", "classifier",
                                                                     "gmean", "auc"])
    res = res[res["sampler"].isin(MUS + ["RUS", "CC"])]
    res = res.sort_values(["dataset", "sampler", "classifier", "repeat", "fold"], kind="mergesort")
    means = res.groupby(["dataset", "sampler", "classifier"], sort=True)[["gmean", "auc"]].mean().round(12).reset_index()
    datasets = sorted(means["dataset"].unique())
    out_p = os.path.join(HERE, "out", "A3_bayes_partial.csv")
    done = pd.read_csv(out_p) if os.path.exists(out_p) else pd.DataFrame()
    keys = set(map(tuple, done[["metric", "classifier", "mus", "comparator", "rope"]].values.tolist())) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    t0 = time.time()
    for met in ["gmean", "auc"]:
        for clf in CLFS:
            X = means[means["classifier"] == clf].pivot(index="dataset", columns="sampler", values=met).reindex(datasets)
            for m in MUS:
                for comp in ["RUS", "CC"]:
                    for rope in ROPES:
                        if (met, clf, m, comp, rope) in keys:
                            continue
                        x, y = X[m].values.astype(float), X[comp].values.astype(float)
                        np.random.seed(SEED)
                        smp = SignedRankTest.sample(x, y, rope=rope, prior=PRIOR, nsamples=NSAMPLES)
                        pl, pe, pr = Posterior(smp).probs(True)
                        d = x - y
                        rows.append(dict(metric=met, classifier=clf, mus=m, comparator=comp, rope=rope, n_datasets=len(d),
                                         mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                                         n_abs_diff_lt_rope=int((np.abs(d) < rope).sum()),
                                         p_mus_better=float(pl), p_rope=float(pe), p_mus_worse=float(pr),
                                         prior=PRIOR, nsamples=NSAMPLES, seed=SEED, verdict=verdict(pl, pe, pr)))
                        if len(rows) % 10 == 0:
                            pd.DataFrame(rows).to_csv(out_p + ".tmp", index=False)
                            os.replace(out_p + ".tmp", out_p)
                            print(f"{len(rows)} tests, {(time.time() - t0) / 60:.1f} min", flush=True)
    pd.DataFrame(rows).to_csv(out_p + ".tmp", index=False)
    os.replace(out_p + ".tmp", out_p)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
