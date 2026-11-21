"""Referee-revision analyses A1-A9 for the matching-undersampling manuscript.

    python rev_analyses.py            (after rev_heavy.py [w1, rounds, complete] and rev_bayes.py have finished)

Inputs: rev/paths.json (artifact paths), rev/out/{w1,rounds,complete}/*.joblib, rev/out/A3_bayes_partial.csv.
Outputs (rev/deliver/): A1..A9 CSV files, LaTeX tables (booktabs) for A1, A2, A3, A6, A7/A8, and
analyses_numbers.json (every number quoted in analyses_report.md).
Conventions copied from stats_pipeline.py / mechanisms_pipeline.py: per-dataset means over the 25 folds rounded to
1e-12; Wilcoxon signed-rank two-sided, zero_method='zsplit', method='auto'; W/T/L tie |diff| < 0.001; Holm.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import glob
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DEL = os.path.join(HERE, "deliver")
MUS = ["MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
CLFS = ["CART", "kNN", "SVM", "LR", "RF"]
COMPS = ["RUS", "CC"]
ALLM = ["gmean", "auc", "bacc", "f1", "mcc", "prauc", "sens", "spec", "prec"]
MLAB = {"gmean": "G-mean", "auc": "ROC-AUC", "f1": "F1", "mcc": "MCC"}
TIE_EPS = 1e-3
B_BOOT, SEED_BOOT = 5000, 20261120
W1_METHODS = ["RUS", "NM1", "NM2", "NM3", "CC", "MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
COVARIATES = ["IR", "n", "d", "n_min", "F1", "N1", "N2", "N3", "LSC", "safe", "borderline", "rare", "outlier",
              "maj_overlap_share"]
NUM = {}   # every number quoted in the report


def P(name):
    return os.path.join(DEL, name)


def holm(p):
    rej, padj, _, _ = multipletests(np.asarray(p, float), alpha=0.05, method="holm")
    return padj, rej


def wilcoxon_pair(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    wins, losses = int(np.sum(d >= TIE_EPS)), int(np.sum(d <= -TIE_EPS))
    ties = n - wins - losses
    r = stats.rankdata(np.abs(d))
    rp = r[d > 0].sum() + 0.5 * r[d == 0].sum()
    rm = r[d < 0].sum() + 0.5 * r[d == 0].sum()
    r_rb = float((rp - rm) / (n * (n + 1) / 2.0))
    if np.all(d == 0):
        return dict(n=n, mean_diff=0.0, median_diff=0.0, wins=wins, ties=ties, losses=losses, W=np.nan, p_raw=1.0,
                    p_method="all zero", r_rb=r_rb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rr = stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided", method="auto")
        p_ex = float(stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided",
                                    method="exact").pvalue) if n <= 50 else np.nan
    meth = "exact" if (n <= 50 and np.isclose(rr.pvalue, p_ex, rtol=0, atol=1e-15)) else "asymptotic"
    return dict(n=n, mean_diff=float(d.mean()), median_diff=float(np.median(d)), wins=wins, ties=ties, losses=losses,
                W=float(rr.statistic), p_raw=float(rr.pvalue), p_method=meth, r_rb=r_rb)


def verdict(sig, r_rb):
    return "n.s." if not sig else ("MUS better" if r_rb > 0 else "MUS worse")


def q(v, p):
    v = pd.Series(v).dropna()
    return float(v.quantile(p)) if len(v) else np.nan


def fnum(x, nd=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def fp(p):
    return "n/a" if p is None or not np.isfinite(p) else (f"{p:.2e}" if p < 0.001 else f"{p:.3f}")


def tex_num(x, nd=4, sign=True):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def tex_p(p):
    return "--" if not np.isfinite(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


# ============================================================================================ loading
def load():
    paths = json.load(open(os.path.join(HERE, "paths.json")))
    res = pd.read_parquet(paths["results_main_final.parquet"])
    data = joblib.load(paths["benchmark_datasets.joblib"])
    folds = joblib.load(paths["cv_folds.joblib"])
    reg = pd.read_csv(paths["registry"])
    reg = reg[reg["included"].astype(bool)]
    cpx = pd.read_csv(paths["complexity"])
    datasets = sorted(data)
    r_ = res.sort_values(["dataset", "sampler", "classifier", "repeat", "fold"], kind="mergesort")
    means = r_.groupby(["dataset", "sampler", "classifier"], sort=True)[ALLM].mean().round(12).reset_index()
    spdm = pd.read_csv(paths["stats_per_dataset_means.csv"])
    chk = means.merge(spdm[["dataset", "sampler", "classifier"] + ALLM], on=["dataset", "sampler", "classifier"],
                      suffixes=("", "_f"))
    NUM["per_dataset_means_max_abs_diff_vs_stats_file"] = max(float((chk[m] - chk[m + "_f"]).abs().max()) for m in ALLM)
    meta = reg.set_index("name").loc[datasets, ["family", "IR", "family_representative"]].copy()
    meta["safe"] = cpx.set_index("name").loc[datasets, "safe"]
    return dict(paths=paths, res=res, data=data, folds=folds, reg=reg, cpx=cpx, datasets=datasets, means=means,
                meta=meta)


def mat(C, metric, clf, methods):
    m = C["means"]
    s = m[(m.classifier == clf) & m.sampler.isin(methods)].pivot(index="dataset", columns="sampler", values=metric)
    return s.reindex(index=C["datasets"], columns=methods)


# ============================================================================================ A1
def a1(C):
    datasets, fam = C["datasets"], C["meta"]["family"]
    fam_list = sorted(fam.unique())
    F, nD = len(fam_list), len(datasets)
    fam_idx = np.array([fam_list.index(fam[d]) for d in datasets])
    famsize = np.bincount(fam_idx, minlength=F).astype(float)
    rng = np.random.default_rng(SEED_BOOT)
    fam_draws = rng.integers(0, F, size=(B_BOOT, F))
    Cf = np.zeros((B_BOOT, F))
    np.add.at(Cf, (np.repeat(np.arange(B_BOOT), F), fam_draws.ravel()), 1)
    ds_draws = rng.integers(0, nD, size=(B_BOOT, nD))
    Cd = np.zeros((B_BOOT, nD))
    np.add.at(Cd, (np.repeat(np.arange(B_BOOT), nD), ds_draws.ravel()), 1)
    den_f = Cf @ famsize
    rows = []
    for metric in ["gmean", "auc"]:
        for clf in CLFS:
            X = mat(C, metric, clf, MUS + COMPS)
            for mth in MUS:
                for comp in COMPS:
                    d = (X[mth] - X[comp]).values
                    famsum = np.bincount(fam_idx, weights=d, minlength=F)
                    bf = (Cf @ famsum) / den_f
                    bd = (Cd @ d) / nD
                    lo_f, hi_f = np.percentile(bf, [2.5, 97.5])
                    lo_d, hi_d = np.percentile(bd, [2.5, 97.5])
                    rows.append(dict(metric=metric, classifier=clf, mus=mth, comparator=comp, n_datasets=nD,
                                     n_families=F, mean_diff=float(d.mean()),
                                     mean_of_family_means=float((famsum / famsize).mean()),
                                     ds_boot_se=float(bd.std(ddof=1)), ds_ci_lo=float(lo_d), ds_ci_hi=float(hi_d),
                                     fam_boot_se=float(bf.std(ddof=1)), fam_ci_lo=float(lo_f), fam_ci_hi=float(hi_f),
                                     se_ratio_fam_over_ds=float(bf.std(ddof=1) / bd.std(ddof=1)),
                                     ds_ci_excludes_0=bool(lo_d > 0 or hi_d < 0),
                                     fam_ci_excludes_0=bool(lo_f > 0 or hi_f < 0)))
    A = pd.DataFrame(rows)
    A["conclusion_changes"] = A.ds_ci_excludes_0 != A.fam_ci_excludes_0
    A["B"], A["seed"] = B_BOOT, SEED_BOOT
    A["ci_method"] = "percentile 95%; family bootstrap: 24 families resampled with replacement, statistic = pooled mean over the datasets of the drawn families"
    A.to_csv(P("A1_family_bootstrap.csv"), index=False)
    NUM["A1"] = dict(
        n_tests=len(A), n_changes=int(A.conclusion_changes.sum()), n_se_ratio_gt1=int((A.se_ratio_fam_over_ds > 1).sum()),
        se_ratio_median=float(A.se_ratio_fam_over_ds.median()), se_ratio_min=float(A.se_ratio_fam_over_ds.min()),
        se_ratio_max=float(A.se_ratio_fam_over_ds.max()),
        by=A.groupby(["comparator", "metric"]).agg(ds_excl=("ds_ci_excludes_0", "sum"),
                                                     fam_excl=("fam_ci_excludes_0", "sum"),
                                                     changes=("conclusion_changes", "sum"),
                                                     se_ratio_med=("se_ratio_fam_over_ds", "median")).reset_index()
        .to_dict("records"),
        changes=A[A.conclusion_changes][["metric", "classifier", "mus", "comparator", "mean_diff", "ds_ci_lo", "ds_ci_hi",
                                         "fam_ci_lo", "fam_ci_hi", "se_ratio_fam_over_ds"]].to_dict("records"),
        boundary_gmean=A[(A.metric == "gmean") & A.mus.isin(["MUS-BS", "MUS-BO"])][
            ["classifier", "mus", "comparator", "mean_diff", "fam_ci_lo", "fam_ci_hi"]].to_dict("records"),
        coverage_max_abs_mean=float(A[A.mus.isin(["MUS-CS", "MUS-CO"])].mean_diff.abs().max()),
        coverage_max_abs_ci=float(A[A.mus.isin(["MUS-CS", "MUS-CO"])][["fam_ci_lo", "fam_ci_hi"]].abs().max().max()),
        coverage_fam_excl=int(A[A.mus.isin(["MUS-CS", "MUS-CO"])].fam_ci_excludes_0.sum()),
        coverage_fam_excl_rows=A[A.mus.isin(["MUS-CS", "MUS-CO"]) & A.fam_ci_excludes_0][
            ["metric", "classifier", "mus", "comparator", "mean_diff", "fam_ci_lo", "fam_ci_hi"]].to_dict("records"))
    # LaTeX
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Family dependence (A1): mean paired difference (MUS $-$ comparator) over the 55 datasets with the "
         r"95\% percentile interval of the family-clustered bootstrap (24 base-data families resampled with "
         r"replacement, $B=5000$, seed 20261120). $^{\dagger}$: the dataset-level bootstrap interval leads to the "
         r"opposite conclusion about zero. Median ratio of family to dataset bootstrap standard errors: "
         + f"{A.se_ratio_fam_over_ds.median():.2f} (range {A.se_ratio_fam_over_ds.min():.2f}--{A.se_ratio_fam_over_ds.max():.2f})" + r".}",
         r"\label{tab:a1_family_bootstrap}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r" & & \multicolumn{2}{c}{G-mean} & \multicolumn{2}{c}{ROC-AUC} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Method & Clf. & vs RUS & vs CC & vs RUS & vs CC \\", r"\midrule"]
    for mi, mth in enumerate(MUS):
        if mi:
            L.append(r"\midrule")
        for ci, clf in enumerate(CLFS):
            cells = [mth if ci == 0 else "", clf]
            for metric in ["gmean", "auc"]:
                for comp in COMPS:
                    r = A[(A.metric == metric) & (A.classifier == clf) & (A.mus == mth) & (A.comparator == comp)].iloc[0]
                    s = f"{tex_num(r.mean_diff, 3)} [{tex_num(r.fam_ci_lo, 3)}, {tex_num(r.fam_ci_hi, 3)}]"
                    if r.conclusion_changes:
                        s += r"$^{\dagger}$"
                    cells.append(s)
            L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(P("A1_family_bootstrap.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    return A


# ============================================================================================ A2
def a2(C):
    res, data, folds, datasets = C["res"], C["data"], C["folds"], C["datasets"]
    cnt_rows = []
    for ds in datasets:
        y = np.asarray(data[ds]["y"]).astype(int)
        for f in folds[ds]["folds"]:
            te = np.asarray(f["test_idx"])
            cnt_rows.append(dict(dataset=ds, repeat=int(f["repeat"]), fold=int(f["fold"]), P_te=int((y[te] == 1).sum()),
                                 N_te=int((y[te] == 0).sum())))
    cnts = pd.DataFrame(cnt_rows)
    sub = res[res.sampler.isin(MUS + COMPS)][["dataset", "repeat", "fold", "sampler", "classifier", "sens", "spec", "prec",
                                              "gmean"]].merge(cnts, on=["dataset", "repeat", "fold"])
    tp_f, tn_f = sub.sens * sub.P_te, sub.spec * sub.N_te
    sub["TP"], sub["TN"] = np.rint(tp_f).astype(int), np.rint(tn_f).astype(int)
    sub["FN"], sub["FP"] = sub.P_te - sub.TP, sub.N_te - sub.TN
    prec_rec = np.where(sub.TP + sub.FP > 0, sub.TP / (sub.TP + sub.FP).clip(lower=1), 0.0)
    gm_rec = np.sqrt((sub.TP / sub.P_te) * (sub.TN / sub.N_te))
    NUM["A2_checks"] = dict(fold_rows=len(sub), max_abs_TP_nonint=float((tp_f - sub.TP).abs().max()),
                            max_abs_TN_nonint=float((tn_f - sub.TN).abs().max()),
                            max_abs_prec_mismatch=float(np.abs(prec_rec - sub.prec).max()),
                            max_abs_gmean_mismatch=float(np.abs(gm_rec - sub.gmean).max()))
    pool = sub.groupby(["dataset", "sampler", "classifier", "repeat"])[["TP", "FN", "TN", "FP"]].sum().reset_index()
    TP, FN, TN, FP = [pool[c].astype(float) for c in ["TP", "FN", "TN", "FP"]]
    pool["gmean"] = np.sqrt((TP / (TP + FN)) * (TN / (TN + FP)))
    pool["f1"] = np.where(2 * TP + FP + FN > 0, 2 * TP / (2 * TP + FP + FN), 0.0)
    den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    pool["mcc"] = np.where(den > 0, (TP * TN - FP * FN) / np.where(den > 0, den, 1), 0.0)
    pm = pool.groupby(["dataset", "sampler", "classifier"])[["gmean", "f1", "mcc"]].mean().round(12).reset_index()
    rows = []
    for metric in ["gmean", "f1", "mcc"]:
        for clf in CLFS:
            Xp = pm[pm.classifier == clf].pivot(index="dataset", columns="sampler", values=metric).reindex(index=datasets)
            Xf = mat(C, metric, clf, MUS + COMPS)
            for scheme, X in [("pooled", Xp), ("fold_avg", Xf)]:
                block = []
                for mth in MUS:
                    for comp in COMPS:
                        rr = wilcoxon_pair(X[mth].values, X[comp].values)
                        rr.update(metric=metric, classifier=clf, scheme=scheme, mus=mth, comparator=comp)
                        block.append(rr)
                padj, rej = holm([b["p_raw"] for b in block])
                for b, pa, rj in zip(block, padj, rej):
                    b.update(p_holm8=float(pa), sig_holm8=bool(rj), verdict8=verdict(rj, b["r_rb"]))
                rows += block
    L_ = pd.DataFrame(rows)
    L_["WTL"] = L_.wins.astype(str) + "/" + L_.ties.astype(str) + "/" + L_.losses.astype(str)
    w = L_.pivot_table(index=["metric", "classifier", "mus", "comparator"], columns="scheme",
                       values=["mean_diff", "median_diff", "WTL", "p_raw", "p_holm8", "verdict8"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    A = w.reset_index()
    A["verdict_changed"] = A.verdict8_pooled != A.verdict8_fold_avg
    sw = pd.read_csv(C["paths"]["stats_wilcoxon_holm.csv"])
    orig = sw[(sw.condition == "all") & sw.baseline.isin(COMPS)].set_index(["metric", "classifier", "mus", "baseline"])["verdict"]
    A["verdict_original_48family_fold_avg"] = [orig.get((r.metric, r.classifier, r.mus, r.comparator), np.nan)
                                               for r in A.itertuples()]
    A["holm_family"] = "8 comparisons (4 MUS x {RUS, CC}) per classifier x metric"
    order = {"gmean": 0, "f1": 1, "mcc": 2}
    A = A.sort_values(["metric", "comparator", "mus", "classifier"], key=lambda c: c.map(order) if c.name == "metric" else c)
    A.to_csv(P("A2_pooled_confusion.csv"), index=False)
    cnt = A.groupby(["metric", "mus", "comparator"]).agg(
        pooled_worse=("verdict8_pooled", lambda s: int((s == "MUS worse").sum())),
        pooled_better=("verdict8_pooled", lambda s: int((s == "MUS better").sum())),
        fold_worse=("verdict8_fold_avg", lambda s: int((s == "MUS worse").sum())),
        fold_better=("verdict8_fold_avg", lambda s: int((s == "MUS better").sum()))).reset_index()
    o48 = A[A.verdict_original_48family_fold_avg.notna() & (A.verdict_original_48family_fold_avg != A.verdict8_fold_avg)]
    NUM["A2"] = dict(n_tests_per_scheme=int(len(A)), n_changed=int(A.verdict_changed.sum()), counts=cnt.to_dict("records"),
                     diff_vs_original48=o48[["metric", "classifier", "mus", "comparator",
                                             "verdict_original_48family_fold_avg", "verdict8_fold_avg"]].to_dict("records"),
                     gmean_pooled_median_diff=A[A.metric == "gmean"][["classifier", "mus", "comparator",
                                                                      "median_diff_pooled", "median_diff_fold_avg"]]
                     .to_dict("records"))
    sym = {"MUS worse": r"$-$", "MUS better": r"$+$", "n.s.": r"$\cdot$"}
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Pooled confusion matrices (A2): verdicts of the two-sided Wilcoxon signed-rank test (zero-split) on "
         r"per-dataset means of metrics computed from confusion matrices pooled over the 5 folds of each repeat and "
         r"averaged over the 5 repeats; Holm over the 8 comparisons (4 MUS methods $\times$ \{RUS, CC\}) per "
         r"classifier and metric. $-$/$+$: MUS significantly worse/better ($p_\mathrm{Holm}<0.05$); $\cdot$: n.s. "
         + f"The same verdicts are obtained from fold-averaged metrics in {int((~A.verdict_changed).sum())} of {len(A)} tests." + r"}",
         r"\label{tab:a2_pooled}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{ll" + "c" * 15 + "}", r"\toprule",
         r" & & \multicolumn{5}{c}{G-mean} & \multicolumn{5}{c}{F1} & \multicolumn{5}{c}{MCC} \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}",
         "Method & vs & " + " & ".join(CLFS * 3) + r" \\", r"\midrule"]
    for comp in COMPS:
        for mth in MUS:
            cells = [mth, comp]
            for metric in ["gmean", "f1", "mcc"]:
                for clf in CLFS:
                    r = A[(A.metric == metric) & (A.classifier == clf) & (A.mus == mth) & (A.comparator == comp)].iloc[0]
                    cells.append(sym[r.verdict8_pooled])
            L.append(" & ".join(cells) + r" \\")
        if comp == "RUS":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(P("A2_pooled_confusion.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    return A


# ============================================================================================ A3
def a3(C):
    A = pd.read_csv(os.path.join(OUT, "A3_bayes_partial.csv"))
    assert len(A) == 2 * 5 * 4 * 2 * 4, len(A)
    sb = pd.read_csv(C["paths"]["stats_bayes.csv"])
    ref = sb[sb.comparator_role == "RUS"].set_index(["metric", "classifier", "mus"])
    x = A[(A.comparator == "RUS") & (A.rope == 0.01)]
    diffs = [max(abs(r.p_mus_better - ref.loc[(r.metric, r.classifier, r.mus), "p_mus_better"]),
                 abs(r.p_rope - ref.loc[(r.metric, r.classifier, r.mus), "p_rope"]),
                 abs(r.p_mus_worse - ref.loc[(r.metric, r.classifier, r.mus), "p_mus_worse"])) for r in x.itertuples()]
    A["category"] = np.select([A.p_mus_better >= 0.95, A.p_mus_worse >= 0.95, A.p_rope >= 0.95],
                              ["better", "worse", "equivalent"], "inconclusive")
    A.to_csv(P("A3_bayes_rope.csv"), index=False)
    cat = A.groupby(["metric", "comparator", "rope", "mus"]).category.value_counts().unstack(fill_value=0).reset_index()
    for c in ["better", "worse", "equivalent", "inconclusive"]:
        if c not in cat.columns:
            cat[c] = 0
    cov = A[A.mus.isin(["MUS-CS", "MUS-CO"])]
    bnd = A[A.mus.isin(["MUS-BS", "MUS-BO"])]
    NUM["A3"] = dict(max_abs_diff_vs_stats_bayes_rope001_RUS=float(max(diffs)), n_reproduced=len(diffs),
                     counts=cat.to_dict("records"),
                     better_rows=A[A.category == "better"][["metric", "comparator", "rope", "mus", "classifier", "mean_diff",
                                                           "p_mus_better", "p_rope", "p_mus_worse"]].to_dict("records"),
                     coverage_prope=cov.groupby(["metric", "comparator", "rope"]).p_rope.agg(["min", "median", "max"])
                     .reset_index().to_dict("records"),
                     coverage_equiv=cov.assign(e=cov.category == "equivalent").groupby(["metric", "comparator", "rope"]).e.sum()
                     .reset_index().to_dict("records"),
                     boundary_worse=bnd.assign(w=bnd.category == "worse").groupby(["metric", "comparator", "rope"]).w.sum()
                     .reset_index().to_dict("records"),
                     boundary_pworse_min=bnd.groupby(["metric", "comparator", "rope"]).p_mus_worse.min().reset_index()
                     .to_dict("records"),
                     boundary_auc_not_worse=bnd[(bnd.metric == "auc") & (bnd.category != "worse")][
                         ["comparator", "rope", "mus", "classifier", "mean_diff", "p_mus_worse", "p_rope"]].to_dict("records"))
    code = {"better": "B", "worse": "W", "equivalent": "E", "inconclusive": "i"}
    ropes = sorted(A.rope.unique())
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Bayesian signed-rank tests (A3; baycomp, prior 0.5, 50{,}000 posterior samples, seed 20261115) of "
         r"each MUS method against RUS and ClusterCentroids (CC) for ROPE half-widths "
         + ", ".join(f"{r:g}" for r in ropes) + r" (left to right within each classifier). "
         r"W/B/E: posterior probability $\geq 0.95$ that the MUS method is worse / better by more than the ROPE / "
         r"practically equivalent; i: inconclusive. Probabilities in \texttt{A3\_bayes\_rope.csv}.}",
         r"\label{tab:a3_bayes_rope}", r"\scriptsize", r"\setlength{\tabcolsep}{1.6pt}",
         r"\begin{tabular}{lll" + "c" * (5 * len(ropes)) + "}", r"\toprule",
         r" & & & " + " & ".join(f"\\multicolumn{{{len(ropes)}}}{{c}}{{{c}}}" for c in CLFS) + r" \\",
         "".join(f"\\cmidrule(lr){{{4 + i * len(ropes)}-{3 + (i + 1) * len(ropes)}}}" for i in range(5)),
         r"Metric & Method & vs & " + " & ".join([f"{r:g}".replace("0.", ".") for r in ropes] * 5) + r" \\", r"\midrule"]
    for mi, metric in enumerate(["gmean", "auc"]):
        if mi:
            L.append(r"\midrule")
        first = True
        for comp in COMPS:
            for mth in MUS:
                cells = [MLAB[metric] if first else "", mth, comp]
                first = False
                for clf in CLFS:
                    for rp in ropes:
                        r = A[(A.metric == metric) & (A.classifier == clf) & (A.mus == mth) & (A.comparator == comp)
                              & (A.rope == rp)].iloc[0]
                        cells.append(code[r.category])
                L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(P("A3_bayes_rope.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    return A


# ============================================================================================ A4
def a4(C):
    sw = pd.read_csv(C["paths"]["stats_wilcoxon_holm.csv"])
    c = sw[(sw.condition == "all") & (sw.baseline == "RUS") & sw.metric.isin(["gmean", "auc"])].copy()
    c["p_raw_recomputed"] = [wilcoxon_pair(mat(C, r.metric, r.classifier, [r.mus, "RUS"])[r.mus].values,
                                           mat(C, r.metric, r.classifier, [r.mus, "RUS"])["RUS"].values)["p_raw"]
                             for r in c.itertuples()]
    padj, rej = holm(c.p_raw.values)
    c["p_holm40"], c["sig_holm40"] = padj, rej
    c["verdict40"] = [verdict(s, r) for s, r in zip(c.sig_holm40, c.r_rb)]
    A = c[["metric", "classifier", "mus", "baseline", "n", "mean_diff", "median_diff", "wins", "ties", "losses", "W",
           "r_rb", "p_raw", "p_raw_recomputed", "p_method", "p_holm", "verdict", "p_holm40", "sig_holm40", "verdict40"]] \
        .rename(columns={"p_holm": "p_holm48_original", "verdict": "verdict48_original"})
    A["verdict_changed"] = A.verdict48_original != A.verdict40
    A = A.sort_values(["metric", "mus", "classifier"], key=lambda s: s.map({"gmean": 0, "auc": 1}) if s.name == "metric" else s)
    A.to_csv(P("A4_confirmatory_holm40.csv"), index=False)
    NUM["A4"] = dict(n=len(A), max_abs_p_diff=float((A.p_raw - A.p_raw_recomputed).abs().max()),
                     n_sig=int(A.sig_holm40.sum()), n_sig_orig=int((A.verdict48_original != "n.s.").sum()),
                     sig=A[A.sig_holm40][["metric", "mus", "classifier", "median_diff", "p_raw", "p_holm40", "verdict40"]]
                     .to_dict("records"),
                     changed=A[A.verdict_changed][["metric", "mus", "classifier", "median_diff", "p_raw",
                                                   "p_holm48_original", "p_holm40", "verdict48_original", "verdict40"]]
                     .to_dict("records"),
                     ns=A[~A.sig_holm40][["metric", "mus", "classifier", "median_diff", "p_raw", "p_holm40"]].to_dict("records"))
    return A


# ============================================================================================ A5
def a5(C):
    dd = pd.read_csv(C["paths"]["diag_by_dataset.csv"])
    res = C["res"]
    perf = res.groupby(["dataset", "sampler", "classifier"])["gmean"].mean().reset_index()
    gavg = perf.groupby(["dataset", "sampler"])["gmean"].mean()
    link = pd.read_csv(C["paths"]["diag_performance_link.csv"])
    pairs = [("gmean", "BP"), ("gmean", "CE"), ("BP", "CE")]
    sets = {"A5_six_methods": ["RUS", "NM1", "NM2", "CC", "MUS-BS", "MUS-CS"],
            "reference_size_balanced9": ["RUS", "NM1", "NM2", "NM3", "CC", "MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]}
    rows = []
    for sname, smps in sets.items():
        per = {pr: [] for pr in pairs}
        for ds in C["datasets"]:
            x = dd[(dd.dataset == ds) & dd.sampler.isin(smps)].set_index("sampler")
            j = pd.concat([x["BP"], x["CE"], gavg.loc[ds].reindex(x.index).rename("gmean")], axis=1).dropna()
            for a, b in pairs:
                rho = np.nan
                if len(j) >= 5 and j[a].nunique() > 1 and j[b].nunique() > 1:
                    rho = float(stats.spearmanr(j[a], j[b]).statistic)
                per[(a, b)].append(dict(dataset=ds, rho=rho))
        block = []
        for pr in pairs:
            dfp = pd.DataFrame(per[pr])
            v = dfp.rho.dropna()
            pw = float(stats.wilcoxon(v).pvalue) if len(v) >= 6 and (v != 0).any() else np.nan
            block.append(dict(set=sname, methods=";".join(smps), n_methods=len(smps), correlation=f"{pr[0]} vs {pr[1]}",
                              n_datasets=len(v), datasets_undefined=";".join(dfp[dfp.rho.isna()].dataset),
                              rho_median=float(v.median()), rho_q1=q(v, .25), rho_q3=q(v, .75),
                              frac_pos=float((v > 0).mean()), frac_neg=float((v < 0).mean()), wilcoxon_p=pw))
        adj = holm([b["wilcoxon_p"] for b in block])[0]
        for b, a_ in zip(block, adj):
            b["wilcoxon_p_holm3"] = float(a_)
        rows += block
    A = pd.DataFrame(rows)
    ref = link[(link.set == "size_balanced") & (link.classifier == "mean")].set_index("diagnostic")
    A["original_link_rho_median"] = [float(ref.loc[c.split(" vs ")[1], "rho_median"])
                                     if s.startswith("reference") and c.startswith("gmean") else np.nan
                                     for s, c in zip(A.set, A.correlation)]
    A["test"] = "one-sample two-sided Wilcoxon signed-rank (scipy defaults) of per-dataset Spearman rho vs 0; Holm over the 3 correlations"
    A.to_csv(P("A5_mechanism_correlations.csv"), index=False)
    NUM["A5"] = A.drop(columns=["methods", "test"]).to_dict("records")
    return A


# ============================================================================================ A6
def a6(C):
    res, meta = C["res"], C["meta"]
    g = res[res.sampler.isin(MUS + ["RUS"])].pivot_table(index=["dataset", "repeat", "fold", "classifier"],
                                                         columns="sampler", values="gmean")
    J, RATIO = 25, 0.25
    rows = []
    for mth in MUS:
        dk = (g[mth] - g["RUS"]).groupby(level=["dataset", "repeat", "fold"]).mean()
        for ds, x in dk.groupby(level="dataset"):
            v = x.values
            assert len(v) == J
            s2 = v.var(ddof=1)
            se = np.sqrt((1 / J + RATIO) * s2)
            diff = v.mean()
            t = diff / se if se > 0 else np.nan
            flag = bool(abs(diff) > 2 * se)
            rows.append(dict(dataset=ds, family=meta.loc[ds, "family"], IR=float(meta.loc[ds, "IR"]),
                             safe_share=float(meta.loc[ds, "safe"]), mus=mth, diff_gmean_vs_RUS=float(diff),
                             fold_sd=float(np.sqrt(s2)), se_nadeau_bengio=float(se), naive_se=float(np.sqrt(s2 / J)),
                             t_corrected=float(t), p_corrected_t24=float(2 * stats.t.sf(abs(t), J - 1)) if se > 0 else np.nan,
                             flag_abs_diff_gt_2se=flag,
                             flag_direction=("MUS better" if diff > 0 else "MUS worse") if flag else ""))
    A = pd.DataFrame(rows)
    A.to_csv(P("A6_per_dataset_nadeau_bengio.csv"), index=False)
    s = A.groupby("mus").agg(flagged=("flag_abs_diff_gt_2se", "sum"),
                             better=("flag_direction", lambda x: int((x == "MUS better").sum())),
                             worse=("flag_direction", lambda x: int((x == "MUS worse").sum())),
                             median_diff=("diff_gmean_vs_RUS", "median"), median_se=("se_nadeau_bengio", "median"))
    NUM["A6"] = dict(summary=s.reset_index().to_dict("records"),
                     coverage_flagged=A[A.flag_abs_diff_gt_2se & A.mus.isin(["MUS-CS", "MUS-CO"])][
                         ["dataset", "mus", "diff_gmean_vs_RUS", "se_nadeau_bengio", "safe_share"]].to_dict("records"),
                     boundary_better=A[A.flag_abs_diff_gt_2se & (A.flag_direction == "MUS better")][
                         ["dataset", "mus", "diff_gmean_vs_RUS", "se_nadeau_bengio", "safe_share"]].to_dict("records"),
                     boundary_unflagged=A[~A.flag_abs_diff_gt_2se & A.mus.isin(["MUS-BS", "MUS-BO"])].groupby("mus")
                     .safe_share.median().to_dict(),
                     boundary_flagged_safe=A[A.flag_abs_diff_gt_2se & A.mus.isin(["MUS-BS", "MUS-BO"])].groupby("mus")
                     .safe_share.median().to_dict(),
                     null_flag_rate=float(2 * stats.t.sf(2.0, 24)))
    W = A.pivot(index="dataset", columns="mus", values=["diff_gmean_vs_RUS", "se_nadeau_bengio", "flag_abs_diff_gt_2se"])
    L = [r"\begin{table*}[!htbp]", r"\centering",
         r"\caption{Per-dataset classifier-averaged G-mean difference to RUS (A6; mean over the 25 folds of the "
         r"5-classifier average) with the corrected resampled-$t$ standard error of Nadeau and Bengio (2003), "
         r"$\mathrm{SE}=\sqrt{(1/25+1/4)\,s^2}$ from the 25 fold-level differences (test/train ratio 1/4). "
         r"$^{*}$: $|\Delta|>2\,\mathrm{SE}$. IR = imbalance ratio; safe = safe-minority share.}",
         r"\label{tab:a6_per_dataset}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lllrr" + "r" * len(MUS) + "}", r"\toprule",
         r"Dataset & Family & & IR & Safe & " + " & ".join(MUS) + r" \\", r"\midrule"]
    for ds in C["datasets"]:
        cells = [ds.replace("_", r"\_"), str(meta.loc[ds, "family"]).replace("_", r"\_"), "",
                 f"{meta.loc[ds, 'IR']:.1f}", f"{meta.loc[ds, 'safe']:.2f}"]
        for mth in MUS:
            d_ = W.loc[ds, ("diff_gmean_vs_RUS", mth)]
            se_ = W.loc[ds, ("se_nadeau_bengio", mth)]
            st = r"$^{*}$" if W.loc[ds, ("flag_abs_diff_gt_2se", mth)] else ""
            cells.append(f"{tex_num(d_, 3)} ({se_:.3f}){st}")
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(P("A6_per_dataset_nadeau_bengio.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    return A


# ============================================================================================ A7
def a7(C):
    rows, checks, errs = [], [], []
    for f in sorted(glob.glob(os.path.join(OUT, "w1", "*.joblib"))):
        o = joblib.load(f)
        if "error" in o:
            errs.append(o)
            continue
        rows += o["rows"]
        checks.append(o["check"])
    if errs:
        raise RuntimeError(f"{len(errs)} W1 units failed: {errs[:2]}")
    W = pd.DataFrame(rows)
    K = pd.DataFrame(checks)
    assert len(K) == 275 and len(W) == 275 * 9, (len(K), len(W))
    S = pd.DataFrame([joblib.load(f)["summary"] for f in sorted(glob.glob(os.path.join(OUT, "rounds", "*.joblib")))])
    S = S[S.repeat == 0]
    S["bound_included"] = (S.concept == "stable") | S.a7
    Sm = S.assign(sampler=S.concept.map({"stable": "MUS-CS", "optimal": "MUS-CO"}))
    Sm = Sm[Sm.bound_included][["dataset", "repeat", "fold", "sampler", "n_rounds", "reproduces_record", "seconds_total",
                                "t_main", "bound_sum", "bound_term1_sum", "bound_term2_sum"]]
    F = W.merge(Sm, on=["dataset", "repeat", "fold", "sampler"], how="left")
    F["bound_over_W1"] = F.bound_sum / F.W1
    for mth, col in [("MUS-BO", "W1_BO"), ("MUS-BS", "W1_BS")]:
        kk = K.set_index(["dataset", "repeat", "fold"])
        idx = F.sampler == mth
        keys = list(zip(F.loc[idx, "dataset"], F.loc[idx, "repeat"], F.loc[idx, "fold"]))
        F.loc[idx, "W1_capacity_minority_vs_S"] = [float(kk.loc[k_, col]) for k_ in keys]
        F.loc[idx, "W1_star_best_subset"] = [float(kk.loc[k_, "W1_star"]) for k_ in keys]
    F["rel_diff_to_W1_star"] = (F.W1_capacity_minority_vs_S - F.W1_star_best_subset) / F.W1_star_best_subset
    F.to_csv(P("A7_wasserstein_folds.csv"), index=False)
    # dataset level
    D = W.groupby(["dataset", "sampler"]).agg(W1_mean=("W1", "mean"), n_maj=("n_maj", "mean"), m=("m", "mean")).reset_index()
    rus = D[D.sampler == "RUS"].set_index("dataset").W1_mean
    D["W1_ratio_to_RUS"] = D.W1_mean / D.dataset.map(rus)
    Sb = F[F.bound_sum.notna()].groupby(["dataset", "sampler"]).agg(bound_mean=("bound_sum", "mean"),
                                                                    bound_folds=("bound_sum", "size"),
                                                                    bound_over_W1_mean=("bound_over_W1", "mean")).reset_index()
    D = D.merge(Sb, on=["dataset", "sampler"], how="left")
    D["family"] = D.dataset.map(C["meta"]["family"])
    D.to_csv(P("A7_wasserstein_by_dataset.csv"), index=False)
    summ = []
    for mth in W1_METHODS:
        x = D[D.sampler == mth]
        summ.append(dict(sampler=mth, n_datasets=len(x), W1_median=float(x.W1_mean.median()), W1_q1=q(x.W1_mean, .25),
                         W1_q3=q(x.W1_mean, .75), ratio_median=float(x.W1_ratio_to_RUS.median()),
                         ratio_q1=q(x.W1_ratio_to_RUS, .25), ratio_q3=q(x.W1_ratio_to_RUS, .75),
                         n_ratio_lt_1=int((x.W1_ratio_to_RUS < 1).sum()),
                         fold_W1_median=float(W[W.sampler == mth].W1.median())))
    SU = pd.DataFrame(summ)
    bnd = []
    for mth in ["MUS-CS", "MUS-CO"]:
        x = F[(F.sampler == mth) & F.bound_sum.notna()]
        share2 = x.bound_term2_sum / x.bound_sum
        bnd.append(dict(sampler=mth, n_folds=len(x), n_datasets=x.dataset.nunique(),
                        all_reproduce=bool(x.reproduces_record.all()), bound_ge_W1_all=bool((x.bound_sum >= x.W1 - 1e-12).all()),
                        bound_over_W1_median=float(x.bound_over_W1.median()), bound_over_W1_min=float(x.bound_over_W1.min()),
                        bound_over_W1_max=float(x.bound_over_W1.max()), bound_median=float(x.bound_sum.median()),
                        W1_median=float(x.W1.median()), term2_share_median=float(share2.median()),
                        n_term2_gt_term1=int((x.bound_term2_sum > x.bound_term1_sum).sum()),
                        term1_over_W1_median=float((x.bound_term1_sum / x.W1).median()),
                        n_term1_ge_W1=int((x.bound_term1_sum >= x.W1).sum())))
    BD = pd.DataFrame(bnd)
    kb = K.assign(rel_BO=(K.W1_BO - K.W1_star) / K.W1_star, rel_BS=(K.W1_BS - K.W1_star) / K.W1_star,
                  rel_module=(K.W1_module_assignment - K.W1_star) / K.W1_star)
    excluded = sorted(set(S[(S.concept == "optimal")].dataset) ^ set(C["datasets"]))
    co_ex = S[(S.concept == "optimal") & ~S.a7]
    t_main = pd.read_parquet(C["paths"]["results_main_final.parquet"], columns=["dataset", "repeat", "fold", "sampler", "t_resample"])
    t_main = t_main[(t_main.repeat == 0) & (t_main.sampler == "MUS-CO")].drop_duplicates(["dataset", "fold"])
    ex = t_main[t_main.t_resample >= 120.0]
    NUM["A7"] = dict(summary=SU.to_dict("records"), bounds=BD.to_dict("records"),
                     bo_max_abs_rel_diff=float(kb.rel_BO.abs().max()), bo_n=len(kb),
                     bo_module_max_abs_rel_diff=float(kb.rel_module.abs().max()),
                     bo_module_keep_equals_record=int(kb.module_keep_equals_record.sum()),
                     bs_rel_median=float(kb.rel_BS.median()), bs_rel_max=float(kb.rel_BS.max()),
                     bs_n_at_opt=int((kb.rel_BS.abs() <= 1e-9).sum()), bs_n_below=int((kb.rel_BS < -1e-9).sum()),
                     solver_counts=W.solver.value_counts().to_dict(), all_certified=bool(W.certified.all()),
                     co_bound_excluded_folds=int(len(ex)), co_bound_excluded_datasets=sorted(ex.dataset.unique()),
                     co_bound_excluded_by_dataset=ex.groupby("dataset").size().to_dict(),
                     q_all_ones=bool((K.q_min == 1).all() and (K.q_max == 1).all()),
                     n_tight=int(((F.bound_over_W1 - 1).abs() < 1e-9).sum()))
    spc = os.path.join(DEL, "A7_spotcheck_network_simplex.csv")
    if os.path.exists(spc):
        sc = pd.read_csv(spc)
        NUM["A7"]["spotcheck_n"] = int(len(sc))
        NUM["A7"]["spotcheck_max_abs_rel"] = float(sc.rel.abs().max())
        NUM["A7"]["spotcheck_datasets"] = int(sc.dataset.nunique())
    return F, D, SU, BD, kb


# ============================================================================================ A8
def a8(C):
    R = pd.DataFrame([r for f in sorted(glob.glob(os.path.join(OUT, "rounds", "*.joblib")))
                      for r in joblib.load(f)["rounds"]])
    S = pd.DataFrame([joblib.load(f)["summary"] for f in sorted(glob.glob(os.path.join(OUT, "rounds", "*.joblib")))])
    R = R.merge(S[["dataset", "repeat", "fold", "concept", "a7", "a8", "reproduces_record"]],
                on=["dataset", "repeat", "fold", "concept"], how="left")
    co = R.concept == "optimal"
    R.loc[co, "ratio_greedy_over_opt_perfect"] = R.loc[co, "greedy_cost_perfect"] / R.loc[co, "cost_perfect"]
    R.loc[co, "ratio_greedy_over_opt_used"] = R.loc[co, "greedy_cost_used"] / R.loc[co, "cost_used"]
    R.to_csv(P("A8_coverage_rounds.csv"), index=False)
    a = R[co & (R.repeat == 0) & (R.fold == 0)]
    byr = a.groupby("round").agg(n=("ratio_greedy_over_opt_perfect", "size"),
                                 median=("ratio_greedy_over_opt_perfect", "median"),
                                 max=("ratio_greedy_over_opt_perfect", "max"),
                                 min=("ratio_greedy_over_opt_perfect", "min")).reset_index()
    byd = a.groupby("dataset").agg(rounds=("round", "max"), median=("ratio_greedy_over_opt_perfect", "median"),
                                   max=("ratio_greedy_over_opt_perfect", "max")).reset_index()
    comp = pd.DataFrame([joblib.load(f) for f in sorted(glob.glob(os.path.join(OUT, "complete", "*.joblib")))])
    comp["rel_gap_pct"] = 100 * comp.rel_gap
    r1 = a[a["round"] == 1].set_index("dataset").cost_perfect
    comp["k10_cost_equals_round1"] = np.isclose(comp.cost_k10.values, r1.reindex(comp.dataset).values, rtol=0, atol=1e-9)
    comp.to_csv(P("A8_complete_graph.csv"), index=False)
    ks = pd.read_csv(C["paths"]["mus_co_k_sweep.csv"])
    kc = ks.groupby("instance").k.apply(lambda s: sorted(s.tolist()))
    uni = ks[ks.instance.str.startswith("uniform")]
    tcol = "seconds_milp_solver" if "seconds_milp_solver" in ks.columns else "seconds"  # pipeline/mus_co_k_sweep.py writes 'seconds'
    NUM["A8"] = dict(
        n_datasets=int(a.dataset.nunique()), n_rounds=len(a), n_full=int(a.full.sum()), n_partial=int((~a.full).sum()),
        ratio_min=float(a.ratio_greedy_over_opt_perfect.min()), ratio_median=float(a.ratio_greedy_over_opt_perfect.median()),
        ratio_max=float(a.ratio_greedy_over_opt_perfect.max()),
        ratio_max_where=a.loc[a.ratio_greedy_over_opt_perfect.idxmax(), ["dataset", "round", "s"]].to_dict(),
        used_partial_min=float(a[~a.full].ratio_greedy_over_opt_used.min()),
        used_partial_median=float(a[~a.full].ratio_greedy_over_opt_used.median()),
        used_partial_max=float(a[~a.full].ratio_greedy_over_opt_used.max()),
        by_round=byr.to_dict("records"), per_dataset_median_of_medians=float(byd["median"].median()),
        per_dataset_max_median=float(byd["max"].median()),
        all_co_runs_ratio_median=float(R.loc[co, "ratio_greedy_over_opt_perfect"].median()),
        all_co_runs_ratio_max=float(R.loc[co, "ratio_greedy_over_opt_perfect"].max()),
        all_co_runs_n_rounds=int(co.sum()),
        solver_rounds=R.loc[co, "solver"].value_counts().to_dict(),
        complete_n=len(comp), complete_status=comp.status.value_counts().to_dict(),
        complete_gap_median_pct=float(comp.rel_gap_pct.median()), complete_gap_q3_pct=q(comp.rel_gap_pct, .75),
        complete_gap_max_pct=float(comp.rel_gap_pct.max()),
        complete_gap_max_where=str(comp.loc[comp.rel_gap_pct.idxmax(), "dataset"]),
        complete_n_zero=int((comp.rel_gap.abs() < 1e-12).sum()), complete_n_negative=int((comp.rel_gap < -1e-12).sum()),
        complete_s_range=[int(comp.s.min()), int(comp.s.max())], complete_solver=comp.solver_complete.value_counts().to_dict(),
        complete_seconds_max=float(comp.seconds_complete.max()), complete_seconds_median=float(comp.seconds_complete.median()),
        complete_k10_consistent=bool(comp.k10_cost_equals_round1.all()),
        ksweep_instances=int(len(kc)), ksweep_rows=int(len(ks)), ksweep_k20=int(kc.apply(lambda l: 20 in l).sum()),
        ksweep_missing_k20=kc[kc.apply(lambda l: 20 not in l)].index.tolist(),
        ksweep_uniform=int(uni.instance.nunique()), ksweep_uniform_k20=int(uni[uni.k == 20].instance.nunique()),
        ksweep_n5000_d10_seconds={int(k_): float(v) for k_, v in ks[(ks.n == 5000) & (ks.d == 10)].set_index("k")[tcol].items()},
        ksweep_n5000_d40_seconds={int(k_): float(v) for k_, v in ks[(ks.n == 5000) & (ks.d == 40)].set_index("k")[tcol].items()})
    return R, comp, byr


def tex_a7a8(SU, BD, kb, byr):
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Wasserstein distortion of the retained majority (A7) and greedy versus optimal matching cost (A8). "
         r"Top: exact $W_1$ between the uniform measures on the full (preprocessed) training majority and on the "
         r"retained majority, first CV repeat (5 folds per dataset); median over the 55 datasets of the fold means "
         r"[first--third quartile], per-dataset ratio to RUS, and number of datasets with a ratio below 1. Middle: "
         r"realised $W_1$ of the coverage arm against the summed per-round bound $\sum_t B_t$ of the Lemma. Bottom: "
         r"ratio of the greedy (stable roommates) to the optimal (candidate-graph) perfect-matching cost on the same "
         r"set in each MUS-CO round (repeat 0, fold 0).}",
         r"\label{tab:a7a8}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"Method & $W_1$ median [IQR] & Ratio to RUS [IQR] & Ratio $<1$ \\", r"\midrule"]
    for r in SU.itertuples():
        L.append(f"{r.sampler} & {r.W1_median:.4f} [{r.W1_q1:.4f}, {r.W1_q3:.4f}] & "
                 f"{r.ratio_median:.3f} [{r.ratio_q1:.3f}, {r.ratio_q3:.3f}] & {r.n_ratio_lt_1}/{r.n_datasets}" + r" \\")
    L += [r"\midrule", r"Coverage arm & folds & $\sum_t B_t / W_1$ median [min, max] & 2nd-term share \\", r"\midrule"]
    for r in BD.itertuples():
        L.append(f"{r.sampler} & {r.n_folds} & {r.bound_over_W1_median:.2f} [{r.bound_over_W1_min:.2f}, {r.bound_over_W1_max:.2f}] & "
                 f"{r.term2_share_median:.2f}" + r" \\")
    L += [r"\midrule", r"MUS-CO round & rounds & greedy/optimal median & max \\", r"\midrule"]
    for r in byr.itertuples():
        L.append(f"{int(r.round)} & {int(r.n)} & {r.median:.3f} & {r.max:.3f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(P("A7_A8_wasserstein_matching.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")


# ============================================================================================ A9
def a9(C):
    rows = []

    def add(check, item, value=None, text=""):
        rows.append(dict(check=check, item=item, value=value, text=text))
    ab = pd.read_csv(C["paths"]["stats_ablation.csv"])
    x = ab[(ab.metric == "auc") & ab.variant.isin(["r1.5", "r2"])].copy()
    x["sig_holm"] = x["sig_holm"].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    for r in x.itertuples():
        add("i_ablation_auc", f"{r.family} {r.classifier} {r.variant}", float(r.mean_diff),
            f"p_holm={r.p_holm:.4g}; {r.verdict}; W/T/L={int(r.wins)}/{int(r.ties)}/{int(r.losses)}")
    NUM["A9i"] = dict(n=len(x), n_negative=int((x.mean_diff < 0).sum()), n_sig=int(x.sig_holm.sum()),
                      n_sig_negative=int((x.sig_holm & (x.mean_diff < 0)).sum()),
                      negative=x[x.mean_diff < 0][["family", "classifier", "variant", "mean_diff", "p_holm"]].to_dict("records"),
                      ns=x[~x.sig_holm][["family", "classifier", "variant", "mean_diff", "p_holm"]].to_dict("records"),
                      mean_diff_range=[float(x.mean_diff.min()), float(x.mean_diff.max())],
                      holm_family_sizes=sorted(set(int(v) for v in x.holm_family_size.dropna())))
    # (ii) trees
    from sklearn.tree import DecisionTreeRegressor
    cov = C["cpx"].set_index("name")[COVARIATES].astype(float).loc[C["datasets"]]
    mg = pd.read_csv(C["paths"]["meta_gains_per_dataset.csv"])
    ga = mg[(mg.classifier == "mean") & (mg.metric == "gmean")]
    leaves = []
    for mth in MUS:
        yv = ga[ga.method == mth].set_index("dataset")["delta_rus"].reindex(cov.index)
        t = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, random_state=0).fit(cov, yv)
        tr_, rules = t.tree_, {}

        def walk(node, conds):
            if tr_.children_left[node] == -1:
                rules[node] = " and ".join(conds)
                return
            f_, th = COVARIATES[tr_.feature[node]], tr_.threshold[node]
            walk(tr_.children_left[node], conds + [f"{f_} <= {th:.3f}"])
            walk(tr_.children_right[node], conds + [f"{f_} > {th:.3f}"])
        walk(0, [])
        lf = t.apply(cov)
        for l_ in np.unique(lf):
            yy = yv[lf == l_]
            leaves.append(dict(method=mth, rule=rules[int(l_)], n=len(yy), mean_gain=float(yy.mean()),
                               r2=float(t.score(cov, yv)), datasets=";".join(yy.index)))
    TL = pd.DataFrame(leaves)
    for r in TL[TL.method == "MUS-BS"].itertuples():
        add("ii_tree_MUS-BS_leaf", r.rule, r.mean_gain, f"n={r.n}; in-sample R2={r.r2:.3f}; datasets={r.datasets}")
    safe = C["meta"]["safe"]
    ybs = ga[ga.method == "MUS-BS"].set_index("dataset")["delta_rus"].reindex(C["datasets"])
    s19 = list(safe[safe >= 0.8].index)
    leaf = TL[(TL.method == "MUS-BS") & (TL.rule == "outlier <= 0.213 and safe > 0.688")].iloc[0]
    l22 = leaf.datasets.split(";")
    extra = sorted(set(l22) - set(s19))
    add("ii_coincidence", "mean MUS-BS gain, datasets with safe share >= 0.8", float(ybs[s19].mean()), f"n={len(s19)}")
    add("ii_coincidence", "mean MUS-BS gain, leaf outlier<=0.213 & safe>0.688", float(ybs[l22].mean()), f"n={len(l22)}")
    add("ii_coincidence", "datasets in leaf but safe < 0.8", float(ybs[extra].mean()),
        "; ".join(f"{d} (gain {ybs[d]:+.4f}, safe {safe[d]:.3f})" for d in extra))
    NUM["A9ii"] = dict(leaves=TL[TL.method == "MUS-BS"][["rule", "n", "mean_gain", "r2"]].to_dict("records"),
                       mean19=float(ybs[s19].mean()), n19=len(s19), mean22=float(ybs[l22].mean()), n22=len(l22),
                       subset=bool(set(s19) <= set(l22)), extra=extra, extra_mean=float(ybs[extra].mean()),
                       extra_gains={d: float(ybs[d]) for d in extra}, extra_safe={d: float(safe[d]) for d in extra},
                       max19=float(ybs[s19].max()),
                       tree_r2={m: float(TL[TL.method == m].r2.iloc[0]) for m in MUS})
    # (iii) meta correlations of the coverage arm
    mc = pd.read_csv(C["paths"]["meta_gain_correlations.csv"])
    ca = mc[mc.method.isin(["MUS-CS", "MUS-CO"])]
    sig = ca[ca.p_holm < 0.05]
    for r in sig.itertuples():
        add("iii_holm_significant", f"{r.method} {r.metric} vs {r.reference} ({r.subset}) ~ {r.covariate}", float(r.rho),
            f"n={r.n}; p={r.p:.3g}; p_holm={r.p_holm:.3g}")
    rus_sig = ca[(ca.reference == "rus") & (ca.p_holm < 0.05)]
    gcov = ga[ga.method.isin(["MUS-CS", "MUS-CO"])].pivot(index="dataset", columns="method", values="delta_rus").reindex(C["datasets"])
    gcov["safe"] = safe
    ext = gcov[(gcov[["MUS-CS", "MUS-CO"]] > 0.02).any(axis=1) | (gcov[["MUS-CS", "MUS-CO"]] < -0.02).any(axis=1)]
    for d, r in ext.iterrows():
        add("iii_extreme_coverage_gain", d, float(r["MUS-CS"]), f"MUS-CO {r['MUS-CO']:+.4f}; safe share {r['safe']:.3f}")
    NUM["A9iii"] = dict(n_tests=len(ca), n_sig=len(sig), n_unadj=int((ca.p < 0.05).sum()), n_sig_rus=len(rus_sig),
                        sig=sig[["subset", "metric", "reference", "method", "covariate", "n", "rho", "p", "p_holm"]].to_dict("records"),
                        block_size=int(ca.groupby(["subset", "metric", "reference", "method"]).size().iloc[0]),
                        gains=ext[ext[["MUS-CS", "MUS-CO"]].max(axis=1) > 0.02].round(4).reset_index().to_dict("records"),
                        losses=ext[ext[["MUS-CS", "MUS-CO"]].min(axis=1) < -0.02].round(4).reset_index().to_dict("records"))
    # (iv) NM3 W1 handling and enrichment
    sel = joblib.load(C["paths"]["selections_main_final.joblib"])
    nmin = C["res"].drop_duplicates(["dataset", "repeat", "fold"]).set_index(["dataset", "repeat", "fold"])["n_min_train"]
    nm3 = pd.DataFrame([dict(dataset=r["dataset"], repeat=r["repeat"], fold=r["fold"], m=len(r["kept_majority"]))
                        for r in sel if r["sampler"] == "NM3"])
    nm3["p"] = [int(nmin.loc[(a_, b_, c_)]) for a_, b_, c_ in zip(nm3.dataset, nm3.repeat, nm3.fold)]
    per = nm3.assign(short=nm3.m < nm3.p).groupby("dataset").short.sum()
    dd = pd.read_csv(C["paths"]["diag_by_dataset.csv"])
    dsum = pd.read_csv(C["paths"]["diag_summary.csv"]).set_index("sampler")
    ddn = dd[dd.sampler == "NM3"].set_index("dataset")
    nan_eq = bool(((ddn.W1_ratio.isna()) == (per.reindex(ddn.index) == 25)).all())
    ef = dd[dd.sampler == "RUS"].set_index("dataset").enrich_folds
    NUM["A9iv"] = dict(nm3_short_folds=int((nm3.m < nm3.p).sum()), nm3_folds=len(nm3), nm3_over_folds=int((nm3.m > nm3.p).sum()),
                       nm3_ds_any_short=int((per > 0).sum()), nm3_ds_all_short=int((per == 25).sum()),
                       nm3_ds_partial=int(((per > 0) & (per < 25)).sum()),
                       nm3_ds_partial_list={d: int(v) for d, v in per[(per > 0) & (per < 25)].items()},
                       nm3_w1_ndatasets=int(dsum.loc["NM3", "W1_ratio_ndatasets"]), nm3_w1_median=float(dsum.loc["NM3", "W1_ratio_median"]),
                       nan_iff_all_short=nan_eq,
                       enrich_ndatasets=int(dsum.loc["RUS", "enrich_ndatasets"]),
                       enrich_ndatasets_all_methods=sorted(set(int(v) for v in dsum["enrich_ndatasets"])),
                       enrich_folds_25=int((ef == 25).sum()), enrich_folds_partial=int(((ef > 0) & (ef < 25)).sum()),
                       enrich_folds_zero=int((ef == 0).sum()), enrich_folds_min_pos=int(ef[ef > 0].min()))
    add("iv_nm3_w1", "NM3 folds with |S|<|P| (of 1375)", NUM["A9iv"]["nm3_short_folds"],
        "W1_SP set to NaN unless |S|=|P| (fold_diagnostics); fold NaNs skipped in the dataset mean (pandas mean)")
    add("iv_nm3_w1", "NM3 datasets short in all 25 folds -> W1 ratio undefined, dropped from median", NUM["A9iv"]["nm3_ds_all_short"])
    add("iv_nm3_w1", "NM3 datasets short in some folds -> mean over the |S|=|P| folds only", NUM["A9iv"]["nm3_ds_partial"],
        json.dumps(NUM["A9iv"]["nm3_ds_partial_list"]))
    add("iv_nm3_w1", "NM3 datasets entering the W1(S,P)/W1* median", NUM["A9iv"]["nm3_w1_ndatasets"])
    add("iv_enrichment", "datasets with defined unsafe-majority enrichment (all methods)", NUM["A9iv"]["enrich_ndatasets"],
        f"{NUM['A9iv']['enrich_folds_zero']} datasets have no unsafe majority instance in any fold; "
        f"{NUM['A9iv']['enrich_folds_25']} defined in all 25 folds, {NUM['A9iv']['enrich_folds_partial']} in 1-24 folds")
    # (v) test and Holm family of the within-dataset correlations
    link = pd.read_csv(C["paths"]["diag_performance_link.csv"])
    fam_sizes = link.groupby(["set", "classifier"]).size()
    NUM["A9v"] = dict(n_rows=len(link), n_families=int(len(fam_sizes)), family_sizes=sorted(set(int(v) for v in fam_sizes)),
                      sets=sorted(link.set.unique()), diags_all=sorted(link[link.set == "all_selections"].diagnostic.unique()),
                      diags_other=sorted(link[link.set == "size_balanced"].diagnostic.unique()))
    add("v_link_test", "test", None,
        "one-sample two-sided Wilcoxon signed-rank test, scipy.stats.wilcoxon(v) with defaults (zero_method='wilcox', "
        "method='auto'), on the per-dataset within-dataset Spearman correlations (across the selection methods of the set) "
        "between a fold-averaged diagnostic and the mean G-mean; requires >= 6 defined correlations and at least one non-zero")
    add("v_link_test", "Holm family", NUM["A9v"]["family_sizes"][0],
        f"Holm across the 7 diagnostics within each method set x classifier block ({NUM['A9v']['n_families']} families: "
        f"4 sets x 6 classifier levels incl. the classifier mean); all_selections uses BP, CE, ED_SN, ED_SP, enrich, "
        f"iso_enrich, size_ratio; the other sets use W1_SP instead of size_ratio; only finite p-values enter")
    A = pd.DataFrame(rows)
    A.to_csv(P("A9_checks.csv"), index=False)
    return A, TL




# ============================================================================================ driver
def main(parts=None):
    os.makedirs(DEL, exist_ok=True)
    C = load()
    parts = parts or ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]
    out = {}
    if "A1" in parts:
        out["A1"] = a1(C)
    if "A2" in parts:
        out["A2"] = a2(C)
    if "A3" in parts:
        out["A3"] = a3(C)
    if "A4" in parts:
        out["A4"] = a4(C)
    if "A5" in parts:
        out["A5"] = a5(C)
    if "A6" in parts:
        out["A6"] = a6(C)
    if "A8" in parts:
        out["A8"] = a8(C)
    if "A7" in parts:
        out["A7"] = a7(C)
        if "A8" in out:
            F, D, SU, BD, kb = out["A7"]
            tex_a7a8(SU, BD, kb, out["A8"][2])
    if "A9" in parts:
        out["A9"] = a9(C)
    json.dump(NUM, open(P("analyses_numbers.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if set(parts) >= {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"}:
        import rev_report
        rev_report.write_report(NUM, P("analyses_report.md"))
    return out


if __name__ == "__main__":
    main(sys.argv[1:] or None)
