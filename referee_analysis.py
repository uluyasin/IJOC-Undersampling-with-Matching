"""Analysis of the referee experiments X1-X6 (per-dataset means, Wilcoxon/Holm tests, ranks, LaTeX).

Conventions (all experiments): per-dataset means over the folds of the experiment; two-sided Wilcoxon
signed-rank test on the per-dataset differences (method - reference) with zero_method='zsplit';
Holm correction within the family stated in each function; median difference; W/T/L counts of datasets
with difference > 0.001 / |difference| <= 0.001 / < -0.001 (W = method better for every metric used
here because all are higher-is-better, except the across-seed SD in X5 where lower is better and the
sign is reported as is).
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

CLASSIFIERS = ["CART", "kNN", "SVM", "LR", "RF"]
BALANCING = ["RUS", "NM1", "NM2", "NM3", "CC", "MUS-BS", "MUS-BO", "MUS-CS"]
MUS3 = ["MUS-BS", "MUS-BO", "MUS-CS"]
TIE = 0.001


# ----------------------------------------------------------------------------- statistics helpers
def wtest(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    out = dict(n=n, median_diff=np.nan, mean_diff=np.nan, p=np.nan, W=0, T=0, L=0)
    if n == 0:
        return out
    out.update(median_diff=float(np.median(d)), mean_diff=float(np.mean(d)),
               W=int((d > TIE).sum()), L=int((d < -TIE).sum()))
    out["T"] = n - out["W"] - out["L"]
    if np.all(d == 0):
        out["p"] = 1.0
    else:
        out["p"] = float(wilcoxon(d, zero_method="zsplit", alternative="two-sided").pvalue)
    return out


def holm(p):
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    adj = np.full(len(p), np.nan)
    order = np.argsort(np.where(ok, p, np.inf), kind="stable")
    running = 0.0
    for rank, i in enumerate(order[:m]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def add_holm(df, family_cols):
    df = df.copy()
    df["p_holm"] = np.nan
    for _, g in df.groupby(family_cols, dropna=False):
        df.loc[g.index, "p_holm"] = holm(g["p"].values)
    df["holm_family_size"] = df.groupby(family_cols, dropna=False)["p"].transform("count")
    return df


def ds_means(df, keys, metrics):
    return df.groupby(["dataset"] + keys, dropna=False)[metrics].mean().reset_index()


def paired(dm_a, dm_b, metric, on=("dataset",)):
    j = pd.merge(dm_a[list(on) + [metric]], dm_b[list(on) + [metric]], on=list(on), suffixes=("_a", "_b"))
    return (j[metric + "_a"] - j[metric + "_b"]).values


def fmt_cell(med, p_holm, digits=3):
    if not np.isfinite(med):
        return "--"
    s = f"{med:+.{digits}f}".replace("+0.000", "0.000").replace("-0.000", "0.000")
    s = s.replace("-", "$-$")
    if np.isfinite(p_holm) and p_holm < 0.05:
        s = r"\textbf{" + s + "}"
    return s


def read_exp(run_dir, exp, extra=False):
    parts = sorted(glob.glob(os.path.join(run_dir, exp + ("_extra" if extra else ""), f"{exp}_*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


def avg_ranks(dm, samplers, metric, by):
    """Average ranks (1 = best) of `samplers` over datasets, per group `by`."""
    out = []
    for key, g in dm[dm["sampler"].isin(samplers)].groupby(by):
        w = g.pivot(index="dataset", columns="sampler", values=metric)[samplers].dropna()
        r = np.vstack([rankdata(-row) for row in w.values]) if len(w) else np.empty((0, len(samplers)))
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(metric=metric, n_datasets=len(w))
        rec.update({s: float(r[:, i].mean()) if len(w) else np.nan for i, s in enumerate(samplers)})
        out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- X1
def analyse_x1(x1, main):
    mets = ["gmean", "auc", "bacc", "sens", "spec", "f1", "mcc", "prauc"]
    x1 = x1[x1["classifier"].isin(CLASSIFIERS)].copy()
    # sanity: default variant vs main run, same unit
    d = x1[x1["variant"] == "default"].merge(main, on=["dataset", "repeat", "fold", "sampler", "classifier"],
                                             suffixes=("", "_main"))
    sanity = {m: float(np.nanmax(np.abs(d[m] - d[m + "_main"]))) for m in mets}
    sanity["n_rows_compared"] = int(len(d))
    sanity["max_abs_diff_all_metrics"] = float(max(v for k, v in sanity.items() if k in mets))
    dm = ds_means(x1, ["sampler", "classifier", "variant"], mets)
    rows = []
    for var in ("default", "thr_adjusted", "tuned"):
        for c in CLASSIFIERS:
            ref = dm[(dm.sampler == "RUS") & (dm.classifier == c) & (dm.variant == var)]
            for met in ("gmean", "auc", "sens", "spec", "bacc"):
                for s in MUS3:
                    a = dm[(dm.sampler == s) & (dm.classifier == c) & (dm.variant == var)]
                    r = dict(variant=var, classifier=c, metric=met, method=s, reference="RUS")
                    r.update(wtest(paired(a, ref, met)))
                    rows.append(r)
    tests = add_holm(pd.DataFrame(rows), ["variant", "classifier", "metric"])
    ranks = pd.concat([avg_ranks(dm, BALANCING, met, ["classifier", "variant"]) for met in ("gmean", "auc")],
                      ignore_index=True)
    # variant effects within each sampler (thr_adjusted / tuned minus default), descriptive + tests
    vrows = []
    for c in CLASSIFIERS:
        for s in BALANCING:
            base = dm[(dm.sampler == s) & (dm.classifier == c) & (dm.variant == "default")]
            for var in ("thr_adjusted", "tuned"):
                a = dm[(dm.sampler == s) & (dm.classifier == c) & (dm.variant == var)]
                for met in ("gmean", "auc", "sens", "spec"):
                    r = dict(classifier=c, sampler=s, variant=var, metric=met, reference="default")
                    r.update(wtest(paired(a, base, met)))
                    vrows.append(r)
    vt = add_holm(pd.DataFrame(vrows), ["classifier", "variant", "metric"])
    # choices
    ch = x1[x1["variant"] == "tuned"].groupby(["sampler", "classifier"]).agg(
        tuned_is_default=("tuned_is_default", "mean"), thr_moved=("thr_chosen", lambda s: np.nan)).reset_index()
    thr = x1[x1["variant"] == "thr_adjusted"].copy()
    thr["thr_moved"] = (thr["thr_chosen"] != thr["thr_default"]).astype(float)
    ch = ch.drop(columns="thr_moved").merge(
        thr.groupby(["sampler", "classifier"]).agg(thr_moved=("thr_moved", "mean"),
                                                   thr_median=("thr_chosen", "median")).reset_index(),
        on=["sampler", "classifier"])
    return dict(sanity=sanity, dm=dm, tests=tests, ranks=ranks, variant_tests=vt, choices=ch)


def latex_x1(res, n_folds):
    t = res["tests"]
    cols = [("gmean", "default"), ("gmean", "thr_adjusted"), ("gmean", "tuned"), ("auc", "default"), ("auc", "tuned")]
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Nested threshold adjustment and tuning (X1; 55 datasets, " + str(n_folds) +
             r" outer folds). Median over datasets of the per-dataset difference to RUS under the same variant; "
             r"bold: Holm-adjusted two-sided Wilcoxon $p<0.05$ (family: 3 MUS methods per classifier, metric and "
             r"variant). ROC-AUC is unchanged by threshold adjustment. A bold 0.000 means that the median dataset is unchanged (absolute difference $\leq 0.0005$) while the non-zero differences are systematically signed; W/T/L counts and exact $p$-values are in the summary CSV.}",
             r"\label{tab:x1}", r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{G-mean} & \multicolumn{2}{c}{ROC-AUC}\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"Classifier & Method & default & thr.-adj. & tuned & default & tuned\\", r"\midrule"]
    for c in CLASSIFIERS:
        for i, s in enumerate(MUS3):
            cells = []
            for met, var in cols:
                r = t[(t.classifier == c) & (t.method == s) & (t.metric == met) & (t.variant == var)]
                cells.append(fmt_cell(r["median_diff"].iloc[0], r["p_holm"].iloc[0]) if len(r) else "--")
            lines.append((c if i == 0 else "") + " & " + s + " & " + " & ".join(cells) + r"\\")
        if c != CLASSIFIERS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\midrule"]
    rk = res["ranks"]
    for met, lab in (("gmean", "G-mean"), ("auc", "ROC-AUC")):
        lines.append(r"\multicolumn{7}{l}{\emph{Average rank among the 8 balancing samplers (" + lab +
                     r", mean over classifiers; RUS / MUS-BS / MUS-BO / MUS-CS)}}\\")
        cells = []
        for var in ("default", "thr_adjusted", "tuned"):
            g = rk[(rk.metric == met) & (rk.variant == var)]
            cells.append(" / ".join(f"{g[s].mean():.2f}" for s in ["RUS"] + MUS3) if len(g) else "--")
        lines.append(r"\multicolumn{7}{l}{default: " + cells[0] + r"; thr.-adj.: " + cells[1] + r"; tuned: " +
                     cells[2] + r"}\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- X2
def analyse_x2(x2, main):
    mets = ["gmean", "auc", "sens", "spec"]
    x2 = x2[x2["classifier"].isin(CLASSIFIERS)]
    dm_p = ds_means(x2, ["sampler", "classifier"], mets)
    dm_o = ds_means(main[main.sampler.isin(MUS3 + ["RUS"])], ["sampler", "classifier"], mets)
    rows = []
    for c in CLASSIFIERS:
        for met in mets:
            for s in MUS3:
                a = dm_p[(dm_p.sampler == s) & (dm_p.classifier == c)]
                o = dm_o[(dm_o.sampler == s) & (dm_o.classifier == c)]
                rus = dm_o[(dm_o.sampler == "RUS") & (dm_o.classifier == c)]
                r = dict(classifier=c, metric=met, method=s, comparison="permuted_vs_original")
                r.update(wtest(paired(a, o, met)))
                rows.append(r)
                r = dict(classifier=c, metric=met, method=s, comparison="permuted_vs_RUS")
                r.update(wtest(paired(a, rus, met)))
                rows.append(r)
                r = dict(classifier=c, metric=met, method=s, comparison="original_vs_RUS")
                r.update(wtest(paired(o, rus, met)))
                rows.append(r)
    tests = add_holm(pd.DataFrame(rows), ["comparison", "classifier", "metric"])
    u = x2.drop_duplicates(["dataset", "repeat", "fold", "sampler"])
    jac = u.groupby("sampler").agg(n_folds=("jaccard", "size"), median_jaccard=("jaccard", "median"),
                                   mean_jaccard=("jaccard", "mean"), q25_jaccard=("jaccard", lambda s: s.quantile(.25)),
                                   share_identical=("same_selection", "mean"),
                                   median_chance_jaccard=("jaccard_chance", "median"),
                                   recomputed_matches_main=("recomputed_matches_main", "mean")).reset_index()
    if "pair_jaccard_round1" in u:
        pj = u[u.sampler == "MUS-CS"]["pair_jaccard_round1"]
        jac["median_pair_jaccard_round1"] = np.where(jac.sampler == "MUS-CS", pj.median(), np.nan)
        jac["share_pairs_identical_round1"] = np.where(jac.sampler == "MUS-CS", (pj == 1).mean(), np.nan)
    jds = u.groupby(["dataset", "sampler"])["jaccard"].mean().unstack()
    return dict(tests=tests, jaccard=jac, jaccard_by_dataset=jds)


# ----------------------------------------------------------------------------- X3
def analyse_x3(x3, main):
    mets = ["gmean", "sens", "spec", "auc"]
    x3 = x3[x3["classifier"].isin(CLASSIFIERS)]
    dm_v = ds_means(x3, ["sampler", "classifier"], mets)
    dm_o = ds_means(main[main.sampler.isin(["MUS-BS", "RUS"])], ["sampler", "classifier"], mets)
    rows = []
    for c in CLASSIFIERS:
        for met in mets:
            for v in ("MUS-BS:safecap", "MUS-BS:nounsafe"):
                a = dm_v[(dm_v.sampler == v) & (dm_v.classifier == c)]
                for ref in ("MUS-BS", "RUS"):
                    b = dm_o[(dm_o.sampler == ref) & (dm_o.classifier == c)]
                    r = dict(classifier=c, metric=met, method=v, reference=ref)
                    r.update(wtest(paired(a, b, met)))
                    rows.append(r)
            a = dm_o[(dm_o.sampler == "MUS-BS") & (dm_o.classifier == c)]
            b = dm_o[(dm_o.sampler == "RUS") & (dm_o.classifier == c)]
            r = dict(classifier=c, metric=met, method="MUS-BS", reference="RUS (main, for context)")
            r.update(wtest(paired(a, b, met)))
            rows.append(r)
    tests = add_holm(pd.DataFrame(rows), ["classifier", "metric"])
    tests.loc[tests.reference.str.startswith("RUS (main"), "p_holm"] = np.nan
    tests = add_holm_subset(tests, ["classifier", "metric"], tests.reference.isin(["MUS-BS", "RUS"]))
    u = x3.drop_duplicates(["dataset", "repeat", "fold", "sampler"])
    flags = u.groupby("sampler").agg(n_folds=("dataset", "size"), fallback_folds=("fallback", "sum"),
                                     topped_up_folds=("topped_up", "sum"),
                                     median_jaccard_vs_musbs=("jaccard_vs_musbs", "median"),
                                     mean_n_hospitals=("n_hospitals", "mean"),
                                     mean_n_unacceptable=("n_unacceptable", "mean")).reset_index()
    fb = u[u.fallback.fillna(False).astype(bool)].groupby("sampler")["dataset"].unique().to_dict()
    tu = u[u.topped_up.fillna(False).astype(bool)].groupby("sampler")["dataset"].unique().to_dict()
    return dict(tests=tests, flags=flags, fallback_datasets=fb, topped_up_datasets=tu)


def add_holm_subset(df, family_cols, mask):
    df = df.copy()
    df.loc[~mask, "p_holm"] = np.nan
    for _, g in df[mask].groupby(family_cols):
        df.loc[g.index, "p_holm"] = holm(g["p"].values)
    df.loc[mask, "holm_family_size"] = df[mask].groupby(family_cols)["p"].transform("count")
    df.loc[~mask, "holm_family_size"] = np.nan
    return df


def latex_x3(res):
    t = res["tests"]
    mets = [("gmean", "G-mean"), ("sens", "Sens."), ("spec", "Spec."), ("auc", "AUC")]
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
             r"\caption{Intervention on the boundary arm (X3; 55 datasets, 25 folds). Median per-dataset difference "
             r"of each MUS-BS variant to standard MUS-BS and to RUS; bold: Holm-adjusted Wilcoxon $p<0.05$ "
             r"(family: 2 variants $\times$ 2 references per classifier and metric). safecap: capacities only for "
             r"safe/borderline minority instances; nounsafe: majority instances with $\geq 3$ minority neighbours "
             r"among their 5 nearest neighbours are unacceptable. A bold 0.000 means that the median dataset is unchanged (absolute difference $\leq 0.0005$) while the non-zero differences are systematically signed; W/T/L counts and exact $p$-values are in the summary CSV.}",
             r"\label{tab:x3}", r"\begin{tabular}{ll" + "r" * 8 + "}", r"\toprule",
             r" & & \multicolumn{4}{c}{vs standard MUS-BS} & \multicolumn{4}{c}{vs RUS}\\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r"Classifier & Variant & " + " & ".join(m[1] for m in mets) + " & " + " & ".join(m[1] for m in mets) + r"\\",
             r"\midrule"]
    for c in CLASSIFIERS:
        for i, v in enumerate(("MUS-BS:safecap", "MUS-BS:nounsafe")):
            cells = []
            for ref in ("MUS-BS", "RUS"):
                for met, _ in mets:
                    r = t[(t.classifier == c) & (t.method == v) & (t.reference == ref) & (t.metric == met)]
                    cells.append(fmt_cell(r["median_diff"].iloc[0], r["p_holm"].iloc[0]) if len(r) else "--")
            lines.append((c if i == 0 else "") + " & " + v.split(":")[1] + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- X4
def analyse_x4(x4, main):
    mets = ["gmean", "auc", "sens", "spec"]
    x4 = x4[x4["classifier"].isin(CLASSIFIERS)]
    dm_n = ds_means(x4, ["sampler", "classifier"], mets)
    dm_o = ds_means(main[main.sampler.isin(["MUS-CS", "RUS"])], ["sampler", "classifier"], mets)
    rows = []
    for c in CLASSIFIERS:
        for met in mets:
            for s in ("LPM", "KSRUS"):
                a = dm_n[(dm_n.sampler == s) & (dm_n.classifier == c)]
                for ref in ("RUS", "MUS-CS"):
                    b = dm_o[(dm_o.sampler == ref) & (dm_o.classifier == c)]
                    r = dict(classifier=c, metric=met, method=s, reference=ref)
                    r.update(wtest(paired(a, b, met)))
                    rows.append(r)
    tests = add_holm(pd.DataFrame(rows), ["classifier", "metric"])
    # coverage diagnostics (one row per unit)
    u = x4.drop_duplicates(["dataset", "repeat", "fold", "sampler"])
    lp = u[u.sampler == "LPM"].groupby("dataset")[["CE_self", "CE_RUS", "CE_MUS-CS", "BP_self", "BP_RUS", "BP_MUS-CS",
                                                     "SB_self", "SB_RUS", "SB_MUS-CS"]].mean()
    ks = u[u.sampler == "KSRUS"].groupby("dataset")[["CE_self", "BP_self", "SB_self"]].mean()
    diag = pd.DataFrame({"CE_LPM": lp["CE_self"], "CE_RUS": lp["CE_RUS"], "CE_MUS-CS": lp["CE_MUS-CS"],
                         "CE_KSRUS": ks["CE_self"], "BP_LPM": lp["BP_self"], "BP_RUS": lp["BP_RUS"],
                         "BP_MUS-CS": lp["BP_MUS-CS"], "BP_KSRUS": ks["BP_self"], "SB_LPM": lp["SB_self"],
                         "SB_RUS": lp["SB_RUS"], "SB_MUS-CS": lp["SB_MUS-CS"], "SB_KSRUS": ks["SB_self"]})
    drows = []
    for met in ("CE", "BP", "SB"):
        for a_, b_ in (("LPM", "RUS"), ("LPM", "MUS-CS"), ("KSRUS", "RUS"), ("MUS-CS", "RUS"), ("LPM", "KSRUS")):
            r = dict(diagnostic=met, method=a_, reference=b_, median_method=float(diag[f"{met}_{a_}"].median()),
                     median_reference=float(diag[f"{met}_{b_}"].median()),
                     n_method_lower=int((diag[f"{met}_{a_}"] < diag[f"{met}_{b_}"]).sum()))
            r.update(wtest((diag[f"{met}_{a_}"] - diag[f"{met}_{b_}"]).values))
            drows.append(r)
    dtests = add_holm(pd.DataFrame(drows), ["diagnostic"])
    return dict(tests=tests, diag=diag, diag_tests=dtests)


# ----------------------------------------------------------------------------- X5
def analyse_x5(x5, main, x4):
    mets = ["gmean", "auc", "sens", "spec"]
    x5 = x5[x5["classifier"].isin(CLASSIFIERS)].copy()
    dm_e = ds_means(x5, ["sampler", "classifier"], mets + ["member_gmean_mean", "member_auc_mean",
                                                           "member_gmean_sd", "member_auc_sd"])
    single = pd.concat([main[main.sampler.isin(["RUS", "MUS-CS"])], x4[x4.sampler == "LPM"]], ignore_index=True)
    dm_s = ds_means(single[single.classifier.isin(CLASSIFIERS)], ["sampler", "classifier"], mets)
    rows = []
    for c in CLASSIFIERS:
        for met in ("gmean", "auc", "sens", "spec"):
            for s in ("RUS", "MUS-CS", "LPM"):
                a = dm_e[(dm_e.sampler == s) & (dm_e.classifier == c)]
                b = dm_s[(dm_s.sampler == s) & (dm_s.classifier == c)]
                r = dict(classifier=c, metric=met, method=f"{s} ensemble", reference=f"{s} single (seed = unit seed)",
                         family="ensemble_vs_single")
                r.update(wtest(paired(a, b, met)))
                rows.append(r)
                if met in ("gmean", "auc"):
                    mm = "member_gmean_mean" if met == "gmean" else "member_auc_mean"
                    bb = a[["dataset", mm]].rename(columns={mm: met})
                    r = dict(classifier=c, metric=met, method=f"{s} ensemble", reference=f"{s} mean of 10 members",
                             family="ensemble_vs_member_mean")
                    r.update(wtest(paired(a, bb, met)))
                    rows.append(r)
            rus = dm_e[(dm_e.sampler == "RUS") & (dm_e.classifier == c)]
            for s in ("MUS-CS", "LPM"):
                a = dm_e[(dm_e.sampler == s) & (dm_e.classifier == c)]
                r = dict(classifier=c, metric=met, method=f"{s} ensemble", reference="RUS ensemble",
                         family="ensemble_vs_RUS_ensemble")
                r.update(wtest(paired(a, rus, met)))
                rows.append(r)
        for met, col in (("gmean", "member_gmean_sd"), ("auc", "member_auc_sd")):
            for s_a, s_b in (("MUS-CS", "RUS"), ("LPM", "RUS"), ("MUS-CS", "LPM")):
                a = dm_e[(dm_e.sampler == s_a) & (dm_e.classifier == c)]
                b = dm_e[(dm_e.sampler == s_b) & (dm_e.classifier == c)]
                r = dict(classifier=c, metric=f"sd_single_{met}", method=s_a, reference=s_b, family="across_seed_sd",
                         median_method=float(a[col].median()), median_reference=float(b[col].median()))
                r.update(wtest(paired(a, b, col)))
                rows.append(r)
    tests = add_holm(pd.DataFrame(rows), ["family", "classifier", "metric"])
    sd = dm_e.groupby(["sampler", "classifier"])[["member_gmean_sd", "member_auc_sd"]].median().reset_index()
    return dict(tests=tests, sd=sd, dm=dm_e)


def latex_x4x5(r4, r5):
    t = r4["tests"]
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Spatially balanced baselines (X4; 55 datasets, 25 folds). Median per-dataset difference of "
             r"LPM (local pivotal method, LPM2) and KSRUS ($k$-means-stratified RUS) to RUS and to MUS-CS; bold: "
             r"Holm-adjusted Wilcoxon $p<0.05$ (family: 2 methods $\times$ 2 references per classifier and metric). A bold 0.000 means that the median dataset is unchanged (absolute difference $\leq 0.0005$) while the non-zero differences are systematically signed; W/T/L counts and exact $p$-values are in the summary CSV.}",
             r"\label{tab:x4}", r"\begin{tabular}{l" + "r" * 8 + "}", r"\toprule",
             r" & \multicolumn{4}{c}{G-mean} & \multicolumn{4}{c}{ROC-AUC}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Classifier & \multicolumn{2}{c}{LPM vs} & \multicolumn{2}{c}{KSRUS vs} & \multicolumn{2}{c}{LPM vs} & "
             r"\multicolumn{2}{c}{KSRUS vs}\\",
             r" & RUS & MUS-CS & RUS & MUS-CS & RUS & MUS-CS & RUS & MUS-CS\\", r"\midrule"]
    for c in CLASSIFIERS:
        cells = []
        for met in ("gmean", "auc"):
            for s in ("LPM", "KSRUS"):
                for ref in ("RUS", "MUS-CS"):
                    r = t[(t.classifier == c) & (t.method == s) & (t.reference == ref) & (t.metric == met)]
                    cells.append(fmt_cell(r["median_diff"].iloc[0], r["p_holm"].iloc[0]) if len(r) else "--")
        lines.append(c + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    t = r5["tests"]
    lines += [r"\begin{table}[htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
              r"\caption{Ensembles of $T=10$ undersampled subsets (X5; 55 datasets, 25 folds; mean of member scores). "
              r"Columns 3--6: median per-dataset difference ensemble $-$ single subset (main-run seed) and "
              r"ensemble $-$ RUS ensemble; columns 7--8: median over datasets of the across-seed SD of a single "
              r"subset's G-mean and ROC-AUC (10 members). Bold: Holm-adjusted Wilcoxon $p<0.05$. A bold 0.000 means that the median dataset is unchanged (absolute difference $\leq 0.0005$) while the non-zero differences are systematically signed; W/T/L counts and exact $p$-values are in the summary CSV.}",
              r"\label{tab:x5}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
              r" & & \multicolumn{2}{c}{Ens.\ $-$ single} & \multicolumn{2}{c}{Ens.\ $-$ RUS ens.} & "
              r"\multicolumn{2}{c}{SD single subset}\\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
              r"Classifier & Sampler & G-mean & AUC & G-mean & AUC & G-mean & AUC\\", r"\midrule"]
    sd = r5["sd"]
    for c in CLASSIFIERS:
        for i, s in enumerate(("RUS", "MUS-CS", "LPM")):
            cells = []
            for met in ("gmean", "auc"):
                r = t[(t.classifier == c) & (t.method == f"{s} ensemble") & (t.family == "ensemble_vs_single") &
                      (t.metric == met)]
                cells.append(fmt_cell(r["median_diff"].iloc[0], r["p_holm"].iloc[0]) if len(r) else "--")
            for met in ("gmean", "auc"):
                if s == "RUS":
                    cells.append("--")
                    continue
                r = t[(t.classifier == c) & (t.method == f"{s} ensemble") & (t.family == "ensemble_vs_RUS_ensemble") &
                      (t.metric == met)]
                cells.append(fmt_cell(r["median_diff"].iloc[0], r["p_holm"].iloc[0]) if len(r) else "--")
            g = sd[(sd.sampler == s) & (sd.classifier == c)]
            cells += [f"{g['member_gmean_sd'].iloc[0]:.3f}" if len(g) else "--",
                      f"{g['member_auc_sd'].iloc[0]:.3f}" if len(g) else "--"]
            lines.append((c if i == 0 else "") + " & " + s + " & " + " & ".join(cells) + r"\\")
        if c != CLASSIFIERS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- X6
def analyse_x6(x6, main, dup_counts):
    mets = ["gmean", "auc", "sens", "spec"]
    x6 = x6[x6["classifier"].isin(CLASSIFIERS)]
    dd = sorted(x6["dataset"].unique())
    keep_main = main[~main.dataset.isin(dd) & main.sampler.isin(["NONE"] + BALANCING)]
    comb = pd.concat([keep_main, x6], ignore_index=True)
    dm_c = ds_means(comb, ["sampler", "classifier"], mets)
    dm_o = ds_means(main[main.sampler.isin(["NONE"] + BALANCING)], ["sampler", "classifier"], mets)
    rows = []
    for data_lab, dm in (("deduplicated", dm_c), ("original", dm_o),
                         ("deduplicated_only_29", dm_c[dm_c.dataset.isin(dd)]),
                         ("original_only_29", dm_o[dm_o.dataset.isin(dd)])):
        for c in CLASSIFIERS:
            for met in mets:
                rus = dm[(dm.sampler == "RUS") & (dm.classifier == c)]
                for s in MUS3:
                    a = dm[(dm.sampler == s) & (dm.classifier == c)]
                    r = dict(data=data_lab, classifier=c, metric=met, method=s, reference="RUS")
                    r.update(wtest(paired(a, rus, met)))
                    rows.append(r)
    tests = add_holm(pd.DataFrame(rows), ["data", "classifier", "metric"])
    ranks = pd.concat([avg_ranks(dm_c.assign(variant="dedup"), BALANCING, met, ["classifier"]) for met in
                       ("gmean", "auc")], ignore_index=True)
    return dict(tests=tests, ranks=ranks, datasets=dd)


def latex_x6(r6, dup_counts):
    t = r6["tests"]
    n_dd = len(r6["datasets"])
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Exact duplicates (X6). " + str(n_dd) + r" datasets contain exact duplicate rows; after removing "
             r"them (first occurrence kept, cross-class duplicates kept) and regenerating the $5\times5$ folds, the "
             r"main comparison is repeated on the 55 datasets (deduplicated where applicable, otherwise main results). "
             r"Median per-dataset difference to RUS; bold: Holm-adjusted Wilcoxon $p<0.05$ (3 MUS methods per "
             r"classifier and metric). A bold 0.000 means that the median dataset is unchanged (absolute difference $\leq 0.0005$) while the non-zero differences are systematically signed; W/T/L counts and exact $p$-values are in the summary CSV.}",
             r"\label{tab:x6}", r"\begin{tabular}{llrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{G-mean} & \multicolumn{2}{c}{ROC-AUC}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Classifier & Method & original & dedup. & original & dedup.\\", r"\midrule"]
    for c in CLASSIFIERS:
        for i, s in enumerate(MUS3):
            cells = []
            for met in ("gmean", "auc"):
                for lab in ("original", "deduplicated"):
                    r = t[(t.classifier == c) & (t.method == s) & (t.metric == met) & (t.data == lab)]
                    cells.append(fmt_cell(r["median_diff"].iloc[0], r["p_holm"].iloc[0]) if len(r) else "--")
            lines.append((c if i == 0 else "") + " & " + s + " & " + " & ".join(cells) + r"\\")
        if c != CLASSIFIERS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- orchestration
def run_all(run_dir, main_path, dup_counts_path, out_dir=".", n_folds_x1=15, experiments=("x1", "x2", "x3", "x4", "x5", "x6")):
    """Collect per-fold results, write per-fold parquets, summary CSVs and LaTeX tables; return results."""
    main = pd.read_parquet(main_path)
    main = main[main["classifier"].isin(CLASSIFIERS)]
    dup = pd.read_csv(dup_counts_path)
    R, D = {}, {}
    for e in experiments:
        D[e] = read_exp(run_dir, e)
    if "x1" in experiments and len(D["x1"]):
        D["x1"].to_parquet(os.path.join(out_dir, "x1_perfold.parquet"))
        g = read_exp(run_dir, "x1", extra=True)
        g.to_parquet(os.path.join(out_dir, "x1_inner_grid.parquet"))
        r = analyse_x1(D["x1"], main)
        R["x1"] = r
        r["tests"].assign(section="MUS_vs_RUS_same_variant").to_csv(os.path.join(out_dir, "x1_summary.csv"), index=False)
        r["ranks"].to_csv(os.path.join(out_dir, "x1_ranks.csv"), index=False)
        r["variant_tests"].to_csv(os.path.join(out_dir, "x1_variant_effects.csv"), index=False)
        r["choices"].to_csv(os.path.join(out_dir, "x1_choices.csv"), index=False)
        json.dump(r["sanity"], open(os.path.join(out_dir, "x1_sanity.json"), "w"), indent=1)
        open(os.path.join(out_dir, "x1_table.tex"), "w").write(latex_x1(r, n_folds_x1))
    if "x2" in experiments and len(D["x2"]):
        D["x2"].to_parquet(os.path.join(out_dir, "x2_perfold.parquet"))
        r = analyse_x2(D["x2"], main)
        R["x2"] = r
        out = pd.concat([r["tests"].assign(section="tests"), r["jaccard"].assign(section="jaccard")], ignore_index=True)
        out.to_csv(os.path.join(out_dir, "x2_summary.csv"), index=False)
    if "x3" in experiments and len(D["x3"]):
        D["x3"].to_parquet(os.path.join(out_dir, "x3_perfold.parquet"))
        r = analyse_x3(D["x3"], main)
        R["x3"] = r
        out = pd.concat([r["tests"].assign(section="tests"), r["flags"].assign(section="flags")], ignore_index=True)
        out.to_csv(os.path.join(out_dir, "x3_summary.csv"), index=False)
        open(os.path.join(out_dir, "x3_table.tex"), "w").write(latex_x3(r))
    if "x4" in experiments and len(D["x4"]):
        D["x4"].to_parquet(os.path.join(out_dir, "x4_perfold.parquet"))
        r = analyse_x4(D["x4"], main)
        R["x4"] = r
        out = pd.concat([r["tests"].assign(section="tests"), r["diag_tests"].assign(section="coverage_diagnostics")],
                        ignore_index=True)
        out.to_csv(os.path.join(out_dir, "x4_summary.csv"), index=False)
    if "x5" in experiments and len(D["x5"]):
        D["x5"].to_parquet(os.path.join(out_dir, "x5_perfold.parquet"))
        r = analyse_x5(D["x5"], main, D.get("x4", pd.DataFrame()))
        R["x5"] = r
        out = pd.concat([r["tests"].assign(section="tests"), r["sd"].assign(section="median_sd_single_subset")],
                        ignore_index=True)
        out.to_csv(os.path.join(out_dir, "x5_summary.csv"), index=False)
    if "x4" in R and "x5" in R:
        open(os.path.join(out_dir, "x4x5_tables.tex"), "w").write(latex_x4x5(R["x4"], R["x5"]))
    if "x6" in experiments and len(D["x6"]):
        D["x6"].to_parquet(os.path.join(out_dir, "x6_perfold.parquet"))
        r = analyse_x6(D["x6"], main, dup)
        R["x6"] = r
        out = pd.concat([r["tests"].assign(section="tests"), r["ranks"].assign(section="avg_ranks_dedup")],
                        ignore_index=True)
        out.to_csv(os.path.join(out_dir, "x6_summary.csv"), index=False)
        open(os.path.join(out_dir, "x6_table.tex"), "w").write(latex_x6(r, dup))
    return R, D


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    run_all(cfg["out_dir"], cfg["main_results_path"], cfg["dup_counts_path"], cfg.get("analysis_out", "."),
            cfg["experiments"]["x1"]["n_folds"])
