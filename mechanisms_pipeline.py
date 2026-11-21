"""Selection diagnostics and dataset-level meta-analysis for matching undersampling.

End-to-end, re-runnable from input file paths:

    python mechanisms_pipeline.py --results results.parquet --selections selections.joblib \
        --data benchmark_datasets.joblib --folds cv_folds.joblib --registry dataset_registry.csv \
        --complexity dataset_complexity.csv --out out_dir [--workers 6]

Stage 1 (diagnostics) reproduces the benchmark preprocessing of every training fold
(median imputation + MinMax[0,1] fitted on the training fold, Euclidean distances) and
computes, for every (dataset, repeat, fold, selection), with P = minority, N = all
majority and S = retained majority (a multiset: ClusterCentroids hard voting can return
the same instance twice):
  BP    boundary proximity  = mean_{z in S} d(z,P) / mean_{z in N} d(z,P)
  CE    coverage error      = mean_{z in N} d(z,S) / mean_{z in N} d(z, N minus z)
  ED_SN, ED_SP              energy distance (metric form sqrt(2A-B-C), V-statistics,
                            computed exactly on all points) between S and N, S and P
  W1_SP                     exact 1-Wasserstein distance between S and P when |S| = |P|
                            (min-cost perfect matching / |P|); W1_opt = min over all
                            |P|-subsets of N (rectangular assignment) for reference
  enrich                    unsafe-majority enrichment = share of S whose 5 nearest
                            training-fold neighbours contain <= 2 majority instances,
                            divided by the same share in N
  iso_enrich                isolated-majority retention = share of S among the 10% majority
                            instances farthest from their nearest majority neighbour, / 0.10
  size_ratio                |S| / |P|
NONE (no resampling) enters as the identity selection S = N.

Stage 2 (analysis) writes the summary tables, figures and the report numbers.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

UNDERSAMPLERS = ["RUS", "NM1", "NM2", "NM3", "TL", "ENN", "NCR", "OSS", "CNN", "IHT", "CC",
                 "MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
SELECTIONS = ["NONE"] + UNDERSAMPLERS            # NONE = identity selection S = N
MUS = ["MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
NON_MUS_US = [s for s in UNDERSAMPLERS if s not in MUS]
BALANCED = ["RUS", "NM1", "NM2", "NM3", "CC"] + MUS   # target |S| = |P| by design
CLASSIFIERS = ["CART", "kNN", "SVM", "LR", "RF"]
K_UNSAFE = 5          # neighbourhood size of the unsafe-majority definition
ISO_FRAC = 0.10       # isolated majority = top 10% by distance to the nearest other majority instance
UNSAFE_MAX_MAJ = 2    # unsafe if <= 2 of the 5 nearest neighbours are majority
BLOCK = 512


# --------------------------------------------------------------------------- #
# Stage 1: selection diagnostics
# --------------------------------------------------------------------------- #
def scale_training_fold(X, tr):
    """Benchmark preprocessing: median imputation + MinMax[0,1], fitted on the training fold."""
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sc = MinMaxScaler()
    return sc.fit_transform(imp.fit_transform(X[tr]))


def _block_sum(D, rows, cols):
    tot = 0.0
    for b in range(0, len(rows), BLOCK):
        tot += float(D[np.ix_(rows[b:b + BLOCK], cols)].sum())
    return tot


def fold_diagnostics(Xtr, ytr, tr, selections, with_w1_opt=True):
    """Diagnostics of every selection on one training fold.

    Xtr, ytr : scaled training fold and labels (1 = minority); tr : original row indices of
    the training fold; selections : dict sampler -> original row indices of the retained
    majority (duplicates allowed). Returns a list of dict rows (one per selection)."""
    pos_min = np.flatnonzero(ytr == 1)
    pos_maj = np.flatnonzero(ytr == 0)
    P, N = Xtr[pos_min], Xtr[pos_maj]
    p, n = len(P), len(N)
    to_pos = np.full(int(tr.max()) + 1, -1, dtype=np.int64)
    to_pos[tr[pos_maj]] = np.arange(n)

    D_NN = cdist(N, N)
    D_NP = cdist(N, P)
    D_PP = cdist(P, P)
    rs_NN = D_NN.sum(axis=1)
    tot_NN = float(rs_NN.sum())
    rs_NP = D_NP.sum(axis=1)
    mean_PP = float(D_PP.mean())
    mean_NN = tot_NN / n ** 2
    dP = D_NP.min(axis=1)                      # d(z, P) for every majority z
    np.fill_diagonal(D_NN, np.inf)
    dN1 = D_NN.min(axis=1)                     # nearest other majority instance
    nmaj5 = np.empty(n, dtype=np.int64)        # majority count among the 5 NN in the fold
    for b in range(0, n, BLOCK):
        blk = np.hstack([D_NN[b:b + BLOCK], D_NP[b:b + BLOCK]])
        nn = np.argpartition(blk, K_UNSAFE - 1, axis=1)[:, :K_UNSAFE]
        nmaj5[b:b + BLOCK] = (nn < n).sum(axis=1)
    np.fill_diagonal(D_NN, 0.0)
    unsafe = nmaj5 <= UNSAFE_MAX_MAJ
    share_N = float(unsafe.mean())
    # isolated majority: the ceil(10%) majority instances farthest from their nearest majority neighbour
    n_iso = int(np.ceil(ISO_FRAC * n))
    iso = np.zeros(n, dtype=bool)
    iso[np.lexsort((np.arange(n), -dN1))[:n_iso]] = True
    iso_share_N = n_iso / n
    ed_NP = float(np.sqrt(max(0.0, 2 * rs_NP.sum() / (n * p) - mean_NN - mean_PP)))
    w1_opt = np.nan
    if with_w1_opt and n >= p:
        r, c = linear_sum_assignment(D_NP)
        w1_opt = float(D_NP[r, c].sum() / p)

    common = dict(n_min=p, n_maj=n, mean_dP_N=float(dP.mean()), mean_dN1=float(dN1.mean()),
                  unsafe_share_N=share_N, ED_NP=ed_NP, W1_opt=w1_opt)
    rows = []
    sels = dict(selections)
    sels.setdefault("NONE", tr[pos_maj])
    for smp, kept in sels.items():
        s = to_pos[np.asarray(kept, dtype=np.int64)]
        if (s < 0).any():
            raise ValueError(f"{smp}: retained index outside the training-fold majority")
        mS = len(s)
        row = dict(common, sampler=smp, n_sel=mS)
        if mS == 0:
            rows.append(row)
            continue
        su = np.unique(s)
        dup = len(su) < mS
        mask = np.zeros(n, dtype=bool)
        mask[su] = True
        R = np.flatnonzero(~mask)
        dS = np.zeros(n)
        for b in range(0, len(R), BLOCK):
            rb = R[b:b + BLOCK]
            dS[rb] = D_NN[np.ix_(rb, su)].min(axis=1)
        if dup or mS <= n // 2:
            B = _block_sum(D_NN, s, s)
        else:   # complement identity, S a set: sum_SS = sum_NN - 2 sum_RN + sum_RR
            B = tot_NN - 2 * float(rs_NN[R].sum()) + _block_sum(D_NN, R, R)
        mean_SS = B / mS ** 2
        A_SN = float(rs_NN[s].sum()) / (mS * n)
        A_SP = float(rs_NP[s].sum()) / (mS * p)
        w1 = np.nan
        if mS == p:
            Dsp = D_NP[s]
            r, c = linear_sum_assignment(Dsp)
            w1 = float(Dsp[r, c].sum() / p)
        row.update(
            n_unique=len(su), size_ratio=mS / p,
            BP=float(dP[s].mean() / dP.mean()),
            CE=float(dS.mean() / dN1.mean()) if dN1.mean() > 0 else np.nan,
            ED_SN=float(np.sqrt(max(0.0, 2 * A_SN - mean_SS - mean_NN))),
            ED_SP=float(np.sqrt(max(0.0, 2 * A_SP - mean_SS - mean_PP))),
            W1_SP=w1,
            unsafe_share_S=float(unsafe[s].mean()),
            enrich=float(unsafe[s].mean() / share_N) if share_N > 0 else np.nan,
            iso_enrich=float(iso[s].mean() / iso_share_N),
        )
        rows.append(row)
    return rows


def diagnostics_for_dataset(name, data, folds, sel_index, samplers=SELECTIONS):
    """All folds of one dataset. sel_index: dict (dataset, repeat, fold, sampler) -> kept idx."""
    D = data[name]
    X = np.asarray(D["X"], dtype=float)
    y = np.asarray(D["y"]).astype(int)
    out = []
    for f in folds[name]["folds"]:
        tr = np.asarray(f["train_idx"])
        rep, fo = int(f["repeat"]), int(f["fold"])
        Xtr = scale_training_fold(X, tr)
        sels = {}
        for smp in samplers:
            key = (name, rep, fo, smp)
            if smp == "NONE":
                continue
            if key in sel_index:
                sels[smp] = sel_index[key]
        t0 = time.perf_counter()
        rows = fold_diagnostics(Xtr, y[tr], tr, sels)
        dt = time.perf_counter() - t0
        for r in rows:
            r.update(dataset=name, repeat=rep, fold=fo, t_diag_fold=dt)
        out.extend(rows)
    return pd.DataFrame(out)


def load_selection_index(sel_path, datasets=None):
    sel = joblib.load(sel_path)
    idx = {}
    for r in sel:
        if datasets is not None and r["dataset"] not in datasets:
            continue
        key = (r["dataset"], int(r["repeat"]), int(r["fold"]), r["sampler"])
        if key in idx:
            raise ValueError(f"duplicate selection record {key}")
        idx[key] = np.asarray(r["kept_majority"], dtype=np.int64)
    return idx


def diag_worker(data_path, folds_path, sel_path, datasets, out_path):
    data = joblib.load(data_path)
    folds = joblib.load(folds_path)
    sel_index = load_selection_index(sel_path, set(datasets))
    parts = []
    for name in datasets:
        t0 = time.time()
        parts.append(diagnostics_for_dataset(name, data, folds, sel_index))
        print(f"{name}: {time.time() - t0:.1f}s", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(out_path)


def run_diagnostics(data_path, folds_path, sel_path, out_dir, workers=6, python_exe=None):
    """Stage 1 with `workers` independent subprocesses (the sandbox has no process pools)."""
    data = joblib.load(data_path)
    names = sorted(data)
    cost = {k: float((np.asarray(data[k]["y"]) == 0).sum()) ** 2 for k in names}
    shards = [[] for _ in range(workers)]
    load = np.zeros(workers)
    for k in sorted(names, key=lambda z: -cost[z]):        # greedy longest-processing-time
        i = int(np.argmin(load))
        shards[i].append(k)
        load[i] += cost[k]
    tmp = os.path.join(out_dir, "_diag_parts")
    os.makedirs(tmp, exist_ok=True)
    for f in glob.glob(os.path.join(tmp, "*")):
        os.remove(f)
    python_exe = python_exe or sys.executable
    procs = []
    for i, sh in enumerate(shards):
        if not sh:
            continue
        cfg = os.path.join(tmp, f"shard{i:02d}.json")
        json.dump(dict(data=data_path, folds=folds_path, selections=sel_path, datasets=sh,
                       out=os.path.join(tmp, f"part{i:02d}.parquet")), open(cfg, "w"))
        fh = open(os.path.join(tmp, f"shard{i:02d}.log"), "w")
        procs.append((subprocess.Popen([python_exe, os.path.abspath(__file__), "diag-worker", cfg],
                                       stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ)), fh))
    codes = []
    for p, fh in procs:
        codes.append(p.wait())
        fh.close()
    if any(codes):
        raise RuntimeError(f"diagnostics workers failed: exit codes {codes}; see {tmp}")
    diag = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(tmp, "part*.parquet")))],
                     ignore_index=True)
    first = ["dataset", "repeat", "fold", "sampler"]
    diag = diag[first + [c for c in diag.columns if c not in first]]
    diag = diag.sort_values(first).reset_index(drop=True)
    out = os.path.join(out_dir, "diag_selection.parquet")
    diag.to_parquet(out)
    return diag


# --------------------------------------------------------------------------- #
# Stage 2: analysis
# --------------------------------------------------------------------------- #
METRICS = ["gmean", "auc", "bacc", "sens", "spec", "f1", "mcc", "prauc", "prec"]
LINK_DIAGS = ["BP", "CE", "ED_SN", "ED_SP", "enrich", "iso_enrich", "size_ratio"]
SUMMARY_DIAGS = ["size_ratio", "BP", "CE", "ED_SN", "ED_SP", "ED_SP_ratio", "W1_ratio", "enrich", "iso_enrich"]
COVERAGE_CLUSTER = ["RUS", "CC", "MUS-CS", "MUS-CO"]
BOUNDARY_CLUSTER = ["NM1", "NM2", "NM3", "MUS-BS", "MUS-BO"]
COVARIATES = ["IR", "n", "d", "n_min", "F1", "N1", "N2", "N3", "LSC", "safe", "borderline", "rare",
              "outlier", "maj_overlap_share"]
COV_LABEL = {"IR": "imbalance ratio (IR)", "n": "number of instances (n)", "d": "number of features (d)",
             "n_min": "minority size", "F1": "feature overlap F1 (higher = harder)",
             "N1": "boundary-point fraction N1", "N2": "NN distance ratio N2",
             "N3": "1-NN error rate N3", "LSC": "local-set cardinality LSC",
             "safe": "safe minority share", "borderline": "borderline minority share",
             "rare": "rare minority share", "outlier": "outlier minority share",
             "maj_overlap_share": "overlapping majority share"}
COV_GROUP = {"IR": "size", "n": "size", "d": "size", "n_min": "size", "F1": "feature overlap",
             "N1": "neighbourhood", "N2": "neighbourhood", "N3": "neighbourhood", "LSC": "neighbourhood",
             "safe": "typology", "borderline": "typology", "rare": "typology", "outlier": "typology",
             "maj_overlap_share": "typology"}
LOG_COVS = {"IR", "n", "d", "n_min"}
DIAG_LABEL = {"BP": "boundary proximity", "CE": "coverage error", "ED_SN": "energy distance to the full majority",
              "ED_SP": "energy distance to the minority", "enrich": "unsafe-majority enrichment",
              "iso_enrich": "isolated-majority retention", "size_ratio": "retained size", "W1_SP": "W1 to the minority"}
PAIRS = [("MUS-CS", "MUS-CO"), ("MUS-BS", "MUS-BO"), ("MUS-BS", "NM3"), ("MUS-CS", "RUS"), ("MUS-CS", "CC")]
GROUP = {s: ("matching (proposed)" if s in MUS else "size-balancing" if s in BALANCED else
             "no resampling" if s == "NONE" else "cleaning / partial") for s in SELECTIONS}
MUS_COLOR = {"MUS-CS": "#1f6fb4", "MUS-CO": "#0b3d6b", "MUS-BS": "#d9711c", "MUS-BO": "#8a3f00"}
STYLE_FN = None          # set to the figure-style skill's apply_figure_style when run in the kernel
STYLE_USED = None


def apply_style():
    """Publication rcParams: the figure-style skill's apply_figure_style() when available
    (STYLE_FN or env FIGURE_STYLE_KERNEL = path to the skill's kernel.py), else the same rc set."""
    global STYLE_USED
    import matplotlib
    matplotlib.use("Agg")
    fn = STYLE_FN
    p = os.environ.get("FIGURE_STYLE_KERNEL")
    if fn is None and p and os.path.exists(p):
        import importlib.util
        spec = importlib.util.spec_from_file_location("figure_style_kernel", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "apply_figure_style", None)
    if fn is not None:
        fn()
        STYLE_USED = "figure-style skill apply_figure_style()"
        return
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "legend.frameon": False, "figure.dpi": 200,
        "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlelocation": "left",
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42})
    STYLE_USED = "embedded copy of the figure-style rcParams"


def _q(v, q):
    v = pd.Series(v).dropna()
    return float(v.quantile(q)) if len(v) else np.nan


def load_results(results_path):
    res = pd.read_parquet(results_path)
    if "tag" in res.columns:
        res = res[res["tag"] == "main"]
    return res


def performance_table(res):
    g = res.groupby(["dataset", "sampler", "classifier"])
    perf = g[METRICS].mean()
    perf["n_folds"] = g["gmean"].size()
    return perf.reset_index()


def average_ranks(perf, samplers, metric="gmean", classifiers=CLASSIFIERS):
    """Rank `samplers` within each (dataset, classifier) block by mean metric (1 = best,
    ties averaged); average over all complete blocks."""
    sub = perf[perf["sampler"].isin(samplers) & perf["classifier"].isin(classifiers)]
    wide = sub.pivot_table(index=["dataset", "classifier"], columns="sampler", values=metric)
    wide = wide.dropna(axis=0, how="any")
    rk = wide.rank(axis=1, ascending=False, method="average")
    return rk.mean().sort_values(), len(wide)


def dataset_level_diagnostics(diag):
    d = diag.copy()
    # W1(S,P) is reported only for the size-balancing methods (|S| = |P| by design); for the
    # cleaning methods |S| = |P| happens only by coincidence in a few folds.
    d.loc[~d["sampler"].isin(BALANCED), "W1_SP"] = np.nan
    d["W1_ratio"] = d["W1_SP"] / d["W1_opt"]
    d["ED_SP_ratio"] = d["ED_SP"] / d["ED_NP"]
    cols = ["n_sel", "size_ratio", "BP", "CE", "ED_SN", "ED_SP", "ED_NP", "ED_SP_ratio", "W1_SP", "W1_opt",
            "W1_ratio", "unsafe_share_S", "unsafe_share_N", "enrich", "iso_enrich", "n_maj", "n_min"]
    g = d.groupby(["dataset", "sampler"])
    dd = g[cols].mean()
    dd["enrich_folds"] = g["enrich"].count()
    return dd.reset_index()


def diag_summary_table(dd, perf, ranks, n_blocks):
    present = [s for s in SELECTIONS if s in set(dd["sampler"])]
    pa = perf.groupby(["dataset", "sampler"])[["gmean", "auc", "sens", "spec"]].mean().reset_index()
    rus = dd[dd["sampler"] == "RUS"].set_index("dataset")
    rows = []
    for s in present:
        x = dd[dd["sampler"] == s].set_index("dataset")
        r = dict(sampler=s, group=GROUP[s], n_datasets=len(x))
        for c in SUMMARY_DIAGS:
            v = x[c].dropna()
            r[f"{c}_median"] = _q(v, 0.5)
            r[f"{c}_q1"] = _q(v, 0.25)
            r[f"{c}_q3"] = _q(v, 0.75)
            r[f"{c}_ndatasets"] = int(len(v))
        for c in ("BP", "CE", "ED_SN", "ED_SP", "enrich", "iso_enrich"):
            both = pd.concat([x[c], rus[c]], axis=1, keys=["s", "r"]).dropna()
            r[f"{c}_lt_RUS"] = int((both["s"] < both["r"]).sum()) if s != "RUS" else np.nan
            r[f"{c}_cmp_RUS_n"] = int(len(both)) if s != "RUS" else np.nan
        p = pa[pa["sampler"] == s]
        for c in ("gmean", "auc", "sens", "spec"):
            r[f"{c}_clfavg_median"] = _q(p[c], 0.5)
        r["gmean_avg_rank"] = float(ranks.get(s, np.nan))
        r["rank_blocks"] = n_blocks
        rows.append(r)
    return pd.DataFrame(rows)


def diag_summary_tex(summ, path, caption_note=""):
    cols = [("size_ratio", r"$|S|/|P|$"), ("BP", "BP"), ("CE", "CE"), ("ED_SN", r"$\mathcal{E}(S,N)$"),
            ("ED_SP", r"$\mathcal{E}(S,P)$"), ("W1_ratio", r"$W_1(S,P)/W_1^{*}$"), ("enrich", "Unsafe enr."),
            ("iso_enrich", "Isolated enr.")]
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{Selection diagnostics per undersampler on the training folds: median over the "
             r"datasets [first--third quartile] of the fold-averaged values. BP = boundary proximity "
             r"($<1$: retained majority closer to the minority than the full majority); CE = coverage error "
             r"(0 when nothing is removed); $\mathcal{E}$ = energy distance; $W_1(S,P)/W_1^{*}$ = "
             r"Wasserstein-1 distance to the minority relative to its minimum over all $|P|$-subsets "
             r"(defined when $|S|=|P|$); unsafe enr. = unsafe-majority enrichment; isolated enr. = retention of the 10\% "
             r"most isolated majority instances relative to their share. Rank = average G-mean rank "
             r"among the " + str(int(summ["sampler"].nunique())) + r" selection methods, pooled over five "
             r"classifiers." + caption_note + "}",
             r"\label{tab:diag_summary}", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", r"\toprule",
             "Method & " + " & ".join(c[1] for c in cols) + r" & Rank \\", r"\midrule"]
    order = [s for s in SELECTIONS if s in set(summ["sampler"])]
    for s in order:
        r = summ.set_index("sampler").loc[s]
        cells = []
        for c, _ in cols:
            m, a, b = r[f"{c}_median"], r[f"{c}_q1"], r[f"{c}_q3"]
            cells.append("--" if not np.isfinite(m) else f"{m:.2f} [{a:.2f}, {b:.2f}]")
        name = r"\textbf{" + s + "}" if s in MUS else ("None" if s == "NONE" else s)
        lines.append(name + " & " + " & ".join(cells) + f" & {r['gmean_avg_rank']:.2f}" + r" \\")
        if s in ("NONE", "IHT", "CC"):
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def diag_performance_link(dd, perf, present):
    """Within-dataset Spearman correlation across selection methods between each fold-averaged
    diagnostic and the mean G-mean (per classifier and classifier-averaged)."""
    from scipy.stats import spearmanr, wilcoxon
    from statsmodels.stats.multitest import multipletests
    pa = perf.groupby(["dataset", "sampler"])["gmean"].mean().reset_index().assign(classifier="mean")
    pp = pd.concat([perf[["dataset", "sampler", "classifier", "gmean"]], pa], ignore_index=True)
    sets = {"all_selections": [s for s in SELECTIONS if s in present],
            "size_balanced": [s for s in BALANCED if s in present],
            "coverage_cluster": [s for s in COVERAGE_CLUSTER if s in present],
            "boundary_cluster": [s for s in BOUNDARY_CLUSTER if s in present]}
    per_ds, summ = [], []
    for setname, smps in sets.items():
        diags = LINK_DIAGS if setname == "all_selections" else [c for c in LINK_DIAGS if c != "size_ratio"] + ["W1_SP"]
        min_methods = 5 if setname in ("all_selections", "size_balanced") else 3
        for clf in CLASSIFIERS + ["mean"]:
            yall = pp[pp["classifier"] == clf].set_index(["dataset", "sampler"])["gmean"]
            block = []
            for dg in diags:
                rhos = []
                for ds, x in dd[dd["sampler"].isin(smps)].groupby("dataset"):
                    xv = x.set_index("sampler")[dg]
                    yv = yall.loc[ds]
                    j = pd.concat([xv, yv], axis=1, keys=["x", "y"]).dropna()
                    rho = np.nan
                    if len(j) >= min_methods and j["x"].nunique() > 1 and j["y"].nunique() > 1:
                        rho = float(spearmanr(j["x"], j["y"]).statistic)
                    rhos.append(rho)
                    per_ds.append(dict(set=setname, classifier=clf, diagnostic=dg, dataset=ds, rho=rho,
                                       n_methods=len(j)))
                v = pd.Series(rhos).dropna()
                pw = float(wilcoxon(v).pvalue) if len(v) >= 6 and (v != 0).any() else np.nan
                block.append(dict(set=setname, n_methods=len(smps), classifier=clf, diagnostic=dg,
                                  n_datasets=len(v), rho_median=_q(v, .5), rho_q1=_q(v, .25), rho_q3=_q(v, .75),
                                  frac_pos=float((v > 0).mean()) if len(v) else np.nan,
                                  frac_neg=float((v < 0).mean()) if len(v) else np.nan,
                                  wilcoxon_p=pw))
            ps = [b["wilcoxon_p"] for b in block]
            ok = [i for i, p in enumerate(ps) if np.isfinite(p)]
            adj = np.full(len(ps), np.nan)
            if ok:
                adj[ok] = multipletests([ps[i] for i in ok], method="holm")[1]
            for b, a in zip(block, adj):
                b["wilcoxon_p_holm"] = float(a)
            summ.extend(block)
    return pd.DataFrame(summ), pd.DataFrame(per_ds)


def selection_overlap(sel_index, diag):
    """Jaccard similarity of retained-majority sets (set semantics) per (dataset, fold)."""
    nmaj = diag.drop_duplicates(["dataset", "repeat", "fold"]).set_index(["dataset", "repeat", "fold"])["n_maj"]
    units = sorted({k[:3] for k in sel_index})
    rows = []
    for u in units:
        for a, b in PAIRS:
            ka, kb = u + (a,), u + (b,)
            if ka not in sel_index or kb not in sel_index:
                continue
            A, B = set(sel_index[ka].tolist()), set(sel_index[kb].tolist())
            inter, union = len(A & B), len(A | B)
            n = float(nmaj.loc[u])
            e_int = len(A) * len(B) / n
            rows.append(dict(dataset=u[0], repeat=u[1], fold=u[2], pair=f"{a} vs {b}", jaccard=inter / union,
                             jaccard_chance=e_int / (len(A) + len(B) - e_int), size_a=len(A), size_b=len(B)))
    pf = pd.DataFrame(rows)
    summ = []
    by_ds = (pf.groupby(["pair", "dataset"])[["jaccard", "jaccard_chance"]].mean().reset_index()
             if len(pf) else pd.DataFrame(columns=["pair", "dataset", "jaccard", "jaccard_chance"]))
    for a, b in PAIRS:
        pr = f"{a} vs {b}"
        x = by_ds[by_ds["pair"] == pr]
        if x.empty:
            summ.append(dict(pair=pr, n_datasets=0, n_folds=0))
            continue
        summ.append(dict(pair=pr, n_datasets=len(x), n_folds=int((pf["pair"] == pr).sum()),
                         jaccard_median=_q(x["jaccard"], .5), jaccard_q1=_q(x["jaccard"], .25),
                         jaccard_q3=_q(x["jaccard"], .75), jaccard_min=float(x["jaccard"].min()),
                         jaccard_max=float(x["jaccard"].max()),
                         chance_median=_q(x["jaccard_chance"], .5),
                         frac_datasets_above_chance=float((x["jaccard"] > x["jaccard_chance"]).mean())))
    return pd.DataFrame(summ), by_ds


def seed_reference_overlap(data_path, folds_path, sel_index, ms_path, repeats=(0,), seed_offset=7919):
    """Reference for the coverage arm: Jaccard between the benchmark MUS-CS selection and MUS-CS re-run with a
    different random_state on the same training fold (the keep coin is the only difference). Also verifies that
    the benchmark selection is reproduced with the original seed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("matchsampling", ms_path)
    ms = importlib.util.module_from_spec(spec)
    sys.modules["matchsampling"] = ms
    spec.loader.exec_module(ms)
    data, folds = joblib.load(data_path), joblib.load(folds_path)
    rows = []
    for name in sorted(data):
        X = np.asarray(data[name]["X"], dtype=float)
        y = np.asarray(data[name]["y"]).astype(int)
        for k, f in enumerate(folds[name]["folds"]):
            if int(f["repeat"]) not in repeats:
                continue
            tr = np.asarray(f["train_idx"])
            seed = int(folds[name]["seed"]) * 100 + k          # benchmark seed rule
            Xtr, ytr = scale_training_fold(X, tr), y[tr]
            kept = []
            for rs in (seed, seed + seed_offset):
                smp = ms.MatchingUnderSampler(arm="coverage", concept="stable", random_state=rs)
                smp.fit_resample(Xtr, ytr)
                idx = smp.sample_indices_
                kept.append(np.sort(tr[idx[ytr[idx] == 0]]))
            rec = sel_index.get((name, int(f["repeat"]), int(f["fold"]), "MUS-CS"))
            A, B = set(kept[0].tolist()), set(kept[1].tolist())
            n = int((ytr == 0).sum())
            e = len(A) * len(B) / n
            rows.append(dict(dataset=name, repeat=int(f["repeat"]), fold=int(f["fold"]),
                             reproduces_record=bool(rec is not None and np.array_equal(kept[0], np.sort(rec))),
                             jaccard=len(A & B) / len(A | B), jaccard_chance=e / (len(A) + len(B) - e)))
    pf = pd.DataFrame(rows)
    by_ds = pf.groupby("dataset")[["jaccard", "jaccard_chance"]].mean().reset_index()
    summ = dict(pair="MUS-CS vs MUS-CS (other seed)", n_datasets=len(by_ds), n_folds=len(pf),
                jaccard_median=_q(by_ds["jaccard"], .5), jaccard_q1=_q(by_ds["jaccard"], .25),
                jaccard_q3=_q(by_ds["jaccard"], .75), jaccard_min=float(by_ds["jaccard"].min()),
                jaccard_max=float(by_ds["jaccard"].max()), chance_median=_q(by_ds["jaccard_chance"], .5),
                frac_datasets_above_chance=float((by_ds["jaccard"] > by_ds["jaccard_chance"]).mean()),
                reproduces_benchmark_share=float(pf["reproduces_record"].mean()),
                note=(f"first CV repeat only, {len(pf)} folds" if tuple(repeats) == (0,)
                      else f"CV repeats {list(repeats)}, {len(pf)} folds"))
    return summ, pf


def best_baselines(perf, metrics=("gmean", "auc")):
    rows = []
    for clf in CLASSIFIERS:
        for met in metrics:
            ar, nb = average_ranks(perf, NON_MUS_US, met, [clf])
            rows.append(dict(classifier=clf, metric=met, best=ar.index[0], best_avg_rank=float(ar.iloc[0]),
                             second=ar.index[1], second_avg_rank=float(ar.iloc[1]),
                             RUS_avg_rank=float(ar.get("RUS", np.nan)), n_datasets=nb,
                             ranking=";".join(f"{k}:{v:.2f}" for k, v in ar.items())))
    return pd.DataFrame(rows)


def gains_table(perf, bb, metrics=("gmean", "auc", "sens", "spec")):
    """Delta = mean metric(MUS-x) - mean metric(reference) per dataset and classifier.
    Reference 'best' = the non-matching undersampler with the best average rank for that classifier
    and metric; for sens/spec (decomposition of the G-mean gain) the G-mean-best baseline is used."""
    present = [m for m in MUS if m in set(perf["sampler"])]
    pv = perf.set_index(["dataset", "sampler", "classifier"])
    best = {(r.classifier, r.metric): r.best for r in bb.itertuples()}
    rows = []
    for ds in sorted(perf["dataset"].unique()):
        for clf in CLASSIFIERS:
            for met in metrics:
                ref_r = pv.loc[(ds, "RUS", clf), met]
                b = best[(clf, met if met in ("gmean", "auc") else "gmean")]
                ref_b = pv.loc[(ds, b, clf), met]
                for m in present:
                    v = pv.loc[(ds, m, clf), met]
                    rows.append(dict(dataset=ds, classifier=clf, metric=met, method=m, value=v,
                                     delta_rus=v - ref_r, delta_best=v - ref_b, best_baseline=b))
    g = pd.DataFrame(rows)
    ga = g.groupby(["dataset", "metric", "method"])[["value", "delta_rus", "delta_best"]].mean().reset_index()
    ga["classifier"] = "mean"
    ga["best_baseline"] = "per-classifier"
    return pd.concat([g, ga[g.columns]], ignore_index=True)


def gain_summary(gains):
    from scipy.stats import wilcoxon
    rows = []
    for (clf, met, m), x in gains.groupby(["classifier", "metric", "method"]):
        for ref in ("rus", "best"):
            v = x[f"delta_{ref}"].dropna()
            nz = v[v != 0]
            rows.append(dict(classifier=clf, metric=met, method=m, reference=ref, n_datasets=len(v),
                             delta_mean=float(v.mean()), delta_median=float(v.median()),
                             delta_q1=_q(v, .25), delta_q3=_q(v, .75), wins=int((v > 0).sum()),
                             ties=int((v == 0).sum()), losses=int((v < 0).sum()),
                             wilcoxon_p=float(wilcoxon(nz).pvalue) if len(nz) >= 6 else np.nan))
    return pd.DataFrame(rows)


def covariate_table(complexity_path, registry_path):
    cx = pd.read_csv(complexity_path).set_index("name")
    reg = pd.read_csv(registry_path)
    if "included" in reg.columns:
        reg = reg[reg["included"].astype(str).str.lower().isin(["true", "1", "yes"])]
    reg = reg.set_index("name")
    cov = cx[COVARIATES].astype(float).copy()
    cov["family"] = reg.reindex(cov.index)["family"]
    fr = reg.reindex(cov.index)["family_representative"]
    cov["family_representative"] = fr.astype(str).str.lower().isin(["true", "1", "yes"])
    return cov


def meta_correlations(gains, cov, metrics=("gmean", "auc")):
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests
    ga = gains[gains["classifier"] == "mean"]
    subsets = {"all": list(cov.index), "family_representative": list(cov.index[cov["family_representative"]])}
    rows = []
    for sub, dss in subsets.items():
        for met in metrics:
            for ref in ("rus", "best"):
                for m in [m for m in MUS if m in set(ga["method"])]:
                    y = ga[(ga["metric"] == met) & (ga["method"] == m)].set_index("dataset")[f"delta_{ref}"]
                    block = []
                    for c in COVARIATES:
                        j = pd.concat([cov.loc[dss, c], y.reindex(dss)], axis=1, keys=["x", "y"]).dropna()
                        r = spearmanr(j["x"], j["y"])
                        block.append(dict(subset=sub, metric=met, reference=ref, method=m, covariate=c,
                                          n=len(j), rho=float(r.statistic), p=float(r.pvalue)))
                    adj = multipletests([b["p"] for b in block], method="holm")[1]
                    for b, a in zip(block, adj):
                        b["p_holm"] = float(a)
                    rows.extend(block)
    return pd.DataFrame(rows)


def meta_corr_tex(mc, path, note=""):
    sub = mc[(mc["subset"] == "all") & (mc["metric"] == "gmean")]
    meths = [m for m in MUS if m in set(sub["method"])]
    refs = [("rus", "vs RUS"), ("best", "vs best baseline")]
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             r"\caption{Spearman correlation between dataset characteristics and the classifier-averaged "
             r"G-mean gain of each matching undersampler over random undersampling (RUS) and over the best "
             r"non-matching undersampler of each classifier (" + str(int(sub["n"].max())) + r" datasets). Bold: "
             r"Holm-adjusted $p<0.05$ across the " + str(len(COVARIATES)) + r" characteristics; "
             r"$^{\dagger}$: unadjusted $p<0.05$ only." + note + "}",
             r"\label{tab:meta_gain_corr}",
             r"\begin{tabular}{l" + "r" * (len(meths) * 2) + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{" + str(len(meths)) + r"}{c}{" + t + "}" for _, t in refs) + r" \\",
             " ".join(r"\cmidrule(lr){%d-%d}" % (2 + i * len(meths), 1 + (i + 1) * len(meths))
                      for i in range(len(refs))),
             "Characteristic & " + " & ".join(meths * len(refs)) + r" \\", r"\midrule"]
    for c in COVARIATES:
        cells = []
        for ref, _ in refs:
            for m in meths:
                r = sub[(sub["reference"] == ref) & (sub["method"] == m) & (sub["covariate"] == c)].iloc[0]
                t = f"{r['rho']:+.2f}"
                if r["p_holm"] < 0.05:
                    t = r"\textbf{" + t + "}"
                elif r["p"] < 0.05:
                    t = t + r"$^{\dagger}$"
                cells.append(t)
        lines.append(COV_LABEL[c] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def gain_trees(gains, cov, metric="gmean", ref="rus"):
    from sklearn.tree import DecisionTreeRegressor, export_text
    ga = gains[(gains["classifier"] == "mean") & (gains["metric"] == metric)]
    out, rows = [], []
    for m in [m for m in MUS if m in set(ga["method"])]:
        y = ga[ga["method"] == m].set_index("dataset")[f"delta_{ref}"].reindex(cov.index)
        X = cov[COVARIATES]
        ok = y.notna() & X.notna().all(axis=1)
        t = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, random_state=0).fit(X[ok], y[ok])
        r2 = float(t.score(X[ok], y[ok]))
        txt = export_text(t, feature_names=COVARIATES, decimals=3)
        leaf = t.apply(X[ok])
        tr_, rules = t.tree_, {}

        def walk(node, conds):
            if tr_.children_left[node] == -1:
                rules[node] = " and ".join(conds) if conds else "all"
                return
            f, th = COVARIATES[tr_.feature[node]], tr_.threshold[node]
            walk(tr_.children_left[node], conds + [f"{f} <= {th:.3f}"])
            walk(tr_.children_right[node], conds + [f"{f} > {th:.3f}"])
        walk(0, [])
        for lf in np.unique(leaf):
            yy = y[ok][leaf == lf]
            rows.append(dict(method=m, metric=metric, reference=ref, leaf=int(lf), rule=rules[int(lf)], n=len(yy),
                             mean_delta=float(yy.mean()), median_delta=float(yy.median()),
                             frac_positive=float((yy > 0).mean()), r2_in_sample=r2))
        out.append(f"### {m}: depth-2 regression tree of classifier-averaged delta {metric} vs {ref.upper()} "
                   f"(n={int(ok.sum())}, min_samples_leaf=5, in-sample R^2={r2:.3f})\n{txt}")
    return "\n".join(out), pd.DataFrame(rows)


def runtime_table(res):
    from scipy import stats
    u = res.drop_duplicates(["dataset", "repeat", "fold", "sampler"])
    u = u[~u["sampler"].isin(["NONE", "CW"])]
    per_ds = u.groupby(["sampler", "dataset"]).agg(t_med=("t_resample", "median"),
                                                  n_maj=("n_maj_train", "median"),
                                                  n_min=("n_min_train", "median")).reset_index()
    rows = []
    for s, x in u.groupby("sampler"):
        pdx = per_ds[per_ds["sampler"] == s]
        lt, ln = np.log10(pdx["t_med"].clip(lower=1e-6)), np.log10(pdx["n_maj"])
        lr = stats.linregress(ln, lt)
        tcrit = stats.t.ppf(0.975, len(pdx) - 2)
        rows.append(dict(sampler=s, n_units=len(x), t_median=float(x["t_resample"].median()),
                         t_p95=float(x["t_resample"].quantile(0.95)), t_max=float(x["t_resample"].max()),
                         t_total=float(x["t_resample"].sum()), slope=float(lr.slope),
                         slope_ci_lo=float(lr.slope - tcrit * lr.stderr),
                         slope_ci_hi=float(lr.slope + tcrit * lr.stderr),
                         intercept=float(lr.intercept), r2=float(lr.rvalue ** 2), n_datasets=len(pdx)))
    rt = pd.DataFrame(rows).sort_values("t_median").reset_index(drop=True)
    return rt, per_ds


def _sig(v):
    """Seconds with three significant digits, no scientific notation."""
    if not np.isfinite(v):
        return "--"
    if v >= 100:
        return f"{v:,.0f}".replace(",", "{,}")
    if v >= 1:
        return f"{v:.3g}"
    return f"{v:.3g}" if v >= 1e-3 else f"{v:.1e}"


def runtime_tex(rt, path):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Resampling time per training fold in seconds (" + str(int(rt["n_units"].max())) +
             r" folds per method, measured while several benchmark processes ran in parallel) and the "
             r"log--log slope of the per-dataset median time against the majority size of the training fold "
             r"[95\% confidence interval].}", r"\label{tab:runtime}",
             r"\begin{tabular}{lrrrc}", r"\toprule",
             r"Method & Median & 95th pct. & Max. & Slope [95\% CI] \\", r"\midrule"]
    for r in rt.itertuples():
        name = r"\textbf{" + r.sampler + "}" if r.sampler in MUS else r.sampler
        lines.append(f"{name} & {_sig(r.t_median)} & {_sig(r.t_p95)} & {_sig(r.t_max)} & "
                     f"{r.slope:.2f} [{r.slope_ci_lo:.2f}, {r.slope_ci_hi:.2f}]" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def overlap_tex(ov, path):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Jaccard similarity of the retained majority sets of two methods on the same training "
             r"fold: median over datasets [first--third quartile] of the fold-averaged similarity, and the "
             r"median similarity expected for two independent uniformly random subsets of the same sizes.}",
             r"\label{tab:overlap}", r"\begin{tabular}{lccc}", r"\toprule",
             r"Pair & Jaccard & Chance & Datasets \\", r"\midrule"]
    for r in ov.to_dict("records"):
        if not r.get("n_datasets") or not np.isfinite(r.get("jaccard_median", np.nan)):
            lines.append(f"{r['pair']} & -- & -- & 0" + r" \\")
            continue
        mark = r"$^{a}$" if isinstance(r.get("note"), str) else ""
        lines.append(f"{r['pair']}{mark} & {r['jaccard_median']:.2f} [{r['jaccard_q1']:.2f}, {r['jaccard_q3']:.2f}] & "
                     f"{r['chance_median']:.2f} & {r['n_datasets']}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    refs = [r for r in ov.to_dict("records") if isinstance(r.get("note"), str)]
    if refs:
        lines += [r"\par\smallskip{\scriptsize $^{a}$MUS-CS re-run with a different random seed on the same training folds "
                  r"(" + str(int(refs[0]["n_folds"])) + r" folds of the first CV repeat): the kept member of each matched "
                  r"pair is a coin flip, so instance-level overlap between two runs of the same method is at chance level.}"]
    lines += [r"\end{table}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
# label offsets (points) for the desiderata map; chosen to keep labels off the neighbouring marks
MAP_OFFSETS = {"NONE": (-16, -10), "TL": (34, -6), "ENN": (34, 7), "NCR": (34, 20), "OSS": (-22, 6),
               "IHT": (16, 0), "CC": (16, -6), "RUS": (-26, 6), "MUS-CS": (18, 2), "MUS-CO": (18, 12),
               "CNN": (-26, 6), "NM1": (0, 11), "NM2": (12, 6), "NM3": (12, 7),
               "MUS-BS": (-16, -16), "MUS-BO": (16, -16)}
MERGED_SEP = {"MUS-CS": " /\n"}
MERGED_OFFSETS = {"NONE": (34, -9), "TL": (34, -9), "ENN": (34, 10), "MUS-BS": (0, -20), "MUS-CS": (18, 12),
                  "NM1": (0, 12), "NM3": (12, 8)}


def _save(fig, out_dir, stem):
    pdf, png = os.path.join(out_dir, stem + ".pdf"), os.path.join(out_dir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    return pdf, png


def fig_desiderata_map(summ, out_dir, stem="fig_desiderata_map"):
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.lines import Line2D
    apply_style()
    s = summ.set_index("sampler")
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    vals = s["gmean_avg_rank"]
    norm = colors.Normalize(vmin=np.floor(vals.min()), vmax=np.ceil(vals.max()))
    cmap = plt.get_cmap("viridis")
    xlo, xhi = s["BP_median"].min(), s["BP_median"].max()
    ylo, yhi = s["CE_median"].min(), s["CE_median"].max()
    ax.set_xlim(xlo - 0.06 * (xhi - xlo), xhi + 0.30 * (xhi - xlo))
    ax.set_ylim(ylo - 0.08 * (yhi - ylo), yhi + 0.10 * (yhi - ylo))
    ax.axvline(1.0, color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=0)
    ax.text(1.004, 0.995, "no boundary\npreference (BP = 1)", transform=ax.get_xaxis_transform(), ha="left",
            va="top", fontsize=plt.rcParams["legend.fontsize"], color="#666666")
    marker = {"matching (proposed)": "o", "size-balancing": "o", "cleaning / partial": "s", "no resampling": "D"}
    # merge labels of (nearly) coincident points
    xr, yr = ax.get_xlim(), ax.get_ylim()
    pos = {k: ((r["BP_median"] - xr[0]) / (xr[1] - xr[0]), (r["CE_median"] - yr[0]) / (yr[1] - yr[0]))
           for k, r in s.iterrows()}
    merged, skip = {}, set()
    keys = list(s.index)
    for a in keys:
        if a in skip:
            continue
        group = [a] + [b for b in keys if b != a and b not in skip and
                       np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]) < 0.015]
        skip.update(group)
        merged[a] = group
    for smp, r in s.iterrows():
        is_mus = smp in MUS
        ax.scatter([r["BP_median"]], [r["CE_median"]], s=95 if is_mus else 38, marker=marker[r["group"]],
                   c=[cmap(norm(r["gmean_avg_rank"]))], edgecolors="black", linewidths=1.4 if is_mus else 0.5,
                   zorder=4 if is_mus else 3)
    for lead, group in merged.items():
        r = s.loc[lead]
        x, y = r["BP_median"], r["CE_median"]
        is_mus = any(g in MUS for g in group)
        dx, dy = MAP_OFFSETS.get(lead, (12, 4))
        if len(group) > 1:
            dx, dy = MERGED_OFFSETS.get(lead, (dx, dy))
        label = (MERGED_SEP.get(lead, " / ") if len(group) > 1 else "").join(
            "None" if g == "NONE" else g for g in group)
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=plt.rcParams["font.size"] if is_mus else plt.rcParams["legend.fontsize"],
                    fontweight="bold" if is_mus else "normal",
                    ha="center" if dx == 0 else "left" if dx > 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#777777", shrinkA=0, shrinkB=4)
                    if (abs(dx) > 14 or abs(dy) > 10) else None, zorder=5)
    ax.set_xlabel("boundary proximity BP  (< 1: retained majority lies closer to the minority)")
    ax.set_ylabel("coverage error CE  (lower = majority better covered)")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("average G-mean rank (lower = better)")
    cb.ax.invert_yaxis()
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=1.4, ms=8, label="matching (proposed)"),
               Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=0.5, ms=5.5, label="size-balancing baseline"),
               Line2D([], [], marker="s", ls="", mfc="white", mec="black", mew=0.5, ms=5, label="cleaning / partial"),
               Line2D([], [], marker="D", ls="", mfc="white", mec="black", mew=0.5, ms=5, label="no resampling")]
    ax.legend(handles=handles, loc="lower left", handletextpad=0.3, borderaxespad=0.3)
    return fig, _save(fig, out_dir, stem)


def _choose_covariates(mc, max_k=4, min_abs=0.2):
    sub = mc[(mc["subset"] == "all") & (mc["metric"] == "gmean") & (mc["reference"] == "rus")].copy()
    sub["abs"] = sub["rho"].abs()
    best = sub.groupby("covariate")["abs"].max()
    pick = []
    for g in ["size", "feature overlap", "neighbourhood", "typology"]:
        cs = [c for c in COVARIATES if COV_GROUP[c] == g]
        c = best[cs].idxmax()
        pick.append((best[c], c))
    pick.sort(reverse=True)
    chosen = [c for a, c in pick if a >= min_abs][:max_k]
    if len(chosen) < 2:
        chosen = [c for a, c in pick[:2]]
    return chosen, best


def fig_gain_vs_complexity(gains, cov, mc, out_dir, stem="fig_gain_vs_complexity"):
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess
    apply_style()
    chosen, best = _choose_covariates(mc)
    ga = gains[(gains["classifier"] == "mean") & (gains["metric"] == "gmean")]
    arms = [("coverage arm", ["MUS-CS", "MUS-CO"]), ("boundary arm", ["MUS-BS", "MUS-BO"])]
    fig, axes = plt.subplots(2, len(chosen), figsize=(1.75 * len(chosen) + 1.5, 4.6), sharey="row",
                             squeeze=False)
    mcs = mc[(mc["subset"] == "all") & (mc["metric"] == "gmean") & (mc["reference"] == "rus")].set_index(
        ["method", "covariate"])
    for i, (arm, meths) in enumerate(arms):
        for j, c in enumerate(chosen):
            ax = axes[i, j]
            ax.axhline(0, color="#999999", lw=0.7, zorder=0)
            notes = []
            for k, m in enumerate(meths):
                y = ga[ga["method"] == m].set_index("dataset")["delta_rus"].reindex(cov.index)
                if y.notna().sum() == 0:
                    continue
                x = cov[c]
                ok = x.notna() & y.notna()
                xx = np.log10(x[ok]) if c in LOG_COVS else x[ok]
                col = MUS_COLOR[m]
                ax.scatter(x[ok], y[ok], s=11, marker="o", facecolors=col if k == 0 else "none",
                           edgecolors=col, linewidths=0.7, alpha=0.85, zorder=3, label=m)
                lw = lowess(y[ok].values, xx.values, frac=0.7, return_sorted=True)
                xs = 10 ** lw[:, 0] if c in LOG_COVS else lw[:, 0]
                ax.plot(xs, lw[:, 1], color=col, lw=1.4, ls="-" if k == 0 else (0, (4, 2)), zorder=4)
                r = mcs.loc[(m, c)]
                star = "*" if r["p_holm"] < 0.05 else ""
                notes.append((f"{m}: \u03c1 = {r['rho']:+.2f}{star}", col))
            for t, (txt, col) in enumerate(notes):
                ax.text(0.0, 1.03 + 0.11 * (len(notes) - 1 - t), txt, transform=ax.transAxes, fontsize=6,
                        color=col, ha="left", va="bottom")
            if c in LOG_COVS:
                ax.set_xscale("log")
            if i == 1:
                ax.set_xlabel(COV_LABEL[c])
            if j == 0:
                ax.set_ylabel(f"{arm}: \u0394 G-mean vs RUS\n(> 0: matching better)")
            ax.margins(x=0.05, y=0.08)
        axes[i, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.2, borderaxespad=0.0,
                           markerscale=1.2)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(w_pad=0.4, h_pad=2.2)
    return fig, chosen, _save(fig, out_dir, stem)


def fig_runtime(rt, per_ds, out_dir, stem="fig_runtime_benchmark"):
    import matplotlib.pyplot as plt
    apply_style()
    order = list(rt["sampler"])
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6, 1.45 * nrow + 0.4), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    xmin, xmax = per_ds["n_maj"].min() / 1.3, per_ds["n_maj"].max() * 1.3
    ymin, ymax = per_ds["t_med"].clip(lower=1e-4).min() / 2, per_ds["t_med"].max() * 2
    for k, s in enumerate(order):
        ax = axes[k // ncol, k % ncol]
        x = per_ds[per_ds["sampler"] == s]
        col = MUS_COLOR.get(s, "#7a7a7a")
        ax.scatter(x["n_maj"], x["t_med"].clip(lower=1e-4), s=6, color=col, alpha=0.75, lw=0, zorder=3)
        r = rt.set_index("sampler").loc[s]
        xs = np.array([x["n_maj"].min(), x["n_maj"].max()])
        ax.plot(xs, 10 ** (r["intercept"] + r["slope"] * np.log10(xs)), color="black" if s not in MUS else col,
                lw=1.0, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(s, fontweight="bold" if s in MUS else "normal", color=col if s in MUS else "black", pad=2)
        ax.text(0.04, 0.96, f"slope {r['slope']:.2f}\nmedian {r['t_median']:.2g} s", transform=ax.transAxes,
                fontsize=6, va="top", ha="left")
    for k in range(len(order), nrow * ncol):
        axes[k // ncol, k % ncol].set_visible(False)
    for i in range(nrow):
        axes[i, 0].set_ylabel("time (s)")
    for j in range(ncol):
        last = max(i for i in range(nrow) if i * ncol + j < len(order))
        axes[last, j].set_xlabel("majority size")
        axes[last, j].xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    return fig, _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# Across-dataset link between diagnostic differences and gains (vs RUS)
# --------------------------------------------------------------------------- #
def diag_gain_link(dd, gains):
    from scipy.stats import spearmanr
    rus = dd[dd["sampler"] == "RUS"].set_index("dataset")
    ga = gains[(gains["classifier"] == "mean")]
    rows = []
    for m in [m for m in MUS if m in set(dd["sampler"])]:
        x = dd[dd["sampler"] == m].set_index("dataset")
        for met in ("gmean", "auc"):
            y = ga[(ga["metric"] == met) & (ga["method"] == m)].set_index("dataset")["delta_rus"]
            for c in ("BP", "CE", "ED_SN", "ED_SP", "enrich", "iso_enrich"):
                dx = (x[c] - rus[c]).rename("dx")
                j = pd.concat([dx, y], axis=1).dropna()
                r = spearmanr(j["dx"], j["delta_rus"])
                rows.append(dict(method=m, metric=met, diagnostic=c, n=len(j),
                                 diag_diff_median=float(j["dx"].median()),
                                 frac_diag_lower_than_RUS=float((j["dx"] < 0).mean()),
                                 rho=float(r.statistic), p=float(r.pvalue)))
    out = pd.DataFrame(rows)
    if len(out):
        from statsmodels.stats.multitest import multipletests
        out["p_holm"] = np.nan
        for (m, met), idx in out.groupby(["method", "metric"]).groups.items():
            out.loc[idx, "p_holm"] = multipletests(out.loc[idx, "p"], method="holm")[1]
    return out


def coverage_arm_agreement(dd, gains):
    """Dataset-level agreement of MUS-CS and MUS-CO (diagnostics and gains over RUS)."""
    from scipy.stats import spearmanr
    present = set(dd["sampler"])
    if not {"MUS-CS", "MUS-CO"} <= present:
        return {}
    out = {}
    ga = gains[gains["classifier"] == "mean"]
    for met in ("gmean", "auc"):
        a = ga[(ga["metric"] == met) & (ga["method"] == "MUS-CS")].set_index("dataset")["delta_rus"]
        b = ga[(ga["metric"] == met) & (ga["method"] == "MUS-CO")].set_index("dataset")["delta_rus"].reindex(a.index)
        r = spearmanr(a, b)
        out[f"gain_rho_{met}"] = float(r.statistic)
        out[f"gain_p_{met}"] = float(r.pvalue)
        out[f"gain_same_sign_{met}"] = float((np.sign(a) == np.sign(b)).mean())
    for c in ("BP", "CE", "ED_SN", "enrich", "iso_enrich"):
        x = dd[dd["sampler"] == "MUS-CS"].set_index("dataset")[c]
        y = dd[dd["sampler"] == "MUS-CO"].set_index("dataset")[c].reindex(x.index)
        j = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
        out[f"median_absdiff_{c}"] = float((j["x"] - j["y"]).abs().median())
        out[f"rho_{c}"] = float(spearmanr(j["x"], j["y"]).statistic)
    return out


def diag_checks(diag):
    """Internal consistency checks of the selections (reported, not used for inference)."""
    out = {}
    for m in ("MUS-BO", "MUS-BS"):
        x = diag[(diag["sampler"] == m) & diag["W1_SP"].notna() & diag["W1_opt"].notna()]
        if len(x):
            rel = (x["W1_SP"] - x["W1_opt"]) / x["W1_opt"]
            out[f"{m}_folds"] = int(len(x))
            out[f"{m}_W1_rel_gap_max"] = float(rel.max())
            out[f"{m}_W1_rel_gap_median"] = float(rel.median())
            out[f"{m}_folds_at_optimum_1e-9"] = int((rel.abs() <= 1e-9).sum())
            out[f"{m}_folds_below_optimum_1e-9"] = int((rel < -1e-9).sum())
    bal = diag[diag["sampler"].isin(BALANCED)]
    out["balanced_size_ratio_min"] = float(bal["size_ratio"].min())
    out["balanced_size_ratio_max"] = float(bal["size_ratio"].max())
    out["folds_with_no_unsafe_majority"] = int((diag.drop_duplicates(["dataset", "repeat", "fold"])
                                                ["unsafe_share_N"] == 0).sum())
    out["datasets_with_no_unsafe_majority_in_any_fold"] = int(
        (diag.groupby("dataset")["unsafe_share_N"].max() == 0).sum())
    out["n_fold_units"] = int(diag.drop_duplicates(["dataset", "repeat", "fold"]).shape[0])
    out["diag_seconds_total"] = float(diag.drop_duplicates(["dataset", "repeat", "fold"])["t_diag_fold"].sum())
    return out


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Key findings (sentences generated from the tables; wording follows the numbers)
# --------------------------------------------------------------------------- #
def overlap_verdict(ov, agree):
    o = ov.set_index("pair")
    if "MUS-CS vs MUS-CO" not in o.index or not o.loc["MUS-CS vs MUS-CO"].get("n_datasets", 0):
        return "MUS-CS vs MUS-CO selection overlap: not available in this run (no MUS-CO selection records)."
    r = o.loc["MUS-CS vs MUS-CO"]
    j = float(r["jaccard_median"])
    txt = (f"MUS-CS vs MUS-CO selection overlap: median Jaccard {j:.2f} [{r['jaccard_q1']:.2f}, {r['jaccard_q3']:.2f}] "
           f"over {int(r['n_datasets'])} datasets (chance {r['chance_median']:.2f}) (selection_overlap.csv). ")
    if j >= 0.5:
        return txt + ("By the pre-specified rule (Jaccard >= 0.5) the two retain largely the same instances, so MUS-CS "
                      "ablation results can be read as indicative for MUS-CO.")
    txt += ("By the pre-specified rule (Jaccard >= 0.5) the MUS-CS ablations are NOT instance-level evidence for MUS-CO. ")
    ref = "MUS-CS vs MUS-CS (other seed)"
    if ref in o.index and np.isfinite(o.loc[ref].get("jaccard_median", np.nan)):
        rr = o.loc[ref]
        txt += (f"The rule is, however, uninformative for the coverage arm: re-running MUS-CS itself with a different "
                f"random_state on the same folds gives a median Jaccard of {rr['jaccard_median']:.2f} [{rr['jaccard_q1']:.2f}, "
                f"{rr['jaccard_q3']:.2f}] (chance {rr['chance_median']:.2f}; {rr['note']}; the original seed reproduces the "
                f"benchmark selection in {100 * rr['reproduces_benchmark_share']:.0f}% of these folds), because the member "
                f"kept from each matched pair is a fair coin flip. ")
    if agree:
        txt += (f"At the distribution level the two coverage methods agree closely: dataset-level median absolute "
                f"differences BP {agree['median_absdiff_BP']:.3f}, CE {agree['median_absdiff_CE']:.3f}, energy distance "
                f"to the full majority {agree['median_absdiff_ED_SN']:.3f}, isolated-majority retention "
                f"{agree['median_absdiff_iso_enrich']:.3f}; their classifier-averaged G-mean gains over RUS correlate "
                f"across datasets with Spearman rho {agree['gain_rho_gmean']:.2f} (p {agree['gain_p_gmean']:.2g}; same "
                f"sign in {100 * agree['gain_same_sign_gmean']:.0f}% of datasets) (coverage_arm_agreement.csv). "
                f"Reading: MUS-CS ablations of the coverage mechanism can be taken as indicative for MUS-CO at the level "
                f"of distributional behaviour and dataset-level gains, not as evidence about the specific instances "
                f"MUS-CO retains; the extent of transfer is a judgement, not a test result.")
    return txt


def _verdict(r):
    if not np.isfinite(r["wilcoxon_p"]) or r["wilcoxon_p"] >= 0.05:
        return "on par with"
    return "better than" if r["delta_median"] > 0 else "worse than"


def key_findings(summ, link, ov, bb, gsum, mc, tree_leaves, dgl, rt, checks, agree=None):
    S = summ.set_index("sampler")
    present = list(S.index)
    nds = int(S["n_datasets"].max())
    cov_arm = [m for m in ("MUS-CS", "MUS-CO") if m in present]
    bnd_arm = [m for m in ("MUS-BS", "MUS-BO") if m in present]
    md = lambda s_, c: float(S.loc[s_, f"{c}_median"])
    gm = gsum[gsum["classifier"] == "mean"].set_index(["method", "metric", "reference"])
    out = []
    ref_txt = ", ".join(f"{b} {md(b, 'BP'):.2f}/{md(b, 'CE'):.2f}" for b in ("RUS", "CC") if b in S.index)
    nm_txt = ", ".join(f"{b} {md(b, 'BP'):.2f}/{md(b, 'CE'):.2f}" for b in ("NM1", "NM2", "NM3") if b in S.index)
    # 1. map: boundary arm
    if bnd_arm:
        b0 = bnd_arm[0]
        txt = (f"Boundary arm ({' and '.join(bnd_arm)}) sits with NearMiss on the boundary-coverage map: median "
               f"BP/CE " + ", ".join(f"{m} {md(m, 'BP'):.2f}/{md(m, 'CE'):.2f}" for m in bnd_arm) +
               f" vs NearMiss {nm_txt} and {ref_txt} (diag_summary.csv). The boundary arm keeps majority instances "
               f"closer to the minority than RUS does in {int(S.loc[b0, 'BP_lt_RUS'])}/{int(S.loc[b0, 'BP_cmp_RUS_n'])} "
               f"datasets but covers the majority better than RUS in only {int(S.loc[b0, 'CE_lt_RUS'])}/"
               f"{int(S.loc[b0, 'CE_cmp_RUS_n'])}; it retains unsafe majority instances at "
               + ", ".join(f"{md(m, 'enrich'):.1f}x" for m in bnd_arm) + f" their base rate (RUS {md('RUS', 'enrich'):.2f}x).")
        if "MUS-BO_folds" in checks:
            txt += (f" MUS-BO attains the exact minimum W1 distance to the minority over all |P|-subsets in "
                    f"{checks['MUS-BO_folds_at_optimum_1e-9']}/{checks['MUS-BO_folds']} folds")
            if "MUS-BS_W1_rel_gap_median" in checks:
                txt += (f"; MUS-BS is within a median {100 * checks['MUS-BS_W1_rel_gap_median']:.2f}% "
                        f"(max {100 * checks['MUS-BS_W1_rel_gap_max']:.1f}%) of it")
            o = ov.set_index("pair")
            if "MUS-BS vs MUS-BO" in o.index and o.loc["MUS-BS vs MUS-BO"].get("n_datasets", 0):
                txt += (f" and the two select largely the same instances (median Jaccard "
                        f"{o.loc['MUS-BS vs MUS-BO', 'jaccard_median']:.2f}; MUS-BS vs NM3 "
                        f"{o.loc['MUS-BS vs NM3', 'jaccard_median']:.2f}, chance "
                        f"{o.loc['MUS-BS vs NM3', 'chance_median']:.2f}) (selection_overlap.csv)")
            txt += "."
        out.append(txt)
    # 2. map: coverage arm
    if cov_arm:
        bal = [b for b in BALANCED if b in S.index]
        ed_rank = S.loc[bal, "ED_SN_median"].sort_values()
        top = ed_rank.index[0]
        txt = (f"Coverage arm ({' and '.join(cov_arm)}) sits with RUS and ClusterCentroids: median BP/CE " +
               ", ".join(f"{m} {md(m, 'BP'):.2f}/{md(m, 'CE'):.2f}" for m in cov_arm) + f" vs {ref_txt}; "
               + "; ".join(f"{m} has lower CE than RUS in {int(S.loc[m, 'CE_lt_RUS'])}/{int(S.loc[m, 'CE_cmp_RUS_n'])} "
                           f"datasets and lower energy distance to the full majority in "
                           f"{int(S.loc[m, 'ED_SN_lt_RUS'])}/{int(S.loc[m, 'ED_SN_cmp_RUS_n'])}" for m in cov_arm)
               + ". Among the size-balancing methods the energy distance to the full majority is smallest for "
               + f"{top} ({ed_rank.iloc[0]:.3f}), followed by "
               + ", ".join(f"{k} ({v:.3f})" for k, v in ed_rank.iloc[1:4].items()) + ". "
               + ("Pairwise thinning therefore preserves the majority distribution better than random or "
                  "cluster-based selection, whereas " if top in cov_arm else "")
               + f"ClusterCentroids spreads its picks evenly over space (lowest median CE among the size-balancing "
               f"methods: {md('CC', 'CE'):.2f}) and retains the 10% most isolated majority instances at "
               f"{md('CC', 'iso_enrich'):.2f}x their base rate; "
               + ", ".join(f"{m} {md(m, 'iso_enrich'):.2f}x" for m in cov_arm)
               + f", RUS {md('RUS', 'iso_enrich'):.2f}x (descriptive; isolation retention does not track the gain, "
               + "diag_gain_link.csv).")
        out.append(txt)
    # 3. link
    lk = link[link["classifier"] == "mean"].set_index(["set", "diagnostic"])
    def lr(st, dgn):
        return lk.loc[(st, dgn)] if (st, dgn) in lk.index else None
    a = [lr("size_balanced", d_) for d_ in ("CE", "ED_SN", "enrich", "BP", "iso_enrich")]
    txt = ("Which diagnostics track performance (within-dataset Spearman across methods, classifier-averaged "
           "G-mean; diag_performance_link.csv): among the size-balancing methods, coverage and "
           f"representativeness track G-mean (median rho CE {a[0]['rho_median']:+.2f}, energy distance to the "
           f"full majority {a[1]['rho_median']:+.2f}), retaining unsafe majority hurts (enrichment "
           f"{a[2]['rho_median']:+.2f}) and boundary focus hurts (BP {a[3]['rho_median']:+.2f}; higher BP = less "
           f"boundary-focused); across all selection methods the retained size dominates (|S|/|P| "
           f"{lr('all_selections', 'size_ratio')['rho_median']:+.2f}).")
    b1, b2 = lr("boundary_cluster", "CE"), lr("boundary_cluster", "BP")
    if b1 is not None and b2 is not None:
        txt += (f" Within the boundary-focused cluster ({int(b1['n_methods'])} methods: NearMiss and the boundary arm) "
                f"coverage still orders performance (CE {b1['rho_median']:+.2f}, Holm p {b1['wilcoxon_p_holm']:.2g}) "
                f"while boundary proximity does not (BP {b2['rho_median']:+.2f}, Holm p {b2['wilcoxon_p_holm']:.2g}).")
    c1, c2 = lr("coverage_cluster", "CE"), lr("coverage_cluster", "ED_SN")
    if c1 is not None and c2 is not None:
        weak = c1["wilcoxon_p_holm"] >= 0.05 and c2["wilcoxon_p_holm"] >= 0.05
        txt += (f" Within the coverage-oriented cluster ({int(c1['n_methods'])} methods: RUS, CC and the coverage arm) "
                f"the diagnostics {'do not separate the methods' if weak else 'also order performance'} "
                f"(CE {c1['rho_median']:+.2f}, energy distance {c2['rho_median']:+.2f}; Holm p "
                f"{c1['wilcoxon_p_holm']:.2g} and {c2['wilcoxon_p_holm']:.2g})")
        sigc = lk.loc["coverage_cluster"] if "coverage_cluster" in lk.index.get_level_values(0) else None
        if sigc is not None:
            sigc = sigc[sigc["wilcoxon_p_holm"] < 0.05]
            if len(sigc):
                txt += "; only " + ", ".join(f"{DIAG_LABEL.get(d_, d_)} (rho {r_['rho_median']:+.2f}, Holm p "
                                             f"{r_['wilcoxon_p_holm']:.2g})" for d_, r_ in sigc.iterrows()) + \
                       (" reaches" if len(sigc) == 1 else " reach") + " Holm p < 0.05 there"
        txt += "."
    out.append(txt)
    # 4. boundary arm: why and when
    if bnd_arm:
        g = [gm.loc[(m, "gmean", "rus")] for m in bnd_arm]
        se = [gm.loc[(m, "sens", "rus")] for m in bnd_arm]
        sp = [gm.loc[(m, "spec", "rus")] for m in bnd_arm]
        txt = (f"Why the boundary arm hurts: its classifier-averaged G-mean is "
               + ", ".join(f"{m} {_verdict(r)} RUS (median delta {r['delta_median']:+.3f}, W/T/L "
                           f"{r['wins']}/{r['ties']}/{r['losses']})" for m, r in zip(bnd_arm, g))
               + "; the loss is a sensitivity loss (median delta sensitivity "
               + ", ".join(f"{r['delta_median']:+.3f}" for r in se) + ", losses on "
               + ", ".join(f"{r['losses']}/{r['n_datasets']}" for r in se)
               + ") only partly offset by specificity (" + ", ".join(f"{r['delta_median']:+.3f}" for r in sp)
               + ") (meta_gain_summary.csv): the matched majority instances sit inside the minority region and "
               "claim it for the majority class.")
        out.append(txt)
        sub = mc[(mc["metric"] == "gmean") & (mc["reference"] == "rus")]
        def rho(m, c, subset="all"):
            r = sub[(sub["subset"] == subset) & (sub["method"] == m) & (sub["covariate"] == c)].iloc[0]
            return f"{r['rho']:+.2f}{'*' if r['p_holm'] < 0.05 else ''}"
        b0 = bnd_arm[0]
        tl = tree_leaves[tree_leaves["method"] == b0].sort_values("mean_delta")
        txt = ("When the boundary arm is harmless: its G-mean gain over RUS rises with the safe-minority share and "
               "falls with the outlier and rare shares and with class overlap (Spearman, * = Holm p < 0.05; "
               "meta_gain_correlations.csv): "
               + "; ".join(f"{m}: safe {rho(m, 'safe')}, outlier {rho(m, 'outlier')}, rare {rho(m, 'rare')}, "
                           f"F1 {rho(m, 'F1')}, N3 {rho(m, 'N3')}" for m in bnd_arm)
               + "; on the 24 family representatives: "
               + "; ".join(f"{m} safe {rho(m, 'safe', 'family_representative')}, outlier "
                           f"{rho(m, 'outlier', 'family_representative')}" for m in bnd_arm)
               + f". Depth-2 tree for {b0} (meta_gain_tree_leaves.csv): best leaf [{tl.iloc[-1]['rule']}] mean delta "
               f"{tl.iloc[-1]['mean_delta']:+.3f} (n = {int(tl.iloc[-1]['n'])}), worst leaf [{tl.iloc[0]['rule']}] "
               f"{tl.iloc[0]['mean_delta']:+.3f} (n = {int(tl.iloc[0]['n'])}). Size, dimensionality and IR do not "
               f"predict the gain (|rho| <= "
               + f"{sub[(sub['subset'] == 'all') & sub['method'].isin(bnd_arm) & sub['covariate'].isin(['IR', 'n', 'd', 'n_min'])]['rho'].abs().max():.2f}).")
        out.append(txt)
    # 5. coverage arm: on par / mechanism / no predictor
    if cov_arm:
        parts = []
        for m in cov_arm:
            g_r, g_b = gm.loc[(m, "gmean", "rus")], gm.loc[(m, "gmean", "best")]
            a_r, a_b = gm.loc[(m, "auc", "rus")], gm.loc[(m, "auc", "best")]
            se, sp = gm.loc[(m, "sens", "rus")], gm.loc[(m, "spec", "rus")]
            parts.append(
                f"{m} is {_verdict(g_r)} RUS in G-mean (median delta {g_r['delta_median']:+.4f}, W/T/L "
                f"{g_r['wins']}/{g_r['ties']}/{g_r['losses']}, Wilcoxon p {g_r['wilcoxon_p']:.3g}) and "
                f"{_verdict(g_b)} the best non-matching undersampler ({g_b['delta_median']:+.4f}, "
                f"{g_b['wins']}/{g_b['ties']}/{g_b['losses']}, p {g_b['wilcoxon_p']:.3g}); in ROC-AUC it is "
                f"{_verdict(a_r)} RUS ({a_r['delta_median']:+.4f}, {a_r['wins']}/{a_r['ties']}/{a_r['losses']}, "
                f"p {a_r['wilcoxon_p']:.3g}) and {_verdict(a_b)} the best AUC baseline ({a_b['delta_median']:+.4f}, "
                f"p {a_b['wilcoxon_p']:.3g}); versus RUS it trades sensitivity ({se['delta_median']:+.4f}, "
                f"losses {se['losses']}/{se['n_datasets']}) for specificity ({sp['delta_median']:+.4f}, wins "
                f"{sp['wins']}/{sp['n_datasets']})")
        sub = mc[(mc["metric"] == "gmean") & (mc["reference"] == "rus") & (mc["subset"] == "all")]
        nsig = {m: int((sub[(sub["method"] == m)]["p_holm"] < 0.05).sum()) for m in cov_arm}
        mx = {m: sub[sub["method"] == m].reindex(sub[sub["method"] == m]["rho"].abs().sort_values().index).iloc[-1]
              for m in cov_arm}
        rk = S["gmean_avg_rank"].sort_values()
        txt = ("Coverage arm (meta_gain_summary.csv): " + "; ".join(parts) + ". "
               + f"Average G-mean rank among the {len(present)} selection methods (diag_summary.csv): "
               + ", ".join(f"{k_} {v_:.2f}" for k_, v_ in rk.head(5).items()) + ". "
               + " ".join(f"{m}: {nsig[m]} of {len(COVARIATES)} characteristics Holm-significant for the G-mean "
                          f"gain over RUS (strongest {mx[m]['covariate']} rho {mx[m]['rho']:+.2f})." for m in cov_arm))
        out.append(txt)
    # 6. overlap verdict
    out.append(overlap_verdict(ov, agree or {}))
    # 7. runtime
    r = rt.set_index("sampler")
    txt = ("Runtime (median s per training fold under parallel load; runtime_benchmark.csv): "
           + " < ".join(f"{k} {v:.3g}" for k, v in r["t_median"].items())
           + ". Log-log slopes vs majority size: " + ", ".join(f"{m} {r.loc[m, 'slope']:.2f}" for m in MUS if m in r.index)
           + f" (RUS {r.loc['RUS', 'slope']:.2f}, CC {r.loc['CC', 'slope']:.2f}, CNN {r.loc['CNN', 'slope']:.2f}); "
           "95th percentiles: " + ", ".join(f"{m} {r.loc[m, 't_p95']:.3g}" for m in MUS if m in r.index) + ".")
    out.append(txt)
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_analysis(results_path, sel_path, registry_path, complexity_path, out_dir, diag=None,
                 provenance=None, data_path=None, folds_path=None, ms_path=None):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    P = lambda f: os.path.join(out_dir, f)
    if diag is None:
        diag = pd.read_parquet(P("diag_selection.parquet"))
    res = load_results(results_path)
    perf = performance_table(res)
    present = [s for s in SELECTIONS if s in set(diag["sampler"])]
    missing = [s for s in SELECTIONS if s not in present]
    samplers_res = sorted(set(res["sampler"]))
    manifest = dict(
        inputs={k: dict(path=os.path.basename(v), sha256=_sha256(v)) for k, v in
                dict(results=results_path, selections=sel_path, registry=registry_path,
                     complexity=complexity_path, data=data_path, folds=folds_path,
                     matchsampling=ms_path).items() if v},
        provenance=provenance or {}, samplers_in_results=samplers_res, selections_present=present,
        selections_missing=missing, n_result_rows=int(len(res)), n_diag_rows=int(len(diag)),
        folds_per_cell=sorted(perf["n_folds"].unique().tolist()),
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    # ---- step 2/3: diagnostics summary and desiderata map
    ranks, nblocks = average_ranks(perf, present)
    dd = dataset_level_diagnostics(diag)
    dd.to_csv(P("diag_by_dataset.csv"), index=False)
    summ = diag_summary_table(dd, perf, ranks, nblocks)
    summ.to_csv(P("diag_summary.csv"), index=False)
    note = "" if not missing else (r" PROVISIONAL: " + ", ".join(missing) + " not available.")
    diag_summary_tex(summ, P("diag_summary.tex"), note)
    fig, _ = fig_desiderata_map(summ, out_dir)
    plt.close(fig)
    checks = diag_checks(diag)
    # ---- step 4: diagnostics-performance link (within dataset, across selection methods)
    link, link_ds = diag_performance_link(dd, perf, present)
    link.to_csv(P("diag_performance_link.csv"), index=False)
    link_ds.to_csv(P("diag_performance_link_by_dataset.csv"), index=False)
    # ---- step 4b: selection overlap
    sel_index = load_selection_index(sel_path)
    ov, ov_ds = selection_overlap(sel_index, diag)
    if ms_path and data_path and folds_path:
        ref, ref_folds = seed_reference_overlap(data_path, folds_path, sel_index, ms_path)
        ref_folds.to_csv(P("seed_reference_overlap_folds.csv"), index=False)
        ov = pd.concat([ov, pd.DataFrame([ref])], ignore_index=True)
    ov.to_csv(P("selection_overlap.csv"), index=False)
    ov_ds.to_csv(P("selection_overlap_by_dataset.csv"), index=False)
    overlap_tex(ov, P("selection_overlap.tex"))
    # ---- step 5: gains and meta-analysis
    bb = best_baselines(perf)
    bb.to_csv(P("best_baselines.csv"), index=False)
    gains = gains_table(perf, bb)
    gains.to_csv(P("meta_gains_per_dataset.csv"), index=False)
    gsum = gain_summary(gains)
    gsum.to_csv(P("meta_gain_summary.csv"), index=False)
    cov = covariate_table(complexity_path, registry_path)
    mc = meta_correlations(gains, cov)
    mc.to_csv(P("meta_gain_correlations.csv"), index=False)
    meta_corr_tex(mc, P("meta_gain_correlations.tex"), note)
    tree_txt, tree_leaves = gain_trees(gains, cov)
    open(P("meta_gain_trees.txt"), "w", encoding="utf-8").write(tree_txt)
    tree_leaves.to_csv(P("meta_gain_tree_leaves.csv"), index=False)
    dgl = diag_gain_link(dd, gains)
    dgl.to_csv(P("diag_gain_link.csv"), index=False)
    agree = coverage_arm_agreement(dd, gains)
    if agree:
        pd.DataFrame([agree]).to_csv(P("coverage_arm_agreement.csv"), index=False)
    manifest["coverage_arm_agreement"] = agree
    fig, chosen, _ = fig_gain_vs_complexity(gains, cov, mc, out_dir)
    plt.close(fig)
    # ---- step 6: runtime
    rt, rt_ds = runtime_table(res)
    rt.to_csv(P("runtime_benchmark.csv"), index=False)
    rt_ds.to_csv(P("runtime_by_dataset.csv"), index=False)
    runtime_tex(rt, P("runtime_benchmark.tex"))
    fig, _ = fig_runtime(rt, rt_ds, out_dir)
    plt.close(fig)
    manifest.update(style=STYLE_USED, checks=checks, gain_figure_covariates=chosen)
    json.dump(manifest, open(P("run_manifest.json"), "w"), indent=2, default=str)
    nums = write_report(out_dir, manifest, summ, link, ov, bb, gsum, mc, tree_txt, tree_leaves, dgl, rt,
                        chosen, cov)
    return dict(summary=summ, link=link, overlap=ov, best=bb, gains=gains, gain_summary=gsum, meta=mc,
                trees=tree_txt, diag_gain=dgl, runtime=rt, manifest=manifest, numbers=nums)


# --------------------------------------------------------------------------- #
# Report (every number is read from the tables written above)
# --------------------------------------------------------------------------- #
def write_report(out_dir, manifest, summ, link, ov, bb, gsum, mc, tree_txt, tree_leaves, dgl, rt, chosen,
                 cov):
    S = summ.set_index("sampler")
    present = list(S.index)
    mus = [m for m in MUS if m in present]
    f2 = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.2f}"
    f3 = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"
    f4 = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.4f}"
    fp = lambda v: "n/a" if v is None or not np.isfinite(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}")

    def dg(s, c):
        r = S.loc[s]
        return f"{f2(r[c + '_median'])} [{f2(r[c + '_q1'])}, {f2(r[c + '_q3'])}]"

    nds = int(S["n_datasets"].max())
    nums = {}
    L = []
    prov = "PROVISIONAL" if manifest["selections_missing"] else "FINAL"
    L.append(f"# Why and when matching undersamplers help or hurt: selection diagnostics and meta-analysis\n")
    L.append(f"**Status: {prov} run** ({manifest['created']}). Selection methods analysed: "
             f"{', '.join(present)}" + (f"; missing: {', '.join(manifest['selections_missing'])}."
                                        if manifest['selections_missing'] else "."))
    L.append(f"Inputs (SHA-256 in `run_manifest.json`): " +
             "; ".join(f"{k} = `{os.path.basename(v['path'])}`" for k, v in manifest["inputs"].items()) + ".")
    if manifest.get("provenance"):
        L.append("Input artifact versions: " + "; ".join(f"{k} = {v}" for k, v in manifest["provenance"].items()) + ".")
    L.append(f"Figures use the figure-style rules ({manifest.get('style')}).\n")
    kfs = key_findings(summ, link, ov, bb, gsum, mc, tree_leaves, dgl, rt, manifest.get("checks", {}),
                       manifest.get("coverage_arm_agreement", {}))
    L.append("## 0. Key findings\n")
    for i, k in enumerate(kfs, 1):
        L.append(f"{i}. {k}")
    L.append("")
    nums["key_findings"] = kfs

    # ------------------------------------------------------------ methods
    L.append("## 1. Methods\n")
    L.append("**Preprocessing.** Every training fold of the benchmark (55 datasets x 5 x 5 repeated stratified CV) "
             "is re-created exactly as in the benchmark runner: median imputation (`keep_empty_features=True`) and "
             "MinMax[0, 1] scaling fitted on the training fold; Euclidean distances. The retained majority set S of "
             "each undersampler is read from the benchmark's selection records (original row indices). P = minority "
             "and N = majority of the training fold. ClusterCentroids (hard voting) can return the same instance "
             "for two centroids, so S is a multiset for the mean-based diagnostics (as seen by the classifier) and "
             "a set for coverage and overlap. No resampling (NONE) enters as the identity selection S = N.")
    L.append("**Diagnostics** (per dataset, repeat, fold and method):\n"
             "- boundary proximity BP = mean_{z in S} d(z, P) / mean_{z in N} d(z, P), with d(z, P) the distance "
             "to the nearest minority instance (< 1: S lies closer to the minority than the majority as a whole);\n"
             "- coverage error CE = mean_{z in N} d(z, S) / mean_{z in N} d(z, N minus z) (0 when S = N; the "
             "denominator is the mean nearest-neighbour spacing of the majority);\n"
             "- energy distance E(A, B) = sqrt(2 E|a-b| - E|a-a'| - E|b-b'|) (V-statistics, metric form) between S "
             "and N and between S and P, computed **exactly on all points** (no subsampling was needed: the "
             "largest training-fold majority has 4,000 or fewer points);\n"
             "- W1(S, P): exact 1-Wasserstein distance between S and P when |S| = |P| (minimum-cost perfect "
             "matching / |P|), reported relative to W1* = the minimum over all |P|-subsets of N (rectangular "
             "assignment) and only for the size-balancing methods;\n"
             "- unsafe-majority enrichment = share of S whose 5 nearest training-fold neighbours (both classes, "
             "self excluded) contain <= 2 majority instances, divided by the same share in N (undefined in folds "
             "without unsafe majority instances);\n"
             "- isolated-majority retention (added to the planned set to probe outlier retention) = share of S among "
             "the 10% majority instances farthest from their nearest other majority instance, divided by 0.10;\n"
             "- retained size |S| / |P|.\n")
    L.append("Fold values are averaged per dataset (folds with an undefined value are skipped); per-method "
             "summaries are the median and quartiles over datasets. Average G-mean ranks rank the "
             f"{len(present)} selection methods within each (dataset, classifier) block "
             f"({int(S['rank_blocks'].iloc[0])} blocks = {nds} datasets x 5 classifiers; 1 = best).")
    L.append("**Diagnostics-performance link.** Within each dataset, the Spearman correlation across selection "
             "methods between a fold-averaged diagnostic and the mean G-mean (per classifier and "
             "classifier-averaged); distribution over datasets with a Wilcoxon signed-rank test of the "
             "correlations against 0 (Holm-adjusted across diagnostics within each method set and classifier). "
             "Two method sets: all selections (sizes differ widely) and the size-balancing methods only "
             "(|S| = |P|), which isolates *which* majority instances are kept from *how many*.")
    L.append("**Selection overlap.** Jaccard similarity of the retained majority sets of two methods on the same "
             "training fold; the chance level is that of two independent uniformly random subsets with the same "
             "sizes, (ab/n) / (a + b - ab/n).")
    L.append("**Gains and meta-analysis.** Delta = mean metric(MUS-x) - mean metric(reference) over the 25 folds, "
             "per dataset and classifier, for G-mean and ROC-AUC; references: RUS, and the non-matching "
             "undersampler with the best average rank for that classifier and metric (chosen post hoc on all "
             "datasets, which favours the reference). Classifier-averaged Deltas are related to 14 dataset "
             "characteristics (IR, n, d, minority size, F1, N1, N2, N3, LSC, minority safe/borderline/rare/outlier "
             "shares, overlapping-majority share) by Spearman correlation with Holm adjustment across the 14 "
             "characteristics, on all datasets and on the 24 family representatives. F1, N1, N2, N3 and LSC "
             "follow Lorena et al. (higher = harder). A depth-2 regression tree (min. 5 datasets per leaf) "
             "summarises the G-mean gain over RUS descriptively (in-sample, not validated). G-mean gains are "
             "decomposed into sensitivity and specificity changes.")
    L.append("**Runtime.** Resampling time per training fold from the benchmark (measured while several benchmark "
             "processes ran in parallel, so absolute values are inflated and noisy); log-log OLS slope of the "
             "per-dataset median time on the training-fold majority size.\n")

    # ------------------------------------------------------------ results 1
    L.append("## 2. Where the methods sit on the boundary-coverage map\n")
    L.append(f"Source: `diag_summary.csv` (median [Q1, Q3] over {nds} datasets), Figure `fig_desiderata_map`.\n")
    hdr = ("| Method | size ratio | BP | CE | E(S,N) | E(S,P) | W1/W1* | unsafe enr. | isolated enr. | "
           "avg G-mean rank |")
    L.append(hdr)
    L.append("|" + "---|" * 10)
    for s in present:
        r = S.loc[s]
        L.append(f"| {s} | {f2(r['size_ratio_median'])} | {dg(s, 'BP')} | {dg(s, 'CE')} | {dg(s, 'ED_SN')} | "
                 f"{dg(s, 'ED_SP')} | {dg(s, 'W1_ratio') if np.isfinite(r['W1_ratio_median']) else 'n/a'} | "
                 f"{dg(s, 'enrich')} | {dg(s, 'iso_enrich')} | {f2(r['gmean_avg_rank'])} |")
    L.append("")
    ck = manifest.get("checks", {})
    if "MUS-BO_folds" in ck:
        L.append(f"- Check: MUS-BO attains the minimum W1(S, P) over all |P|-subsets in "
                 f"{ck['MUS-BO_folds_at_optimum_1e-9']} of {ck['MUS-BO_folds']} folds (relative tolerance 1e-9; "
                 f"max. relative gap {ck['MUS-BO_W1_rel_gap_max']:.2e}), as the theory requires. MUS-BS (stable) "
                 f"is within a median relative gap of {ck.get('MUS-BS_W1_rel_gap_median', np.nan):.2e} "
                 f"(max. {ck.get('MUS-BS_W1_rel_gap_max', np.nan):.2e}) of that optimum.")
    for m in mus:
        r = S.loc[m]
        L.append(f"- {m}: BP {dg(m, 'BP')}, CE {dg(m, 'CE')}; BP below RUS in {int(r['BP_lt_RUS'])}/"
                 f"{int(r['BP_cmp_RUS_n'])} datasets, CE below RUS in {int(r['CE_lt_RUS'])}/{int(r['CE_cmp_RUS_n'])}, "
                 f"E(S,N) below RUS in {int(r['ED_SN_lt_RUS'])}/{int(r['ED_SN_cmp_RUS_n'])}, unsafe enrichment below "
                 f"RUS in {int(r['enrich_lt_RUS'])}/{int(r['enrich_cmp_RUS_n'])}.")
        nums[f"{m}_BP_median"] = r["BP_median"]
        nums[f"{m}_CE_median"] = r["CE_median"]
        nums[f"{m}_ED_SN_median"] = r["ED_SN_median"]
        nums[f"{m}_enrich_median"] = r["enrich_median"]
        nums[f"{m}_rank"] = r["gmean_avg_rank"]
    for b in ("RUS", "CC", "NM1", "NM2", "NM3"):
        if b in S.index:
            nums[f"{b}_BP_median"] = S.loc[b, "BP_median"]
            nums[f"{b}_CE_median"] = S.loc[b, "CE_median"]
            nums[f"{b}_ED_SN_median"] = S.loc[b, "ED_SN_median"]
            nums[f"{b}_enrich_median"] = S.loc[b, "enrich_median"]
            nums[f"{b}_rank"] = S.loc[b, "gmean_avg_rank"]
    rank_order = S["gmean_avg_rank"].sort_values()
    L.append(f"- Average G-mean rank among the {len(present)} selection methods (lower = better): " +
             ", ".join(f"{k} {v:.2f}" for k, v in rank_order.items()) + ".")
    L.append(f"- Enrichment is undefined in {ck.get('datasets_with_no_unsafe_majority_in_any_fold', 'n/a')} "
             f"datasets that have no unsafe majority instance in any fold "
             f"({ck.get('folds_with_no_unsafe_majority', 'n/a')} of {ck.get('n_fold_units', 'n/a')} folds).\n")

    # ------------------------------------------------------------ results 2
    L.append("## 3. Which diagnostics track performance\n")
    L.append("Source: `diag_performance_link.csv` (per-dataset correlations in "
             "`diag_performance_link_by_dataset.csv`).\n")
    L.append("| Method set | diagnostic | datasets | median rho [Q1, Q3] | share rho > 0 | Wilcoxon p (Holm) |")
    L.append("|---|---|---|---|---|---|")
    lk = link[link["classifier"] == "mean"]
    for r in lk.itertuples():
        L.append(f"| {r.set} ({r.n_methods}) | {r.diagnostic} | {r.n_datasets} | {f2(r.rho_median)} "
                 f"[{f2(r.rho_q1)}, {f2(r.rho_q3)}] | {f2(r.frac_pos)} | {fp(r.wilcoxon_p_holm)} |")
        nums[f"link_{r.set}_{r.diagnostic}_rho_median"] = r.rho_median
    L.append("\nPer classifier (size-balancing set), median rho: ")
    lb = link[(link["set"] == "size_balanced") & (link["classifier"] != "mean")]
    for dgn, x in lb.groupby("diagnostic", sort=False):
        L.append(f"- {dgn}: " + ", ".join(f"{c} {f2(v)}" for c, v in zip(x["classifier"], x["rho_median"])))
    L.append("")
    L.append("Across datasets (`diag_gain_link.csv`): Spearman correlation between the dataset-level diagnostic "
             "difference (MUS-x minus RUS) and the classifier-averaged G-mean gain over RUS.\n")
    L.append("| Method | diagnostic | median diff. vs RUS | rho | p (Holm) |")
    L.append("|---|---|---|---|---|")
    for r in dgl[dgl["metric"] == "gmean"].itertuples():
        L.append(f"| {r.method} | {r.diagnostic} | {f3(r.diag_diff_median)} | {f2(r.rho)} | {fp(r.p_holm)} |")
    L.append("")

    # ------------------------------------------------------------ results 3
    L.append("## 4. Selection overlap\n")
    L.append("Source: `selection_overlap.csv` (median [Q1, Q3] over datasets of the fold-averaged Jaccard).\n")
    L.append("| Pair | Jaccard | chance | datasets above chance | datasets |")
    L.append("|---|---|---|---|---|")
    for r in ov.to_dict("records"):
        if not r.get("n_datasets"):
            L.append(f"| {r['pair']} | not available | | | 0 |")
            continue
        extra = f" ({r['note']})" if isinstance(r.get("note"), str) else ""
        L.append(f"| {r['pair']}{extra} | {f2(r['jaccard_median'])} [{f2(r['jaccard_q1'])}, {f2(r['jaccard_q3'])}] | "
                 f"{f2(r['chance_median'])} | {f2(r['frac_datasets_above_chance'])} | {r['n_datasets']} |")
        nums[f"jaccard_{r['pair']}"] = r["jaccard_median"]
    L.append("\n" + overlap_verdict(ov, manifest.get("coverage_arm_agreement", {})) + "\n")

    # ------------------------------------------------------------ results 4
    L.append("## 5. When matching helps: gains and dataset characteristics\n")
    L.append("Best non-matching undersampler per classifier and metric (`best_baselines.csv`): " +
             "; ".join(f"{r.classifier}/{r.metric}: {r.best} (avg rank {r.best_avg_rank:.2f})"
                       for r in bb.itertuples()) + ".\n")
    L.append("Classifier-averaged gains (`meta_gain_summary.csv`; median [Q1, Q3] over datasets, wins/ties/losses, "
             "Wilcoxon signed-rank p, unadjusted):\n")
    L.append("| Method | metric | reference | median Delta | wins/ties/losses | p |")
    L.append("|---|---|---|---|---|---|")
    gm = gsum[gsum["classifier"] == "mean"]
    for met in ("gmean", "auc", "sens", "spec"):
        for m in mus:
            for ref in ("rus", "best"):
                r = gm[(gm["metric"] == met) & (gm["method"] == m) & (gm["reference"] == ref)].iloc[0]
                L.append(f"| {m} | {met} | {ref} | {f4(r['delta_median'])} [{f4(r['delta_q1'])}, "
                         f"{f4(r['delta_q3'])}] | {r['wins']}/{r['ties']}/{r['losses']} | {fp(r['wilcoxon_p'])} |")
                nums[f"gain_{m}_{met}_{ref}_median"] = r["delta_median"]
                nums[f"gain_{m}_{met}_{ref}_wtl"] = f"{r['wins']}/{r['ties']}/{r['losses']}"
    L.append("")
    L.append("Spearman correlations of the classifier-averaged G-mean gain over RUS with dataset characteristics "
             "(`meta_gain_correlations.csv`; * = Holm-adjusted p < 0.05 across the 14 characteristics):\n")
    sub = mc[(mc["metric"] == "gmean") & (mc["reference"] == "rus")]
    L.append("| Characteristic | " + " | ".join(f"{m} (all)" for m in mus) + " | " +
             " | ".join(f"{m} (24 reps)" for m in mus) + " |")
    L.append("|---|" + "---|" * (2 * len(mus)))
    for c in COVARIATES:
        cells = []
        for subset in ("all", "family_representative"):
            for m in mus:
                r = sub[(sub["subset"] == subset) & (sub["method"] == m) & (sub["covariate"] == c)].iloc[0]
                cells.append(f"{r['rho']:+.2f}{'*' if r['p_holm'] < 0.05 else ''}")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
    sig = mc[(mc["p_holm"] < 0.05)]
    L.append(f"Holm-significant associations over all metric x reference x subset combinations: {len(sig)} of "
             f"{len(mc)} tests; by method: " +
             ", ".join(f"{m} {int((sig['method'] == m).sum())}" for m in mus) + ".")
    for m in mus:
        x = sig[sig["method"] == m]
        if len(x):
            top = x.reindex(x["rho"].abs().sort_values(ascending=False).index).head(6)
            L.append(f"- {m}: " + "; ".join(f"{r.covariate} rho {r.rho:+.2f} ({r.metric} vs {r.reference}, "
                                           f"{'24 reps' if r.subset != 'all' else 'all'})"
                                           for r in top.itertuples()))
        else:
            L.append(f"- {m}: no Holm-significant association with any characteristic.")
    L.append(f"\nGain-vs-characteristic figure (`fig_gain_vs_complexity`): panels show the strongest characteristic "
             f"of each group (size, feature overlap, neighbourhood, minority typology) with max |rho| >= 0.2 across "
             f"the matching methods: {', '.join(chosen)}; LOWESS (frac 0.7) per method.\n")
    L.append("Depth-2 regression trees (descriptive; `meta_gain_trees.txt`, leaves in `meta_gain_tree_leaves.csv`):\n")
    L.append("```\n" + tree_txt.strip() + "\n```\n")

    # ------------------------------------------------------------ runtime
    L.append("## 6. Runtime\n")
    L.append("Source: `runtime_benchmark.csv` (seconds per training fold; timings under parallel load).\n")
    L.append("| Method | median | 95th pct. | max | log-log slope [95% CI] | R^2 |")
    L.append("|---|---|---|---|---|---|")
    for r in rt.itertuples():
        L.append(f"| {r.sampler} | {r.t_median:.4g} | {r.t_p95:.4g} | {r.t_max:.4g} | {r.slope:.2f} "
                 f"[{r.slope_ci_lo:.2f}, {r.slope_ci_hi:.2f}] | {r.r2:.2f} |")
        nums[f"rt_{r.sampler}_median"] = r.t_median
        nums[f"rt_{r.sampler}_p95"] = r.t_p95
        nums[f"rt_{r.sampler}_slope"] = r.slope
    L.append("\nOrdering by median time: " + " < ".join(rt["sampler"]) + ".\n")
    L.append("## 7. Caveats\n")
    L.append("- Diagnostics are computed on the training folds (what the classifier sees), not on test data.\n"
             "- The best-baseline reference is chosen post hoc on all datasets, which favours the baseline.\n"
             "- The within-dataset link uses at most 16 methods per dataset, and in the size-balancing set the "
             "methods form two clusters (boundary-focused: NearMiss and the boundary arm; coverage-oriented: RUS, "
             "CC and the coverage arm), so the correlations largely contrast these clusters.\n"
             "- Meta-correlations are exploratory (55 datasets from 24 families; the 24-representative subset is "
             "the robustness check); the trees are in-sample descriptions.\n"
             "- Runtimes were measured with several benchmark processes running concurrently.\n")
    L.append("## 8. Files\n")
    for f, d in [("diag_selection.parquet", "fold-level diagnostics"),
                 ("diag_by_dataset.csv", "fold-averaged diagnostics per dataset and method"),
                 ("diag_summary.csv / .tex", "per-method summary (Section 2)"),
                 ("fig_desiderata_map.pdf/.png", "boundary-coverage map"),
                 ("diag_performance_link.csv", "within-dataset diagnostic-performance correlations (Section 3)"),
                 ("diag_performance_link_by_dataset.csv", "per-dataset correlations"),
                 ("diag_gain_link.csv", "across-dataset diagnostic-difference vs gain correlations"),
                 ("selection_overlap.csv / .tex", "Jaccard overlaps (Section 4)"),
                 ("selection_overlap_by_dataset.csv", "per-dataset Jaccard"),
                 ("best_baselines.csv", "best non-matching undersampler per classifier and metric"),
                 ("meta_gains_per_dataset.csv", "per-dataset gains"),
                 ("meta_gain_summary.csv", "gain distribution summaries (Section 5)"),
                 ("meta_gain_correlations.csv / .tex", "Spearman meta-correlations with Holm adjustment"),
                 ("meta_gain_trees.txt, meta_gain_tree_leaves.csv", "depth-2 regression trees"),
                 ("fig_gain_vs_complexity.pdf/.png", "gains vs strongest characteristics"),
                 ("runtime_benchmark.csv / .tex, runtime_by_dataset.csv", "runtime (Section 6)"),
                 ("fig_runtime_benchmark.pdf/.png", "runtime scaling"),
                 ("run_manifest.json", "inputs, checksums, checks"),
                 ("mechanisms_numbers.json", "machine-readable copy of the quoted numbers")]:
        L.append(f"- `{f}`: {d}")
    open(os.path.join(out_dir, "mechanisms_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    nums = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in nums.items()}
    json.dump(nums, open(os.path.join(out_dir, "mechanisms_numbers.json"), "w"), indent=1)
    return nums


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "diag-worker":
        cfg = json.load(open(argv[1]))
        diag_worker(cfg["data"], cfg["folds"], cfg["selections"], cfg["datasets"], cfg["out"])
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--selections", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--complexity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-diagnostics", action="store_true",
                    help="reuse OUT/diag_selection.parquet instead of recomputing stage 1")
    ap.add_argument("--ms-path", default=None,
                    help="optional path to matchsampling.py: adds the MUS-CS seed-to-seed overlap reference")
    ap.add_argument("--provenance", default=None,
                    help="optional JSON object (label -> artifact version id) recorded in the report")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.skip_diagnostics:
        diag = pd.read_parquet(os.path.join(a.out, "diag_selection.parquet"))
    else:
        diag = run_diagnostics(a.data, a.folds, a.selections, a.out, a.workers)
    print(f"stage 1 done: {len(diag)} diagnostic rows, {time.time() - t0:.0f}s", flush=True)
    run_analysis(a.results, a.selections, a.registry, a.complexity, a.out, diag=diag,
                 provenance=json.loads(a.provenance) if a.provenance else None, data_path=a.data,
                 folds_path=a.folds, ms_path=a.ms_path)
    print(f"all done: {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
