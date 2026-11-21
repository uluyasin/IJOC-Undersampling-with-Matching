"""figures_pipeline.py -- main figure arc (Figures 1-5) and supplementary CD diagrams for the
matching-theory undersampling manuscript (2x2 family: boundary/coverage arm x stable/optimal concept).

Re-runnable from input file paths:

    python figures_pipeline.py --inputs inputs.json --out figs

``inputs.json`` maps keys to files (the frozen analysis outputs):
    perfold       results_main_final.parquet          per-fold benchmark results
    dsmeans       stats_per_dataset_means.csv         per-dataset means over the 25 folds
    factorial     stats_factorial_2x2.csv             2x2 factorial Wilcoxon-Holm tests
    bayes         stats_bayes.csv                     Bayesian signed-rank posteriors (ROPE 0.01)
    ranks         stats_ranks_friedman.csv            average ranks / Friedman / Nemenyi CD
    wilcoxon      stats_wilcoxon_holm.csv             MUS vs baseline Wilcoxon-Holm tests
    diag_ds       diag_by_dataset.csv                 selection diagnostics (BP, CE, ...) per dataset
    diag_sum      diag_summary.csv                    diagnostic medians / IQR per sampler
    diag_perf     diag_performance_link.csv           within-dataset diagnostic-performance rho summary
    diag_perf_ds  diag_performance_link_by_dataset.csv  ... per dataset
    diag_gain     diag_gain_link.csv                  diagnostic change vs gain correlations
    gains_ds      meta_gains_per_dataset.csv          per-dataset gains over RUS / best baseline
    gain_corr     meta_gain_correlations.csv          gain vs dataset-characteristic correlations
    complexity    dataset_complexity.csv              dataset characteristics (safe share etc.)
    rt_sum        runtime_benchmark.csv               resampling-time summary per sampler
    rt_ds         runtime_by_dataset.csv              resampling time per dataset
    matchsampling matchsampling.py                    frozen sampler module (MatchingUnderSampler)

Outputs (in --out): fig1_hook, fig2_evidence, fig3_mechanism, fig4_when, fig5_cost, figS_cd_gmean,
figS_cd_auc (.pdf + .png at 300 dpi); figure_numbers.csv (every number the figures and captions
quote, long format with figure / panel / source); figure_plotdata.csv (per-dataset values drawn in
the profile and scatter panels); render_checks.json (figure-style 9.1 geometric check per figure).

Every number is recomputed from the per-dataset means and asserted against the frozen statistics
tables (factorial means, gains, diagnostic summaries, link correlations, runtime medians), so a
mismatch stops the run instead of producing a silently different figure.

Colour key (threaded through every figure): MUS-CO #E66100, MUS-CS #A6430F, MUS-BS #08519C,
MUS-BO #4A98C9, RUS black, ClusterCentroids #636363, NearMiss-1/2/3 #969696/#BDBDBD/#525252 with
distinct markers, other baselines light grey. Text drawn in a key colour lighter than #767676 is
darkened to #767676 for legibility (marks keep the exact key colour).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator, NullFormatter
from matplotlib import image as mpimg
from scipy.stats import linregress, spearmanr
from scipy.stats import t as student_t
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------------------------------------------------------------------------
# Style: identical to figure-style apply_figure_style(frame='open', sizes=(8, 7, 6))
# ------------------------------------------------------------------------------------------
STYLE_RC = {
    "font.size": 8.0, "axes.labelsize": 8.0, "axes.titlesize": 8.0, "legend.fontsize": 7.0,
    "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "axes.linewidth": 0.6,
    "axes.spines.right": False, "axes.spines.top": False, "axes.titlelocation": "left",
    "figure.dpi": 200.0, "legend.frameon": False, "lines.linewidth": 1.2, "patch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300.0,
    "xtick.major.size": 3.0, "xtick.major.width": 0.6, "ytick.major.size": 3.0,
    "ytick.major.width": 0.6,
}
FS_BASE, FS_SEC, FS_TICK = 8, 7, 6
HATCH_LW = 0.5
SAVE_PAD = 0.03         # inches around the tight bounding box (keeps saved figures within 7.2 in)

CLFS = ["CART", "kNN", "SVM", "LR", "RF"]
CLF_MARK = {"CART": "o", "kNN": "s", "SVM": "^", "LR": "D", "RF": "v"}   # Fig 3b per-classifier marks (hollow)
MUS = ["MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
ORDER16 = ["NONE", "RUS", "NM1", "NM2", "NM3", "TL", "ENN", "NCR", "OSS", "CNN", "IHT", "CC",
           "MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
US15 = ORDER16[1:]                                   # the 15 undersamplers
BAL9 = ["RUS", "NM1", "NM2", "NM3", "CC", "MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
BCL5 = ["NM1", "NM2", "NM3", "MUS-BS", "MUS-BO"]     # boundary-focused cluster
B8 = [m for m in BAL9 if m != "RUS"]
NONBAL = ["NONE", "TL", "ENN", "NCR", "OSS", "CNN", "IHT"]   # selections whose output size is not controlled
ORDER_RT = ["RUS", "CC", "NM1", "NM2", "NM3", "MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO",
            "TL", "ENN", "NCR", "OSS", "CNN", "IHT", "SMOTE"]
ROPE = 0.01             # region of practical equivalence on the metric scale (as in the Bayesian tests)
TIE_EPS = 1e-3          # W/T/L tie threshold (as in the statistics pipeline)
B_BOOT = 5000           # percentile bootstrap over datasets
SEED_BOOT = 20261116
SAFE_CUT = 0.8
JITTER_HALF = 0.006     # Fig 4: horizontal display jitter of the point mass at safe share 0 (uniform, seeded)
JITTER_SEED = 20261117
LOWESS_FRAC, LOWESS_IT = 2 / 3, 3   # statsmodels defaults
N_EXPECTED = 55

COL = {"MUS-CO": "#E66100", "MUS-CS": "#A6430F", "MUS-BS": "#08519C", "MUS-BO": "#4A98C9",
       "RUS": "#000000", "CC": "#636363", "NM1": "#969696", "NM2": "#BDBDBD", "NM3": "#525252"}
OTHER_GREY = "#CCCCCC"
DARK_EDGE = "#7F7F7F"
MINORITY_COL = "#762A83"
GRID_GREY = "#9E9E9E"
BAND_GREY = "#E3E3E3"
MARK = {"RUS": "o", "CC": "X", "NM1": "s", "NM2": "D", "NM3": "v",
        "MUS-BS": "o", "MUS-BO": "^", "MUS-CS": "o", "MUS-CO": "^"}
NAME = {"NONE": "no resampling (NONE)", "RUS": "random undersampling (RUS)",
        "NM1": "NearMiss-1 (NM1)", "NM2": "NearMiss-2 (NM2)", "NM3": "NearMiss-3 (NM3)",
        "TL": "Tomek links (TL)", "ENN": "edited NN (ENN)", "NCR": "neighbourhood cleaning (NCR)",
        "OSS": "one-sided selection (OSS)", "CNN": "condensed NN (CNN)",
        "IHT": "instance hardness (IHT)", "CC": "cluster centroids (CC)",
        "MUS-BS": "MUS-BS (boundary, stable)", "MUS-BO": "MUS-BO (boundary, optimal)",
        "MUS-CS": "MUS-CS (coverage, stable)", "MUS-CO": "MUS-CO (coverage, optimal)",
        "SMOTE": "SMOTE (oversampling reference)"}
# contrast key, minuend, subtrahend, factor, held fixed, factorial-table contrast label
CONTRASTS = [
    ("arm_stable", "MUS-CS", "MUS-BS", "arm (coverage minus boundary)", "stable concept", "MUS-BS vs MUS-CS"),
    ("arm_optimal", "MUS-CO", "MUS-BO", "arm (coverage minus boundary)", "optimal concept", "MUS-BO vs MUS-CO"),
    ("concept_boundary", "MUS-BO", "MUS-BS", "concept (optimal minus stable)", "boundary arm", "MUS-BS vs MUS-BO"),
    ("concept_coverage", "MUS-CO", "MUS-CS", "concept (optimal minus stable)", "coverage arm", "MUS-CS vs MUS-CO"),
]
MINUS = "\u2212"


def col(m):
    return COL.get(m, OTHER_GREY)


def tcol(m):
    """Text colour: the key colour, with greys lighter than #767676 darkened to #767676."""
    r, g, b = mpl.colors.to_rgb(col(m))
    if abs(r - g) < 1e-9 and abs(g - b) < 1e-9 and r > 118 / 255:
        return "#767676"
    return col(m)


def edge(m):
    """Marker edge: light greys get a darker edge so they stay visible; others a thin white edge."""
    r, g, b = mpl.colors.to_rgb(col(m))
    return DARK_EDGE if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.6 else "white"


def tint(c, a=0.33):
    rgb = np.array(mpl.colors.to_rgb(c))
    return tuple(a * rgb + (1 - a) * np.ones(3))


def apply_style(style_fn=None):
    if style_fn is not None:
        style_fn()
    mpl.rcParams.update(STYLE_RC)
    mpl.rcParams["hatch.linewidth"] = HATCH_LW
    mpl.rcParams["savefig.pad_inches"] = SAVE_PAD


def panel_letter(ax, letter, dx=-0.18, dy=1.02, fontsize=None):
    """figure-style 5.7: bold lower-case panel letter outside the top-left of the axes."""
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    ax.text(dx, dy, letter.lower(), transform=ax.transAxes, fontweight="bold", fontsize=fontsize,
            va="bottom", ha="left")


def axes_in(fig, x0, y0, w, h, **kw):
    """Axes placed in inches from the figure's bottom-left corner."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}" if p < 0.1 else f"= {p:.2f}"


# ------------------------------------------------------------------------------------------
# Number registry
# ------------------------------------------------------------------------------------------
class Registry:
    def __init__(self):
        self.rows = []

    def add(self, figure, panel, quantity, value, *, method="", classifier="", contrast="",
            ci_lo=np.nan, ci_hi=np.nan, n=np.nan, unit="", source="", note=""):
        self.rows.append(dict(figure=figure, panel=panel, quantity=quantity, method=method,
                              classifier=classifier, contrast=contrast, value=value, ci_lo=ci_lo,
                              ci_hi=ci_hi, n=n, unit=unit, source=source, note=note))

    def frame(self):
        return pd.DataFrame(self.rows)


def _band_entry(lw, rope):
    """Smallest x from which the sorted LOWESS curve stays inside [-rope, rope] (linear interpolation)."""
    x, y = lw[:, 0], lw[:, 1]
    inside = np.abs(y) <= rope
    if not inside[-1]:
        return float("nan")
    k = len(y) - 1
    while k > 0 and inside[k - 1]:
        k -= 1
    if k == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[k - 1], x[k], y[k - 1], y[k]
    target = -rope if y0 < -rope else rope
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)


def boot_matrix(n, B=B_BOOT, seed=SEED_BOOT):
    """One common resampling matrix over datasets, used for every bootstrap interval."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def boot_ci(d, idx):
    d = np.asarray(d, float)
    bm = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


# ------------------------------------------------------------------------------------------
# Compute layer
# ------------------------------------------------------------------------------------------
def load_inputs(inp):
    T = {}
    for k, p in inp.items():
        if k == "matchsampling":
            continue
        T[k] = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    src = {k: os.path.basename(p) for k, p in inp.items()}
    return T, src


def toy_selections(match_path):
    from imblearn.under_sampling import NearMiss, RandomUnderSampler
    spec = importlib.util.spec_from_file_location("matchsampling", match_path)
    msm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(msm)
    rng = np.random.default_rng(7)
    Xmaj = rng.normal(size=(520, 2)) * [1.0, 0.85]
    theta = rng.uniform(0.15 * np.pi, 0.85 * np.pi, 58)
    Xmin = np.c_[1.25 * np.cos(theta), 0.35 + np.sin(theta)] + rng.normal(scale=0.12, size=(58, 2))
    X = np.vstack([Xmaj, Xmin])
    y = np.r_[np.zeros(len(Xmaj), int), np.ones(len(Xmin), int)]
    samplers = {"RUS": RandomUnderSampler(random_state=0), "NM1": NearMiss(version=1),
                "MUS-BS": msm.MatchingUnderSampler("boundary", "stable", random_state=0),
                "MUS-CS": msm.MatchingUnderSampler("coverage", "stable", random_state=0)}
    kept = {}
    for k, s in samplers.items():
        s.fit_resample(X, y)
        idx = np.asarray(s.sample_indices_)
        kept[k] = idx[idx < len(Xmaj)]
        assert (idx >= len(Xmaj)).sum() == len(Xmin), f"{k} altered the minority class"
    return Xmaj, Xmin, kept


def compute(inp):
    T, src = load_inputs(inp)
    R = Registry()
    C = {"src": src, "reg": R}
    dm = T["dsmeans"]
    datasets = sorted(dm.dataset.unique())
    assert len(datasets) == N_EXPECTED, len(datasets)
    C["datasets"] = datasets
    n = len(datasets)
    idx = boot_matrix(n)

    def piv(metric):
        return dm.pivot_table(index="dataset", columns=["classifier", "sampler"], values=metric).loc[datasets]

    G, SENS, SPEC = piv("gmean"), piv("sens"), piv("spec")

    def clfavg_delta(P, m, ref="RUS"):
        return pd.concat([P[(c, m)] - P[(c, ref)] for c in CLFS], axis=1).mean(axis=1)

    s_dm, s_fac = src["dsmeans"], src["factorial"]
    # ---------------- F1b factor effects ----------------
    fac = T["factorial"]
    rows = []
    for clf in CLFS:
        for key, a, b, factor, fixed, lab in CONTRASTS:
            d = (G[(clf, a)] - G[(clf, b)]).values
            m, lo, hi = boot_ci(d, idx)
            fr = fac[(fac.metric == "gmean") & (fac.classifier == clf) & (fac.contrast == lab)].iloc[0]
            oriented = fr.mean_diff if fr["first"] == a else -fr.mean_diff
            assert abs(oriented - m) < 1e-9, (clf, key, oriented, m)
            W, Ti, L = (fr.wins, fr.ties, fr.losses) if fr["first"] == a else (fr.losses, fr.ties, fr.wins)
            w2, l2 = int((d > TIE_EPS).sum()), int((d < -TIE_EPS).sum())
            assert (w2, n - w2 - l2, l2) == (int(W), int(Ti), int(L)), (clf, key)
            rows.append(dict(classifier=clf, contrast=key, minuend=a, subtrahend=b, factor=factor,
                             held_fixed=fixed, n_datasets=n, mean_diff=m, ci_lo=lo, ci_hi=hi,
                             median_diff=float(np.median(d)), wins_minuend=int(W), ties=int(Ti),
                             losses_minuend=int(L), p_holm=float(fr.p_holm), sig_holm=bool(fr.sig_holm),
                             holm_family_size=int(fr.holm_family_size), verdict=fr.verdict))
            lbl = f"{a} {MINUS} {b}"
            R.add("fig1_hook", "b", "mean paired G-mean difference", m, classifier=clf, contrast=lbl,
                  ci_lo=lo, ci_hi=hi, n=n, unit="G-mean",
                  source=f"{s_dm}; percentile bootstrap over datasets B={B_BOOT} seed {SEED_BOOT}",
                  note=f"{factor}; held fixed: {fixed}")
            R.add("fig1_hook", "b", "wins/ties/losses of minuend (tie |d|<=0.001)",
                  f"{int(W)}/{int(Ti)}/{int(L)}", classifier=clf, contrast=lbl, n=n, source=s_fac)
            R.add("fig1_hook", "b", "Wilcoxon signed-rank Holm-adjusted p", float(fr.p_holm), classifier=clf,
                  contrast=lbl, n=n, source=s_fac,
                  note=f"Holm family = {int(fr.holm_family_size)} factorial contrasts per classifier; verdict: {fr.verdict}")
    C["F1b"] = pd.DataFrame(rows)
    arm = C["F1b"][C["F1b"].contrast.str.startswith("arm")]
    con = C["F1b"][C["F1b"].contrast.str.startswith("concept")]
    R.add("fig1_hook", "b", "range of arm-effect means over classifiers and concepts",
          f"{arm.mean_diff.min():.4f} to {arm.mean_diff.max():.4f}", n=10, unit="G-mean", source=s_dm)
    R.add("fig1_hook", "b", "max |concept-effect mean| over classifiers and arms",
          float(con.mean_diff.abs().max()), n=10, unit="G-mean", source=s_dm)
    R.add("fig1_hook", "b", "concept-effect CIs that include 0", int(((con.ci_lo <= 0) & (con.ci_hi >= 0)).sum()),
          n=10, source=s_dm)
    R.add("fig1_hook", "b", "arm contrasts significant after Holm", int(arm.sig_holm.sum()), n=10, source=s_fac)
    R.add("fig1_hook", "b", "concept contrasts significant after Holm", int(con.sig_holm.sum()), n=10, source=s_fac)

    # ---------------- F1c per-dataset differences vs RUS ----------------
    D1c = pd.DataFrame({m: clfavg_delta(G, m) for m in MUS})
    gd = T["gains_ds"]
    chk = gd[(gd.metric == "gmean") & (gd.classifier == "mean")].pivot(index="dataset", columns="method",
                                                                        values="delta_rus").loc[datasets]
    assert float((D1c - chk[MUS]).abs().max().max()) < 1e-9
    C["D1c"] = D1c
    rows = []
    for m in MUS:
        d = D1c[m]
        r = dict(method=m, n_datasets=n, below=int((d < -ROPE).sum()),
                 within=int(((d >= -ROPE) & (d <= ROPE)).sum()), above=int((d > ROPE).sum()),
                 mean=float(d.mean()), median=float(d.median()), min=float(d.min()), max=float(d.max()),
                 min_dataset=d.idxmin(), max_dataset=d.idxmax())
        rows.append(r)
        for q in ["below", "within", "above"]:
            R.add("fig1_hook", "c", f"datasets {q} the ROPE (+-{ROPE}) of G-mean difference vs RUS", r[q],
                  method=m, n=n, unit="datasets", source=s_dm,
                  note="classifier-averaged (5 classifiers) per-dataset difference; below: d<-0.01, within: |d|<=0.01, above: d>0.01")
        for q in ["mean", "median", "min", "max"]:
            R.add("fig1_hook", "c", f"{q} classifier-averaged G-mean difference vs RUS", r[q], method=m, n=n,
                  unit="G-mean", source=s_dm,
                  note=(f"dataset: {r['min_dataset']}" if q == "min" else f"dataset: {r['max_dataset']}" if q == "max" else ""))
    C["F1c"] = pd.DataFrame(rows)

    # ---------------- F1a toy ----------------
    C["toy"] = toy_selections(inp["matchsampling"])
    Xmaj, Xmin, kept = C["toy"]
    from scipy.spatial.distance import cdist
    dP = cdist(Xmaj, Xmin).min(1)
    Dn = cdist(Xmaj, Xmaj)
    np.fill_diagonal(Dn, np.inf)
    dN1 = Dn.min(1)
    np.fill_diagonal(Dn, 0.0)
    C["toy_diag"] = {k: (float(dP[kk].mean() / dP.mean()), float(Dn[:, kk].min(1).mean() / dN1.mean())) for k, kk in kept.items()}
    for k, kk in kept.items():
        R.add("fig1_hook", "a", "toy: retained majority instances", len(kk), method=k, n=len(Xmaj),
              source="toy data (rng seed 7), sampler random_state=0", note="raw 2-D coordinates, no scaling")
        R.add("fig1_hook", "a", "toy: boundary proximity BP of the retained majority",
              float(dP[kk].mean() / dP.mean()), method=k, source="toy data", note="BP as in the mechanism pipeline")
        R.add("fig1_hook", "a", "toy: coverage error CE of the retained majority",
              float(Dn[:, kk].min(1).mean() / dN1.mean()), method=k, source="toy data")

    # ---------------- F2a Bayesian ----------------
    by = T["bayes"]
    F2a = by[by.comparator_role == "RUS"].copy()
    assert set(F2a.comparator) == {"RUS"} and len(F2a) == 40
    C["F2a"] = F2a
    for _, r in F2a.iterrows():
        for q, lab in [("p_mus_better", "P(MUS better)"), ("p_rope", "P(practically equivalent)"),
                       ("p_mus_worse", "P(MUS worse)")]:
            R.add("fig2_evidence", "a", f"{lab} vs RUS, {r.metric}", float(r[q]), method=r.mus,
                  classifier=r.classifier, n=int(r.n_datasets), unit="posterior probability",
                  source=src["bayes"], note=f"Bayesian signed-rank, ROPE {r.rope}, prior {r.prior}, {int(r.nsamples)} samples, seed {int(r.seed)}; verdict: {r.bayes_verdict}")
    g = F2a[F2a.metric == "gmean"]
    for m in MUS:
        gm = g[g.mus == m]
        R.add("fig2_evidence", "a", "classifiers with P(practically equivalent) >= 0.95 (G-mean)",
              int((gm.p_rope >= 0.95).sum()), method=m, n=5, source=src["bayes"])
        R.add("fig2_evidence", "a", "classifiers with P(MUS worse) >= 0.95 (G-mean)",
              int((gm.p_mus_worse >= 0.95).sum()), method=m, n=5, source=src["bayes"])
        am = F2a[(F2a.metric == "auc") & (F2a.mus == m)]
        R.add("fig2_evidence", "a", "classifiers with P(practically equivalent) >= 0.95 (ROC-AUC)",
              int((am.p_rope >= 0.95).sum()), method=m, n=5, source=src["bayes"])
        R.add("fig2_evidence", "a", "classifiers with P(MUS worse) >= 0.95 (ROC-AUC)",
              int((am.p_mus_worse >= 0.95).sum()), method=m, n=5, source=src["bayes"])
        R.add("fig2_evidence", "a", "min P(practically equivalent) over classifiers (G-mean)", float(gm.p_rope.min()),
              method=m, n=5, source=src["bayes"])

    # ---------------- F2b ranks + Wilcoxon ----------------
    rk = T["ranks"]
    RMs = {}
    for metric in ["gmean", "auc"]:
        Rm = rk[(rk.condition == "all") & (rk.metric == metric)].set_index("classifier")
        RMs[metric] = (pd.DataFrame({c: [Rm.loc[c, f"rank_{m}"] for m in ORDER16] for c in CLFS + ["pooled"]},
                                    index=ORDER16), Rm)
    C["ranks"] = RMs
    RM, Rg = RMs["gmean"]
    wx = T["wilcoxon"]
    W = wx[(wx.metric == "gmean") & (wx.baseline == "RUS") & (wx.condition == "all")].set_index(["mus", "classifier"])
    C["W_rus"] = W
    for m in ORDER16:
        for c in CLFS:
            R.add("fig2_evidence", "b", "average G-mean rank (1 = best, k = 16)", float(RM.loc[m, c]), method=m,
                  classifier=c, n=int(Rg.loc[c, "N"]), source=src["ranks"])
    for m in MUS:
        for c in CLFS:
            w = W.loc[(m, c)]
            R.add("fig2_evidence", "b", "Wilcoxon signed-rank vs RUS, Holm-adjusted p (G-mean)", float(w.p_holm),
                  method=m, classifier=c, n=int(w.n), source=src["wilcoxon"],
                  note=f"Holm family = {int(w.holm_family_size)} (4 MUS x 12 baselines per classifier); verdict: {w.verdict}; W/T/L {int(w.wins)}/{int(w.ties)}/{int(w.losses)}")
    gaps = []
    for c in CLFS:
        cdv = float(Rg.loc[c, "CD"])
        for m in MUS:
            gap = float(RM.loc[m, c] - RM.loc["RUS", c])
            gaps.append(dict(classifier=c, method=m, gap=gap, inside_cd=abs(gap) < cdv, cd=cdv))
            R.add("figS_cd_gmean", c, "average G-mean rank gap to RUS (MUS minus RUS)", gap, method=m, classifier=c,
                  n=55, source=src["ranks"], note=f"inside Nemenyi CD {cdv:.2f}: {abs(gap) < cdv}")
    C["cd_gaps"] = pd.DataFrame(gaps)
    bg = C["cd_gaps"][C["cd_gaps"].method.isin(["MUS-BS", "MUS-BO"])]
    R.add("figS_cd_gmean", "-", "boundary-arm vs RUS G-mean rank gaps inside the Nemenyi CD", int(bg.inside_cd.sum()),
          n=len(bg), source=src["ranks"], note="Holm-Wilcoxon finds all 10 significant (Fig. 2b)")
    for c in CLFS:
        best = RM[c].idxmin()
        R.add("fig2_evidence", "b", "best average G-mean rank", f"{best} ({RM.loc[best, c]:.2f})", classifier=c,
              n=55, source=src["ranks"])
        R.add("fig2_evidence", "b", "Friedman p (G-mean, k = 16)", float(Rg.loc[c, "p_chi2"]), classifier=c, n=55,
              source=src["ranks"])

    # ---------------- F3a map ----------------
    dd = T["diag_ds"]
    dsum = T["diag_sum"].set_index("sampler")
    rows = []
    for s in BAL9:
        x = dd[dd.sampler == s].set_index("dataset").loc[datasets]
        r = dict(method=s, n_datasets=len(x))
        for v in ["BP", "CE"]:
            q1, med, q3 = np.percentile(x[v], [25, 50, 75])
            assert np.allclose([med, q1, q3], [dsum.loc[s, f"{v}_median"], dsum.loc[s, f"{v}_q1"], dsum.loc[s, f"{v}_q3"]])
            r.update({f"{v}_median": med, f"{v}_q1": q1, f"{v}_q3": q3})
            R.add("fig3_mechanism", "a", f"median {v} over datasets", float(med), method=s, ci_lo=float(q1),
                  ci_hi=float(q3), n=len(x), source=src["diag_ds"], note="ci_lo/ci_hi = first/third quartile")
        r["n_not_balanced"] = int((np.abs(x.size_ratio - 1) > 1e-9).sum())
        r["size_ratio_min"] = float(x.size_ratio.min())
        rows.append(r)
        R.add("fig3_mechanism", "a", "datasets where |S|/|P| != 1 (retained majority / minority)", r["n_not_balanced"],
              method=s, n=len(x), source=src["diag_ds"], note=f"minimum size ratio {r['size_ratio_min']:.3f}")
    C["F3a"] = pd.DataFrame(rows)

    # ---------------- F3b failure routes ----------------
    rows = []
    for m in B8:
        for c in CLFS:
            dsn = SENS[(c, m)] - SENS[(c, "RUS")]
            dsp = SPEC[(c, m)] - SPEC[(c, "RUS")]
            rows.append(dict(method=m, classifier=c, n_datasets=n, d_sens=float(dsn.mean()), d_spec=float(dsp.mean()),
                             n_sens_loss=int((dsn < -TIE_EPS).sum()), n_spec_loss=int((dsp < -TIE_EPS).sum())))
            R.add("fig3_mechanism", "b", "mean change in sensitivity vs RUS", float(dsn.mean()), method=m, classifier=c,
                  n=n, source=s_dm)
            R.add("fig3_mechanism", "b", "mean change in specificity vs RUS", float(dsp.mean()), method=m, classifier=c,
                  n=n, source=s_dm)
        dsn, dsp = clfavg_delta(SENS, m), clfavg_delta(SPEC, m)
        ms_, lo_s, hi_s = boot_ci(dsn, idx)
        mp_, lo_p, hi_p = boot_ci(dsp, idx)
        rows.append(dict(method=m, classifier="mean", n_datasets=n, d_sens=ms_, d_sens_lo=lo_s, d_sens_hi=hi_s,
                         d_spec=mp_, d_spec_lo=lo_p, d_spec_hi=hi_p))
        R.add("fig3_mechanism", "b", "mean change in sensitivity vs RUS (mean over classifiers)", ms_, method=m,
              classifier="mean of 5", ci_lo=lo_s, ci_hi=hi_s, n=n, source=f"{s_dm}; bootstrap over datasets B={B_BOOT}")
        R.add("fig3_mechanism", "b", "mean change in specificity vs RUS (mean over classifiers)", mp_, method=m,
              classifier="mean of 5", ci_lo=lo_p, ci_hi=hi_p, n=n, source=f"{s_dm}; bootstrap over datasets B={B_BOOT}")
    C["F3b"] = pd.DataFrame(rows)

    # ---------------- F3c diagnostic-performance link ----------------
    Gavg = pd.DataFrame({s: pd.concat([G[(c, s)] for c in CLFS], axis=1).mean(axis=1) for s in ORDER16})
    DX = {v: dd.pivot(index="dataset", columns="sampler", values=v).loc[datasets] for v in ["BP", "CE"]}
    dp, dpd = T["diag_perf"], T["diag_perf_ds"]
    rows, per = [], []
    for setname, meths in [("size_balanced", BAL9), ("boundary_cluster", BCL5)]:
        for a, b in [("gmean", "BP"), ("gmean", "CE"), ("BP", "CE")]:
            rh = []
            for dsn in datasets:
                xa = Gavg.loc[dsn, meths].values if a == "gmean" else DX[a].loc[dsn, meths].values
                rh.append(spearmanr(xa, DX[b].loc[dsn, meths].values).statistic)
                per.append(dict(set=setname, pair=f"{a} vs {b}", dataset=dsn, rho=rh[-1]))
            rh = np.asarray(rh)
            q1, med, q3 = np.nanpercentile(rh, [25, 50, 75])
            r = dict(set=setname, n_methods=len(meths), pair=f"{a} vs {b}", n_datasets=int(np.isfinite(rh).sum()),
                     rho_median=med, rho_q1=q1, rho_q3=q3, frac_pos=float((rh > 0).mean()), frac_neg=float((rh < 0).mean()))
            note = "computed here (absent from the link tables)"
            if a == "gmean":
                t = dp[(dp.set == setname) & (dp.classifier == "mean") & (dp.diagnostic == b)].iloc[0]
                tt = dpd[(dpd.set == setname) & (dpd.classifier == "mean") & (dpd.diagnostic == b)].set_index("dataset").loc[datasets, "rho"].values
                assert np.nanmax(np.abs(tt - rh)) < 1e-9 and abs(t.rho_median - med) < 1e-9
                r["wilcoxon_p_holm"] = float(t.wilcoxon_p_holm)
                note = f"matches link table; Wilcoxon (rho = 0) Holm p {t.wilcoxon_p_holm:.2g}"
            rows.append(r)
            R.add("fig3_mechanism", "c", f"within-dataset Spearman rho, {a} vs {b} (median over datasets)", float(med),
                  method=f"{setname} ({len(meths)} samplers: {', '.join(meths)})", ci_lo=float(q1), ci_hi=float(q3),
                  n=r["n_datasets"], source=f"{src['diag_ds']}; {s_dm}",
                  note=f"ci_lo/ci_hi = IQR; G-mean = mean over 5 classifiers; {note}")
    C["F3c"], C["F3c_ds"] = pd.DataFrame(rows), pd.DataFrame(per)

    # ---------------- F3d coverage saturation ----------------
    dgl = T["diag_gain"]
    rows = []
    for m in ["MUS-CS", "MUS-CO"]:
        dCE = DX["CE"][m] - DX["CE"]["RUS"]
        dG = D1c[m]
        rr = spearmanr(dCE, dG)
        t = dgl[(dgl.method == m) & (dgl.metric == "gmean") & (dgl.diagnostic == "CE")].iloc[0]
        assert abs(t.rho - rr.statistic) < 1e-9
        rows.append(dict(method=m, n_datasets=n, rho=float(rr.statistic), p=float(rr.pvalue), p_holm=float(t.p_holm),
                         n_dCE_neg=int((dCE < 0).sum()), dCE_median=float(dCE.median()), dG_median=float(dG.median())))
        R.add("fig3_mechanism", "d", "Spearman rho, change in CE vs change in G-mean (both vs RUS)", float(rr.statistic),
              method=m, n=n, source=f"{src['diag_ds']}; {s_dm}",
              note=f"p = {rr.pvalue:.3f} (unadjusted); Holm p in diag_gain_link = {t.p_holm:.2f}")
        R.add("fig3_mechanism", "d", "datasets with lower CE than RUS", int((dCE < 0).sum()), method=m, n=n,
              source=src["diag_ds"])
        R.add("fig3_mechanism", "d", "median change in CE vs RUS", float(dCE.median()), method=m, n=n, source=src["diag_ds"])
        R.add("fig3_mechanism", "d", "median change in classifier-averaged G-mean vs RUS", float(dG.median()), method=m,
              n=n, source=s_dm)
        C.setdefault("dCE", {})[m] = dCE
    C["F3d"] = pd.DataFrame(rows)

    # ---------------- F4 safe share ----------------
    cx = T["complexity"].set_index("name").loc[datasets]
    safe = cx["safe"]
    C["safe"] = safe
    rows = []
    hi = safe >= SAFE_CUT
    for m in MUS:
        d = D1c[m]
        rr = spearmanr(safe, d)
        lw = lowess(d.values, safe.values, frac=LOWESS_FRAC, it=LOWESS_IT, return_sorted=True)
        r = dict(method=m, n_datasets=n, n_gain_gt_rope=int((d > ROPE).sum()),
                 datasets_gain_gt_rope=";".join(d[d > ROPE].index), n_safe_hi=int(hi.sum()),
                 mean_gain_safe_hi=float(d[hi].mean()), median_gain_safe_hi=float(d[hi].median()),
                 max_gain_safe_hi=float(d[hi].max()), n_within_safe_hi=int((d[hi].abs() <= ROPE).sum()),
                 n_below_safe_hi=int((d[hi] < -ROPE).sum()), n_above_safe_hi=int((d[hi] > ROPE).sum()),
                 mean_gain_safe_lo=float(d[~hi].mean()), n_within_safe_lo=int((d[~hi].abs() <= ROPE).sum()),
                 rho_safe=float(rr.statistic), p_safe=float(rr.pvalue),
                 lowess_at_1=float(np.interp(1.0, lw[:, 0], lw[:, 1])), lowess_at_0=float(np.interp(0.0, lw[:, 0], lw[:, 1])),
                 lowess_enters_band=_band_entry(lw, ROPE))
        rows.append(r)
        sf = f"{s_dm}; {src['complexity']}"
        R.add("fig4_when", "-", "datasets with classifier-averaged G-mean gain over RUS > +0.01", r["n_gain_gt_rope"],
              method=m, n=n, source=sf, note=f"datasets: {r['datasets_gain_gt_rope']}")
        R.add("fig4_when", "-", f"mean gain over RUS on datasets with safe-minority share >= {SAFE_CUT}",
              r["mean_gain_safe_hi"], method=m, n=r["n_safe_hi"], unit="G-mean", source=sf,
              note=f"median {r['median_gain_safe_hi']:.4f}, max {r['max_gain_safe_hi']:.4f}; within ROPE {r['n_within_safe_hi']}, below {r['n_below_safe_hi']}, above {r['n_above_safe_hi']} of {r['n_safe_hi']}")
        R.add("fig4_when", "-", f"mean gain over RUS on datasets with safe-minority share < {SAFE_CUT}",
              r["mean_gain_safe_lo"], method=m, n=n - r["n_safe_hi"], unit="G-mean", source=sf,
              note=f"within ROPE on {r['n_within_safe_lo']} of {n - r['n_safe_hi']}")
        R.add("fig4_when", "-", "Spearman rho, safe-minority share vs gain over RUS", r["rho_safe"], method=m, n=n,
              source=sf, note=f"p = {r['p_safe']:.2g} (unadjusted)")
        R.add("fig4_when", "-", "smallest safe share from which the LOWESS curve stays within +-0.01", r["lowess_enters_band"],
              method=m, n=n, source=sf, note="linear interpolation between LOWESS support points; nan = never enters")
        R.add("fig4_when", "-", "LOWESS fit at safe share 1.0 / 0.0", f"{r['lowess_at_1']:.4f} / {r['lowess_at_0']:.4f}",
              method=m, n=n, source=sf, note=f"statsmodels lowess frac={LOWESS_FRAC:.3f}, it={LOWESS_IT}")
    bavg = (D1c["MUS-BS"] + D1c["MUS-BO"]) / 2
    R.add("fig4_when", "-", "datasets where either boundary-arm method beats RUS by > 0.01",
          int(((D1c["MUS-BS"] > ROPE) | (D1c["MUS-BO"] > ROPE)).sum()), method="MUS-BS or MUS-BO", n=n,
          source=s_dm)
    R.add("fig4_when", "-", f"boundary arm (mean of MUS-BS, MUS-BO) mean gain at safe share >= {SAFE_CUT}",
          float(bavg[hi].mean()), method="MUS-BS/MUS-BO average", n=int(hi.sum()), unit="G-mean", source=s_dm)
    C["F4"] = pd.DataFrame(rows)
    zero = safe == 0
    pos = ~zero
    f4z = dict(n_zero=int(zero.sum()), n_pos=int(pos.sum()))
    for m in ["MUS-BS", "MUS-BO"]:
        rr = spearmanr(safe[pos], D1c[m][pos])
        f4z[m] = dict(rho=float(rr.statistic), p=float(rr.pvalue))
    f4z["lowess_pos_BS"] = lowess(D1c["MUS-BS"][pos].values, safe[pos].values, frac=LOWESS_FRAC, it=LOWESS_IT,
                                  return_sorted=True)
    jit = pd.Series(0.0, index=safe.index)
    jit[zero] = np.random.default_rng(JITTER_SEED).uniform(-JITTER_HALF, JITTER_HALF, int(zero.sum()))
    f4z["jitter"] = jit                     # display only; every statistic uses the unjittered safe share
    C["F4z"] = f4z

    # ---------------- F5 cost-benefit ----------------
    rt = T["rt_sum"].set_index("sampler")
    pf = T["perfold"]
    tmed = pf.drop_duplicates(["dataset", "repeat", "fold", "sampler"]).groupby("sampler").t_resample.median()
    rows = []
    for m in US15:
        assert abs(tmed[m] - rt.loc[m, "t_median"]) < 1e-9, m
        rows.append(dict(method=m, t_median=float(rt.loc[m, "t_median"]), t_p95=float(rt.loc[m, "t_p95"]),
                         n_folds=int(rt.loc[m, "n_units"]), pooled_rank=float(RM.loc[m, "pooled"])))
        R.add("fig5_cost", "-", "median resampling time per training fold", float(rt.loc[m, "t_median"]), method=m,
              n=int(rt.loc[m, "n_units"]), unit="s", source=src["rt_sum"],
              note="55 datasets x 25 training folds; measured under parallel load")
        R.add("fig5_cost", "-", "pooled average G-mean rank (275 dataset-classifier blocks, k = 16; descriptive)",
              float(RM.loc[m, "pooled"]), method=m, n=275, source=src["ranks"])
    C["F5"] = pd.DataFrame(rows)
    ratio = rt.loc["MUS-CO", "t_median"] / rt.loc["MUS-CS", "t_median"]
    ratio_b = rt.loc["MUS-BO", "t_median"] / rt.loc["MUS-BS", "t_median"]
    C["ratio_CO_CS"], C["ratio_BO_BS"] = float(ratio), float(ratio_b)
    shifts = {f"{b} - {a}": float(RM.loc[b, "pooled"] - RM.loc[a, "pooled"]) for a, b in [("MUS-BS", "MUS-BO"), ("MUS-CS", "MUS-CO")]}
    C["max_rank_shift"] = max(abs(v) for v in shifts.values())
    for k_, v_ in shifts.items():
        R.add("fig5_cost", "-", "pooled average G-mean rank shift, optimal minus stable", v_, method=k_, n=275,
              source=src["ranks"], note="negative = optimal ranks better")
    R.add("fig5_cost", "-", "median-time ratio MUS-CO / MUS-CS", float(ratio), method="MUS-CO/MUS-CS", unit="x",
          source=src["rt_sum"])
    R.add("fig5_cost", "-", "median-time ratio MUS-BO / MUS-BS", float(ratio_b), method="MUS-BO/MUS-BS", unit="x",
          source=src["rt_sum"])
    R.add("fig5_cost", "-", "95th-percentile time ratio MUS-CO / MUS-CS",
          float(rt.loc["MUS-CO", "t_p95"] / rt.loc["MUS-CS", "t_p95"]), method="MUS-CO/MUS-CS", unit="x",
          source=src["rt_sum"])

    # ---------------- supplementary CD diagrams ----------------
    for metric in ["gmean", "auc"]:
        RMm, Rmm = RMs[metric]
        for c in CLFS:
            R.add(f"figS_cd_{metric}", c, "Nemenyi critical difference", float(Rmm.loc[c, "CD"]), classifier=c,
                  n=int(Rmm.loc[c, "N"]), source=src["ranks"], note="k = 16, alpha = 0.05")
            R.add(f"figS_cd_{metric}", c, "Friedman p", float(Rmm.loc[c, "p_chi2"]), classifier=c,
                  n=int(Rmm.loc[c, "N"]), source=src["ranks"])
            for m in ORDER16:
                R.add(f"figS_cd_{metric}", c, f"average {metric} rank", float(RMm.loc[m, c]), method=m, classifier=c,
                      n=int(Rmm.loc[c, "N"]), source=src["ranks"])

    # ---------------- figS_runtime_grid ----------------
    rtd = T["rt_ds"]
    uu = pf.drop_duplicates(["dataset", "repeat", "fold", "sampler"])
    uu = uu[~uu["sampler"].isin(["NONE", "CW"])]
    per = uu.groupby(["sampler", "dataset"]).agg(t_med=("t_resample", "median"), n_maj=("n_maj_train", "median")).reset_index()
    mm = per.merge(rtd, on=["sampler", "dataset"], suffixes=("", "_tab"))
    assert len(mm) == len(rtd) and float((mm.t_med - mm.t_med_tab).abs().max()) < 1e-12 \
        and float((mm.n_maj - mm.n_maj_tab).abs().max()) < 1e-12
    rows = []
    for s in ORDER_RT:
        x = rtd[rtd.sampler == s]
        lr = linregress(np.log10(x.n_maj), np.log10(x.t_med.clip(lower=1e-6)))
        tc = student_t.ppf(0.975, len(x) - 2)
        lo, hi = lr.slope - tc * lr.stderr, lr.slope + tc * lr.stderr
        tm = float(uu[uu.sampler == s].t_resample.median())
        ref = rt.loc[s]
        assert max(abs(lr.slope - ref.slope), abs(lo - ref.slope_ci_lo), abs(hi - ref.slope_ci_hi),
                   abs(lr.intercept - ref.intercept), abs(tm - ref.t_median)) < 1e-9, s
        rows.append(dict(method=s, n_datasets=len(x), slope=float(lr.slope), slope_lo=float(lo), slope_hi=float(hi),
                         intercept=float(lr.intercept), r2=float(lr.rvalue ** 2), t_median=tm, n_folds=int(ref.n_units)))
        R.add("figS_runtime_grid", s, "log-log slope of per-dataset median time per training fold on median training-fold majority size",
              float(lr.slope), method=s, ci_lo=float(lo), ci_hi=float(hi), n=len(x), source=f"{src['rt_ds']}; matches {src['rt_sum']}",
              note=f"OLS on log10 values, 95% t interval; R^2 = {lr.rvalue ** 2:.2f}")
        R.add("figS_runtime_grid", s, "median resampling time per training fold", tm, method=s, n=int(ref.n_units), unit="s",
              source=f"{src['perfold']}; matches {src['rt_sum']}", note="measured under parallel load")
    C["RTG"], C["rt_ds"] = pd.DataFrame(rows), rtd

    # ---------------- figS_map_full ----------------
    rows = []
    for s in ORDER16:
        x = dd[dd.sampler == s].set_index("dataset").loc[datasets]
        r = dict(method=s, size_controlled=s in BAL9)
        for v in ["BP", "CE", "size_ratio"]:
            med = float(np.median(x[v]))
            assert abs(med - dsum.loc[s, f"{v}_median"]) < 1e-9, (s, v)
            r[f"{v}_median"] = med
        rows.append(r)
        for v, q in [("BP", "median boundary proximity BP"), ("CE", "median coverage error CE"),
                     ("size_ratio", "median output-size ratio |S|/|P|")]:
            R.add("figS_map_full", "-", q, r[f"{v}_median"], method=s, n=len(x), source=f"{src['diag_sum']} (recomputed from {src['diag_ds']})",
                  note="output size fixed at |S| = |P|" if s in BAL9 else "output size not controlled")
    C["MAPF"] = pd.DataFrame(rows)

    # ---------------- fig4 revision: point mass at safe share 0 (rows appended last) ----------------
    f4z = C["F4z"]
    sf4 = f"{s_dm}; {src['complexity']}"
    R.add("fig4_when", "-", "datasets with safe-minority share exactly 0", f4z["n_zero"], n=n, unit="datasets",
          source=src["complexity"],
          note=f"point mass at x = 0, drawn with seeded horizontal jitter (uniform +-{JITTER_HALF}, seed {JITTER_SEED}); display only")
    for m in ["MUS-BS", "MUS-BO"]:
        R.add("fig4_when", "-", "Spearman rho, safe-minority share vs gain over RUS, excluding datasets with safe share 0",
              f4z[m]["rho"], method=m, n=f4z["n_pos"], source=sf4, note=f"p = {f4z[m]['p']:.2g} (unadjusted)")
    lwz = f4z["lowess_pos_BS"]
    R.add("fig4_when", "-", "LOWESS fitted without safe-share-0 datasets: value at smallest positive safe share / at 1.0",
          f"{lwz[0, 1]:.4f} / {np.interp(1.0, lwz[:, 0], lwz[:, 1]):.4f}", method="MUS-BS", n=f4z["n_pos"], source=sf4,
          note=f"dashed line; smallest positive safe share {lwz[0, 0]:.4f}; statsmodels lowess frac={LOWESS_FRAC:.3f}, it={LOWESS_IT}")

    # ---------------- plot data (per dataset) ----------------
    pdx = []
    for m in MUS:
        for dsn in datasets:
            pdx.append(dict(figure="fig1_hook", panel="c", dataset=dsn, method=m, x=np.nan,
                            y=float(D1c.loc[dsn, m]), y_name="classifier-averaged G-mean difference vs RUS"))
            pdx.append(dict(figure="fig4_when", panel="-", dataset=dsn, method=m, x=float(safe[dsn]),
                            x_name="safe-minority share", y=float(D1c.loc[dsn, m]),
                            y_name="classifier-averaged G-mean difference vs RUS"))
    for m in ["MUS-CS", "MUS-CO"]:
        for dsn in datasets:
            pdx.append(dict(figure="fig3_mechanism", panel="d", dataset=dsn, method=m,
                            x=float(C["dCE"][m][dsn]), x_name="CE change vs RUS",
                            y=float(D1c.loc[dsn, m]), y_name="classifier-averaged G-mean difference vs RUS"))
    C["plotdata"] = pd.DataFrame(pdx)
    return C


# ------------------------------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------------------------------
def ax_text_in(fig, ax, x_in, y_in, s, **kw):
    """Text owned by `ax`, positioned in figure inches (converted to axes fraction)."""
    W, H = fig.get_size_inches()
    p = ax.get_position()
    xf = (x_in / W - p.x0) / p.width
    yf = (y_in / H - p.y0) / p.height
    return ax.text(xf, yf, s, transform=ax.transAxes, **kw)


def letter_in(fig, ax, x_in, y_in, letter):
    """Panel letter (bold, one step above base size), top-left of the panel block, owned by `ax`."""
    return ax_text_in(fig, ax, x_in, y_in, letter, fontweight="bold", fontsize=FS_BASE + 1, ha="left", va="top")


def title_in(fig, ax, x_in, y_in, s):
    return ax_text_in(fig, ax, x_in, y_in, s, fontsize=FS_BASE, ha="left", va="top", linespacing=1.15)


def two_tone(ax, x, y, c_left, c_right, ms=5.6, zorder=3):
    ax.plot([x], [y], ls="none", marker="o", ms=ms, fillstyle="left", markerfacecolor=c_left,
            markerfacecoloralt=c_right, markeredgecolor="#262626", markeredgewidth=0.45, zorder=zorder)


def two_tone_handle(c_left, c_right, ms=5.6):
    return Line2D([], [], ls="none", marker="o", ms=ms, fillstyle="left", markerfacecolor=c_left,
                  markerfacecoloralt=c_right, markeredgecolor="#262626", markeredgewidth=0.45)


def leader(ax, xy_text, xy_point, s, color, ha="left", va="center", fontsize=FS_SEC, bold=False, lw=0.5):
    """Direct label with a thin leader from the text anchor to the point (data coordinates)."""
    ax.annotate(s, xy=xy_point, xytext=xy_text, textcoords="data", ha=ha, va=va, fontsize=fontsize,
                color=color, fontweight="bold" if bold else "normal",
                arrowprops=dict(arrowstyle="-", color=color, lw=lw, shrinkA=1.5, shrinkB=3.0), zorder=6)


def check_overlaps(fig):
    """figure-style 9.1 geometric check: overlapping visible text boxes, text over spines, text
    outside the figure. Tick labels of hidden axes / ticks outside the view limits are excluded."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden, tl_vis = set(), {}
    for ax in fig.axes:
        ticks = set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"))
        vis = set()
        if ax.axison:
            for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                lo, hi = min(lim), max(lim)
                for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels()):
                    if lo - 1e-9 <= loc <= hi + 1e-9 and lab.get_visible():
                        vis.add(lab)
        hidden |= ticks - vis
        tl_vis[ax] = vis
    texts = [(t, mpl.text.Text.get_window_extent(t, r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in hidden]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values()
              if s.get_visible() and ax.axison]
    bad = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts)
           for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    bad += [(t.get_text(), "spine") for t, bt in texts for s, bs in spines
            if bt.overlaps(bs) and t not in tl_vis[s.axes]]
    fb = fig.bbox
    outside = [t.get_text() for t, bt in texts
               if bt.x0 < fb.x0 - 1 or bt.y0 < fb.y0 - 1 or bt.x1 > fb.x1 + 1 or bt.y1 > fb.y1 + 1]
    return bad, outside


# ------------------------------------------------------------------------------------------
# Figure 1 -- hook
# ------------------------------------------------------------------------------------------
TOY = ["RUS", "NM1", "MUS-BS", "MUS-CS"]
TOY_TITLE = {"RUS": "random undersampling (RUS)", "NM1": "NearMiss-1 (NM1)",
             "MUS-BS": "MUS-BS (boundary, stable)", "MUS-CS": "MUS-CS (coverage, stable)"}


def fig1(C):
    W, H = 7.2, 5.45
    fig = plt.figure(figsize=(W, H))
    Xmaj, Xmin, kept = C["toy"]
    xl, yl = (-3.45, 2.8), (-2.6, 2.35)
    pw, gap = 1.58, 0.105
    ph = pw * (yl[1] - yl[0]) / (xl[1] - xl[0])
    y0a = 3.78
    axa = []
    for j, m in enumerate(TOY):
        ax = axes_in(fig, 0.42 + j * (pw + gap), y0a, pw, ph)
        axa.append(ax)
        mask = np.zeros(len(Xmaj), bool)
        mask[kept[m]] = True
        ax.scatter(Xmaj[~mask, 0], Xmaj[~mask, 1], s=3, c="#CFCFCF", linewidths=0, zorder=1)
        ax.scatter(Xmaj[mask, 0], Xmaj[mask, 1], s=14, c=col(m), edgecolors="white", linewidths=0.35, zorder=2)
        ax.scatter(Xmin[:, 0], Xmin[:, 1], s=15, marker="+", c=MINORITY_COL, linewidths=0.75, zorder=3)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("#BDBDBD")
            s.set_linewidth(0.5)
        ax.set_title(TOY_TITLE[m], color=tcol(m), pad=2.5, fontsize=FS_BASE, loc="left")
    top_a = y0a + ph
    letter_in(fig, axa[0], 0.05, top_a + 0.40, "a")
    title_in(fig, axa[0], 0.24, top_a + 0.40,
             "Which majority instances each sampler keeps (toy data: 520 majority, 58 minority)")
    h_min = Line2D([], [], ls="none", marker="+", ms=5, mew=0.8, color=MINORITY_COL)
    h_keep = tuple(Line2D([], [], ls="none", marker="o", ms=3.8, mfc=col(m), mec="white", mew=0.3) for m in TOY)
    h_disc = Line2D([], [], ls="none", marker="o", ms=2.2, mfc="#CFCFCF", mec="none")
    fig.legend([h_min, h_keep, h_disc],
               ["minority (58)", "majority kept (58; colour = sampler)", "majority discarded (462)"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, loc="upper center",
               bbox_to_anchor=(0.5, (y0a - 0.04) / H), ncol=3, fontsize=FS_SEC, handletextpad=0.4,
               columnspacing=1.8, borderaxespad=0.0, handlelength=2.6)

    # (b) factor effects
    yb0, hb = 0.42, 2.18
    axb = axes_in(fig, 0.52, yb0, 2.86, hb)
    F = C["F1b"]
    offs = {"arm_stable": -0.27, "arm_optimal": -0.09, "concept_boundary": 0.09, "concept_coverage": 0.27}
    for i, clf in enumerate(CLFS):
        for key, a, b, *_ in CONTRASTS:
            r = F[(F.classifier == clf) & (F.contrast == key)].iloc[0]
            yy = i + offs[key]
            axb.plot([r.ci_lo, r.ci_hi], [yy, yy], color="#4D4D4D", lw=0.8, solid_capstyle="butt", zorder=2)
            two_tone(axb, r.mean_diff, yy, col(a), col(b))
    axb.axvline(0, color=GRID_GREY, lw=0.6, zorder=1)
    for i in range(1, 5):
        axb.axhline(i - 0.5, color="#EDEDED", lw=0.5, zorder=0)
    axb.set_ylim(4.5, -1.3)
    axb.set_yticks(range(5))
    axb.set_yticklabels(CLFS)
    axb.tick_params(axis="y", length=0)
    axb.set_xlim(-0.016, 0.13)
    axb.set_xticks([0, 0.04, 0.08, 0.12])
    axb.set_xticklabels(["0", "0.04", "0.08", "0.12"])
    axb.set_xlabel("G-mean difference (mean over 55 datasets, 95% CI)")
    axb.text(-0.0145, -0.83, f"concept\n(optimal {MINUS} stable)", ha="left", va="center", fontsize=FS_SEC,
             color="#404040", linespacing=1.1, zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    axb.text(0.077, -0.83, f"arm\n(coverage {MINUS} boundary)", ha="center", va="center", fontsize=FS_SEC,
             color="#404040", linespacing=1.1)
    hs = [two_tone_handle(col(a), col(b)) for _, a, b, *_ in CONTRASTS]
    labs = [f"{a} {MINUS} {b}" for _, a, b, *_ in CONTRASTS]
    order = [0, 2, 1, 3]
    axb.legend([hs[i] for i in order], [labs[i] for i in order], ncol=2, loc="lower left",
               bbox_to_anchor=(-0.02, 1.0), fontsize=FS_SEC, handletextpad=0.2, columnspacing=1.0,
               borderaxespad=0.15, handlelength=1.2)
    top_b = yb0 + hb + 0.36
    letter_in(fig, axb, 0.05, top_b + 0.34, "b")
    title_in(fig, axb, 0.24, top_b + 0.34,
             f"Switching arm moves G-mean by 0.05{chr(8211)}0.09;\nswitching concept by < 0.003")

    # (c) per-dataset profiles
    axc = axes_in(fig, 4.22, yb0, 2.86, hb + 0.36)
    axc.axhspan(-ROPE, ROPE, color=BAND_GREY, lw=0, zorder=0)
    axc.axhline(0, color=GRID_GREY, lw=0.5, zorder=1)
    n = len(C["datasets"])
    xx = np.arange(1, n + 1)
    cnt = C["F1c"].set_index("method")
    for m in ["MUS-BO", "MUS-BS", "MUS-CO", "MUS-CS"]:
        d = np.sort(C["D1c"][m].values)
        axc.plot(xx, d, color=col(m), lw=0.9, marker=MARK[m], ms=2.4, mew=0,
                 zorder=4 if m in ("MUS-BS", "MUS-CS") else 3)
    hdl = [Line2D([], [], color=col(m), lw=0.9, marker=MARK[m], ms=3, mew=0) for m in MUS]
    lab = [f"{m}:  {cnt.loc[m, 'below']} / {cnt.loc[m, 'within']} / {cnt.loc[m, 'above']}" for m in MUS]
    lg = axc.legend(hdl, lab, loc="lower right", fontsize=FS_SEC, handlelength=1.6, handletextpad=0.4,
                    title=f"datasets below / within / above {chr(177)}0.01", title_fontsize=FS_SEC,
                    borderaxespad=0.3)
    lg._legend_box.align = "left"
    axc.text(1.5, ROPE + 0.004, f"{chr(177)}0.01: practical equivalence", fontsize=FS_SEC, color="#6E6E6E",
             ha="left", va="bottom")
    axc.set_xlim(-0.5, n + 1.5)
    axc.set_xticks([1, 10, 20, 30, 40, 50, 55])
    axc.set_xlabel("datasets, sorted within each method (n = 55)")
    axc.set_ylabel("G-mean difference vs RUS\n(mean over 5 classifiers)")
    axc.margins(y=0.04)
    letter_in(fig, axc, 3.62, top_b + 0.34, "c")
    title_in(fig, axc, 3.80, top_b + 0.34,
             "Per dataset, the boundary arm mostly loses to RUS;\nthe coverage arm mostly ties")
    return fig


# ------------------------------------------------------------------------------------------
# Figure 2 -- evidence
# ------------------------------------------------------------------------------------------
def fig2(C):
    W, H = 7.2, 4.47
    fig = plt.figure(figsize=(W, H))
    F = C["F2a"]
    gap = 6.5
    ypos = {(m, c): g * gap + i for g, m in enumerate(MUS) for i, c in enumerate(CLFS)}
    y0, h = 0.45, 3.22
    axg = axes_in(fig, 0.55, y0, 1.30, h)                 # 0.16-in gap keeps the '1' and '0' end ticks apart
    axu = axes_in(fig, 2.01, y0, 1.30, h, sharey=axg)
    for ax, metric, ttl in [(axg, "gmean", "G-mean"), (axu, "auc", "ROC-AUC")]:
        sub = F[F.metric == metric].set_index(["mus", "classifier"])
        for m in MUS:
            c = col(m)
            for clf in CLFS:
                r = sub.loc[(m, clf)]
                yy = ypos[(m, clf)]
                pb, pr, pw_ = float(r.p_mus_better), float(r.p_rope), float(r.p_mus_worse)
                ax.barh(yy, pb, left=0, height=0.78, color=c, edgecolor=c, lw=0.4)
                ax.barh(yy, pr, left=pb, height=0.78, color=tint(c), edgecolor=c, lw=0.4)
                ax.barh(yy, pw_, left=pb + pr, height=0.78, color="white", edgecolor=c, lw=0.4, hatch="//////")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_title(ttl, pad=2.5, fontsize=FS_BASE, loc="center")
    axg.set_ylim(3 * gap + 4 + 0.6, -1.6)
    axg.set_yticks([ypos[(m, c)] for m in MUS for c in CLFS])
    axg.set_yticklabels(CLFS * 4)
    axg.tick_params(axis="y", length=0)
    axu.tick_params(axis="y", left=False, labelleft=False)
    axu.spines["left"].set_visible(False)
    axu.patch.set_visible(False)
    tr = mtransforms.blended_transform_factory(axg.transAxes, axg.transData)
    for g, m in enumerate(MUS):
        axg.text(0.015, g * gap - 1.0, NAME[m], color=tcol(m), fontweight="bold", fontsize=FS_SEC, ha="left",
                 va="center", transform=tr, zorder=6)
    ax_text_in(fig, axg, 1.93, 0.08, "posterior probability (MUS vs RUS, 55 datasets)", fontsize=FS_BASE,
               ha="center", va="bottom")
    grey = "#6E6E6E"
    hd = [Patch(facecolor=grey, edgecolor=grey, lw=0.4), Patch(facecolor=tint(grey), edgecolor=grey, lw=0.4),
          Patch(facecolor="white", edgecolor=grey, lw=0.4, hatch="//////")]
    fig.legend(hd, ["MUS better", f"practically equivalent (|{chr(916)}| < 0.01)", "MUS worse"], ncol=3,
               loc="lower left", bbox_to_anchor=(0.20 / W, (y0 + h + 0.20) / H), fontsize=FS_SEC,
               handlelength=1.4, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)
    letter_in(fig, axg, 0.05, H - 0.04, "a")
    title_in(fig, axg, 0.24, H - 0.04, "Coverage arm practically equivalent to RUS;\nboundary arm worse in G-mean")

    # (b) rank matrix
    RMf = C["ranks"]["gmean"][0]
    vals = RMf[CLFS].values
    x0h, y0h, wh, hh = 5.10, 1.12, 2.02, 2.55
    axh = axes_in(fig, x0h, y0h, wh, hh)
    cmap = plt.get_cmap("Purples_r")
    norm = mpl.colors.Normalize(vmin=1, vmax=16)
    im = axh.imshow(vals, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    W_rus = C["W_rus"]
    for i, m in enumerate(ORDER16):
        best_cols = set()
        for j, c in enumerate(CLFS):
            v = vals[i, j]
            rgb = cmap(norm(v))[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            tc = "white" if lum < 0.45 else "black"
            is_best = v == vals[:, j].min()
            sig = m in MUS and bool(W_rus.loc[(m, c), "sig_holm"])
            axh.text(j - (0.10 if sig else 0.0), i, f"{v:.2f}", ha="center", va="center", fontsize=FS_TICK,
                     color=tc, fontweight="bold" if is_best else "normal")
            if sig:
                worse = W_rus.loc[(m, c), "mean_diff"] < 0
                axh.text(j + 0.33, i, "\u25BC" if worse else "\u25B2", ha="center", va="center", fontsize=FS_TICK,
                         color=tc)
    axh.set_xticks(range(len(CLFS)))
    axh.set_xticklabels(CLFS, fontsize=FS_SEC)
    axh.xaxis.set_ticks_position("top")
    axh.xaxis.set_label_position("top")
    axh.tick_params(axis="both", length=0)
    axh.set_yticks(range(len(ORDER16)))
    axh.set_yticklabels([NAME[m] for m in ORDER16], fontsize=FS_SEC)
    for t, m in zip(axh.get_yticklabels(), ORDER16):
        t.set_color(tcol(m))
        if m in MUS:
            t.set_fontweight("bold")
    for s in axh.spines.values():
        s.set_visible(False)
    axh.set_xticks(np.arange(-0.5, len(CLFS)), minor=True)
    axh.set_yticks(np.arange(-0.5, len(ORDER16)), minor=True)
    axh.grid(which="minor", color="white", lw=0.8)
    axh.tick_params(which="minor", length=0)
    axh.axhline(11.5, color="white", lw=2.4)
    cax = axes_in(fig, x0h, 0.78, wh, 0.07)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[1, 4, 8, 12, 16])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=FS_TICK, length=2)
    cb.set_label("average G-mean rank (lower = better)", fontsize=FS_SEC, labelpad=2)
    ax_text_in(fig, axh, 3.45, 0.04,
               "\u25BC significantly worse than RUS (Wilcoxon signed-rank,\n"
               "Holm-adjusted p < 0.05); bold = best rank per classifier", fontsize=FS_SEC, ha="left", va="bottom",
               linespacing=1.15)
    letter_in(fig, axh, 3.30, H - 0.04, "b")
    title_in(fig, axh, 3.49, H - 0.04,
             "A coverage-arm sampler ranks first for every classifier,\nbut not significantly ahead of RUS")
    return fig


# ------------------------------------------------------------------------------------------
# Figure 3 -- mechanism
# ------------------------------------------------------------------------------------------
def fig3(C):
    W, H = 7.2, 6.65
    fig = plt.figure(figsize=(W, H))
    ya, yc, hp = 3.82, 0.55, 2.02
    # (a) boundary-coverage map
    ax = axes_in(fig, 0.60, ya, 2.80, hp)
    A = C["F3a"].set_index("method")
    for m in ["NM2", "NM1", "NM3", "CC", "RUS", "MUS-BO", "MUS-BS", "MUS-CO", "MUS-CS"]:
        r = A.loc[m]
        ax.plot([r.BP_q1, r.BP_q3], [r.CE_median] * 2, color=col(m), lw=0.8, zorder=2, solid_capstyle="butt")
        ax.plot([r.BP_median] * 2, [r.CE_q1, r.CE_q3], color=col(m), lw=0.8, zorder=2, solid_capstyle="butt")
        big = m in ("MUS-BO", "MUS-CO")
        ax.plot(r.BP_median, r.CE_median, ls="none", marker=MARK[m], ms=7.8 if big else 5.4, mfc=col(m),
                mec=edge(m), mew=0.5, zorder=4 if not big else 3)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 11)
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(["1", "2", "4", "8"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.30, 1.62)
    ax.axvline(1.0, color=GRID_GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(1.012, 9.6, "BP = 1: full-majority level", fontsize=FS_SEC, color="#6E6E6E", ha="left", va="center")
    lab_a = {"RUS": ((1.17, 2.35), "RUS"), "MUS-CS": ((1.17, 1.66), "MUS-CS / MUS-CO"),
             "CC": ((1.10, 1.0), "CC"), "NM1": ((0.33, 4.9), "NM1"), "NM2": ((0.95, 5.3), "NM2"),
             "NM3": ((0.86, 1.25), "NM3"), "MUS-BS": ((0.33, 1.08), "MUS-BS / MUS-BO")}
    for m, (xy, s) in lab_a.items():
        r = A.loc[m]
        leader(ax, xy, (r.BP_median, r.CE_median), s, tcol(m), bold=m in MUS)
    ax.text(1.60, 0.845, "lower CE =\nbetter coverage", ha="right", va="bottom", fontsize=FS_SEC,
            color="#6E6E6E", linespacing=1.1)
    ax.set_xlabel("boundary proximity, BP (< 1: kept majority nearer the minority)")
    ax.set_ylabel("coverage error, CE (log scale)")
    letter_in(fig, ax, 0.04, H - 0.04, "a")
    title_in(fig, ax, 0.23, H - 0.04, "Size-balancing samplers: the boundary arm selects like\nNearMiss, the coverage arm like RUS and CC")

    # (b) failure routes
    axb = axes_in(fig, 4.25, ya, 2.79, hp)
    Fb = C["F3b"]
    axb.axhline(0, color=GRID_GREY, lw=0.6, zorder=1)
    axb.axvline(0, color=GRID_GREY, lw=0.6, zorder=1)
    for m in ["NM2", "NM1", "NM3", "CC", "MUS-BO", "MUS-BS", "MUS-CO", "MUS-CS"]:
        per = Fb[(Fb.method == m) & (Fb.classifier != "mean")].set_index("classifier")
        for clf in CLFS:
            axb.plot(per.loc[clf, "d_sens"], per.loc[clf, "d_spec"], ls="none", marker=CLF_MARK[clf], ms=3.9,
                     mfc="none", mec=col(m), mew=0.8, zorder=2)
        r = Fb[(Fb.method == m) & (Fb.classifier == "mean")].iloc[0]
        axb.plot([r.d_sens_lo, r.d_sens_hi], [r.d_spec] * 2, color=col(m), lw=0.9, zorder=3, solid_capstyle="butt")
        axb.plot([r.d_sens] * 2, [r.d_spec_lo, r.d_spec_hi], color=col(m), lw=0.9, zorder=3, solid_capstyle="butt")
        big = m in ("MUS-BO", "MUS-CO")
        axb.plot(r.d_sens, r.d_spec, ls="none", marker=MARK[m], ms=7.8 if big else 5.6, mfc=col(m), mec=edge(m),
                 mew=0.5, zorder=4 if not big else 3.5)
    mean = Fb[Fb.classifier == "mean"].set_index("method")
    lab_b = {"MUS-BS": ((-0.19, 0.084), "MUS-BS / MUS-BO"), "MUS-CS": ((-0.078, 0.084), "MUS-CS / MUS-CO"),
             "CC": ((0.03, 0.035), "CC"), "NM1": ((-0.16, -0.14), "NM1"), "NM2": ((0.05, -0.212), "NM2"),
             "NM3": ((-0.17, -0.06), "NM3")}
    for m, (xy, s) in lab_b.items():
        leader(axb, xy, (mean.loc[m, "d_sens"], mean.loc[m, "d_spec"]), s, tcol(m), bold=m in MUS)
    axb.set_xlim(-0.195, 0.085)
    axb.set_ylim(-0.265, 0.097)
    hk = [Line2D([], [], ls="none", marker=CLF_MARK[c], ms=3.9, mfc="none", mec="#404040", mew=0.8) for c in CLFS]
    lgk = axb.legend(hk, CLFS, ncol=5, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=FS_SEC,
                     handlelength=0.8, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.15,
                     title="small open marks: one classifier each", title_fontsize=FS_SEC)
    lgk._legend_box.align = "left"
    axb.set_xlabel("change in sensitivity vs RUS")
    axb.set_ylabel("change in specificity vs RUS")
    letter_in(fig, axb, 3.62, H - 0.04, "b")
    title_in(fig, axb, 3.81, H - 0.04,
             "Two failure routes: the boundary arm loses sensitivity,\nNearMiss-1/2 lose specificity")

    # (c) which diagnostic tracks G-mean
    axc = axes_in(fig, 1.36, yc, 2.04, hp)
    F3c = C["F3c"]
    rowsY = [("gmean vs BP", "G-mean vs BP"), ("gmean vs CE", "G-mean vs CE"),
             ("BP vs CE", "BP vs CE\n(collinearity check)")]
    sty = {"size_balanced": dict(marker="o", mfc="black", mec="black", off=-0.14),
           "boundary_cluster": dict(marker="s", mfc="white", mec="black", off=0.14)}
    for setname, st in sty.items():
        for k, (pair, _) in enumerate(rowsY):
            r = F3c[(F3c.set == setname) & (F3c.pair == pair)].iloc[0]
            yy = k + st["off"]
            axc.plot([r.rho_q1, r.rho_q3], [yy, yy], color="black", lw=0.9, zorder=2, solid_capstyle="butt")
            axc.plot(r.rho_median, yy, ls="none", marker=st["marker"], ms=5.2, mfc=st["mfc"], mec=st["mec"],
                     mew=0.8, zorder=3)
    axc.axvline(0, color=GRID_GREY, lw=0.6, zorder=1)
    axc.set_xlim(-1.05, 1.05)
    axc.set_xticks([-1, -0.5, 0, 0.5, 1])
    axc.set_xticklabels(["\u22121", "\u22120.5", "0", "0.5", "1"])
    axc.set_ylim(2.55, -0.55)
    axc.set_yticks([0, 1, 2])
    axc.set_yticklabels([lbl for _, lbl in rowsY])
    axc.tick_params(axis="y", length=0)
    hh = [Line2D([], [], ls="-", lw=0.9, color="black", marker="o", ms=5.2, mfc="black", mec="black"),
          Line2D([], [], ls="-", lw=0.9, color="black", marker="s", ms=5.2, mfc="white", mec="black", mew=0.8)]
    axc.legend(hh, ["9 size-balancing samplers", f"boundary cluster (NM1{chr(8211)}3, MUS-BS, MUS-BO)"],
               loc="lower left", bbox_to_anchor=(-0.55, 1.0), fontsize=FS_SEC, handlelength=1.6,
               borderaxespad=0.15, labelspacing=0.3)
    axc.set_xlabel(f"within-dataset Spearman {chr(961)} across samplers\n(median and IQR over 55 datasets)")
    letter_in(fig, axc, 0.04, yc + hp + 0.62, "c")
    title_in(fig, axc, 0.23, yc + hp + 0.62,
             "Coverage error tracks G-mean in both sets;\nproximity only where collinear with it")

    # (d) coverage saturation
    axd = axes_in(fig, 4.25, yc, 2.79, hp)
    axd.axhspan(-ROPE, ROPE, color=BAND_GREY, lw=0, zorder=0)
    axd.axhline(0, color=GRID_GREY, lw=0.5, zorder=1)
    F3d = C["F3d"].set_index("method")
    for m in ["MUS-CO", "MUS-CS"]:
        axd.plot(C["dCE"][m], C["D1c"][m], ls="none", marker=MARK[m], ms=4.0 if m == "MUS-CS" else 4.6,
                 mfc=col(m), mec="white", mew=0.35, alpha=0.9, zorder=3)
    for k, m in enumerate(["MUS-CS", "MUS-CO"]):
        r = F3d.loc[m]
        axd.text(0.03, 0.97 - k * 0.085, f"{m}: Spearman {chr(961)} = {r.rho:.2f} (p = {r.p:.2f})".replace("-", MINUS).replace(f"MUS{MINUS}", "MUS-"),
                 transform=axd.transAxes, ha="left", va="top", fontsize=FS_SEC, color=tcol(m),
                 fontweight="bold")
    axd.set_xlabel("change in coverage error vs RUS (< 0: better coverage)")
    axd.set_ylabel("change in G-mean vs RUS\n(mean over 5 classifiers)")
    axd.margins(x=0.04)
    yl = axd.get_ylim()
    axd.set_ylim(min(yl[0], -0.045), max(yl[1], 0.062))
    axd.text(-2.25, ROPE + 0.0015, f"{chr(177)}0.01", ha="center", va="bottom", fontsize=FS_SEC, color="#6E6E6E")
    letter_in(fig, axd, 3.62, yc + hp + 0.62, "d")
    title_in(fig, axd, 3.81, yc + hp + 0.62,
             f"Better coverage than RUS on {int(F3d.n_dCE_neg.min())} of {len(C['datasets'])} datasets\n"
             "is not associated with a G-mean gain")
    return fig


# ------------------------------------------------------------------------------------------
# Figure 4 -- when is the boundary arm harmless
# ------------------------------------------------------------------------------------------
def fig4(C):
    W, H = 7.2, 3.35
    fig = plt.figure(figsize=(W, H))
    ax = axes_in(fig, 0.80, 0.42, 4.45, 2.40)
    safe, D = C["safe"], C["D1c"]
    F4 = C["F4"].set_index("method")
    ax.axhspan(-ROPE, ROPE, color=BAND_GREY, lw=0, zorder=0)
    ax.axhline(0, color=GRID_GREY, lw=0.5, zorder=1)
    ax.axvline(SAFE_CUT, color=GRID_GREY, lw=0.7, ls=(0, (1.5, 1.5)), zorder=1)
    f4z = C["F4z"]
    xj = safe + f4z["jitter"]               # display jitter at safe share 0 only
    for m in ["MUS-CS", "MUS-CO"]:
        ax.plot(xj, D[m], ls="none", marker=MARK[m], ms=3.8, mfc="none", mec=col(m), mew=0.7, zorder=2)
    for m, lwl in [("MUS-BO", 2.3), ("MUS-BS", 1.1)]:
        ax.plot(xj, D[m], ls="none", marker=MARK[m], ms=4.2 if m == "MUS-BS" else 4.8, mfc=col(m), mec="white",
                mew=0.35, zorder=3)
        lw_ = lowess(D[m].values, safe.values, frac=LOWESS_FRAC, it=LOWESS_IT, return_sorted=True)
        ax.plot(lw_[:, 0], lw_[:, 1], color=col(m), lw=lwl, zorder=4, solid_capstyle="round")
    lwz = f4z["lowess_pos_BS"]
    ax.plot(lwz[:, 0], lwz[:, 1], color=col("MUS-BS"), lw=1.1, ls=(0, (3.0, 2.0)), zorder=4.5)
    ax.set_xlim(-0.03, 1.03)
    ax.margins(y=0.04)
    ax.set_xlabel(f"share of safe minority examples (Napiera{chr(322)}a{chr(8211)}Stefanowski typology)")
    ax.set_ylabel("G-mean difference vs RUS\n(mean over 5 classifiers)")
    hb = F4.loc["MUS-BS"]
    hb2 = F4.loc["MUS-BO"]
    ann = (f"Right of the dotted line\n(safe share {chr(8805)} {SAFE_CUT}, {int(hb.n_safe_hi)} datasets):\n"
           f"boundary-arm mean gain\nMUS-BS {hb.mean_gain_safe_hi:.3f}, MUS-BO {hb2.mean_gain_safe_hi:.3f};\n"
           f"within {chr(177)}0.01 on {int(hb.n_within_safe_hi)} of {int(hb.n_safe_hi)}.\n\n"
           f"Boundary arm beats RUS\nby > 0.01 on {int(hb.n_gain_gt_rope)} of {len(C['datasets'])} datasets.").replace("-0.", MINUS + "0.")
    ax_text_in(fig, ax, 5.42, 0.42, ann, ha="left", va="bottom", fontsize=FS_SEC, color="#404040", linespacing=1.25)
    hd = [Line2D([], [], color=col("MUS-BS"), lw=1.1, marker="o", ms=4.2, mfc=col("MUS-BS"), mec="white", mew=0.35),
          Line2D([], [], color=col("MUS-BO"), lw=2.3, marker="^", ms=4.8, mfc=col("MUS-BO"), mec="white", mew=0.35),
          Line2D([], [], color=col("MUS-BS"), lw=1.1, ls=(0, (3.0, 2.0))),
          Line2D([], [], ls="none", marker="o", ms=3.8, mfc="none", mec=col("MUS-CS"), mew=0.7),
          Line2D([], [], ls="none", marker="^", ms=3.8, mfc="none", mec=col("MUS-CO"), mew=0.7),
          Patch(facecolor=BAND_GREY, edgecolor="none")]
    ax.legend(hd, [f"MUS-BS (line: LOWESS)", f"MUS-BO (line: LOWESS)",
                   f"MUS-BS LOWESS without the\n{f4z['n_zero']} datasets at safe share 0", "MUS-CS", "MUS-CO",
                   f"{chr(177)}0.01: practical\nequivalence"], loc="upper left", fontsize=FS_SEC, ncol=1,
              handlelength=2.0, borderaxespad=0.0,
              bbox_to_anchor=((5.36 - 0.80) / 4.45, 1.0))
    title_in(fig, ax, 0.04, H - 0.04,
             f"Boundary-arm losses fade as the minority becomes safe, but gains beyond {chr(177)}0.01 are rare "
             f"({int(hb.n_gain_gt_rope)} of {len(C['datasets'])} datasets)")
    return fig


# ------------------------------------------------------------------------------------------
# Figure 5 -- cost-benefit
# ------------------------------------------------------------------------------------------
F5_LABEL = {"RUS": ("random undersampling (RUS)", (6, -7), "left"), "CC": ("cluster centroids (CC)", (0, -9), "center"),
            "MUS-CS": ("MUS-CS", (0, 7), "center"), "MUS-CO": ("MUS-CO", (0, 7), "center"),
            "MUS-BS": ("MUS-BS", (2, 7), "center"), "MUS-BO": ("MUS-BO", (-6, 0), "right"),
            "NM1": ("NearMiss-1", (-6, 0), "right"), "NM2": ("NearMiss-2", (-6, 0), "right"),
            "NM3": ("NearMiss-3", (6, -5), "left"), "TL": ("TL", (6, 0), "left"), "ENN": ("ENN", (-6, 0), "right"),
            "NCR": ("NCR", (6, 0), "left"), "OSS": ("OSS", (6, 0), "left"), "CNN": ("CNN", (6, 0), "left"),
            "IHT": ("IHT", (-6, 0), "right")}


def fig5(C):
    W, H = 7.2, 3.45
    fig = plt.figure(figsize=(W, H))
    ax = axes_in(fig, 0.66, 0.42, 6.40, 2.52)
    F = C["F5"].set_index("method")
    ax.set_xscale("log")
    for m in US15:
        x, y = F.loc[m, "t_median"], F.loc[m, "pooled_rank"]
        if m in COL:
            big = m in ("MUS-BO", "MUS-CO")
            ax.plot(x, y, ls="none", marker=MARK[m], ms=7.0 if big else 6.0, mfc=col(m), mec=edge(m), mew=0.5,
                    zorder=4)
        else:
            ax.plot(x, y, ls="none", marker="o", ms=5.0, mfc=OTHER_GREY, mec=DARK_EDGE, mew=0.5, zorder=3)
        s, off, ha = F5_LABEL[m]
        ax.annotate(s, (x, y), xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=FS_SEC,
                    color=tcol(m), fontweight="bold" if m in MUS else "normal", zorder=6)
    for a, b in [("MUS-BS", "MUS-BO"), ("MUS-CS", "MUS-CO")]:
        ar = FancyArrowPatch((F.loc[a, "t_median"], F.loc[a, "pooled_rank"]),
                             (F.loc[b, "t_median"], F.loc[b, "pooled_rank"]), arrowstyle="-|>", mutation_scale=7,
                             color=col(b), lw=0.9, shrinkA=5, shrinkB=6, zorder=5)
        ax.add_patch(ar)
    xm = np.sqrt(F.loc["MUS-CS", "t_median"] * F.loc["MUS-CO", "t_median"])
    ym = (F.loc["MUS-CS", "pooled_rank"] + F.loc["MUS-CO", "pooled_rank"]) / 2
    ax.annotate(f"median time MUS-CO / MUS-CS = {C['ratio_CO_CS']:.1f}{chr(215)}", (xm, ym), xytext=(0, -11),
                textcoords="offset points", ha="center", va="center", fontsize=FS_SEC, color=tcol("MUS-CO"))
    ax.set_xlim(0.0042, 8.0)
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1]))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(["10 ms", "100 ms", "1 s"]))
    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(12.4, 3.9)
    ax.set_xlabel("median resampling time per training fold (log scale)")
    ax.set_ylabel("pooled average G-mean rank\n(5 classifiers x 55 datasets)")
    ax.text(0.012, 0.975, f"{chr(8598)} better: faster and lower (better) rank", transform=ax.transAxes, ha="left",
            va="top", fontsize=FS_SEC, color="#404040")
    title_in(fig, ax, 0.04, H - 0.04,
             f"Optimal matching moves the pooled rank by at most {C['max_rank_shift']:.1f}, but on the coverage arm "
             f"it costs {C['ratio_CO_CS']:.1f}{chr(215)} the median time")
    return fig


# ------------------------------------------------------------------------------------------
# Supplementary: runtime grid and full boundary-coverage map
# ------------------------------------------------------------------------------------------
def _fmt_time(t):
    return f"{t * 1000:.3g} ms" if t < 1 else f"{t:.3g} s"


def fig_runtime_grid(C):
    W, H = 7.2, 6.8
    fig = plt.figure(figsize=(W, H))
    ncol, nrow = 4, 4
    x0, y0, pw, ph, gx, gy = 0.68, 0.50, 1.53, 1.24, 0.09, 0.32
    rtd, G = C["rt_ds"], C["RTG"].set_index("method")
    xlim, ylim = (55.0, 6000.0), (2e-3, 3000.0)
    for k, s in enumerate(ORDER_RT):
        i, j = divmod(k, ncol)
        ax = axes_in(fig, x0 + j * (pw + gx), y0 + (nrow - 1 - i) * (ph + gy), pw, ph)
        x = rtd[rtd.sampler == s]
        r = G.loc[s]
        ax.plot(x.n_maj, x.t_med, ls="none", marker=MARK.get(s, "o"), ms=2.8, mfc=col(s), mec=edge(s), mew=0.3, zorder=3)
        xs = np.array([x.n_maj.min(), x.n_maj.max()])
        ax.plot(xs, 10 ** (r.intercept + r.slope * np.log10(xs)), color=tcol(s), lw=1.1, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.xaxis.set_major_locator(FixedLocator([100, 1000]))
        ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(["100", "1k"]))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1, 10, 100, 1000]))
        ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(["10 ms", "100 ms", "1 s", "10 s", "100 s", "1000 s"]))
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        if j > 0:
            ax.tick_params(axis="y", labelleft=False)
        if i < nrow - 1:
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_title("SMOTE (oversampling)" if s == "SMOTE" else s, loc="left", pad=2, fontsize=FS_BASE, color=tcol(s),
                     fontweight="bold" if s in MUS else "normal")
        ax.text(0.04, 0.96, f"slope {r.slope:.2f} [{r.slope_lo:.2f}, {r.slope_hi:.2f}]\nmedian {_fmt_time(r.t_median)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS_SEC, color="#404040", linespacing=1.15)
    gh = nrow * ph + (nrow - 1) * gy
    gw = ncol * pw + (ncol - 1) * gx
    fig.text(0.06 / W, (y0 + gh / 2) / H, "median resampling time per training fold (log scale)", rotation=90,
             ha="left", va="center", fontsize=FS_BASE)
    fig.text((x0 + gw / 2) / W, 0.04 / H, "median training-fold majority size (log scale)", ha="center", va="bottom",
             fontsize=FS_BASE)
    top = G.slope.idxmax()
    fig.text(0.04 / W, (H - 0.04) / H,
             f"Resampling time per training fold grows with majority size for every sampler; {top} grows fastest "
             f"(log-log slope {G.loc[top, 'slope']:.2f})", ha="left", va="top", fontsize=FS_BASE)
    return fig


def fig_map_full(C):
    W, H = 7.2, 4.6
    fig = plt.figure(figsize=(W, H))
    ax = axes_in(fig, 0.62, 0.45, 6.45, 3.55)
    M = C["MAPF"].set_index("method")
    ax.axvline(1.0, color=GRID_GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(1.006, 3.22, "BP = 1: full-majority level", fontsize=FS_SEC, color="#6E6E6E", ha="left", va="center")
    for s in ["NM2", "NM1", "NM3", "CC", "RUS", "MUS-BO", "MUS-BS", "MUS-CO", "MUS-CS"]:
        r = M.loc[s]
        big = s in ("MUS-BO", "MUS-CO")
        ax.plot(r.BP_median, r.CE_median, ls="none", marker=MARK[s], ms=8.0 if big else 6.0, mfc=col(s), mec=edge(s),
                mew=0.5, zorder=3 if big else 4)
    for s in NONBAL:
        r = M.loc[s]
        ax.plot(r.BP_median, r.CE_median, ls="none", marker="o", ms=5.5, mfc="white", mec=DARK_EDGE, mew=0.9, zorder=4)
    lab = {"RUS": ((1.08, 1.97), "RUS"), "MUS-CS": ((1.08, 1.64), "MUS-CS / MUS-CO"), "CC": ((1.08, 1.30), "CC"),
           "NM1": ((0.545, 3.12), "NM1"), "NM2": ((0.90, 3.10), "NM2"), "NM3": ((0.80, 2.25), "NM3"),
           "MUS-BS": ((0.555, 2.05), "MUS-BS / MUS-BO")}
    for s, (xy, txt) in lab.items():
        r = M.loc[s]
        leader(ax, xy, (r.BP_median, r.CE_median), txt, tcol(s), bold=s in MUS)
    for s, xy in {"CNN": (0.60, 1.52), "OSS": (0.80, 0.22), "IHT": (0.80, 0.80)}.items():
        r = M.loc[s]
        leader(ax, xy, (r.BP_median, r.CE_median), f"{s}  |S|/|P| = {r.size_ratio_median:.2f}", tcol(s))
    ax.set_xlim(0.53, 1.50)
    ax.set_ylim(-0.15, 3.32)
    ax.set_xlabel("boundary proximity, BP (< 1: kept majority nearer the minority)")
    ax.set_ylabel("coverage error, CE (lower = majority better covered)")
    axi = ax.inset_axes([0.66, 0.05, 0.32, 0.30])
    for s in ["NONE", "TL", "ENN", "NCR"]:
        r = M.loc[s]
        axi.plot(r.BP_median, r.CE_median, ls="none", marker="o", ms=5.5, mfc="white", mec=DARK_EDGE, mew=0.9, zorder=4)
    axi.set_xlim(0.993, 1.024)
    axi.set_ylim(-0.012, 0.080)
    for s, (xy, ha) in {"NONE": ((0.9945, 0.030), "left"), "TL": ((1.0048, 0.004), "left"),
                        "ENN": ((1.0145, 0.036), "left"), "NCR": ((1.0150, 0.068), "right")}.items():
        r = M.loc[s]
        leader(axi, xy, (r.BP_median, r.CE_median), f"{s}  {r.size_ratio_median:.2f}", tcol(s), ha=ha)
    axi.set_xticks([1.00, 1.01, 1.02])
    axi.set_xticklabels(["1.00", "1.01", "1.02"])
    axi.set_yticks([0, 0.05])
    axi.set_yticklabels(["0", "0.05"])
    axi.tick_params(labelsize=FS_TICK, length=2)
    for sp in axi.spines.values():
        sp.set_visible(True)
        sp.set_color("#9E9E9E")
        sp.set_linewidth(0.6)
    axi.set_title("zoom (label: median |S|/|P|)", fontsize=FS_SEC, loc="left", pad=2, color="#404040")
    ax.indicate_inset_zoom(axi, edgecolor="#9E9E9E", alpha=1.0, linewidth=0.6)
    hd = [Line2D([], [], ls="none", marker="o", ms=6, mfc="#595959", mec="white", mew=0.5),
          Line2D([], [], ls="none", marker="o", ms=5.5, mfc="white", mec=DARK_EDGE, mew=0.9)]
    ax.legend(hd, ["output size fixed at |S| = |P| (colour = sampler)", "output size not controlled (label: median |S|/|P|)"],
              loc="upper right", bbox_to_anchor=(1.0, 0.80), fontsize=FS_SEC, handletextpad=0.4, borderaxespad=0.3)
    title_in(fig, ax, 0.04, H - 0.04,
             "Boundary-coverage map of all 16 selections: samplers that fix |S| = |P| (filled) and selections whose "
             "output size\nis not controlled (open, labelled with the median ratio of kept majority to minority instances)")
    return fig


# ------------------------------------------------------------------------------------------
# Supplementary CD diagrams (drawing code reused from the statistics pipeline)
# ------------------------------------------------------------------------------------------
def cd_cliques(vals, cd):
    k = len(vals)
    out = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(i2 <= i and j <= j2 for (i2, j2) in out):
            out.append((i, j))
    return out


def pack_levels(cliques, vals, gap=0.3):
    levels, assign = [], []
    for (i, j) in cliques:
        x0, x1 = vals[i], vals[j]
        for li, lev in enumerate(levels):
            if all(x1 + gap < a or x0 - gap > b for (a, b) in lev):
                lev.append((x0, x1))
                assign.append(li)
                break
        else:
            levels.append([(x0, x1)])
            assign.append(len(levels) - 1)
    return assign, len(levels)


CD_GEOM = dict(lev0=0.20, lev_dy=0.13, dy=0.27, y_cd=0.62)


def cd_layout(avg, cd):
    order = avg.sort_values(kind="mergesort")
    vals = order.values.astype(float)
    cl = cd_cliques(vals, cd)
    assign, nlev = pack_levels(cl, vals)
    return order, cl, assign, nlev


def draw_cd(ax, avg, cd, title, nlev_common):
    g = CD_GEOM
    order, cliques, assign, _ = cd_layout(avg, cd)
    names, vals = list(order.index), order.values.astype(float)
    k = len(vals)
    lo, hi = 1, k
    span = hi - lo
    marg = 0.43 * span + 0.4
    c0 = g["lev0"] + max(nlev_common, 1) * g["lev_dy"] + 0.06
    h = (k + 1) // 2
    ax.plot([lo, hi], [0, 0], color="k", lw=0.8, solid_capstyle="butt")
    for r_ in range(lo, hi + 1):
        ax.plot([r_, r_], [0, 0.09], color="k", lw=0.6)
        ax.text(r_, 0.12, str(r_), ha="center", va="bottom", fontsize=FS_TICK)
    for r_ in np.arange(lo + 0.5, hi, 1.0):
        ax.plot([r_, r_], [0, 0.045], color="k", lw=0.5)
    yb = g["y_cd"]
    ax.plot([lo, lo + cd], [yb, yb], color="k", lw=0.9, solid_capstyle="butt")
    for xx in (lo, lo + cd):
        ax.plot([xx, xx], [yb - 0.05, yb + 0.05], color="k", lw=0.9)
    ax.text(lo + cd + 0.25, yb, f"CD = {cd:.2f}", ha="left", va="center", fontsize=FS_SEC)
    for i in range(k):
        name, v = names[i], vals[i]
        is_mus = name in MUS
        lw = 1.0 if is_mus else 0.8 if name in COL else 0.6
        if i < h:
            y = -(c0 + i * g["dy"])
            xe, xt, ha, txt = lo - 0.15, lo - 0.3, "right", f"{name}  {v:.2f}"
        else:
            y = -(c0 + (k - 1 - i) * g["dy"])
            xe, xt, ha, txt = hi + 0.15, hi + 0.3, "left", f"{v:.2f}  {name}"
        ax.plot([v, v, xe], [0, y, y], color=col(name), lw=lw)
        ax.text(xt, y, txt, ha=ha, va="center", fontsize=FS_SEC, color=tcol(name),
                fontweight="bold" if is_mus else "normal")
    for (i, j), li in zip(cliques, assign):
        y = -(g["lev0"] + li * g["lev_dy"])
        ax.plot([vals[i] - 0.07, vals[j] + 0.07], [y, y], color="k", lw=2.2, solid_capstyle="butt")
    ax.set_xlim(lo - marg, hi + marg)
    ax.set_ylim(-(c0 + (h - 1) * g["dy"]) - 0.18, 0.80)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=FS_BASE, pad=2)


def fig_cd(C, metric):
    RMm, Rmm = C["ranks"][metric]
    lay = {c: (RMm[c], Rmm.loc[c]) for c in CLFS}
    nlev = max(cd_layout(a, fr["CD"])[3] for a, fr in lay.values())
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.3))
    for ax, clf in zip(axes.ravel(), CLFS):
        avg, fr = lay[clf]
        draw_cd(ax, avg, fr["CD"], f"{clf}: {int(fr['N'])} datasets, Friedman p {fmt_p(fr['p_chi2'])}", nlev)
    axn = axes.ravel()[-1]
    axn.axis("off")
    mname = {"gmean": "G-mean", "auc": "ROC-AUC"}[metric]
    axn.text(0.02, 0.92,
             f"Average {mname} rank of 16 samplers per classifier\n(1 = best, at the left end of each axis).\n\n"
             "Thick bars join samplers whose average ranks differ\nby less than the Nemenyi critical difference "
             "(CD,\n" + f"{chr(945)} = 0.05, k = 16, N = 55).\n\n"
             "Colours: MUS-BS / MUS-BO (boundary arm, blues),\nMUS-CS / MUS-CO (coverage arm, oranges),\n"
             "RUS black, CC and NearMiss in greys,\nother baselines light grey.",
             transform=axn.transAxes, ha="left", va="top", fontsize=FS_SEC, color="#404040", linespacing=1.2)
    for ax, letter in zip(axes.ravel()[:5], "abcde"):
        panel_letter(ax, letter, dx=-0.045, dy=1.0)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.95, bottom=0.01, hspace=0.18, wspace=0.09)
    return fig


# ------------------------------------------------------------------------------------------
# Caption drafts (every number read from the computed context, none typed by hand)
# ------------------------------------------------------------------------------------------
def _up(x, nd):
    """Upper bound rounded up (so a quoted '<= value' is always true)."""
    return f"{np.ceil(x * 10 ** nd) / 10 ** nd:.{nd}f}"


def _lo(x, nd):
    """Lower bound rounded down (so a quoted '>= value' is always true)."""
    return f"{np.floor(x * 10 ** nd) / 10 ** nd:.{nd}f}"


def _m(x, nd=3):
    """Signed number with a typographic minus."""
    return f"{x:+.{nd}f}".replace("-", MINUS) if x < 0 else f"{x:+.{nd}f}"


def captions(C):
    n = len(C["datasets"])
    td = C["toy_diag"]
    F1b, F1c = C["F1b"], C["F1c"].set_index("method")
    arm = F1b[F1b.contrast.str.startswith("arm")]
    con = F1b[F1b.contrast.str.startswith("concept")]
    F2a = C["F2a"]
    g, a = F2a[F2a.metric == "gmean"], F2a[F2a.metric == "auc"]
    cov = ["MUS-CS", "MUS-CO"]
    bnd = ["MUS-BS", "MUS-BO"]
    auc_inc = a[a.mus.isin(bnd) & (a.p_mus_worse < 0.95)]
    W = C["W_rus"].reset_index()
    RM = C["ranks"]["gmean"][0]
    covgap = max(float(RM.loc["RUS", c] - RM.loc[m, c]) for c in CLFS for m in cov)
    A3 = C["F3a"].set_index("method")
    B3 = C["F3b"][C["F3b"].classifier == "mean"].set_index("method")
    near = max(abs(B3.loc[m, q]) for m in ["CC"] + cov for q in ["d_sens", "d_spec"])
    C3 = C["F3c"].set_index(["set", "pair"])
    D3 = C["F3d"].set_index("method")
    F4 = C["F4"].set_index("method")
    F5 = C["F5"].set_index("method")
    bg = C["cd_gaps"][C["cd_gaps"].method.isin(bnd)]
    cd = float(C["ranks"]["gmean"][1].loc["CART", "CD"])
    key = ("Colour key throughout: MUS-CO orange, MUS-CS dark orange-brown (coverage arm); MUS-BS dark blue, "
           "MUS-BO light blue (boundary arm); random undersampling (RUS) black; cluster centroids (CC) dark grey; "
           "NearMiss-1/2/3 greys with distinct markers; other baselines light grey.")
    L = []
    L.append("# Figure captions (drafts)\n")
    L.append("Numbers below are generated by `figures_pipeline.py` from the same computation that writes "
             "`figure_numbers.csv`. Unit of replication: dataset (55 KEEL datasets; each dataset value is the mean "
             "over 5 x 5 stratified cross-validation folds). " + key + "\n")
    L.append("## Figure 1 (fig1_hook)\n")
    L.append(
        "**Which majority instances an undersampler keeps matters; how optimally it matches does not.** "
        "(a) Orientation on a two-dimensional toy problem (520 majority and 58 minority instances; samplers applied to "
        "the raw coordinates with random_state = 0). Each panel shows the majority instances kept by random "
        "undersampling (RUS), NearMiss-1 (NM1), the boundary arm (MUS-BS: stable minority-majority matching) and the "
        "coverage arm (MUS-CS: stable within-majority matching); every sampler keeps 58 majority instances. Like NM1, "
        f"the boundary arm keeps majority instances next to the minority (toy BP {td['MUS-BS'][0]:.2f} vs "
        f"{td['NM1'][0]:.2f} for NM1); the coverage arm spreads them over the majority like RUS (BP "
        f"{td['MUS-CS'][0]:.2f} vs {td['RUS'][0]:.2f}) with lower coverage error (CE {td['MUS-CS'][1]:.2f} vs "
        f"{td['RUS'][1]:.2f}). "
        f"(b) Factor effects of the 2 x 2 design on G-mean for each classifier: mean paired difference over the {n} "
        f"datasets with 95% percentile bootstrap intervals over datasets (B = {B_BOOT}). Two-tone markers show the "
        "colours of the minuend (left half) and the subtrahend (right half). Arm contrasts (coverage minus boundary, "
        f"within the stable and within the optimal concept) range from {_m(arm.mean_diff.min())} to "
        f"{_m(arm.mean_diff.max())} and are significant after Holm correction for all five classifiers (two-sided "
        f"Wilcoxon signed-rank tests, Holm over the four factorial contrasts per classifier, p_holm "
        f"{chr(8804)} {_up(arm.p_holm.max(), 4)}). Concept contrasts (optimal minus stable, within each arm) lie within "
        f"{chr(177)}{_up(con.mean_diff.abs().max(), 4)}; all {len(con)} intervals include zero (p_holm "
        f"{chr(8805)} {_lo(con.p_holm.min(), 2)}). "
        "(c) Per-dataset G-mean difference from RUS, averaged over the five classifiers and sorted within each method "
        f"(n = {n} datasets per line). Grey band: region of practical equivalence ({chr(177)}{ROPE}, as in the Bayesian "
        "tests). Legend counts: datasets below / within / above the band (boundary arm "
        f"{F1c.loc['MUS-BS', 'below']} and {F1c.loc['MUS-BO', 'below']} below; coverage arm "
        f"{F1c.loc['MUS-CS', 'within']} and {F1c.loc['MUS-CO', 'within']} within).\n")
    L.append("## Figure 2 (fig2_evidence)\n")
    L.append(
        "**The coverage arm is practically equivalent to random undersampling; the boundary arm is worse in G-mean.** "
        f"(a) Bayesian signed-rank test of each MUS method against RUS over {n} datasets (region of practical "
        f"equivalence {chr(177)}{ROPE} on the metric scale; prior weight 0.5; 50,000 posterior samples). Each bar stacks "
        "the posterior probabilities that the MUS method is better (solid), practically equivalent (light tint) or "
        "worse (hatched) than RUS. Coverage arm: P(practically equivalent) "
        f"{chr(8805)} {np.floor(g[g.mus.isin(cov)].p_rope.min() * 1000) / 1000:.3f} for G-mean and "
        f"{chr(8805)} {np.floor(a[a.mus.isin(cov)].p_rope.min() * 1000) / 1000:.3f} for ROC-AUC, for every classifier. "
        f"Boundary arm: P(worse) {chr(8805)} {np.floor(g[g.mus.isin(bnd)].p_mus_worse.min() * 1000) / 1000:.3f} for "
        f"G-mean for every classifier; for ROC-AUC, P(worse) = {auc_inc.p_mus_worse.min():.2f}{chr(8211)}"
        f"{auc_inc.p_mus_worse.max():.2f} with {', '.join(sorted(set(auc_inc.classifier)))} (below the 0.95 decision "
        "threshold, most probable outcome: worse) and > 0.95 otherwise. "
        "(b) Average G-mean rank of the 16 undersampling-family methods for each classifier (Friedman ranks over "
        f"{n} datasets; 1 = best, lower = better). {chr(9660)}: MUS method significantly worse than RUS (two-sided "
        "Wilcoxon signed-rank test, Holm adjustment over the 48 MUS-versus-baseline comparisons per classifier; "
        f"p_holm {chr(8804)} {_up(W[W.mus.isin(bnd)].p_holm.max(), 4)} for all ten boundary-arm cells). No coverage-arm "
        f"cell differs significantly from RUS (p_holm {chr(8805)} {np.floor(W[W.mus.isin(cov)].p_holm.min() * 100) / 100:.2f}). "
        "Bold: best average rank per classifier, a coverage-arm method in every column; its lead over RUS "
        f"({chr(8804)} {_up(covgap, 2)} ranks) is far inside the Nemenyi critical difference ({cd:.2f}).\n")
    L.append("## Figure 3 (fig3_mechanism)\n")
    L.append(
        "**The two arms keep different majority instances and fail through different error routes; covering the "
        "majority better than RUS is not associated with a G-mean gain.** "
        f"(a) Boundary-coverage map of the nine size-balancing samplers (target: as many kept majority as minority "
        "instances): median "
        f"over {n} datasets, whiskers = interquartile range. Boundary proximity BP = mean distance of the kept majority "
        "to the minority divided by that of the full majority (1 = full-majority level, < 1 = nearer the minority). "
        "Coverage error CE = mean distance of every majority instance to its nearest kept majority instance divided by "
        "the mean nearest-neighbour distance within the majority (log scale; lower = better coverage). Stable and "
        "optimal versions coincide (larger triangle = optimal, drawn underneath). NM3 keeps fewer majority than "
        f"minority instances on {int(A3.loc['NM3', 'n_not_balanced'])} of {n} datasets (minimum size ratio "
        f"{A3.loc['NM3', 'size_ratio_min']:.2f}); all other samplers keep exactly as many. "
        "(b) Mean change in sensitivity (x) and specificity (y) relative to RUS. Small open marks: one classifier each "
        f"(marker shape = classifier, see key; colour = sampler; mean over {n} datasets). Large marks: mean over the five classifiers, with 95% bootstrap intervals over datasets "
        f"(B = {B_BOOT}). Boundary arm: {_m(B3.loc['MUS-BS', 'd_sens'])} sensitivity, {_m(B3.loc['MUS-BS', 'd_spec'])} "
        f"specificity; NM1 {_m(B3.loc['NM1', 'd_sens'])} / {_m(B3.loc['NM1', 'd_spec'])}; NM2 "
        f"{_m(B3.loc['NM2', 'd_sens'])} / {_m(B3.loc['NM2', 'd_spec'])}; NM3 {_m(B3.loc['NM3', 'd_sens'])} / "
        f"{_m(B3.loc['NM3', 'd_spec'])}; CC and the coverage arm stay within {_up(near, 3)} of RUS on both axes. "
        "Descriptive; not significance-tested. "
        "(c) Within-dataset Spearman correlation across samplers between classifier-averaged G-mean and BP or CE, "
        f"and between BP and CE (collinearity check): median and interquartile range over {n} datasets, for the nine "
        "size-balancing samplers (filled circles) and for the boundary cluster NM1-3, MUS-BS and MUS-BO (open squares; "
        "with five samplers rho moves in steps of 0.1). Size-balancing set: G-mean-BP "
        f"{_m(C3.loc[('size_balanced', 'gmean vs BP'), 'rho_median'], 2)}, G-mean-CE "
        f"{_m(C3.loc[('size_balanced', 'gmean vs CE'), 'rho_median'], 2)}, BP-CE "
        f"{_m(C3.loc[('size_balanced', 'BP vs CE'), 'rho_median'], 2)}. Boundary cluster: G-mean-BP "
        f"{_m(C3.loc[('boundary_cluster', 'gmean vs BP'), 'rho_median'], 2)} (Wilcoxon test of rho = 0, Holm p = "
        f"{C3.loc[('boundary_cluster', 'gmean vs BP'), 'wilcoxon_p_holm']:.2f}), G-mean-CE "
        f"{_m(C3.loc[('boundary_cluster', 'gmean vs CE'), 'rho_median'], 2)}, BP-CE "
        f"{_m(C3.loc[('boundary_cluster', 'BP vs CE'), 'rho_median'], 2)}. "
        "(d) Per-dataset change in CE (coverage-arm method minus RUS) against the change in classifier-averaged G-mean "
        f"for MUS-CS (circles) and MUS-CO (triangles), {n} datasets each; band {chr(177)}{ROPE}. Both lower CE on "
        f"{int(D3.loc['MUS-CS', 'n_dCE_neg'])} and {int(D3.loc['MUS-CO', 'n_dCE_neg'])} of {n} datasets (median change "
        f"{_m(D3.loc['MUS-CS', 'dCE_median'], 2)} and {_m(D3.loc['MUS-CO', 'dCE_median'], 2)}), but the size of the coverage gain is not associated with the change in G-mean "
        f"(Spearman rho = {_m(D3.loc['MUS-CS', 'rho'], 2)}, p = {D3.loc['MUS-CS', 'p']:.2f}; "
        f"{_m(D3.loc['MUS-CO', 'rho'], 2)}, p = {D3.loc['MUS-CO', 'p']:.2f}).\n")
    L.append("## Figure 4 (fig4_when)\n")
    hb, hb2 = F4.loc["MUS-BS"], F4.loc["MUS-BO"]
    f4z = C["F4z"]
    pmax = max(f4z[m]["p"] for m in ("MUS-BS", "MUS-BO"))
    pz = "p < 0.001" if pmax < 0.001 else f"p {chr(8804)} {_up(pmax, 3)}"
    L.append(
        "**The boundary arm approaches random undersampling only where the minority is safe, and rarely beats it.** "
        "Classifier-averaged "
        "G-mean difference from RUS against the share of safe minority examples (Napierala-Stefanowski "
        f"neighbourhood typology); one mark per dataset and method, n = {n} datasets. Filled marks and LOWESS curves "
        f"(statsmodels, frac = 2/3, {LOWESS_IT} robustifying iterations): boundary arm; open marks: coverage arm. Band: "
        f"{chr(177)}{ROPE}. Marks at safe share 0 ({f4z['n_zero']} datasets) are spread by a seeded horizontal jitter of at "
        f"most {chr(177)}{JITTER_HALF} for visibility; all fits and statistics use the unjittered values. Dashed line: "
        f"LOWESS for MUS-BS fitted without these {f4z['n_zero']} datasets (n = {f4z['n_pos']}). The boundary arm's loss "
        f"shrinks as the safe share rises (Spearman rho = {hb.rho_safe:.2f} for MUS-BS, {hb2.rho_safe:.2f} for MUS-BO; "
        f"without the datasets at safe share 0, {f4z['MUS-BS']['rho']:.2f} and {f4z['MUS-BO']['rho']:.2f}, {pz}); its "
        "LOWESS curve stays inside the band only from a "
        f"safe share of {hb.lowess_enters_band:.2f} (MUS-BS) and {hb2.lowess_enters_band:.2f} (MUS-BO) upwards "
        f"({_m(hb.lowess_at_1)} and {_m(hb2.lowess_at_1)} at safe share 1, against {_m(hb.lowess_at_0)} and "
        f"{_m(hb2.lowess_at_0)} at 0). On the {int(hb.n_safe_hi)} datasets with safe share {chr(8805)} {SAFE_CUT} its "
        f"mean gain is {_m(hb.mean_gain_safe_hi)} (MUS-BS) and {_m(hb2.mean_gain_safe_hi)} (MUS-BO), within the band on "
        f"{int(hb.n_within_safe_hi)} of {int(hb.n_safe_hi)} (against {int(hb.n_within_safe_lo)} and "
        f"{int(hb2.n_within_safe_lo)} of the {n - int(hb.n_safe_hi)} less safe datasets); it beats RUS by more than "
        f"{ROPE} on {int(hb.n_gain_gt_rope)} of {n} datasets ({hb.datasets_gain_gt_rope.replace(';', ', ')}).\n")
    L.append("## Figure 5 (fig5_cost)\n")
    L.append(
        "**Optimal matching moves the pooled rank by at most 0.2 but can cost several times the run time.** Median "
        "resampling time per training fold (log scale; "
        f"{int(F5.loc['RUS', 'n_folds'])} folds = {n} datasets x 25 folds per sampler) against the pooled average "
        "G-mean rank (275 dataset-classifier blocks, k = 16 methods including no resampling; descriptive; lower = "
        "better) for the 15 undersamplers. Arrows join the stable and the optimal version of each arm. Optimal "
        f"matching changes the pooled rank by {_m(F5.loc['MUS-CO', 'pooled_rank'] - F5.loc['MUS-CS', 'pooled_rank'], 2)} "
        f"on the coverage arm and {_m(F5.loc['MUS-BO', 'pooled_rank'] - F5.loc['MUS-BS', 'pooled_rank'], 2)} on the "
        f"boundary arm. It takes {C['ratio_CO_CS']:.1f}x the median time of stable matching on the coverage arm "
        f"(MUS-CO {F5.loc['MUS-CO', 't_median']:.2f} s vs MUS-CS {F5.loc['MUS-CS', 't_median']:.2f} s; 95th percentiles "
        f"{F5.loc['MUS-CO', 't_p95']:.0f} s vs {F5.loc['MUS-CS', 't_p95']:.1f} s), but {C['ratio_BO_BS']:.2f}x on the "
        "boundary arm. Times were measured while the benchmark ran under parallel load on a shared machine, so "
        "absolute values are indicative only. Abbreviations: TL Tomek links, ENN edited nearest neighbours, NCR "
        "neighbourhood cleaning rule, OSS one-sided selection, CNN condensed nearest neighbour, IHT instance hardness "
        "threshold.\n")
    L.append("## Supplementary Figures (figS_cd_gmean, figS_cd_auc)\n")
    L.append(
        "**Critical-difference diagrams of average G-mean (figS_cd_gmean) and ROC-AUC (figS_cd_auc) ranks**, one panel "
        f"per classifier (N = {n} datasets, k = 16 methods, Nemenyi CD = {cd:.2f} at alpha = 0.05; Friedman p < 0.001 "
        "for every classifier and both metrics). Thick bars join methods whose average ranks differ by less than the "
        "CD. The pooled-over-classifiers panel of the earlier version is omitted. The Nemenyi test is conservative: "
        f"{int(bg.inside_cd.sum())} of the {len(bg)} boundary-arm-versus-RUS G-mean rank gaps lie inside the CD, whereas "
        "the Holm-adjusted Wilcoxon tests (Fig. 2b) find all ten significant.\n")
    G = C["RTG"].set_index("method")
    L.append("## Supplementary Figure (figS_runtime_grid)\n")
    L.append(
        "**Resampling time per training fold against majority size.** One panel per sampler (SMOTE: oversampling "
        f"reference). Each mark is one dataset ({int(G.loc['RUS', 'n_datasets'])} per panel): the median resampling time "
        "over its 25 training folds against the median training-fold majority size (both axes log scale). Line: "
        "least-squares fit of log10 time on log10 majority size; annotated: its slope with a 95% t interval and the "
        f"median time over all {int(G.loc['RUS', 'n_folds'])} training folds. Slopes range from "
        f"{G.slope.min():.2f} ({G.slope.idxmin()}) to {G.slope.max():.2f} ({G.slope.idxmax()}); all 95% intervals exclude "
        f"{'zero' if bool((G.slope_lo > 0).all()) else 'zero except for some samplers'}. MUS-CS: slope {G.loc['MUS-CS', 'slope']:.2f}, median {_fmt_time(G.loc['MUS-CS', 't_median'])}; MUS-CO: "
        f"slope {G.loc['MUS-CO', 'slope']:.2f}, median {_fmt_time(G.loc['MUS-CO', 't_median'])}. Times were measured "
        "under parallel load on a shared machine, so absolute values are indicative only. Abbreviations as in Fig. 5.\n")
    Mf = C["MAPF"].set_index("method")
    nb = ", ".join(f"{s} {Mf.loc[s, 'size_ratio_median']:.2f}" for s in NONBAL)
    near4 = Mf.loc[["NONE", "TL", "ENN", "NCR"]]
    L.append("## Supplementary Figure (figS_map_full)\n")
    L.append(
        "**Boundary-coverage map of all 16 selections** (15 undersamplers and no resampling, NONE): median boundary "
        f"proximity BP against median coverage error CE over {n} datasets (linear CE axis; definitions as in Fig. 3a). "
        "Filled marks: the nine samplers that fix the output size at |S| = |P| (colour key as in the main figures; "
        f"NM3 keeps fewer majority than minority instances on {int(A3.loc['NM3', 'n_not_balanced'])} of {n} datasets but "
        "its median ratio is 1.00). Open marks: selections whose output size is not controlled, labelled with the median "
        f"ratio of kept majority to minority instances ({nb}). NONE, TL, ENN and NCR keep almost the whole majority "
        f"(median |S|/|P| {chr(8805)} {_lo(near4.size_ratio_median.min(), 1)}) and have CE {chr(8804)} "
        f"{_up(near4.CE_median.max(), 2)}, so their position is not comparable with the size-controlled samplers. CNN "
        f"keeps slightly fewer majority than minority instances (median |S|/|P| {Mf.loc['CNN', 'size_ratio_median']:.2f}) "
        f"and is the only uncontrolled selection with BP well below 1 ({Mf.loc['CNN', 'BP_median']:.2f}). Inset: zoom on "
        "NONE, TL, ENN and NCR.\n")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------------
FIGURES = [("fig1_hook", fig1), ("fig2_evidence", fig2), ("fig3_mechanism", fig3), ("fig4_when", fig4),
           ("fig5_cost", fig5), ("figS_cd_gmean", lambda C: fig_cd(C, "gmean")),
           ("figS_cd_auc", lambda C: fig_cd(C, "auc")), ("figS_runtime_grid", fig_runtime_grid),
           ("figS_map_full", fig_map_full)]


def run(inp, out, style_fn=None, keep_figs=False, only=None):
    os.makedirs(out, exist_ok=True)
    apply_style(style_fn)
    C = compute(inp)
    rc_path = os.path.join(out, "render_checks.json")
    prev = {}
    if os.path.exists(rc_path):          # partial runs (only=...) update their entries instead of erasing the rest
        with open(rc_path, encoding="utf-8") as f:
            prev = json.load(f)
    checks, figs = {}, {}
    for stem, fn in FIGURES:
        if only and stem not in only:
            continue
        fig = fn(C)
        fig.savefig(os.path.join(out, stem + ".pdf"))
        fig.savefig(os.path.join(out, stem + ".png"), dpi=300)
        bad, outside = check_overlaps(fig)
        h_px, w_px = mpimg.imread(os.path.join(out, stem + ".png")).shape[:2]
        entry = dict(bbox_overlaps=[[str(a), str(b)] for a, b in bad], text_outside_figure=[str(t) for t in outside],
                     saved_png_px=[int(w_px), int(h_px)], saved_size_in=[round(w_px / 300, 2), round(h_px / 300, 2)])
        if "perceptual_check" in prev.get(stem, {}):
            entry["perceptual_check"] = prev[stem]["perceptual_check"]   # visual-review note, refreshed by hand
        checks[stem] = entry
        if keep_figs:
            figs[stem] = fig
        else:
            plt.close(fig)
    num = C["reg"].frame()
    num.to_csv(os.path.join(out, "figure_numbers.csv"), index=False)
    C["plotdata"].to_csv(os.path.join(out, "figure_plotdata.csv"), index=False)
    with open(os.path.join(out, "figures_captions.md"), "w", encoding="utf-8") as f:
        f.write(captions(C))
    merged = {**prev, **checks} if only else checks
    with open(rc_path, "w", encoding="utf-8") as f:
        json.dump({s: merged[s] for s, _ in FIGURES if s in merged}, f, indent=1)
    return C, checks, figs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True, help="JSON file mapping input keys to file paths")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args(argv)
    with open(a.inputs, encoding="utf-8") as f:
        inp = json.load(f)
    _, checks, _ = run(inp, a.out)
    for k, v in checks.items():
        print(k, "overlaps:", len(v["bbox_overlaps"]), "outside:", len(v["text_outside_figure"]))


if __name__ == "__main__":
    main()
