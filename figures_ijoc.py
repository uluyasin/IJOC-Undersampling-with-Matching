"""IJOC versions of the figures: identical content and layout to figures_pipeline.py, but all font sizes are
scaled by 7.2/6.5 so that the figures, drawn on a 7.2-in canvas and included at the 6.5-in text width of the
INFORMS submission format (12 pt, 1-in margins), print at the intended 8/7/6 pt. Figure 5 uses the isolated
single-thread timings (as pipeline/fig5_isolated.py).

Usage (from the package root):  python pipeline/figures_ijoc.py configs/figures_inputs.json figs_ijoc
"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import matplotlib.pyplot as plt
import figures_pipeline as fpm

K = 7.2 / 6.5
fpm.FS_BASE, fpm.FS_SEC, fpm.FS_TICK = [round(v * K, 2) for v in (fpm.FS_BASE, fpm.FS_SEC, fpm.FS_TICK)]
for key in ["font.size", "axes.labelsize", "axes.titlesize", "legend.fontsize", "xtick.labelsize", "ytick.labelsize"]:
    if key in fpm.STYLE_RC and isinstance(fpm.STYLE_RC[key], (int, float)):
        fpm.STYLE_RC[key] = round(fpm.STYLE_RC[key] * K, 2)


# IJOC style: figure titles live in the captions, so the in-figure headline texts are not drawn
fpm.title_in = lambda fig, ax, x_in, y_in, s: None
fpm.TOY_TITLE = {"RUS": "random (RUS)", "NM1": "NearMiss-1", "MUS-BS": "MUS-BS (boundary)", "MUS-CS": "MUS-CS (coverage)"}
_fig3, _fig4 = fpm.fig3, fpm.fig4
def _fig3_ijoc(C):
    fig = _fig3(C)
    for ax in fig.axes:
        if ax.get_xlabel().startswith("change in coverage error"):
            ax.set_xlabel("change in CE vs RUS (< 0: better coverage)")
    return fig
def _fig4_ijoc(C):
    fig = _fig4(C)
    for ax in fig.axes:
        for t_ in list(ax.texts):
            if t_.get_text().startswith("Right of the dotted line"):
                t_.remove()
    return fig
fpm.fig3, fpm.fig4 = _fig3_ijoc, _fig4_ijoc

inp = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2] if len(sys.argv) > 2 else "figs_ijoc"
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)
fpm.apply_style()
C = fpm.compute(inp)

def fig5_isolated(C):
    iso = pd.read_csv(os.path.join("results", "revision", "timing_isolated_summary.csv")).set_index("sampler")
    F = C["F5"].copy()
    F["t_median"] = F["method"].map(iso["median"])
    assert F["t_median"].notna().all()
    C2 = dict(C); C2["F5"] = F
    C2["ratio_CO_CS"] = float(iso.loc["MUS-CO", "median"] / iso.loc["MUS-CS", "median"])
    fpm.F5_LABEL["CC"] = ("cluster centroids (CC)", (0, 9), "center")
    fig = fpm.fig5(C2)
    ax = fig.axes[0]
    ax.set_xlim(0.0025, 6.0)
    ax.xaxis.set_major_locator(fpm.FixedLocator([0.01, 0.1, 1]))
    ax.set_xlabel("median resampling time per training fold, measured in isolation (log scale)")
    return fig

FIGS = [("fig1_hook", fpm.fig1), ("fig2_evidence", fpm.fig2), ("fig3_mechanism", _fig3_ijoc), ("fig4_when", _fig4_ijoc),
        ("fig5_cost", fig5_isolated), ("figS_cd_gmean", lambda C: fpm.fig_cd(C, "gmean")),
        ("figS_cd_auc", lambda C: fpm.fig_cd(C, "auc")), ("figS_runtime_grid", fpm.fig_runtime_grid),
        ("figS_map_full", fpm.fig_map_full)]
report = {}
for stem, fn in FIGS:
    if only and stem not in only:
        continue
    fig = fn(C)
    fig.savefig(os.path.join(out, stem + ".pdf"))
    fig.savefig(os.path.join(out, stem + ".png"), dpi=300)
    bad, outside = fpm.check_overlaps(fig)
    w, h = fig.get_size_inches()
    report[stem] = dict(overlaps=[[str(a), str(b)] for a, b in bad], outside=[str(x) for x in outside],
                        canvas_in=[round(float(w), 2), round(float(h), 2)])
    print(stem, "overlaps:", len(bad), "outside:", len(outside))
    plt.close(fig)
json.dump(report, open(os.path.join(out, "render_checks_ijoc.json"), "w"), indent=1)
