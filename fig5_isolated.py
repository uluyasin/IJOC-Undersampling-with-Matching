"""Figure 5 with the isolated single-thread timings (results/revision/timing_isolated_summary.csv) instead of the
timings taken under parallel load. Reuses compute() and fig5() of figures_pipeline.py unchanged; only the median
times, the MUS-CO/MUS-CS ratio, the x-range and the x-label differ.

Usage (from the package root):  python pipeline/fig5_isolated.py configs/figures_inputs.json figs
"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd
import figures_pipeline as fpm

inp = json.load(open(sys.argv[1], encoding="utf-8"))
out = sys.argv[2] if len(sys.argv) > 2 else "figs"
os.makedirs(out, exist_ok=True)
fpm.apply_style()
C = fpm.compute(inp)
iso = pd.read_csv(os.path.join("results", "revision", "timing_isolated_summary.csv")).set_index("sampler")
F = C["F5"].copy()
F["t_median"] = F["method"].map(iso["median"])
assert F["t_median"].notna().all()
C["F5"] = F
C["ratio_CO_CS"] = float(iso.loc["MUS-CO", "median"] / iso.loc["MUS-CS", "median"])
fpm.F5_LABEL["CC"] = ("cluster centroids (CC)", (0, 9), "center")   # above the marker: CC moves next to RUS in isolation
fig = fpm.fig5(C)
ax = fig.axes[0]
ax.set_xlim(0.0025, 6.0)
ax.xaxis.set_major_locator(fpm.FixedLocator([0.01, 0.1, 1]))
ax.set_xlabel("median resampling time per training fold, measured in isolation (log scale)")
bad, outside = fpm.check_overlaps(fig)
print("overlaps", len(bad), "outside", len(outside))
fig.savefig(os.path.join(out, "fig5_cost.pdf"))
fig.savefig(os.path.join(out, "fig5_cost.png"), dpi=300)
print("ratio MUS-CO/MUS-CS", round(C["ratio_CO_CS"], 2), "max rank shift", round(C["max_rank_shift"], 2))
