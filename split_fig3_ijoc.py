"""Split the IJOC version of the mechanism figure into panels (a)-(b) and (c)-(d).

Figure 4 of the paper shows panels (a)-(b) of the mechanism figure and Figure S1 of the Online Supplement panels (c)-(d).
The script builds the figure with the set-up code of pipeline/figures_ijoc.py (same inputs: configs/figures_inputs.json,
that is, the analysis tables of steps 3 and 4 of the README) and cuts it between the two panel rows.

Usage (from the repository root, after pipeline/figures_ijoc.py):
    python pipeline/split_fig3_ijoc.py
Writes figs_ijoc/fig3_mechanism_ab.pdf/.png (Figure 4) and figs_ijoc/fig3_mechanism_cd.pdf/.png (Figure S1).
"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
sys.argv = ["figures_ijoc.py", "configs/figures_inputs.json", "figs_ijoc", "fig3_mechanism"]
HERE = os.path.join(os.getcwd(), "pipeline"); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "figures_ijoc.py"), encoding="utf-8").read()
src = src.split("FIGS = [")[0]          # set-up part only: font scaling, patches, compute(C)
exec(compile(src, "figures_ijoc_setup", "exec"))
from matplotlib.transforms import Bbox
fig = _fig3_ijoc(C)
fig.canvas.draw()
r = fig.canvas.get_renderer()
W_, H_ = fig.get_size_inches()
axes = [a for a in fig.axes if a.get_visible()]
mids = sorted(((a.get_position().y0 + a.get_position().y1) / 2, i) for i, a in enumerate(axes))
ys = [m for m, _ in mids]
cut_y = None
# rows: the largest gap in axes centres separates the top row (a, b) from the bottom row (c, d)
gaps = [(ys[i + 1] - ys[i], i) for i in range(len(ys) - 1)]
g, gi = max(gaps)
bottom_idx = [mids[j][1] for j in range(gi + 1)]; top_idx = [mids[j][1] for j in range(gi + 1, len(mids))]
def union(idx):
    bbs = [axes[i].get_tightbbox(r) for i in idx]
    return Bbox.union(bbs)
tb, bb = union(top_idx), union(bottom_idx)
top_lo = tb.y0 / fig.dpi; bot_hi = bb.y1 / fig.dpi
cut = (top_lo + bot_hi) / 2
print("top row lower edge", round(top_lo, 3), "bottom row upper edge", round(bot_hi, 3), "cut", round(cut, 3), "H", H_)
for t in fig.texts:
    bbt = t.get_window_extent(r); y0, y1 = bbt.y0 / fig.dpi, bbt.y1 / fig.dpi
    if y0 < cut < y1: print("TEXT CROSSES CUT:", t.get_text()[:40])
fig.savefig("figs_ijoc/fig3_mechanism_ab.pdf", bbox_inches=Bbox([[0, cut], [W_, H_]]))
fig.savefig("figs_ijoc/fig3_mechanism_ab.png", bbox_inches=Bbox([[0, cut], [W_, H_]]), dpi=300)
fig.savefig("figs_ijoc/fig3_mechanism_cd.pdf", bbox_inches=Bbox([[0, 0], [W_, cut]]))
fig.savefig("figs_ijoc/fig3_mechanism_cd.png", bbox_inches=Bbox([[0, 0], [W_, cut]]), dpi=300)
print("saved")
