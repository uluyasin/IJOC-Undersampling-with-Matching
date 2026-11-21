"""Selection illustration of the extended results (figures/figS_selection_illustration.pdf).

Applies random undersampling, NearMiss-1, ClusterCentroids (hard voting) and the four matching undersamplers to the
two-dimensional toy problem of Figure 1(a) of the paper (520 majority and 58 minority instances, generated with
numpy.random.default_rng(7) exactly as in pipeline/figures_pipeline.py) and plots the retained majority instances.
The samplers are applied to the raw coordinates with random_state=0; the ClusterCentroids output is mapped to the nearest
majority instances as in pipeline/bench_runner.py.

Usage (from the repository root):
    python pipeline/selection_illustration.py [--out docs/extended_results/figures] [--ms-path src/matchsampling.py]
Writes figS_selection_illustration.pdf and figS_selection_illustration.png to --out.
"""
import argparse
import importlib.util
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

C_MIN, C_KEEP, C_DROP, C_ALL = "#D55E00", "#0B4F8A", "#BDBDBD", "#555555"
PANELS = [("RUS", "Random undersampling", "(RUS)"), ("NM1", "NearMiss-1", "(closest to minority)"),
          ("CC", "ClusterCentroids", "(instance nearest each centroid)"),
          ("MUS-BS", "MUS-BS", "boundary arm, stable"), ("MUS-BO", "MUS-BO", "boundary arm, optimal"),
          ("MUS-CS", "MUS-CS", "coverage arm, stable"), ("MUS-CO", "MUS-CO", "coverage arm, optimal")]


def toy_data():
    """Toy problem of Figure 1(a), identical to toy_selections() in pipeline/figures_pipeline.py."""
    rng = np.random.default_rng(7)
    Xmaj = rng.normal(size=(520, 2)) * [1.0, 0.85]
    theta = rng.uniform(0.15 * np.pi, 0.85 * np.pi, 58)
    Xmin = np.c_[1.25 * np.cos(theta), 0.35 + np.sin(theta)] + rng.normal(scale=0.12, size=(58, 2))
    return Xmaj, Xmin


def selections(Xmaj, Xmin, ms_path):
    spec = importlib.util.spec_from_file_location("matchsampling", ms_path)
    msm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(msm)
    X = np.vstack([Xmaj, Xmin])
    y = np.r_[np.zeros(len(Xmaj), int), np.ones(len(Xmin), int)]
    samplers = {"RUS": RandomUnderSampler(random_state=0), "NM1": NearMiss(version=1),
                "CC": ClusterCentroids(voting="hard", random_state=0)}
    for key in ("MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"):
        arm = "boundary" if key[4] == "B" else "coverage"
        variant = "stable" if key[5] == "S" else "optimal"
        samplers[key] = msm.MatchingUnderSampler(arm, variant, random_state=0)
    kept = {}
    for key, s in samplers.items():
        X_res, y_res = s.fit_resample(X, y)
        if key == "CC":   # prototype output: map each retained majority row to the nearest majority instance
            _, ind = NearestNeighbors(n_neighbors=1).fit(Xmaj).kneighbors(X_res[y_res == 0])
            kept[key] = ind.ravel()
        else:
            idx = np.asarray(s.sample_indices_)
            kept[key] = idx[idx < len(Xmaj)]
        assert (y_res == 0).sum() == len(Xmin), f"{key} did not balance the classes"
    return kept


def plot(Xmaj, Xmin, kept):
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.0), sharex=True, sharey=True)
    axes = axes.ravel()
    circ = dict(marker="o", s=11, linewidths=0.6)
    tri = dict(marker="^", s=18, c=C_MIN, edgecolors="none", zorder=3)
    for j, ax in enumerate(axes):
        if j == 0:
            ax.scatter(*Xmaj.T, facecolors="none", edgecolors=C_ALL, **circ)
            title = "Training data\n520 majority, 58 minority"
        else:
            key, name, sub = PANELS[j - 1]
            mask = np.zeros(len(Xmaj), bool)
            mask[kept[key]] = True
            ax.scatter(*Xmaj[~mask].T, facecolors="none", edgecolors=C_DROP, **circ)
            ax.scatter(*Xmaj[mask].T, c=C_KEEP, edgecolors="none", marker="o", s=13, zorder=2)
            title = f"{name}\n{sub}"
        ax.scatter(*Xmin.T, **tri)
        ax.set_title(title, loc="left", fontsize=8, pad=3)
        ax.text(-0.06, 1.13, "abcdefgh"[j], transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], ls="none", marker="^", ms=6, mfc=C_MIN, mec="none"),
               Line2D([], [], ls="none", marker="o", ms=5, mfc=C_KEEP, mec="none"),
               Line2D([], [], ls="none", marker="o", ms=5, mfc="none", mec=C_DROP),
               Line2D([], [], ls="none", marker="o", ms=5, mfc="none", mec=C_ALL)]
    fig.legend(handles, ["minority", "majority, retained", "majority, discarded", "majority (panel a)"],
               loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.08, wspace=0.18, hspace=0.42)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/extended_results/figures")
    ap.add_argument("--ms-path", default="src/matchsampling.py")
    a = ap.parse_args()
    Xmaj, Xmin = toy_data()
    kept = selections(Xmaj, Xmin, a.ms_path)
    fig = plot(Xmaj, Xmin, kept)
    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"figS_selection_illustration.{ext}"), dpi=300, bbox_inches="tight")
    print("retained majority instances:", {k: len(np.unique(v)) for k, v in kept.items()})


if __name__ == "__main__":
    main()
