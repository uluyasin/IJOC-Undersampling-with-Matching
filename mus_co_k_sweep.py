"""Candidate-graph study of MUS-CO (Table S1 of the Online Supplement).

For every test instance and k in {5, 10, 20}, the minimum-weight perfect matching on the candidate graph G_k of MUS-CO
(k nearest neighbours plus the stable matching) is computed with matchsampling.min_weight_matching, and its cost is
compared with the minimum-weight matching cost on the complete graph.

Instances (listed in data/mus_co_complete_graph_optima.csv):
  uniform    n points drawn uniformly from [0, 1]^d with numpy.random.default_rng(1000 * n + d + rep);
  clustered  the 500 majority points (d = 10) of synth(500, 50, 10, seed=0) below, a Gaussian mixture scaled to [0, 1].

The complete-graph optima in data/mus_co_complete_graph_optima.csv were certified during development by linear programming
over all pairs (odd-set cuts and column generation; every reduced cost >= -1e-7 at the optimum). That development solver
is not part of this repository, so the optima are provided as data. The candidate-graph costs computed here are exact
optima on G_k and do not depend on the solver used; the run times do (they were measured on a machine shared with other
processes and are not reported in the paper).

Output: results/mus_co_k_sweep.csv with the columns instance, n, d, rep, k, seconds, n_edges, cost_candidate_graph,
cost_complete_graph_optimum, rel_gap_pct (= 100 * (candidate - complete) / complete), unmatched_beyond_parity,
optimal_on_candidates, solver; the per-k summary of rel_gap_pct is printed (Table S1).

Usage (from the repository root):
    python pipeline/mus_co_k_sweep.py                      # all instances, k = 5, 10, 20 (about 10-30 minutes)
    python pipeline/mus_co_k_sweep.py --check FILE         # also compare the candidate-graph costs with an earlier run
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

OPTIMA = "data/mus_co_complete_graph_optima.csv"
OUT = "results/mus_co_k_sweep.csv"


def load_matchsampling(path):
    spec = importlib.util.spec_from_file_location("matchsampling", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synth(n, p, d, seed):
    """Gaussian-mixture data of the run-time benchmark: n majority points around 3 centres, p minority points around 2."""
    r = np.random.default_rng(seed)
    cm = r.normal(0, 2.0, (3, d))
    cp = cm[:2] + r.normal(0, 1.0, (2, d))
    N = cm[r.integers(0, 3, n)] + r.normal(0, 1.0, (n, d))
    P = cp[r.integers(0, 2, p)] + r.normal(0, 0.7, (p, d))
    X = np.vstack([N, P])
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    return X, np.r_[np.zeros(n, int), np.ones(p, int)]


def points(row):
    n, d, rep = int(row.n), int(row.d), int(row.rep)
    if row.kind == "uniform":
        return np.random.default_rng(1000 * n + d + rep).random((n, d))
    if row.kind == "clustered":
        X, y = synth(n, 50, d, rep)
        return X[y == 0]
    raise ValueError(row.kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms-path", default="src/matchsampling.py")
    ap.add_argument("--optima", default=OPTIMA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--check", default=None, help="earlier k-sweep CSV whose candidate-graph costs are compared")
    a = ap.parse_args()
    ms = load_matchsampling(a.ms_path)
    opt = pd.read_csv(a.optima)
    rows = []
    for row in opt.itertuples(index=False):
        X = points(row)
        for k in a.ks:
            t0 = time.perf_counter()
            pairs, costs, info = ms.min_weight_matching(X, n_candidates=k)
            sec = time.perf_counter() - t0
            c, c_opt = float(np.sum(costs)), float(row.cost_complete_graph_optimum)
            rows.append(dict(instance=row.instance, n=int(row.n), d=int(row.d), rep=int(row.rep), k=int(k), seconds=sec,
                             n_edges=int(info["n_edges"]), cost_candidate_graph=c, cost_complete_graph_optimum=c_opt,
                             rel_gap_pct=100 * (c - c_opt) / c_opt, unmatched_beyond_parity=int(row.n) // 2 - len(pairs),
                             optimal_on_candidates=bool(info["optimal"]), solver=info["solver"]))
            print(f"{row.instance}, k={k}: gap {rows[-1]['rel_gap_pct']:.4f}%, {sec:.1f} s, {info['solver']}", flush=True)
    sw = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sw.to_csv(a.out, index=False)
    sw["kind"] = np.where(sw.instance.str.startswith("uniform"), "uniform", "clustered")
    print(sw.groupby(["kind", "k"]).agg(instances=("instance", "nunique"), mean_gap_pct=("rel_gap_pct", "mean"),
                                         max_gap_pct=("rel_gap_pct", "max")).round(4).to_string())
    if a.check:
        old = pd.read_csv(a.check)
        m = sw.merge(old[["instance", "k", "cost_candidate_graph"]], on=["instance", "k"], suffixes=("", "_earlier"))
        rel = np.abs(m.cost_candidate_graph - m.cost_candidate_graph_earlier) / m.cost_candidate_graph_earlier
        print(f"compared {len(m)} rows with {a.check}: maximum relative cost difference {rel.max():.2e}")
        sys.exit(0 if rel.max() < 1e-9 else 1)


if __name__ == "__main__":
    main()
