"""Shared helpers for the referee-revision analyses (A7/A8 heavy computations).

All paths come from rev/paths.json (written by the analysis kernel from the artifact store).
Preprocessing replicates bench_runner.run_unit exactly: median imputation (keep_empty_features)
+ MinMax[0,1], both fitted on the training fold; Euclidean distances.
Selection records (selections_main_final.joblib): kept_majority = ORIGINAL dataset row indices of
the retained majority instances (bench_runner: kept_maj = tr[idx[y_tr[idx] == 0]]), duplicates
possible for ClusterCentroids (hard voting).

Exact W1 is computed as a transportation LP solved by HiGHS (scipy.optimize.linprog); POT could not
be installed (environment creation not approved, sandbox offline). The LP is the same one POT's
ot.emd2 solves (network simplex); HiGHS returns an optimal vertex to ~1e-9 relative tolerance.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))


def load_paths():
    return json.load(open(os.path.join(HERE, "paths.json")))


def scale_training_fold(X, tr):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sc = MinMaxScaler()
    return sc.fit_transform(imp.fit_transform(X[tr]))


def fold_arrays(data, folds, name, k):
    D = data[name]
    X = np.asarray(D["X"], dtype=float)
    y = np.asarray(D["y"]).astype(int)
    f = folds[name]["folds"][k]
    tr = np.asarray(f["train_idx"])
    Xtr = scale_training_fold(X, tr)
    ytr = y[tr]
    seed = int(folds[name]["seed"]) * 100 + k
    return Xtr, ytr, tr, int(f["repeat"]), int(f["fold"]), seed


def selection_index(sel):
    idx = {}
    for r in sel:
        key = (r["dataset"], int(r["repeat"]), int(r["fold"]), r["sampler"])
        if key in idx:
            raise ValueError(f"duplicate selection record {key}")
        idx[key] = np.asarray(r["kept_majority"], dtype=np.int64)
    return idx


def transport_lp(a, b, M):
    """min <M, x> s.t. x 1 = a, x^T 1 = b, x >= 0 (HiGHS). a, b integral-scaled masses with equal sums.
    Returns (objective, status message)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ns, nt = M.shape
    if ns == 0 or nt == 0:
        return 0.0, "empty"
    # row-sum constraints (sources) and column-sum constraints (sinks); x is row-major (i * nt + j)
    rows_s = np.repeat(np.arange(ns), nt)
    rows_t = ns + np.tile(np.arange(nt), ns)
    cols = np.arange(ns * nt)
    A = sp.csc_matrix((np.ones(2 * ns * nt), (np.concatenate([rows_s, rows_t]), np.concatenate([cols, cols]))),
                      shape=(ns + nt, ns * nt))
    beq = np.concatenate([a, b])
    # one redundant equality (total mass) is harmless for HiGHS presolve
    r = linprog(M.ravel(), A_eq=A, b_eq=beq, bounds=(0, None), method="highs",
                options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10, presolve=True))
    if r.status != 0:
        raise RuntimeError(f"transport LP failed: {r.message}")
    return float(r.fun), r.message


def _nw_corner(a, b):
    """North-west-corner feasible plan arcs (i, j) for integral masses a, b with equal sums."""
    a = np.asarray(a, dtype=np.int64).copy()
    b = np.asarray(b, dtype=np.int64).copy()
    i = j = 0
    arcs = []
    while i < len(a) and j < len(b):
        arcs.append((i, j))
        t = min(a[i], b[j])
        a[i] -= t
        b[j] -= t
        if a[i] == 0 and b[j] == 0:
            if i + 1 < len(a) and j + 1 < len(b):
                arcs.append((i + 1, j))   # keep the arc set connected (degenerate step)
            i += 1
            j += 1
        elif a[i] == 0:
            i += 1
        else:
            j += 1
    return arcs


def transport_cg(a, b, Xs, Xt, k=12, extra_cost_rows=None, max_iter=200, add_per_row=8, block=1024):
    """Exact transportation LP  min sum_ij d(Xs_i, Xt_j) x_ij, rows sum a, columns sum b, x >= 0,
    solved by column generation: HiGHS LP on a sparse arc set (k nearest sinks of every source, k nearest
    sources of every sink, a north-west-corner plan for feasibility), then pricing of ALL arcs with the LP
    duals; violated arcs (reduced cost < -tol) are added until none remains -> certified optimum over the
    complete arc set. `extra_cost_rows`: optional number of trailing zero-cost dummy sources (rows with
    index >= len(Xs)) whose arcs to every sink are always included.
    Returns dict(obj, iters, n_arcs, certified, min_rc)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ns_real, nt = len(Xs), len(Xt)
    nd = int(extra_cost_rows or 0)
    ns = ns_real + nd
    assert len(a) == ns
    kk = min(k, nt)
    # candidate arcs
    I, J = [], []
    for st in range(0, ns_real, block):
        D = cdist(Xs[st:st + block], Xt)
        nn = np.argpartition(D, kk - 1, axis=1)[:, :kk] if kk < nt else np.tile(np.arange(nt), (len(D), 1))
        I.append(np.repeat(np.arange(st, st + len(D)), nn.shape[1]))
        J.append(nn.ravel())
    ks = min(k, ns_real)
    for st in range(0, nt, block):
        D = cdist(Xt[st:st + block], Xs)
        nn = np.argpartition(D, ks - 1, axis=1)[:, :ks] if ks < ns_real else np.tile(np.arange(ns_real), (len(D), 1))
        J.append(np.repeat(np.arange(st, st + len(D)), nn.shape[1]))
        I.append(nn.ravel())
    nw = np.array(_nw_corner(np.rint(a).astype(np.int64), np.rint(b).astype(np.int64)), dtype=np.int64).reshape(-1, 2)
    I.append(nw[:, 0]); J.append(nw[:, 1])
    if nd:
        I.append(np.repeat(np.arange(ns_real, ns), nt)); J.append(np.tile(np.arange(nt), nd))
    key = np.unique(np.concatenate(I).astype(np.int64) * nt + np.concatenate(J))
    scale = 1.0
    it = 0
    certified = False
    min_rc = np.nan
    while it < max_iter:
        it += 1
        ii, jj = key // nt, key % nt
        real = ii < ns_real
        c = np.zeros(len(key))
        if real.any():
            c[real] = np.sqrt(np.sum((Xs[ii[real]] - Xt[jj[real]]) ** 2, axis=1))
        if it == 1:
            scale = max(1e-12, float(c.max()) if len(c) else 1.0)
        A = sp.csc_matrix((np.ones(2 * len(key)), (np.concatenate([ii, ns + jj]),
                                                    np.concatenate([np.arange(len(key))] * 2))),
                          shape=(ns + nt, len(key)))
        r = linprog(c, A_eq=A, b_eq=np.concatenate([a, b]), bounds=(0, None), method="highs",
                    options=dict(primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10))
        if r.status != 0:
            raise RuntimeError(f"transport LP failed: {r.message}")
        y = np.asarray(r.eqlin.marginals, dtype=float)
        u, v = y[:ns], y[ns:]
        tol = 1e-9 * scale
        new = []
        min_rc = np.inf
        for st in range(0, ns_real, block):
            en = min(st + block, ns_real)
            R = cdist(Xs[st:en], Xt) - u[st:en, None] - v[None, :]
            min_rc = min(min_rc, float(R.min()))
            bad = R < -tol
            if bad.any():
                rows = np.flatnonzero(bad.any(axis=1))
                Rb = R[rows]
                take = min(add_per_row, nt)
                sel_ = np.argpartition(Rb, take - 1, axis=1)[:, :take]
                vals = np.take_along_axis(Rb, sel_, axis=1)
                ok = vals < -tol
                new.append(((st + rows)[:, None].repeat(take, 1)[ok]).astype(np.int64) * nt + sel_[ok])
        if nd:  # dummy arcs are all present; their reduced costs are >= -tol at an LP optimum
            pass
        if not new:
            certified = True
            break
        key = np.unique(np.concatenate([key] + new))
    return dict(obj=float(r.fun), iters=it, n_arcs=int(len(key)), certified=certified, min_rc=float(min_rc))


def transport_cg_hot(a, b, Xs, Xt, k=24, extra_cost_rows=0, max_iter=500, add_per_row=24, block=1024,
                     max_add=400_000):
    """Exact transportation LP by column generation with HOT-STARTED HiGHS (low-level API, as used by the
    sampler module): initial arcs = k nearest sinks of every source + k nearest sources of every sink +
    north-west-corner plan (feasibility) + all arcs of the trailing zero-cost dummy sources; after each solve
    ALL source-sink arcs are priced with the row duals (reduced cost d_ij - u_i - v_j) and violated arcs are
    appended (addCols) and the LP is re-solved from the current basis. Stops when no arc has reduced cost
    < -1e-9 * max cost -> certified optimum over the complete arc set."""
    from scipy.optimize._highspy import _core as hc
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ns_real, nt = len(Xs), len(Xt)
    nd = int(extra_cost_rows or 0)
    ns = ns_real + nd
    assert len(a) == ns
    included = np.zeros((ns_real, nt), dtype=bool)
    kk, ks = min(k, nt), min(k, ns_real)
    dmax = 0.0
    for st in range(0, ns_real, block):
        en = min(st + block, ns_real)
        D = cdist(Xs[st:en], Xt)
        dmax = max(dmax, float(D.max()))
        nn = np.argpartition(D, kk - 1, axis=1)[:, :kk] if kk < nt else np.tile(np.arange(nt), (en - st, 1))
        included[np.repeat(np.arange(st, en), nn.shape[1]), nn.ravel()] = True
    for st in range(0, nt, block):
        en = min(st + block, nt)
        D = cdist(Xt[st:en], Xs)
        nn = np.argpartition(D, ks - 1, axis=1)[:, :ks] if ks < ns_real else np.tile(np.arange(ns_real), (en - st, 1))
        included[nn.ravel(), np.repeat(np.arange(st, en), nn.shape[1])] = True
    nw = np.array(_nw_corner(np.rint(a).astype(np.int64), np.rint(b).astype(np.int64)), dtype=np.int64).reshape(-1, 2)
    nwr = nw[nw[:, 0] < ns_real]
    included[nwr[:, 0], nwr[:, 1]] = True
    ii, jj = np.nonzero(included)
    cost = np.sqrt(np.sum((Xs[ii] - Xt[jj]) ** 2, axis=1))
    if nd:
        ii = np.concatenate([ii, np.repeat(np.arange(ns_real, ns), nt)])
        jj = np.concatenate([jj, np.tile(np.arange(nt), nd)])
        cost = np.concatenate([cost, np.zeros(nd * nt)])
    E = len(ii)
    h = hc._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("parallel", "off")
    h.setOptionValue("presolve", "off")
    h.setOptionValue("primal_feasibility_tolerance", 1e-10)
    h.setOptionValue("dual_feasibility_tolerance", 1e-10)
    lp = hc.HighsLp()
    lp.num_col_ = E
    lp.num_row_ = ns + nt
    lp.col_cost_ = cost
    lp.col_lower_ = np.zeros(E)
    lp.col_upper_ = np.full(E, hc.kHighsInf)
    lp.row_lower_ = np.concatenate([a, b])
    lp.row_upper_ = np.concatenate([a, b])
    lp.a_matrix_.format_ = hc.MatrixFormat.kColwise
    lp.a_matrix_.start_ = (2 * np.arange(E + 1)).astype(np.int32)
    lp.a_matrix_.index_ = np.stack([ii, ns + jj], axis=1).ravel().astype(np.int32)
    lp.a_matrix_.value_ = np.ones(2 * E)
    lp.a_matrix_.num_col_ = E
    lp.a_matrix_.num_row_ = ns + nt
    h.passModel(lp)
    tol = 1e-9 * max(dmax, 1e-12)
    it, certified, min_rc, n_arcs = 0, False, np.nan, E
    while it < max_iter:
        it += 1
        h.run()
        if h.getModelStatus() != hc.HighsModelStatus.kOptimal:
            raise RuntimeError(f"transport LP not optimal: {h.modelStatusToString(h.getModelStatus())}")
        y = np.asarray(h.getSolution().row_dual, dtype=float)
        u, v = y[:ns], y[ns:]
        newi, newj, newc = [], [], []
        min_rc = np.inf
        for st in range(0, ns_real, block):
            en = min(st + block, ns_real)
            D = cdist(Xs[st:en], Xt)
            R = D - u[st:en, None] - v[None, :]
            min_rc = min(min_rc, float(R.min()))
            R[included[st:en]] = np.inf
            bad = R < -tol
            if not bad.any():
                continue
            rows = np.flatnonzero(bad.any(axis=1))
            Rb = R[rows]
            take = min(add_per_row, nt)
            sel_ = np.argpartition(Rb, take - 1, axis=1)[:, :take]
            vals = np.take_along_axis(Rb, sel_, axis=1)
            ok = vals < -tol
            ri = (st + rows)[:, None].repeat(take, 1)[ok]
            cj = sel_[ok]
            newi.append(ri); newj.append(cj); newc.append(D[ri - st, cj])
        if not newi:
            certified = True
            break
        ni, nj, nc = np.concatenate(newi), np.concatenate(newj), np.concatenate(newc)
        if len(ni) > max_add:
            o = np.argsort(nc - u[ni] - v[nj])[:max_add]
            ni, nj, nc = ni[o], nj[o], nc[o]
        included[ni, nj] = True
        K = len(ni)
        h.addCols(K, nc, np.zeros(K), np.full(K, hc.kHighsInf), 2 * K, (2 * np.arange(K)).astype(np.int32),
                  np.stack([ni, ns + nj], axis=1).ravel().astype(np.int32), np.ones(2 * K))
        n_arcs += K
    obj = float(h.getInfo().objective_function_value)
    return dict(obj=obj, iters=it, n_arcs=int(n_arcs), certified=certified, min_rc=float(min_rc), tol=tol)


def w1_full_vs_subset_hot(N, s_pos, k=24):
    n = len(N)
    s_pos = np.asarray(s_pos, dtype=np.int64)
    m = len(s_pos)
    if m == 0:
        return np.nan, {}
    u, c = np.unique(s_pos, return_counts=True)
    resid = n * c - m
    if np.any(resid < 0):
        raise ValueError("unexpected negative residual (m > n?)")
    keep = resid > 0
    mask = np.ones(n, dtype=bool)
    mask[u] = False
    u, resid = u[keep], resid[keep]
    src = np.flatnonzero(mask)
    if len(src) == 0 or len(u) == 0:
        return 0.0, dict(iters=0, certified=True)
    out = transport_cg_hot(np.full(len(src), float(m)), resid.astype(float), N[src], N[u], k=k)
    return out["obj"] / (n * m), out


def w1_weighted_hot(P, q, S, k=24):
    m = int(np.sum(q))
    out = transport_cg_hot(np.asarray(q, float), np.ones(len(S)), P, S, k=k)
    return out["obj"] / m, out


def w1_best_subset_hot(P, q, N, k=24):
    m = int(np.sum(q))
    n = len(N)
    a = np.concatenate([np.asarray(q, float), [float(n - m)]])
    out = transport_cg_hot(a, np.ones(n), P, N, k=k, extra_cost_rows=1)
    return out["obj"] / m, out


def w1_full_vs_subset_cg(N, s_pos, k=12):
    """As w1_full_vs_subset, solved by certified column generation."""
    n = len(N)
    s_pos = np.asarray(s_pos, dtype=np.int64)
    m = len(s_pos)
    if m == 0:
        return np.nan, {}
    u, c = np.unique(s_pos, return_counts=True)
    resid = n * c - m
    if np.any(resid < 0):
        raise ValueError("unexpected negative residual (m > n?)")
    keep = resid > 0
    mask = np.ones(n, dtype=bool)
    mask[u] = False
    u, resid = u[keep], resid[keep]
    src = np.flatnonzero(mask)
    if len(src) == 0 or len(u) == 0:
        return 0.0, dict(iters=0, certified=True)
    out = transport_cg(np.full(len(src), float(m)), resid.astype(float), N[src], N[u], k=k)
    return out["obj"] / (n * m), out


def w1_weighted_cg(P, q, S, k=12):
    m = int(np.sum(q))
    out = transport_cg(np.asarray(q, float), np.ones(len(S)), P, S, k=k)
    return out["obj"] / m, out


def w1_best_subset_cg(P, q, N, k=12):
    m = int(np.sum(q))
    n = len(N)
    a = np.concatenate([np.asarray(q, float), [float(n - m)]])
    out = transport_cg(a, np.ones(n), P, N, k=k, extra_cost_rows=1)
    return out["obj"] / m, out


def w1_full_vs_subset(N, s_pos):
    """Exact W1 between the uniform measure on the rows of N (n points) and the uniform measure on
    the multiset N[s_pos] (m points, multiplicities allowed). Kantorovich-Rubinstein: W1 depends only on
    mu_N - mu_S, so the common mass min(1/n, c_j/m) = 1/n stays in place at every retained point and the
    LP is solved between the positive and negative parts (scaled by n*m to integers)."""
    n = len(N)
    s_pos = np.asarray(s_pos, dtype=np.int64)
    m = len(s_pos)
    if m == 0:
        return np.nan
    u, c = np.unique(s_pos, return_counts=True)
    resid = n * c - m                         # sink masses (scaled); >= 0 because m <= n and c >= 1
    if np.any(resid < 0):
        raise ValueError("unexpected negative residual (m > n?)")
    keep = resid > 0
    u, resid = u[keep], resid[keep]
    mask = np.ones(n, dtype=bool)
    mask[np.unique(s_pos)] = False
    src = np.flatnonzero(mask)
    if len(src) == 0 or len(u) == 0:
        return 0.0
    M = cdist(N[src], N[u])
    a = np.full(len(src), float(m))
    obj, _ = transport_lp(a, resid.astype(float), M)
    return obj / (n * m)


def w1_weighted(P, q, S):
    """Exact W1 between sum_i q_i/m delta_{P_i} and the uniform measure on the rows of S (m = sum q)."""
    m = int(np.sum(q))
    M = cdist(P, S)
    obj, _ = transport_lp(np.asarray(q, float), np.ones(len(S)), M)
    return obj / m


def w1_best_subset(P, q, N):
    """min over m-subsets S of N of W1(sum q_i/m delta_{P_i}, uniform(S)), m = sum q, via the transportation LP
    with majority capacities 1 and a zero-cost dummy source absorbing the n - m unused units (integral polytope)."""
    m = int(np.sum(q))
    n = len(N)
    M = np.vstack([cdist(P, N), np.zeros((1, n))])
    a = np.concatenate([np.asarray(q, float), [float(n - m)]])
    obj, _ = transport_lp(a, np.ones(n), M)
    return obj / m


def diameter(X, block=2048):
    """Exact diameter (max pairwise Euclidean distance) of the rows of X."""
    X = np.asarray(X, float)
    s = len(X)
    if s < 2:
        return 0.0
    best = 0.0
    for st in range(0, s, block):
        D = cdist(X[st:st + block], X[st:])
        best = max(best, float(D.max()))
    return best



NS_SCALE = 1e9        # network simplex works on integer costs: rint(d * NS_SCALE / max d)


def transport_ns(a, b, M):
    """Exact transportation problem by networkx network simplex (integer supplies a, demands b, dense costs M).
    Costs are scaled to integers (rint(M * 1e9 / max M)); the optimal flow for the scaled costs is re-evaluated with
    the exact float costs. Rounding bound on the objective error: total flow * max(M) * 1e-9.
    Returns dict(obj, total_flow, rounding_bound)."""
    import networkx as nx
    a = np.rint(np.asarray(a, float)).astype(np.int64)
    b = np.rint(np.asarray(b, float)).astype(np.int64)
    if a.sum() != b.sum():
        raise ValueError("unbalanced transportation problem")
    ns, nt = M.shape
    mx = float(M.max()) if M.size else 0.0
    scale = NS_SCALE / mx if mx > 0 else 1.0
    W = np.rint(M * scale).astype(np.int64)
    G = nx.DiGraph()
    G.add_nodes_from((i, {"demand": -int(a[i])}) for i in range(ns))
    G.add_nodes_from((ns + j, {"demand": int(b[j])}) for j in range(nt))
    ii, jj = np.nonzero(np.ones((ns, nt), dtype=bool))
    G.add_edges_from(zip(ii.tolist(), (ns + jj).tolist(), ({"weight": int(w)} for w in W.ravel().tolist())))
    _, flow = nx.network_simplex(G)
    obj, out_s, in_t = 0.0, np.zeros(ns), np.zeros(nt)
    for i in range(ns):
        for t, f_ in flow[i].items():
            if f_:
                obj += M[i, t - ns] * f_
                out_s[i] += f_
                in_t[t - ns] += f_
    if not (np.array_equal(out_s, a) and np.array_equal(in_t, b)):
        raise RuntimeError("network simplex returned an infeasible flow")
    return dict(obj=float(obj), total_flow=int(a.sum()), rounding_bound=float(a.sum() * mx / NS_SCALE))


def w1_full_vs_subset_ns(N, s_pos):
    n = len(N)
    s_pos = np.asarray(s_pos, dtype=np.int64)
    m = len(s_pos)
    u, c = np.unique(s_pos, return_counts=True)
    resid = n * c - m
    keep = resid > 0
    mask = np.ones(n, dtype=bool)
    mask[u] = False
    u, resid = u[keep], resid[keep]
    src = np.flatnonzero(mask)
    if len(src) == 0 or len(u) == 0:
        return 0.0, dict(rounding_bound_rel=0.0)
    out = transport_ns(np.full(len(src), m), resid, cdist(N[src], N[u]))
    return out["obj"] / (n * m), dict(rounding_bound_rel=out["rounding_bound"] / max(out["obj"], 1e-300))


def w1_weighted_ns(P, q, S):
    m = int(np.sum(q))
    out = transport_ns(q, np.ones(len(S), dtype=np.int64), cdist(P, S))
    return out["obj"] / m, out


def w1_best_subset_ns(P, q, N):
    m = int(np.sum(q))
    n = len(N)
    M = np.vstack([cdist(P, N), np.zeros((1, n))])
    out = transport_ns(np.concatenate([np.asarray(q, np.int64), [n - m]]), np.ones(n, dtype=np.int64), M)
    return out["obj"] / m, out
