"""Test-suite for matchsampling.py (run with ``pytest -q test_matchsampling.py``).

Covers: stability (no blocking pairs), greedy == minority-proposing DA ==
majority-proposing DA under distance-induced preferences (P1), invariance of the
selected majority subset between the two DA directions under random strict
preferences (P2, rural hospitals), agreement with the reference library
``matching`` (HospitalResident, StableRoommates), the ball characterisation (P3),
Wasserstein optimality of MUS-BO against brute force (P4), the per-round
Wasserstein bound of the coverage arm (P5), exactness of the minimum-weight
matching against networkx and brute force, exact output sizes, determinism and
imbalanced-learn Pipeline compatibility.
"""
import itertools
import warnings

import numpy as np
import pytest
from scipy.optimize import linprog

import matchsampling as ms

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------- helpers
def _instance(seed, p=None, n=None, d=None, discrete=False):
    r = np.random.default_rng(seed)
    p = int(r.integers(1, 7)) if p is None else p
    n = int(r.integers(p, 3 * p + 8)) if n is None else n
    d = int(r.integers(1, 5)) if d is None else d
    P, N = r.random((p, d)), r.random((n, d))
    if discrete:
        P, N = np.round(P * 3) / 3, np.round(N * 3) / 3
    return P, N, r


def _w1_uniform_lp(A, wa, B, wb):
    """Exact W1 between discrete measures sum wa_i delta_{A_i} and sum wb_j delta_{B_j}."""
    C = ms.exact_distances(A, B)
    na, nb = C.shape
    Aeq = np.zeros((na + nb, na * nb))
    for i in range(na):
        Aeq[i, i * nb:(i + 1) * nb] = 1
    for j in range(nb):
        Aeq[na + j, j::nb] = 1
    res = linprog(C.ravel(), A_eq=Aeq, b_eq=np.r_[wa, wb], bounds=(0, None), method="highs")
    assert res.status == 0
    return res.fun


def _brute_mwpm(D):
    """Minimum-weight maximum-cardinality matching by exhaustive search (tiny n)."""
    n = len(D)

    def rec(free):
        if len(free) <= 1:
            return 0.0
        a = free[0]
        best = np.inf
        rest = free[1:]
        if len(free) % 2 == 1:  # a may stay unmatched (odd count)
            best = rec(rest)
        for t, b in enumerate(rest):
            best = min(best, D[a, b] + rec(rest[:t] + rest[t + 1:]))
        return best

    return rec(list(range(n)))


def _data(n_maj, n_min, d=3, seed=0):
    r = np.random.default_rng(seed)
    X = np.vstack([r.normal(0, 1, (n_maj, d)), r.normal(1.0, 0.7, (n_min, d))])
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    y = np.r_[np.zeros(n_maj, int), np.ones(n_min, int)]
    return X, y


ALL = [ms.MUS_BS, ms.MUS_BO, ms.MUS_CS, ms.MUS_CO]


# ----------------------------------------------------------------------------- checkers
def test_blocking_pair_checkers_detect_instability():
    D = np.array([[0.1, 0.5, 0.9], [0.4, 0.2, 0.3]])  # 2 hospitals x 3 residents
    mo, mj = ms.preference_orders(D)
    bad = np.array([1, 0, -1])  # r0->h1, r1->h0 is unstable (h0,r0 both prefer each other)
    assert len(ms.hr_blocking_pairs(bad, mo, mj, np.array([1, 1]))) > 0
    good, _ = ms.hr_greedy(np.array([[0.0], [1.0]]), np.array([[0.1], [0.9], [0.5]]), np.array([1, 1]))
    Dg = ms.exact_distances(np.array([[0.0], [1.0]]), np.array([[0.1], [0.9], [0.5]]))
    assert len(ms.hr_blocking_pairs(good, *ms.preference_orders(Dg), np.array([1, 1]))) == 0
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    D2 = ms.exact_distances(X, X)
    assert len(ms.roommates_blocking_pairs(np.array([[0, 2], [1, 3]]), D2)) > 0
    assert len(ms.roommates_blocking_pairs(np.array([[0, 1], [2, 3]]), D2)) == 0


# ----------------------------------------------------------------------------- P1 (HR)
@pytest.mark.parametrize("seed", range(60))
def test_hr_greedy_equals_both_da_directions_symmetric(seed):
    P, N, r = _instance(seed, discrete=(seed % 3 == 0))
    m = int(r.integers(1, len(N) + 1))
    q = ms.capacities_for_target(len(P), m, seed)
    g, _ = ms.hr_greedy(P, N, q)
    D = ms.exact_distances(P, N)
    mo, mj = ms.preference_orders(D)
    a1 = ms.hr_deferred_acceptance(mo, mj, q, "minority")
    a2 = ms.hr_deferred_acceptance(mo, mj, q, "majority")
    assert np.array_equal(g, a1) and np.array_equal(g, a2)
    assert len(ms.hr_blocking_pairs(g, mo, mj, q)) == 0
    assert (g >= 0).sum() == m and np.all(np.bincount(g[g >= 0], minlength=len(P)) == q)


def test_hr_greedy_with_duplicates_and_d1():
    P = np.array([[0.5], [0.5], [0.2]])
    N = np.array([[0.5], [0.5], [0.5], [0.1], [0.9], [0.2], [0.2]])
    q = np.array([2, 1, 2])
    g, _ = ms.hr_greedy(P, N, q)
    mo, mj = ms.preference_orders(ms.exact_distances(P, N))
    assert np.array_equal(g, ms.hr_deferred_acceptance(mo, mj, q, "minority"))
    assert np.array_equal(g, ms.hr_deferred_acceptance(mo, mj, q, "majority"))
    assert len(ms.hr_blocking_pairs(g, mo, mj, q)) == 0


# ----------------------------------------------------------------------------- P2 (rural hospitals)
def test_selected_subset_invariant_between_da_directions_random_preferences():
    differing_assignments = 0
    for seed in range(200):
        r = np.random.default_rng(seed)
        p = int(r.integers(1, 6))
        n = int(r.integers(p, 14))
        q = r.multinomial(int(r.integers(1, n + 1)) - 0, np.ones(p) / p)  # arbitrary capacities, sum <= n
        q = np.minimum(q, n)
        if q.sum() > n:
            continue
        mo = np.array([r.permutation(n) for _ in range(p)])
        mj = np.array([r.permutation(p) for _ in range(n)])
        a1 = ms.hr_deferred_acceptance(mo, mj, q, "minority")
        a2 = ms.hr_deferred_acceptance(mo, mj, q, "majority")
        assert len(ms.hr_blocking_pairs(a1, mo, mj, q)) == 0
        assert len(ms.hr_blocking_pairs(a2, mo, mj, q)) == 0
        assert np.array_equal(a1 >= 0, a2 >= 0)  # same selected majority subset
        assert np.array_equal(np.bincount(a1[a1 >= 0], minlength=p), np.bincount(a2[a2 >= 0], minlength=p))
        assert (a1 >= 0).sum() == q.sum()
        differing_assignments += int(not np.array_equal(a1, a2))
    assert differing_assignments > 0  # the invariance is non-trivial: assignments do differ


def test_asymmetric_sampler_direction_invariance():
    X, y = _data(80, 12, d=4, seed=3)
    s = ms.MatchingUnderSampler("boundary", "stable", preferences="asymmetric", random_state=0)
    s.fit_resample(X, y)
    P, N = X[y == 1], X[y == 0]
    Wp, Wn = ms.mahalanobis_whitener(P), ms.mahalanobis_whitener(N)
    mo, mj = ms.preference_orders(ms.exact_distances(P @ Wp, N @ Wp), ms.exact_distances(N @ Wn, P @ Wn))
    a2 = ms.hr_deferred_acceptance(mo, mj, s.capacities_, "majority")
    assert np.array_equal(np.sort(s.pairs_[:, 1]), np.flatnonzero(y == 0)[a2 >= 0])


# ----------------------------------------------------------------------------- reference library
@pytest.mark.parametrize("seed", range(25))
def test_hr_agrees_with_matching_library(seed):
    HospitalResident = pytest.importorskip("matching.games").HospitalResident
    r = np.random.default_rng(1000 + seed)
    p = int(r.integers(1, 5))
    n = int(r.integers(p, 11))
    q = np.ones(p, int) + r.multinomial(int(r.integers(0, n - p + 1)), np.ones(p) / p)
    if seed % 2 == 0:  # random strict preferences
        mo = np.array([r.permutation(n) for _ in range(p)])
        mj = np.array([r.permutation(p) for _ in range(n)])
    else:  # distance preferences
        P, N = r.random((p, 2)), r.random((n, 2))
        mo, mj = ms.preference_orders(ms.exact_distances(P, N))
    res_prefs = {f"r{j}": [f"h{i}" for i in mj[j]] for j in range(n)}
    hos_prefs = {f"h{i}": [f"r{j}" for j in mo[i]] for i in range(p)}
    caps = {f"h{i}": int(q[i]) for i in range(p)}
    for optimal, proposer in (("hospital", "minority"), ("resident", "majority")):
        game = HospitalResident.create_from_dictionaries(res_prefs, hos_prefs, caps)
        sol = game.solve(optimal=optimal)
        lib = np.full(n, -1)
        for h, rs in sol.items():
            for rr in rs:
                lib[int(rr.name[1:])] = int(h.name[1:])
        ours = ms.hr_deferred_acceptance(mo, mj, q, proposer)
        assert np.array_equal(ours, lib)


@pytest.mark.parametrize("seed", range(25))
def test_roommates_greedy_agrees_with_matching_library(seed):
    StableRoommates = pytest.importorskip("matching.games").StableRoommates
    r = np.random.default_rng(2000 + seed)
    s = 2 * int(r.integers(1, 8))
    X = r.random((s, int(r.integers(1, 4))))
    if seed % 3 == 0:
        X = np.round(X * 4) / 4  # ties and duplicates
    D = ms.exact_distances(X, X)
    pairs, _ = ms.roommates_greedy(X)
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    order = np.argsort(Dm, axis=1, kind="stable")
    prefs = {f"p{a}": [f"p{b}" for b in order[a] if b != a] for a in range(s)}
    game = StableRoommates.create_from_dictionary(prefs)
    sol = game.solve()
    lib = {tuple(sorted((int(a.name[1:]), int(b.name[1:])))) for a, b in sol.items() if b is not None}
    assert lib == {tuple(pq) for pq in pairs.tolist()}


# ----------------------------------------------------------------------------- P1 (roommates)
@pytest.mark.parametrize("seed", range(40))
def test_roommates_greedy_is_stable_and_equals_mutual_nn(seed):
    r = np.random.default_rng(3000 + seed)
    s = int(r.integers(2, 40))
    X = r.random((s, int(r.integers(1, 4))))
    if seed % 2 == 0:
        X = np.round(X * 5) / 5
    pairs, costs = ms.roommates_greedy(X)
    D = ms.exact_distances(X, X)
    assert len(pairs) == s // 2
    assert len(ms.roommates_blocking_pairs(pairs, D)) == 0
    assert np.all(np.diff(costs) >= 0)  # accepted in increasing key order
    # repeated mutual-nearest-neighbour pairing under the global tie-break
    free = list(range(s))
    mnn = set()
    while len(free) > 1:
        F = np.array(free)
        sub = D[np.ix_(F, F)].copy()
        np.fill_diagonal(sub, np.inf)
        nn = F[np.lexsort((np.tile(F, (len(F), 1)), sub), axis=1)[:, 0]]
        nn_of = dict(zip(F.tolist(), nn.tolist()))
        new = {(min(a, b), max(a, b)) for a, b in nn_of.items() if nn_of[b] == a}
        assert new
        mnn |= new
        used = {v for pq in new for v in pq}
        free = [a for a in free if a not in used]
    assert mnn == {tuple(pq) for pq in pairs.tolist()}
    # partial greedy = the r closest pairs of the stable matching
    rr = int(r.integers(0, s // 2 + 1))
    part, _ = ms.roommates_greedy(X, max_pairs=rr)
    assert np.array_equal(part, pairs[:rr])


# ----------------------------------------------------------------------------- P3 (balls)
@pytest.mark.parametrize("seed", range(30))
def test_ball_characterisation_boundary_stable(seed):
    X, y = _data(int(30 + 7 * (seed % 5)), int(3 + seed % 6), d=2 + seed % 3, seed=seed)
    if seed % 3 == 0:
        X = np.round(X * 4) / 4  # ties on spheres
    alpha = [None, 0.5, 0.8][seed % 3]
    s = ms.MUS_BS(sampling_strategy="auto" if alpha is None else alpha, random_state=seed)
    s.fit_resample(X, y)
    P, N = X[y == 1], X[y == 0]
    D = ms.exact_distances(P, N)
    sel = np.zeros(len(N), bool)
    sel[np.searchsorted(np.flatnonzero(y == 0), s.pairs_[:, 1])] = True
    closed = np.any(D <= s.radii_[:, None], axis=0)
    opened = np.any(D < s.radii_[:, None], axis=0)
    assert np.all(sel <= closed) and np.all(opened <= sel)
    if seed % 3 != 0:  # general position: equality with the union of closed balls
        assert np.array_equal(sel, closed)
    # exact statement with the tie-broken order: z selected iff (d(x_i,z), z) <=lex (r_i, w_i) for some i
    owner = np.full(len(N), -1)
    owner[np.searchsorted(np.flatnonzero(y == 0), s.pairs_[:, 1])] = np.searchsorted(np.flatnonzero(y == 1), s.pairs_[:, 0])
    lex = np.zeros(len(N), bool)
    for i in range(len(P)):
        mates = np.flatnonzero(owner == i)
        w = mates[np.lexsort((mates, D[i, mates]))][-1]
        lex |= (D[i] < D[i, w]) | ((D[i] == D[i, w]) & (np.arange(len(N)) <= w))
    assert np.array_equal(sel, lex)


def test_ball_characterisation_asymmetric_metric():
    X, y = _data(70, 9, d=3, seed=11)
    s = ms.MUS_BS(preferences="asymmetric", random_state=0)
    s.fit_resample(X, y)
    P, N = X[y == 1], X[y == 0]
    Wp = ms.mahalanobis_whitener(P)
    D = ms.exact_distances(P @ Wp, N @ Wp)
    sel = np.isin(np.flatnonzero(y == 0), s.pairs_[:, 1])
    assert np.array_equal(sel, np.any(D <= s.radii_[:, None] + 1e-12, axis=0))


# ----------------------------------------------------------------------------- P4 (W1 optimality of MUS-BO)
@pytest.mark.parametrize("seed", range(20))
def test_boundary_optimal_is_w1_optimal_bruteforce(seed):
    r = np.random.default_rng(4000 + seed)
    p = int(r.integers(1, 4))
    n = int(r.integers(p, 7))
    m = int(r.integers(1, n + 1))
    P, N = r.random((p, 2)), r.random((n, 2))
    q = ms.capacities_for_target(p, m, seed)
    owner, cols, costs = ms.boundary_optimal_assignment(P, N, q)
    D = ms.exact_distances(P, N)
    rows = np.repeat(np.arange(p), q)
    best = min(sum(D[rows[t], perm[t]] for t in range(m)) for perm in itertools.permutations(range(n), m))
    assert abs(costs.sum() - best) < 1e-9
    w_alpha = q / m
    w1 = {S: _w1_uniform_lp(P, w_alpha, N[list(S)], np.full(m, 1 / m)) for S in itertools.combinations(range(n), m)}
    assert abs(min(w1.values()) - best / m) < 1e-7
    assert abs(w1[tuple(sorted(cols.tolist()))] - min(w1.values())) < 1e-7
    # the stable (greedy) selection is never cheaper than the optimal one
    g, gd = ms.hr_greedy(P, N, q)
    assert np.nansum(gd) >= costs.sum() - 1e-12


# ----------------------------------------------------------------------------- min-weight matching
def _nx_mwm_cost(nx, ea, eb, w):
    G = nx.Graph()
    G.add_weighted_edges_from(zip(ea.tolist(), eb.tolist(), w.tolist()))
    M = nx.min_weight_matching(G)
    return sum(G[u][v]["weight"] for u, v in M), len(M)


@pytest.mark.parametrize("seed", range(30))
def test_min_weight_matching_vs_networkx_and_bruteforce(seed):
    nx = pytest.importorskip("networkx")
    r = np.random.default_rng(5000 + seed)
    s = int(r.integers(2, 42))
    X = r.random((s, int(r.integers(1, 6))))
    if seed % 4 == 0:
        X = np.round(X * 3) / 3  # ties and duplicates
    k = int(r.integers(1, 6))
    pairs, costs, info = ms.min_weight_matching(X, n_candidates=k)
    assert info["optimal"] and info["solver"] in ("lp-cuts", "edmonds")
    assert len(pairs) == s // 2 and len(np.unique(pairs)) == 2 * (s // 2)
    # optimal on the candidate graph (k-NN + stable-matching edges): equals networkx Edmonds on the same graph
    ea, eb, w = ms.candidate_graph(X, k)
    ref, card = _nx_mwm_cost(nx, ea, eb, w)
    assert card == s // 2 and abs(costs.sum() - ref) < 1e-9
    # complete candidate graph: equals the complete-graph optimum (networkx and brute force)
    pc, cc, ic = ms.min_weight_matching(X, n_candidates=s)
    D = ms.exact_distances(X, X)
    iu = np.triu_indices(s, 1)
    refc, _ = _nx_mwm_cost(nx, iu[0], iu[1], D[iu])
    assert abs(cc.sum() - refc) < 1e-9
    if s <= 10:
        assert abs(cc.sum() - _brute_mwpm(D)) < 1e-9
    assert cc.sum() <= costs.sum() + 1e-12
    gp, gc = ms.roommates_greedy(X)
    assert costs.sum() <= gc.sum() + 1e-12  # optimal on a graph containing the stable matching


def test_mus_co_round_cost_equals_networkx():
    nx = pytest.importorskip("networkx")
    X, y = _data(37, 6, d=3, seed=5)
    N = X[y == 0]
    s = ms.MUS_CO(random_state=0, n_candidates=2)
    s.fit_resample(X, y)
    ref, _ = _nx_mwm_cost(nx, *ms.candidate_graph(N, 2))
    assert abs(s.rounds_[0]["cost"] - ref) < 1e-9
    assert all(rd["solver"] in ("lp-cuts", "edmonds") for rd in s.rounds_)
    s2 = ms.MUS_CO(random_state=0, n_candidates=1000)  # complete candidate graph
    s2.fit_resample(X, y)
    D = ms.exact_distances(N, N)
    iu = np.triu_indices(len(N), 1)
    ref2, _ = _nx_mwm_cost(nx, iu[0], iu[1], D[iu])
    assert abs(s2.rounds_[0]["cost"] - ref2) < 1e-9


def test_mus_co_duplicates_deterministic():
    X = np.r_[np.zeros((25, 2)), np.full((20, 2), 0.5), np.ones((6, 2)) * 0.9]
    y = np.r_[np.zeros(45, int), np.ones(6, int)]
    a = ms.MUS_CO(random_state=3).fit_resample(X, y)
    b = ms.MUS_CO(random_state=3).fit_resample(X, y)
    assert np.array_equal(a[0], b[0]) and (a[1] == 0).sum() == 6


def test_min_weight_matching_solver_paths_agree(monkeypatch):
    nx = pytest.importorskip("networkx")
    for seed in range(12):
        r = np.random.default_rng(7000 + seed)
        s = int(r.integers(10, 60))
        X = r.random((s, int(r.integers(1, 4))))
        if seed % 3 == 0:
            X = np.round(X * 4) / 4  # duplicates
        k = int(r.integers(2, 8))
        p1, c1, i1 = ms.min_weight_matching(X, n_candidates=k)                     # default deterministic rule
        p2, c2, i2 = ms.min_weight_matching(X, n_candidates=k, max_lp_solves=1)    # forces Edmonds unless LP integral
        assert i1["solver"] in ("lp-cuts", "edmonds") and i2["solver"] in ("lp-cuts", "edmonds")
        assert abs(c1.sum() - c2.sum()) < 1e-9
        ref, _ = _nx_mwm_cost(nx, *ms.candidate_graph(X, k))
        assert abs(c1.sum() - ref) < 1e-9
        p3, c3, i3 = ms.min_weight_matching(X, n_candidates=k)
        assert np.array_equal(p1, p3)  # deterministic

    class _Boom:
        def __init__(self, *a, **kw):
            raise RuntimeError("forced HiGHS failure")

    monkeypatch.setattr(ms, "_HighsLP", _Boom)
    X = np.random.default_rng(1).random((31, 2))
    p4, c4, i4 = ms.min_weight_matching(X, n_candidates=3)
    assert i4["solver"] == "edmonds" and len(p4) == 15
    ref, _ = _nx_mwm_cost(nx, *ms.candidate_graph(X, 3))
    assert abs(c4.sum() - ref) < 1e-9


# ----------------------------------------------------------------------------- P5 (W1 bound, coverage arm)
@pytest.mark.parametrize("cls,seed", [(c, sd) for c in (ms.MUS_CS, ms.MUS_CO) for sd in range(6)])
def test_coverage_w1_bound(cls, seed):
    X, y = _data(int(25 + 6 * seed), 4 + seed % 3, d=2, seed=seed)
    s = cls(random_state=seed)
    Xr, yr = s.fit_resample(X, y)
    N, S = X[y == 0], Xr[yr == 0]
    w1 = _w1_uniform_lp(N, np.full(len(N), 1 / len(N)), S, np.full(len(S), 1 / len(S)))
    diam = ms.exact_distances(N, N).max()
    bound = 0.0
    for rd in s.rounds_:
        sz, rr = rd["size_in"], rd["n_pairs"]
        bound += rd["cost"] / sz + diam * rr * (sz - 2 * rr) / (sz * (sz - rr))
    assert w1 <= bound + 1e-9


# ----------------------------------------------------------------------------- estimator behaviour
@pytest.mark.parametrize("cls", ALL)
@pytest.mark.parametrize("n_maj,n_min,strategy", [(60, 7, "auto"), (61, 9, 0.5), (45, 1, "auto"), (33, 4, 0.75),
                                                  (20, 20, "auto"), (50, 5, {0: 13}), (50, 5, {0: 50})])
def test_exact_output_sizes(cls, n_maj, n_min, strategy):
    X, y = _data(n_maj, n_min, d=2, seed=n_maj + n_min)
    s = cls(sampling_strategy=strategy, random_state=0)
    Xr, yr = s.fit_resample(X, y)
    if isinstance(strategy, dict):
        m = strategy[0]
    elif strategy == "auto":
        m = n_min
    else:
        m = int(n_min / strategy)
    assert (yr == 0).sum() == min(m, n_maj)
    assert (yr == 1).sum() == n_min
    assert np.array_equal(Xr[yr == 1], X[y == 1])  # minority untouched
    assert np.array_equal(Xr, X[s.sample_indices_])


@pytest.mark.parametrize("cls", ALL)
def test_degenerate_inputs_d1_duplicates(cls):
    X = np.r_[np.full(30, 0.5), np.linspace(0, 1, 30), np.full(6, 0.7)][:, None]
    y = np.r_[np.zeros(60, int), np.ones(6, int)]
    s = cls(random_state=1)
    Xr, yr = s.fit_resample(X, y)
    assert (yr == 0).sum() == 6 and (yr == 1).sum() == 6


@pytest.mark.parametrize("cls", ALL)
def test_determinism(cls):
    X, y = _data(90, 8, d=3, seed=2)
    a = cls(random_state=7, sampling_strategy=0.6).fit_resample(X, y)
    b = cls(random_state=7, sampling_strategy=0.6).fit_resample(X, y)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_keep_rule_boundary_keeps_closer_member():
    X, y = _data(40, 5, d=2, seed=9)
    s = ms.MUS_CS(keep_rule="boundary", sampling_strategy={0: 20})
    Xr, yr = s.fit_resample(X, y)
    P, N = X[y == 1], X[y == 0]
    dmin = ms._nearest_distance(N, P)
    pairs, _ = ms.roommates_greedy(N)
    kept = np.isin(np.flatnonzero(y == 0), s.sample_indices_)
    for a, b in pairs:
        assert kept[a] != kept[b]
        assert kept[a] == ((dmin[a], a) < (dmin[b], b))


def test_errors():
    X, y = _data(30, 5, d=2, seed=0)
    with pytest.raises(ValueError):
        ms.MatchingUnderSampler("coverage", "optimal", preferences="asymmetric").fit_resample(X, y)
    y3 = y.copy()
    y3[:5] = 2
    with pytest.raises(ValueError):
        ms.MUS_BS().fit_resample(X, y3)
    import scipy.sparse as sp
    with pytest.raises((TypeError, ValueError)):
        ms.MUS_BS().fit_resample(sp.csr_matrix(X), y)
    with pytest.raises(ValueError):
        ms.MatchingUnderSampler(arm="middle").fit_resample(X, y)


def test_pipeline_clone_and_pandas():
    pd = pytest.importorskip("pandas")
    from imblearn.pipeline import make_pipeline
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.tree import DecisionTreeClassifier

    X, y = _data(120, 15, d=4, seed=4)
    for est in [ms.MUS_BS(random_state=0), ms.MUS_BO(random_state=0), ms.MUS_CS(random_state=0),
                ms.MUS_CO(random_state=0), ms.MatchingUnderSampler("coverage", "stable", keep_rule="boundary")]:
        pipe = make_pipeline(MinMaxScaler(), est, DecisionTreeClassifier(random_state=0))
        sc = cross_val_score(pipe, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc")
        assert np.all(np.isfinite(sc))
        c = clone(est)
        assert c.get_params() == est.get_params()
    est = ms.MatchingUnderSampler("boundary", "optimal", random_state=0)
    est.set_params(arm="coverage", concept="stable")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    Xr, yr = est.fit_resample(df, pd.Series(y))
    assert isinstance(Xr, pd.DataFrame) and (yr == 0).sum() == 15
