"""Matching-theory undersamplers for imbalanced binary classification.

The module implements a 2 x 2 family of majority-class undersamplers
(``arm`` x ``concept``) as one imbalanced-learn compatible estimator,
:class:`MatchingUnderSampler`, plus four thin aliases.

=========  ==========  ===========  ==================================================
alias      arm         concept      matching problem solved
=========  ==========  ===========  ==================================================
MUS_BS     boundary    stable       hospitals/residents (minority = hospitals with
                                    capacities, majority = residents), deferred
                                    acceptance; fast exact greedy for symmetric
                                    (distance-induced) preferences
MUS_BO     boundary    optimal      rectangular min-cost assignment between the
                                    capacity-replicated minority and the majority
MUS_CS     coverage    stable       iterated stable roommates on the majority class
                                    (keep one instance per pair)
MUS_CO     coverage    optimal      iterated minimum-weight (maximum-cardinality)
                                    matching on the majority class
=========  ==========  ===========  ==================================================

Notation: minority set P (p points), majority set N (n points), target majority
size m (``sampling_strategy_`` of imbalanced-learn; ``'auto'`` gives m = p and a
float alpha gives m = int(p / alpha), exactly as for ``RandomUnderSampler``).
Distances are Euclidean on the features as given (the experimental protocol
MinMax-scales inside the training folds before resampling).

Tie-breaking.  All exact algorithms use one global strict order on pairs:
boundary arm: (d(x_i, z_j), i, j); coverage arm: (d(z_a, z_b), min(a, b),
max(a, b)), where i, j, a, b are positions in the (sorted) index arrays of the
classes.  Every distance used for an ordering decision is computed by the same
function (:func:`_pd`), which is bitwise symmetric and shape independent, so the
order is consistent across all code paths.

The optimal coverage variant solves every round exactly on a candidate graph:
the symmetric k-nearest-neighbour graph (k = ``n_candidates``, default 10)
plus the edges of the stable (greedy) roommates matching, which guarantees a
perfect matching up to parity (a dummy vertex at cost 0 absorbs the unmatched
point when the number of points is odd).  Deterministic exact solver rule: the
perfect-matching LP with violated odd-set (blossom) inequalities is solved by
the HiGHS dual simplex (hot starts) for at most MAX_LP_SOLVES = 50 solves; an
integral LP optimum is optimal; otherwise the exact Edmonds blossom algorithm
(networkx) is run on the same candidate graph.  No step depends on wall-clock
time.  With ``n_candidates >= s - 1`` the candidate graph is complete.
"""
from __future__ import annotations

import heapq
import numbers
import time
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils import _safe_indexing

from imblearn.under_sampling.base import BaseUnderSampler

__all__ = [
    "MatchingUnderSampler",
    "MUS_BS",
    "MUS_BO",
    "MUS_CS",
    "MUS_CO",
    "capacities_for_target",
    "exact_distances",
    "hr_greedy",
    "hr_deferred_acceptance",
    "hr_blocking_pairs",
    "preference_orders",
    "roommates_greedy",
    "roommates_blocking_pairs",
    "candidate_graph",
    "min_weight_matching",
    "boundary_optimal_assignment",
    "mahalanobis_whitener",
]

DEFAULT_N_CANDIDATES = 10  # k of the MUS-CO candidate graph; k sweep (k in {5, 10, 20}: runtime and gap to the
#                             certified complete-graph optimum, 15 point sets) is saved as mus_co_k_sweep.csv
GREEDY_PREFIX = 32  # exact candidate-prefix length of the greedy algorithms (speed only)


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def _pd(A, B):
    """Paired / broadcast Euclidean distance, sqrt(sum((A - B)**2, axis=-1)).

    This is the single distance routine used for every ordering decision; it
    is bitwise symmetric (d(a, b) == d(b, a)) and independent of the batch
    shape, so all code paths induce the same global order on pairs.
    """
    diff = A - B
    return np.sqrt(np.sum(diff * diff, axis=-1))


def exact_distances(A, B, block_elems=2 ** 22):
    """Dense matrix of exact distances d(A[i], B[j]) computed with :func:`_pd`."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    out = np.empty((len(A), len(B)))
    d = max(1, A.shape[1])
    bs = max(1, block_elems // max(1, len(B) * d))
    for st in range(0, len(A), bs):
        en = min(len(A), st + bs)
        out[st:en] = _pd(A[st:en, None, :], B[None, :, :])
    return out


def _nearest_distance(A, B, block_elems=2 ** 22):
    """min_j d(A[i], B[j]) for every row of A (scipy cdist, blockwise)."""
    out = np.empty(len(A))
    bs = max(1, block_elems // max(1, len(B)))
    for st in range(0, len(A), bs):
        en = min(len(A), st + bs)
        out[st:en] = cdist(A[st:en], B).min(axis=1)
    return out


def _prefix_lists(A, B, K, exclude_self=False, extra=8, block_elems=2 ** 22):
    """Exact sorted candidate prefixes.

    For every row a of A returns the indices (and distances) of all points b of
    B with d(a, b) <= d_K(a), the K-th smallest distance, sorted by the strict
    order (d(a, b), b).  This is an exact prefix of the complete preference list
    of a.  Squared distances obtained from a matrix product are used only for
    screening; every returned distance is recomputed with :func:`_pd`, and rows
    whose screening is inconclusive (near ties) are recomputed exactly.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    na, nb = len(A), len(B)
    avail = nb - (1 if exclude_self else 0)
    out_idx = [None] * na
    out_d = [None] * na
    if na == 0 or avail <= 0:
        for r in range(na):
            out_idx[r] = np.empty(0, dtype=np.int64)
            out_d[r] = np.empty(0)
        return out_idx, out_d
    K = int(min(max(K, 1), avail))
    Kc = int(min(K + extra, avail))
    sqA = np.sum(A * A, axis=1)
    sqB = np.sum(B * B, axis=1)
    tol = 1e-9 * (1.0 + float(sqA.max()) + float(sqB.max()))
    bs = max(1, block_elems // nb)
    for st in range(0, na, bs):
        en = min(na, st + bs)
        b = en - st
        S = A[st:en] @ B.T
        S *= -2.0
        S += sqA[st:en, None]
        S += sqB[None, :]
        if exclude_self:
            S[np.arange(b), np.arange(st, en)] = np.inf
        if Kc < avail:
            P = np.argpartition(S, Kc, axis=1)
            nxt = S[np.arange(b), P[:, Kc]]
            P = P[:, :Kc]
        else:
            P = np.argsort(S, axis=1, kind="stable")[:, :avail]
            nxt = np.full(b, np.inf)
        E = _pd(A[st:en, None, :], B[P])
        o = np.lexsort((P, E), axis=1)
        P = np.take_along_axis(P, o, axis=1)
        E = np.take_along_axis(E, o, axis=1)
        dK = E[:, K - 1]
        safe = nxt > dK * dK + tol
        for r in range(b):
            if safe[r]:
                cnt = int(np.searchsorted(E[r], dK[r], side="right"))
                out_idx[st + r] = P[r, :cnt].astype(np.int64)
                out_d[st + r] = E[r, :cnt].copy()
            else:
                e = _pd(A[st + r][None, :], B)
                if exclude_self:
                    e[st + r] = np.inf
                kth = np.partition(e, K - 1)[K - 1]
                idx = np.flatnonzero(e <= kth)
                o2 = np.lexsort((idx, e[idx]))
                out_idx[st + r] = idx[o2].astype(np.int64)
                out_d[st + r] = e[idx][o2]
    return out_idx, out_d


# --------------------------------------------------------------------------- #
# capacities
# --------------------------------------------------------------------------- #
def capacities_for_target(p, m, random_state=None):
    """Capacities q_i in {floor(m/p), ceil(m/p)} with sum(q) = m.

    The m mod p instances receiving the larger capacity are chosen by a seeded
    random permutation (no randomness is consumed when p divides m).
    """
    if p <= 0:
        raise ValueError("p must be positive")
    base, rem = divmod(int(m), int(p))
    q = np.full(p, base, dtype=np.int64)
    if rem:
        rng = check_random_state(random_state)
        q[rng.permutation(p)[:rem]] += 1
    return q


# --------------------------------------------------------------------------- #
# hospitals / residents  (boundary arm, stable concept)
# --------------------------------------------------------------------------- #
def hr_greedy(P, N, q, K=None):
    """Unique stable matching of the distance-induced HR instance (exact greedy).

    Minority points P are hospitals with capacities q, majority points N are
    residents; both sides rank by Euclidean distance and ties are broken by the
    global pair order (d(x_i, z_j), i, j).  Pairs are processed in increasing
    global order and accepted when the hospital has a free seat and the
    resident is unmatched (lazy, heap-based implementation with exact candidate
    prefixes and exact refills).

    Returns
    -------
    assign : ndarray of shape (n,), minority index of each majority point or -1
    assign_d : ndarray of shape (n,), distance of the assigned pair (nan if none)
    """
    P = np.asarray(P, dtype=float)
    N = np.asarray(N, dtype=float)
    q = np.asarray(q, dtype=np.int64)
    p, n = len(P), len(N)
    assign = np.full(n, -1, dtype=np.int64)
    assign_d = np.full(n, np.nan)
    if p == 0 or n == 0 or q.sum() == 0:
        return assign, assign_d
    if q.sum() > n:
        raise ValueError("sum of capacities exceeds the number of residents")
    if K is None:
        K = int(q.max()) + 16
    lists, dists = _prefix_lists(P, N, K)
    ptr = np.zeros(p, dtype=np.int64)
    full = np.zeros(p, dtype=bool)
    rem = q.copy()
    heap = [(float(dists[i][0]), i, int(lists[i][0])) for i in range(p) if rem[i] > 0]
    heapq.heapify(heap)

    def push_next(i, start):
        L = lists[i]
        pp = start
        while pp < len(L) and assign[L[pp]] != -1:
            pp += 1
        if pp >= len(L):
            if full[i]:
                return
            free = np.flatnonzero(assign == -1)
            if len(free) == 0:
                return
            e = _pd(P[i][None, :], N[free])
            o = np.lexsort((free, e))
            lists[i] = free[o]
            dists[i] = e[o]
            full[i] = True
            pp = 0
            L = lists[i]
        ptr[i] = pp
        heapq.heappush(heap, (float(dists[i][pp]), i, int(L[pp])))

    while heap:
        d, i, j = heapq.heappop(heap)
        if rem[i] == 0:
            continue
        if assign[j] == -1:
            assign[j] = i
            assign_d[j] = d
            rem[i] -= 1
            if rem[i] > 0:
                push_next(i, ptr[i] + 1)
        else:
            push_next(i, ptr[i] + 1)
    return assign, assign_d


def preference_orders(D_min, D_maj=None):
    """Complete strict preference orders from distance matrices.

    Parameters
    ----------
    D_min : (p, n) distances used by the minority side (row i ranks majority).
    D_maj : (n, p) distances used by the majority side; defaults to D_min.T.

    Returns ``min_order`` (p, n) and ``maj_order`` (n, p); ties are broken by
    the index of the ranked agent (stable sort), i.e. by the global pair order
    when both matrices come from one symmetric distance.
    """
    D_min = np.asarray(D_min, dtype=float)
    D_maj = D_min.T if D_maj is None else np.asarray(D_maj, dtype=float)
    min_order = np.argsort(D_min, axis=1, kind="stable")
    maj_order = np.argsort(D_maj, axis=1, kind="stable")
    return min_order, maj_order


def _ranks_from_orders(order):
    rank = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    rank[rows, order] = np.arange(order.shape[1])[None, :]
    return rank


def hr_deferred_acceptance(min_order, maj_order, q, proposer="minority"):
    """Deferred acceptance for hospitals/residents with complete strict lists.

    Parameters
    ----------
    min_order : (p, n) int array, row i = majority indices in decreasing
        preference of minority (hospital) i.
    maj_order : (n, p) int array, row j = minority indices in decreasing
        preference of majority (resident) j.
    q : (p,) capacities.
    proposer : {'minority', 'majority'}; 'minority' gives the hospital-optimal
        and 'majority' the resident-optimal stable matching.

    Returns ``assign`` (n,): minority index of each majority instance or -1.
    """
    min_order = np.ascontiguousarray(min_order, dtype=np.int64)
    maj_order = np.ascontiguousarray(maj_order, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    p, n = min_order.shape
    assign = [-1] * n  # python containers + memoryviews: much faster scalar access than numpy
    if proposer == "minority":
        mrank = memoryview(np.ascontiguousarray(_ranks_from_orders(maj_order)))
        mo = memoryview(min_order)
        ptr = [0] * p
        rem = q.tolist()
        stack = [i for i in range(p) if rem[i] > 0]
        while stack:
            i = stack.pop()
            while rem[i] > 0 and ptr[i] < n:
                j = mo[i, ptr[i]]
                ptr[i] += 1
                cur = assign[j]
                if cur == -1:
                    assign[j] = i
                    rem[i] -= 1
                elif mrank[j, i] < mrank[j, cur]:
                    assign[j] = i
                    rem[i] -= 1
                    rem[cur] += 1
                    stack.append(cur)
        return np.asarray(assign, dtype=np.int64)
    if proposer == "majority":
        nrank = memoryview(np.ascontiguousarray(_ranks_from_orders(min_order)))
        mj = memoryview(maj_order)
        qq = q.tolist()
        held = [[] for _ in range(p)]  # max-heap on rank via (-rank, j)
        ptr = [0] * n
        free = list(range(n - 1, -1, -1))
        while free:
            j = free.pop()
            while ptr[j] < p:
                i = mj[j, ptr[j]]
                ptr[j] += 1
                if qq[i] == 0:
                    continue
                r = nrank[i, j]
                if len(held[i]) < qq[i]:
                    heapq.heappush(held[i], (-r, j))
                    assign[j] = i
                    break
                worst_negr, worst_j = held[i][0]
                if r < -worst_negr:
                    heapq.heapreplace(held[i], (-r, j))
                    assign[j] = i
                    assign[worst_j] = -1
                    free.append(worst_j)
                    break
        return np.asarray(assign, dtype=np.int64)
    raise ValueError("proposer must be 'minority' or 'majority'")


def hr_blocking_pairs(assign, min_order, maj_order, q):
    """All blocking pairs (i, j) of an HR matching with complete strict lists."""
    min_order = np.asarray(min_order)
    maj_order = np.asarray(maj_order)
    assign = np.asarray(assign)
    q = np.asarray(q)
    p, n = min_order.shape
    min_rank = _ranks_from_orders(min_order)
    maj_rank = _ranks_from_orders(maj_order)
    worst = np.full(p, -1, dtype=np.int64)  # i accepts j iff min_rank[i, j] < worst[i]
    for i in range(p):
        mates = np.flatnonzero(assign == i)
        if len(mates) > q[i]:
            raise ValueError("capacity violated")
        if len(mates) < q[i]:
            worst[i] = n  # under-subscribed: accepts anybody
        elif len(mates):
            worst[i] = min_rank[i, mates].max()
    cur_rank = np.where(assign >= 0, maj_rank[np.arange(n), np.maximum(assign, 0)], p)
    B = (min_rank < worst[:, None]) & (maj_rank.T < cur_rank[None, :])
    B &= assign[None, :] != np.arange(p)[:, None]
    return np.argwhere(B)


def mahalanobis_whitener(Z):
    """Whitening matrix W with |W^T (a - b)| = Mahalanobis distance under the
    Ledoit-Wolf shrunk covariance of Z (pseudo-inverse on its range).

    Falls back to the identity (Euclidean) when the covariance is numerically
    zero (e.g. a single instance or constant features).
    """
    from sklearn.covariance import LedoitWolf

    Z = np.asarray(Z, dtype=float)
    d = Z.shape[1]
    if len(Z) < 2:
        return np.eye(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        C = LedoitWolf().fit(Z).covariance_
    evals, evecs = np.linalg.eigh((C + C.T) / 2.0)
    top = float(evals.max()) if evals.size else 0.0
    if not np.isfinite(top) or top <= 0:
        return np.eye(d)
    keep = evals > top * max(d, len(Z)) * np.finfo(float).eps
    return evecs[:, keep] / np.sqrt(evals[keep])


# --------------------------------------------------------------------------- #
# stable roommates  (coverage arm, stable concept)
# --------------------------------------------------------------------------- #
def roommates_greedy(X, max_pairs=None, K=GREEDY_PREFIX, lists=None):
    """Unique stable matching of the distance-induced roommates instance.

    Agents rank each other by Euclidean distance with ties broken by the global
    pair order (d(a, b), min(a, b), max(a, b)); the unique stable matching is
    the greedy matching that repeatedly pairs the globally closest free pair
    (equivalently: repeated mutual-nearest-neighbour pairing).  With
    ``max_pairs = r`` the procedure stops after the r first (closest) pairs.

    Returns ``pairs`` (r, 2) in acceptance (= increasing key) order with
    ``pairs[:, 0] < pairs[:, 1]`` and the pair distances ``costs``.
    """
    X = np.asarray(X, dtype=float)
    s = len(X)
    if s < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0)
    if lists is None:
        lists = _prefix_lists(X, X, K, exclude_self=True)
    bufs, bufd = list(lists[0]), list(lists[1])
    limit = s // 2 if max_pairs is None else int(min(max_pairs, s // 2))
    ptr = np.zeros(s, dtype=np.int64)
    free = np.ones(s, dtype=bool)
    full = np.zeros(s, dtype=bool)
    heap = []
    for a in range(s):
        b = int(bufs[a][0])
        heap.append((float(bufd[a][0]), min(a, b), max(a, b), a))
    heapq.heapify(heap)
    pairs, costs = [], []

    def push_next(a, start):
        B = bufs[a]
        pp = start
        while pp < len(B) and not free[B[pp]]:
            pp += 1
        if pp >= len(B):
            if full[a]:
                return
            fr = np.flatnonzero(free)
            fr = fr[fr != a]
            if len(fr) == 0:
                return
            e = _pd(X[a][None, :], X[fr])
            o = np.lexsort((fr, e))
            bufs[a] = fr[o]
            bufd[a] = e[o]
            full[a] = True
            pp = 0
            B = bufs[a]
        ptr[a] = pp
        b = int(B[pp])
        heapq.heappush(heap, (float(bufd[a][pp]), min(a, b), max(a, b), a))

    while heap and len(pairs) < limit:
        d, lo, hi, a = heapq.heappop(heap)
        if not free[a]:
            continue
        b = hi if lo == a else lo
        if free[b]:
            free[a] = False
            free[b] = False
            pairs.append((lo, hi))
            costs.append(d)
        else:
            push_next(a, ptr[a] + 1)
    return np.array(pairs, dtype=np.int64).reshape(-1, 2), np.array(costs, dtype=float)


def roommates_blocking_pairs(pairs, D):
    """Blocking pairs of a roommates matching under distance preferences with
    the global tie-break (d, min, max) (agent a ranks b by (D[a, b], b))."""
    D = np.asarray(D, dtype=float)
    s = len(D)
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    order = np.argsort(Dm, axis=1, kind="stable")
    rank = _ranks_from_orders(order)
    partner = np.full(s, -1)
    for a, b in np.asarray(pairs).reshape(-1, 2):
        partner[a] = b
        partner[b] = a
    cur = np.where(partner >= 0, rank[np.arange(s), np.maximum(partner, 0)], s)
    B = (rank < cur[:, None]) & (rank.T < cur[None, :])
    np.fill_diagonal(B, False)
    B &= partner[:, None] != np.arange(s)[None, :]
    iu = np.argwhere(np.triu(B, 1))
    return iu


# --------------------------------------------------------------------------- #
# exact minimum-weight perfect matching  (coverage arm, optimal concept)
# --------------------------------------------------------------------------- #
def candidate_graph(X, n_candidates=DEFAULT_N_CANDIDATES, lists=None, init_pairs=None):
    """Edge set of the MUS-CO candidate graph on the rows of X.

    Union of the symmetric k-nearest-neighbour graph (k = ``n_candidates``;
    neighbours taken from exact prefixes sorted by (distance, index), so ties
    and duplicates are handled deterministically) and the edges of the stable
    (greedy) roommates matching, which guarantees that the candidate graph
    contains a matching leaving at most one vertex unmatched.

    Returns ``ea < eb`` (int arrays) and edge lengths ``w``.
    """
    X = np.asarray(X, dtype=float)
    s = len(X)
    k = int(min(max(1, n_candidates), s - 1))
    if lists is None:
        lists = _prefix_lists(X, X, max(k, GREEDY_PREFIX), exclude_self=True)
    if init_pairs is None:
        init_pairs, _ = roommates_greedy(X, lists=lists)
    a = np.concatenate([np.repeat(np.arange(s), [min(len(L), k) for L in lists[0]]), init_pairs[:, 0]])
    b = np.concatenate([np.concatenate([L[:k] for L in lists[0]]), init_pairs[:, 1]])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = np.unique(lo.astype(np.int64) * s + hi)
    ea, eb = key // s, key % s
    return ea, eb, _pd(X[ea], X[eb])


MAX_LP_SOLVES = 50  # deterministic cap of the cutting-plane phase of MUS-CO (then exact Edmonds fallback)


class _HighsLP:
    """Incremental LP  min w.x  s.t.  x(delta(v)) = 1,  x(E(U)) <= (|U|-1)/2 (odd sets U),  x >= 0,
    solved by the HiGHS dual simplex (serial, hot-started after adding cuts)."""

    def __init__(self, nv, ea, eb, w):
        from scipy.optimize._highspy import _core as hc

        self.hc = hc
        self.nv = nv
        self.ea = np.asarray(ea, dtype=np.int64)
        self.eb = np.asarray(eb, dtype=np.int64)
        self.w = np.asarray(w, dtype=float)
        E = len(self.ea)
        A = sp.csc_matrix((np.ones(2 * E), (np.concatenate([self.ea, self.eb]), np.concatenate([np.arange(E)] * 2))),
                          shape=(nv, E))
        A.sort_indices()
        h = hc._Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("parallel", "off")
        h.setOptionValue("primal_feasibility_tolerance", 1e-9)
        h.setOptionValue("dual_feasibility_tolerance", 1e-9)
        lp = hc.HighsLp()
        lp.num_col_ = E
        lp.num_row_ = nv
        lp.col_cost_ = self.w
        lp.col_lower_ = np.zeros(E)
        lp.col_upper_ = np.full(E, hc.kHighsInf)
        lp.row_lower_ = np.ones(nv)
        lp.row_upper_ = np.ones(nv)
        lp.a_matrix_.format_ = hc.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data
        lp.a_matrix_.num_col_ = E
        lp.a_matrix_.num_row_ = nv
        h.passModel(lp)
        self.h = h

    def solve(self):
        self.h.run()
        if self.h.getModelStatus() != self.hc.HighsModelStatus.kOptimal:
            raise RuntimeError("LP not optimal")
        return np.asarray(self.h.getSolution().col_value, dtype=float)

    def add_cuts(self, Us):
        starts, idx, ub = [], [], []
        for U in Us:
            inU = np.zeros(self.nv, dtype=bool)
            inU[U] = True
            es = np.flatnonzero(inU[self.ea] & inU[self.eb])
            starts.append(len(idx))
            idx.extend(es.tolist())
            ub.append((len(U) - 1) / 2.0)
        self.h.addRows(len(Us), np.full(len(Us), -self.hc.kHighsInf), np.asarray(ub, dtype=float), len(idx),
                       np.asarray(starts, dtype=np.int32), np.asarray(idx, dtype=np.int32), np.ones(len(idx)))


def _odd_set_cuts(x, ea, eb, nv, eps=1e-7, scale=2 ** 28):
    """Violated odd-set (blossom) inequalities x(E(U)) <= (|U|-1)/2 at a fractional LP point.

    Odd connected components of the support are violated.  Inside even
    components with fractional edges, Gusfield's all-pairs minimum-cut
    procedure (n-1 maximum flows; scipy, capacities scaled to integers) is run
    and every computed minimum cut with an odd shore and value < 1 is returned
    (Padberg-Rao).  Deterministic.
    """
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    supp = x > eps
    frac = supp & (x < 1 - eps)
    if not frac.any():
        return []
    sa, sb, sx = ea[supp], eb[supp], x[supp]
    G = sp.csr_matrix((np.ones(len(sa)), (sa, sb)), shape=(nv, nv))
    _, lab = connected_components(G, directed=False)
    fl = np.unique(lab[np.concatenate([ea[frac], eb[frac]])])
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(lab.max() + 2))
    cuts = []
    for c in fl:
        nodes = np.sort(order[bounds[c]:bounds[c + 1]])
        nc = len(nodes)
        if nc % 2 == 1:
            cuts.append(nodes)
            continue
        loc = np.full(nv, -1, dtype=np.int64)
        loc[nodes] = np.arange(nc)
        m = loc[sa] >= 0
        a, b = loc[sa[m]], loc[sb[m]]
        cap = np.maximum(1, np.round(sx[m] * scale)).astype(np.int64)
        C = sp.csr_matrix((np.concatenate([cap, cap]), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(nc, nc))
        C.sum_duplicates()
        parent = np.zeros(nc, dtype=np.int64)
        for s_ in range(1, nc):
            t_ = int(parent[s_])
            res = maximum_flow(C, s_, t_)
            R = (C - res.flow).tocsr()
            R.data = (R.data > 0).astype(np.int8)
            R.eliminate_zeros()
            Xs = breadth_first_order(R, s_, directed=True, return_predecessors=False)
            inX = np.zeros(nc, dtype=bool)
            inX[Xs] = True
            if res.flow_value < (1.0 - 1e-6) * scale and len(Xs) % 2 == 1:
                side = Xs if len(Xs) <= nc - len(Xs) else np.flatnonzero(~inX)
                cuts.append(np.sort(nodes[side]))
            upd = inX & (parent == t_)
            upd[: s_ + 1] = False
            parent[upd] = s_
    uniq = {tuple(U.tolist()): U for U in cuts}
    return [uniq[kk] for kk in sorted(uniq)]


def min_weight_matching(X, n_candidates=DEFAULT_N_CANDIDATES, edges=None, max_lp_solves=MAX_LP_SOLVES):
    """Minimum-weight maximum-cardinality matching on the MUS-CO candidate graph (exact, deterministic).

    Solver rule (no wall-clock dependence): the perfect-matching LP on the
    candidate edges (plus a dummy vertex joined to every point at cost 0 when
    len(X) is odd) is solved with the HiGHS dual simplex; violated odd-set
    inequalities are added and the LP is re-solved (hot start).  An integral LP
    optimum is an optimal matching of the candidate graph.  If the LP is still
    fractional after ``max_lp_solves`` solves, if no violated odd set is found,
    or if HiGHS is unavailable or fails, the exact Edmonds blossom algorithm
    (networkx ``min_weight_matching``) is run on the same candidate graph.
    Every step is a deterministic function of the input.

    Returns ``pairs`` (floor(s/2), 2) with pairs[:, 0] < pairs[:, 1], their
    lengths ``costs`` and ``info`` (``solver`` in {'lp-cuts', 'edmonds'},
    ``lp_solves``, ``cuts``, ``n_edges``, ``optimal`` = True).
    """
    X = np.asarray(X, dtype=float)
    s = len(X)
    info = dict(optimal=True, solver="none", lp_solves=0, cuts=0, n_edges=0, seconds=0.0)
    if s < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0), info
    t0 = time.perf_counter()
    ea, eb, w = candidate_graph(X, n_candidates) if edges is None else edges
    odd = s % 2
    nv = s + odd
    if odd:
        ea = np.concatenate([ea, np.arange(s)])
        eb = np.concatenate([eb, np.full(s, s)])
        w = np.concatenate([w, np.zeros(s)])
    sel = None
    try:
        lp = _HighsLP(nv, ea, eb, w)
        while info["lp_solves"] < max_lp_solves:
            x = lp.solve()
            info["lp_solves"] += 1
            if not np.any((x > 1e-6) & (x < 1 - 1e-6)):
                cand = np.flatnonzero(x > 0.5)
                if np.all(np.bincount(np.concatenate([ea[cand], eb[cand]]), minlength=nv) == 1):
                    sel = cand
                    info["solver"] = "lp-cuts"
                break
            if info["lp_solves"] >= max_lp_solves:
                break
            new = _odd_set_cuts(x, ea, eb, nv)
            if not new:
                break
            lp.add_cuts(new)
            info["cuts"] += len(new)
    except Exception:  # HiGHS unavailable or failed -> exact fallback below
        sel = None
    if sel is None:
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(s))
        real_e = eb < s
        G.add_weighted_edges_from(zip(ea[real_e].tolist(), eb[real_e].tolist(), w[real_e].tolist()))
        M = nx.min_weight_matching(G)
        mp = np.array(sorted(tuple(sorted(e)) for e in M), dtype=np.int64).reshape(-1, 2)
        if odd:
            left = np.setdiff1d(np.arange(s), mp.ravel())
            mp = np.vstack([mp, np.stack([left, np.full(len(left), s)], axis=1)])
        key = {(int(u), int(v)): t for t, (u, v) in enumerate(zip(ea.tolist(), eb.tolist()))}
        sel = np.array([key[(int(u), int(v))] for u, v in mp], dtype=np.int64)
        info["solver"] = "edmonds"
    pa, pb = ea[sel], eb[sel]
    deg = np.bincount(np.concatenate([pa, pb]), minlength=nv)
    if not np.all(deg == 1):
        raise RuntimeError("matching solver returned a non-perfect matching")
    real = pb < s
    pairs = np.sort(np.stack([pa[real], pb[real]], axis=1), axis=1)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    costs = _pd(X[pairs[:, 0]], X[pairs[:, 1]])
    info.update(n_edges=int(len(ea) - (s if odd else 0)), seconds=time.perf_counter() - t0)
    return pairs, costs, info


# --------------------------------------------------------------------------- #
# rectangular assignment  (boundary arm, optimal concept)
# --------------------------------------------------------------------------- #
def boundary_optimal_assignment(P, N, q, max_cells=2.5e8):
    """Min-cost assignment of the capacity-replicated minority to the majority.

    Minority instance i is replicated q_i times (m = sum(q) rows) and every row
    is assigned to a distinct majority instance minimising the total Euclidean
    distance (scipy.optimize.linear_sum_assignment, Jonker-Volgenant).

    Returns (rows -> minority index, assigned majority index, pair distances).
    """
    P = np.asarray(P, dtype=float)
    N = np.asarray(N, dtype=float)
    q = np.asarray(q, dtype=np.int64)
    m = int(q.sum())
    if m == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0)
    if m * len(N) > max_cells:
        raise MemoryError("cost matrix of %d x %d exceeds max_cells" % (m, len(N)))
    D = cdist(P, N)  # ties are irrelevant for an optimisation objective; cdist is ~10x faster than _pd here
    owner = np.repeat(np.arange(len(P)), q)
    C = D[owner]
    r, c = linear_sum_assignment(C)
    return owner[r], c.astype(np.int64), C[r, c]


# --------------------------------------------------------------------------- #
# the estimator
# --------------------------------------------------------------------------- #
class MatchingUnderSampler(BaseUnderSampler):
    """Matching-theory undersampling of the majority class (binary problems).

    Parameters
    ----------
    arm : {'boundary', 'coverage'}
        'boundary': minority-majority bipartite matching, the matched majority
        instances are kept.  'coverage': within-majority matching, one instance
        per matched pair is dropped, repeated until the target size is reached.
    concept : {'stable', 'optimal'}
        Stability (deferred acceptance / stable roommates) or minimum total
        distance (assignment / minimum-weight perfect matching).
    sampling_strategy : float, str, dict or callable, default='auto'
        imbalanced-learn semantics; must not alter the minority class.
    keep_rule : {'random', 'boundary'}, default='random'
        Coverage arm only: which member of a matched pair is kept ('random' =
        seeded fair coin; 'boundary' = the member closer to the minority class,
        ties to the smaller index).
    preferences : {'symmetric', 'asymmetric'}, default='symmetric'
        Boundary-stable only.  'symmetric': both sides rank by Euclidean
        distance (unique stable matching, exact greedy).  'asymmetric':
        class-conditional Mahalanobis distances (Ledoit-Wolf), minority-proposing
        deferred acceptance.
    n_candidates : int or None, default=None
        Coverage-optimal only: k of the symmetric k-NN candidate graph on which
        every round is solved exactly (None -> DEFAULT_N_CANDIDATES = 10).  The
        matching is optimal on the candidate graph; with n_candidates >= s - 1
        the graph is complete.  Larger k moves the result towards the
        complete-graph optimum at a higher cost (see mus_co_k_sweep.csv).
    random_state : int, RandomState instance or None, default=None
        Seeds the capacity remainder (boundary arm) and the keep coin (coverage
        arm, keep_rule='random').

    Attributes
    ----------
    sample_indices_ : ndarray, indices of the retained samples (sorted).
    minority_class_, majority_class_ : labels.
    n_majority_in_, n_majority_out_ : majority sizes before / after (= m).
    capacities_ : (p,) capacities (boundary arm).
    pairs_ : (m, 2) original indices [minority, majority] (boundary arm).
    radii_ : (p,) distance from each minority instance to its farthest assigned
        majority instance in the minority-side metric (boundary arm).
    matching_cost_ : total Euclidean length of all matched pairs (all rounds).
    rounds_ : list of dicts with per-round diagnostics (coverage arm).
    n_rounds_ : number of matching rounds.
    runtime_ : wall-clock seconds of the last ``fit_resample``.
    """

    _parameter_constraints: dict = {
        **BaseUnderSampler._parameter_constraints,
        "arm": [StrOptions({"boundary", "coverage"})],
        "concept": [StrOptions({"stable", "optimal"})],
        "keep_rule": [StrOptions({"random", "boundary"})],
        "preferences": [StrOptions({"symmetric", "asymmetric"})],
        "n_candidates": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    def __init__(self, arm="boundary", concept="stable", *, sampling_strategy="auto", keep_rule="random",
                 preferences="symmetric", n_candidates=None, random_state=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.arm = arm
        self.concept = concept
        self.keep_rule = keep_rule
        self.preferences = preferences
        self.n_candidates = n_candidates
        self.random_state = random_state

    # configuration hook (overridden by the aliases)
    def _config(self):
        return dict(arm=self.arm, concept=self.concept, keep_rule=self.keep_rule,
                    preferences=self.preferences, n_candidates=self.n_candidates)

    def _check_X_y(self, X, y, accept_sparse=None):
        return super()._check_X_y(X, y, accept_sparse=False)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        try:
            tags.input_tags.sparse = False
        except Exception:  # pragma: no cover
            pass
        return tags

    def _fit_resample(self, X, y):
        t0 = time.perf_counter()
        cfg = self._config()
        if cfg["preferences"] == "asymmetric" and not (cfg["arm"] == "boundary" and cfg["concept"] == "stable"):
            raise ValueError("preferences='asymmetric' is only defined for arm='boundary', concept='stable'")
        y_arr = np.asarray(y)
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) != 2:
            raise ValueError("MatchingUnderSampler supports binary problems only; got %d classes" % len(classes))
        imin = int(np.argmin(counts))
        minority, majority = classes[imin], classes[1 - imin]
        p, n = int(counts[imin]), int(counts[1 - imin])
        targets = self.sampling_strategy_
        if minority in targets and int(targets[minority]) != p:
            raise ValueError("the minority class is never altered by MatchingUnderSampler")
        m = int(targets.get(majority, n))
        idx_min = np.flatnonzero(y_arr == minority)
        idx_maj = np.flatnonzero(y_arr == majority)
        Xf = np.asarray(X, dtype=float)
        P, N = Xf[idx_min], Xf[idx_maj]
        rng = check_random_state(self.random_state)

        self.minority_class_, self.majority_class_ = minority, majority
        self.n_majority_in_ = n
        self.capacities_ = None
        self.pairs_ = None
        self.radii_ = None
        self.rounds_ = []
        self.n_rounds_ = 0
        self.matching_cost_ = 0.0
        if m >= n:
            keep = np.arange(n)
        elif m <= 0:
            keep = np.empty(0, dtype=np.int64)
        elif cfg["arm"] == "boundary":
            keep = self._boundary(P, N, m, cfg, rng, idx_min, idx_maj)
        else:
            keep = self._coverage(P, N, m, cfg, rng)
        keep = np.sort(np.asarray(keep, dtype=np.int64))
        if len(keep) != min(max(m, 0), n):
            raise RuntimeError("internal error: wrong output size")
        self.n_majority_out_ = len(keep)
        self.sample_indices_ = np.sort(np.concatenate([idx_min, idx_maj[keep]]))
        self.runtime_ = time.perf_counter() - t0
        return _safe_indexing(X, self.sample_indices_), _safe_indexing(y, self.sample_indices_)

    # ------------------------------------------------------------------ boundary arm
    def _boundary(self, P, N, m, cfg, rng, idx_min, idx_maj):
        p = len(P)
        q = capacities_for_target(p, m, rng)
        self.capacities_ = q
        if cfg["concept"] == "stable":
            if cfg["preferences"] == "symmetric":
                assign, adist = hr_greedy(P, N, q)
                metric_d = adist
            else:
                Wp = mahalanobis_whitener(P)
                Wn = mahalanobis_whitener(N)
                D_min = exact_distances(P @ Wp, N @ Wp)
                D_maj = exact_distances(N @ Wn, P @ Wn)
                min_order, maj_order = preference_orders(D_min, D_maj)
                assign = hr_deferred_acceptance(min_order, maj_order, q, proposer="minority")
                metric_d = np.where(assign >= 0, D_min[np.maximum(assign, 0), np.arange(len(N))], np.nan)
            keep = np.flatnonzero(assign >= 0)
            owner = assign[keep]
            eu = _pd(P[owner], N[keep])
            radii = np.zeros(p)
            np.maximum.at(radii, owner, metric_d[keep])
            self.radii_ = radii
        else:
            owner, keep, eu = boundary_optimal_assignment(P, N, q)
            radii = np.zeros(p)
            np.maximum.at(radii, owner, eu)
            self.radii_ = radii
        self.pairs_ = np.stack([idx_min[owner], idx_maj[keep]], axis=1)
        self.matching_cost_ = float(np.sum(eu))
        self.n_rounds_ = 1
        return keep

    # ------------------------------------------------------------------ coverage arm
    def _coverage(self, P, N, m, cfg, rng):
        k = DEFAULT_N_CANDIDATES if cfg["n_candidates"] is None else int(cfg["n_candidates"])
        dmin = _nearest_distance(N, P) if cfg["keep_rule"] == "boundary" else None
        S = np.arange(len(N))
        total = 0.0
        while len(S) > m:
            s = len(S)
            full_round = (s - s // 2) >= m
            r = s // 2 if full_round else s - m
            Xs = N[S]
            tr = time.perf_counter()
            info = {}
            if cfg["concept"] == "stable":
                pairs, costs = roommates_greedy(Xs, max_pairs=r)
            else:
                pairs, costs, info = min_weight_matching(Xs, n_candidates=k)
                if not full_round:
                    o = np.lexsort((pairs[:, 1], pairs[:, 0], costs))[:r]
                    pairs, costs = pairs[o], costs[o]
            if cfg["keep_rule"] == "random":
                first = rng.random_sample(len(pairs)) < 0.5
            else:
                da, db = dmin[S[pairs[:, 0]]], dmin[S[pairs[:, 1]]]
                first = (da < db) | ((da == db) & (S[pairs[:, 0]] < S[pairs[:, 1]]))
            drop = np.where(first, pairs[:, 1], pairs[:, 0])
            mask = np.ones(s, dtype=bool)
            mask[drop] = False
            S = S[mask]
            total += float(costs.sum())
            self.rounds_.append(dict(size_in=s, size_out=len(S), n_pairs=int(len(pairs)), full=bool(full_round),
                                     cost=float(costs.sum()), seconds=time.perf_counter() - tr,
                                     **{kk: vv for kk, vv in info.items() if kk in
                                        ("solver", "lp_solves", "cuts", "n_edges")}))
        self.n_rounds_ = len(self.rounds_)
        self.matching_cost_ = total
        return S


class MUS_BS(MatchingUnderSampler):
    """Boundary-stable matching undersampler (hospitals/residents)."""

    _parameter_constraints: dict = {
        kk: MatchingUnderSampler._parameter_constraints[kk]
        for kk in ("sampling_strategy", "preferences", "random_state")
    }

    def __init__(self, *, sampling_strategy="auto", preferences="symmetric", random_state=None):
        self.sampling_strategy = sampling_strategy
        self.preferences = preferences
        self.random_state = random_state

    def _config(self):
        return dict(arm="boundary", concept="stable", keep_rule="random", preferences=self.preferences,
                    n_candidates=None)


class MUS_BO(MatchingUnderSampler):
    """Boundary-optimal matching undersampler (rectangular assignment)."""

    _parameter_constraints: dict = {
        kk: MatchingUnderSampler._parameter_constraints[kk] for kk in ("sampling_strategy", "random_state")
    }

    def __init__(self, *, sampling_strategy="auto", random_state=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def _config(self):
        return dict(arm="boundary", concept="optimal", keep_rule="random", preferences="symmetric",
                    n_candidates=None)


class MUS_CS(MatchingUnderSampler):
    """Coverage-stable matching undersampler (iterated stable roommates)."""

    _parameter_constraints: dict = {
        kk: MatchingUnderSampler._parameter_constraints[kk]
        for kk in ("sampling_strategy", "keep_rule", "random_state")
    }

    def __init__(self, *, sampling_strategy="auto", keep_rule="random", random_state=None):
        self.sampling_strategy = sampling_strategy
        self.keep_rule = keep_rule
        self.random_state = random_state

    def _config(self):
        return dict(arm="coverage", concept="stable", keep_rule=self.keep_rule, preferences="symmetric",
                    n_candidates=None)


class MUS_CO(MatchingUnderSampler):
    """Coverage-optimal matching undersampler (iterated min-weight perfect matching)."""

    _parameter_constraints: dict = {
        kk: MatchingUnderSampler._parameter_constraints[kk]
        for kk in ("sampling_strategy", "keep_rule", "n_candidates", "random_state")
    }

    def __init__(self, *, sampling_strategy="auto", keep_rule="random", n_candidates=None, random_state=None):
        self.sampling_strategy = sampling_strategy
        self.keep_rule = keep_rule
        self.n_candidates = n_candidates
        self.random_state = random_state

    def _config(self):
        return dict(arm="coverage", concept="optimal", keep_rule=self.keep_rule, preferences="symmetric",
                    n_candidates=self.n_candidates)
