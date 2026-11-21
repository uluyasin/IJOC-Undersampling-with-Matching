"""Additional undersamplers for the referee experiments (imbalanced-learn compatible).

All samplers are binary, never alter the minority class, follow imbalanced-learn's
``sampling_strategy`` semantics (``'auto'`` -> m = p retained majority instances) and
expose ``sample_indices_`` (sorted positions of the retained rows).

Classes
-------
LocalPivotalUnderSampler      LPM2 (Grafstrom, Lundstrom & Schelin 2012, Biometrics 68:514-520)
                              on the majority class, equal inclusion probabilities m/n,
                              Euclidean distance; exact integer arithmetic.
KMeansStratifiedUnderSampler  k-means-stratified random undersampling (KSRUS): K = min(10, m)
                              clusters on the majority class, proportional allocation of m with
                              largest-remainder rounding, simple random sampling within clusters.
PermutedSampler               wrapper: resample a seeded row permutation, map the selected rows
                              back and return them in original row order.
SafeCapMUSBS                  MUS-BS with capacities only for safe/borderline minority instances.
NoUnsafeMUSBS                 MUS-BS where majority instances with >= 3 minority instances among
                              their 5 nearest neighbours are unacceptable.

Helpers: lpm2_select, ksrus_select, largest_remainder, knn_class_counts, ns_typology,
coverage_error, boundary_proximity, spatial_balance.
"""
from __future__ import annotations

import numbers

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.utils._param_validation import Interval

from imblearn.under_sampling.base import BaseUnderSampler

__all__ = ["LocalPivotalUnderSampler", "KMeansStratifiedUnderSampler", "PermutedSampler", "SafeCapMUSBS",
           "NoUnsafeMUSBS", "lpm2_select", "ksrus_select", "largest_remainder", "knn_class_counts",
           "ns_typology", "coverage_error", "boundary_proximity", "spatial_balance"]


# --------------------------------------------------------------------------- helpers
def _binary_targets(sampler, y):
    """(minority, majority, p, n, m, idx_min, idx_maj) under imblearn sampling_strategy_."""
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) != 2:
        raise ValueError("binary problems only; got %d classes" % len(classes))
    imin = int(np.argmin(counts))
    minority, majority = classes[imin], classes[1 - imin]
    p, n = int(counts[imin]), int(counts[1 - imin])
    targets = sampler.sampling_strategy_
    if minority in targets and int(targets[minority]) != p:
        raise ValueError("the minority class is never altered")
    m = int(min(max(int(targets.get(majority, n)), 0), n))
    return minority, majority, p, n, m, np.flatnonzero(y_arr == minority), np.flatnonzero(y_arr == majority)


def lpm2_select(X, m, random_state=None):
    """Local pivotal method LPM2 with equal inclusion probabilities m/n (exact sample size m).

    Probabilities are held as integer numerators a_i over the common denominator n
    (pi_i = a_i / n), so every update is exact and the final sample has exactly m units.
    Step: draw an undecided unit i uniformly; j = its nearest undecided neighbour
    (Euclidean; ties broken uniformly at random); pivotal update of (pi_i, pi_j)
    (Deville & Tille 1998): if pi_i + pi_j < 1, one of the two gets probability 0 and the
    other pi_i + pi_j (j gets the sum with probability pi_j / (pi_i + pi_j)); otherwise one
    gets 1 and the other pi_i + pi_j - 1 (i gets 1 with probability (1 - pi_j) /
    (2 - pi_i - pi_j)).  Units with pi in {0, 1} are decided.  Returns sorted positions.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    m = int(m)
    if m >= n:
        return np.arange(n)
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    rng = check_random_state(random_state)
    a = np.full(n, m, dtype=np.int64)
    und = np.arange(n)
    while len(und) > 1:
        t = int(rng.randint(len(und)))
        i = int(und[t])
        others = np.concatenate([und[:t], und[t + 1:]])
        diff = X[others] - X[i]
        d = np.einsum("ij,ij->i", diff, diff)
        cand = others[d == d.min()]
        j = int(cand[rng.randint(len(cand))]) if len(cand) > 1 else int(cand[0])
        ai, aj = int(a[i]), int(a[j])
        s = ai + aj
        u = rng.random_sample()
        if s < n:
            if u * s < aj:
                a[i], a[j] = 0, s
            else:
                a[i], a[j] = s, 0
        else:
            if u * (2 * n - s) < (n - aj):
                a[i], a[j] = n, s - n
            else:
                a[i], a[j] = s - n, n
        au = a[und]
        und = und[(au > 0) & (au < n)]
    sel = np.flatnonzero(a == n)
    if len(sel) != m:  # cannot happen: sum(a) = m * n is invariant and the last unit is decided
        raise RuntimeError("LPM2 produced %d instead of %d units" % (len(sel), m))
    return sel


def largest_remainder(m, sizes):
    """Proportional allocation of m units to strata of the given sizes (Hamilton rounding).

    Exact integer arithmetic; remainders tied -> lower stratum index first."""
    sizes = np.asarray(sizes, dtype=np.int64)
    n = int(sizes.sum())
    num = int(m) * sizes
    base = num // n
    rem = num % n
    short = int(m) - int(base.sum())
    order = np.lexsort((np.arange(len(sizes)), -rem))
    alloc = base.copy()
    alloc[order[:short]] += 1
    return alloc


def ksrus_select(X, m, n_clusters=10, n_init=10, random_state=None):
    """k-means-stratified random undersampling; returns (sorted positions, labels, allocation)."""
    from sklearn.cluster import KMeans

    X = np.asarray(X, dtype=float)
    n = len(X)
    m = int(m)
    if m >= n:
        return np.arange(n), np.zeros(n, dtype=np.int64), np.array([n])
    if m <= 0:
        return np.empty(0, dtype=np.int64), np.zeros(n, dtype=np.int64), np.array([0])
    rng = check_random_state(random_state)
    n_distinct = len(np.unique(X, axis=0))
    K = int(max(1, min(n_clusters, m, n_distinct)))
    km_seed = int(rng.randint(np.iinfo(np.int32).max))
    if K == 1:
        lab = np.zeros(n, dtype=np.int64)
    else:
        lab = KMeans(n_clusters=K, n_init=n_init, random_state=km_seed).fit(X).labels_.astype(np.int64)
    sizes = np.bincount(lab, minlength=K)
    alloc = largest_remainder(m, sizes)
    sel = []
    for c in range(K):
        mem = np.flatnonzero(lab == c)
        if alloc[c] > 0:
            sel.append(rng.choice(mem, size=int(alloc[c]), replace=False))
    sel = np.sort(np.concatenate(sel)) if sel else np.empty(0, dtype=np.int64)
    if len(sel) != m:
        raise RuntimeError("KSRUS produced %d instead of %d units" % (len(sel), m))
    return sel, lab, alloc


def knn_class_counts(X, y, query_pos, k=5, count_label=1):
    """For each training row in ``query_pos``: number of its k nearest neighbours (in X,
    excluding the row itself; Euclidean; ties broken by row position) with label count_label."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    query_pos = np.asarray(query_pos, dtype=np.int64)
    out = np.empty(len(query_pos), dtype=np.int64)
    kk = int(min(k, len(X) - 1))
    bs = max(1, 2 ** 22 // max(1, len(X)))
    for st in range(0, len(query_pos), bs):
        q = query_pos[st:st + bs]
        D = cdist(X[q], X)
        D[np.arange(len(q)), q] = np.inf
        o = np.argsort(D, axis=1, kind="stable")[:, :kk]
        out[st:st + bs] = (y[o] == count_label).sum(axis=1)
    return out


def ns_typology(n_majority_neighbours):
    """Napierala-Stefanowski type from the number of majority instances among 5 NN."""
    t = np.asarray(n_majority_neighbours)
    return np.where(t <= 1, "safe", np.where(t <= 3, "borderline", np.where(t == 4, "rare", "outlier")))


def coverage_error(N, S_pos):
    """CE = mean_z d(z, S) / mean_z d(z, N minus z) over the majority set N (mechanisms pipeline)."""
    N = np.asarray(N, dtype=float)
    S_pos = np.unique(np.asarray(S_pos, dtype=np.int64))
    n = len(N)
    dS = np.empty(n)
    dN1 = np.empty(n)
    bs = max(1, 2 ** 22 // max(1, n))
    for st in range(0, n, bs):
        en = min(n, st + bs)
        D = cdist(N[st:en], N)
        dS[st:en] = D[:, S_pos].min(axis=1)
        D[np.arange(en - st), np.arange(st, en)] = np.inf
        dN1[st:en] = D.min(axis=1)
    return float(dS.mean() / dN1.mean()) if dN1.mean() > 0 else np.nan


def boundary_proximity(N, P, S_pos):
    """BP = mean_{z in S} d(z, P) / mean_{z in N} d(z, P) (mechanisms pipeline)."""
    N = np.asarray(N, dtype=float)
    P = np.asarray(P, dtype=float)
    dP = np.empty(len(N))
    bs = max(1, 2 ** 22 // max(1, len(P)))
    for st in range(0, len(N), bs):
        dP[st:st + bs] = cdist(N[st:st + bs], P).min(axis=1)
    return float(dP[np.asarray(S_pos, dtype=np.int64)].mean() / dP.mean())


def spatial_balance(N, S_pos, m=None):
    """Voronoi spatial-balance index B = mean_{s in S} (v_s - 1)^2 (Stevens & Olsen 2004) with equal
    inclusion probabilities m/n; v_s = sum of pi over the population units whose nearest sample unit is s
    (ties to the lower sample position).  0 = perfectly balanced."""
    N = np.asarray(N, dtype=float)
    S_pos = np.sort(np.asarray(S_pos, dtype=np.int64))
    n = len(N)
    m = len(S_pos) if m is None else m
    owner = np.empty(n, dtype=np.int64)
    bs = max(1, 2 ** 22 // max(1, len(S_pos)))
    for st in range(0, n, bs):
        owner[st:st + bs] = np.argmin(cdist(N[st:st + bs], N[S_pos]), axis=1)
    v = np.bincount(owner, minlength=len(S_pos)) * (m / n)
    return float(np.mean((v - 1.0) ** 2))


# --------------------------------------------------------------------------- samplers
class LocalPivotalUnderSampler(BaseUnderSampler):
    """Spatially balanced undersampling of the majority class by the local pivotal method (LPM2)."""

    _parameter_constraints: dict = {**BaseUnderSampler._parameter_constraints, "random_state": ["random_state"]}

    def __init__(self, *, sampling_strategy="auto", random_state=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state

    def _fit_resample(self, X, y):
        _, _, p, n, m, idx_min, idx_maj = _binary_targets(self, y)
        keep = lpm2_select(np.asarray(X, dtype=float)[idx_maj], m, self.random_state)
        self.sample_indices_ = np.sort(np.concatenate([idx_min, idx_maj[keep]]))
        return _safe_indexing(X, self.sample_indices_), _safe_indexing(y, self.sample_indices_)


class KMeansStratifiedUnderSampler(BaseUnderSampler):
    """k-means-stratified random undersampling of the majority class (KSRUS)."""

    _parameter_constraints: dict = {**BaseUnderSampler._parameter_constraints,
                                    "n_clusters": [Interval(numbers.Integral, 1, None, closed="left")],
                                    "n_init": [Interval(numbers.Integral, 1, None, closed="left")],
                                    "random_state": ["random_state"]}

    def __init__(self, *, sampling_strategy="auto", n_clusters=10, n_init=10, random_state=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state

    def _fit_resample(self, X, y):
        _, _, p, n, m, idx_min, idx_maj = _binary_targets(self, y)
        keep, lab, alloc = ksrus_select(np.asarray(X, dtype=float)[idx_maj], m, self.n_clusters, self.n_init,
                                        self.random_state)
        self.cluster_labels_, self.allocation_ = lab, alloc
        self.sample_indices_ = np.sort(np.concatenate([idx_min, idx_maj[keep]]))
        return _safe_indexing(X, self.sample_indices_), _safe_indexing(y, self.sample_indices_)


class PermutedSampler(BaseEstimator):
    """Run ``sampler`` on a seeded random permutation of the rows and return the selected rows in the
    original row order (``sample_indices_`` = sorted original positions).  Isolates the effect of the
    row order (index tie-breaks) on index-based samplers."""

    def __init__(self, sampler, *, random_state=None):
        self.sampler = sampler
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        perm = check_random_state(self.random_state).permutation(len(y))
        inner = clone(self.sampler)
        inner.fit_resample(X[perm], y[perm])
        self.sampler_ = inner
        self.permutation_ = perm
        self.sample_indices_ = np.sort(perm[np.asarray(inner.sample_indices_)])
        return X[self.sample_indices_], y[self.sample_indices_]

    def fit(self, X, y):
        self.fit_resample(X, y)
        return self


class _MUSBSVariant(BaseUnderSampler):
    _parameter_constraints: dict = {**BaseUnderSampler._parameter_constraints,
                                    "n_neighbors": [Interval(numbers.Integral, 1, None, closed="left")],
                                    "random_state": ["random_state"]}

    def __init__(self, *, sampling_strategy="auto", n_neighbors=5, random_state=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    @staticmethod
    def _ms():
        import matchsampling  # the frozen module (bench_runner loads it into sys.modules)
        return matchsampling


class SafeCapMUSBS(_MUSBSVariant):
    """MUS-BS in which only safe and borderline minority instances (Napierala-Stefanowski typology from
    the 5 nearest neighbours in the training data: 0-1 majority neighbours safe, 2-3 borderline, 4 rare,
    5 outlier) receive capacities; m is split among them as in ``capacities_for_target``; all minority
    instances are kept.  Falls back to standard MUS-BS (``fallback_ = True``) when no minority instance
    is safe or borderline."""

    def _fit_resample(self, X, y):
        ms = self._ms()
        minority, majority, p, n, m, idx_min, idx_maj = _binary_targets(self, y)
        Xf = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        nmaj = knn_class_counts(Xf, y_arr, idx_min, self.n_neighbors, count_label=majority)
        self.minority_types_ = ns_typology(nmaj)
        hosp = np.flatnonzero(nmaj <= 3)
        self.fallback_ = len(hosp) == 0
        if self.fallback_:
            hosp = np.arange(p)
        rng = check_random_state(self.random_state)
        if m >= n:
            keep = np.arange(n)
        else:
            q = ms.capacities_for_target(len(hosp), m, rng)
            assign, _ = ms.hr_greedy(Xf[idx_min[hosp]], Xf[idx_maj], q)
            keep = np.flatnonzero(assign >= 0)
            self.capacities_ = q
        self.n_hospitals_ = len(hosp)
        self.sample_indices_ = np.sort(np.concatenate([idx_min, idx_maj[keep]]))
        return _safe_indexing(X, self.sample_indices_), _safe_indexing(y, self.sample_indices_)


class NoUnsafeMUSBS(_MUSBSVariant):
    """MUS-BS in which majority instances with at least 3 minority instances among their 5 nearest
    neighbours are unacceptable (removed from the matching).  If fewer than m acceptable majority
    instances remain, the least-unsafe unacceptable ones (fewest minority neighbours, then lowest
    position) are added until m are available (``topped_up_ = True``)."""

    def _fit_resample(self, X, y):
        ms = self._ms()
        minority, majority, p, n, m, idx_min, idx_maj = _binary_targets(self, y)
        Xf = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        nmin = knn_class_counts(Xf, y_arr, idx_maj, self.n_neighbors, count_label=minority)
        acc = nmin < 3
        self.n_unacceptable_ = int((~acc).sum())
        self.topped_up_ = int(acc.sum()) < m
        if self.topped_up_:
            un = np.flatnonzero(~acc)
            add = un[np.lexsort((un, nmin[un]))][: m - int(acc.sum())]
            acc[add] = True
            self.n_added_ = len(add)
        else:
            self.n_added_ = 0
        pool = np.flatnonzero(acc)
        rng = check_random_state(self.random_state)
        if m >= len(pool):
            keep = pool
        else:
            q = ms.capacities_for_target(p, m, rng)
            assign, _ = ms.hr_greedy(Xf[idx_min], Xf[idx_maj[pool]], q)
            keep = pool[np.flatnonzero(assign >= 0)]
        self.sample_indices_ = np.sort(np.concatenate([idx_min, idx_maj[keep]]))
        return _safe_indexing(X, self.sample_indices_), _safe_indexing(y, self.sample_indices_)
