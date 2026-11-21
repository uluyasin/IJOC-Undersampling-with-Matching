"""Runner for the referee experiments X1-X6 of the matching-undersampling study.

Reuses bench_runner (make_sampler, make_classifier, scores_of, compute_metrics, preprocessing
convention, seed = folds[ds]['seed'] * 100 + k) so that every outer fit is identical to the main run.

Work distribution: a deterministic global task list; independent shard processes claim tasks by
atomically creating claim files (os.O_CREAT | os.O_EXCL) and write one parquet per task (the sandbox
forbids named pipes, so process pools are unavailable).

Usage:  python referee_runner.py <cfg.json> <shard_id>
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, json, time, glob, warnings, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import joblib
import bench_runner as br
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

CLASSIFIERS = ["CART", "kNN", "SVM", "LR", "RF"]
BALANCING = ["RUS", "NM1", "NM2", "NM3", "CC", "MUS-BS", "MUS-BO", "MUS-CS"]   # MUS-CO excluded (cost)
MUS3 = ["MUS-BS", "MUS-BO", "MUS-CS"]
METRICS = ["gmean", "bacc", "sens", "spec", "prec", "f1", "mcc", "auc", "prauc"]
T_ENS = 10

CFG = None
_CACHE = {}
ES = None   # extra_samplers module


# ----------------------------------------------------------------------------- setup / data
def init(cfg):
    global CFG, ES
    CFG = cfg
    br._load_matchsampling(cfg["ms_path"])      # registers sys.modules['matchsampling'] (frozen module)
    import extra_samplers as es
    ES = es


def _load(key):
    if key not in _CACHE:
        _CACHE[key] = joblib.load(CFG[key])
    return _CACHE[key]


def sub_seed(seed, code):
    """Deterministic 32-bit seed derived from (unit seed, code)."""
    return int(np.random.SeedSequence([int(seed), int(code)]).generate_state(1, dtype=np.uint32)[0])


def prep(ds, k, dedup=False):
    data = _load("dedup_data_path" if dedup else "data_path")
    folds = _load("dedup_folds_path" if dedup else "folds_path")
    D = data[ds]
    X, y = np.asarray(D["X"], dtype=float), np.asarray(D["y"]).astype(int)
    f = folds[ds]["folds"][k]
    tr, te = np.asarray(f["train_idx"]), np.asarray(f["test_idx"])
    seed = int(folds[ds]["seed"]) * 100 + k
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sc = MinMaxScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X[tr]))
    X_te = sc.transform(imp.transform(X[te]))
    return dict(X_tr=X_tr, y_tr=y[tr], X_te=X_te, y_te=y[te], tr=tr, te=te, seed=seed,
                repeat=int(f["repeat"]), fold=int(f["fold"]), n=len(y))


def make_any_sampler(name, seed):
    base = name.split(":")[0]
    if base == "LPM":
        return ES.LocalPivotalUnderSampler(random_state=seed)
    if base == "KSRUS":
        return ES.KMeansStratifiedUnderSampler(random_state=seed)
    if name == "MUS-BS:safecap":
        return ES.SafeCapMUSBS(random_state=seed)
    if name == "MUS-BS:nounsafe":
        return ES.NoUnsafeMUSBS(random_state=seed)
    return br.make_sampler(name, seed, CFG["ms_path"])


def resample(sampler, X_tr, y_tr, name):
    """bench_runner semantics: (X_res, y_res, idx, err)."""
    if sampler is None:
        return X_tr, y_tr, np.arange(len(y_tr)), None
    try:
        X_res, y_res = sampler.fit_resample(X_tr, y_tr)
        if hasattr(sampler, "sample_indices_"):
            idx = np.asarray(sampler.sample_indices_)
        elif name.split(":")[0] == "CC":
            idx = br._cc_indices(X_tr, y_tr, X_res, y_res)
        else:
            idx = None
        return X_res, y_res, idx, None
    except Exception as e:
        return X_tr, y_tr, None, f"{type(e).__name__}: {e}"[:300]


def fit_score(clf, X_res, y_res, X_te, y_te):
    t1 = time.perf_counter()
    if len(np.unique(y_res)) < 2:
        raise ValueError("resampled training set has a single class")
    clf.fit(X_res, y_res)
    t_fit = time.perf_counter() - t1
    pred = clf.predict(X_te)
    sc = br.scores_of(clf, X_te)
    return pred, sc, t_fit


def default_threshold(cname):
    return 0.0 if cname == "SVM" else 0.5      # SVC(probability=False) -> decision_function


def base_row(tag, ds, P, sname, y_res, t_res, err):
    y_tr = P["y_tr"]
    return dict(tag=tag, dataset=ds, repeat=P["repeat"], fold=P["fold"], sampler=sname,
                n_train=len(y_tr), n_min_train=int((y_tr == 1).sum()), n_maj_train=int((y_tr == 0).sum()),
                n_min_res=int((y_res == 1).sum()), n_maj_res=int((y_res == 0).sum()),
                t_resample=t_res, sampler_error=err)


def kept_major(P, idx):
    return np.sort(P["tr"][idx[P["y_tr"][idx] == 0]]).astype(np.int64)


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    u = len(a | b)
    return len(a & b) / u if u else 1.0


# ----------------------------------------------------------------------------- X1 nested
def svm_scale_gamma(X):
    v = X.var()
    return 1.0 / (X.shape[1] * v) if v != 0 else 1.0


def grid_for(cname, knn_cap):
    """List of (label, params); params None = the bench_runner default classifier."""
    if cname == "CART":
        return [("msl=%d" % v, None if v == 1 else dict(min_samples_leaf=v)) for v in (1, 2, 5, 10)]
    if cname == "kNN":
        vals, out = [], []
        for v in (1, 3, 5, 9, 15):
            vv = int(min(v, knn_cap))
            if vv >= 1 and vv not in vals:
                vals.append(vv)
                out.append(("k=%d" % vv, None if vv == 5 else dict(n_neighbors=vv)))
        return out
    if cname == "SVM":
        return [("C=%g,g=%gx" % (C, gm), None if (C == 1 and gm == 1) else dict(C=C, gmult=gm))
                for C in (0.1, 1, 10, 100) for gm in (0.1, 1, 10)]
    if cname == "LR":
        return [("C=%g" % C, None if C == 1 else dict(C=C)) for C in (0.01, 0.1, 1, 10, 100)]
    if cname == "RF":
        return [("msl=%d,mf=%s" % (l, f), None if (l == 1 and f == "sqrt") else dict(min_samples_leaf=l, max_features=f))
                for l in (1, 3) for f in ("sqrt", 0.5)]
    raise ValueError(cname)


def build_clf(cname, params, seed, X_fit):
    if params is None:
        return br.make_classifier(cname, seed)
    if cname == "CART":
        return DecisionTreeClassifier(random_state=seed, min_samples_leaf=params["min_samples_leaf"])
    if cname == "kNN":
        return KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    if cname == "SVM":
        return SVC(kernel="rbf", C=params["C"], gamma=params["gmult"] * svm_scale_gamma(X_fit))
    if cname == "LR":
        return LogisticRegression(C=params["C"], max_iter=2000)
    if cname == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1,
                                      min_samples_leaf=params["min_samples_leaf"],
                                      max_features=params["max_features"])
    raise ValueError(cname)


def gmean_of(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    P_, N_ = (y == 1).sum(), (y == 0).sum()
    tpr = ((pred == 1) & (y == 1)).sum() / P_ if P_ else 0.0
    tnr = ((pred == 0) & (y == 0)).sum() / N_ if N_ else 0.0
    return float(np.sqrt(tpr * tnr))


def best_threshold(y, s, default):
    """Threshold maximising G-mean of (s > t) on pooled OOF scores; candidates = midpoints between
    sorted unique scores plus the default; ties -> closest to the default, then the smaller one."""
    ok = np.isfinite(s)
    y, s = np.asarray(y)[ok], np.asarray(s)[ok]
    if len(s) == 0 or (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float(default), np.nan, np.nan
    u = np.unique(s)
    cands = np.concatenate([(u[:-1] + u[1:]) / 2.0, [float(default)]])
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    tp = len(pos) - np.searchsorted(pos, cands, side="right")
    tn = np.searchsorted(neg, cands, side="right")
    gm = np.sqrt((tp / len(pos)) * (tn / len(neg)))
    best = gm.max()
    idx = np.flatnonzero(gm >= best * (1 - 1e-12))
    c = cands[idx]
    o = np.lexsort((c, np.abs(c - default)))
    t = float(c[o[0]])
    gm_def = float(gm[-1])
    return t, float(best), gm_def


def run_x1(unit):
    ds, k, sname = unit
    P = prep(ds, k)
    X_tr, y_tr, X_te, y_te, us = P["X_tr"], P["y_tr"], P["X_te"], P["y_te"], P["seed"]
    t_unit = time.perf_counter()
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=sub_seed(us, 1))
    inner = []
    for j, (itr, iva) in enumerate(skf.split(X_tr, y_tr)):
        smp = make_any_sampler(sname, sub_seed(us, 10 + j))
        Xr, yr, _, err = resample(smp, X_tr[itr], y_tr[itr], sname)
        inner.append((Xr, yr, iva, err))
    n_res_min = min(len(t[1]) for t in inner)
    rows, grid_rows = [], []
    choice = {}
    for cname in CLASSIFIERS:
        G = grid_for(cname, n_res_min - 1)
        gm = np.full((len(G), 3), np.nan)
        oof = np.full(len(y_tr), np.nan)
        dflt = [gi for gi, (_, prm) in enumerate(G) if prm is None][0]
        for gi, (lab, prm) in enumerate(G):
            for j, (Xr, yr, iva, err) in enumerate(inner):
                if err is not None:
                    continue
                try:
                    clf = build_clf(cname, prm, us, Xr)
                    pred, sc, _ = fit_score(clf, Xr, yr, X_tr[iva], y_tr[iva])
                    gm[gi, j] = gmean_of(y_tr[iva], pred)
                    if gi == dflt:
                        oof[iva] = sc
                except Exception:
                    pass
        with np.errstate(all="ignore"):
            mean_gm = np.nanmean(gm, axis=1)
        if np.all(np.isnan(mean_gm)):
            best = dflt
        else:
            mx = np.nanmax(mean_gm)
            tied = np.flatnonzero(mean_gm >= mx - 1e-12)
            best = dflt if dflt in tied else int(tied[0])
        thr, oof_best, oof_def = best_threshold(y_tr, oof, default_threshold(cname))
        choice[cname] = (G, dflt, best, thr, oof_best, oof_def, mean_gm)
        for gi, (lab, prm) in enumerate(G):
            grid_rows.append(dict(dataset=ds, repeat=P["repeat"], fold=P["fold"], sampler=sname, classifier=cname,
                                  grid=lab, is_default=gi == dflt, g0=gm[gi, 0], g1=gm[gi, 1], g2=gm[gi, 2],
                                  mean_gmean=mean_gm[gi]))
    # outer: exactly as bench_runner
    smp = make_any_sampler(sname, us)
    t0 = time.perf_counter()
    X_res, y_res, idx, err = resample(smp, X_tr, y_tr, sname)
    t_res = time.perf_counter() - t0
    b = base_row("x1", ds, P, sname, y_res, t_res, err)
    b.update(inner_errors=sum(t[3] is not None for t in inner), n_res_inner_min=n_res_min)
    for cname in CLASSIFIERS:
        G, dflt, best, thr, oof_best, oof_def, mean_gm = choice[cname]
        common = dict(b, classifier=cname, thr_default=default_threshold(cname), thr_chosen=thr,
                      oof_gmean_default_thr=oof_def, oof_gmean_chosen_thr=oof_best,
                      tuned_params=G[best][0], tuned_is_default=bool(best == dflt),
                      inner_gmean_default=mean_gm[dflt], inner_gmean_tuned=mean_gm[best])
        try:
            clf = br.make_classifier(cname, us)
            pred, sc, t_fit = fit_score(clf, X_res, y_res, X_te, y_te)
            m_def = br.compute_metrics(y_te, pred, sc)
            m_thr = br.compute_metrics(y_te, (sc > thr).astype(int), sc)
            if best == dflt:
                m_tun = dict(m_def)
            else:
                clf2 = build_clf(cname, G[best][1], us, X_res)
                pred2, sc2, _ = fit_score(clf2, X_res, y_res, X_te, y_te)
                m_tun = br.compute_metrics(y_te, pred2, sc2)
            for var, mm in (("default", m_def), ("thr_adjusted", m_thr), ("tuned", m_tun)):
                rows.append(dict(common, variant=var, t_fit=t_fit, **mm))
        except Exception as e:
            for var in ("default", "thr_adjusted", "tuned"):
                rows.append(dict(common, variant=var, clf_error=f"{type(e).__name__}: {e}"[:300]))
    tu = time.perf_counter() - t_unit
    for r in rows:
        r["t_unit"] = tu
    return rows, grid_rows


# ----------------------------------------------------------------------------- X2 tie-breaking
def _sel_index():
    if "sel" not in _CACHE:
        _CACHE["sel"] = joblib.load(CFG["sel_small_path"])
    return _CACHE["sel"]


def run_x2(unit):
    ds, k, sname = unit
    P = prep(ds, k)
    X_tr, y_tr, X_te, y_te, us = P["X_tr"], P["y_tr"], P["X_te"], P["y_te"], P["seed"]
    # original selection (recomputed) and stored main-run selection
    s_o = make_any_sampler(sname, us)
    _, _, idx_o, err_o = resample(s_o, X_tr, y_tr, sname)
    kept_o = kept_major(P, idx_o) if idx_o is not None else np.array([], dtype=np.int64)
    stored = _sel_index().get((ds, P["repeat"], P["fold"], sname))
    match_main = bool(stored is not None and np.array_equal(np.sort(stored), kept_o))
    perm_seed = sub_seed(us, 2002)
    ps = ES.PermutedSampler(make_any_sampler(sname, us), random_state=perm_seed)
    t0 = time.perf_counter()
    X_res, y_res, idx_p, err = resample(ps, X_tr, y_tr, sname)
    t_res = time.perf_counter() - t0
    kept_p = kept_major(P, idx_p) if idx_p is not None else np.array([], dtype=np.int64)
    n_maj, m = int((y_tr == 0).sum()), len(kept_o)
    extra = dict(jaccard=jaccard(kept_o, kept_p), jaccard_chance=m / (2 * n_maj - m) if n_maj else np.nan,
                 same_selection=bool(np.array_equal(kept_o, kept_p)), recomputed_matches_main=match_main,
                 perm_seed=perm_seed)
    if sname == "MUS-CS":   # round-1 pair structure (separates index tie-breaks from the keep coin)
        ms = sys.modules["matchsampling"]
        pos_maj = np.flatnonzero(y_tr == 0)
        perm = ps.permutation_
        pos_maj_p = np.flatnonzero(y_tr[perm] == 0)          # permuted positions of the majority rows
        pa, _ = ms.roommates_greedy(X_tr[pos_maj])
        pb, _ = ms.roommates_greedy(X_tr[perm][pos_maj_p])
        A = {tuple(sorted((int(pos_maj[a]), int(pos_maj[b])))) for a, b in pa}
        orig_of_p = perm[pos_maj_p]
        B = {tuple(sorted((int(orig_of_p[a]), int(orig_of_p[b])))) for a, b in pb}
        extra["pair_jaccard_round1"] = len(A & B) / len(A | B) if (A | B) else 1.0
    b = base_row("x2", ds, P, sname, y_res, t_res, err)
    b.update(extra)
    rows = []
    for cname in CLASSIFIERS:
        r = dict(b, classifier=cname)
        try:
            pred, sc, t_fit = fit_score(br.make_classifier(cname, us), X_res, y_res, X_te, y_te)
            r["t_fit"] = t_fit
            r.update(br.compute_metrics(y_te, pred, sc))
        except Exception as e:
            r["clf_error"] = f"{type(e).__name__}: {e}"[:300]
        rows.append(r)
    return rows, []


# ----------------------------------------------------------------------------- X3 / X4 / X6 (single subset)
def run_single(unit, tag, dedup=False):
    ds, k, sname = unit
    P = prep(ds, k, dedup=dedup)
    X_tr, y_tr, X_te, y_te, us = P["X_tr"], P["y_tr"], P["X_te"], P["y_te"], P["seed"]
    smp = None if sname in ("NONE", "CW") else make_any_sampler(sname, us)
    t0 = time.perf_counter()
    X_res, y_res, idx, err = resample(smp, X_tr, y_tr, sname)
    t_res = time.perf_counter() - t0
    b = base_row(tag, ds, P, sname, y_res, t_res, err)
    if tag == "x3" and smp is not None and err is None:
        b.update(fallback=bool(getattr(smp, "fallback_", False)), n_hospitals=getattr(smp, "n_hospitals_", np.nan),
                 topped_up=bool(getattr(smp, "topped_up_", False)), n_added=getattr(smp, "n_added_", np.nan),
                 n_unacceptable=getattr(smp, "n_unacceptable_", np.nan))
        if hasattr(smp, "minority_types_"):
            t = smp.minority_types_
            b.update({f"n_{c}": int((t == c).sum()) for c in ("safe", "borderline", "rare", "outlier")})
        stored = _sel_index().get((ds, P["repeat"], P["fold"], "MUS-BS"))
        if stored is not None and idx is not None:
            b["jaccard_vs_musbs"] = jaccard(stored, kept_major(P, idx))
    if tag == "x4" and idx is not None:
        pos_maj = np.flatnonzero(y_tr == 0)
        N, Pm = X_tr[pos_maj], X_tr[y_tr == 1]
        to_pos = np.full(P["n"], -1, dtype=np.int64)
        to_pos[P["tr"][pos_maj]] = np.arange(len(pos_maj))
        sels = {sname: to_pos[kept_major(P, idx)]}
        if sname == "LPM":
            for other in ("RUS", "MUS-CS"):
                st = _sel_index().get((ds, P["repeat"], P["fold"], other))
                if st is not None:
                    sels[other] = to_pos[np.asarray(st, dtype=np.int64)]
        for nm, S in sels.items():
            key = "self" if nm == sname else nm
            b[f"CE_{key}"] = ES.coverage_error(N, S)
            b[f"BP_{key}"] = ES.boundary_proximity(N, Pm, S)
            b[f"SB_{key}"] = ES.spatial_balance(N, S)
    rows = []
    for cname in CLASSIFIERS:
        r = dict(b, classifier=cname)
        try:
            pred, sc, t_fit = fit_score(br.make_classifier(cname, us), X_res, y_res, X_te, y_te)
            r["t_fit"] = t_fit
            r.update(br.compute_metrics(y_te, pred, sc))
        except Exception as e:
            r["clf_error"] = f"{type(e).__name__}: {e}"[:300]
        rows.append(r)
    return rows, []


# ----------------------------------------------------------------------------- X5 ensembles
def run_x5(unit):
    ds, k, sname = unit
    P = prep(ds, k)
    X_tr, y_tr, X_te, y_te, us = P["X_tr"], P["y_tr"], P["X_te"], P["y_te"], P["seed"]
    subs, t_res, errs = [], 0.0, 0
    for t in range(T_ENS):
        s_t = (us * 10 + t) % (2 ** 32)
        t0 = time.perf_counter()
        Xr, yr, idx, err = resample(make_any_sampler(sname, s_t), X_tr, y_tr, sname)
        t_res += time.perf_counter() - t0
        if err is not None:
            errs += 1
            continue
        subs.append((Xr, yr))
    b = base_row("x5", ds, P, sname, subs[0][1] if subs else y_tr, t_res, None)
    b.update(n_members=len(subs), member_errors=errs)
    rows = []
    for cname in CLASSIFIERS:
        r = dict(b, classifier=cname)
        try:
            S, gms, aucs, t_fit = [], [], [], 0.0
            for Xr, yr in subs:
                pred, sc, tf = fit_score(br.make_classifier(cname, us), Xr, yr, X_te, y_te)
                t_fit += tf
                mm = br.compute_metrics(y_te, pred, sc)
                gms.append(mm["gmean"])
                aucs.append(mm["auc"])
                S.append(sc)
            ens = np.mean(np.vstack(S), axis=0)
            r.update(br.compute_metrics(y_te, (ens > default_threshold(cname)).astype(int), ens))
            r["t_fit"] = t_fit
            for t, (g, a) in enumerate(zip(gms, aucs)):
                r[f"gm_m{t}"] = g
                r[f"auc_m{t}"] = a
            r["member_gmean_mean"], r["member_gmean_sd"] = float(np.mean(gms)), float(np.std(gms, ddof=1))
            r["member_auc_mean"], r["member_auc_sd"] = float(np.nanmean(aucs)), float(np.nanstd(aucs, ddof=1))
        except Exception as e:
            r["clf_error"] = f"{type(e).__name__}: {e}"[:300]
        rows.append(r)
    return rows, []


RUNNERS = {"x1": run_x1, "x2": run_x2,
           "x3": lambda u: run_single(u, "x3"), "x4": lambda u: run_single(u, "x4"),
           "x5": run_x5, "x6": lambda u: run_single(u, "x6", dedup=True)}


# ----------------------------------------------------------------------------- task list / queue
def build_tasks(cfg):
    data, folds = joblib.load(cfg["data_path"]), joblib.load(cfg["folds_path"])
    names = sorted(data)
    ex = cfg["experiments"]
    rng = np.random.default_rng(20261120)
    tasks = {}

    def chunks(exp, units, size):
        units = [units[i] for i in rng.permutation(len(units))]
        return [(f"{exp}_{i:05d}", exp, units[st:st + size]) for i, st in enumerate(range(0, len(units), size))]

    if "x1" in ex:
        K1 = ex["x1"]["n_folds"]
        tasks["x1"] = chunks("x1", [(d, k, s) for d in names for k in range(K1) for s in BALANCING], ex["x1"]["chunk"])
    if "x2" in ex:
        tasks["x2"] = chunks("x2", [(d, k, s) for d in names for k in range(25) for s in MUS3], ex["x2"]["chunk"])
    if "x3" in ex:
        tasks["x3"] = chunks("x3", [(d, k, s) for d in names for k in range(25)
                                    for s in ("MUS-BS:safecap", "MUS-BS:nounsafe")], ex["x3"]["chunk"])
    if "x4" in ex:
        tasks["x4"] = chunks("x4", [(d, k, s) for d in names for k in range(25) for s in ("LPM", "KSRUS")],
                             ex["x4"]["chunk"])
    if "x5" in ex:
        tasks["x5"] = chunks("x5", [(d, k, s) for d in names for k in range(25) for s in ("RUS", "MUS-CS", "LPM")],
                             ex["x5"]["chunk"])
    if "x6" in ex:
        dd = sorted(joblib.load(cfg["dedup_data_path"]))
        tasks["x6"] = chunks("x6", [(d, k, s) for d in dd for k in range(25) for s in ["NONE"] + BALANCING],
                             ex["x6"]["chunk"])
    order = []
    for e in cfg.get("first", []):
        order += tasks.pop(e, [])
    rest = list(tasks.values())          # interleave the remaining experiments
    while any(rest):
        for L in rest:
            if L:
                order.append(L.pop(0))
    return order


def task_paths(out_dir, task_id, exp):
    return (os.path.join(out_dir, exp, task_id + ".parquet"), os.path.join(out_dir, exp + "_extra", task_id + ".parquet"),
            os.path.join(out_dir, "claims", task_id + ".claim"))


def shard_main(cfg, shard):
    init(cfg)
    out_dir = cfg["out_dir"]
    tasks = build_tasks(cfg)
    for d in set(t[1] for t in tasks):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
        os.makedirs(os.path.join(out_dir, d + "_extra"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "claims"), exist_ok=True)
    stop_file = os.path.join(out_dir, "STOP")
    n_done, t0 = 0, time.time()
    print(f"[s{shard:02d}] {len(tasks)} tasks in queue", flush=True)
    for task_id, exp, units in tasks:
        if os.path.exists(stop_file):
            print(f"[s{shard:02d}] STOP file found", flush=True)
            break
        out, out_x, claim = task_paths(out_dir, task_id, exp)
        if os.path.exists(out):
            continue
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(shard).encode())
            os.close(fd)
        except FileExistsError:
            continue
        ts = time.time()
        rows, extra = [], []
        for u in units:
            try:
                r, x = RUNNERS[exp](u)
            except Exception as e:
                r, x = [dict(tag=exp, dataset=u[0], fold_index=u[1], sampler=u[2],
                             unit_error=(f"{type(e).__name__}: {e} | " + traceback.format_exc()[-500:])[:900])], []
            rows += r
            extra += x
        tmp = out + f".tmp{shard}"
        pd.DataFrame(rows).to_parquet(tmp)
        if extra:
            pd.DataFrame(extra).to_parquet(out_x)
        os.replace(tmp, out)
        n_done += 1
        print(f"[s{shard:02d}] {task_id} {len(units)} units {time.time() - ts:.1f}s | done {n_done} "
              f"| {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"[s{shard:02d}] finished {n_done} tasks in {(time.time() - t0) / 60:.1f} min", flush=True)


def launch(cfg_path, nshards, first_shard=0, python_exe=None):
    import subprocess
    cfg = json.load(open(cfg_path))
    log_dir = os.path.join(cfg["out_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for i in range(first_shard, first_shard + nshards):
        fh = open(os.path.join(log_dir, f"shard{i:02d}.log"), "a")
        p = subprocess.Popen([python_exe or sys.executable, os.path.abspath(__file__), cfg_path, str(i)],
                             stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
        procs.append((p, fh))
    return procs


def collect(out_dir, exp, extra=False):
    parts = sorted(glob.glob(os.path.join(out_dir, exp + ("_extra" if extra else ""), f"{exp}_*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


if __name__ == "__main__":
    shard_main(json.load(open(sys.argv[1])), int(sys.argv[2]))
