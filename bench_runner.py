"""Leakage-safe benchmark runner for the matching-undersampling study.

Unit of work = (dataset, repeat, fold, sampler). For each unit:
  1. median imputation + MinMax[0,1] scaling fitted on the training fold only;
  2. the sampler is fitted on the scaled training fold (one resampling per unit);
  3. every classifier is trained on the resampled set and scored on the untouched test fold.
Outputs per unit: metric rows (one per classifier) and the selected original-row indices of
retained majority instances (for selection diagnostics).
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time, warnings, glob, json, importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, confusion_matrix)
from imblearn.under_sampling import (RandomUnderSampler, NearMiss, TomekLinks,
    EditedNearestNeighbours, NeighbourhoodCleaningRule, OneSidedSelection,
    CondensedNearestNeighbour, InstanceHardnessThreshold, ClusterCentroids)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

CLASSIFIERS = ["CART", "kNN", "SVM", "LR", "RF"]
BASELINES = ["NONE", "RUS", "NM1", "NM2", "NM3", "TL", "ENN", "NCR", "OSS", "CNN", "IHT", "CC"]
REFERENCES = ["SMOTE", "CW"]          # oversampling and cost-sensitive references
PROPOSED = ["MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]

_DATA = None
_FOLDS = None
_MS = None   # matchsampling module


def _load_matchsampling(path):
    global _MS
    if _MS is None and path:
        spec = importlib.util.spec_from_file_location("matchsampling", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["matchsampling"] = mod
        spec.loader.exec_module(mod)
        _MS = mod
    return _MS


def _get(data_path, folds_path):
    global _DATA, _FOLDS
    if _DATA is None:
        _DATA = joblib.load(data_path)
    if _FOLDS is None:
        _FOLDS = joblib.load(folds_path)
    return _DATA, _FOLDS


def make_sampler(name, seed, ms_path=None, params=None):
    """Return a fresh sampler (None for NONE / CW).

    `name` may carry a variant label after a colon (e.g. "RUS:r2", "MUS-CS:keep=boundary");
    the label is free text, the constructor parameters come from `params`
    (cfg["sampler_params"][name])."""
    p = dict(params or {})
    base = name.split(":")[0]
    if base in ("NONE", "CW"):
        return None
    if base == "RUS":
        return RandomUnderSampler(random_state=seed, **p)
    if base in ("NM1", "NM2", "NM3"):
        return NearMiss(version=int(base[-1]), **p)
    if base == "TL":
        return TomekLinks(**p)
    if base == "ENN":
        return EditedNearestNeighbours(**p)
    if base == "NCR":
        return NeighbourhoodCleaningRule(**p)
    if base == "OSS":
        return OneSidedSelection(random_state=seed, **p)
    if base == "CNN":
        return CondensedNearestNeighbour(random_state=seed, **p)
    if base == "IHT":
        return InstanceHardnessThreshold(random_state=seed, **p)
    if base == "CC":
        return ClusterCentroids(voting="hard", random_state=seed, **p)
    if base == "SMOTE":
        return SMOTE(random_state=seed, **p)
    if base.startswith("MUS-"):
        ms = _load_matchsampling(ms_path)
        arm = {"B": "boundary", "C": "coverage"}[base[4]]
        concept = {"S": "stable", "O": "optimal"}[base[5]]
        return ms.MatchingUnderSampler(arm=arm, concept=concept, random_state=seed, **p)
    raise ValueError(name)


def make_classifier(name, seed, class_weight=None):
    if name == "CART":
        return DecisionTreeClassifier(random_state=seed, class_weight=class_weight)
    if name == "kNN":
        return None if class_weight else KNeighborsClassifier(n_neighbors=5)
    if name == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=class_weight)
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=2000, class_weight=class_weight)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1,
                                      class_weight=class_weight)
    raise ValueError(name)


def scores_of(clf, X):
    if hasattr(clf, "predict_proba"):
        try:
            return clf.predict_proba(X)[:, 1]
        except Exception:
            pass
    return clf.decision_function(X)


def compute_metrics(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    out = dict(gmean=float(np.sqrt(tpr * tnr)), bacc=0.5 * (tpr + tnr), sens=tpr, spec=tnr,
               prec=prec, f1=float(f1_score(y_true, y_pred, zero_division=0)),
               mcc=float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0)
    try:
        out["auc"] = float(roc_auc_score(y_true, score))
        out["prauc"] = float(average_precision_score(y_true, score))
    except Exception:
        out["auc"], out["prauc"] = np.nan, np.nan
    return out


def _cc_indices(X_tr, y_tr, X_res, y_res):
    """Recover original training-fold indices of hard-voting ClusterCentroids output."""
    maj = np.flatnonzero(y_tr == 0)
    nn = NearestNeighbors(n_neighbors=1).fit(X_tr[maj])
    _, ind = nn.kneighbors(X_res[y_res == 0])
    return np.sort(np.concatenate([maj[ind.ravel()], np.flatnonzero(y_tr == 1)]))


def run_unit(unit, data_path, folds_path, ms_path=None, classifiers=CLASSIFIERS,
             sampler_params=None, tag="main"):
    """unit = (dataset, fold_no, sampler_name). Returns (rows, sel_record)."""
    ds, k, sname = unit
    data, folds = _get(data_path, folds_path)
    D = data[ds]
    X, y = np.asarray(D["X"], dtype=float), np.asarray(D["y"]).astype(int)
    f = folds[ds]["folds"][k]
    tr, te = np.asarray(f["train_idx"]), np.asarray(f["test_idx"])
    rep, fold = int(f["repeat"]), int(f["fold"])
    seed = int(folds[ds]["seed"]) * 100 + k
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sc = MinMaxScaler()
    X_tr = sc.fit_transform(imp.fit_transform(X[tr]))
    X_te = sc.transform(imp.transform(X[te]))
    y_tr, y_te = y[tr], y[te]

    sp = dict((sampler_params or {}).get(sname) or {})
    if "target_ratio" in sp:   # majority:minority ratio r, capped at the available majority
        r_ = float(sp.pop("target_ratio"))
        n_min_, n_maj_ = int((y_tr == 1).sum()), int((y_tr == 0).sum())
        sp["sampling_strategy"] = {0: int(min(np.ceil(r_ * n_min_), n_maj_))}
    sampler = make_sampler(sname, seed, ms_path, sp)
    t0 = time.perf_counter()
    err = None
    if sampler is None:
        X_res, y_res, idx = X_tr, y_tr, np.arange(len(y_tr))
    else:
        try:
            X_res, y_res = sampler.fit_resample(X_tr, y_tr)
            if hasattr(sampler, "sample_indices_"):
                idx = np.asarray(sampler.sample_indices_)
            elif sname.split(":")[0] == "CC":
                idx = _cc_indices(X_tr, y_tr, X_res, y_res)
            else:
                idx = None
        except Exception as e:  # recorded, never silently dropped
            err = f"{type(e).__name__}: {e}"[:300]
            X_res, y_res, idx = X_tr, y_tr, None
    t_res = time.perf_counter() - t0

    base = dict(tag=tag, dataset=ds, repeat=rep, fold=fold, sampler=sname,
                n_train=len(y_tr), n_min_train=int((y_tr == 1).sum()), n_maj_train=int((y_tr == 0).sum()),
                n_min_res=int((y_res == 1).sum()), n_maj_res=int((y_res == 0).sum()),
                t_resample=t_res, sampler_error=err)
    rows = []
    for cname in classifiers:
        cw = "balanced" if sname.split(":")[0] == "CW" else None
        clf = make_classifier(cname, seed, class_weight=cw)
        if clf is None:          # kNN has no class_weight -> CW undefined
            continue
        r = dict(base, classifier=cname)
        try:
            t1 = time.perf_counter()
            if len(np.unique(y_res)) < 2:
                raise ValueError("resampled training set has a single class")
            clf.fit(X_res, y_res)
            r["t_fit"] = time.perf_counter() - t1
            r.update(compute_metrics(y_te, clf.predict(X_te), scores_of(clf, X_te)))
        except Exception as e:
            r["clf_error"] = f"{type(e).__name__}: {e}"[:300]
        rows.append(r)

    sel = None
    if idx is not None and sname.split(":")[0] not in ("NONE", "CW", "SMOTE"):
        kept_maj = tr[idx[y_tr[idx] == 0]] if len(idx) else np.array([], dtype=int)
        sel = dict(tag=tag, dataset=ds, repeat=rep, fold=fold, sampler=sname,
                   kept_majority=np.sort(kept_maj).astype(np.int32))
    return rows, sel


def _done_keys(out_dir, tag):
    done = set()
    for p in glob.glob(os.path.join(out_dir, f"{tag}_*part*.parquet")):
        d = pd.read_parquet(p, columns=["dataset", "repeat", "fold", "sampler"])
        done |= set(map(tuple, d.drop_duplicates().values.tolist()))
    return done


def run_units_serial(units, data_path, folds_path, out_dir, ms_path=None, classifiers=CLASSIFIERS,
                     sampler_params=None, tag="main", shard_label="s00", chunk=50):
    """Run units serially in this process; checkpoint every `chunk` units to own parquet parts.
    (The sandbox forbids named pipes, so process pools are unavailable; parallelism comes from
    launching several independent shard processes - see launch().)"""
    os.makedirs(out_dir, exist_ok=True)
    data, folds = _get(data_path, folds_path)
    done = _done_keys(out_dir, tag)

    def key(u):
        f = folds[u[0]]["folds"][u[1]]
        return (u[0], int(f["repeat"]), int(f["fold"]), u[2])
    todo = [u for u in units if key(u) not in done]
    print(f"[{tag}/{shard_label}] {len(units)} units, {len(units) - len(todo)} done, {len(todo)} to run", flush=True)
    part = len(glob.glob(os.path.join(out_dir, f"{tag}_{shard_label}_part*.parquet")))
    buf_rows, buf_sel, n_fin, t0 = [], [], 0, time.time()
    for u in todo:
        rows, sel = run_unit(u, data_path, folds_path, ms_path, classifiers, sampler_params, tag)
        buf_rows.extend(rows)
        if sel is not None:
            buf_sel.append(sel)
        n_fin += 1
        if n_fin % chunk == 0 or n_fin == len(todo):
            stem = os.path.join(out_dir, f"{tag}_{shard_label}_part{part:04d}")
            pd.DataFrame(buf_rows).to_parquet(stem + ".parquet")
            joblib.dump(buf_sel, stem.replace("_part", "_sel") + ".joblib", compress=3)
            part += 1
            buf_rows, buf_sel = [], []
            el = time.time() - t0
            print(f"[{tag}/{shard_label}] {n_fin}/{len(todo)} units, {el/60:.1f} min, "
                  f"ETA {el/n_fin*(len(todo)-n_fin)/60:.1f} min", flush=True)
    return n_fin


def build_units(cfg):
    data, folds = _get(cfg["data_path"], cfg["folds_path"])
    names = cfg.get("datasets") or sorted(data)
    units = [(ds, k, s) for ds in names for k in range(len(folds[ds]["folds"])) for s in cfg["samplers"]]
    rng = np.random.default_rng(12345)      # deterministic shuffle -> balanced shards
    order = rng.permutation(len(units))
    return [units[i] for i in order]


def launch(cfg_path, nshards, python_exe=None, log_dir=None):
    """Start `nshards` independent shard processes; returns list of (Popen, logfile)."""
    import subprocess
    python_exe = python_exe or sys.executable
    cfg = json.load(open(cfg_path))
    log_dir = log_dir or os.path.join(cfg["out_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for i in range(nshards):
        fh = open(os.path.join(log_dir, f"{cfg.get('tag', 'main')}_shard{i:02d}.log"), "w")
        p = subprocess.Popen([python_exe, os.path.abspath(__file__), cfg_path, str(i), str(nshards)],
                             stdout=fh, stderr=subprocess.STDOUT, cwd=os.getcwd())
        procs.append((p, fh))
    return procs


def collect(out_dir, tag="main"):
    parts = sorted(glob.glob(os.path.join(out_dir, f"{tag}_*part*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


def collect_selections(out_dir, tag="main"):
    out = []
    for p in sorted(glob.glob(os.path.join(out_dir, f"{tag}_*sel*.joblib"))):
        out.extend(joblib.load(p))
    return out


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1]))
    shard, nshards = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    units = build_units(cfg)[shard::nshards]
    run_units_serial(units, cfg["data_path"], cfg["folds_path"], cfg["out_dir"], cfg.get("ms_path"),
                     cfg.get("classifiers", CLASSIFIERS), cfg.get("sampler_params"),
                     cfg.get("tag", "main"), f"s{shard:02d}", cfg.get("chunk", 50))
