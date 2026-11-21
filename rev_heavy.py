"""Heavy computations for the referee-revision analyses A7 and A8 (resumable, sharded).

    python rev_heavy.py plan                      -> writes rev/out/units_<task>.json
    python rev_heavy.py run <task> <shard> <nshards>
    python rev_heavy.py child-complete <dataset>  (internal: one complete-graph solve)

Tasks
  w1        A7: exact W1(uniform full training majority N, uniform retained multiset S) for 9 methods on the
            25 folds... restricted to the first CV repeat (5 folds per dataset), plus the boundary-optimal check:
            W1(capacity-weighted minority, MUS-BO subset) vs the minimum over all m-subsets (LP with majority
            capacities 1 and a zero-cost dummy source), and the same quantity for MUS-BS.
  rounds    A7/A8: coverage-arm rounds re-run with the module (same preprocessing and seed rule as the
            benchmark): per round s, r, full/partial, cost(M_t), diam(S_t), the Lemma bound
            B_t = cost(M_t)/s + diam(S_t) r (s-2r) / (s (s-r)); for MUS-CO additionally the greedy (stable
            roommates) matching cost on the SAME set S_t. The final retained set is compared with the stored
            benchmark selection.
  complete  A8: round 1 (S_1 = full training majority, repeat 0 fold 0, n <= 1200) solved on the complete graph
            (n_candidates = s - 1, the module's exact solver) vs the default k = 10 candidate graph.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import json
import time
import glob
import importlib.util
import subprocess
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_common as rc  # noqa: E402

OUT = os.path.join(HERE, "out")
W1_METHODS = ["RUS", "NM1", "NM2", "NM3", "CC", "MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
DENSE_MAX = 50_000           # use the dense LP below this many arcs, hot-started column generation above
CO_TIME_LIMIT_MAIN = 120.0   # A7: MUS-CO bound only on folds whose benchmark resampling took < 120 s
COMPLETE_MAX_N = 1200        # A8: complete-graph round-1 solve only for training majorities <= 1200
COMPLETE_TIMEOUT = 2400      # seconds per complete-graph child process

_MS = None


def ms():
    global _MS
    if _MS is None:
        p = rc.load_paths()["matchsampling.py"]
        spec = importlib.util.spec_from_file_location("matchsampling", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["matchsampling"] = mod
        spec.loader.exec_module(mod)
        _MS = mod
    return _MS


_CACHE = {}


def inputs():
    if not _CACHE:
        P = rc.load_paths()
        _CACHE["data"] = joblib.load(P["benchmark_datasets.joblib"])
        _CACHE["folds"] = joblib.load(P["cv_folds.joblib"])
        _CACHE["sel"] = rc.selection_index(joblib.load(P["selections_main_final.joblib"]))
    return _CACHE["data"], _CACHE["folds"], _CACHE["sel"]


# --------------------------------------------------------------------------------------------- planning
def plan():
    P = rc.load_paths()
    res = pd.read_parquet(P["results_main_final.parquet"], columns=["dataset", "repeat", "fold", "sampler",
                                                                     "n_maj_train", "n_min_train", "t_resample"])
    u = res.drop_duplicates(["dataset", "repeat", "fold", "sampler"])
    u = u[u["repeat"] == 0]
    os.makedirs(OUT, exist_ok=True)
    t = u.pivot_table(index=["dataset", "fold"], columns="sampler", values="t_resample")
    nm = u[u["sampler"] == "RUS"].set_index(["dataset", "fold"])[["n_maj_train", "n_min_train"]]
    w1_units, rounds_units, complete_units = [], [], []
    for (ds, fo), r in nm.iterrows():
        n, p = int(r["n_maj_train"]), int(r["n_min_train"])
        w1_units.append(dict(dataset=ds, k=int(fo), cost=float(9 * min(n * p, 4e6) + 2 * (p + 1) * n)))
        rounds_units.append(dict(dataset=ds, k=int(fo), concept="stable", cost=float(t.loc[(ds, fo), "MUS-CS"]) + 0.5,
                                 a7=True, a8=False))
        tco = float(t.loc[(ds, fo), "MUS-CO"])
        a7 = tco < CO_TIME_LIMIT_MAIN
        a8 = int(fo) == 0
        if a7 or a8:
            rounds_units.append(dict(dataset=ds, k=int(fo), concept="optimal", cost=tco + 1.0, a7=bool(a7),
                                     a8=bool(a8), t_main=tco))
        if int(fo) == 0 and n <= COMPLETE_MAX_N:
            complete_units.append(dict(dataset=ds, k=0, cost=float(n) ** 2.5))
    for name, units in [("w1", w1_units), ("rounds", rounds_units), ("complete", complete_units)]:
        json.dump(units, open(os.path.join(OUT, f"units_{name}.json"), "w"), indent=0)
        print(name, len(units))


def shard_units(task, shard, nshards):
    units = json.load(open(os.path.join(OUT, f"units_{task}.json")))
    load = np.zeros(nshards)
    mine = []
    for i in np.argsort([-u["cost"] for u in units], kind="stable"):    # longest processing time first
        s = int(np.argmin(load))
        load[s] += units[i]["cost"]
        if s == shard:
            mine.append(units[i])
    return mine


def unit_path(task, u):
    tag = f"{u['dataset']}__k{u['k']}" + (f"__{u['concept']}" if "concept" in u else "")
    return os.path.join(OUT, task, tag + ".joblib")


# --------------------------------------------------------------------------------------------- W1 (A7)
def _w1_auto(N, s_pos):
    n = len(N)
    u = np.unique(s_pos)
    size = (n - len(u)) * len(u)
    t0 = time.perf_counter()
    if size <= DENSE_MAX:
        w = rc.w1_full_vs_subset(N, s_pos)
        info = dict(solver="dense-highs", iters=1, certified=True)
    else:
        w, inf = rc.w1_full_vs_subset_hot(N, s_pos)
        info = dict(solver="cg-hot-highs", iters=inf["iters"], certified=bool(inf["certified"]))
    info["seconds"] = time.perf_counter() - t0
    return w, info


def _weighted_auto(P, q, S):
    if len(P) * len(S) <= DENSE_MAX:
        return rc.w1_weighted(P, q, S), "dense-highs"
    w, inf = rc.w1_weighted_hot(P, q, S)
    assert inf["certified"]
    return w, "cg-hot-highs"


def _best_auto(P, q, N):
    if (len(P) + 1) * len(N) <= DENSE_MAX:
        return rc.w1_best_subset(P, q, N), "dense-highs"
    w, inf = rc.w1_best_subset_hot(P, q, N)
    assert inf["certified"]
    return w, "cg-hot-highs"


def do_w1(u):
    data, folds, SI = inputs()
    M = ms()
    ds, k = u["dataset"], u["k"]
    Xtr, ytr, tr, rep, fo, seed = rc.fold_arrays(data, folds, ds, k)
    pos_maj, pos_min = np.flatnonzero(ytr == 0), np.flatnonzero(ytr == 1)
    N, P = Xtr[pos_maj], Xtr[pos_min]
    n, p = len(N), len(P)
    to_pos = np.full(int(tr.max()) + 1, -1, dtype=np.int64)
    to_pos[tr[pos_maj]] = np.arange(n)
    rows = []
    for smp in W1_METHODS:
        s = to_pos[SI[(ds, rep, fo, smp)]]
        if (s < 0).any():
            raise ValueError(f"{ds} {smp}: index outside training majority")
        w, info = _w1_auto(N, s)
        rows.append(dict(dataset=ds, repeat=rep, fold=fo, k=k, sampler=smp, n_maj=n, n_min=p, m=len(s),
                         m_unique=int(len(np.unique(s))), W1=w, **info))
    # boundary-optimal check (capacities exactly as the sampler draws them: first use of RandomState(seed))
    m = p                                         # sampling_strategy='auto' -> m = p
    q = M.capacities_for_target(p, m, np.random.RandomState(seed))
    chk = dict(dataset=ds, repeat=rep, fold=fo, k=k, n_maj=n, n_min=p, m=m, q_min=int(q.min()), q_max=int(q.max()))
    t0 = time.perf_counter()
    s_bo = to_pos[SI[(ds, rep, fo, "MUS-BO")]]
    s_bs = to_pos[SI[(ds, rep, fo, "MUS-BS")]]
    chk["W1_BO"], chk["solver_BO"] = _weighted_auto(P, q, N[s_bo])
    chk["W1_BS"], chk["solver_BS"] = _weighted_auto(P, q, N[s_bs])
    chk["W1_star"], chk["solver_star"] = _best_auto(P, q, N)
    # module's own objective (Jonker-Volgenant assignment) for reference
    owner, keep, eu = M.boundary_optimal_assignment(P, N, q)
    chk["W1_module_assignment"] = float(np.sum(eu) / m)
    chk["module_keep_equals_record"] = bool(np.array_equal(np.sort(keep), np.sort(s_bo)))
    chk["seconds"] = time.perf_counter() - t0
    return dict(rows=rows, check=chk)


# --------------------------------------------------------------------------------------------- rounds (A7/A8)
def do_rounds(u):
    data, folds, SI = inputs()
    M = ms()
    ds, k, concept = u["dataset"], u["k"], u["concept"]
    Xtr, ytr, tr, rep, fo, seed = rc.fold_arrays(data, folds, ds, k)
    pos_maj, pos_min = np.flatnonzero(ytr == 0), np.flatnonzero(ytr == 1)
    N = Xtr[pos_maj]
    n, p = len(N), len(pos_min)
    m = p                                         # sampling_strategy='auto'
    kc = M.DEFAULT_N_CANDIDATES
    rng = np.random.RandomState(seed)             # check_random_state(int) inside the sampler
    S = np.arange(n)
    rounds = []
    t_all = time.perf_counter()
    t = 0
    while len(S) > m:
        t += 1
        s = len(S)
        full_round = (s - s // 2) >= m
        r = s // 2 if full_round else s - m
        Xs = N[S]
        rec = dict(dataset=ds, repeat=rep, fold=fo, k=k, concept=concept, round=t, s=s, r=r, full=bool(full_round))
        t0 = time.perf_counter()
        if concept == "stable":
            pairs, costs = M.roommates_greedy(Xs, max_pairs=r)
            rec["cost_perfect"] = np.nan
        else:
            pairs, costs, info = M.min_weight_matching(Xs, n_candidates=kc)
            rec["cost_perfect"] = float(costs.sum())        # optimal (candidate graph) perfect matching
            rec.update(solver=info["solver"], lp_solves=int(info["lp_solves"]), cuts=int(info["cuts"]),
                       n_edges=int(info["n_edges"]))
            if not full_round:
                o = np.lexsort((pairs[:, 1], pairs[:, 0], costs))[:r]
                pairs, costs = pairs[o], costs[o]
        rec["seconds_match"] = time.perf_counter() - t0
        rec["cost_used"] = float(costs.sum())
        rec["n_pairs"] = int(len(pairs))
        first = rng.random_sample(len(pairs)) < 0.5        # keep_rule='random'
        drop = np.where(first, pairs[:, 1], pairs[:, 0])
        mask = np.ones(s, dtype=bool)
        mask[drop] = False
        t1 = time.perf_counter()
        diam = rc.diameter(Xs)
        rec["diam"] = diam
        rec["B_term1"] = rec["cost_used"] / s
        rec["B_term2"] = diam * r * (s - 2 * r) / (s * (s - r))
        rec["B"] = rec["B_term1"] + rec["B_term2"]
        if concept == "optimal":
            gp, gc = M.roommates_greedy(Xs)                  # greedy (stable) perfect matching on the SAME set
            rec["greedy_cost_perfect"] = float(gc.sum())
            rec["greedy_pairs"] = int(len(gp))
            if not full_round:
                gpr, gcr = M.roommates_greedy(Xs, max_pairs=r)
                rec["greedy_cost_used"] = float(gcr.sum())
            else:
                rec["greedy_cost_used"] = rec["greedy_cost_perfect"]
        rec["seconds_extra"] = time.perf_counter() - t1
        S = S[mask]
        rec["size_out"] = len(S)
        rounds.append(rec)
    kept = np.sort(tr[pos_maj[S]])
    smp = "MUS-CS" if concept == "stable" else "MUS-CO"
    stored = np.sort(SI[(ds, rep, fo, smp)])
    summ = dict(dataset=ds, repeat=rep, fold=fo, k=k, concept=concept, sampler=smp, n_maj=n, n_min=p,
                n_rounds=len(rounds), reproduces_record=bool(np.array_equal(kept, stored)),
                seconds_total=time.perf_counter() - t_all, a7=bool(u.get("a7", True)), a8=bool(u.get("a8", False)),
                t_main=float(u.get("t_main", np.nan)),
                bound_sum=float(sum(x["B"] for x in rounds)), bound_term1_sum=float(sum(x["B_term1"] for x in rounds)),
                bound_term2_sum=float(sum(x["B_term2"] for x in rounds)))
    return dict(rounds=rounds, summary=summ)


# --------------------------------------------------------------------------------------------- complete graph (A8)
def child_complete(ds):
    data, folds, _ = inputs()
    M = ms()
    Xtr, ytr, tr, rep, fo, seed = rc.fold_arrays(data, folds, ds, 0)
    N = Xtr[ytr == 0]
    s = len(N)
    t0 = time.perf_counter()
    pk, ck, ik = M.min_weight_matching(N, n_candidates=M.DEFAULT_N_CANDIDATES)
    t1 = time.perf_counter()
    pc, cc, ic = M.min_weight_matching(N, n_candidates=s - 1)
    t2 = time.perf_counter()
    out = dict(dataset=ds, repeat=rep, fold=fo, s=s, d=int(N.shape[1]), cost_k10=float(ck.sum()),
               cost_complete=float(cc.sum()), solver_k10=ik["solver"], solver_complete=ic["solver"],
               lp_solves_complete=int(ic["lp_solves"]), cuts_complete=int(ic["cuts"]),
               n_edges_k10=int(ik["n_edges"]), n_edges_complete=int(ic["n_edges"]),
               seconds_k10=t1 - t0, seconds_complete=t2 - t1, status="ok")
    out["rel_gap"] = (out["cost_k10"] - out["cost_complete"]) / out["cost_complete"]
    return out


def do_complete(u):
    ds = u["dataset"]
    tmp = unit_path("complete", u) + ".child.joblib"
    t0 = time.perf_counter()
    try:
        cp = subprocess.run([sys.executable, os.path.abspath(__file__), "child-complete", ds, tmp],
                            capture_output=True, text=True, timeout=COMPLETE_TIMEOUT, env=dict(os.environ))
        if cp.returncode != 0 or not os.path.exists(tmp):
            return dict(dataset=ds, status=f"error rc={cp.returncode}: {cp.stderr[-400:]}",
                        seconds_wall=time.perf_counter() - t0)
        out = joblib.load(tmp)
        os.remove(tmp)
        out["seconds_wall"] = time.perf_counter() - t0
        return out
    except subprocess.TimeoutExpired:
        return dict(dataset=ds, status=f"timeout>{COMPLETE_TIMEOUT}s", seconds_wall=time.perf_counter() - t0)


# --------------------------------------------------------------------------------------------- driver
def run(task, shard, nshards):
    units = shard_units(task, shard, nshards)
    os.makedirs(os.path.join(OUT, task), exist_ok=True)
    fn = dict(w1=do_w1, rounds=do_rounds, complete=do_complete)[task]
    todo = [u for u in units if not os.path.exists(unit_path(task, u))]
    print(f"[{task} {shard}/{nshards}] {len(units)} units, {len(todo)} to do", flush=True)
    t0 = time.time()
    for i, u in enumerate(todo):
        t1 = time.time()
        try:
            out = fn(u)
        except Exception as e:  # recorded, never silently dropped
            out = dict(error=f"{type(e).__name__}: {e}"[:500], unit=u)
        joblib.dump(out, unit_path(task, u) + ".tmp", compress=3)
        os.replace(unit_path(task, u) + ".tmp", unit_path(task, u))
        print(f"[{task} {shard}] {i + 1}/{len(todo)} {u['dataset']} k{u['k']} {u.get('concept', '')} "
              f"{time.time() - t1:.1f}s (elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "plan":
        plan()
    elif a[0] == "run":
        run(a[1], int(a[2]), int(a[3]))
    elif a[0] == "child-complete":
        joblib.dump(child_complete(a[1]), a[2])
