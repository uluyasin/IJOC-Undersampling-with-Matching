"""Isolated re-timing of the undersamplers (response to the runtime comment of the referee).

Every timed call is preceded by an untimed warm-up call on a small synthetic set.
Part A: one training fold per dataset (repeat 0, fold 0; benchmark preprocessing), every sampler
        timed in a single process with one thread; 5 repetitions per sampler (MUS-CO: 5 if the
        first run takes < 30 s, 3 if < 120 s, else 1).
Part B: synthetic scaling, two Gaussian classes in d = 10, imbalance ratio 20, majority sizes
        1e3 ... 1e5; each run in a fresh single-threaded subprocess with a 600-s limit; a method
        is not run at larger sizes after a timeout or an error (e.g. memory).

Usage (from the package root):  python pipeline/timing_isolated.py [A|B|AB]
Outputs: results/revision/timing_isolated_benchmark.csv, results/revision/timing_isolated_scaling.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys, time, json, subprocess
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bench_runner as br
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

MS = os.path.join(ROOT, "src", "matchsampling.py")
SAMPLERS = ["RUS", "NM1", "NM2", "NM3", "TL", "ENN", "NCR", "OSS", "CNN", "IHT", "CC", "SMOTE",
            "MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
SCALING = ["RUS", "NM1", "CC", "MUS-BS", "MUS-BO", "MUS-CS", "MUS-CO"]
SIZES = [1000, 3000, 10000, 30000, 100000]
OUT = os.path.join(ROOT, "results", "revision")


def part_a():
    br._load_matchsampling(MS)
    out_csv = os.path.join(OUT, "timing_isolated_benchmark.csv")
    data = joblib.load(os.path.join(ROOT, "data", "benchmark_datasets.joblib"))
    folds = joblib.load(os.path.join(ROOT, "data", "cv_folds.joblib"))
    rows, done = [], set()
    if os.path.exists(out_csv):
        rows = pd.read_csv(out_csv).to_dict("records")
        done = {(r["dataset"], r["sampler"]) for r in rows}
    Xw, yw = synth(200, seed=1)
    for s in SAMPLERS:                                # warm-up of every sampler before timing
        br.make_sampler(s, 1, MS, {}).fit_resample(Xw, yw)
    for ds in sorted(data):
        D = data[ds]
        X, y = np.asarray(D["X"], dtype=float), np.asarray(D["y"]).astype(int)
        f = folds[ds]["folds"][0]
        tr = np.asarray(f["train_idx"])
        seed = int(folds[ds]["seed"]) * 100 + 0
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        X_tr = MinMaxScaler().fit_transform(imp.fit_transform(X[tr]))
        y_tr = y[tr]
        for s in SAMPLERS:
            if (ds, s) in done:
                continue
            times, nrep, r = [], 5, 0
            while r < nrep:
                smp = br.make_sampler(s, seed, MS, {})
                t0 = time.perf_counter()
                smp.fit_resample(X_tr, y_tr)
                t = time.perf_counter() - t0
                times.append(t)
                r += 1
                if s == "MUS-CO" and r == 1:
                    nrep = 5 if t < 30 else (3 if t < 120 else 1)
            rows.append(dict(dataset=ds, sampler=s, n_maj=int((y_tr == 0).sum()), n_min=int((y_tr == 1).sum()),
                             d=int(X_tr.shape[1]), reps=len(times), t_median=float(np.median(times)),
                             t_min=float(min(times)), t_max=float(max(times)), times=json.dumps(times)))
            pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("A", ds, flush=True)


def synth(n_maj, d=10, ir=20, seed=0):
    rng = np.random.default_rng(seed)
    p = max(10, n_maj // ir)
    X = np.vstack([rng.normal(0.0, 1.0, (n_maj, d)), rng.normal(1.0, 1.0, (p, d))])
    y = np.r_[np.zeros(n_maj, dtype=int), np.ones(p, dtype=int)]
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    return X, y


def child(sampler, n_maj, rep):
    br._load_matchsampling(MS)
    Xw, yw = synth(200, seed=1)                      # warm-up: lazy imports and first-call overheads
    br.make_sampler(sampler, 1, MS, {}).fit_resample(Xw, yw)
    X, y = synth(n_maj, seed=0)
    smp = br.make_sampler(sampler, 1000 + rep, MS, {})
    t0 = time.perf_counter()
    smp.fit_resample(X, y)
    print(json.dumps(dict(t=time.perf_counter() - t0)))


def part_b():
    out_csv = os.path.join(OUT, "timing_isolated_scaling.csv")
    rows = []
    for s in SCALING:
        stop = False
        for n in SIZES:
            if stop:
                break
            for rep in range(3 if n <= 10000 else 1):
                cmd = [sys.executable, os.path.abspath(__file__), "child", s, str(n), str(rep)]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if r.returncode == 0 and r.stdout.strip():
                        t = json.loads(r.stdout.strip().splitlines()[-1])["t"]
                        rows.append(dict(sampler=s, n_maj=n, n_min=max(10, n // 20), rep=rep, t=t, status="ok"))
                    else:
                        err = (r.stderr.strip().splitlines() or ["error"])[-1][:200]
                        rows.append(dict(sampler=s, n_maj=n, n_min=max(10, n // 20), rep=rep, t=np.nan, status="error: " + err))
                        stop = True
                except subprocess.TimeoutExpired:
                    rows.append(dict(sampler=s, n_maj=n, n_min=max(10, n // 20), rep=rep, t=np.nan, status="timeout>600s"))
                    stop = True
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print("B", s, n, rep, rows[-1]["status"], rows[-1]["t"], flush=True)
                if stop:
                    break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        what = sys.argv[1] if len(sys.argv) > 1 else "AB"
        os.makedirs(OUT, exist_ok=True)
        if "A" in what:
            part_a()
        if "B" in what:
            part_b()
