"""Build results/revision/sel_small.joblib (stored main-run selections of the matching
undersamplers, keyed by (dataset, repeat, fold, sampler)) from results/selections_main_final.joblib.
Only used by the row-order experiment (X2) to check that recomputed selections equal the stored ones.

Usage: python pipeline/make_sel_small.py results/selections_main_final.joblib results/revision/sel_small.joblib
"""
import sys
import joblib

src, dst = sys.argv[1], sys.argv[2]
rows = joblib.load(src)
out = {(r["dataset"], int(r["repeat"]), int(r["fold"]), r["sampler"]): r["kept_majority"]
       for r in rows if r["sampler"] in ("MUS-BS", "MUS-BO", "MUS-CS")}
joblib.dump(out, dst)
print(len(out), "selections written to", dst)
