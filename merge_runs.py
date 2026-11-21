"""Merge the shard outputs of pipeline/bench_runner.py into the files read by the analysis scripts.

    results/main/main_*part*.parquet     -> results/results_main_final.parquet    (main experiment, 18 configurations)
    results/main/main_*sel*.joblib       -> results/selections_main_final.joblib  (retained majority indices per unit)
    results/ablation/abl_*part*.parquet  -> results/results_ablation.parquet      (ablations)

All three main configurations (cfg_main_baselines.json, cfg_main_proposed_bsc.json, cfg_main_musco.json) write to
results/main with the tag "main"; both ablation configurations write to results/ablation with the tag "abl".
Rows and records are concatenated in the order of the shard files (bench_runner.collect, bench_runner.collect_selections).

Usage (from the repository root): python pipeline/merge_runs.py
"""
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_runner as br  # noqa: E402


def main():
    main_rows = br.collect("results/main", "main")
    keys = ["dataset", "repeat", "fold", "sampler", "classifier"]
    assert len(main_rows) and not main_rows.duplicated(keys).any(), "no shard output in results/main, or duplicate units"
    main_rows.to_parquet("results/results_main_final.parquet")
    sel = br.collect_selections("results/main", "main")
    joblib.dump(sel, "results/selections_main_final.joblib", compress=3)
    print(f"results/results_main_final.parquet: {len(main_rows)} rows; results/selections_main_final.joblib: {len(sel)} records")
    abl = br.collect("results/ablation", "abl")
    if len(abl):
        assert not abl.duplicated(keys).any(), "duplicate units in results/ablation"
        abl.to_parquet("results/results_ablation.parquet")
        print(f"results/results_ablation.parquet: {len(abl)} rows")
    else:
        print("no ablation output found in results/ablation")


if __name__ == "__main__":
    main()
