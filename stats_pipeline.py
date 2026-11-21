#!/usr/bin/env python
"""
stats_pipeline.py -- pre-specified statistical comparison of the matching undersamplers
(MUS-BS, MUS-BO, MUS-CS, MUS-CO) against standard undersampling on the KEEL benchmark.

Re-runs end to end from the input file paths:

    python stats_pipeline.py --main results_main.parquet --ablation results_ablation.parquet \
        --registry dataset_registry.csv --complexity dataset_complexity.csv --out stats_out

Analysis plan (pre-specified; see statistics_report.md for the full description)
  1. Per (dataset, sampler, classifier): mean over the 25 folds of every metric.
  2. Comparison set = NONE + 11 standard undersamplers + 4 MUS methods (16 methods);
     SMOTE and class weighting (CW) are references, analysed separately.
  3. Per classifier x primary metric (G-mean, ROC-AUC): average ranks (ties averaged),
     Friedman chi-square (scipy, tie-corrected) and Iman-Davenport F, Nemenyi CD (alpha 0.05),
     CD diagrams (+ a pooled, descriptive panel over dataset x classifier blocks);
     Wilcoxon signed-rank (two-sided, zero_method='zsplit') of each MUS method vs each of the
     12 non-MUS methods, Holm over the 48 comparisons per classifier x metric; W/T/L with
     tie threshold |diff| < 0.001; 2x2 factorial contrasts with Holm over the 4 contrasts.
  4. Bayesian signed-rank test (baycomp, ROPE 0.01) of each MUS method vs RUS and vs the best
     non-MUS method (lowest average rank); robustness re-runs on subsets / secondary metrics;
     secondary reference comparison vs SMOTE and CW.
  5. Ablation: target ratio 1 / 1.5 / 2 and the two design variants vs their defaults.

Figures use the rcParams of the figure-style skill's apply_figure_style(frame='open',
sizes=(8, 7, 6)), replicated in apply_style() so that the CLI run is self-contained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------------------------------
# Constants (pre-specified)
# --------------------------------------------------------------------------------------
MUS = ['MUS-BS', 'MUS-BO', 'MUS-CS', 'MUS-CO']
STD = ['RUS', 'NM1', 'NM2', 'NM3', 'TL', 'ENN', 'NCR', 'OSS', 'CNN', 'IHT', 'CC']
NONMUS = ['NONE'] + STD
REFS = ['SMOTE', 'CW']
EXPECTED_SAMPLERS = NONMUS + MUS + REFS          # 18
CLFS = ['CART', 'kNN', 'SVM', 'LR', 'RF']
ABL_CLFS = ['CART', 'kNN', 'SVM']
PRIMARY = ['gmean', 'auc']
SECONDARY = ['bacc', 'f1', 'mcc', 'prauc']
ALL_METRICS = ['gmean', 'auc', 'bacc', 'f1', 'mcc', 'prauc', 'sens', 'spec', 'prec']
MLAB = {'gmean': 'G-mean', 'auc': 'ROC-AUC', 'bacc': 'Balanced accuracy', 'f1': 'F1 (minority)',
        'mcc': 'MCC', 'prauc': 'PR-AUC', 'sens': 'Sensitivity', 'spec': 'Specificity',
        'prec': 'Precision'}
MLAB_SHORT = {'gmean': 'G-mean', 'auc': 'ROC-AUC', 'bacc': 'BAcc', 'f1': 'F1', 'mcc': 'MCC',
              'prauc': 'PR-AUC'}
ALPHA = 0.05
TIE_EPS = 1e-3          # W/T/L tie threshold on the metric scale
ROPE = 0.01             # Bayesian ROPE on the metric scale
SEED = 20261115         # seed for every Bayesian posterior sample (np.random.seed before each test)
NSAMPLES = 50000        # baycomp default
PRIOR = 0.5             # baycomp default prior weight of the pseudo-observation at 0
ROUND = 12              # per-dataset means rounded to 1e-12 (absorbs float summation noise)
N_FOLDS = 25
FACTORIAL = [('MUS-BS', 'MUS-BO', 'stable vs optimal', 'boundary arm'),
             ('MUS-CS', 'MUS-CO', 'stable vs optimal', 'coverage arm'),
             ('MUS-BS', 'MUS-CS', 'boundary vs coverage', 'stable concept'),
             ('MUS-BO', 'MUS-CO', 'boundary vs coverage', 'optimal concept')]
ABL_FAMILIES = ['RUS', 'NM1', 'CC', 'MUS-BS', 'MUS-BO', 'MUS-CS']
RATIO_SUFFIX = {1.0: None, 1.5: 'r1.5', 2.0: 'r2'}
OTHER_VARIANTS = OrderedDict([('MUS-CS:keep=boundary', 'MUS-CS'), ('MUS-BS:asym', 'MUS-BS')])
VARIANT_LABEL = {'r1.5': 'ratio 1.5', 'r2': 'ratio 2', 'keep=boundary': 'keep boundary member',
                 'asym': 'asymmetric pref.'}

# colour threading: boundary arm = blues, coverage arm = oranges; stable darker, optimal lighter
COLORS = {'MUS-BS': '#08519C', 'MUS-BO': '#4292C6', 'MUS-CS': '#A63603', 'MUS-CO': '#E6550D',
          'RUS': '#000000', 'CC': '#737373', 'NM1': '#A3A3A3'}
BASE_GREY = '#404040'
MARKERS = {'MUS-BS': 'o', 'MUS-BO': 's', 'MUS-CS': 'D', 'MUS-CO': '^',
           'RUS': 'o', 'NM1': 's', 'CC': '^'}


def color_for(name):
    return COLORS.get(name, BASE_GREY)


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------
def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_table(path):
    if str(path).lower().endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def holm(p):
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return np.array([]), np.array([], dtype=bool)
    rej, padj, _, _ = multipletests(p, alpha=ALPHA, method='holm')
    return padj, rej


def wilcoxon_pair(a, b):
    """Two-sided Wilcoxon signed-rank test of a vs b (paired over datasets), zero_method='zsplit'.

    Returns statistics from the perspective of `a` (difference d = a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    wins = int(np.sum(d >= TIE_EPS))
    losses = int(np.sum(d <= -TIE_EPS))
    ties = n - wins - losses
    # matched-pairs rank-biserial correlation with the zero-split convention
    r = stats.rankdata(np.abs(d))
    rp = r[d > 0].sum() + 0.5 * r[d == 0].sum()
    rm = r[d < 0].sum() + 0.5 * r[d == 0].sum()
    r_rb = float((rp - rm) / (n * (n + 1) / 2.0))
    if np.all(d == 0):
        W, p, meth = np.nan, 1.0, 'all differences zero'
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = stats.wilcoxon(a, b, zero_method='zsplit', alternative='two-sided', method='auto')
            W, p = float(res.statistic), float(res.pvalue)
            p_ex = float(stats.wilcoxon(a, b, zero_method='zsplit', alternative='two-sided',
                                        method='exact').pvalue) if n <= 50 else np.nan
        meth = 'exact' if (n <= 50 and np.isclose(p, p_ex, rtol=0, atol=1e-15)) else 'asymptotic'
    return dict(n=n, mean_diff=float(d.mean()), median_diff=float(np.median(d)), wins=wins,
                ties=ties, losses=losses, W=W, R_plus=float(rp), R_minus=float(rm),
                p_raw=float(p), p_method=meth, r_rb=r_rb)


def verdict(sig, r_rb, pos='MUS better', neg='MUS worse'):
    if not sig:
        return 'n.s.'
    return pos if r_rb > 0 else neg


def friedman_block(X):
    """X: DataFrame (blocks x methods). Returns ranks, avg ranks, omnibus statistics."""
    N, k = X.shape
    R = pd.DataFrame(stats.rankdata(-X.values, axis=1, method='average'), index=X.index,
                     columns=X.columns)
    avg = R.mean(axis=0)
    chi2, p = stats.friedmanchisquare(*[X[c].values for c in X.columns])
    chi2 = float(chi2)
    denom = N * (k - 1) - chi2
    ff = (N - 1) * chi2 / denom if denom > 0 else np.inf
    df1, df2 = k - 1, (k - 1) * (N - 1)
    p_ff = float(stats.f.sf(ff, df1, df2)) if np.isfinite(ff) else 0.0
    q = float(stats.studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2.0))
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    # Demsar (2006) form without tie correction, recomputable from the average ranks alone
    chi2_nc = float(12.0 * N / (k * (k + 1)) * (np.sum(avg.values ** 2) - k * (k + 1) ** 2 / 4.0))
    ff_nc = (N - 1) * chi2_nc / (N * (k - 1) - chi2_nc) if N * (k - 1) - chi2_nc > 0 else np.inf
    return R, avg, dict(N=N, k=k, chi2=chi2, p_chi2=float(p), F_ID=float(ff), df1=df1, df2=df2,
                        p_ID=p_ff, q_alpha=q, CD=float(cd), chi2_nocorr=chi2_nc,
                        p_chi2_nocorr=float(stats.chi2.sf(chi2_nc, k - 1)), F_ID_nocorr=float(ff_nc),
                        p_ID_nocorr=float(stats.f.sf(ff_nc, df1, df2)) if np.isfinite(ff_nc) else 0.0)


# --------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------
REQ_COLS = ['dataset', 'repeat', 'fold', 'sampler', 'classifier'] + ALL_METRICS


def load_inputs(main_path, abl_path, reg_path, cpx_path, log):
    main = read_table(main_path)
    abl = read_table(abl_path) if abl_path else None
    reg = read_table(reg_path)
    cpx = read_table(cpx_path)
    for name, df in [('main', main)] + ([('ablation', abl)] if abl is not None else []):
        miss = [c for c in REQ_COLS if c not in df.columns]
        if miss:
            raise ValueError(f'{name}: missing columns {miss}')
    reg = reg[reg['included'].astype(bool)].copy()
    datasets = sorted(reg['name'].tolist())
    info = OrderedDict()
    info['n_rows_main'] = int(len(main))
    info['n_rows_ablation'] = int(len(abl)) if abl is not None else 0
    info['datasets_registry'] = len(datasets)
    dmain = sorted(main['dataset'].unique().tolist())
    if dmain != datasets:
        raise ValueError(f'main datasets differ from registry: '
                         f'{sorted(set(dmain) ^ set(datasets))}')
    present = [s for s in EXPECTED_SAMPLERS if s in set(main['sampler'])]
    unexpected = sorted(set(main['sampler']) - set(EXPECTED_SAMPLERS))
    missing = [s for s in EXPECTED_SAMPLERS if s not in present]
    if unexpected:
        raise ValueError(f'unexpected samplers in main file: {unexpected}')
    if any(s not in present for s in NONMUS):
        raise ValueError('a non-MUS comparison method is missing: '
                         f'{[s for s in NONMUS if s not in present]}')
    info['samplers_present'] = present
    info['samplers_missing'] = missing
    # completeness: 25 folds per (dataset, sampler, classifier); CW undefined for kNN
    cnt = main.groupby(['dataset', 'sampler', 'classifier']).size()
    info['cells_main'] = int(len(cnt))
    info['folds_per_cell'] = {int(k): int(v) for k, v in cnt.value_counts().items()}
    exp_cells = 0
    for s in present:
        exp_cells += len(datasets) * (len(CLFS) - (1 if s == 'CW' else 0))
    info['cells_expected'] = exp_cells
    if len(cnt) != exp_cells or (cnt != N_FOLDS).any():
        raise ValueError(f'incomplete main results: {len(cnt)} cells vs {exp_cells} expected; '
                         f'folds per cell {info["folds_per_cell"]}')
    nan_main = int(main[ALL_METRICS].isna().sum().sum())
    info['nan_metric_values_main'] = nan_main
    if nan_main:
        raise ValueError(f'{nan_main} NaN metric values in main results')
    if 'sampler_error' in main.columns:
        info['sampler_errors_main'] = int(main['sampler_error'].notna().sum())
        if info['sampler_errors_main']:
            raise ValueError('sampler errors present in main results')
    if abl is not None:
        ca = abl.groupby(['dataset', 'sampler', 'classifier']).size()
        info['cells_ablation'] = int(len(ca))
        info['folds_per_cell_ablation'] = {int(k): int(v) for k, v in ca.value_counts().items()}
        info['ablation_variants'] = sorted(abl['sampler'].unique().tolist())
        info['nan_metric_values_ablation'] = int(abl[ALL_METRICS].isna().sum().sum())
        if (ca != N_FOLDS).any() or info['nan_metric_values_ablation']:
            raise ValueError('incomplete ablation results')
    cpx = cpx.set_index('name').loc[datasets].reset_index()
    meta = reg.set_index('name').loc[datasets, ['family', 'IR', 'IR_group',
                                               'family_representative']].copy()
    meta['family_representative'] = meta['family_representative'].astype(bool)
    meta['single_feature_separable'] = cpx.set_index('name').loc[datasets,
                                                                  'single_feature_separable'].astype(bool)
    info['n_family_representative'] = int(meta['family_representative'].sum())
    info['n_single_feature_separable'] = int(meta['single_feature_separable'].sum())
    info['n_IR_low'] = int((meta['IR_group'] == 'low').sum())
    info['n_IR_high'] = int((meta['IR_group'] == 'high').sum())
    # IR groups must coincide with the pre-specified threshold IR < 9 vs IR >= 9
    chk = ((meta['IR'] < 9) == (meta['IR_group'] == 'low')).all()
    info['IR_group_matches_threshold_9'] = bool(chk)
    if not chk:
        log('WARNING: registry IR_group does not coincide with IR < 9; using IR < 9 threshold')
        meta['IR_group'] = np.where(meta['IR'] < 9, 'low', 'high')
    return main, abl, meta, datasets, info


def per_dataset_means(df, metrics=ALL_METRICS, extra=('t_resample', 'n_maj_res', 'n_min_res')):
    df = df.sort_values(['dataset', 'sampler', 'classifier', 'repeat', 'fold'], kind='mergesort')
    g = df.groupby(['dataset', 'sampler', 'classifier'], sort=True)
    cols = list(metrics) + [c for c in extra if c in df.columns]
    m = g[cols].mean()
    m[list(metrics)] = m[list(metrics)].round(ROUND)
    sd = g[['gmean', 'auc']].std(ddof=1).add_suffix('_fold_sd')
    m = m.join(sd)
    m['n_folds'] = g.size()
    return m.reset_index()


def matrix(means, clf, metric, methods, datasets):
    sub = means[(means['classifier'] == clf) & (means['sampler'].isin(methods))
                & (means['dataset'].isin(datasets))]
    X = sub.pivot(index='dataset', columns='sampler', values=metric)
    X = X.reindex(index=sorted(datasets), columns=list(methods))
    if X.isna().any().any():
        raise ValueError(f'missing values in matrix {clf}/{metric}')
    return X


# --------------------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------------------
def rank_analysis(means, comp, datasets, metrics, condition):
    """Average ranks + Friedman/Iman-Davenport + Nemenyi CD per classifier and pooled."""
    rows, avg_store = [], {}
    nonmus = [m for m in comp if m not in MUS]
    for met in metrics:
        blocks = []
        for clf in CLFS + ['pooled']:
            if clf == 'pooled':
                X = pd.concat(blocks, keys=CLFS, names=['classifier', 'dataset'])
            else:
                X = matrix(means, clf, met, comp, datasets)
                blocks.append(X)
            R, avg, fr = friedman_block(X)
            best = avg[nonmus].idxmin()
            pos = avg.rank(method='min')
            row = OrderedDict(condition=condition, metric=met, classifier=clf,
                              descriptive=(clf == 'pooled'), n_datasets=len(datasets), **fr,
                              best_nonmus=best, best_nonmus_avg_rank=float(avg[best]))
            for m in comp:
                row[f'rank_{m}'] = float(avg[m])
            for m in comp:
                row[f'pos_{m}'] = float(pos[m])
            rows.append(row)
            avg_store[(condition, met, clf)] = (avg, fr)
    return pd.DataFrame(rows), avg_store


def pairwise_analysis(means, comp, datasets, metrics, condition, avg_store):
    """Each MUS method vs each non-MUS method; Holm over all comparisons per clf x metric."""
    mus = [m for m in comp if m in MUS]
    nonmus = [m for m in comp if m not in MUS]
    out = []
    for met in metrics:
        for clf in CLFS:
            X = matrix(means, clf, met, comp, datasets)
            avg, fr = avg_store[(condition, met, clf)]
            best = avg[nonmus].idxmin()
            block = []
            for m in mus:
                for b in nonmus:
                    res = wilcoxon_pair(X[m].values, X[b].values)
                    row = OrderedDict(condition=condition, metric=met,
                                      metric_role='primary' if met in PRIMARY else 'secondary',
                                      classifier=clf, n_datasets=len(datasets), k_methods=len(comp),
                                      mus=m, baseline=b, is_rus=(b == 'RUS'), is_cc=(b == 'CC'),
                                      is_best=(b == best), best_nonmus=best,
                                      mus_avg_rank=float(avg[m]), baseline_avg_rank=float(avg[b]))
                    row.update(res)
                    block.append(row)
            padj, rej = holm([r['p_raw'] for r in block])
            for r, pa, rj in zip(block, padj, rej):
                r['p_holm'] = float(pa)
                r['sig_holm'] = bool(rj)
                r['holm_family_size'] = len(block)
                r['verdict'] = verdict(rj, r['r_rb'])
            out += block
    return pd.DataFrame(out)


def factorial_analysis(means, datasets, metrics, present):
    out = []
    contrasts = [c for c in FACTORIAL if c[0] in present and c[1] in present]
    for met in metrics:
        for clf in CLFS:
            X = matrix(means, clf, met, sorted({c[0] for c in contrasts} | {c[1] for c in contrasts}),
                       datasets)
            block = []
            for a, b, factor, within in contrasts:
                res = wilcoxon_pair(X[a].values, X[b].values)
                row = OrderedDict(metric=met, metric_role='primary' if met in PRIMARY else 'secondary',
                                  classifier=clf, n_datasets=len(datasets), contrast=f'{a} vs {b}',
                                  first=a, second=b, factor=factor, within=within)
                row.update(res)
                block.append(row)
            padj, rej = holm([r['p_raw'] for r in block])
            for r, pa, rj in zip(block, padj, rej):
                r['p_holm'] = float(pa)
                r['sig_holm'] = bool(rj)
                r['holm_family_size'] = len(block)
                r['verdict'] = verdict(rj, r['r_rb'], pos=f'{r["first"]} better',
                                       neg=f'{r["second"]} better')
            out += block
    return pd.DataFrame(out), contrasts


def bayes_signed_rank(x, y):
    """baycomp Bayesian signed-rank test; returns (P(x better), P(rope), P(y better)), samples."""
    from baycomp.multiple import SignedRankTest, Posterior
    np.random.seed(SEED)
    samples = SignedRankTest.sample(np.asarray(x, float), np.asarray(y, float), rope=ROPE,
                                    prior=PRIOR, nsamples=NSAMPLES)
    pl, pe, pr = Posterior(samples).probs(True)
    return (float(pl), float(pe), float(pr)), samples


def bayes_analysis(means, comp, datasets, avg_store, keep_samples):
    mus = [m for m in comp if m in MUS]
    nonmus = [m for m in comp if m not in MUS]
    out, samples_store = [], {}
    for met in PRIMARY:
        for clf in CLFS:
            X = matrix(means, clf, met, comp, datasets)
            avg, _ = avg_store[(MAIN_COND, met, clf)]
            best = avg[nonmus].idxmin()
            for m in mus:
                for role, b in [('RUS', 'RUS'), ('best non-MUS', best)]:
                    (pl, pe, pr), smp = bayes_signed_rank(X[m].values, X[b].values)
                    d = X[m].values - X[b].values
                    out.append(OrderedDict(metric=met, classifier=clf, n_datasets=len(datasets),
                                           mus=m, comparator_role=role, comparator=b,
                                           best_is_rus=(best == 'RUS'), mean_diff=float(d.mean()),
                                           median_diff=float(np.median(d)), p_mus_better=pl,
                                           p_rope=pe, p_mus_worse=pr, rope=ROPE, prior=PRIOR,
                                           nsamples=NSAMPLES, seed=SEED,
                                           bayes_verdict=bayes_verdict(pl, pe, pr)))
                    if keep_samples(met, clf, m, b):
                        samples_store[(met, clf, m, b)] = smp[:4000].copy()
    return pd.DataFrame(out), samples_store


def bayes_verdict(pl, pe, pr, thr=0.95):
    if pl >= thr:
        return 'MUS better (P>=0.95)'
    if pr >= thr:
        return 'MUS worse (P>=0.95)'
    if pe >= thr:
        return 'practically equivalent (P>=0.95)'
    lab = ['MUS better', 'equivalent', 'MUS worse'][int(np.argmax([pl, pe, pr]))]
    return f'inconclusive (most probable: {lab})'


def reference_analysis(means, comp, datasets, metrics, present):
    mus = [m for m in comp if m in MUS]
    refs = [r for r in REFS if r in present]
    out = []
    for met in metrics:
        for clf in CLFS:
            rr = [r for r in refs if not (r == 'CW' and clf == 'kNN')]
            X = matrix(means, clf, met, mus + rr, datasets)
            block = []
            for m in mus:
                for r in rr:
                    res = wilcoxon_pair(X[m].values, X[r].values)
                    row = OrderedDict(metric=met, metric_role='primary' if met in PRIMARY else 'secondary',
                                      classifier=clf, n_datasets=len(datasets), mus=m, reference=r)
                    row.update(res)
                    block.append(row)
            padj, rej = holm([r['p_raw'] for r in block])
            for r, pa, rj in zip(block, padj, rej):
                r['p_holm'] = float(pa)
                r['sig_holm'] = bool(rj)
                r['holm_family_size'] = len(block)
                r['verdict'] = verdict(rj, r['r_rb'])
            out += block
    return pd.DataFrame(out)


def robustness_extract(pw, condition_label):
    """Rows vs RUS, CC and the best non-MUS method (one row per role)."""
    rows = []
    for _, r in pw.iterrows():
        roles = []
        if r['is_rus']:
            roles.append('RUS')
        if r['is_cc']:
            roles.append('CC')
        if r['is_best']:
            roles.append('best non-MUS')
        for role in roles:
            d = r.to_dict()
            d['baseline_role'] = role
            d['condition_label'] = condition_label
            rows.append(d)
    return pd.DataFrame(rows)


def ablation_analysis(main_means, abl, datasets, metrics):
    abl_means = per_dataset_means(abl)
    base = main_means[main_means['sampler'].isin(ABL_FAMILIES)
                      & main_means['classifier'].isin(ABL_CLFS)].copy()
    comb = pd.concat([base, abl_means], ignore_index=True)
    comb['family'] = comb['sampler'].str.split(':').str[0]
    comb['variant'] = comb['sampler'].map(lambda s: s.split(':', 1)[1] if ':' in s else 'default')
    ratio_map = {'default': 1.0, 'r1.5': 1.5, 'r2': 2.0}
    comb['ratio'] = comb['variant'].map(ratio_map)
    # whether the target ratio is capped at the available majority (no majority removal)
    capped = (abl.assign(cap=abl['n_maj_res'] >= abl['n_maj_train'])
              .groupby(['sampler', 'dataset'])['cap'].any().groupby('sampler').sum())
    ratio_cfgs = [f if r == 1.0 else f'{f}:{RATIO_SUFFIX[r]}' for f in ABL_FAMILIES
                  for r in (1.0, 1.5, 2.0)]
    missing = [c for c in ratio_cfgs if c not in set(comb['sampler'])]
    if missing:
        raise ValueError(f'ablation configurations missing: {missing}')
    rows = []
    for met in metrics:
        for clf in ABL_CLFS:
            X = matrix(comb, clf, met, ratio_cfgs, datasets)
            R = pd.DataFrame(stats.rankdata(-X.values, axis=1, method='average'), index=X.index,
                             columns=X.columns)
            avg = R.mean(axis=0)
            allcfg = ratio_cfgs + [v for v in OTHER_VARIANTS if v in set(comb['sampler'])]
            Xa = matrix(comb, clf, met, allcfg, datasets)
            tests = {}
            for fam in ABL_FAMILIES:
                variants = [f'{fam}:r1.5', f'{fam}:r2'] + [v for v, f in OTHER_VARIANTS.items()
                                                          if f == fam and v in Xa.columns]
                block = []
                for v in variants:
                    res = wilcoxon_pair(Xa[v].values, Xa[fam].values)
                    block.append((v, res))
                padj, rej = holm([r['p_raw'] for _, r in block])
                for (v, res), pa, rj in zip(block, padj, rej):
                    res = dict(res)
                    res.update(p_holm=float(pa), sig_holm=bool(rj), holm_family_size=len(block),
                               verdict=verdict(rj, res['r_rb'], pos='variant better',
                                               neg='variant worse'))
                    tests[v] = res
            for cfg in allcfg:
                fam = cfg.split(':')[0]
                var = cfg.split(':', 1)[1] if ':' in cfg else 'default'
                row = OrderedDict(metric=met, classifier=clf, n_datasets=len(datasets), family=fam,
                                  config=cfg, variant=var,
                                  target_ratio={'default': 1.0, 'r1.5': 1.5, 'r2': 2.0}.get(var, 1.0),
                                  mean=float(Xa[cfg].mean()), median=float(Xa[cfg].median()),
                                  avg_rank_ratio_grid=float(avg[cfg]) if cfg in avg.index else np.nan,
                                  n_datasets_capped=int(capped.get(cfg, 0)) if var != 'default' else 0)
                if cfg in tests:
                    t = tests[cfg]
                    row.update(vs=fam, **{k: t[k] for k in ['mean_diff', 'median_diff', 'wins', 'ties',
                                                            'losses', 'W', 'p_raw', 'p_method', 'r_rb',
                                                            'p_holm', 'sig_holm', 'holm_family_size',
                                                            'verdict']})
                rows.append(row)
    return pd.DataFrame(rows), abl_means


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
STYLE_RC = {  # identical to figure-style apply_figure_style(frame='open', sizes=(8, 7, 6))
    'font.family': 'sans-serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': 0.6,
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 3,
    'ytick.major.size': 3, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': True,
    'axes.spines.bottom': True, 'axes.grid': False, 'legend.frameon': False, 'figure.dpi': 200,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.titleweight': 'normal',
    'axes.titlelocation': 'left', 'axes.labelweight': 'normal', 'lines.linewidth': 1.2,
    'patch.linewidth': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42}
FS_BASE, FS_SEC, FS_TICK = 8, 7, 6


def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update(STYLE_RC)


def check_overlaps(fig):
    """figure-style 9.1 geometric check on the drawn figure: overlapping visible text boxes, text
    over spines, text outside the figure. Tick labels of hidden axes and of ticks outside the view
    limits are not drawn and are excluded."""
    import matplotlib as mpl
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden, tl_vis = set(), {}
    for ax in fig.axes:
        ticks = set(ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both'))
        vis = set()
        if ax.axison:
            for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                lo, hi = min(lim), max(lim)
                for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels()):
                    if lo - 1e-9 <= loc <= hi + 1e-9:
                        vis.add(lab)
        hidden |= ticks - vis
        tl_vis[ax] = vis
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in hidden]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values()
              if s.get_visible() and ax.axison]
    bad = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(texts)
           for b, bb in texts[i + 1:] if ba.overlaps(bb)]
    bad += [(t.get_text(), 'spine') for t, bt in texts for s, bs in spines
            if bt.overlaps(bs) and t not in tl_vis[s.axes]]
    fb = fig.bbox
    outside = [t.get_text() for t, bt in texts
               if bt.x0 < fb.x0 - 1 or bt.y0 < fb.y0 - 1 or bt.x1 > fb.x1 + 1 or bt.y1 > fb.y1 + 1]
    return bad, outside


def cd_cliques(vals, cd):
    k = len(vals)
    out = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(i2 <= i and j <= j2 for (i2, j2) in out):
            out.append((i, j))
    return out


def pack_levels(cliques, vals, gap=0.3):
    levels, assign = [], []
    for (i, j) in cliques:
        x0, x1 = vals[i], vals[j]
        for li, lev in enumerate(levels):
            if all(x1 + gap < a or x0 - gap > b for (a, b) in lev):
                lev.append((x0, x1))
                assign.append(li)
                break
        else:
            levels.append([(x0, x1)])
            assign.append(len(levels) - 1)
    return assign, len(levels)


CD_GEOM = dict(lev0=0.20, lev_dy=0.13, dy=0.27, y_cd=0.62)


def cd_layout(avg, cd):
    order = avg.sort_values(kind='mergesort')
    vals = order.values.astype(float)
    cl = cd_cliques(vals, cd)
    assign, nlev = pack_levels(cl, vals)
    return order, cl, assign, nlev


def draw_cd(ax, avg, cd, title, nlev_common):
    g = CD_GEOM
    order, cliques, assign, _ = cd_layout(avg, cd)
    names, vals = list(order.index), order.values.astype(float)
    k = len(vals)
    lo, hi = 1, k
    span = hi - lo
    marg = 0.43 * span + 0.4
    c0 = g['lev0'] + max(nlev_common, 1) * g['lev_dy'] + 0.06
    h = (k + 1) // 2
    ax.plot([lo, hi], [0, 0], color='k', lw=0.8, solid_capstyle='butt')
    for r_ in range(lo, hi + 1):
        ax.plot([r_, r_], [0, 0.09], color='k', lw=0.6)
        ax.text(r_, 0.12, str(r_), ha='center', va='bottom', fontsize=FS_TICK)
    for r_ in np.arange(lo + 0.5, hi, 1.0):
        ax.plot([r_, r_], [0, 0.045], color='k', lw=0.5)
    yb = g['y_cd']
    ax.plot([lo, lo + cd], [yb, yb], color='k', lw=0.9, solid_capstyle='butt')
    for xx in (lo, lo + cd):
        ax.plot([xx, xx], [yb - 0.05, yb + 0.05], color='k', lw=0.9)
    ax.text(lo + cd + 0.25, yb, f'CD = {cd:.2f}', ha='left', va='center', fontsize=FS_SEC)
    for i in range(k):
        name, v = names[i], vals[i]
        col = color_for(name) if name in MUS else BASE_GREY
        is_mus = name in MUS
        if i < h:
            y = -(c0 + i * g['dy'])
            xe, xt, ha, txt = lo - 0.15, lo - 0.3, 'right', f'{name}  {v:.2f}'
        else:
            y = -(c0 + (k - 1 - i) * g['dy'])
            xe, xt, ha, txt = hi + 0.15, hi + 0.3, 'left', f'{v:.2f}  {name}'
        ax.plot([v, v, xe], [0, y, y], color=col, lw=0.9 if is_mus else 0.6)
        ax.text(xt, y, txt, ha=ha, va='center', fontsize=FS_SEC, color=col,
                fontweight='bold' if is_mus else 'normal')
    for (i, j), li in zip(cliques, assign):
        y = -(g['lev0'] + li * g['lev_dy'])
        ax.plot([vals[i] - 0.07, vals[j] + 0.07], [y, y], color='k', lw=2.2, solid_capstyle='butt')
    ax.set_xlim(lo - marg, hi + marg)
    ax.set_ylim(-(c0 + (h - 1) * g['dy']) - 0.18, 0.80)
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=FS_BASE, pad=2)


def fig_cd(avg_store, metric, comp, n_datasets, path_base):
    import matplotlib.pyplot as plt
    panels = CLFS + ['pooled']
    lay = {}
    for clf in panels:
        avg, fr = avg_store[(MAIN_COND, metric, clf)]
        lay[clf] = (avg[comp], fr)
    nlev = max(cd_layout(a, fr['CD'])[3] for a, fr in lay.values())
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.3))
    for ax, clf in zip(axes.ravel(), panels):
        avg, fr = lay[clf]
        if clf == 'pooled':
            title = f'Pooled over classifiers: {fr["N"]} blocks (descriptive)'
        else:
            title = f'{clf}: {fr["N"]} datasets, Friedman p {fmt_p_plain(fr["p_chi2"])}'
        draw_cd(ax, avg, fr['CD'], title, nlev)
    fig.text(0.995, 0.997, 'average rank: 1 = best (left end of each axis)', ha='right', va='top',
             fontsize=FS_SEC, color=BASE_GREY)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.18, wspace=0.06)
    fig.savefig(path_base + '.pdf')
    fig.savefig(path_base + '.png')
    bad, outside = check_overlaps(fig)
    plt.close(fig)
    return bad, outside


def fmt_p_plain(p):
    if p < 0.001:
        return '< 0.001'
    return f'= {p:.3f}'


def draw_simplex(ax, smp, probs, color, title):
    s3 = np.sqrt(3) / 2
    ax.plot([0, 0.5, 1, 0], [0, s3, 0, 0], color='k', lw=0.6, zorder=1)
    c = (0.5, s3 / 3)
    for m in [(0.25, s3 / 2), (0.75, s3 / 2), (0.5, 0)]:
        ax.plot([c[0], m[0]], [c[1], m[1]], color='0.65', lw=0.5, ls=(0, (2, 2)), zorder=1)
    xs = smp[:, 1] * 0.5 + smp[:, 2] * 1.0
    ys = smp[:, 1] * s3
    ax.scatter(xs[:2000], ys[:2000], s=1.2, color=color, alpha=0.25, linewidths=0, rasterized=True,
               zorder=2)
    kw = dict(fontsize=FS_SEC)
    ax.text(-0.02, -0.06, f'{probs[0]:.2f}', ha='center', va='top', **kw)
    ax.text(0.5, s3 + 0.05, f'{probs[1]:.2f}', ha='center', va='bottom', **kw)
    ax.text(1.02, -0.06, f'{probs[2]:.2f}', ha='center', va='top', **kw)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.22, s3 + 0.30)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0.5, s3 + 0.30, title, ha='center', va='top', fontsize=FS_SEC)


def fig_bayes(bayes_df, samples_store, best_mus, path_base):
    import matplotlib.pyplot as plt
    rows_spec = [('gmean', 'RUS'), ('gmean', 'best non-MUS'), ('auc', 'RUS'), ('auc', 'best non-MUS')]
    fig, axes = plt.subplots(4, 5, figsize=(7.2, 6.4))
    for ri, (met, role) in enumerate(rows_spec):
        for ci, clf in enumerate(CLFS):
            ax = axes[ri, ci]
            m = best_mus[(met, clf)]
            sub = bayes_df[(bayes_df['metric'] == met) & (bayes_df['classifier'] == clf)
                           & (bayes_df['mus'] == m) & (bayes_df['comparator_role'] == role)].iloc[0]
            smp = samples_store[(met, clf, m, sub['comparator'])]
            draw_simplex(ax, smp, (sub['p_mus_better'], sub['p_rope'], sub['p_mus_worse']),
                         color_for(m), f'{m} vs {sub["comparator"]}')
    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.075, hspace=0.05, wspace=0.05)
    for ci, clf in enumerate(CLFS):
        bb = axes[0, ci].get_position()
        fig.text((bb.x0 + bb.x1) / 2, 0.985, clf, ha='center', va='top', fontsize=FS_BASE,
                 fontweight='bold')
    labels = ['G-mean\nvs RUS', 'G-mean\nvs best non-MUS', 'ROC-AUC\nvs RUS',
              'ROC-AUC\nvs best non-MUS']
    for ri, lab in enumerate(labels):
        bb = axes[ri, 0].get_position()
        fig.text(0.008, (bb.y0 + bb.y1) / 2, lab, ha='left', va='center', fontsize=FS_BASE,
                 rotation=90, multialignment='center')
    fig.text(0.5, 0.012, 'Corners: left = MUS method better, top = practically equivalent '
             f'(|difference| < {ROPE}), right = comparator better; numbers = posterior '
             'probability of each region (Bayesian signed-rank test)', ha='center', va='bottom',
             fontsize=FS_TICK, color=BASE_GREY, wrap=True)
    fig.savefig(path_base + '.pdf')
    fig.savefig(path_base + '.png')
    bad, outside = check_overlaps(fig)
    plt.close(fig)
    return bad, outside


def fig_ablation(abl_df, n_datasets, path_base):
    import matplotlib.pyplot as plt
    rows_spec = [('gmean', 'mean', f'Mean G-mean\n({n_datasets} datasets)'),
                 ('gmean', 'avg_rank_ratio_grid', 'Average rank, G-mean\n(1 = best of 18)'),
                 ('auc', 'mean', f'Mean ROC-AUC\n({n_datasets} datasets)'),
                 ('auc', 'avg_rank_ratio_grid', 'Average rank, ROC-AUC\n(1 = best of 18)')]
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.4), sharex=True)
    grid = abl_df[abl_df['variant'].isin(['default', 'r1.5', 'r2'])]
    for ri, (met, col, ylab) in enumerate(rows_spec):
        for ci, clf in enumerate(ABL_CLFS):
            ax = axes[ri, ci]
            if ci > 0:
                ax.sharey(axes[ri, 0])
            for fam in ABL_FAMILIES:
                sub = grid[(grid['metric'] == met) & (grid['classifier'] == clf)
                           & (grid['family'] == fam)].sort_values('target_ratio')
                is_mus = fam in MUS
                ax.plot(sub['target_ratio'], sub[col], marker=MARKERS[fam], color=color_for(fam),
                        lw=1.5 if is_mus else 0.9, ms=3.6 if is_mus else 3.0,
                        ls='-' if is_mus else (0, (3, 1.5)), label=fam, zorder=3 if is_mus else 2,
                        mfc=color_for(fam) if is_mus else 'white', mew=0.8)
            if col.startswith('avg_rank') and ci == 0:
                ax.invert_yaxis()
            ax.set_xticks([1.0, 1.5, 2.0])
            ax.set_xticklabels(['1', '1.5', '2'])
            ax.margins(x=0.08, y=0.10)
            if ci == 0:
                ax.set_ylabel(ylab)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            if ri == 0:
                ax.set_title(clf, loc='left', fontsize=FS_BASE)
            if ri == len(rows_spec) - 1:
                ax.set_xlabel('Target majority:minority ratio')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.0), fontsize=FS_SEC,
               handlelength=2.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.07, hspace=0.12, wspace=0.08)
    fig.savefig(path_base + '.pdf')
    fig.savefig(path_base + '.png')
    bad, outside = check_overlaps(fig)
    plt.close(fig)
    return bad, outside


# --------------------------------------------------------------------------------------
# LaTeX tables (booktabs)
# --------------------------------------------------------------------------------------
def tp(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '--'
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def ts(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    if abs(x) < 0.5 * 10 ** (-nd):
        return f'{0:.{nd}f}'
    s = f'{x:+.{nd}f}'
    return '$-$' + s[1:] if s.startswith('-') else s


def bold_if(s, cond):
    return f'\\textbf{{{s}}}' if cond else s


def wtl(r):
    return f'{int(r["wins"])}/{int(r["ties"])}/{int(r["losses"])}'


def tex_ranks(ranks, comp, status_note):
    sub = ranks[ranks['condition'] == MAIN_COND]
    cols = CLFS + ['pooled']
    lines = []
    ncol = 1 + len(cols) * len(PRIMARY)
    lines += ['\\begin{table}[!htbp]', '\\centering',
              '\\caption{Average ranks (1 = best; ties averaged) of the %d undersampling-family '
              'methods over the %d KEEL datasets, with the Friedman test (tie-corrected '
              '$\\chi^2_F$), the Iman--Davenport statistic $F_F$ and the Nemenyi critical '
              'difference (CD, $\\alpha=0.05$). The lowest rank in each column is in bold. '
              '\\textsuperscript{\\dag}Ranks within dataset$\\times$classifier blocks pooled over '
              'the five classifiers (descriptive: blocks are not independent).%s}'
              % (len(comp), int(sub['n_datasets'].iloc[0]), status_note),
              '\\label{tab:ranks_friedman}', '\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
              '\\begin{tabular}{l' + 'r' * (ncol - 1) + '}', '\\toprule']
    hdr = ' & ' + ' & '.join(f'\\multicolumn{{{len(cols)}}}{{c}}{{{MLAB[m]}}}' for m in PRIMARY)
    lines.append(hdr + ' \\\\')
    cm = ''.join(f'\\cmidrule(lr){{{2 + i * len(cols)}-{1 + (i + 1) * len(cols)}}}'
                 for i in range(len(PRIMARY)))
    lines.append(cm)
    lines.append('Method & ' + ' & '.join([c if c != 'pooled' else 'Pooled\\textsuperscript{\\dag}'
                                           for c in cols] * len(PRIMARY)) + ' \\\\')
    lines.append('\\midrule')
    look = {(r['metric'], r['classifier']): r for _, r in sub.iterrows()}
    best = {(m, c): min(look[(m, c)][f'rank_{x}'] for x in comp) for m in PRIMARY for c in cols}
    first_mus = next((c for c in comp if c in MUS), None)
    for x in comp:
        if x == first_mus:
            lines.append('\\midrule')
        cells = []
        for m in PRIMARY:
            for c in cols:
                v = look[(m, c)][f'rank_{x}']
                cells.append(bold_if(f'{v:.2f}', np.isclose(v, best[(m, c)])))
        lines.append(f'{x} & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\midrule')
    stat_rows = [('Friedman $\\chi^2_F$', lambda r: f'{r["chi2"]:.1f}'),
                 ('$p$ ($\\chi^2_F$)', lambda r: tp(r['p_chi2'])),
                 ('Iman--Davenport $F_F$', lambda r: f'{r["F_ID"]:.2f}'),
                 ('$p$ ($F_F$)', lambda r: tp(r['p_ID'])),
                 ('Nemenyi CD', lambda r: f'{r["CD"]:.2f}'),
                 ('Blocks $N$', lambda r: f'{int(r["N"])}')]
    for lab, f in stat_rows:
        lines.append(lab + ' & ' + ' & '.join(f(look[(m, c)]) for m in PRIMARY for c in cols) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(lines) + '\n'


def tex_wilcoxon(pw, mus, status_note):
    out = []
    for met in PRIMARY:
        sub = pw[(pw['metric'] == met) & (pw['condition'] == MAIN_COND)]
        fam = int(sub['holm_family_size'].iloc[0])
        out += ['\\begin{table}[!htbp]', '\\centering',
                '\\caption{%s: two-sided Wilcoxon signed-rank tests (zero-split) of each MUS method '
                'against random undersampling (RUS), ClusterCentroids (CC) and the best non-MUS '
                'method (lowest average rank) on the per-dataset means ($N=%d$). '
                '$\\tilde\\Delta$ = median difference (MUS $-$ comparator); W/T/L = wins/ties/losses '
                'of the MUS method (tie: $|\\Delta|<0.001$); $p_\\mathrm{H}$ = Holm-adjusted $p$ '
                'over the %d MUS-vs-non-MUS comparisons per classifier; bold: '
                '$p_\\mathrm{H}<0.05$.%s}' % (MLAB[met], int(sub['n_datasets'].iloc[0]), fam,
                                              status_note),
                '\\label{tab:wilcoxon_%s}' % met, '\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
                '\\begin{tabular}{llrcrrcrlrcr}', '\\toprule',
                ' & & \\multicolumn{3}{c}{vs RUS} & \\multicolumn{3}{c}{vs CC} & '
                '\\multicolumn{4}{c}{vs best non-MUS} \\\\',
                '\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-12}',
                'Method & Clf. & $\\tilde\\Delta$ & W/T/L & $p_\\mathrm{H}$ & $\\tilde\\Delta$ & '
                'W/T/L & $p_\\mathrm{H}$ & Best & $\\tilde\\Delta$ & W/T/L & $p_\\mathrm{H}$ \\\\',
                '\\midrule']
        for mi, m in enumerate(mus):
            if mi:
                out.append('\\midrule')
            for ci, clf in enumerate(CLFS):
                s = sub[(sub['mus'] == m) & (sub['classifier'] == clf)]
                r_rus = s[s['baseline'] == 'RUS'].iloc[0]
                r_cc = s[s['baseline'] == 'CC'].iloc[0]
                r_b = s[s['is_best']].iloc[0]
                cells = [m if ci == 0 else '', clf]
                for r in (r_rus, r_cc):
                    cells += [ts(r['median_diff']), wtl(r), bold_if(tp(r['p_holm']), r['sig_holm'])]
                cells += [r_b['baseline'], ts(r_b['median_diff']), wtl(r_b),
                          bold_if(tp(r_b['p_holm']), r_b['sig_holm'])]
                out.append(' & '.join(cells) + ' \\\\')
        out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


def tex_factorial(fac, status_note):
    sub = fac[fac['metric'].isin(PRIMARY)]
    contrasts = list(OrderedDict.fromkeys(sub['contrast']))
    fam = int(sub['holm_family_size'].iloc[0]) if len(sub) else 0
    out = ['\\begin{table}[!htbp]', '\\centering',
           '\\caption{2$\\times$2 factorial contrasts among the MUS methods (Wilcoxon signed-rank, '
           'zero-split, on per-dataset means; $N=%d$). $\\tilde\\Delta$ = median difference (first '
           '$-$ second); W/T/L from the first method\'s perspective (tie: $|\\Delta|<0.001$); '
           '$p_\\mathrm{H}$ = Holm-adjusted over the %d contrasts per classifier and metric; bold: '
           '$p_\\mathrm{H}<0.05$.%s}' % (int(sub['n_datasets'].iloc[0]) if len(sub) else 0, fam,
                                         status_note),
           '\\label{tab:factorial}', '\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
           '\\begin{tabular}{llrcrrcr}', '\\toprule',
           ' & & \\multicolumn{3}{c}{G-mean} & \\multicolumn{3}{c}{ROC-AUC} \\\\',
           '\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}',
           'Contrast & Clf. & $\\tilde\\Delta$ & W/T/L & $p_\\mathrm{H}$ & $\\tilde\\Delta$ & W/T/L & '
           '$p_\\mathrm{H}$ \\\\', '\\midrule']
    for ki, con in enumerate(contrasts):
        if ki:
            out.append('\\midrule')
        for ci, clf in enumerate(CLFS):
            cells = [con if ci == 0 else '', clf]
            for met in PRIMARY:
                r = sub[(sub['contrast'] == con) & (sub['classifier'] == clf)
                        & (sub['metric'] == met)].iloc[0]
                cells += [ts(r['median_diff']), wtl(r), bold_if(tp(r['p_holm']), r['sig_holm'])]
            out.append(' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


def tex_bayes(bay, mus, status_note):
    out = []
    for met in PRIMARY:
        sub = bay[bay['metric'] == met]
        out += ['\\begin{table}[!htbp]', '\\centering',
                '\\caption{%s: Bayesian signed-rank tests (ROPE $=%.2f$, prior weight %.1f, %d '
                'posterior samples) on per-dataset means ($N=%d$). $P_{\\mathrm{MUS}}$, '
                '$P_{\\mathrm{ROPE}}$, $P_{\\mathrm{comp}}$ = posterior probabilities that the MUS '
                'method is better by more than the ROPE, that the two are practically equivalent, '
                'and that the comparator is better. Bold: probability $\\geq 0.95$.%s}'
                % (MLAB[met], ROPE, PRIOR, NSAMPLES, int(sub['n_datasets'].iloc[0]), status_note),
                '\\label{tab:bayes_%s}' % met, '\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
                '\\begin{tabular}{llrrrlrrr}', '\\toprule',
                ' & & \\multicolumn{3}{c}{vs RUS} & \\multicolumn{4}{c}{vs best non-MUS} \\\\',
                '\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}',
                'Method & Clf. & $P_{\\mathrm{MUS}}$ & $P_{\\mathrm{ROPE}}$ & $P_{\\mathrm{comp}}$ & '
                'Best & $P_{\\mathrm{MUS}}$ & $P_{\\mathrm{ROPE}}$ & $P_{\\mathrm{comp}}$ \\\\',
                '\\midrule']
        for mi, m in enumerate(mus):
            if mi:
                out.append('\\midrule')
            for ci, clf in enumerate(CLFS):
                s = sub[(sub['mus'] == m) & (sub['classifier'] == clf)]
                a = s[s['comparator_role'] == 'RUS'].iloc[0]
                b = s[s['comparator_role'] == 'best non-MUS'].iloc[0]
                cells = [m if ci == 0 else '', clf]
                for r, lab in ((a, None), (b, b['comparator'])):
                    if lab is not None:
                        cells.append(lab)
                    for key in ('p_mus_better', 'p_rope', 'p_mus_worse'):
                        cells.append(bold_if(f'{r[key]:.3f}', r[key] >= 0.95))
                out.append(' & '.join(cells) + ' \\\\')
        out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


def tex_reference(ref, mus, status_note):
    out = []
    for met in PRIMARY:
        sub = ref[ref['metric'] == met]
        out += ['\\begin{table}[!htbp]', '\\centering',
                '\\caption{%s, secondary reference comparison: each MUS method against SMOTE '
                'oversampling and cost-sensitive class weighting (CW; undefined for kNN). Wilcoxon '
                'signed-rank (zero-split) on per-dataset means ($N=%d$); $p_\\mathrm{H}$ = Holm '
                'over the MUS$\\times$reference comparisons per classifier; bold: '
                '$p_\\mathrm{H}<0.05$.%s}' % (MLAB[met], int(sub['n_datasets'].iloc[0]),
                                              status_note),
                '\\label{tab:reference_%s}' % met, '\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
                '\\begin{tabular}{llrcrrcr}', '\\toprule',
                ' & & \\multicolumn{3}{c}{vs SMOTE} & \\multicolumn{3}{c}{vs CW} \\\\',
                '\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}',
                'Method & Clf. & $\\tilde\\Delta$ & W/T/L & $p_\\mathrm{H}$ & $\\tilde\\Delta$ & W/T/L '
                '& $p_\\mathrm{H}$ \\\\', '\\midrule']
        for mi, m in enumerate(mus):
            if mi:
                out.append('\\midrule')
            for ci, clf in enumerate(CLFS):
                cells = [m if ci == 0 else '', clf]
                for rname in REFS:
                    s = sub[(sub['mus'] == m) & (sub['classifier'] == clf) & (sub['reference'] == rname)]
                    if len(s) == 0:
                        cells += ['--', '--', '--']
                    else:
                        r = s.iloc[0]
                        cells += [ts(r['median_diff']), wtl(r), bold_if(tp(r['p_holm']), r['sig_holm'])]
                out.append(' & '.join(cells) + ' \\\\')
        out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


def tex_robustness(rob, mus, conds, status_note):
    k_methods = int(rob['k_methods'].iloc[0])
    out = ['\\begin{table}[!htbp]', '\\centering',
           '\\caption{Robustness of the pairwise conclusions. Each cell gives the number of '
           'classifiers (of 5) in which the MUS method is significantly better / significantly '
           'worse than RUS (R), ClusterCentroids (C) or the best non-MUS method of that condition '
           '(B), Wilcoxon signed-rank with Holm over the full MUS-vs-non-MUS family per classifier '
           '($\\alpha=0.05$). The last block repeats the all-datasets analysis for the four '
           'secondary metrics. $\\bar{R}$ = average rank of the MUS method among the %d methods, '
           'averaged over the 5 classifiers.%s}' % (k_methods, status_note),
           '\\label{tab:robustness}', '\\scriptsize', '\\setlength{\\tabcolsep}{2.5pt}',
           '\\begin{tabular}{ll' + ('ccc' * len(mus)) + 'r' * len(mus) + '}', '\\toprule']
    out.append(' & & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{m}}}' for m in mus)
               + f' & \\multicolumn{{{len(mus)}}}{{c}}{{$\\bar{{R}}$}} \\\\')
    cm = ''.join(f'\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}' for i in range(len(mus)))
    cm += f'\\cmidrule(lr){{{3 + 3 * len(mus)}-{2 + 4 * len(mus)}}}'
    out.append(cm)
    out.append('Condition & Metric & ' + ' & '.join(['R', 'C', 'B'] * len(mus)) + ' & '
               + ' & '.join(m.replace('MUS-', '') for m in mus) + ' \\\\')
    blocks = [(conds[c][0], [(c, m) for m in PRIMARY]) for c in conds]
    blocks.append((conds[list(conds)[0]][0], [(list(conds)[0], m) for m in SECONDARY]))
    for label, items in blocks:
        out.append('\\midrule')
        for ii, (cond, met) in enumerate(items):
            s = rob[(rob['condition'] == cond) & (rob['metric'] == met)]
            if len(s) == 0:
                continue
            cells = [label if ii == 0 else '', MLAB_SHORT[met]]
            for m in mus:
                for role in ('RUS', 'CC', 'best non-MUS'):
                    q = s[(s['mus'] == m) & (s['baseline_role'] == role)]
                    w = int((q['verdict'] == 'MUS better').sum())
                    l = int((q['verdict'] == 'MUS worse').sum())
                    cells.append(f'{w}/{l}')
            for m in mus:
                q = s[(s['mus'] == m) & (s['baseline_role'] == 'RUS')]
                cells.append(f'{q["mus_avg_rank"].mean():.2f}')
            out.append(' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


def tex_ablation(abl_df, status_note):
    tests = abl_df[abl_df['variant'] != 'default']
    out = ['\\begin{table}[!htbp]', '\\centering',
           '\\caption{Ablation: mean difference (variant $-$ default) of the per-dataset means '
           '($N=%d$) for the target majority:minority ratio (default 1; capped at the available '
           'majority) and the two design variants. Bold: Holm-adjusted Wilcoxon $p<0.05$ (Holm '
           'within each method family, classifier and metric). Keep boundary member: the coverage arm '
           'keeps the pair member closer to the minority class (default: random member); asymmetric '
           'pref.: class-conditional Mahalanobis preferences in MUS-BS (default: symmetric). The '
           'last column gives the number of datasets on which the ratio cap binds (no majority '
           'removal) in at least one fold.%s}'
           % (int(abl_df['n_datasets'].iloc[0]), status_note),
           '\\label{tab:ablation}', '\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
           '\\begin{tabular}{ll' + 'r' * (len(ABL_CLFS) * len(PRIMARY)) + 'r}', '\\toprule',
           ' & & ' + ' & '.join(f'\\multicolumn{{{len(ABL_CLFS)}}}{{c}}{{{MLAB[m]}}}'
                                for m in PRIMARY) + ' & \\\\',
           ''.join(f'\\cmidrule(lr){{{3 + i * len(ABL_CLFS)}-{2 + (i + 1) * len(ABL_CLFS)}}}'
                   for i in range(len(PRIMARY))),
           'Family & Variant & ' + ' & '.join(ABL_CLFS * len(PRIMARY)) + ' & Capped \\\\',
           '\\midrule']
    for fi, fam in enumerate(ABL_FAMILIES):
        vs = [v for v in OrderedDict.fromkeys(tests[tests['family'] == fam]['config'])]
        if fi:
            out.append('\\midrule')
        for vi, v in enumerate(vs):
            var = v.split(':', 1)[1]
            cells = [fam if vi == 0 else '', VARIANT_LABEL.get(var, var)]
            for met in PRIMARY:
                for clf in ABL_CLFS:
                    r = tests[(tests['config'] == v) & (tests['metric'] == met)
                              & (tests['classifier'] == clf)].iloc[0]
                    cells.append(bold_if(ts(r['mean_diff']), bool(r['sig_holm'])))
            r0 = tests[(tests['config'] == v)].iloc[0]
            cells.append(str(int(r0['n_datasets_capped'])) if var.startswith('r') else '--')
            out.append(' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


# --------------------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------------------
def fp(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'NA'
    return f'{p:.2e}' if p < 0.001 else f'{p:.3f}'


def f3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'NA'
    return f'{x:+.4f}'


def md_table(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def build_report(ctx):
    info, comp, mus, conds = ctx['info'], ctx['comp'], ctx['mus'], ctx['conds']
    ranks, pw_main, rob, fac = ctx['ranks'], ctx['pw_main'], ctx['rob'], ctx['fac']
    bay, ref, abl_df = ctx['bay'], ctx['ref'], ctx['abl_df']
    L = []
    A = L.append
    A('# Statistical analysis report: matching undersamplers (MUS) vs standard undersampling '
      '(KEEL benchmark)\n')
    if ctx['final']:
        A('**Status: FINAL** - all 18 samplers present in the main results file.\n')
    else:
        A(f'**Status: PROVISIONAL** - samplers missing from the main results file: '
          f'{", ".join(info["samplers_missing"])}. Every number below is provisional.\n')
    A(f'Generated {ctx["timestamp"]} by `stats_pipeline.py` (runtime {ctx["runtime"]:.0f} s). Every '
      'number in this report is written programmatically from the CSV tables saved with it; each '
      'block names its source table and row filter.\n')
    # 1 inputs
    A('## 1. Inputs and data checks\n')
    A(md_table(pd.DataFrame([dict(role=k, file=v['file'], sha256_16=v['sha256'][:16], rows=v['rows'])
                             for k, v in ctx['inputs'].items()])))
    A('')
    A(f'- Datasets: {info["datasets_registry"]} (registry `included == True`), identical to the '
      f'datasets in the main file.')
    A(f'- Main file: {info["cells_main"]} (dataset, sampler, classifier) cells, folds per cell '
      f'{info["folds_per_cell"]}, NaN metric values {info["nan_metric_values_main"]}, sampler errors '
      f'{info.get("sampler_errors_main", "NA")}. CW is undefined for kNN (no rows, as expected).')
    A(f'- Samplers present ({len(info["samplers_present"])}): {", ".join(info["samplers_present"])}.')
    if info.get('cells_ablation'):
        A(f'- Ablation file: {info["cells_ablation"]} cells, folds per cell '
          f'{info["folds_per_cell_ablation"]}, variants: {", ".join(info["ablation_variants"])}.')
    A(f'- Registry IR_group coincides with the pre-specified threshold IR < 9: '
      f'{info["IR_group_matches_threshold_9"]}.\n')
    A(md_table(pd.DataFrame([dict(condition=c, description=v[2], n_datasets=len(v[1]))
                             for c, v in conds.items()])))
    A('')
    # 2 methods
    A('## 2. Methods (pre-specified plan; operational details in 2.9)\n')
    A(f'1. **Aggregation.** For each (dataset, sampler, classifier) the mean over the {N_FOLDS} folds '
      '(5 x 5 repeated stratified CV) of every metric; primary metrics G-mean and ROC-AUC, secondary '
      'balanced accuracy, F1 (minority), MCC and PR-AUC.')
    A(f'2. **Comparison set.** {len(comp)} undersampling-family methods: NONE, the 11 standard '
      f'undersamplers (RUS, NM1, NM2, NM3, TL, ENN, NCR, OSS, CNN, IHT, CC) and the MUS methods '
      f'present ({", ".join(mus)}). SMOTE and CW are references (Section 3.9 only).')
    A('3. **Ranks and omnibus tests.** Per classifier x metric: ranks within each dataset (1 = '
      'highest metric; ties averaged), average ranks, Friedman chi-square (scipy.stats.'
      'friedmanchisquare, tie-corrected) and Iman-Davenport F = (N-1)chi2/(N(k-1)-chi2) with '
      '(k-1, (k-1)(N-1)) df; Nemenyi critical difference CD = q_alpha sqrt(k(k+1)/(6N)) with q_alpha '
      '= studentized-range quantile(0.95, k, inf)/sqrt(2). Pooled panel: ranks within the '
      'dataset x classifier blocks, average over all blocks, labelled descriptive (blocks not '
      'independent).')
    A(f'4. **Pairwise tests.** Each MUS method vs each of the 12 non-MUS methods: two-sided Wilcoxon '
      f'signed-rank test on the per-dataset means (scipy.stats.wilcoxon, zero_method="zsplit", '
      f'method="auto": exact null distribution when n <= 50 without ties/zeros, otherwise the '
      f'normal approximation with tie correction; the method used is recorded per row). Holm '
      f'correction over all MUS-vs-non-MUS comparisons within each classifier x metric '
      f'({len(mus) * 12} comparisons). Differences are MUS minus comparator; W/T/L counts with tie '
      f'if |difference| < {TIE_EPS}. Also reported: matched-pairs rank-biserial correlation '
      f'(zero-split convention), whose sign gives the direction of a significant result.')
    A('5. **2x2 factorial contrasts.** Stable vs optimal within each arm (MUS-BS vs MUS-BO; MUS-CS vs '
      'MUS-CO) and boundary vs coverage within each concept (MUS-BS vs MUS-CS; MUS-BO vs MUS-CO), '
      'Wilcoxon (zsplit) with Holm over the contrasts available within each classifier x metric.')
    A(f'6. **Bayesian signed-rank test** (baycomp {ctx["versions"].get("baycomp", "")}; Benavoli et '
      f'al. 2014/2017) on per-dataset means, ROPE = {ROPE}, prior weight {PRIOR}, {NSAMPLES} '
      f'posterior samples, numpy seed {SEED} set before every test. Each MUS method vs RUS and vs '
      'the best non-MUS method (lowest average rank among the 12 non-MUS methods) of that '
      'classifier x metric. Reported probabilities are baycomp\'s: the share of posterior samples '
      'in which each region (MUS better by more than the ROPE / practically equivalent / comparator '
      'better) is the most probable. Verdict threshold 0.95.')
    A(f'7. **Robustness.** The rank analysis and the full {len(mus) * 12}-comparison Wilcoxon-Holm family are '
      f're-run unchanged on (a) the {len(conds["family_rep"][1])} pre-specified family '
      'representatives, (b) the datasets '
      'without a single-feature-separable (ceiling) flag, (c) IR < 9 and IR >= 9 separately, and '
      '(d) the four secondary metrics on all datasets; the rows vs RUS, CC and the best non-MUS '
      'method of that condition are extracted.')
    A('8. **Ablation.** Classifiers CART, kNN, SVM. Target majority:minority ratio 1 (main run), 1.5 '
      'and 2 (capped at the available majority) for RUS, NM1, CC, MUS-BS, MUS-BO, MUS-CS: mean metric '
      'and average rank among the 18 ratio configurations; Wilcoxon (zsplit) of each variant '
      '(ratio 1.5, ratio 2, MUS-CS keep=boundary, MUS-BS asymmetric preferences) vs its default, '
      'Holm within each method family x classifier x metric.')
    A('9. **Operational details not fixed by the plan text (declared):** (i) Friedman chi-square '
      'includes the standard tie correction (scipy), and the Iman-Davenport F is computed from it; '
      'the uncorrected Demsar (2006) values, which can be recomputed from the average ranks alone, '
      'are given in `stats_ranks_friedman.csv` (columns chi2_nocorr, p_chi2_nocorr, F_ID_nocorr, '
      'p_ID_nocorr); (ii) per-dataset means are rounded to 1e-12 '
      'before ranking to absorb floating-point summation noise (tie threshold for W/T/L is 1e-3, so '
      'this is immaterial); (iii) in each robustness condition the "best non-MUS method" is '
      're-selected as the lowest average rank within that condition, and the Holm family is the '
      'full set of MUS-vs-non-MUS comparisons of that condition (same family definition as the '
      'main analysis); (iv) MUS-CO ablation variants were not run (lead decision; k-sensitivity '
      'covered by a separate cost-gap sweep); (v) the ablation ratio-1 reference is the main run '
      '(identical folds and seeds).\n')
    v = ctx['versions']
    A('Software: ' + ', '.join(f'{k} {v[k]}' for k in v) + '.\n')
    # 3 results
    A('## 3. Results\n')
    A('### 3.1 Omnibus tests\n')
    A('Source: `stats_ranks_friedman.csv`, condition = all.\n')
    s = ranks[ranks['condition'] == 'all'].copy()
    rows = []
    for _, r in s.iterrows():
        avg = {m: r[f'rank_{m}'] for m in comp}
        bm = min(mus, key=lambda m: avg[m]) if mus else None
        rows.append(OrderedDict(metric=r['metric'], classifier=r['classifier'], N=int(r['N']),
                                k=int(r['k']), chi2_F=f'{r["chi2"]:.2f}', p_chi2=fp(r['p_chi2']),
                                F_ID=f'{r["F_ID"]:.3f}', df=f'{int(r["df1"])}, {int(r["df2"])}',
                                p_ID=fp(r['p_ID']), CD=f'{r["CD"]:.3f}',
                                best_nonMUS=f'{r["best_nonmus"]} ({r["best_nonmus_avg_rank"]:.2f})',
                                best_MUS=f'{bm} ({avg[bm]:.2f}, position {int(r["pos_" + bm])})'
                                if bm else 'NA'))
    A(md_table(pd.DataFrame(rows)))
    A('\nPooled rows are descriptive. Position = rank of the method\'s average rank among the '
      f'{len(comp)} methods (1 = best).\n')
    A('### 3.2 Average ranks (1 = best)\n')
    for met in PRIMARY + SECONDARY:
        sm = s[s['metric'] == met].set_index('classifier')
        t = pd.DataFrame({clf: [f'{sm.loc[clf, "rank_" + m]:.2f}' for m in comp]
                          for clf in CLFS + ['pooled']}, index=comp)
        t.insert(0, 'method', comp)
        A(f'**{MLAB[met]}** (source: `stats_ranks_friedman.csv`, condition = all, metric = {met})\n')
        A(md_table(t))
        A('')
    A('### 3.3 MUS methods vs RUS, CC and the best non-MUS method (primary metrics)\n')
    A('Source: `stats_wilcoxon_holm.csv` (condition = all; rows with baseline = RUS, CC, or '
      'is_best = True). p_Holm is adjusted over the full MUS-vs-non-MUS family of each classifier x '
      'metric.\n')
    rows = []
    for met in PRIMARY:
        for m in mus:
            for clf in CLFS:
                q = pw_main[(pw_main['metric'] == met) & (pw_main['mus'] == m)
                            & (pw_main['classifier'] == clf)]
                for role, sel in (('RUS', q['is_rus']), ('CC', q['is_cc']), ('best', q['is_best'])):
                    r = q[sel].iloc[0]
                    rows.append(OrderedDict(metric=met, MUS=m, clf=clf,
                                            comparator=r['baseline'] if role != 'best'
                                            else f'{r["baseline"]} (best)',
                                            mean_diff=f3(r['mean_diff']), median_diff=f3(r['median_diff']),
                                            WTL=f'{r["wins"]}/{r["ties"]}/{r["losses"]}',
                                            W=f'{r["W"]:.1f}', p_raw=fp(r['p_raw']),
                                            p_Holm=fp(r['p_holm']), r_rb=f'{r["r_rb"]:+.3f}',
                                            verdict=r['verdict']))
    A(md_table(pd.DataFrame(rows)))
    A('')
    A('### 3.4 Full MUS-vs-non-MUS family: number of Holm-significant results\n')
    A('Source: `stats_wilcoxon_holm.csv` (condition = all). Cell = number of the 12 non-MUS methods '
      'the MUS method beats / loses to significantly (Holm, alpha 0.05).\n')
    rows = []
    for met in PRIMARY + SECONDARY:
        for m in mus:
            d = OrderedDict(metric=met, MUS=m)
            for clf in CLFS:
                q = pw_main[(pw_main['metric'] == met) & (pw_main['mus'] == m)
                            & (pw_main['classifier'] == clf)]
                d[clf] = f'{int((q["verdict"] == "MUS better").sum())}/{int((q["verdict"] == "MUS worse").sum())}'
            rows.append(d)
    A(md_table(pd.DataFrame(rows)))
    A('')
    sigb = pw_main[(pw_main['verdict'] == 'MUS better') & pw_main['metric'].isin(PRIMARY)]
    if len(sigb):
        A('Holm-significant **MUS better** results (primary metrics): ' + '; '.join(
            f'{r.mus} > {r.baseline} ({r.classifier}, {r.metric}, median diff {r.median_diff:+.4f}, '
            f'p_Holm {fp(r.p_holm)})' for r in sigb.itertuples()) + '.\n')
    else:
        A('No Holm-significant **MUS better** result on the primary metrics.\n')
    A('### 3.5 2x2 factorial contrasts\n')
    A('Source: `stats_factorial_2x2.csv`. Difference = first minus second method.\n')
    rows = []
    for met in PRIMARY + SECONDARY:
        for r in fac[fac['metric'] == met].itertuples():
            rows.append(OrderedDict(metric=met, contrast=r.contrast, factor=f'{r.factor} | {r.within}',
                                    clf=r.classifier, mean_diff=f3(r.mean_diff),
                                    median_diff=f3(r.median_diff), WTL=f'{r.wins}/{r.ties}/{r.losses}',
                                    p_raw=fp(r.p_raw), p_Holm=fp(r.p_holm), verdict=r.verdict))
    A(md_table(pd.DataFrame(rows)))
    A('')
    A('### 3.6 Bayesian signed-rank tests\n')
    A('Source: `stats_bayes.csv`.\n')
    rows = []
    for r in bay.itertuples():
        rows.append(OrderedDict(metric=r.metric, MUS=r.mus, clf=r.classifier,
                                comparator=f'{r.comparator} ({r.comparator_role})',
                                mean_diff=f3(r.mean_diff), P_MUS_better=f'{r.p_mus_better:.3f}',
                                P_ROPE=f'{r.p_rope:.3f}', P_MUS_worse=f'{r.p_mus_worse:.3f}',
                                verdict=r.bayes_verdict))
    A(md_table(pd.DataFrame(rows)))
    A('')
    vc = bay.groupby(['comparator_role', 'bayes_verdict']).size().reset_index(name='n')
    A('Verdict counts (source: `stats_bayes.csv`):\n')
    A(md_table(vc))
    A('')
    A('### 3.7 Robustness\n')
    A('Source: `stats_robustness.csv` (one row per MUS x comparator role x classifier x metric x '
      'condition). Cell = classifiers (of 5) with a Holm-significant MUS win / loss; mean rank = '
      'average rank of the MUS method averaged over the 5 classifiers.\n')
    rows = []
    for c, (tl, ds, ml) in conds.items():
        mets = PRIMARY + SECONDARY if c == 'all' else PRIMARY
        for met in mets:
            for m in mus:
                q = rob[(rob['condition'] == c) & (rob['metric'] == met) & (rob['mus'] == m)]
                d = OrderedDict(condition=c, n=len(ds), metric=met, MUS=m)
                for role in ('RUS', 'CC', 'best non-MUS'):
                    qq = q[q['baseline_role'] == role]
                    d[f'vs {role}'] = (f'{int((qq["verdict"] == "MUS better").sum())}/'
                                       f'{int((qq["verdict"] == "MUS worse").sum())}')
                bests = sorted(set(q[q['baseline_role'] == 'best non-MUS']['baseline']))
                d['best non-MUS (per clf)'] = ', '.join(
                    f'{clf}:{q[(q["baseline_role"] == "best non-MUS") & (q["classifier"] == clf)]["baseline"].iloc[0]}'
                    for clf in CLFS)
                d['mean rank'] = f'{q[q["baseline_role"] == "RUS"]["mus_avg_rank"].mean():.2f}'
                rows.append(d)
    A(md_table(pd.DataFrame(rows)))
    A('')
    sub_rows = rob[rob['condition'] != MAIN_COND].drop_duplicates(
        ['condition', 'metric', 'classifier', 'mus', 'baseline'])
    ct = sub_rows.groupby(['condition', 'change_type']).size().unstack(fill_value=0)
    ct.insert(0, 'condition', ct.index)
    A('Verdict in each subset condition compared with the all-datasets verdict for the same MUS '
      'method, comparator and classifier (primary metrics; unique MUS x comparator x classifier x '
      'metric rows; source: `stats_robustness.csv`, column change_type). "Lost significance" is '
      'expected from the smaller N of the subsets.\n')
    A(md_table(ct.reset_index(drop=True)))
    A('')
    rev = sub_rows[sub_rows['change_type'].isin(['gained significance', 'reversed'])]
    if len(rev):
        A('Rows that gained significance or reversed direction:\n')
        A(md_table(rev[['condition', 'metric', 'classifier', 'mus', 'baseline',
                        'verdict_all_same_baseline', 'verdict', 'median_diff', 'p_holm']].rename(
            columns={'verdict_all_same_baseline': 'verdict_all', 'verdict': 'verdict_subset'})
            .assign(median_diff=lambda d: d['median_diff'].map(f3),
                    p_holm=lambda d: d['p_holm'].map(fp))))
        A('')
    else:
        A('No subset verdict gained significance or reversed direction.\n')
    A('### 3.8 Secondary metrics\n')
    A('Covered by Section 3.4 (counts), Section 3.7 (condition = all, secondary metrics) and '
      '`stats_wilcoxon_holm.csv` (metric_role = secondary).\n')
    A('### 3.9 Reference comparison: MUS vs SMOTE and CW\n')
    A('Source: `stats_reference_smote_cw.csv`; Holm over MUS x reference comparisons per classifier x '
      'metric (CW undefined for kNN).\n')
    rows = []
    for r in ref[ref['metric'].isin(PRIMARY)].itertuples():
        rows.append(OrderedDict(metric=r.metric, MUS=r.mus, clf=r.classifier, reference=r.reference,
                                mean_diff=f3(r.mean_diff), median_diff=f3(r.median_diff),
                                WTL=f'{r.wins}/{r.ties}/{r.losses}', p_Holm=fp(r.p_holm),
                                verdict=r.verdict))
    A(md_table(pd.DataFrame(rows)))
    A('')
    A('### 3.10 Ablation\n')
    if abl_df is not None:
        A('Source: `stats_ablation.csv`. Mean of the per-dataset means; avg rank among the 18 ratio '
          'configurations (6 families x 3 ratios).\n')
        g = abl_df[abl_df['variant'].isin(['default', 'r1.5', 'r2'])]
        rows = []
        for met in PRIMARY:
            for fam in ABL_FAMILIES:
                d = OrderedDict(metric=met, family=fam)
                for clf in ABL_CLFS:
                    for rr in (1.0, 1.5, 2.0):
                        x = g[(g['metric'] == met) & (g['family'] == fam) & (g['classifier'] == clf)
                              & (g['target_ratio'] == rr)].iloc[0]
                        d[f'{clf} r{rr:g}'] = f'{x["mean"]:.4f} ({x["avg_rank_ratio_grid"]:.2f})'
                rows.append(d)
        A(md_table(pd.DataFrame(rows)))
        A('')
        t = abl_df[abl_df['variant'] != 'default']
        rows = []
        for r in t[t['metric'].isin(PRIMARY)].itertuples():
            rows.append(OrderedDict(metric=r.metric, config=r.config, clf=r.classifier,
                                    mean_diff=f3(r.mean_diff), median_diff=f3(r.median_diff),
                                    WTL=f'{int(r.wins)}/{int(r.ties)}/{int(r.losses)}',
                                    p_Holm=fp(r.p_holm), verdict=r.verdict,
                                    capped_datasets=int(r.n_datasets_capped)))
        A('Variant vs default (Wilcoxon zsplit, Holm within family x classifier x metric):\n')
        A(md_table(pd.DataFrame(rows)))
        A('')
    A('### 3.11 Figures\n')
    for f, d in ctx['figures'].items():
        A(f'- `{f}.pdf` / `{f}.png`: {d}')
    A('')
    A('## 4. Quotable summary statements\n')
    for sline in ctx['summary_lines']:
        A(f'- {sline}')
    A('')
    return '\n'.join(L) + '\n'


def summary_lines(ctx):
    """Auto-generated, number-carrying sentences, each with its source."""
    comp, mus, ranks, pw, bay, fac, rob = (ctx['comp'], ctx['mus'], ctx['ranks'], ctx['pw_main'],
                                           ctx['bay'], ctx['fac'], ctx['rob'])
    out = []
    s = ranks[(ranks['condition'] == 'all') & (ranks['classifier'] != 'pooled')
              & ranks['metric'].isin(PRIMARY)]
    out.append(f'Friedman tests over the {len(comp)} methods reject equal average ranks in '
               f'{int((s["p_chi2"] < ALPHA).sum())} of {len(s)} classifier x primary-metric settings '
               f'(largest p = {fp(s["p_chi2"].max())}; Iman-Davenport largest p = {fp(s["p_ID"].max())}); '
               f'Nemenyi CD = {s["CD"].iloc[0]:.3f} for N = {int(s["N"].iloc[0])}, k = {int(s["k"].iloc[0])} '
               '[stats_ranks_friedman.csv, condition = all].')
    for met in PRIMARY:
        for m in mus:
            ss = s[s['metric'] == met]
            parts = ', '.join(f'{r["classifier"]} {r["rank_" + m]:.2f} (pos. {int(r["pos_" + m])})'
                              for _, r in ss.iterrows())
            out.append(f'{MLAB[met]} average rank of {m}: {parts} [stats_ranks_friedman.csv].')
    for met in PRIMARY:
        for m in mus:
            for role, col in (('RUS', 'is_rus'), ('CC', 'is_cc'), ('the best non-MUS method', 'is_best')):
                q = pw[(pw['metric'] == met) & pw[col] & (pw['mus'] == m)]
                nb = int((q['verdict'] == 'MUS better').sum())
                nw = int((q['verdict'] == 'MUS worse').sum())
                det = '; '.join(
                    f'{r["classifier"]}' + (f' ({r["baseline"]})' if col == 'is_best' else '')
                    + f' median diff {r["median_diff"]:+.4f}, W/T/L {r["wins"]}/{r["ties"]}/{r["losses"]}, '
                    f'p_Holm {fp(r["p_holm"])}' for _, r in q.iterrows())
                out.append(f'{MLAB[met]}, {m} vs {role}: Holm-significantly better in {nb} and worse '
                           f'in {nw} of {len(q)} classifiers ({det}) [stats_wilcoxon_holm.csv, '
                           f'condition = all, metric = {met}, mus = {m}, {col} = True].')
        q = pw[pw['metric'] == met]
        out.append(f'{MLAB[met]}, full family: ' + '; '.join(
            f'{m} significantly better than {int((q[q["mus"] == m]["verdict"] == "MUS better").sum())} '
            f'and worse than {int((q[q["mus"] == m]["verdict"] == "MUS worse").sum())} of the '
            f'{int((q["mus"] == m).sum())} MUS-vs-non-MUS comparisons (12 methods x 5 classifiers)'
            for m in mus) + ' [stats_wilcoxon_holm.csv].')
    for met in PRIMARY:
        q = fac[fac['metric'] == met]
        for con in OrderedDict.fromkeys(q['contrast']):
            qq = q[q['contrast'] == con]
            out.append(f'{MLAB[met]}, {con}: Holm-significant in {int(qq["sig_holm"].sum())} of '
                       f'{len(qq)} classifiers; median differences '
                       + ', '.join(f'{r.classifier} {r.median_diff:+.4f}' for r in qq.itertuples())
                       + ' [stats_factorial_2x2.csv].')
    for met in PRIMARY:
        for m in mus:
            for role in ('RUS', 'best non-MUS'):
                q = bay[(bay['metric'] == met) & (bay['comparator_role'] == role) & (bay['mus'] == m)]
                det = '; '.join(f'{r.classifier}' + (f' ({r.comparator})' if role != 'RUS' else '')
                                + f' {r.p_mus_better:.3f}/{r.p_rope:.3f}/{r.p_mus_worse:.3f}'
                                for r in q.itertuples())
                vc = q['bayes_verdict'].value_counts().to_dict()
                out.append(f'{MLAB[met]}, Bayesian {m} vs {role}, P(MUS better)/P(ROPE)/P(MUS worse): '
                           f'{det}; verdicts: ' + ', '.join(f'{k} x{v}' for k, v in vc.items())
                           + ' [stats_bayes.csv].')
    sub_rows = rob[rob['condition'] != MAIN_COND].drop_duplicates(
        ['condition', 'metric', 'classifier', 'mus', 'baseline'])
    ct = sub_rows['change_type'].value_counts().to_dict()
    out.append(f'Robustness: over the {sub_rows["condition"].nunique()} subset conditions, the '
               f'{len(sub_rows)} unique MUS x comparator x classifier x primary-metric verdicts '
               'compare with the all-datasets verdicts as: ' + ', '.join(f'{k} {v}' for k, v in ct.items())
               + ' [stats_robustness.csv, change_type].')
    return out


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
MAIN_COND = 'all'


def versions():
    from importlib import metadata
    v = OrderedDict(python=platform.python_version())
    for pkg in ['numpy', 'pandas', 'scipy', 'statsmodels', 'baycomp', 'matplotlib']:
        try:
            v[pkg] = metadata.version(pkg)
        except Exception:
            v[pkg] = 'unknown'
    return v


def write_text(path, txt):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(txt)


def run(main_path, abl_path, reg_path, cpx_path, outdir, log=print, make_figures=True):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)

    def P(name):
        return os.path.join(outdir, name)

    main, abl, meta, datasets, info = load_inputs(main_path, abl_path, reg_path, cpx_path, log)
    present, missing = info['samplers_present'], info['samplers_missing']
    comp = [m for m in NONMUS + MUS if m in present]
    mus = [m for m in MUS if m in present]
    final = len(missing) == 0
    status_note = '' if final else f' PROVISIONAL: {", ".join(missing)} missing.'
    log(f'status: {"FINAL" if final else "PROVISIONAL"}; missing {missing}; k = {len(comp)}')

    means = per_dataset_means(main)
    role = {**{m: 'non-MUS' for m in NONMUS}, **{m: 'MUS' for m in MUS},
            **{m: 'reference' for m in REFS}}
    sup = means.merge(meta.reset_index().rename(columns={'name': 'dataset'}), on='dataset',
                      how='left')
    sup.insert(2, 'sampler_role', sup['sampler'].map(role))
    sorder = {s: i for i, s in enumerate(EXPECTED_SAMPLERS)}
    corder = {c: i for i, c in enumerate(CLFS)}
    sup = sup.sort_values(['dataset', 'classifier', 'sampler'],
                          key=lambda c: c.map(sorder) if c.name == 'sampler'
                          else (c.map(corder) if c.name == 'classifier' else c)).reset_index(drop=True)
    lead = ['dataset', 'family', 'IR', 'IR_group', 'family_representative',
            'single_feature_separable', 'classifier', 'sampler', 'sampler_role', 'n_folds']
    sup = sup[lead + [c for c in sup.columns if c not in lead]]
    sup.to_csv(P('stats_per_dataset_means.csv'), index=False)

    nD = len(datasets)
    ds_rep = [d for d in datasets if meta.loc[d, 'family_representative']]
    ds_noc = [d for d in datasets if not meta.loc[d, 'single_feature_separable']]
    ds_low = [d for d in datasets if meta.loc[d, 'IR_group'] == 'low']
    ds_high = [d for d in datasets if meta.loc[d, 'IR_group'] == 'high']
    conds = OrderedDict([
        ('all', (f'All ({nD})', datasets, f'all {nD} datasets')),
        ('family_rep', (f'One per family ({len(ds_rep)})', ds_rep,
                        f'{len(ds_rep)} pre-specified family representatives')),
        ('no_ceiling', (f'Excl.\\ ceiling ({len(ds_noc)})', ds_noc,
                        f'{len(ds_noc)} datasets without the single-feature-separable flag')),
        ('IR_low', (f'IR $<$ 9 ({len(ds_low)})', ds_low, f'{len(ds_low)} datasets with IR < 9')),
        ('IR_high', (f'IR $\\geq$ 9 ({len(ds_high)})', ds_high,
                     f'{len(ds_high)} datasets with IR >= 9'))])

    rank_parts, avg_store, pw_parts = [], {}, []
    for c, (_, ds, _) in conds.items():
        mets = PRIMARY + SECONDARY if c == MAIN_COND else PRIMARY
        rr, st = rank_analysis(means, comp, ds, mets, c)
        rank_parts.append(rr)
        avg_store.update(st)
        pw_parts.append(pairwise_analysis(means, comp, ds, mets, c, avg_store))
    ranks = pd.concat(rank_parts, ignore_index=True)
    ranks.to_csv(P('stats_ranks_friedman.csv'), index=False)
    pw_all = pd.concat(pw_parts, ignore_index=True)
    pw_main = pw_all[pw_all['condition'] == MAIN_COND].reset_index(drop=True)
    pw_main.to_csv(P('stats_wilcoxon_holm.csv'), index=False)
    log(f'ranks + pairwise done ({time.time() - t0:.0f} s)')

    rob = robustness_extract(pw_all, '')
    rob['condition_label'] = rob['condition'].map({c: v[2] for c, v in conds.items()})
    ref_verdict = pw_main.set_index(['metric', 'classifier', 'mus', 'baseline'])['verdict']
    keys = list(zip(rob['metric'], rob['classifier'], rob['mus'], rob['baseline']))
    rob['verdict_all_same_baseline'] = [ref_verdict.get(k, np.nan) for k in keys]
    rob['best_nonmus_all'] = [avg_store[(MAIN_COND, m, c)][0][[x for x in comp if x not in MUS]].idxmin()
                              for m, c in zip(rob['metric'], rob['classifier'])]
    rob['changed_vs_all'] = (rob['verdict'] != rob['verdict_all_same_baseline']) & \
        rob['verdict_all_same_baseline'].notna()

    def _ctype(va, vs):
        if not isinstance(va, str):
            return 'not comparable'
        if va == vs:
            return 'unchanged'
        if va == 'n.s.':
            return 'gained significance'
        if vs == 'n.s.':
            return 'lost significance'
        return 'reversed'
    rob['change_type'] = [_ctype(va, vs) if c != MAIN_COND else 'reference'
                          for va, vs, c in zip(rob['verdict_all_same_baseline'], rob['verdict'],
                                               rob['condition'])]
    lead = ['condition', 'condition_label', 'n_datasets', 'metric', 'metric_role', 'classifier',
            'mus', 'baseline_role', 'baseline', 'best_nonmus', 'best_nonmus_all']
    rob = rob[lead + [c for c in rob.columns if c not in lead]]
    rob.to_csv(P('stats_robustness.csv'), index=False)

    fac, contrasts = factorial_analysis(means, datasets, PRIMARY + SECONDARY, present)
    fac.to_csv(P('stats_factorial_2x2.csv'), index=False)

    best_mus = {(met, clf): avg_store[(MAIN_COND, met, clf)][0][mus].idxmin()
                for met in PRIMARY for clf in CLFS}
    bay, smp = bayes_analysis(means, comp, datasets, avg_store,
                              lambda met, clf, m, b: best_mus[(met, clf)] == m)
    bay.to_csv(P('stats_bayes.csv'), index=False)
    log(f'bayes done ({time.time() - t0:.0f} s)')

    ref = reference_analysis(means, comp, datasets, PRIMARY + SECONDARY, present)
    ref.to_csv(P('stats_reference_smote_cw.csv'), index=False)

    abl_df = None
    if abl is not None:
        abl_df, abl_means = ablation_analysis(means, abl, datasets, PRIMARY + SECONDARY)
        abl_df.to_csv(P('stats_ablation.csv'), index=False)
        abl_means.to_csv(P('stats_ablation_per_dataset_means.csv'), index=False)

    figs = OrderedDict()
    checks = OrderedDict()
    if make_figures:
        apply_style()
        checks['fig_cd_gmean'] = fig_cd(avg_store, 'gmean', comp, nD, P('fig_cd_gmean'))
        checks['fig_cd_auc'] = fig_cd(avg_store, 'auc', comp, nD, P('fig_cd_auc'))
        checks['fig_bayes_simplex'] = fig_bayes(bay, smp, best_mus, P('fig_bayes_simplex'))
        figs['fig_cd_gmean'] = ('Nemenyi CD diagrams, G-mean: one panel per classifier + pooled '
                                '(descriptive) panel.')
        figs['fig_cd_auc'] = 'Nemenyi CD diagrams, ROC-AUC (same layout).'
        figs['fig_bayes_simplex'] = ('Posterior simplex plots (Bayesian signed-rank) of the best-ranked '
                                     'MUS method of each classifier x primary metric vs RUS and vs the '
                                     'best non-MUS method.')
        if abl_df is not None:
            checks['fig_ablation_ratio'] = fig_ablation(abl_df, nD, P('fig_ablation_ratio'))
            figs['fig_ablation_ratio'] = ('Mean metric and average rank vs target ratio for the six '
                                          'ablated families (CART, kNN, SVM).')

    write_text(P('stats_ranks_friedman.tex'), tex_ranks(ranks, comp, status_note))
    write_text(P('stats_wilcoxon_holm.tex'), tex_wilcoxon(pw_main, mus, status_note))
    write_text(P('stats_factorial_2x2.tex'), tex_factorial(fac, status_note))
    write_text(P('stats_bayes.tex'), tex_bayes(bay, mus, status_note))
    write_text(P('stats_robustness.tex'), tex_robustness(rob, mus, conds, status_note))
    write_text(P('stats_reference_smote_cw.tex'), tex_reference(ref, mus, status_note))
    if abl_df is not None:
        write_text(P('stats_ablation.tex'), tex_ablation(abl_df, status_note))

    inputs = OrderedDict()
    for k, pth, df in [('main results', main_path, main), ('ablation results', abl_path, abl),
                       ('dataset registry', reg_path, None), ('complexity table', cpx_path, None)]:
        if pth:
            inputs[k] = dict(file=os.path.basename(pth), sha256=sha256(pth),
                             rows=int(len(df)) if df is not None else int(len(read_table(pth))))
    ctx = dict(info=info, comp=comp, mus=mus, conds=conds, ranks=ranks, pw_main=pw_main, rob=rob,
               fac=fac, bay=bay, ref=ref, abl_df=abl_df, final=final, inputs=inputs,
               versions=versions(), figures=figs,
               timestamp=time.strftime('%Y-%m-%d %H:%M:%S'), runtime=time.time() - t0)
    ctx['summary_lines'] = summary_lines(ctx)
    write_text(P('statistics_report.md'), build_report(ctx))
    manifest = OrderedDict(status='FINAL' if final else 'PROVISIONAL', samplers_missing=missing,
                           generated=ctx['timestamp'], runtime_s=round(time.time() - t0, 1),
                           inputs=inputs, data_checks=info, versions=ctx['versions'],
                           parameters=dict(ALPHA=ALPHA, TIE_EPS=TIE_EPS, ROPE=ROPE, SEED=SEED,
                                           NSAMPLES=NSAMPLES, PRIOR=PRIOR, ROUND=ROUND),
                           conditions={c: len(v[1]) for c, v in conds.items()},
                           figure_checks={k: dict(overlaps=v[0], outside=v[1])
                                          for k, v in checks.items()},
                           outputs=sorted(os.listdir(outdir)))
    write_text(P('run_manifest.json'), json.dumps(manifest, indent=2, default=str))
    log(f'done in {time.time() - t0:.0f} s -> {outdir}')
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--main', required=True, help='per-fold main results (.parquet or .csv)')
    ap.add_argument('--ablation', default=None, help='per-fold ablation results')
    ap.add_argument('--registry', required=True, help='dataset registry CSV')
    ap.add_argument('--complexity', required=True, help='dataset complexity CSV')
    ap.add_argument('--out', default='stats_out', help='output directory')
    ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    run(a.main, a.ablation, a.registry, a.complexity, a.out, make_figures=not a.no_figures)


if __name__ == '__main__':
    main()
