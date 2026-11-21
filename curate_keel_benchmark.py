#!/usr/bin/env python
"""Curate the KEEL imbalanced benchmark bundled in imbalanced-databases 0.1.1.

Outputs (written to --out): benchmark_datasets.joblib, dataset_registry.csv,
dataset_complexity.csv, cv_folds.joblib, dataset_table.tex.

Usage:  python curate_keel_benchmark.py --out OUTDIR [--skip-complexity]
Tested: Python 3.11, numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, scikit-learn 1.9.1,
        problexity 0.5.11, imbalanced-databases 0.1.1 (Windows, conda env `imbml`).

Rules (see data_curation_notes.md):
 (i)   KEEL origin (loader citation == package 'keel' BibTeX); bundled dataset-level
       duplicates removed;  (ii) binary;  (iii) n_min >= 10;
 (iv)  per-family cap = the cap whose total lands in [50, 60] (largest such cap);
       within a capped family keep the datasets closest to `cap` targets equally
       spaced in log(IR) between the family's min and max IR (greedy, ties -> name).
"""
import argparse, json, os, pkgutil, re, time, warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import problexity as px
import sklearn
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

import imbalanced_databases._imbalanced_databases as M

BASE_SEED = 20261115          # CV seed = BASE_SEED + dataset_index
SEED_CX = 20261115            # np.random.seed before every complexity measure
MIN_MINORITY = 10
CAP_RANGE = (50, 60)
MISSING_TOKENS = {'?', '<null>', ''}
NAME_FIX = {'habarman': 'haberman', 'flaref': 'flare_f'}
KEEL_DISPLAY_FIX = {'new_thyroid1': 'new-thyroid1', 'new_thyroid2': 'new-thyroid2'}
FAMILY_PREFIX = ['winequality-red', 'winequality-white', 'kr-vs-k', 'new-thyroid', 'page-blocks', 'led7digit',
                 'lymphography', 'abalone', 'car', 'cleveland', 'dermatology', 'ecoli', 'flare', 'glass', 'haberman',
                 'iris', 'kddcup', 'pima', 'poker', 'segment', 'shuttle', 'vehicle', 'vowel', 'wisconsin', 'yeast', 'zoo']
KRNN_ORIGIN = {
    'ada': 'agnostic-learning/performance-prediction challenge (Guyon et al.) file ada_train',
    'hiva': 'agnostic-learning/performance-prediction challenge file hiva_train',
    'sylva': 'agnostic-learning/performance-prediction challenge file sylva_train',
    'cm1': 'NASA/PROMISE software-defect data (cm1.arff)', 'kc1': 'NASA/PROMISE software-defect data (kc1.arff)',
    'pc1': 'NASA/PROMISE software-defect data (pc1.arff)', 'german': 'UCI Statlog German credit (german.data)',
    'hepatitis': 'UCI hepatitis (hepatitis.data)', 'hypothyroid': 'UCI thyroid disease, hypothyroid.data',
    'satimage': 'UCI Statlog Landsat satellite (sat.trn/sat.tst), class 4 vs rest',
    'spectf': 'UCI SPECTF heart (SPECTF.train/test)',
    'glass': 'UCI glass (glass.data), class 3 vs rest; unexported loader',
    'vehicle': 'UCI Statlog vehicle (xa?.dat), van vs rest; unexported loader'}
# Known loader problems in imbalanced-databases 0.1.1 (asserted below: the cross-check must find exactly these)
LOADER_ISSUE = {
    'load_abalone19': "loader raises ValueError: the 32 positive labels carry a leading space (' positive'); read from the bundled .dat with whitespace stripped",
    'load_ecoli_0_vs_1': "loader bug: label written to a new lowercase 'class' column while the original 'Class' label column stays in the data as a feature (target leak); leaked column dropped. In this file 'positive' marks the MAJORITY class (143 vs 77), so y=1 is the 'negative' label",
    'load_new_thyroid2': "loader defined but not exported by the package (__all__); bundled file is an exact row permutation of new_thyroid1",
    'load_glass': "loader defined but not exported by the package (__all__)",
    'load_vehicle': "loader defined but not exported by the package (__all__)"}


# ----------------------------------------------------------------------------- loading / parsing
def rawtext(rel):
    return pkgutil.get_data('imbalanced_databases', rel).decode('utf-8', errors='replace')


def parse_keel(rel):
    """KEEL .dat -> (relation, attrs, rows); attrs = [{'name','kind' ('numeric'|'nominal'),'cats'}], last = class."""
    t = rawtext(rel).splitlines()
    attrs, relation, data_start = [], None, None
    for i, line in enumerate(t):
        s = line.strip()
        if not s or s.startswith('%'):
            continue
        low = s.lower()
        if low.startswith('@relation'):
            relation = s.split(None, 1)[1].strip()
        elif low.startswith('@attribute'):
            body = s[len('@attribute'):].strip()
            if '{' in body:
                attrs.append(dict(name=body[:body.index('{')].strip(), kind='nominal',
                                  cats=[c.strip() for c in body[body.index('{') + 1: body.rindex('}')].split(',')]))
            else:
                m = re.match(r"^(\S+)\s+(real|integer|numeric)\s*(\[.*\])?\s*$", body, flags=re.I)
                assert m, (rel, s)
                attrs.append(dict(name=m.group(1), kind='numeric', cats=None, decl=m.group(2).lower()))
        elif low.startswith('@input') or low.startswith('@output'):
            pass
        elif low.startswith('@data'):
            data_start = i + 1
            break
    rows = [[v.strip() for v in line.strip().split(',')] for line in t[data_start:]
            if line.strip() and not line.strip().startswith('%')]
    assert set(map(len, rows)) == {len(attrs)}, rel
    return relation, attrs, rows


def loader_table():
    src = open(M.__file__, encoding='utf-8').read().splitlines()
    starts = [i for i, l in enumerate(src) if re.match(r'^def load_\w+\(', l)]
    out = []
    for k, i0 in enumerate(starts):
        j = next((i for i in range(i0 + 1, len(src)) if src[i].startswith('def ')), len(src))
        body = '\n'.join(src[i0:j])
        name = re.match(r'^def (\w+)', src[i0]).group(1)
        if name == 'load_artificial_data':
            continue
        files = re.findall(r"'(data/[^']+)'", body)
        cit = re.findall(r"citations\['(\w+)'\]", body)
        out.append(dict(loader=name, file=files[0] if files else None, citation_key=cit[0] if cit else None))
    return pd.DataFrame(out).sort_values('loader').reset_index(drop=True)


def to_str(v):
    return v.decode().strip() if isinstance(v, bytes) else v


def crosscheck(ld, attrs, rows):
    """Compare loader output (encode=False) with the direct parse. Returns a discrepancy string or ''."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d = getattr(M, ld)(return_X_y=False, encode=False)
    except Exception as e:
        return f'error: {type(e).__name__}'
    X, y, fn = d['data'], d['target'], list(d['feature_names'])
    in_names = [a['name'] for a in attrs[:-1]]
    probs = []
    extra = [f for f in fn if f not in in_names]
    if extra:
        probs.append(f'extra columns {extra}')
    if X.shape[0] != len(rows):
        probs.append('row count')
    for j, a in enumerate(attrs[:-1]):
        if a['name'] not in fn:
            probs.append(f"missing column {a['name']}"); continue
        col, filev = X[:, fn.index(a['name'])], [r[j] for r in rows]
        ok = (np.allclose(col.astype(float), np.array(filev, float), equal_nan=True) if a['kind'] == 'numeric'
              else all(to_str(c) == f for c, f in zip(col, filev)))
        if not ok:
            probs.append(f"values differ in {a['name']}")
    if not np.array_equal(np.array([bool(v) for v in y]), np.array([r[-1] == 'positive' for r in rows])):
        probs.append('labels differ')
    return '; '.join(probs)


# ----------------------------------------------------------------------------- encoding
def encode_keel(attrs, rows):
    cols, names, types, dropped, numcode = [], [], [], [], []
    for j, a in enumerate(attrs[:-1]):
        vals = [r[j] for r in rows]
        if a['kind'] == 'numeric':
            cols.append(np.array([np.nan if s in MISSING_TOKENS else float(s) for s in vals], dtype=np.float64))
            names.append(a['name']); types.append('numeric')
        else:
            miss = np.array([s in MISSING_TOKENS for s in vals])
            present = set(vals)
            obs = [c for c in a['cats'] if c in present]                      # header order
            if all(re.fullmatch(r'-?\d+(\.\d+)?', c) for c in a['cats']):
                numcode.append(a['name'])
            if len(obs) <= 1:
                dropped.append(f"{a['name']} (nominal, single observed category)"); continue
            for c in ([obs[1]] if len(obs) == 2 else obs):                    # binary -> one indicator
                v = np.array([1.0 if s == c else 0.0 for s in vals], dtype=np.float64); v[miss] = np.nan
                cols.append(v); names.append(f"{a['name']}={c}"); types.append('onehot')
    X = np.column_stack(cols) if cols else np.empty((len(rows), 0))
    keep = []
    for k in range(X.shape[1]):
        col = X[:, k]
        if np.unique(col[~np.isnan(col)]).size <= 1:
            dropped.append(f"{names[k]} ({types[k]}, constant)")
        else:
            keep.append(k)
    X = X[:, keep]; names = [names[k] for k in keep]; types = [types[k] for k in keep]
    ypos = np.array([r[-1] == 'positive' for r in rows])
    nmiss = sum(r[j] in MISSING_TOKENS for r in rows for j in range(len(attrs) - 1))
    return dict(X=X, ypos=ypos, names=names, types=types, dropped=dropped, nominal_numcode=numcode,
                missing_raw=nmiss, d_raw=len(attrs) - 1, n_nominal_raw=sum(a['kind'] == 'nominal' for a in attrs[:-1]),
                source='KEEL')


def encode_generic(data, target):
    df = pd.DataFrame(data).map(lambda v: v.decode().strip() if isinstance(v, bytes) else (v.strip() if isinstance(v, str) else v))
    df = df.replace({'?': np.nan, '': np.nan})
    cols, names, types, dropped, n_nom = [], [], [], [], 0
    for c in df.columns:
        s = df[c]
        num = pd.to_numeric(s, errors='coerce')
        if num.notna().sum() == s.notna().sum():
            cols.append(num.to_numpy(np.float64)); names.append(str(c)); types.append('numeric')
        else:
            n_nom += 1
            miss = s.isna().to_numpy(); obs = sorted(s.dropna().astype(str).unique())
            if len(obs) <= 1:
                dropped.append(str(c)); continue
            for cat in ([obs[1]] if len(obs) == 2 else obs):
                v = (s.astype(str) == cat).to_numpy(np.float64); v[miss] = np.nan
                cols.append(v); names.append(f"{c}={cat}"); types.append('onehot')
    X = np.column_stack(cols)
    keep = [k for k in range(X.shape[1]) if np.unique(X[~np.isnan(X[:, k]), k]).size > 1]
    dropped += [names[k] for k in range(X.shape[1]) if k not in keep]
    X = X[:, keep]; names = [names[k] for k in keep]; types = [types[k] for k in keep]
    yv = pd.Series(list(target)).astype(str)
    minority_label = yv.value_counts().index[-1]
    return dict(X=X, ypos=(yv == minority_label).to_numpy(), names=names, types=types, dropped=dropped,
                nominal_numcode=[], missing_raw=int(df.isna().sum().sum()), d_raw=df.shape[1], n_nominal_raw=n_nom,
                source='KRNN collection (non-KEEL)', minority_label=minority_label)


def family_of(display):
    s = display.lower()
    for p in FAMILY_PREFIX:
        if s.startswith(p):
            return p
    raise ValueError(display)


# ----------------------------------------------------------------------------- complexity
def n1_ref(X, y):
    """N1 (Lorena et al. 2019): fraction of instances incident to an MST edge joining different classes."""
    D = cdist(X, X)
    pos = D[D > 0]
    eps = (pos.min() * 1e-3) if pos.size else 1e-12
    D[(D == 0) & ~np.eye(len(X), dtype=bool)] = eps          # keep exact duplicates as (near-)zero edges
    T = minimum_spanning_tree(D).tocoo()
    cross = y[T.row] != y[T.col]
    return np.unique(np.r_[T.row[cross], T.col[cross]]).size / len(y)


def n2_ref(X, y):
    """N2 (Lorena et al. 2019): r/(1+r), r = sum nearest same-class dist / sum nearest other-class dist."""
    D = cdist(X, X); np.fill_diagonal(D, np.inf)
    same = (y[:, None] == y[None, :])
    intra = np.where(same, D, np.inf).min(1); extra = np.where(~same, D, np.inf).min(1)
    r = intra.sum() / extra.sum()
    return r / (1 + r)


MEASURES = {'F1': px.f1, 'F1v': px.f1v, 'F2': px.f2, 'F3': px.f3, 'F4': px.f4,
            'L1': px.l1, 'L2': px.l2, 'L3': px.l3,
            'N1': n1_ref, 'N2': n2_ref, 'N3': px.n3, 'N4': px.n4, 'LSC': px.lsc,
            'N1_problexity': px.n1, 'N2_problexity': px.n2,
            'T2': px.t2, 'T3': px.t3, 'T4': px.t4, 'C1': px.c1, 'C2': px.c2}


def prep_full(X):
    return MinMaxScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(X))


def complexity(X, y):
    res, tim = {}, {}
    for m, f in MEASURES.items():
        np.random.seed(SEED_CX)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res[m] = float(f(X, y.astype(int)))
        tim[m] = time.perf_counter() - t0
    return res, tim


def typology(X, y):
    _, ind = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(X).kneighbors()
    same = (y[ind] == y[:, None]).sum(1)
    smin, smaj = same[y == 1], same[y == 0]
    return dict(safe=np.mean(smin >= 4), borderline=np.mean((smin == 2) | (smin == 3)), rare=np.mean(smin == 1),
                outlier=np.mean(smin == 0), n_safe=int((smin >= 4).sum()), n_borderline=int(((smin == 2) | (smin == 3)).sum()),
                n_rare=int((smin == 1).sum()), n_outlier=int((smin == 0).sum()),
                maj_overlap_share=np.mean(smaj <= 2), n_maj_overlap=int((smaj <= 2).sum()))


def ir_span_select(sub, c):
    lir = np.log(sub.IR.to_numpy()); names = sub.name.to_numpy()
    targets = np.linspace(lir.min(), lir.max(), c)
    chosen = []
    for t in targets:
        chosen.append(min((abs(lir[i] - t), names[i]) for i in range(len(sub)) if names[i] not in chosen)[1])
    return chosen, np.exp(targets)


def tex_escape(s):
    return s.replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-complexity', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    LT = loader_table()
    exported = {a for a in M.__all__ if a.startswith('load_')}
    keel = LT[LT.citation_key == 'keel']
    folder = {r.loader: r.file.split('/')[1] for r in LT.itertuples() if r.file}
    keel_file = dict(zip(keel.loader, keel.file))

    enc, parsed, found_issues = {}, {}, {}
    for ld, rel in keel_file.items():
        relation, attrs, rows = parse_keel(rel)
        parsed[ld] = rows
        disc = crosscheck(ld, attrs, rows)
        if disc:
            found_issues[ld] = disc
        enc[ld] = encode_keel(attrs, rows)
    for ld in LT.loader[LT.citation_key == 'krnn']:
        d = getattr(M, ld)(return_X_y=False, encode=False)
        enc[ld] = encode_generic(d['data'], d['target'])
    # the cross-check must reproduce exactly the documented KEEL loader problems
    assert set(found_issues) == {'load_abalone19', 'load_ecoli_0_vs_1'}, found_issues
    assert found_issues['load_ecoli_0_vs_1'].startswith("extra columns ['Class']")
    assert set(LT.loader) - exported == {'load_glass', 'load_new_thyroid2', 'load_vehicle'}

    for ld, e in enc.items():
        if e['source'] == 'KEEL':
            e['minority_label'] = 'positive' if e['ypos'].sum() < (~e['ypos']).sum() else 'negative'
            e['y'] = (e['ypos'] if e['minority_label'] == 'positive' else ~e['ypos']).astype(np.int8)
        else:
            e['y'] = e['ypos'].astype(np.int8)

    reg = []
    for ld, e in sorted(enc.items()):
        base = ld[len('load_'):]
        name = NAME_FIX.get(base, base)
        if e['source'] == 'KEEL':
            disp = KEEL_DISPLAY_FIX.get(folder[ld], folder[ld]); fam = family_of(disp)
        else:
            disp, fam = name, name
        X, y = e['X'], e['y']
        n_min = int(y.sum()); n_maj = int(len(y) - n_min)
        dfX = pd.DataFrame(X)
        g = pd.DataFrame({'k': pd.util.hash_pandas_object(dfX, index=False).values, 'y': y})
        nun = g.groupby('k')['y'].nunique()
        conf = set(nun.index[nun > 1])
        reg.append(dict(name=name, keel_name=disp, loader=ld, in_110_inventory=(ld in exported), source=e['source'],
                        citation_key=('keel' if e['source'] == 'KEEL' else 'krnn'),
                        origin_note=('KEEL imbalanced repository' if e['source'] == 'KEEL' else KRNN_ORIGIN.get(name, '')),
                        family=fam, n=len(y), d_raw=e['d_raw'], n_nominal_raw=e['n_nominal_raw'], d=X.shape[1],
                        n_numeric=e['types'].count('numeric'), n_onehot=e['types'].count('onehot'), n_min=n_min,
                        n_maj=n_maj, IR=n_maj / n_min, IR_group=('low' if n_maj / n_min < 9 else 'high'),
                        minority_label=e.get('minority_label'), missing_cells=int(e['missing_raw']),
                        missing_cells_encoded=int(np.isnan(X).sum()), duplicate_rows=int(dfX.duplicated().sum()),
                        conflicting_dup_groups=len(conf), conflicting_dup_rows=int(g.k.isin(conf).sum()),
                        dropped_columns='; '.join(e['dropped']), nominal_integer_coded='; '.join(e['nominal_numcode']),
                        n_classes=len(np.unique(y))))
    R = pd.DataFrame(reg)

    # --- inclusion rules
    R['exclusion_reason'] = ''
    nk = R.source != 'KEEL'
    R.loc[nk, 'exclusion_reason'] = ('rule (i) non-KEEL source: package cites the KRNN benchmark (Zhang, Tari & Cheriet 2017, '
                                     'Pattern Recognit. 62:33-44); data: ' + R.loc[nk, 'origin_note'])
    sig = {}
    for ld, rows in parsed.items():
        sig.setdefault(frozenset(Counter(tuple(r) for r in rows).items()), []).append(ld)
    for group in sig.values():
        if len(group) > 1:
            keep = sorted(group, key=lambda l: (l not in exported, l))[0]
            for ld in group:
                if ld != keep:
                    nm, kn = ld[len('load_'):], keep[len('load_'):]
                    txt = f'rule (i) duplicate: bundled {nm} file is an exact row permutation of {kn} (identical (x,y) multiset)'
                    if ld not in exported:
                        txt += '; loader not exported by the package'
                    R.loc[(R.loader == ld) & (R.exclusion_reason == ''), 'exclusion_reason'] = txt
    m = (R.n_classes != 2) & (R.exclusion_reason == '')
    R.loc[m, 'exclusion_reason'] = 'rule (ii) not binary'
    m = (R.n_min < MIN_MINORITY) & (R.exclusion_reason == '')
    R.loc[m, 'exclusion_reason'] = R.loc[m, 'n_min'].map(lambda v: f'rule (iii) fewer than 10 minority instances (n_min={v})')
    elig = R[R.exclusion_reason == '']
    fam_counts = elig.family.value_counts()
    totals = {c: int(np.minimum(fam_counts, c).sum()) for c in range(1, int(fam_counts.max()) + 1)}
    cap = max(c for c, t in totals.items() if CAP_RANGE[0] <= t <= CAP_RANGE[1])
    selected, sel_notes = [], {}
    for fam, cnt in fam_counts.items():
        sub = elig[elig.family == fam].sort_values(['IR', 'name'])
        if cnt <= cap:
            selected += sub.name.tolist()
        else:
            ch, tg = ir_span_select(sub, cap)
            selected += ch
            sel_notes[fam] = dict(targets=np.round(tg, 2).tolist(), chosen=ch)
            R.loc[R.name.isin(sub.name[~sub.name.isin(ch)]), 'exclusion_reason'] = \
                f'rule (iv) family cap: {fam} has {cnt} eligible datasets > cap {cap}; not chosen by the log-IR span rule'
            R.loc[R.name.isin(sub.name), 'selection_note'] = f"capped family; log-IR targets {sel_notes[fam]['targets']} -> kept {ch}"
    R['selection_note'] = R.get('selection_note', pd.Series('', index=R.index)).fillna('')
    R['included'] = R.name.isin(selected)
    assert (R.included == (R.exclusion_reason == '')).all()

    R['loader_issue'] = R.loader.map(LOADER_ISSUE).fillna('')
    R.loc[(R.source == 'KEEL') & ~R.loader.isin(LOADER_ISSUE), 'loader_issue'] = \
        'none: loader output (encode=False) identical to direct parse of the bundled KEEL .dat'
    R.loc[(R.source != 'KEEL') & (R.loader_issue == ''), 'loader_issue'] = \
        'not checked (non-KEEL; characteristics from generic typing of loader output)'
    R['family_n_eligible'] = R.family.map(fam_counts).fillna(0).astype(int)
    R['family_cap'] = cap
    sel_names = sorted(selected)
    didx = {nm: i for i, nm in enumerate(sel_names)}
    R['dataset_index'] = R.name.map(didx).astype('Int64')
    R['cv_seed'] = R.dataset_index.map(lambda v: BASE_SEED + int(v) if pd.notna(v) else pd.NA).astype('Int64')
    R['family_representative'] = False
    for fam, g in R[R.included].groupby('family'):
        g = g.sort_values(['IR', 'name']); R.loc[g.index[(len(g) - 1) // 2], 'family_representative'] = True
    order = ['name', 'keel_name', 'loader', 'in_110_inventory', 'source', 'citation_key', 'origin_note', 'family', 'included',
             'exclusion_reason', 'selection_note', 'family_n_eligible', 'family_cap', 'family_representative', 'dataset_index',
             'cv_seed', 'n', 'd_raw', 'n_nominal_raw', 'd', 'n_numeric', 'n_onehot', 'n_min', 'n_maj', 'IR', 'IR_group',
             'minority_label', 'n_classes', 'missing_cells', 'missing_cells_encoded', 'duplicate_rows',
             'conflicting_dup_groups', 'conflicting_dup_rows', 'dropped_columns', 'nominal_integer_coded', 'loader_issue']
    REG = R[order].sort_values(['included', 'source', 'name'], ascending=[False, True, True]).reset_index(drop=True)
    REG['IR'] = REG.IR.round(4)
    REG.to_csv(os.path.join(args.out, 'dataset_registry.csv'), index=False)
    print(f'registry: {len(REG)} rows, {int(REG.included.sum())} selected, cap={cap}, totals by cap={totals}', flush=True)

    # --- benchmark dict
    Ri = R.set_index('name')
    bench = {}
    for nm in sel_names:
        r = Ri.loc[nm]; e = enc[r.loader]
        bench[nm] = {'X': np.ascontiguousarray(e['X'], dtype=np.float64), 'y': np.ascontiguousarray(e['y'], dtype=np.int8),
                     'feature_names': list(e['names']), 'feature_types': list(e['types']), 'family': r.family,
                     'source': 'KEEL', 'keel_name': r.keel_name, 'loader': r.loader, 'keel_file': keel_file[r.loader],
                     'minority_label': r.minority_label}
    joblib.dump(bench, os.path.join(args.out, 'benchmark_datasets.joblib'), compress=3)

    # --- CV folds
    cv = {}
    for di, nm in enumerate(sel_names):
        y = bench[nm]['y']; seed = BASE_SEED + di
        folds = []
        for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
                                     .split(np.zeros((len(y), 1)), y)):
            rep, fold = divmod(k, 5)
            assert y[tr].sum() >= 8 and y[te].sum() >= 2, (nm, rep, fold)
            folds.append({'repeat': rep, 'fold': fold, 'train_idx': tr.astype(np.int32), 'test_idx': te.astype(np.int32)})
        cv[nm] = {'seed': seed, 'dataset_index': di, 'n_splits': 5, 'n_repeats': 5, 'folds': folds,
                  'splitter': f'RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state={seed}) [sklearn {sklearn.__version__}]'}
    joblib.dump(cv, os.path.join(args.out, 'cv_folds.joblib'), compress=3)

    # --- LaTeX table
    T = REG[REG.included].sort_values(['IR', 'keel_name']).reset_index(drop=True)
    n_low, n_high = int((T.IR_group == 'low').sum()), int((T.IR_group == 'high').sum())
    L = [r'% Auto-generated from dataset_registry.csv (55 selected KEEL datasets). Requires \usepackage{booktabs}.',
         r'\begin{table}[!htbp]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{5pt}',
         r'\caption{The %d KEEL benchmark datasets, sorted by imbalance ratio (IR\,$=n_{\mathrm{maj}}/n_{\mathrm{min}}$). '
         r'$d$ is the number of features after one-hot encoding of nominal attributes and removal of constant columns. '
         r'The middle rule separates the low-IR group (IR\,$<9$, %d datasets) from the high-IR group (IR\,$\geq 9$, %d datasets).}' % (len(T), n_low, n_high),
         r'\label{tab:datasets}', r'\begin{tabular}{llrrrr}', r'\toprule',
         r'Dataset & Family & $n$ & $d$ & $n_{\mathrm{min}}$ & IR \\', r'\midrule']
    for i, r in T.iterrows():
        if i > 0 and T.IR_group[i] == 'high' and T.IR_group[i - 1] == 'low':
            L.append(r'\midrule')
        L.append(f"{tex_escape(r.keel_name)} & {tex_escape(r.family)} & {r.n} & {r.d} & {r.n_min} & {r.IR:.2f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(os.path.join(args.out, 'dataset_table.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')

    # --- complexity + typology
    if args.skip_complexity:
        return
    cx, sec, ty, sep = {}, {}, {}, {}
    for nm in sel_names:
        X = prep_full(bench[nm]['X']); y = bench[nm]['y'].astype(int)
        cx[nm], tim = complexity(X, y); sec[nm] = sum(tim.values())
        ty[nm] = typology(X, y)
        Xr, yr = bench[nm]['X'], bench[nm]['y']
        sep[nm] = any((Xr[yr == 1, j].max() < Xr[yr == 0, j].min()) or (Xr[yr == 0, j].max() < Xr[yr == 1, j].min())
                      for j in range(Xr.shape[1]))
        print(nm, round(sec[nm], 1), flush=True)
    meas = ['F1', 'F1v', 'F2', 'F3', 'F4', 'L1', 'L2', 'L3', 'N1', 'N2', 'N3', 'N4', 'LSC', 'T2', 'T3', 'T4', 'C1', 'C2',
            'N1_problexity', 'N2_problexity']
    TY = pd.DataFrame(ty).T
    TY['dominant_type'] = TY[['safe', 'borderline', 'rare', 'outlier']].astype(float).idxmax(axis=1)
    base = REG[REG.included].set_index('name')[['keel_name', 'family', 'IR_group', 'n', 'd', 'n_min', 'n_maj', 'IR',
                                                 'family_representative']]
    C = base.join(pd.DataFrame(cx).T[meas]).join(TY)
    C['single_feature_separable'] = pd.Series(sep)
    C['complexity_rows_used'] = C['n']
    C['complexity_subsampled'] = False
    C['complexity_seconds'] = pd.Series(sec).round(2)
    C = C.reset_index().rename(columns={'index': 'name'}).sort_values('name').reset_index(drop=True)
    C.to_csv(os.path.join(args.out, 'dataset_complexity.csv'), index=False, float_format='%.6g')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
