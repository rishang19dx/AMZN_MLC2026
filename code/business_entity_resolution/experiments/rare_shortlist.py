"""
Prototype: "find with rare words, rank with all words".

Per (country, pass):
  baseline  top-K of the full product (max_df 2% vocabulary)       = blocking.py today
  rare      1. shortlist: top-M of the product over RARE words only (df <= rare_df * n_targets)
            2. rescore the shortlist with the exact baseline cosine (all kept words)
            3. keep top-K
            fallback: S1 rows whose shortlist has < K candidates get the full product
Reports time, product non-zeros per row, per-pass and union recall, F0.5 ceiling,
and how many baseline pairs the variant reproduces.

python experiments/rare_shortlist.py --split local_val   (from code/business_entity_resolution/)
Results and conclusion: docs/REPORT.md, Run D.
"""
import argparse, os, sys, time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)
import config
import blocking as B
from evaluate import read_ground_truth, score_candidates, read_country_map

K = {'addr': 20, 'full': 30}
MAX_DF = 0.02


def vectors(s1_text, tg_text, kw):
    vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=2, **kw)
    vec.fit(pd.concat([s1_text, tg_text]))
    X, Y = vec.transform(s1_text), vec.transform(tg_text)
    df = np.bincount(Y.indices, minlength=Y.shape[1])
    keep = df <= MAX_DF * Y.shape[0]
    return X[:, keep].tocsr(), Y[:, keep].tocsr(), df[keep]


def nnz_per_row(X, Y, sample=512):
    idx = np.random.default_rng(0).choice(X.shape[0], min(sample, X.shape[0]), replace=False)
    return (X[idx] @ Y.T.tocsr()).nnz / len(idx)


def dot_pairs(X, Y, r, c, chunk=1_000_000):
    out = np.empty(len(r), np.float32)
    for s in range(0, len(r), chunk):
        e = s + chunk
        out[s:e] = np.asarray(X[r[s:e]].multiply(Y[c[s:e]]).sum(axis=1)).ravel()
    return out


def keep_topk(r, c, v, k):
    rank = B.rank_within(r, v)
    m = rank <= k
    return r[m], c[m], v[m]


def rare_pass(X, Y, df, k, rare_df, M, workers, fallback):
    rare = df <= rare_df * Y.shape[0]
    Xr, Yr = X[:, rare].tocsr(), Y[:, rare].tocsr()
    r, c, _ = B.topk_sparse(Xr, Yr, M, workers=workers)
    v = dot_pairs(X, Y, r, c)
    r, c, v = keep_topk(r, c, v, k)
    n_fb = 0
    if fallback:
        cnt = np.bincount(r, minlength=X.shape[0])
        fb = np.flatnonzero(cnt < k).astype(np.int32)
        n_fb = len(fb)
        if n_fb:
            r2, c2, v2 = B.topk_sparse(X[fb], Y, k, workers=workers)
            drop = np.isin(r, fb)
            r = np.concatenate([r[~drop], fb[r2]]); c = np.concatenate([c[~drop], c2]); v = np.concatenate([v[~drop], v2])
    return r, c, v, n_fb, nnz_per_row(Xr, Yr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='local_val')
    ap.add_argument('--variants', default='0.001:100:1,0.001:100:0,0.0005:100:1,0.002:100:1,0.001:200:1',
                    help='rare_df:M:fallback,...')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('BER_WORKERS', os.cpu_count())))
    a = ap.parse_args()
    variants = [('base', None)] + [(v, tuple(float(x) for x in v.split(':'))) for v in a.variants.split(',')]

    s1, tg = B.load_split(a.split)
    # pairs[variant][pass] -> list of (s1_idx, tg_idx) arrays ; timing[variant] -> seconds
    pairs = defaultdict(lambda: defaultdict(list))
    timing, stats = defaultdict(float), defaultdict(list)
    for country, s1_c in s1.groupby('country', sort=False):
        tg_c = tg[tg['country'] == country]
        si, ti = s1_c.index.to_numpy(np.int32), tg_c.index.to_numpy(np.int32)
        for p, k in K.items():
            fields, kw = B.PASSES[p]
            X, Y, df = vectors(B.field_text(s1_c, fields), B.field_text(tg_c, fields), kw)
            full_nnz = nnz_per_row(X, Y)
            for name, cfg in variants:
                t = time.time()
                if cfg is None:
                    r, c, v = B.topk_sparse(X, Y, k, workers=a.workers)
                    extra = f'nnz/row {full_nnz:,.0f}'
                else:
                    rare_df, M, fb = cfg
                    r, c, v, n_fb, rn = rare_pass(X, Y, df, k, rare_df, int(M), a.workers, bool(fb))
                    extra = f'nnz/row {rn:,.0f} ({rn / full_nnz:.2f}x), fallback rows {n_fb:,} ({n_fb / len(si):.1%})'
                dt = time.time() - t
                timing[name] += dt
                stats[name].append(f'{country}/{p}: {dt:.0f}s, {extra}')
                pairs[name][p].append(np.stack([si[r], ti[c]], 1))
                B.log(f'{name:>16} {country:>6} {p}: {dt:6.1f}s  {extra}')

    paths = config.split_paths(a.split)
    gt = read_ground_truth(paths['gt'])
    countries = read_country_map(paths['s1'])
    s1_ids, tg_ids = s1['entity_id'].to_numpy(), tg['entity_id'].to_numpy()
    true_pairs = {(a_, b_) for a_, bs in gt.items() for b_ in bs}
    n_true = len(true_pairs)
    base_sets = {}
    rows = []
    for name, _ in variants:
        per_pass, union = {}, None
        for p in K:
            arr = np.concatenate(pairs[name][p])
            codes = arr[:, 0].astype(np.int64) * len(tg) + arr[:, 1]
            per_pass[p] = codes
            union = codes if union is None else np.union1d(union, codes)
        if name == 'base':
            base_sets = {p: per_pass[p] for p in K}
        cands = defaultdict(set)
        for code in union:
            cands[s1_ids[code // len(tg)]].add(tg_ids[code % len(tg)])
        rep = score_candidates(gt, cands, countries)
        row = {'variant': name, 'time_s': round(timing[name]), 'cands/S1': rep['ALL']['cands_mean']}
        for p in K:
            got = {(s1_ids[x // len(tg)], tg_ids[x % len(tg)]) for x in per_pass[p]}
            row[f'recall_{p}'] = len(got & true_pairs) / n_true
            row[f'same_as_base_{p}'] = np.isin(per_pass[p], base_sets[p]).mean()
        row['recall_union'] = rep['ALL']['pair_recall']
        for cn in [c for c in rep if c != 'ALL']:
            row[f'recall_{cn}'] = rep[cn]['pair_recall']
        row['f05_ceiling'] = rep['ALL']['f05_ceiling']
        rows.append(row)
    pd.set_option('display.width', 250)
    print('\n' + pd.DataFrame(rows).to_string(index=False, float_format='%.4f'))
    for name, s in stats.items():
        print(f'\n{name}\n  ' + '\n  '.join(s))


if __name__ == '__main__':
    main()
