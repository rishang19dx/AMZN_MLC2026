"""
Where do true pairs rank in each blocking pass, and how does that change with
the size of the target pool?

For a sample of S1 records, computes the exact rank of every true target in the
addr and full passes (the blocking.py vectorisation, max-df 2%), against all
targets of the same country in the split. rank = 1 + number of targets scoring
strictly higher; inf when the pair shares no kept word.

  --split local_val     small universe (1.0M targets): what blocking.py sees on local_val
  --split local_train   test-sized pool (9.3M targets): what blocking sees at test scale

python experiments/miss_ranks.py --split local_train --sample 20000
"""
import argparse, os, sys, time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import config
import blocking as B
from evaluate import read_ground_truth

MAX_DF = 0.02
PASSES = ('addr', 'full')
KS = (10, 20, 30, 50, 100, 200, 500, 1000)


def vectors(s1_text, tg_text, kw):
    vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=2, **kw)
    vec.fit(pd.concat([s1_text, tg_text]))
    X, Y = vec.transform(s1_text), vec.transform(tg_text)
    df = np.bincount(Y.indices, minlength=Y.shape[1])
    keep = df <= MAX_DF * Y.shape[0]
    return X[:, keep].tocsr(), Y[:, keep].tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='local_val')
    ap.add_argument('--sample', type=int, default=20000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    s1, tg = B.load_split(a.split)
    gt = read_ground_truth(config.split_paths(a.split)['gt'])
    tg_pos = pd.Series(np.arange(len(tg)), index=tg['entity_id'])
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(s1), min(a.sample, len(s1)), replace=False))
    in_sample = np.zeros(len(s1), bool); in_sample[sample] = True

    rows, rows_t = [], []
    for country, s1_c in s1.groupby('country', sort=False):
        tg_c = tg[tg['country'] == country]
        local = pd.Series(np.arange(len(tg_c)), index=tg_c.index)      # global tg row -> column in Y
        smp = np.flatnonzero(in_sample[s1_c.index])                     # rows of s1_c in the sample
        if len(smp) == 0:
            continue
        truth = [[local[tg_pos[t]] for t in gt.get(s1_c['entity_id'].iat[i], [])] for i in smp]
        for p in PASSES:
            t0 = time.time()
            fields, kw = B.PASSES[p]
            X, Y = vectors(B.field_text(s1_c, fields), B.field_text(tg_c, fields), kw)
            YT = Y.T.tocsr()
            for s in range(0, len(smp), 256):
                P = (X[smp[s:s + 256]] @ YT).tocsr()
                P.sort_indices()
                for j in range(P.shape[0]):
                    i = s + j
                    if not truth[i]:
                        continue
                    a_, b_ = P.indptr[j], P.indptr[j + 1]
                    d, c = P.data[a_:b_], P.indices[a_:b_]
                    for t in truth[i]:
                        pos = np.searchsorted(c, t)
                        v = float(d[pos]) if pos < len(c) and c[pos] == t else 0.0
                        rank = np.inf if v == 0 else 1 + int(np.count_nonzero(d > v))
                        rows.append((country, p, s1_c['entity_id'].iat[smp[i]], tg_c['entity_id'].iat[t],
                                     bool(tg_c['nonlatin'].iat[t]), rank, v, b_ - a_))
            # target side: rank of the true S1 among all S1 of the country, for the same true pairs
            XT = X.T.tocsr()
            pi = np.array([smp[i] for i in range(len(smp)) for _ in truth[i]], np.int64)
            pt = np.array([t for i in range(len(smp)) for t in truth[i]], np.int64)
            for s in range(0, len(pt), 512):
                P = (Y[pt[s:s + 512]] @ XT).tocsr()
                P.sort_indices()
                for j in range(P.shape[0]):
                    a_, b_ = P.indptr[j], P.indptr[j + 1]
                    d, c = P.data[a_:b_], P.indices[a_:b_]
                    q = pi[s + j]
                    pos = np.searchsorted(c, q)
                    v = float(d[pos]) if pos < len(c) and c[pos] == q else 0.0
                    rows_t.append((s1_c['entity_id'].iat[q], tg_c['entity_id'].iat[pt[s + j]], p + '_t',
                                   np.inf if v == 0 else 1 + int(np.count_nonzero(d > v))))
            B.log(f'{country:>6} {p}: {len(smp):,} S1 x {len(tg_c):,} targets, {time.time() - t0:.0f}s')

    df = pd.DataFrame(rows, columns=['country', 'pass', 's1_id', 'tg_id', 'nonlatin', 'rank', 'score', 'nnz_row'])
    w = df.pivot_table(index=['country', 's1_id', 'tg_id', 'nonlatin'], columns='pass', values='rank').reset_index()
    wt = pd.DataFrame(rows_t, columns=['s1_id', 'tg_id', 'pass', 'rank']).pivot_table(
        index=['s1_id', 'tg_id'], columns='pass', values='rank').reset_index()
    w = w.merge(wt, on=['s1_id', 'tg_id'], how='left')
    w['source'] = w['tg_id'].str[:2]
    out = a.out or os.path.join(config.CACHE_DIR, f'miss_ranks_{a.split}.pkl')
    w.to_pickle(out)

    print(f'\n== {a.split}: {len(w):,} true pairs of {len(sample):,} sampled S1; recall@K per pass')
    for grp, g in [('ALL', w)] + list(w.groupby('country')) + [('non-Latin', w[w['nonlatin']])]:
        line = {f'{p}@{k}': (g[p] <= k).mean() for p in PASSES for k in KS}
        cur = ((g['addr'] <= 20) | (g['full'] <= 30)).mean()
        print(f'{grp:>10} n={len(g):>6}  current(addr20|full30)={cur:.4f}  ' +
              '  '.join(f'{p}: ' + ' '.join(f'{k}:{(g[p] <= k).mean():.3f}' for k in KS) for p in PASSES))
    miss = w[~((w['addr'] <= 20) | (w['full'] <= 30))]
    print(f'\n== {len(miss):,} pairs missed by the current config')
    print(f'  share of missed with no shared kept word in either pass: {(np.isinf(miss["addr"]) & np.isinf(miss["full"])).mean():.3f}')
    print(f'  best rank (min over passes) buckets:')
    best = np.minimum(miss['addr'], miss['full'])
    print(pd.cut(best, [0, 50, 100, 200, 500, 1000, 1e9, np.inf]).value_counts(sort=False).to_string())
    print(f'  missed: non-Latin share {miss["nonlatin"].mean():.3f} (all pairs {w["nonlatin"].mean():.3f}); '
          f'by country {miss["country"].value_counts().to_dict()}; by source {miss["source"].value_counts().to_dict()}')
    print('\n== target side: recall if each target also keeps its top-k S1 (union with current config)')
    cur = (w['addr'] <= 20) | (w['full'] <= 30)
    for k in (1, 2, 3, 5, 10):
        tk = (w['addr_t'] <= k) | (w['full_t'] <= k)
        print(f'  k={k:>2}: target-side alone {tk.mean():.4f}   union with current {(cur | tk).mean():.4f}   '
              f'recovers {(tk & ~cur).sum():,} of {(~cur).sum():,} missed')
    print(f'  median product non-zeros per S1 row: ' +
          df.groupby(['country', 'pass'])['nnz_row'].median().to_string().replace('\n', '; '))


if __name__ == '__main__':
    main()
