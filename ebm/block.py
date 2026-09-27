"""
Candidate generation (blocking) on a preprocessed cache: per-country TF-IDF
top-K, the same design as the main pipeline's blocking (which keeps 96.9% of
true pairs at test scale with K 20/30, 97.5% with K 30/50).

Two passes, unioned; both use IDF fitted per country on the split's own
records (unsupervised) and drop tokens present in more than --max-df of the
country's targets (they rank nothing and dominate the cost):
  addr  address words, house number, other numbers, postcode
  full  name words + the address tokens above
Cosine top-K per Source 1 via chunked sparse products in N worker processes.

Queries: all Source 1 rows, or (training) a hash-sampled --s1-fraction of the
train-split Source 1 PLUS every validation Source 1. Targets: always ALL
targets of the country, so crowding matches test.

Output: <cache>/candidates.npz with s1 (row), tg (row), score_addr, score_full
(NaN = not retrieved by that pass), and <cache>/candidate_pairs.tsv.

Usage:
  python -m ebm.block --cache cache/test --k-addr 30 --k-full 50
  python -m ebm.block --cache cache/train --k-addr 30 --k-full 50 --s1-fraction 0.3
"""

import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp

from ebm.normalize import F_NAME, F_ADDR, F_NUM, F_HOUSE, F_ZIP

PASSES = {'addr': (F_ADDR, F_NUM, F_HOUSE, F_ZIP), 'full': (F_NAME, F_ADDR, F_NUM, F_HOUSE, F_ZIP)}
_W = {}


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:8.1f}s] {msg}', flush=True)


def matrix(tokens, fields, rows, keep_fields, col_of, n_cols, idf):
    """Row-normalised binary TF-IDF CSR for the given rows (columns remapped)."""
    t = np.asarray(tokens[rows]); f = np.asarray(fields[rows])
    mask = np.isin(f, keep_fields) & (t > 0)
    r, c = np.nonzero(mask)
    cols = col_of[t[r, c]]
    ok = cols >= 0
    r, cols = r[ok], cols[ok]
    m = sp.csr_matrix((idf[cols], (r, cols)), shape=(len(rows), n_cols), dtype=np.float32)
    m.sum_duplicates()
    m.data[:] = idf[m.indices]                                   # binary tf: duplicates count once
    norm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    norm[norm == 0] = 1
    return sp.diags(1 / norm).dot(m).tocsr().astype(np.float32)


def _work(args):
    lo, hi, k = args
    X, YT = _W['X'][lo:hi], _W['YT']
    P = (X @ YT).tocsr()
    out_q, out_t, out_s = [], [], []
    for i in range(P.shape[0]):
        a, b = P.indptr[i], P.indptr[i + 1]
        if a == b:
            continue
        d, idx = P.data[a:b], P.indices[a:b]
        top = np.argpartition(-d, min(k, len(d)) - 1)[:k] if len(d) > k else np.arange(len(d))
        out_q.append(np.full(len(top), lo + i, np.int32)); out_t.append(idx[top]); out_s.append(d[top])
    if not out_q:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    return np.concatenate(out_q), np.concatenate(out_t).astype(np.int32), np.concatenate(out_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--k-addr', type=int, default=30)
    ap.add_argument('--k-full', type=int, default=50)
    ap.add_argument('--max-df', type=float, default=0.02)
    ap.add_argument('--s1-fraction', type=float, default=1.0, help='train: share of train-split Source 1 to block (validation Source 1 always included)')
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('EBM_WORKERS', min(32, os.cpu_count()))))
    args = ap.parse_args()
    c = args.cache
    with open(os.path.join(c, 'meta.json')) as f:
        meta = json.load(f)
    tokens = np.load(os.path.join(c, 'tokens.npy'), mmap_mode='r')
    fields = np.load(os.path.join(c, 'fields.npy'), mmap_mode='r')
    country = np.load(os.path.join(c, 'country.npy'))
    ids = np.load(os.path.join(c, 'ids.npy'))
    n, n_s1 = meta['n'], meta['n_s1']
    val = np.load(os.path.join(c, 'val.npy'))
    s1 = np.arange(n_s1)
    if args.s1_fraction < 1.0:
        h = np.array([int(hashlib.blake2b(f'block:{e}'.encode(), digest_size=8).hexdigest(), 16) / 2**64 for e in ids[:n_s1]])
        s1 = s1[val | (h < args.s1_fraction)]
    log(f'{len(s1):,} query Source 1 of {n_s1:,}; {n - n_s1:,} targets; countries {meta["countries"]}')

    pairs = {}                                                   # pass -> (q, t, score)
    for pname, keep in PASSES.items():
        k = args.k_addr if pname == 'addr' else args.k_full
        qs, ts, ss = [], [], []
        for ci, cname in enumerate(meta['countries']):
            q = s1[country[s1] == ci]
            tg = np.arange(n_s1, n)[country[n_s1:] == ci]
            if len(q) == 0 or len(tg) == 0:
                continue
            # vocabulary + IDF from this country's records (Source 1 + targets)
            vocab_rows = np.concatenate([np.arange(n_s1)[country[:n_s1] == ci], tg])
            df = np.zeros(1 << 20, np.int64)
            tg_df = np.zeros(1 << 20, np.int64)
            for s in range(0, len(vocab_rows), 1_000_000):
                r = vocab_rows[s:s + 1_000_000]
                t = np.asarray(tokens[r]); f = np.asarray(fields[r])
                m = np.isin(f, keep) & (t > 0)
                rr = np.repeat(np.arange(len(r), dtype=np.int64), m.sum(1))
                u = np.unique(rr * (1 << 20) + t[m])                  # (row, token) once per row
                ur, ut = u >> 20, u & ((1 << 20) - 1)
                df += np.bincount(ut, minlength=1 << 20)
                tg_df += np.bincount(ut[r[ur] >= n_s1], minlength=1 << 20)
            usable = (df > 0) & (tg_df <= args.max_df * len(tg))
            col_of = np.full(1 << 20, -1, np.int64)
            col_of[usable] = np.arange(int(usable.sum()))
            idf = np.log((1 + len(vocab_rows)) / (1 + df[usable])).astype(np.float32) + 1
            X = matrix(tokens, fields, q, keep, col_of, int(usable.sum()), idf)
            Y = matrix(tokens, fields, tg, keep, col_of, int(usable.sum()), idf)
            _W['X'], _W['YT'] = X, Y.T.tocsr()
            jobs = [(lo, min(lo + args.chunk, len(q)), k) for lo in range(0, len(q), args.chunk)]
            with Pool(args.workers) as pool:
                for qq, tt, sc in pool.imap_unordered(_work, jobs):
                    qs.append(q[qq]); ts.append(tg[tt]); ss.append(sc)
            log(f'  {pname} {cname}: {len(q):,} x {len(tg):,}, {int(usable.sum()):,} tokens')
        pairs[pname] = (np.concatenate(qs), np.concatenate(ts), np.concatenate(ss))
        log(f'{pname}: {len(pairs[pname][0]):,} pairs')

    # union of passes, one row per (s1, target), score per pass (NaN = not retrieved)
    key = lambda q, t: q.astype(np.int64) * n + t
    allk = np.unique(np.concatenate([key(*p[:2]) for p in pairs.values()]))
    out = {'s1': (allk // n).astype(np.int32), 'tg': (allk % n).astype(np.int32)}
    for pname, (q, t, s) in pairs.items():
        col = np.full(len(allk), np.nan, np.float32)
        col[np.searchsorted(allk, key(q, t))] = s
        out[f'score_{pname}'] = col
    np.savez(os.path.join(c, 'candidates.npz'), **out)
    per = np.bincount(out['s1'], minlength=n_s1)[s1]
    log(f'union: {len(allk):,} pairs, {per.mean():.1f} per query Source 1 (max {per.max()})')

    if meta.get('labelled'):
        indptr, targets = np.load(os.path.join(c, 'gt_indptr.npy')), np.load(os.path.join(c, 'gt_targets.npy'))
        truth = np.concatenate([np.arange(indptr[i], indptr[i + 1]) for i in s1]) if len(s1) else np.array([], int)
        tk = key(np.repeat(s1, np.diff(indptr)[s1]), targets[truth])
        found = np.isin(tk, allk)
        log(f'blocking recall on query Source 1: {found.mean():.4f} of {len(tk):,} true pairs')

    with open(os.path.join(c, 'candidate_pairs.tsv'), 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        order = np.argsort(out['s1'], kind='stable')
        s_sorted, t_sorted = out['s1'][order], out['tg'][order]
        starts = np.searchsorted(s_sorted, np.arange(n_s1 + 1))
        for i in range(n_s1):
            f.write(f"{ids[i]}\t{','.join(ids[t_sorted[starts[i]:starts[i + 1]]])}\n")
    log(f'wrote candidates.npz and candidate_pairs.tsv in {c}')


if __name__ == '__main__':
    main()
