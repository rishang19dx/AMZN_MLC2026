"""
Blocking / candidate generation, v1: lexical TF-IDF nearest neighbours.

For every S1 entity, retrieve the top-K most similar S2+S3 records *within the
same country* (true matches never cross countries; country is compared as an
open string label, so France needs no special handling). Several passes look
at different fields, and the union of their top-K lists is the candidate set:

  name  char 3-grams of the normalised name       typos, spacing, transliteration noise
  addr  word tokens of the normalised address     trade names / name changes (house no. + street)
  full  word tokens of name + address             joint evidence when each field alone is weak

IDF is fitted per country on S1 + targets (unsupervised, uses only the provided
files), which down-weights "pvt", "limited", "llc", "sarl", "road" etc. without
any hand-made lists. Features whose document frequency among the targets is
above --max-df are dropped: they carry little signal and dominate the cost of
the sparse matrix product.

Outputs (per split):
  <BER_OUTPUT_DIR>/<split>/candidate_pairs.tsv   submission-format candidate lists
  <BER_CACHE_DIR>/<split>/candidates.parquet     one row per pair with per-pass
      scores and ranks (the contract the matcher reads; see docs/PIPELINE.md)

Usage:
  python src/blocking.py --split local_val            # dev loop, scores itself vs ground truth
  python src/blocking.py --split test
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import read_tsv
from normalize import is_non_latin, norm

PASSES = {
    # pass name: (field, vectorizer kwargs)
    'name': ('name_n', dict(analyzer='char_wb', ngram_range=(3, 3))),
    'addr': ('addr_n', dict(analyzer='word', token_pattern=r'\S+')),
    'full': ('full_n', dict(analyzer='word', token_pattern=r'\S+')),
}


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:7.1f}s] {msg}', flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_records(path):
    df = read_tsv(path)
    df['name_n'] = df['business_name'].map(norm)
    df['addr_n'] = df['business_address'].map(norm)
    df['full_n'] = df['name_n'] + ' ' + df['addr_n']
    return df


def load_split(split):
    p = config.split_paths(split)
    s1 = load_records(p['s1'])
    log(f'S1: {len(s1):,} records')
    tg = pd.concat([load_records(p['s2']), load_records(p['s3'])], ignore_index=True)
    log(f'S2+S3: {len(tg):,} records')
    return s1, tg


# ---------------------------------------------------------------------------
# Sparse top-K retrieval
# ---------------------------------------------------------------------------

def _topk_rows(X, YT, k, start, chunk):
    """Top-k columns of X[start:start+chunk] @ YT, as flat (row, col, score) arrays."""
    P = (X[start:start + chunk] @ YT).tocsr()
    indptr, indices, data = P.indptr, P.indices, P.data
    rows, cols, vals = [], [], []
    for i in range(P.shape[0]):
        a, b = indptr[i], indptr[i + 1]
        if a == b:
            continue
        d = data[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k)[:k]
            c, d = indices[a:b][sel], d[sel]
        else:
            c = indices[a:b]
        rows.append(np.full(len(c), start + i, dtype=np.int32))
        cols.append(c.astype(np.int32))
        vals.append(d.astype(np.float32))
    if not rows:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


_SHARED = {}  # matrices handed to forked workers without pickling


def _worker(args):
    start, chunk, k = args
    return _topk_rows(_SHARED['X'], _SHARED['YT'], k, start, chunk)


def topk_sparse(X, Y, k, chunk=2048, workers=1):
    """
    For each row of X return the k columns of X @ Y.T with the highest score
    (rows are L2-normalised TF-IDF, so scores are cosines). Returns three flat
    arrays (row, col, score); rows with fewer than k non-zero scores return fewer.
    Row chunks run in parallel in forked worker processes (Linux).
    """
    YT = Y.T.tocsr()
    jobs = [(s, chunk, k) for s in range(0, X.shape[0], chunk)]
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        _SHARED.update(X=X, YT=YT)
        with mp.get_context('fork').Pool(workers) as pool:
            out = pool.map(_worker, jobs)
        _SHARED.clear()
    else:
        out = [_topk_rows(X, YT, k, s, c) for s, c, k in jobs]
    return tuple(np.concatenate([o[j] for o in out]) for j in range(3))


def run_pass(s1_text, tg_text, k, max_df, vec_kwargs, workers=1):
    vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=2, **vec_kwargs)
    vec.fit(pd.concat([s1_text, tg_text]))
    X, Y = vec.transform(s1_text), vec.transform(tg_text)
    # drop features that are too common among targets (after normalisation, so
    # scores are partial cosines; ranking by rare evidence is what we want)
    df = np.bincount(Y.indices, minlength=Y.shape[1])
    keep = df <= max_df * Y.shape[0]
    X, Y = X[:, keep], Y[:, keep]
    return topk_sparse(X.tocsr(), Y.tocsr(), k, workers=workers)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def generate_candidates(s1, tg, ks, max_df, workers=1):
    """Returns a DataFrame with one row per (s1, target) pair and, for each
    pass, the cosine score and within-S1 rank (NaN if the pass missed it)."""
    parts = []
    for country, s1_c in s1.groupby('country', sort=False):
        tg_c = tg[tg['country'] == country]
        if tg_c.empty:
            continue
        s1_idx, tg_idx = s1_c.index.to_numpy(), tg_c.index.to_numpy()
        for pname, (field, kw) in PASSES.items():
            k = ks[pname]
            if k <= 0:
                continue
            r, c, v = run_pass(s1_c[field], tg_c[field], k, max_df, kw, workers)
            part = pd.DataFrame({'s1': s1_idx[r].astype(np.int64), 'tg': tg_idx[c].astype(np.int64)})
            part[f'score_{pname}'] = v
            part[f'rank_{pname}'] = part.groupby('s1')[f'score_{pname}'].rank(
                ascending=False, method='first').astype(np.float32)
            parts.append(part)
            log(f'  {country:>8} {pname:>4}: {len(s1_c):,} x {len(tg_c):,} -> {len(part):,} pairs')
    # one row per pair; a pass that did not retrieve the pair leaves NaN
    wide = pd.concat(parts, ignore_index=True).groupby(['s1', 'tg'], sort=False).max().reset_index()
    for pname in PASSES:  # a disabled pass still gets (all-NaN) columns
        for col in (f'score_{pname}', f'rank_{pname}'):
            if col not in wide:
                wide[col] = np.float32('nan')
    return wide


def write_outputs(split, s1, tg, cand):
    out_dir = os.path.join(config.OUTPUT_DIR, split)
    cache_dir = os.path.join(config.CACHE_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    cand = cand.assign(s1_id=s1['entity_id'].to_numpy()[cand['s1']],
                       cand_id=tg['entity_id'].to_numpy()[cand['tg']])
    cols = ['s1_id', 'cand_id'] + [c for c in cand.columns if c.startswith(('score_', 'rank_'))]
    import duckdb
    pq = os.path.join(cache_dir, 'candidates.parquet')
    duckdb.from_df(cand[cols]).write_parquet(pq, compression='zstd')

    lists = cand.groupby('s1_id')['cand_id'].agg(lambda x: ','.join(sorted(x)))
    lists = lists.reindex(s1['entity_id'], fill_value='')  # every S1 gets a row
    tsv = os.path.join(out_dir, 'candidate_pairs.tsv')
    with open(tsv, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id, ids in lists.items():
            f.write(f'{s1_id}\t{ids}\n')
    log(f'wrote {tsv} and {pq}')
    return tsv


def report_by_pass(split, s1, tg, cand):
    """Dev-loop diagnostics: recall of each pass alone, and of the union,
    overall and on the hard buckets (non-Latin target names)."""
    from evaluate import read_ground_truth
    gt = read_ground_truth(config.split_paths(split)['gt'])
    s1_ids, tg_ids = s1['entity_id'].to_numpy(), tg['entity_id'].to_numpy()
    true = pd.DataFrame([(a, b) for a, bs in gt.items() for b in bs], columns=['s1_id', 'cand_id'])
    cand = cand.assign(s1_id=s1_ids[cand['s1']], cand_id=tg_ids[cand['tg']])
    m = true.merge(cand, on=['s1_id', 'cand_id'], how='left')
    tgc = tg.set_index('entity_id')
    m['country'] = tgc.loc[m['cand_id'], 'country'].to_numpy()
    m['non_latin'] = tgc.loc[m['cand_id'], 'business_name'].map(is_non_latin).to_numpy()
    passes = [p for p in PASSES if f'score_{p}' in m]
    found = {p: m[f'score_{p}'].notna() for p in passes}
    m['union'] = np.logical_or.reduce(list(found.values()))
    print('\n== pair recall by pass (true pairs found / true pairs)')
    buckets = {'ALL': m.index == m.index}
    buckets.update({c: m['country'] == c for c in m['country'].unique()})
    buckets['non-Latin name'] = m['non_latin']
    print(f"{'':16}{'n_true':>10}" + ''.join(f'{p:>9}' for p in passes) + f"{'union':>9}")
    for b, mask in buckets.items():
        row = f'{b:16}{mask.sum():>10,}'
        row += ''.join(f'{found[p][mask].mean():9.4f}' for p in passes)
        print(row + f"{m['union'][mask].mean():9.4f}")
    print(f'\ncandidates: {len(cand):,} pairs, {len(cand) / len(s1):.1f} per S1')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='local_val', choices=config.SPLIT_NAMES)
    ap.add_argument('--k-name', type=int, default=0, help='0 disables the pass (slow, weak on local_val)')
    ap.add_argument('--k-addr', type=int, default=20)
    ap.add_argument('--k-full', type=int, default=30)
    ap.add_argument('--max-df', type=float, default=0.02,
                    help='drop features present in more than this fraction of targets. Lower is much faster '
                         'but costs recall (0.005: -1.7pt, 0.001: -8.8pt on the full pass, local_val)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--no-write', action='store_true', help='diagnostics only')
    args = ap.parse_args()

    s1, tg = load_split(args.split)
    ks = {'name': args.k_name, 'addr': args.k_addr, 'full': args.k_full}
    cand = generate_candidates(s1, tg, ks, args.max_df, args.workers)
    log(f'{len(cand):,} candidate pairs')
    if args.split != 'test':
        report_by_pass(args.split, s1, tg, cand)
    if not args.no_write:
        write_outputs(args.split, s1, tg, cand)


if __name__ == '__main__':
    main()
