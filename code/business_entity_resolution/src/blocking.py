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

def load_records(path, keep_raw=True):
    df = read_tsv(path)
    df['name_n'] = df['business_name'].map(norm)
    df['addr_n'] = df['business_address'].map(norm)
    df['full_n'] = df['name_n'] + ' ' + df['addr_n']
    if not keep_raw:   # raw text is only needed by the dev-loop report; ~2 GB on test
        df = df.drop(columns=['business_name', 'business_address'])
    return df


def load_split(split):
    p = config.split_paths(split)
    keep_raw = split != 'test'
    s1 = load_records(p['s1'], keep_raw)
    log(f'S1: {len(s1):,} records')
    tg = pd.concat([load_records(p['s2'], keep_raw), load_records(p['s3'], keep_raw)], ignore_index=True)
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
    Row chunks run in parallel in forked worker processes (Linux and macOS;
    on macOS set OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES, as scripts/mac_run.sh does).
    Peak memory per worker grows with chunk x len(Y): see auto_chunk().
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


def auto_chunk(n_targets, budget=None):
    """
    S1 rows per sparse-product chunk. The product's size (and so each worker's
    peak memory) grows with rows x targets, so on test (~10x the targets of
    local_val) the chunk shrinks to keep every worker's peak about the same as
    local_val's 2048 rows. BER_BLOCK_BUDGET = rows x targets per chunk.
    """
    budget = budget or float(os.environ.get('BER_BLOCK_BUDGET', 1.2e9))
    return int(np.clip(budget / max(n_targets, 1), 64, 2048))


def run_pass(s1_text, tg_text, k, max_df, vec_kwargs, workers=1, chunk=2048):
    vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=2, **vec_kwargs)
    vec.fit(pd.concat([s1_text, tg_text]))
    X, Y = vec.transform(s1_text), vec.transform(tg_text)
    # drop features that are too common among targets (after normalisation, so
    # scores are partial cosines; ranking by rare evidence is what we want)
    df = np.bincount(Y.indices, minlength=Y.shape[1])
    keep = df <= max_df * Y.shape[0]
    X, Y = X[:, keep], Y[:, keep]
    return topk_sparse(X.tocsr(), Y.tocsr(), k, chunk=chunk, workers=workers)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def within_rank(group, score):
    """1-based rank of `score` (descending) within each `group`, ties broken by
    position: the same as pandas groupby(group).rank(ascending=False,
    method='first'), at ~8 bytes per row instead of pandas' ~100."""
    order = np.lexsort((-score, group))            # stable: equal scores keep their order
    g = group[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    sizes = np.diff(np.r_[starts, len(g)])
    rank = np.empty(len(g), np.float32)
    rank[order] = np.arange(len(g)) - np.repeat(starts, sizes) + 1
    return rank


def generate_candidates(s1, tg, ks, max_df, workers=1, chunk=0):
    """Returns a DataFrame with one row per (s1, target) pair and, for each
    pass, the cosine score and within-S1 rank (NaN if the pass missed it).
    chunk=0 sizes the sparse-product chunks per country with auto_chunk().

    Passes are merged per country (pairs never cross countries) on an int64
    pair key with numpy, not a pandas two-key groupby: on test (~40M pass rows
    for India alone) the groupby needed ~10 GB of temporaries."""
    n_tg = len(tg)
    cols = [f'{kind}_{pname}' for pname in PASSES for kind in ('score', 'rank')]
    wides = []
    for country, s1_c in s1.groupby('country', sort=False):
        tg_c = tg[tg['country'] == country]
        if tg_c.empty:
            continue
        s1_idx, tg_idx = s1_c.index.to_numpy(), tg_c.index.to_numpy()
        ch = chunk or auto_chunk(len(tg_c))
        keys, passes = [], []
        for pname, (field, kw) in PASSES.items():
            k = ks[pname]
            if k <= 0:
                continue
            r, c, v = run_pass(s1_c[field], tg_c[field], k, max_df, kw, workers, ch)
            keys.append(s1_idx[r].astype(np.int64) * n_tg + tg_idx[c])
            passes.append((pname, v, within_rank(r, v)))
            log(f'  {country:>8} {pname:>4}: {len(s1_c):,} x {len(tg_c):,} -> {len(r):,} pairs (chunk {ch})')
            del r, c
        if not passes:
            continue
        # one row per pair; a pass that did not retrieve the pair leaves NaN
        # (within a pass every pair is unique: top-k columns of one row)
        sizes = [len(x) for x in keys]
        uniq, inv = np.unique(np.concatenate(keys), return_inverse=True)
        del keys
        wide = {'s1': (uniq // n_tg).astype(np.int32), 'tg': (uniq % n_tg).astype(np.int32)}
        del uniq
        for col in cols:          # a disabled pass keeps its (all-NaN) columns
            wide[col] = np.full(len(wide['s1']), np.nan, np.float32)
        off = 0
        for (pname, v, rank), n in zip(passes, sizes):
            wide[f'score_{pname}'][inv[off:off + n]] = v
            wide[f'rank_{pname}'][inv[off:off + n]] = rank
            off += n
        del passes, inv
        wides.append(pd.DataFrame(wide))
    return pd.concat(wides, ignore_index=True)


def write_outputs(split, s1, tg, cand):
    out_dir = os.path.join(config.OUTPUT_DIR, split)
    cache_dir = os.path.join(config.CACHE_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # IDs are joined to the integer row positions inside DuckDB (which spills
    # to disk if needed): on test, 70M pairs x 2 ID columns as Python strings
    # would take ~8 GB.
    import duckdb
    s1_ids, tg_ids = s1['entity_id'].to_numpy(), tg['entity_id'].to_numpy()
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{os.environ.get('BER_DUCKDB_MEMORY', '3GB')}'")
    con.execute(f"SET temp_directory = '{os.path.join(cache_dir, 'duckdb_tmp')}'")
    con.execute('SET preserve_insertion_order = false')
    con.register('c', cand)
    con.register('s1x', pd.DataFrame({'s1': np.arange(len(s1), dtype=np.int32), 's1_id': s1_ids}))
    con.register('tgx', pd.DataFrame({'tg': np.arange(len(tg), dtype=np.int32), 'cand_id': tg_ids}))
    score_cols = ', '.join(f'c.{x}' for x in cand.columns if x.startswith(('score_', 'rank_')))
    pq = os.path.join(cache_dir, 'candidates.parquet')
    con.execute('COPY (SELECT s1x.s1_id, tgx.cand_id, ' + score_cols +
                ' FROM c JOIN s1x ON c.s1 = s1x.s1 JOIN tgx ON c.tg = tgx.tg)'
                f" TO '{pq}' (FORMAT parquet, COMPRESSION zstd)")
    con.close()

    # candidate_pairs.tsv: sort pairs by (S1 row, target ID string) with integer
    # keys and stream the lists out. (DuckDB's ordered string_agg cannot spill
    # to disk and runs out of memory at this size.) Every S1 gets a row.
    tg_rank = np.empty(len(tg_ids), np.int64)
    tg_rank[np.argsort(tg_ids, kind='stable')] = np.arange(len(tg_ids))   # = Python string order
    s1c, tgc = cand['s1'].to_numpy(), cand['tg'].to_numpy()
    order = np.lexsort((tg_rank[tgc], s1c))
    del tg_rank
    bounds = np.searchsorted(s1c[order], np.arange(len(s1_ids) + 1))
    ids = tg_ids[tgc[order]]
    del order
    tsv = os.path.join(out_dir, 'candidate_pairs.tsv')
    with open(tsv, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for i, s1_id in enumerate(s1_ids):
            f.write(f"{s1_id}\t{','.join(ids[bounds[i]:bounds[i + 1]])}\n")
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
    ap.add_argument('--chunk', type=int, default=0,
                    help='S1 rows per sparse-product chunk; 0 = auto from the target count (BER_BLOCK_BUDGET)')
    ap.add_argument('--no-write', action='store_true', help='diagnostics only')
    args = ap.parse_args()

    s1, tg = load_split(args.split)
    ks = {'name': args.k_name, 'addr': args.k_addr, 'full': args.k_full}
    cand = generate_candidates(s1, tg, ks, args.max_df, args.workers, args.chunk)
    log(f'{len(cand):,} candidate pairs')
    if args.split != 'test':
        report_by_pass(args.split, s1, tg, cand)
    if not args.no_write:
        write_outputs(args.split, s1, tg, cand)


if __name__ == '__main__':
    main()
