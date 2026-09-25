"""
Blocking / candidate generation, v1: lexical TF-IDF nearest neighbours.

For every S1 entity, retrieve the top-K most similar S2+S3 records *within the
same country* (true matches never cross countries; country is compared as an
open string label, so France needs no special handling). Several passes look
at different fields, and the union of their top-K lists is the candidate set:

  name  char 3-grams of the normalised name       off by default (slow, weak; REPORT §3)
  addr  word tokens of the normalised address     trade names / name changes (house no. + street)
  full  word tokens of name + address             joint evidence; the main pass

IDF is fitted per country on S1 + targets (unsupervised, uses only the provided
files), which down-weights "pvt", "limited", "llc", "sarl", "road" etc. without
any hand-made lists. Features whose document frequency among the targets is
above --max-df are dropped: they carry little signal and dominate the cost of
the sparse matrix product.

Scales to test (~86M raw pairs): each (country, pass) result is written to a
parquet part as soon as it is computed, and DuckDB merges the parts on disk,
so the full pair table is never held in pandas.

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
import shutil
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
    # pass name: (fields joined with a space, vectorizer kwargs)
    'name': (('name_n',), dict(analyzer='char_wb', ngram_range=(3, 3))),
    'addr': (('addr_n',), dict(analyzer='word', token_pattern=r'\S+')),
    'full': (('name_n', 'addr_n'), dict(analyzer='word', token_pattern=r'\S+')),
}


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:7.1f}s] {msg}', flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_records(path):
    df = read_tsv(path)
    return pd.DataFrame({
        'entity_id': df['entity_id'],
        'country': df['country'],
        'name_n': df['business_name'].map(norm),
        'addr_n': df['business_address'].map(norm),
        'nonlatin': df['business_name'].map(is_non_latin),
    })


def load_split(split):
    p = config.split_paths(split)
    s1 = load_records(p['s1'])
    log(f'S1: {len(s1):,} records')
    tg = pd.concat([load_records(p['s2']), load_records(p['s3'])], ignore_index=True)
    log(f'S2+S3: {len(tg):,} records')
    return s1, tg


def field_text(df, fields):
    return df[fields[0]] if len(fields) == 1 else df[fields[0]] + ' ' + df[fields[1]]


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


def rank_within(rows, scores):
    """1-based rank of each score within its row group (descending)."""
    order = np.lexsort((-scores, rows))
    rs = rows[order]
    starts = np.r_[0, np.flatnonzero(np.diff(rs)) + 1]
    sizes = np.diff(np.r_[starts, len(rs)])
    rank = np.empty(len(rows), np.float32)
    rank[order] = np.arange(len(rs)) - np.repeat(starts, sizes) + 1
    return rank


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def duck(split):
    """DuckDB connection with a memory cap and spill directory under the cache."""
    import duckdb
    con = duckdb.connect()
    tmp = os.path.join(config.CACHE_DIR, split, 'duckdb_tmp')
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET memory_limit='{os.environ.get('BER_DUCKDB_MEM', '6GB')}'")
    con.execute(f"SET temp_directory='{tmp}'")
    return con


def generate_candidates(split, s1, tg, ks, max_df, workers=1):
    """Writes one parquet part per (country, pass): columns s1, tg (row indices),
    score_<pass>, rank_<pass>. Returns the part directory."""
    import duckdb
    part_dir = os.path.join(config.CACHE_DIR, split, 'blocking_parts')
    shutil.rmtree(part_dir, ignore_errors=True)
    os.makedirs(part_dir)
    for ci, (country, s1_c) in enumerate(s1.groupby('country', sort=False)):
        tg_c = tg[tg['country'] == country]
        if tg_c.empty:
            continue
        s1_idx, tg_idx = s1_c.index.to_numpy(np.int32), tg_c.index.to_numpy(np.int32)
        for pname, (fields, kw) in PASSES.items():
            if ks[pname] <= 0:
                continue
            r, c, v = run_pass(field_text(s1_c, fields), field_text(tg_c, fields), ks[pname], max_df, kw, workers)
            part = pd.DataFrame({'s1': s1_idx[r], 'tg': tg_idx[c],
                                 f'score_{pname}': v, f'rank_{pname}': rank_within(r, v)})
            duckdb.from_df(part).write_parquet(os.path.join(part_dir, f'c{ci:02d}_{pname}.parquet'))
            log(f'  {country:>8} {pname:>4}: {len(s1_c):,} x {len(tg_c):,} -> {len(part):,} pairs')
            del part, r, c, v
    return part_dir


def write_outputs(split, s1, tg, part_dir, ks):
    """Merge the parts on disk (DuckDB) into candidates.parquet + candidate_pairs.tsv."""
    con = duck(split)
    con.register('s1x', pd.DataFrame({'s1': np.arange(len(s1), dtype=np.int32), 's1_id': s1['entity_id']}))
    con.register('tgx', pd.DataFrame({'tg': np.arange(len(tg), dtype=np.int32), 'cand_id': tg['entity_id']}))
    cols = []
    for p in PASSES:
        if ks[p] > 0:
            cols += [f'max(score_{p}) AS score_{p}', f'min(rank_{p}) AS rank_{p}']
        else:
            cols += [f'CAST(NULL AS FLOAT) AS score_{p}', f'CAST(NULL AS FLOAT) AS rank_{p}']
    con.execute(f"""CREATE TABLE pairs AS
        SELECT s1, tg, {', '.join(cols)}
        FROM read_parquet('{part_dir}/*.parquet', union_by_name=true) GROUP BY s1, tg""")
    n = con.execute('SELECT count(*) FROM pairs').fetchone()[0]
    log(f'{n:,} candidate pairs after merging passes ({n / len(s1):.1f} per S1)')

    cache_dir = os.path.join(config.CACHE_DIR, split)
    pq = os.path.join(cache_dir, 'candidates.parquet')
    score_cols = ', '.join(f'p.score_{q}, p.rank_{q}' for q in PASSES)
    con.execute(f"""COPY (SELECT s1x.s1_id, tgx.cand_id, {score_cols}
        FROM pairs p JOIN s1x USING (s1) JOIN tgx USING (tg) ORDER BY p.s1, p.tg)
        TO '{pq}' (FORMAT parquet, COMPRESSION zstd)""")

    out_dir = os.path.join(config.OUTPUT_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    tsv = os.path.join(out_dir, 'candidate_pairs.tsv')
    cur = con.execute("""SELECT s1x.s1_id, string_agg(tgx.cand_id, ',' ORDER BY tgx.cand_id)
        FROM s1x LEFT JOIN pairs p USING (s1) LEFT JOIN tgx ON p.tg = tgx.tg
        GROUP BY s1x.s1, s1x.s1_id ORDER BY s1x.s1""")        # every S1 gets a row, in file order
    with open(tsv, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        while rows := cur.fetchmany(50_000):
            f.writelines(f'{a}\t{b or ""}\n' for a, b in rows)
    shutil.rmtree(part_dir, ignore_errors=True)
    shutil.rmtree(os.path.join(cache_dir, 'duckdb_tmp'), ignore_errors=True)
    log(f'wrote {tsv} and {pq}')


def report_by_pass(split, tg):
    """Dev-loop diagnostics: recall of each pass alone, and of the union,
    overall, per country and on non-Latin target names."""
    from evaluate import read_ground_truth
    gt = read_ground_truth(config.split_paths(split)['gt'])
    con = duck(split)
    con.register('truth', pd.DataFrame([(a, b) for a, bs in gt.items() for b in bs], columns=['s1_id', 'cand_id']))
    con.register('tgi', tg[['entity_id', 'country', 'nonlatin']])
    pq = os.path.join(config.CACHE_DIR, split, 'candidates.parquet')
    found = ', '.join(f'avg((c.score_{p} IS NOT NULL)::INT) AS {p}' for p in PASSES)
    q = f"""SELECT {{bucket}} AS bucket, count(*) AS n_true, {found}, avg((c.s1_id IS NOT NULL)::INT) AS "union"
        FROM truth t JOIN tgi ON tgi.entity_id = t.cand_id
        LEFT JOIN read_parquet('{pq}') c ON c.s1_id = t.s1_id AND c.cand_id = t.cand_id {{where}} GROUP BY 1"""
    rep = pd.concat([con.execute(q.format(bucket="'ALL'", where='')).df(),
                     con.execute(q.format(bucket='tgi.country', where='')).df(),
                     con.execute(q.format(bucket="'non-Latin name'", where='WHERE tgi.nonlatin')).df()])
    print('\n== pair recall by pass (true pairs found / true pairs)\n' + rep.to_string(index=False, float_format='%.4f'))


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
    args = ap.parse_args()

    s1, tg = load_split(args.split)
    ks = {'name': args.k_name, 'addr': args.k_addr, 'full': args.k_full}
    part_dir = generate_candidates(args.split, s1, tg, ks, args.max_df, args.workers)
    write_outputs(args.split, s1, tg, part_dir, ks)
    if os.path.exists(config.split_paths(args.split)['gt']):
        report_by_pass(args.split, tg)


if __name__ == '__main__':
    main()
