"""
Pair features for the matcher (stage 1: no model outputs involved).

Input : $BER_CACHE_DIR/<split>/candidates.parquet        (from blocking.py)
Output: $BER_CACHE_DIR/<split>/features/part-*.parquet   one row per candidate pair
        (+ `label` 0/1 when the split has ground truth)

Scales to test (~70M pairs): IDs are joined to integer row indices inside
DuckDB (no 70M-string columns in Python), vectorizers are fitted once, and
features are written in parts of about --chunk pairs, split on Source 1
boundaries so every Source 1 list stays in one part.

Feature groups, and the measured fact each one is for (docs/FINDINGS.md):

  name_*   RapidFuzz ratio / token_set / token_sort / partial / Jaro-Winkler.
           Typos, spacing, word order, truncation.
  *_cos    TF-IDF cosine (name words, name char 3-grams, address words).
           Rare-word agreement ("primary care" is common, "quasaredge" is not).
  *_cov_s1 IDF-weighted share of Source 1's words found in the target, and
  *_cov_tg the reverse. Source 1 is perfectly clean and the noise is one-sided
           (targets add brackets, suffixes, extra numbers), so the two
           directions carry different information.
  num_*    House number / number overlap, leading zeros stripped, compared as
           "contained in": targets add numbers in 18-23% of true pairs, so a
           number mismatch must never be a veto.
  blk_*    Blocking scores and ranks (NaN = that pass did not retrieve it).
  dup_*    Exact-duplicate targets (normalised name+address): 92.5% of
           duplicate groups belong to one Source 1 entity.
  hub/list Number of Source 1 lists containing the target (generic "hub"
           records appear in up to 1,009 lists) and the list length.
  misc     Target source (S2/S3 styles differ), missing address, non-Latin
           name, token counts.

Country is deliberately NOT a feature: France is unseen in training, and a
model keyed on country would route it to an arbitrary branch.

Usage:
  python src/features.py --split local_val
"""

import argparse
import glob
import os
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import read_tsv
from normalize import is_non_latin, norm

_NUM = re.compile(r'\d+')


def log(msg, t0=[time.time()]):
    # current / peak resident memory: test runs close to the laptop's RAM limit
    import resource
    rss = int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**30
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f'[{time.time() - t0[0]:7.1f}s] {msg}   [mem {rss:.1f} GB, peak {peak:.1f} GB]', flush=True)


def features_dir(split):
    return os.path.join(config.CACHE_DIR, split, 'features')


def feature_parts(split):
    parts = sorted(glob.glob(os.path.join(features_dir(split), 'part-*.parquet')))
    if not parts:
        raise FileNotFoundError(f'no feature parts in {features_dir(split)}; run features.py --split {split}')
    return parts


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(split):
    """
    Streams the source files and keeps only what the features need: entity_id,
    normalised name and address, and two flags computed from the raw text.
    Raw columns are never held for the whole split (test: 11.7M records; the
    pandas version sat at 5.9 GB, and freed Python strings are not returned to
    the OS). Reads exactly like read_tsv: tab-split, no quote handling.
    """
    p = config.split_paths(split)

    def read(paths):
        ids, names, addrs, missing, nonlatin = [], [], [], [], []
        for path in paths:
            with open(path, encoding='utf-8') as f:
                header = next(f).rstrip('\n').split('\t')
                assert header[:3] == ['entity_id', 'business_name', 'business_address'], header
                for line in f:
                    if line == '\n':
                        continue
                    eid, name, addr = line.rstrip('\n').split('\t')[:3]
                    ids.append(eid)
                    names.append(norm(name))
                    addrs.append(norm(addr))
                    missing.append(addr == '')
                    nonlatin.append(is_non_latin(name))
        return pd.DataFrame({
            'entity_id': np.array(ids, dtype=object), 'name_n': names, 'addr_n': addrs,
            'addr_missing': np.array(missing, np.float32), 'nonlatin': np.array(nonlatin, np.float32)})

    return read([p['s1']]), read([p['s2'], p['s3']])


def _index_path(split):
    return os.path.join(config.CACHE_DIR, split, 'cand_index.npz')


def load_candidate_index(split, s1_ids, tg_ids):
    """Candidate pairs as integer row indices into s1 / tg, sorted by Source 1.
    Cached in cand_index.npz (built by --index-only in its own process on test,
    so the ID join never stacks on top of the records and TF-IDF matrices)."""
    cached = _index_path(split)
    cands = os.path.join(config.CACHE_DIR, split, 'candidates.parquet')
    if os.path.exists(cached) and os.path.getmtime(cached) >= os.path.getmtime(cands):
        with np.load(cached) as z:
            out = {k: z[k] for k in z.files}
        if len(out['i']) and out['i'].max() < len(s1_ids) and out['j'].max() < len(tg_ids):
            return out
    out = _join_candidate_index(split, s1_ids, tg_ids)
    np.savez(cached + '.tmp.npz', **out)
    os.replace(cached + '.tmp.npz', cached)
    return out


def _read_ids(paths):
    ids = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            next(f)
            ids.extend(line.split('\t', 1)[0] for line in f if line != '\n')
    return np.array(ids, dtype=object)


def _join_candidate_index(split, s1_ids, tg_ids):
    import duckdb
    con = duckdb.connect()
    # Capped memory + spill directory: uncapped, this join of ~65M test pairs
    # against ID strings peaked above the laptop's free RAM and was OOM-killed.
    tmp = os.path.join(config.CACHE_DIR, split, 'duckdb_tmp')
    os.makedirs(tmp, exist_ok=True)
    con.execute(f"SET memory_limit='{os.environ.get('BER_DUCKDB_MEM', '3GB')}'")
    con.execute(f"SET temp_directory='{tmp}'")
    con.register('s1x', pd.DataFrame({'id': s1_ids, 'i': np.arange(len(s1_ids), dtype=np.int32)}))
    con.register('tgx', pd.DataFrame({'id': tg_ids, 'j': np.arange(len(tg_ids), dtype=np.int32)}))
    path = os.path.join(config.CACHE_DIR, split, 'candidates.parquet')
    d = con.sql(f"""
        SELECT s1x.i, tgx.j, c.score_addr, c.rank_addr, c.score_full, c.rank_full
        FROM read_parquet('{path}') c
        JOIN s1x ON c.s1_id = s1x.id JOIN tgx ON c.cand_id = tgx.id
        ORDER BY s1x.i, tgx.j""").fetchnumpy()
    out = {}
    for k, a in d.items():
        a = a.filled(np.nan) if np.ma.isMaskedArray(a) else a
        out[k] = a.astype(np.int32 if k in ('i', 'j') else np.float32)   # int32: 65M pairs x 2 saves 0.5 GB
    n_cand = con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    assert len(out['i']) == n_cand, 'some candidate IDs are missing from the split files'
    con.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def load_truth(split):
    p = config.split_paths(split)
    if not os.path.exists(p['gt']):
        return None
    from evaluate import read_ground_truth
    return read_ground_truth(p['gt'])


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def rf(scorer, a, b):
    """Element-wise RapidFuzz scores for two aligned string lists, all cores, 0-1 scale."""
    return process.cpdist(a, b, scorer=scorer, workers=config.N_THREADS, dtype=np.float32) / np.float32(100)


def pair_rowdot(X, Y, i, j, chunk=1_000_000):
    """sum_k X[i_n, k] * Y[j_n, k] for every pair n, without materialising all rows at once."""
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        e = s + chunk
        out[s:e] = np.asarray(X[i[s:e]].multiply(Y[j[s:e]]).sum(axis=1)).ravel()
    return out


# ---------------------------------------------------------------------------
# Memory-bounded TF-IDF (test: ~11.7M records). sklearn's fit() builds a count
# matrix over every record just to learn IDF, and transform() then builds it
# again; together with the other in-memory arrays that exceeded the laptop's
# RAM. Here document frequencies are counted chunk by chunk and the final
# matrices are assembled into preallocated arrays. The IDF formula is sklearn's
# own (smooth_idf, float32), so features are unchanged up to float rounding.
# ---------------------------------------------------------------------------

def _fit_streaming(s1_text, tg_text, kw, chunk=1_000_000):
    from sklearn.feature_extraction.text import CountVectorizer
    df_counts, n = {}, 0
    for series in (s1_text, tg_text):
        for a in range(0, len(series), chunk):
            part = series.iloc[a:a + chunk]
            cv = CountVectorizer(binary=True, dtype=np.float32, **kw)
            try:
                m = cv.fit_transform(part)
            except ValueError:      # chunk with no terms at all
                n += len(part)
                continue
            df = np.bincount(m.indices, minlength=m.shape[1])
            for term, k in cv.vocabulary_.items():
                df_counts[term] = df_counts.get(term, 0) + int(df[k])
            n += len(part)
            del m, cv
    terms = sorted(df_counts)
    vec = TfidfVectorizer(vocabulary={t: k for k, t in enumerate(terms)},
                          binary=True, norm=None, dtype=np.float32, **kw)
    vec.fit([''])                   # sets up the vocabulary; idf_ is replaced below
    df = np.fromiter((df_counts[t] for t in terms), np.float32, len(terms))
    del df_counts
    df += 1.0                       # sklearn: smooth_idf
    idf = np.full_like(df, fill_value=n + 1, dtype=np.float32)
    idf /= df
    np.log(idf, out=idf)
    idf += 1
    vec.idf_ = idf
    return vec


def _transform_chunked(vec, text, chunk=500_000):
    import scipy.sparse as sp
    datas, inds, lens = [], [], []
    for a in range(0, len(text), chunk):
        m = vec.transform(text.iloc[a:a + chunk]).tocsr()
        m.sort_indices()
        datas.append(m.data.astype(np.float32, copy=False))
        inds.append(m.indices.astype(np.int32, copy=False))
        lens.append(np.diff(m.indptr))
        del m
    nnz = sum(len(d) for d in datas)
    data, indices = np.empty(nnz, np.float32), np.empty(nnz, np.int32)
    pos = 0
    for k in range(len(datas)):     # copy chunk by chunk, freeing as we go
        e = pos + len(datas[k])
        data[pos:e], indices[pos:e] = datas[k], inds[k]
        datas[k] = inds[k] = None
        pos = e
    indptr = np.zeros(len(text) + 1, np.int64)
    np.cumsum(np.concatenate(lens), out=indptr[1:])
    return sp.csr_matrix((data, indices, indptr), shape=(len(text), len(vec.vocabulary_)))


def _row_sums(m, square=False, chunk=1_000_000):
    out = np.zeros(m.shape[0], np.float32)
    for a in range(0, m.shape[0], chunk):
        b = min(a + chunk, m.shape[0])
        lo, hi = m.indptr[a], m.indptr[b]
        d = m.data[lo:hi]
        if square:
            d = d * d
        seg = np.diff(m.indptr[a:b + 1])
        rows = np.repeat(np.arange(b - a), seg)
        out[a:b] = np.bincount(rows, weights=d, minlength=b - a)
    return out


class TfidfField:
    """
    Binary IDF vectors of one field for Source 1 and targets. IDF is fitted on
    this split's Source 1 + targets (unlabeled statistics only; on test this is
    the one documented use of test inputs).
    """

    def __init__(self, s1_text, tg_text, analyzer):
        kw = dict(analyzer='char_wb', ngram_range=(3, 3)) if analyzer == 'char' else \
            dict(analyzer='word', token_pattern=r'\S+')
        vec = _fit_streaming(s1_text, tg_text, kw)
        self.word = analyzer == 'word'
        # One matrix per side (memory: test has ~10M targets). With binary tf, a
        # shared term has the same weight idf_k on both sides, so for a pair the
        # element-wise product holds idf_k^2 on shared terms: its sum is the dot
        # product, and the sum of its square roots is the shared idf mass.
        self.X, self.Y = _transform_chunked(vec, s1_text), _transform_chunked(vec, tg_text)
        self.nx = np.sqrt(_row_sums(self.X, square=True))
        self.ny = np.sqrt(_row_sums(self.Y, square=True))
        self.xs = _row_sums(self.X)     # total idf mass per record
        self.ys = _row_sums(self.Y)

    def pair_features(self, prefix, i, j, chunk=1_000_000):
        dot = np.empty(len(i), np.float32)
        shared = np.empty(len(i), np.float32)
        for s in range(0, len(i), chunk):
            P = self.X[i[s:s + chunk]].multiply(self.Y[j[s:s + chunk]]).tocsr()
            dot[s:s + chunk] = np.asarray(P.sum(axis=1)).ravel()
            shared[s:s + chunk] = np.asarray(P.sqrt().sum(axis=1)).ravel()
        with np.errstate(invalid='ignore', divide='ignore'):
            out = {f'{prefix}_cos': np.nan_to_num(dot / (self.nx[i] * self.ny[j])).astype(np.float32)}
            if self.word:
                out[f'{prefix}_cov_s1'] = (shared / self.xs[i]).astype(np.float32)   # Source 1 words in target
                out[f'{prefix}_cov_tg'] = (shared / self.ys[j]).astype(np.float32)   # target words in Source 1
        return out


def _nums(s, _cache={}):
    # unique numbers as a shared, deduplicated tuple (frozensets for 11.7M test
    # records cost several GB); set semantics are kept in number_features
    t = tuple(sorted({x.lstrip('0') or '0' for x in _NUM.findall(s)}))
    return _cache.setdefault(t, t)


def _house(s):
    m = _NUM.search(s)
    return (m.group().lstrip('0') or '0') if m else ''


def number_features(s1_nums, tg_nums, s1_house, i, j):
    n = len(i)
    hn_in = np.full(n, np.nan, np.float32)
    frac_in = np.full(n, np.nan, np.float32)
    extra = np.zeros(n, np.float32)
    for k in range(n):
        a, b = s1_nums[i[k]], tg_nums[j[k]]
        h = s1_house[i[k]]
        if h and b:
            hn_in[k] = h in b
        if a and b:
            frac_in[k] = sum(1 for x in a if x in b) / len(a)      # |a & b| / |a|
        extra[k] = sum(1 for x in b if x not in a)                  # |b - a|
    return {
        'num_house_in_tg': hn_in,          # NaN when either side has no number
        'num_frac_s1_in_tg': frac_in,
        'num_extra_tg': extra,
        'num_n_s1': np.array([len(s1_nums[x]) for x in i], np.float32),
        'num_n_tg': np.array([len(tg_nums[x]) for x in j], np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_features(split, chunk):
    s1, tg = load_records(split)
    log(f'records: S1 {len(s1):,}, targets {len(tg):,}')
    s1_ids, tg_ids = s1['entity_id'].to_numpy(), tg['entity_id'].to_numpy()
    tg_addr_missing, tg_nonlatin = tg['addr_missing'].to_numpy(), tg['nonlatin'].to_numpy()
    tg_is_s3 = np.fromiter((t.startswith('S3-') for t in tg_ids), np.float32, len(tg_ids))
    fields = {}
    for name, col, analyzer in (('namew', 'name_n', 'word'), ('namec', 'name_n', 'char'), ('addrw', 'addr_n', 'word')):
        fields[name] = TfidfField(s1[col], tg[col], analyzer)
        log(f'tf-idf {name}: {fields[name].X.shape[1]:,} terms, {fields[name].X.nnz + fields[name].Y.nnz:,} nonzeros')
    log('tf-idf fitted')

    c = load_candidate_index(split, s1_ids, tg_ids)
    I, J = c['i'], c['j']
    log(f'candidates: {len(I):,}')

    s1n, tgn = s1['name_n'].to_numpy(), tg['name_n'].to_numpy()
    s1a, tga = s1['addr_n'].to_numpy(), tg['addr_n'].to_numpy()
    s1_nums, tg_nums = [_nums(s) for s in s1a], [_nums(s) for s in tga]
    s1_house = [_house(s) for s in s1a]
    ntok = lambda arr: np.fromiter((len(s.split()) for s in arr), np.float32, len(arr))
    s1_name_ntok, tg_name_ntok, s1_addr_ntok, tg_addr_ntok = ntok(s1n), ntok(tgn), ntok(s1a), ntok(tga)

    # global counts (need all pairs, not just a part)
    # duplicate groups = identical (name, address); from two integer codes, not a
    # concatenated 10M-string column
    kn, ka = pd.factorize(tg['name_n'])[0], pd.factorize(tg['addr_n'])[0]
    key = pd.factorize(kn.astype(np.int64) * (int(ka.max()) + 1) + ka)[0]
    dup_size = np.bincount(key)
    hub = np.bincount(J, minlength=len(tg))
    list_n = np.bincount(I, minlength=len(s1))
    truth = load_truth(split)

    out_dir = features_dir(split)
    import duckdb
    import json
    # part boundaries on Source 1 changes, roughly `chunk` pairs each
    change = np.r_[0, np.flatnonzero(np.diff(I)) + 1, len(I)]
    bounds, last = [0], 0
    for b in change[1:]:
        if b - last >= chunk or b == len(I):
            bounds.append(b)
            last = b
    # Resumable: parts from an earlier, stopped run over the same candidates are
    # kept (same pair count and part bounds); anything else starts clean.
    manifest = {'n_pairs': int(len(I)), 'bounds': [int(b) for b in bounds]}
    man_path = os.path.join(out_dir, 'manifest.json')
    try:
        with open(man_path) as fh:
            same = json.load(fh) == manifest
    except (OSError, ValueError):
        same = False
    if not same:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        with open(man_path, 'w') as fh:
            json.dump(manifest, fh)
    for part, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        path = os.path.join(out_dir, f'part-{part:03d}.parquet')
        if os.path.exists(path):
            log(f'part {part}: already done, kept')
            continue
        i, j = I[s:e], J[s:e]
        f = {}
        a, b = list(s1n[i]), list(tgn[j])
        f['name_ratio'] = rf(fuzz.ratio, a, b)
        f['name_tset'] = rf(fuzz.token_set_ratio, a, b)
        f['name_tsort'] = rf(fuzz.token_sort_ratio, a, b)
        f['name_partial'] = rf(fuzz.partial_ratio, a, b)
        f['name_jw'] = process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=config.N_THREADS, dtype=np.float32)
        f['name_exact'] = (s1n[i] == tgn[j]).astype(np.float32)
        a, b = list(s1a[i]), list(tga[j])
        f['addr_ratio'] = rf(fuzz.ratio, a, b)
        f['addr_tset'] = rf(fuzz.token_set_ratio, a, b)
        f['addr_partial'] = rf(fuzz.partial_ratio, a, b)
        f['addr_exact'] = (s1a[i] == tga[j]).astype(np.float32)
        del a, b
        for prefix, fld in fields.items():
            f.update(fld.pair_features(prefix, i, j))
        f.update(number_features(s1_nums, tg_nums, s1_house, i, j))
        f['blk_score_addr'] = c['score_addr'][s:e]
        f['blk_rank_addr'] = c['rank_addr'][s:e]
        f['blk_score_full'] = c['score_full'][s:e]
        f['blk_rank_full'] = c['rank_full'][s:e]
        f['dup_group_size'] = dup_size[key[j]].astype(np.float32)
        f['dup_in_list'] = pd.Series(key[j]).groupby([i, key[j]]).transform('size').to_numpy(np.float32)
        f['hub_n_lists'] = hub[j].astype(np.float32)
        f['list_n_cands'] = list_n[i].astype(np.float32)
        f['tg_is_s3'] = tg_is_s3[j]
        f['tg_addr_missing'] = tg_addr_missing[j]
        f['tg_name_nonlatin'] = tg_nonlatin[j]
        f['s1_name_ntok'] = s1_name_ntok[i]
        f['tg_name_ntok'] = tg_name_ntok[j]
        f['s1_addr_ntok'] = s1_addr_ntok[i]
        f['tg_addr_ntok'] = tg_addr_ntok[j]

        df = pd.DataFrame(f)
        # row indices into the split's Source 1 file and its S2+S3 files (in that order):
        # the matcher works on these, and only maps back to ID strings for output
        df.insert(0, 'tg_idx', j.astype(np.int32))
        df.insert(0, 's1_idx', i.astype(np.int32))
        df.insert(0, 'cand_id', tg_ids[j])
        df.insert(0, 's1_id', s1_ids[i])
        if truth is not None:
            df['label'] = np.fromiter((t in truth.get(s1, ()) for s1, t in zip(df['s1_id'], df['cand_id'])),
                                      np.int8, len(df))
        duckdb.from_df(df).write_parquet(path + '.tmp', compression='zstd')
        os.replace(path + '.tmp', path)      # a stopped run never leaves a half-written part
        log(f'part {part}: {len(df):,} pairs, {df.shape[1] - 4 - (truth is not None)} features -> {path}')
    return len(I)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='local_val', choices=config.SPLIT_NAMES)
    ap.add_argument('--chunk', type=int, default=int(os.environ.get('BER_FEATURE_CHUNK', 5_000_000)),
                    help='pairs per output part; peak memory grows ~0.65 GB per 1M (env BER_FEATURE_CHUNK)')
    ap.add_argument('--index-only', action='store_true',
                    help='only build the cached candidate index (low memory; run before a big split)')
    args = ap.parse_args()
    if args.index_only:
        p = config.split_paths(args.split)
        s1_ids, tg_ids = _read_ids([p['s1']]), _read_ids([p['s2'], p['s3']])
        n = len(load_candidate_index(args.split, s1_ids, tg_ids)['i'])
        log(f'candidate index: {n:,} pairs -> {_index_path(args.split)}')
        return
    build_features(args.split, args.chunk)
    # completion marker, written last: run_pipeline.sh only skips features when it exists
    open(os.path.join(features_dir(args.split), '_DONE'), 'w').close()
    import duckdb
    parts = os.path.join(features_dir(args.split), 'part-*.parquet')
    has_label = 'label' in [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{parts}')").fetchall()]
    if has_label:
        n, pos = duckdb.sql(f"SELECT count(*), sum(label) FROM read_parquet('{parts}')").fetchone()
        print(f'positives: {int(pos):,} of {n:,} pairs ({pos / n:.3%})')


if __name__ == '__main__':
    main()
