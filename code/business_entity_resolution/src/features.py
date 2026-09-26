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
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import read_tsv
from normalize import is_non_latin, norm

_NUM = re.compile(r'\d+')


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:7.1f}s] {msg}', flush=True)


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
    p = config.split_paths(split)
    s1 = read_tsv(p['s1'])
    tg = pd.concat([read_tsv(p['s2']), read_tsv(p['s3'])], ignore_index=True)
    for df in (s1, tg):
        df['name_n'] = df['business_name'].map(norm)
        df['addr_n'] = df['business_address'].map(norm)
    return s1, tg


def load_candidate_index(split, s1_ids, tg_ids):
    """Candidate pairs as integer row indices into s1 / tg, sorted by Source 1."""
    import duckdb
    con = duckdb.connect()
    # bounded (spills to disk): the default is 80% of RAM, on top of everything else
    con.execute(f"SET memory_limit = '{os.environ.get('BER_DUCKDB_MEMORY', '3GB')}'")
    con.execute(f"SET temp_directory = '{os.path.join(config.CACHE_DIR, split, 'duckdb_tmp')}'")
    con.execute('SET preserve_insertion_order = false')
    con.register('s1x', pd.DataFrame({'id': s1_ids, 'i': np.arange(len(s1_ids), dtype=np.int32)}))
    con.register('tgx', pd.DataFrame({'id': tg_ids, 'j': np.arange(len(tg_ids), dtype=np.int32)}))
    path = os.path.join(config.CACHE_DIR, split, 'candidates.parquet')
    # streamed in chunks and sorted in numpy: a DuckDB ORDER BY + full fetch
    # of ~70M rows (test) would have to fit under memory_limit
    res = con.execute(f"""
        SELECT s1x.i, tgx.j, c.score_addr, c.rank_addr, c.score_full, c.rank_full
        FROM read_parquet('{path}') c
        JOIN s1x ON c.s1_id = s1x.id JOIN tgx ON c.cand_id = tgx.id""")
    cols = {k: [] for k in ('i', 'j', 'score_addr', 'rank_addr', 'score_full', 'rank_full')}
    while True:
        chunk = res.fetch_df_chunk(500)          # 500 vectors x 2048 rows ~ 1M rows
        if chunk.empty:
            break
        for k in cols:                           # float nulls arrive as NaN
            cols[k].append(chunk[k].to_numpy(np.int32 if k in ('i', 'j') else np.float32))
    out = {k: np.concatenate(v) if v else np.empty(0, np.int32 if k in ('i', 'j') else np.float32)
           for k, v in cols.items()}
    del cols
    order = np.lexsort((out['j'], out['i']))
    out = {k: a[order] for k, a in out.items()}
    n_cand = con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    assert len(out['i']) == n_cand, 'some candidate IDs are missing from the split files'
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
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / np.float32(100)


def pair_rowdot(X, Y, i, j, chunk=1_000_000):
    """sum_k X[i_n, k] * Y[j_n, k] for every pair n, without materialising all rows at once."""
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        e = s + chunk
        out[s:e] = np.asarray(X[i[s:e]].multiply(Y[j[s:e]]).sum(axis=1)).ravel()
    return out


def _with_data(M, data):
    """CSR matrix with M's sparsity structure (shared, not copied) and new values."""
    return sp.csr_matrix((data, M.indices, M.indptr), shape=M.shape, copy=False)


def _l2_data(M):
    """Values of M with every row scaled to unit L2 norm (empty rows stay empty)."""
    # per-row sum of squares; a trailing 0 keeps every row start a valid index,
    # and reduceat gives an empty row one stray element, so zero those after
    sq = np.add.reduceat(np.append(M.data.astype(np.float64) ** 2, 0.0), M.indptr[:-1])
    sq[np.diff(M.indptr) == 0] = 0
    norms = np.sqrt(sq)
    norms[norms == 0] = 1
    return (M.data / np.repeat(norms, np.diff(M.indptr))).astype(np.float32)


class TfidfField:
    """
    Binary IDF vectors of one field for Source 1 and targets. IDF is fitted on
    this split's Source 1 + targets (unlabeled statistics only; on test this is
    the one documented use of test inputs).
    """

    def __init__(self, s1_text, tg_text, analyzer):
        kw = dict(analyzer='char_wb', ngram_range=(3, 3)) if analyzer == 'char' else \
            dict(analyzer='word', token_pattern=r'\S+')
        vec = TfidfVectorizer(binary=True, norm=None, dtype=np.float32, **kw)
        vec.fit(pd.concat([s1_text, tg_text]))
        self.word = analyzer == 'word'
        Xw, Yw = vec.transform(s1_text).tocsr(), vec.transform(tg_text).tocsr()
        # the normalised and binary variants share Xw/Yw's index arrays
        # (only a new data array each): saves several GB on test
        self.Xn, self.Yn = _with_data(Xw, _l2_data(Xw)), _with_data(Yw, _l2_data(Yw))
        if self.word:
            self.Xw, self.Yw = Xw, Yw
            self.Xb = _with_data(Xw, np.ones_like(Xw.data))
            self.Yb = _with_data(Yw, np.ones_like(Yw.data))
            self.xs = np.asarray(Xw.sum(axis=1)).ravel()
            self.ys = np.asarray(Yw.sum(axis=1)).ravel()

    def pair_features(self, prefix, i, j):
        out = {f'{prefix}_cos': pair_rowdot(self.Xn, self.Yn, i, j)}
        if self.word:
            with np.errstate(invalid='ignore', divide='ignore'):
                out[f'{prefix}_cov_s1'] = pair_rowdot(self.Xw, self.Yb, i, j) / self.xs[i]  # Source 1 words in target
                out[f'{prefix}_cov_tg'] = pair_rowdot(self.Yw, self.Xb, j, i) / self.ys[j]  # target words in Source 1
        return out


def _nums(s):
    return frozenset(t.lstrip('0') or '0' for t in _NUM.findall(s))


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
            frac_in[k] = len(a & b) / len(a)
        extra[k] = len(b - a)
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
    c = load_candidate_index(split, s1_ids, tg_ids)
    I, J = c['i'], c['j']
    log(f'candidates: {len(I):,}')

    # the only uses of the raw text; drop it before the big TF-IDF matrices exist
    tg_addr_missing = (tg['business_address'].to_numpy() == '').astype(np.float32)
    tg_nonlatin = tg['business_name'].map(is_non_latin).to_numpy().astype(np.float32)
    for df in (s1, tg):
        df.drop(columns=['business_name', 'business_address'], inplace=True)

    fields = {
        'namew': TfidfField(s1['name_n'], tg['name_n'], 'word'),
        'namec': TfidfField(s1['name_n'], tg['name_n'], 'char'),
        'addrw': TfidfField(s1['addr_n'], tg['addr_n'], 'word'),
    }
    log('tf-idf fitted')

    s1n, tgn = s1['name_n'].to_numpy(), tg['name_n'].to_numpy()
    s1a, tga = s1['addr_n'].to_numpy(), tg['addr_n'].to_numpy()
    s1_nums, tg_nums = [_nums(s) for s in s1a], [_nums(s) for s in tga]
    s1_house = [_house(s) for s in s1a]
    ntok = lambda arr: np.fromiter((len(s.split()) for s in arr), np.float32, len(arr))
    s1_name_ntok, tg_name_ntok, s1_addr_ntok, tg_addr_ntok = ntok(s1n), ntok(tgn), ntok(s1a), ntok(tga)
    tg_is_s3 = np.char.startswith(tg_ids.astype(str), 'S3-').astype(np.float32)

    # global counts (need all pairs, not just a part)
    key = pd.factorize(tg['name_n'] + '|' + tg['addr_n'])[0]
    dup_size = np.bincount(key)
    hub = np.bincount(J, minlength=len(tg))
    list_n = np.bincount(I, minlength=len(s1))
    truth = load_truth(split)

    out_dir = features_dir(split)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    import duckdb
    # part boundaries on Source 1 changes, roughly `chunk` pairs each
    change = np.r_[0, np.flatnonzero(np.diff(I)) + 1, len(I)]
    bounds, last = [0], 0
    for b in change[1:]:
        if b - last >= chunk or b == len(I):
            bounds.append(b)
            last = b
    for part, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        i, j = I[s:e], J[s:e]
        f = {}
        a, b = list(s1n[i]), list(tgn[j])
        f['name_ratio'] = rf(fuzz.ratio, a, b)
        f['name_tset'] = rf(fuzz.token_set_ratio, a, b)
        f['name_tsort'] = rf(fuzz.token_sort_ratio, a, b)
        f['name_partial'] = rf(fuzz.partial_ratio, a, b)
        f['name_jw'] = process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
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
            df['label'] = np.fromiter((t in truth[s1] for s1, t in zip(df['s1_id'], df['cand_id'])),
                                      np.int8, len(df))
        path = os.path.join(out_dir, f'part-{part:03d}.parquet')
        duckdb.from_df(df).write_parquet(path, compression='zstd')
        log(f'part {part}: {len(df):,} pairs, {df.shape[1] - 4 - (truth is not None)} features -> {path}')
    return len(I)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='local_val', choices=config.SPLIT_NAMES)
    ap.add_argument('--chunk', type=int, default=5_000_000, help='pairs per output part')
    args = ap.parse_args()
    build_features(args.split, args.chunk)
    import duckdb
    parts = os.path.join(features_dir(args.split), 'part-*.parquet')
    has_label = 'label' in [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{parts}')").fetchall()]
    if has_label:
        n, pos = duckdb.sql(f"SELECT count(*), sum(label) FROM read_parquet('{parts}')").fetchone()
        print(f'positives: {int(pos):,} of {n:,} pairs ({pos / n:.3%})')


if __name__ == '__main__':
    main()
