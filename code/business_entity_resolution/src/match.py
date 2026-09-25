"""
Matcher v1: two-stage LightGBM -> one Source 1 per target -> expected-F0.5 decoding.

Why each step (evidence in docs/REPORT.md and docs/FINDINGS.md):

  stage 1  LightGBM on pair features (features.py). A pair classifier is the
           right *scoring* primitive (Ditto casts entity matching as
           sequence-pair classification), but it judges each pair alone.
  stage 2  LightGBM on stage-1 features + *context* built from stage-1
           probabilities: where the pair ranks in its Source 1 list, and among
           all Source 1 entities competing for the same target (margin over
           the runner-up). This is the cheap equivalent of the "select"
           strategy (Wang et al., COLING 2025), which beat independent pair
           matching by ~16 F1 because candidates compete.
  assign   Each target goes only to the Source 1 entity with its highest
           probability. Every S2/S3 record matches at most one Source 1
           entity and Source 1 is unconstrained, so per-target argmax is the
           exact optimum of the assignment; no Hungarian step is needed.
  decode   Per Source 1 entity, pick the k (0 = predict nothing) that
           maximises *expected* F0.5 under the calibrated probabilities
           (decision-theoretic F-measure optimisation: Ye et al. ICML 2012,
           Waegeman et al. JMLR 2014). k = 0 is how singletons are caught.
           Computed by Monte-Carlo over the top candidates. A tuned global
           threshold is kept as a baseline and fallback.
  caps     At most 5 matches from S2 and 6 from S3 per entity (train maxima).

Memory: pairs are keyed by integer row indices (s1_idx / tg_idx, written by
features.py), never by ID strings; features are one float32 matrix; LightGBM
bins the data once per stage and folds are subsets of that Dataset (binning
uses feature values only, never labels, so this does not leak). --predict
streams the feature parts twice, so test (~70M pairs) never has to fit in RAM.

Modes:
  --cv       labelled split (local_val): 2-fold cross-fit grouped by Source 1,
             out-of-fold probabilities -> decode -> score. Honest dev number.
  --fit      train final models on all rows of a labelled split and save them,
             with the decoding setting chosen by --cv.
  --predict  apply saved models to any split (e.g. test).

Usage:
  python src/match.py --split local_val --cv
  python src/match.py --split local_val --fit
  python src/match.py --split test --predict
"""

import argparse
import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from features import feature_parts

NON_FEATURES = ['s1_id', 'cand_id', 's1_idx', 'tg_idx', 'label']
CONTEXT = ['p1', 's1_rank', 's1_gap', 's1_top1', 's1_margin12', 's1_sum', 's1_n50',
           't_rank', 't_margin', 't_top1', 't_n50']
CAPS = {0: 5, 1: 6}          # source (0 = S2, 1 = S3) -> max matches per Source 1 entity
MODEL_DIR = os.path.join(config.CACHE_DIR, 'models')

LGB_PARAMS = dict(objective='binary', learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=os.cpu_count(), verbose=-1, seed=26)


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:7.1f}s] {msg}', flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_part(path, feats=None):
    """One feature part -> (index/label columns dict, float32 feature matrix, feature names)."""
    import duckdb
    names = [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()]
    feats = feats or [c for c in names if c not in NON_FEATURES]
    meta_cols = [c for c in ('s1_idx', 'tg_idx', 'tg_is_s3', 'label') if c in names]
    extra = [c for c in meta_cols if c not in feats]          # tg_is_s3 is also a feature
    d = duckdb.sql(f"SELECT {', '.join(extra + feats)} FROM read_parquet('{path}')").fetchnumpy()
    meta = {c: np.asarray(d[c]) for c in meta_cols}
    for c in extra:
        d.pop(c)
    X = np.empty((len(meta['s1_idx']), len(feats)), np.float32)
    for k, c in enumerate(feats):
        a = d.pop(c)
        X[:, k] = a.filled(np.nan) if np.ma.isMaskedArray(a) else a
    return meta, X, feats


class Pairs:
    """All candidate pairs of one split in memory (fine for local_val-sized splits)."""

    def __init__(self, split, feats=None):
        metas, Xs = [], []
        for path in feature_parts(split):
            meta, X, self.feats = read_part(path, feats)
            metas.append(meta)
            Xs.append(X)
        self.X = np.concatenate(Xs) if len(Xs) > 1 else Xs[0]
        del Xs
        self.s1_code = np.concatenate([m['s1_idx'] for m in metas]).astype(np.int64)
        self.t_code = np.concatenate([m['tg_idx'] for m in metas]).astype(np.int64)
        self.src = np.concatenate([m['tg_is_s3'] for m in metas]).astype(np.int8)
        self.label = np.concatenate([m['label'] for m in metas]).astype(np.int8) if 'label' in metas[0] else None

    def __len__(self):
        return len(self.s1_code)

    def row_fold(self, split, k, seed):
        """Deterministic fold per Source 1 entity (all its pairs stay together)."""
        s1_ids, _ = split_ids(split)
        f = np.array([int(hashlib.md5(f'{seed}:{u}'.encode()).hexdigest()[:8], 16) % k for u in s1_ids])
        return f[self.s1_code]


def split_ids(split):
    """Entity-ID arrays in the row order features.py indexes (S1; then S2 + S3)."""
    from data_loader import read_tsv
    p = config.split_paths(split)
    s1 = read_tsv(p['s1'], usecols=['entity_id'])['entity_id'].to_numpy()
    tg = np.concatenate([read_tsv(p[k], usecols=['entity_id'])['entity_id'].to_numpy() for k in ('s2', 's3')])
    return s1, tg


# ---------------------------------------------------------------------------
# Group statistics (vectorised; groups = Source 1 lists or target competitions)
# ---------------------------------------------------------------------------

def group_stats(code, p):
    """Per row: 1-based rank of p within its group (desc), group top-1, top-2,
    sum, and number of rows with p > 0.5."""
    order = np.lexsort((-p, code))
    cs, ps = code[order], p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
    sizes = np.diff(np.r_[starts, len(p)])
    gid = np.repeat(np.arange(len(starts)), sizes)
    rank = np.empty(len(p), np.float32)
    rank[order] = np.arange(len(p)) - starts[gid] + 1
    top1 = ps[starts]
    top2 = np.where(sizes > 1, ps[np.minimum(starts + 1, len(p) - 1)], 0)
    gsum = np.add.reduceat(ps, starts)
    gn50 = np.add.reduceat((ps > 0.5).astype(np.float32), starts)
    row_g = np.empty(len(p), np.int64)
    row_g[order] = gid
    return rank, top1[row_g], top2[row_g], gsum[row_g], gn50[row_g]


def context_features(s1_code, t_code, p1):
    """Stage-2 context, columns in CONTEXT order."""
    r, t1, t2, s, n50 = group_stats(s1_code, p1)
    cols = [p1, r, t1 - p1, t1, t1 - t2, s, n50]
    r, t1, t2, s, n50 = group_stats(t_code, p1)
    other_best = np.where(r == 1, t2, t1)          # best competing Source 1 for this target
    cols += [r, p1 - other_best, t1, n50]
    return np.column_stack(cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def predict(model, X, rows=None, chunk=1_000_000):
    rows = np.arange(len(X)) if rows is None else rows
    out = np.empty(len(rows), np.float32)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = model.predict(X[rows[s:s + chunk]], num_threads=os.cpu_count())
    return out


def fit_on(full, rows, es_fold, rounds=3000):
    """Train on `rows` of a constructed Dataset; early-stop on its es_fold==0 part."""
    tr, va = rows[es_fold[rows] != 0], rows[es_fold[rows] == 0]
    return lgb.train(LGB_PARAMS, full.subset(tr), rounds, valid_sets=[full.subset(va)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def cross_fit(pairs, X, folds, es_fold, k=2):
    """Out-of-fold probabilities for every row, plus the last fold's model."""
    full = lgb.Dataset(X, pairs.label, params=LGB_PARAMS, free_raw_data=False).construct()
    oof = np.zeros(len(pairs), np.float32)
    for f in range(k):
        te = np.flatnonzero(folds == f)
        m = fit_on(full, np.flatnonzero(folds != f), es_fold)
        oof[te] = predict(m, X, te)
        log(f'  fold {f}: {m.best_iteration} trees')
    return oof, m


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def assign_and_cap(s1_code, t_code, src, p):
    """Rows still eligible after one-Source-1-per-target and per-source caps."""
    keep = group_stats(t_code, p)[0] == 1
    for source, cap in CAPS.items():
        is_src = keep & (src == source)
        r = group_stats(np.where(is_src, s1_code, -1), np.where(is_src, p, -1.0))[0]
        keep &= ~(is_src & (r > cap))
    return keep


def decode_threshold(p, keep, t):
    return keep & (p >= t)


def decode_expected_f(s1_code, p, keep, n_max=15, samples=512, seed=0, chunk=4096):
    """
    For each Source 1 entity choose k in 0..n_max maximising E[F0.5] when
    predicting its top-k eligible candidates, with the match indicators
    ~ independent Bernoulli(p). E[F0.5] is estimated with shared Monte-Carlo
    draws (common random numbers across k).
    """
    idx = np.flatnonzero(keep & (p > 1e-3))
    order = idx[np.lexsort((-p[idx], s1_code[idx]))]
    code = s1_code[order]
    starts = np.r_[0, np.flatnonzero(np.diff(code)) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    pos = np.arange(len(order)) - np.repeat(starts, sizes)
    in_top = pos < n_max
    P = np.zeros((len(starts), n_max), np.float32)
    P[np.repeat(np.arange(len(starts)), sizes)[in_top], pos[in_top]] = p[order][in_top]

    rng = np.random.default_rng(seed)
    best_k = np.zeros(len(starts), np.int64)
    ks = np.arange(1, n_max + 1, dtype=np.float32)
    for s in range(0, len(P), chunk):
        Pc = P[s:s + chunk]
        draws = rng.random((len(Pc), samples, n_max), dtype=np.float32) < Pc[:, None, :]
        tp = np.cumsum(draws, axis=2, dtype=np.float32)          # TP if we predict top-k
        n_true = tp[..., -1:]                                      # true matches among the top n_max
        ef = np.empty((len(Pc), n_max + 1), np.float32)
        ef[:, 0] = (n_true[..., 0] == 0).mean(axis=1)             # predict nothing
        ef[:, 1:] = (1.25 * tp / (ks + 0.25 * n_true)).mean(axis=1)
        best_k[s:s + chunk] = ef.argmax(axis=1)
    chosen = np.zeros(len(p), bool)
    chosen[order[pos < np.repeat(best_k, sizes)]] = True
    return chosen


def logit_shift(p, b):
    """Shift probabilities in logit space: b < 0 makes decoding stricter."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return (1 / (1 + np.exp(-(np.log(q / (1 - q)) + b)))).astype(np.float32)


def decode(s1_code, t_code, src, p, method, param):
    keep = assign_and_cap(s1_code, t_code, src, p)
    if method == 'threshold':
        return decode_threshold(p, keep, param)
    return decode_expected_f(s1_code, logit_shift(p, param), keep)


# ---------------------------------------------------------------------------
# Output and scoring
# ---------------------------------------------------------------------------

def write_matches(split, s1_code, t_code, chosen):
    s1_ids, tg_ids = split_ids(split)
    sel = pd.DataFrame({'s1': s1_code[chosen], 'cand_id': tg_ids[t_code[chosen]]})
    lists = sel.groupby('s1')['cand_id'].agg(lambda x: ','.join(sorted(x)))
    lists = lists.reindex(np.arange(len(s1_ids)), fill_value='')   # every Source 1 gets a row
    out_dir = os.path.join(config.OUTPUT_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'matching_results.tsv')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id, ids in zip(s1_ids, lists.to_numpy()):
            f.write(f'{s1_id}\t{ids}\n')
    return path


class Scorer:
    """Fast in-memory F0.5 for decoding sweeps (same metric as evaluate.py)."""

    def __init__(self, split, pairs):
        from evaluate import read_country_map, read_ground_truth
        p = config.split_paths(split)
        self.split, self.pairs = split, pairs
        self.gt = read_ground_truth(p['gt'])
        self.countries = read_country_map(p['s1'])
        self.s1_ids, self.tg_ids = split_ids(split)
        self.nt = np.array([len(self.gt[s]) for s in self.s1_ids], float)   # every Source 1, incl. no candidates

    def f05(self, chosen):
        codes, lab = self.pairs.s1_code[chosen], self.pairs.label[chosen]
        n = len(self.nt)
        tp = np.bincount(codes, weights=lab, minlength=n)
        npred = np.bincount(codes, minlength=n).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            f = np.where(self.nt == 0, (npred == 0).astype(float),
                         1.25 * tp / (1.25 * tp + 0.25 * (self.nt - tp) + (npred - tp)))
        return float(np.nan_to_num(f).mean())

    def full_report(self, chosen, title):
        from evaluate import MATCHING_KEYS, print_report, score_matching
        sel = pd.DataFrame({'s1': self.s1_ids[self.pairs.s1_code[chosen]],
                            'c': self.tg_ids[self.pairs.t_code[chosen]]})
        preds = sel.groupby('s1')['c'].agg(set).to_dict()
        rep = score_matching(self.gt, preds, self.countries)
        print_report(title, rep, MATCHING_KEYS)
        return rep


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_cv(split):
    pairs = Pairs(split)
    y = pairs.label
    log(f'{len(pairs):,} pairs, {len(pairs.feats)} base features, {y.mean():.2%} positive')
    folds, es_fold = pairs.row_fold(split, 2, 'mlc26-cv'), pairs.row_fold(split, 10, 'mlc26-es')

    log('stage 1 (pair features)')
    p1, _ = cross_fit(pairs, pairs.X, folds, es_fold)
    X2 = np.hstack([pairs.X, context_features(pairs.s1_code, pairs.t_code, p1)])
    log('stage 2 (+ list and competition context)')
    p2, m2 = cross_fit(pairs, X2, folds, es_fold)
    del X2

    from sklearn.metrics import log_loss, roc_auc_score
    for name, p in (('stage 1', p1), ('stage 2', p2)):
        log(f'{name}: AUC {roc_auc_score(y, p):.5f}  logloss {log_loss(y, p):.5f}')
    bins = pd.cut(p2, [0, .1, .3, .5, .7, .9, .97, 1.0001])
    calib = pd.DataFrame({'p': p2, 'y': y}).groupby(bins, observed=True).agg(
        n=('y', 'size'), mean_p=('p', 'mean'), frac_pos=('y', 'mean'))
    print('\n== calibration of stage-2 out-of-fold probabilities\n' + calib.to_string(float_format='%.4f'))

    sc = Scorer(split, pairs)
    args = (pairs.s1_code, pairs.t_code, pairs.src)
    results = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        results.append(('threshold', t, sc.f05(decode(*args, p2, 'threshold', t))))
    for b in (-1.0, -0.5, 0.0, 0.5):
        results.append(('expected_f', b, sc.f05(decode(*args, p2, 'expected_f', b))))
    res = pd.DataFrame(results, columns=['method', 'param', 'f05'])
    print('\n== decoding sweep (out-of-fold F0.5)\n' + res.to_string(index=False, float_format='%.5f'))
    # ablations: what the structural steps are worth
    no_assign = decode_expected_f(pairs.s1_code, p2, np.ones(len(pairs), bool))
    print(f'ablation, expected_f without one-Source-1-per-target and caps: {sc.f05(no_assign):.5f}')
    print(f'ablation, stage-1 probabilities only (no context features):   '
          f'{sc.f05(decode(*args, p1, "expected_f", 0.0)):.5f}')

    best = res.loc[res['f05'].idxmax()]
    chosen = decode(*args, p2, best['method'], best['param'])
    sc.full_report(chosen, f'Matching, out-of-fold ({split})')
    path = write_matches(split, pairs.s1_code, pairs.t_code, chosen)
    choice = {'method': best['method'], 'param': float(best['param']), 'cv_f05': float(best['f05']), 'split': split}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'decode.json'), 'w') as f:
        json.dump(choice, f, indent=2)
    log(f'best: {choice} -> {path}')

    imp = pd.Series(m2.feature_importance('gain'), index=pairs.feats + CONTEXT).sort_values(ascending=False)
    print('\n== stage-2 feature importance (gain share, top 20)\n' + (imp / imp.sum()).head(20).to_string(float_format='%.4f'))
    np.save(os.path.join(config.CACHE_DIR, split, 'oof_p2.npy'), p2)   # for error analysis


def run_fit(split):
    pairs = Pairs(split)
    all_rows = np.arange(len(pairs))
    folds, es_fold = pairs.row_fold(split, 2, 'mlc26-cv'), pairs.row_fold(split, 10, 'mlc26-es')
    # stage 2 must be trained on *out-of-fold* stage-1 probabilities, as at test time
    p1, _ = cross_fit(pairs, pairs.X, folds, es_fold)
    full = lgb.Dataset(pairs.X, pairs.label, params=LGB_PARAMS, free_raw_data=False).construct()
    m1 = fit_on(full, all_rows, es_fold)
    del full
    X2 = np.hstack([pairs.X, context_features(pairs.s1_code, pairs.t_code, p1)])
    full = lgb.Dataset(X2, pairs.label, params=LGB_PARAMS, free_raw_data=False).construct()
    m2 = fit_on(full, all_rows, es_fold)
    os.makedirs(MODEL_DIR, exist_ok=True)
    m1.save_model(os.path.join(MODEL_DIR, 'stage1.txt'))
    m2.save_model(os.path.join(MODEL_DIR, 'stage2.txt'))
    decode_path = os.path.join(MODEL_DIR, 'decode.json')
    if not os.path.exists(decode_path):   # no --cv run: use the setting --cv chose on local_val
        with open(decode_path, 'w') as f:
            json.dump({'method': 'expected_f', 'param': 0.0, 'cv_f05': None, 'split': 'default'}, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'features.json'), 'w') as f:
        json.dump({'stage1': pairs.feats, 'stage2': pairs.feats + CONTEXT, 'trained_on': split}, f, indent=2)
    log(f'saved models to {MODEL_DIR} ({m1.best_iteration} / {m2.best_iteration} trees)')


def run_predict(split, shift=None):
    """Two streaming passes over the feature parts; only per-row codes and
    probabilities are held for the whole split."""
    with open(os.path.join(MODEL_DIR, 'features.json')) as f:
        fl = json.load(f)
    with open(os.path.join(MODEL_DIR, 'decode.json')) as f:
        choice = json.load(f)
    m1 = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'stage1.txt'))
    m2 = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'stage2.txt'))
    parts = feature_parts(split)

    s1c, tc, src, p1 = [], [], [], []
    for path in parts:
        meta, X, _ = read_part(path, fl['stage1'])
        s1c.append(meta['s1_idx'].astype(np.int64))
        tc.append(meta['tg_idx'].astype(np.int64))
        src.append(meta['tg_is_s3'].astype(np.int8))
        p1.append(predict(m1, X))
    sizes = [len(x) for x in p1]
    s1_code, t_code, src, p1 = (np.concatenate(a) for a in (s1c, tc, src, p1))
    log(f'stage 1: {len(p1):,} pairs in {len(parts)} parts')
    ctx = context_features(s1_code, t_code, p1)
    p2, off = np.empty(len(p1), np.float32), 0
    for path, n in zip(parts, sizes):
        _, X, _ = read_part(path, fl['stage1'])
        p2[off:off + n] = predict(m2, np.hstack([X, ctx[off:off + n]]))
        off += n
    del ctx
    log('stage 2 done')
    param = choice['param'] if shift is None or choice['method'] == 'threshold' else shift
    chosen = decode(s1_code, t_code, src, p2, choice['method'], param)
    path = write_matches(split, s1_code, t_code, chosen)
    log(f'{int(chosen.sum()):,} matches ({choice["method"]}, param {param}) -> {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='local_val', choices=config.SPLIT_NAMES)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cv', action='store_true')
    mode.add_argument('--fit', action='store_true')
    mode.add_argument('--predict', action='store_true')
    ap.add_argument('--shift', type=float, default=None,
                    help='override the logit shift for expected-F decoding at predict time (negative = stricter)')
    args = ap.parse_args()
    if args.cv:
        run_cv(args.split)
    elif args.fit:
        run_fit(args.split)
    else:
        run_predict(args.split, args.shift)


if __name__ == '__main__':
    main()
