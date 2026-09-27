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
                  num_threads=config.N_THREADS, verbose=-1, seed=26)


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


CE_COLS = ['ce', 'ce_s1_rank', 'ce_s1_gap', 'ce_t_margin']


def ce_features(split, s1_code, t_code):
    """
    Cross-encoder member (cross_encoder.py) as stage-2 columns, or None if the
    split has no ce_scores.parquet. Rows outside the scored band get NaN.
    Context mirrors context_features: rank / gap within the Source 1 list and
    margin over the best competing Source 1 for the target (among scored rows).
    """
    path = os.path.join(config.CACHE_DIR, split, 'ce_scores.parquet')
    if not os.path.exists(path):
        return None
    import duckdb
    d = duckdb.sql(f"SELECT s1_idx, tg_idx, ce FROM read_parquet('{path}')").fetchnumpy()
    n_tg = int(max(t_code.max(), np.asarray(d['tg_idx']).max())) + 1
    row_key = pd.Index(s1_code * n_tg + t_code)
    pos = row_key.get_indexer(np.asarray(d['s1_idx'], np.int64) * n_tg + np.asarray(d['tg_idx'], np.int64))
    ok = pos >= 0
    out = np.full((len(s1_code), len(CE_COLS)), np.nan, np.float32)
    rows, ce = pos[ok], np.asarray(d['ce'], np.float32)[ok]
    out[rows, 0] = ce
    r, t1, _, _, _ = group_stats(s1_code[rows], ce)
    out[rows, 1], out[rows, 2] = r, t1 - ce
    r, t1, t2, _, _ = group_stats(t_code[rows], ce)
    out[rows, 3] = ce - np.where(r == 1, t2, t1)
    log(f'cross-encoder scores for {int(ok.sum()):,} rows ({split})')
    return out


def stage2_matrix(split, pairs_or_codes, X, p1):
    """Base features + p1 context (+ cross-encoder columns when available)."""
    s1_code, t_code = pairs_or_codes
    parts = [X, context_features(s1_code, t_code, p1)]
    ce = ce_features(split, s1_code, t_code)
    if ce is not None:
        parts.append(ce)
    return np.hstack(parts), (CONTEXT + (CE_COLS if ce is not None else []))


def context_features(s1_code, t_code, p1):
    """Stage-2 context, columns in CONTEXT order. Columns are written straight
    into one preallocated float32 matrix (the old column_stack + astype made two
    extra full copies: ~9 GB peak on test's 65M pairs, OOM-killed twice)."""
    out = np.empty((len(p1), 11), np.float32)
    out[:, 0] = p1
    r, t1, t2, s, n50 = group_stats(s1_code, p1)
    out[:, 1], out[:, 2], out[:, 3], out[:, 4], out[:, 5], out[:, 6] = r, t1 - p1, t1, t1 - t2, s, n50
    del r, t1, t2, s, n50
    r, t1, t2, s, n50 = group_stats(t_code, p1)
    out[:, 7] = r
    out[:, 8] = p1 - np.where(r == 1, t2, t1)      # margin over the best competing Source 1 for this target
    out[:, 9], out[:, 10] = t1, n50
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def predict(model, X, rows=None, chunk=1_000_000):
    rows = np.arange(len(X)) if rows is None else rows
    out = np.empty(len(rows), np.float32)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = model.predict(X[rows[s:s + chunk]], num_threads=config.N_THREADS)
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

def write_matches(split, s1_code, t_code, chosen, name='matching_results.tsv'):
    s1_ids, tg_ids = split_ids(split)
    sel = pd.DataFrame({'s1': s1_code[chosen], 'cand_id': tg_ids[t_code[chosen]]})
    lists = sel.groupby('s1')['cand_id'].agg(lambda x: ','.join(sorted(x)))
    lists = lists.reindex(np.arange(len(s1_ids)), fill_value='')   # every Source 1 gets a row
    out_dir = os.path.join(config.OUTPUT_DIR, split)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
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
    np.save(os.path.join(config.CACHE_DIR, split, 'oof_p1.npy'), p1)   # band rule for cross_encoder.py
    X2, extra = stage2_matrix(split, (pairs.s1_code, pairs.t_code), pairs.X, p1)
    log(f'stage 2 (+ {", ".join(extra)})')
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

    imp = pd.Series(m2.feature_importance('gain'), index=pairs.feats + extra).sort_values(ascending=False)
    print('\n== stage-2 feature importance (gain share, top 20)\n' + (imp / imp.sum()).head(20).to_string(float_format='%.4f'))
    np.save(os.path.join(config.CACHE_DIR, split, 'oof_p2.npy'), p2)   # for error analysis


def load_oof_p1(split, n):
    """Out-of-fold stage-1 probabilities saved by an earlier --cv / --fit on the
    same feature parts (same folds and seeds, so identical to recomputing), or
    None if missing, the wrong length, or older than the feature parts."""
    path = os.path.join(config.CACHE_DIR, split, 'oof_p1.npy')
    if not os.path.exists(path):
        return None
    parts = feature_parts(split)
    if parts and os.path.getmtime(path) < max(os.path.getmtime(x) for x in parts):
        return None
    p1 = np.load(path)
    return p1 if len(p1) == n else None


def run_fit(split, stage2_only=False):
    pairs = Pairs(split)
    all_rows = np.arange(len(pairs))
    folds, es_fold = pairs.row_fold(split, 2, 'mlc26-cv'), pairs.row_fold(split, 10, 'mlc26-es')
    # stage 2 must be trained on *out-of-fold* stage-1 probabilities, as at test time.
    # --cv already computed them with the same folds and seeds: reuse when valid
    # (saves a full stage-1 cross-fit, ~45 min on scale_val).
    p1 = load_oof_p1(split, len(pairs))
    if p1 is None:
        p1, _ = cross_fit(pairs, pairs.X, folds, es_fold)
        np.save(os.path.join(config.CACHE_DIR, split, 'oof_p1.npy'), p1)   # band rule for cross_encoder.py
    else:
        log('stage 1 out-of-fold probabilities: reused from --cv (oof_p1.npy)')
    if stage2_only:
        # stage 1 does not change when a stage-2 member (e.g. the cross-encoder)
        # is added: keep the saved final stage-1 model instead of retraining it
        m1 = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'stage1.txt'))
        log('stage 1: kept the saved model (--stage2-only)')
    else:
        full = lgb.Dataset(pairs.X, pairs.label, params=LGB_PARAMS, free_raw_data=False).construct()
        m1 = fit_on(full, all_rows, es_fold)
        del full
    X2, extra = stage2_matrix(split, (pairs.s1_code, pairs.t_code), pairs.X, p1)
    full = lgb.Dataset(X2, pairs.label, params=LGB_PARAMS, free_raw_data=False).construct()
    m2 = fit_on(full, all_rows, es_fold)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not stage2_only:
        m1.save_model(os.path.join(MODEL_DIR, 'stage1.txt'))
    m2.save_model(os.path.join(MODEL_DIR, 'stage2.txt'))
    decode_path = os.path.join(MODEL_DIR, 'decode.json')
    if not os.path.exists(decode_path):   # no --cv run: use the setting --cv chose on local_val
        with open(decode_path, 'w') as f:
            json.dump({'method': 'expected_f', 'param': 0.0, 'cv_f05': None, 'split': 'default'}, f, indent=2)
    with open(os.path.join(MODEL_DIR, 'features.json'), 'w') as f:
        json.dump({'stage1': pairs.feats, 'stage2': pairs.feats + extra, 'trained_on': split}, f, indent=2)
    log(f'saved models to {MODEL_DIR} ({m1.current_iteration() if stage2_only else m1.best_iteration} / {m2.best_iteration} trees)')


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

    # Stage 1 is the slow pass (~1 h on test); its output is cached so a failure
    # in stage 2 (e.g. the machine running out of memory) does not repeat it.
    cache = os.path.join(config.CACHE_DIR, split, 'pred_stage1.npz')
    newest_input = max([os.path.getmtime(os.path.join(MODEL_DIR, 'stage1.txt'))] + [os.path.getmtime(x) for x in parts])
    if os.path.exists(cache) and os.path.getmtime(cache) > newest_input:
        with np.load(cache) as z:
            s1_code, t_code, src, p1, sizes = (z[k] for k in ('s1_code', 't_code', 'src', 'p1', 'sizes'))
        s1_code, t_code, sizes = s1_code.astype(np.int64), t_code.astype(np.int64), sizes.tolist()
        log(f'stage 1: reused {len(p1):,} probabilities from {cache}')
    else:
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
        np.savez(cache + '.tmp.npz', s1_code=s1_code.astype(np.int32), t_code=t_code.astype(np.int32),
                 src=src, p1=p1, sizes=np.array(sizes, np.int64))
        os.replace(cache + '.tmp.npz', cache)
        log(f'stage 1 cached -> {cache}')
    ctx = context_features(s1_code, t_code, p1)
    if 'ce' in fl['stage2']:   # model was trained with the cross-encoder member
        ce = ce_features(split, s1_code, t_code)
        assert ce is not None, f'model expects cross-encoder scores: run cross_encoder.py score --split {split}'
        ctx = np.hstack([ctx, ce])
        del ce
    p2, off = np.empty(len(p1), np.float32), 0
    for path, n in zip(parts, sizes):
        _, X, _ = read_part(path, fl['stage1'])
        p2[off:off + n] = predict(m2, np.hstack([X, ctx[off:off + n]]))
        off += n
    del ctx
    log('stage 2 done')
    # Saved so decoding variants (--redecode) take minutes instead of a full predict.
    pred = os.path.join(config.CACHE_DIR, split, 'pred.npz')
    np.savez(pred + '.tmp.npz', s1_code=s1_code.astype(np.int32), t_code=t_code.astype(np.int32),
             src=src.astype(np.int8), p2=p2)
    os.replace(pred + '.tmp.npz', pred)
    log(f'saved probabilities -> {pred}')
    param = choice['param'] if shift is None or choice['method'] == 'threshold' else shift
    chosen = decode(s1_code, t_code, src, p2, choice['method'], param)
    path = write_matches(split, s1_code, t_code, chosen)
    log(f'{int(chosen.sum()):,} matches ({choice["method"]}, param {param}) -> {path}')


def unseen_country_rows(split, s1_code):
    """Pairs whose Source 1 country never occurs in the training Source 1 (on test:
    France). Keyed on 'unseen in training', never on a country name."""
    from data_loader import read_tsv
    seen = set(read_tsv(config.TRAIN_S1, usecols=['country'])['country'].unique())
    c = read_tsv(config.split_paths(split)['s1'], usecols=['country'])['country'].to_numpy()
    unseen = ~np.isin(c, list(seen))
    log(f'unseen countries: {sorted(set(c[unseen]))} ({int(unseen.sum()):,} Source 1)')
    return unseen[s1_code]


def run_redecode(split, shift, unseen_shift=None):
    """Re-decode saved --predict probabilities with another logit shift, optionally a
    separate one for Source 1 in countries unseen in training; writes a new
    matching_results_*.tsv next to the main file (which stays untouched)."""
    with open(os.path.join(MODEL_DIR, 'decode.json')) as f:
        choice = json.load(f)
    with np.load(os.path.join(config.CACHE_DIR, split, 'pred.npz')) as z:
        s1_code, t_code, src, p2 = (z[k] for k in ('s1_code', 't_code', 'src', 'p2'))
    param = choice['param'] if shift is None else shift
    name = f'matching_results_shift{param:+.2f}.tsv'
    if unseen_shift is not None and choice['method'] != 'threshold':
        rows = unseen_country_rows(split, s1_code)
        p2 = logit_shift(p2, param)                        # seen countries: the chosen shift
        p2[rows] = logit_shift(p2[rows], unseen_shift)     # unseen: an extra shift on top
        param = 0.0
        name = f'matching_results_unseen{unseen_shift:+.2f}.tsv'
    chosen = decode(s1_code.astype(np.int64), t_code.astype(np.int64), src, p2, choice['method'], param)
    path = write_matches(split, s1_code, t_code, chosen, name=name)
    log(f'{int(chosen.sum()):,} matches ({choice["method"]}, shift {param:+.2f}) -> {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='local_val', choices=config.SPLIT_NAMES)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cv', action='store_true')
    mode.add_argument('--fit', action='store_true')
    ap.add_argument('--stage2-only', action='store_true',
                    help='with --fit: keep cache/models/stage1.txt, retrain stage 2 only (e.g. after adding cross-encoder scores)')
    mode.add_argument('--predict', action='store_true')
    mode.add_argument('--redecode', action='store_true',
                      help='re-decode saved --predict probabilities with --shift (minutes, no model run)')
    ap.add_argument('--unseen-shift', type=float, default=None,
                    help='with --redecode: extra logit shift for Source 1 in countries unseen in training (negative = stricter)')
    ap.add_argument('--shift', type=float, default=None,
                    help='override the logit shift for expected-F decoding at predict time (negative = stricter)')
    args = ap.parse_args()
    if args.cv:
        run_cv(args.split)
    elif args.fit:
        run_fit(args.split, stage2_only=args.stage2_only)
    elif args.redecode:
        run_redecode(args.split, args.shift, args.unseen_shift)
    else:
        run_predict(args.split, args.shift)


if __name__ == '__main__':
    main()
