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
uses feature values only, never labels, so this does not leak). --eval and
--predict stream the feature parts twice, so test (~70M pairs) never has to
fit in RAM.

K-fold scheme (--folds, default 5): folds are grouped by Source 1 entity
(md5 of the ID), so all pairs of an entity, and so all of its targets, are in
one fold. Stage 1 is cross-fitted to give out-of-fold p1; stage 2 is
cross-fitted on those. The K fold models of each stage ARE the final model:
at inference their probabilities are averaged (no separate full refit).

Modes:
  --fit      labelled split: K-fold cross-fit of both stages, every fold model
             saved to $BER_CACHE_DIR/models/ as soon as it is trained; reports
             out-of-fold AUC / calibration / F0.5 and writes decode.json.
  --cv       same as --fit but saves no models (dev number only).
  --eval     apply the saved ensemble to a *different* labelled split
             (local_val, a holdout), tune decoding there, overwrite decode.json.
  --predict  apply the saved ensemble + decode.json to any split (e.g. test).

Usage (16 GB machine; see scripts/mac_run.sh):
  python src/match.py --split local_fit --fit --folds 5
  python src/match.py --split local_val --eval
  python src/match.py --split test --predict
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

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
                  num_threads=config.THREADS, verbose=-1, seed=26)
SHIFTS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5)     # expected-F logit shifts swept
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


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
    # tg_is_s3 is both a meta column and a feature: select it once, pop it once
    d = duckdb.sql(f"SELECT {', '.join(dict.fromkeys(meta_cols + feats))} FROM read_parquet('{path}')").fetchnumpy()
    meta = {c: np.asarray(d[c] if c in feats else d.pop(c)) for c in meta_cols}
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
        out[s:s + chunk] = model.predict(X[rows[s:s + chunk]], num_threads=config.THREADS)
    return out


def fit_on(full, rows, es_fold, rounds=3000):
    """Train on `rows` of a constructed Dataset; early-stop on its es_fold==0 part."""
    tr, va = rows[es_fold[rows] != 0], rows[es_fold[rows] == 0]
    return lgb.train(LGB_PARAMS, full.subset(tr), rounds, valid_sets=[full.subset(va)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def model_path(stage, fold):
    return os.path.join(MODEL_DIR, f'stage{stage}_fold{fold}.txt')


def cross_fit(pairs, X, folds, es_fold, k=2, stage=None):
    """Out-of-fold probabilities for every row, the last fold's model and each
    fold's tree count. With `stage` set, every fold model is saved to
    MODEL_DIR as soon as it is trained (a crash loses at most one fold)."""
    full = lgb.Dataset(X, pairs.label, params=LGB_PARAMS, free_raw_data=False).construct()
    oof = np.zeros(len(pairs), np.float32)
    iters = []
    for f in range(k):
        t0 = time.time()
        te = np.flatnonzero(folds == f)
        m = fit_on(full, np.flatnonzero(folds != f), es_fold)
        oof[te] = predict(m, X, te)
        iters.append(int(m.best_iteration))
        if stage is not None:
            m.save_model(model_path(stage, f))
        log(f'  fold {f + 1}/{k}: {m.best_iteration} trees, {time.time() - t0:.0f}s'
            + (f' -> {model_path(stage, f)}' if stage is not None else ''))
    return oof, m, iters


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

def sweep_decoding(sc, s1_code, t_code, src, p):
    """F0.5 of every decoding setting; returns the table and the best row."""
    args = (s1_code, t_code, src)
    results = [('threshold', t, sc.f05(decode(*args, p, 'threshold', t))) for t in THRESHOLDS]
    results += [('expected_f', b, sc.f05(decode(*args, p, 'expected_f', b))) for b in SHIFTS]
    res = pd.DataFrame(results, columns=['method', 'param', 'f05'])
    return res, res.loc[res['f05'].idxmax()]


def report_probs(y, named):
    from sklearn.metrics import log_loss, roc_auc_score
    for name, p in named:
        log(f'{name}: AUC {roc_auc_score(y, p):.5f}  logloss {log_loss(y, p):.5f}')
    p = named[-1][1]
    bins = pd.cut(p, [0, .1, .3, .5, .7, .9, .97, 1.0001])
    calib = pd.DataFrame({'p': p, 'y': y}).groupby(bins, observed=True).agg(
        n=('y', 'size'), mean_p=('p', 'mean'), frac_pos=('y', 'mean'))
    print(f'\n== calibration of {named[-1][0]} probabilities\n' + calib.to_string(float_format='%.4f'))


def write_decode(best, split, how):
    choice = {'method': best['method'], 'param': float(best['param']), 'f05': float(best['f05']),
              'tuned_on': split, 'how': how}
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'decode.json'), 'w') as f:
        json.dump(choice, f, indent=2)
    return choice


def run_train(split, k, save):
    """K-fold cross-fit of both stages on a labelled split (see module docstring)."""
    pairs = Pairs(split)
    y = pairs.label
    log(f'{len(pairs):,} pairs, {len(pairs.feats)} base features, {y.mean():.2%} positive, {k} folds')
    folds, es_fold = pairs.row_fold(split, k, 'mlc26-cv'), pairs.row_fold(split, 10, 'mlc26-es')
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        for old in glob.glob(os.path.join(MODEL_DIR, 'stage*_fold*.txt')):   # stale folds of an earlier run
            os.remove(old)

    log('stage 1 (pair features)')
    p1, _, it1 = cross_fit(pairs, pairs.X, folds, es_fold, k, stage=1 if save else None)
    X2 = np.hstack([pairs.X, context_features(pairs.s1_code, pairs.t_code, p1)])
    log('stage 2 (+ list and competition context)')
    p2, m2, it2 = cross_fit(pairs, X2, folds, es_fold, k, stage=2 if save else None)
    del X2
    if save:
        with open(os.path.join(MODEL_DIR, 'features.json'), 'w') as f:
            json.dump({'stage1': pairs.feats, 'stage2': pairs.feats + CONTEXT, 'trained_on': split,
                       'n_folds': k, 'best_iterations': {'stage1': it1, 'stage2': it2},
                       'lgb_params': LGB_PARAMS, 'created': time.strftime('%Y-%m-%d %H:%M:%S %Z')}, f, indent=2)
        log(f'saved {2 * k} fold models + features.json to {MODEL_DIR}')
    np.save(os.path.join(config.CACHE_DIR, split, 'oof_p1.npy'), p1)   # for error analysis
    np.save(os.path.join(config.CACHE_DIR, split, 'oof_p2.npy'), p2)

    report_probs(y, [('stage 1 out-of-fold', p1), ('stage 2 out-of-fold', p2)])
    sc = Scorer(split, pairs)
    res, best = sweep_decoding(sc, pairs.s1_code, pairs.t_code, pairs.src, p2)
    print('\n== decoding sweep (out-of-fold F0.5)\n' + res.to_string(index=False, float_format='%.5f'))
    # ablations: what the structural steps are worth
    no_assign = decode_expected_f(pairs.s1_code, p2, np.ones(len(pairs), bool))
    print(f'ablation, expected_f without one-Source-1-per-target and caps: {sc.f05(no_assign):.5f}')
    print(f'ablation, stage-1 probabilities only (no context features):   '
          f'{sc.f05(decode(pairs.s1_code, pairs.t_code, pairs.src, p1, "expected_f", 0.0)):.5f}')

    chosen = decode(pairs.s1_code, pairs.t_code, pairs.src, p2, best['method'], best['param'])
    sc.full_report(chosen, f'Matching, out-of-fold ({split}, {k} folds)')
    path = write_matches(split, pairs.s1_code, pairs.t_code, chosen)
    choice = write_decode(best, split, 'out-of-fold')
    log(f'best: {choice} -> {path}')

    imp = pd.Series(m2.feature_importance('gain'), index=pairs.feats + CONTEXT).sort_values(ascending=False)
    print('\n== stage-2 feature importance (gain share, top 20)\n' + (imp / imp.sum()).head(20).to_string(float_format='%.4f'))


def load_ensemble():
    """Saved fold models of both stages (or the single models of the old --fit)."""
    with open(os.path.join(MODEL_DIR, 'features.json')) as f:
        fl = json.load(f)
    if 'n_folds' in fl:
        paths = {s: [model_path(s, k) for k in range(fl['n_folds'])] for s in (1, 2)}
    else:
        paths = {s: [os.path.join(MODEL_DIR, f'stage{s}.txt')] for s in (1, 2)}
    models = {s: [lgb.Booster(model_file=p) for p in ps] for s, ps in paths.items()}
    log(f"loaded {len(models[1])} + {len(models[2])} models (trained on {fl['trained_on']})")
    return fl, models[1], models[2]


def predict_mean(models, X):
    return np.mean([predict(m, X) for m in models], axis=0).astype(np.float32)


def ensemble_predict(split):
    """Two streaming passes over the feature parts; only per-row codes and
    probabilities are held for the whole split. Returns (s1_code, t_code, src,
    p1, p2, label or None, feature list info)."""
    fl, m1, m2 = load_ensemble()
    parts = feature_parts(split)
    s1c, tc, src, p1, lab = [], [], [], [], []
    for path in parts:
        meta, X, _ = read_part(path, fl['stage1'])
        s1c.append(meta['s1_idx'].astype(np.int64))
        tc.append(meta['tg_idx'].astype(np.int64))
        src.append(meta['tg_is_s3'].astype(np.int8))
        if 'label' in meta:
            lab.append(meta['label'].astype(np.int8))
        p1.append(predict_mean(m1, X))
    sizes = [len(x) for x in p1]
    s1_code, t_code, src, p1 = (np.concatenate(a) for a in (s1c, tc, src, p1))
    label = np.concatenate(lab) if len(lab) == len(parts) else None
    log(f'stage 1: {len(p1):,} pairs in {len(parts)} parts')
    ctx = context_features(s1_code, t_code, p1)
    p2, off = np.empty(len(p1), np.float32), 0
    for path, n in zip(parts, sizes):
        _, X, _ = read_part(path, fl['stage1'])
        p2[off:off + n] = predict_mean(m2, np.hstack([X, ctx[off:off + n]]))
        off += n
    del ctx
    log('stage 2 done')
    return s1_code, t_code, src, p1, p2, label, fl


def run_eval(split):
    """Holdout check: saved ensemble on a labelled split it was not trained on;
    the decoding setting is tuned here and written to decode.json."""
    s1_code, t_code, src, p1, p2, label, fl = ensemble_predict(split)
    if label is None:
        raise SystemExit(f'--eval needs a labelled split; {split} has no labels')
    if fl['trained_on'] == split:
        log(f'WARNING: models were trained on {split}; these numbers are in-sample')
    report_probs(label, [('stage 1 holdout', p1), ('stage 2 holdout', p2)])
    pairs = SimpleNamespace(s1_code=s1_code, t_code=t_code, label=label)
    sc = Scorer(split, pairs)
    res, best = sweep_decoding(sc, s1_code, t_code, src, p2)
    print(f'\n== decoding sweep (holdout F0.5 on {split})\n' + res.to_string(index=False, float_format='%.5f'))
    chosen = decode(s1_code, t_code, src, p2, best['method'], best['param'])
    sc.full_report(chosen, f"Matching, holdout ({split}; trained on {fl['trained_on']})")
    path = write_matches(split, s1_code, t_code, chosen)
    choice = write_decode(best, split, 'holdout')
    log(f'best: {choice} -> {path}')


def run_predict(split, shift=None):
    s1_code, t_code, src, _, p2, _, _ = ensemble_predict(split)
    with open(os.path.join(MODEL_DIR, 'decode.json')) as f:
        choice = json.load(f)
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
    mode.add_argument('--eval', action='store_true')
    mode.add_argument('--predict', action='store_true')
    ap.add_argument('--folds', type=int, default=5, help='K for the grouped K-fold cross-fit')
    ap.add_argument('--shift', type=float, default=None,
                    help='override the logit shift for expected-F decoding at predict time (negative = stricter)')
    args = ap.parse_args()
    if args.cv or args.fit:
        run_train(args.split, args.folds, save=args.fit)
    elif args.eval:
        run_eval(args.split)
    else:
        run_predict(args.split, args.shift)


if __name__ == '__main__':
    main()
