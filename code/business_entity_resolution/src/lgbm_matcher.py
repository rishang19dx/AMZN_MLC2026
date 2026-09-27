"""
Basic LightGBM matcher on top of the multi-pipeline blocker: turns
candidate_pairs.tsv into matching_results.tsv (a subset of the candidates).

Training data = the blocker's candidates on a LABELLED directory the encoders
never trained on (the validation holdout, e.g. splits/local_val, written by
train_blocker.py to <artifacts>/validation/ or by generate_candidates.py).
Group K-fold by Source 1 gives honest out-of-fold probabilities, on which the
decision threshold is tuned for F0.5.

Features per candidate pair (no country feature: France is unseen in training):
  blocker provenance   score_* / rank_* of every pass (NaN = not retrieved), hit_*,
                       n_pipelines, n_passes, priority, rf_name, rf_addr
  string similarity    RapidFuzz ratio / token_sort / partial / Jaro-Winkler on names,
                       ratio / token_set / partial on addresses, name_core ratio
  exact flags          name, name_core, address, house number, postal code
  list context         rank of priority / of p-proxy within the Source 1 list, list length,
                       gap to the best priority; target "hub" count (lists containing it)
  record shape         target source (S2/S3), empty address, non-Latin name, token counts

Decoding (reuses src/match.py): each target goes to its best Source 1 only
(every S2/S3 record matches at most one entity), per-source caps, then the
tuned probability threshold.

Usage:
  python src/lgbm_matcher.py fit --data-dir <labelled dir> --candidates-dir <its blocker output> \
      --artifacts-dir ../../artifacts [--train-fraction 0.5] [--folds 5]
  python src/lgbm_matcher.py predict --data-dir <test dir> --candidates-dir ../../output/test \
      --artifacts-dir ../../artifacts --output-dir ../../output/test
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocker  # noqa: F401  (OpenMP / fork settings before lightgbm loads)
import lightgbm as lgb
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from blocker.config import add_config_args, load_config
from blocker.data import load_dir
from blocker.normalization import extract_postal, house_number, prepare_data
from blocker.run import use_translit
from blocker.utils import log, n_workers, read_json, write_json
from match import assign_and_cap, group_stats
from normalize import is_non_latin

ID_COLS = ('s1_id', 'cand_id', 's1_idx', 'tg_idx')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def debug_parts(cand_dir):
    import glob
    parts = sorted(glob.glob(os.path.join(cand_dir, 'debug_candidate_scores', 'part-*.parquet')))
    if not parts:
        raise FileNotFoundError(f'no debug_candidate_scores/part-*.parquet in {cand_dir}; '
                                f'run generate_candidates.py without --no-debug')
    return parts


def check_manifest(cand_dir, data):
    path = os.path.join(cand_dir, 'candidate_manifest.json')
    if not os.path.exists(path):
        log(f'WARNING: no {path}; cannot verify row alignment')
        return
    m = read_json(path)
    ok = (m['n_s1'] == len(data.s1) and m['n_tg'] == len(data.tg)
          and m['first_s1'] == str(data.s1_ids[0]) and m['last_tg'] == str(data.tg_ids[-1]))
    if not ok:
        raise SystemExit(f'{path} was written for different data ({m["data"]}: {m["n_s1"]} S1, {m["n_tg"]} targets); '
                         f'regenerate candidates for {data.name}')


def read_part(path):
    import duckdb
    d = duckdb.sql(f"SELECT * EXCLUDE (s1_id, cand_id) FROM read_parquet('{path}')").fetchnumpy()
    return {k: (np.asarray(v.filled(np.nan)) if np.ma.isMaskedArray(v) else np.asarray(v)) for k, v in d.items()}


def load_record_arrays(data):
    """Per-record arrays the features need (row-aligned with s1 / tg)."""
    rec = {}
    for side, df in (('s1', data.s1), ('tg', data.tg)):
        rec[side] = {
            'name': df['name_n'].to_numpy(), 'core': df['name_core'].to_numpy(), 'addr': df['addr_n'].to_numpy(),
            'house': np.array([house_number(a) for a in df['addr_n']], dtype=object),
            'postal': np.array([' '.join(extract_postal(a)) for a in df['business_address']], dtype=object),
            'name_ntok': df['name_n'].str.count(' ').to_numpy(np.float32) + 1,
            'addr_ntok': df['addr_n'].str.count(' ').to_numpy(np.float32) + 1,
        }
    rec['tg']['is_s3'] = (data.tg['source'].to_numpy() == 3).astype(np.float32)
    rec['tg']['addr_missing'] = (data.tg['addr_n'].to_numpy() == '').astype(np.float32)
    rec['tg']['nonlatin'] = data.tg['business_name'].map(is_non_latin).to_numpy(np.float32)
    return rec


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _rf(scorer, a, b):
    return process.cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32) / np.float32(100)


def _eq(a, b):
    return ((a == b) & (a != '')).astype(np.float32)


def pair_features(part, rec, hub):
    i, j = part['s1_idx'].astype(np.int64), part['tg_idx'].astype(np.int64)
    s, t = rec['s1'], rec['tg']
    f = {k: np.asarray(v, np.float32) for k, v in part.items()
         if k not in ID_COLS and np.asarray(v).dtype.kind in 'fiub'}
    an, bn = s['name'][i], t['name'][j]
    f['name_ratio'] = _rf(fuzz.ratio, an, bn)
    f['name_tsort'] = _rf(fuzz.token_sort_ratio, an, bn)
    f['name_tset'] = _rf(fuzz.token_set_ratio, an, bn)
    f['name_partial'] = _rf(fuzz.partial_ratio, an, bn)
    f['name_jw'] = process.cpdist(list(an), list(bn), scorer=JaroWinkler.normalized_similarity, workers=-1,
                                  dtype=np.float32)
    f['core_ratio'] = _rf(fuzz.ratio, s['core'][i], t['core'][j])
    aa, ba = s['addr'][i], t['addr'][j]
    f['addr_ratio'] = _rf(fuzz.ratio, aa, ba)
    f['addr_tset'] = _rf(fuzz.token_set_ratio, aa, ba)
    f['addr_partial'] = _rf(fuzz.partial_ratio, aa, ba)
    f['name_exact'] = _eq(an, bn)
    f['core_exact'] = _eq(s['core'][i], t['core'][j])
    f['addr_exact'] = _eq(aa, ba)
    f['house_eq'] = _eq(s['house'][i], t['house'][j])
    f['postal_eq'] = _eq(s['postal'][i], t['postal'][j])
    for k in ('name_ntok', 'addr_ntok'):
        f[f's1_{k}'], f[f'tg_{k}'] = s[k][i], t[k][j]
    for k in ('is_s3', 'addr_missing', 'nonlatin'):
        f[f'tg_{k}'] = t[k][j]
    # list context (parts never split a Source 1 list)
    pr = f.get('priority', np.zeros(len(i), np.float32))
    rank, top1, _, _, _ = group_stats(i, pr)
    f['ctx_prio_rank'] = rank
    f['ctx_prio_gap'] = top1 - pr
    f['ctx_list_len'] = np.bincount(i, minlength=i.max() + 1)[i].astype(np.float32) if len(i) else np.empty(0, np.float32)
    nsim = f['name_tset'] + f['addr_tset']
    r2, t2, _, _, _ = group_stats(i, nsim)
    f['ctx_sim_rank'] = r2
    f['ctx_sim_gap'] = t2 - nsim
    f['tg_hub'] = hub[j].astype(np.float32)
    return f


def feature_matrix(f, names):
    X = np.empty((len(next(iter(f.values()))), len(names)), np.float32)
    for k, n in enumerate(names):
        X[:, k] = f[n] if n in f else np.nan
    return X


# ---------------------------------------------------------------------------
# Decoding and scoring
# ---------------------------------------------------------------------------

def decode(s1, tg, src, p, threshold):
    keep = assign_and_cap(s1.astype(np.int64), tg.astype(np.int64), src.astype(np.int8), p)
    return keep & (p >= threshold)


def macro_f05(n_true_per_s1, s1, label, chosen):
    """Leaderboard F0.5 averaged over EVERY Source 1 row (singletons included)."""
    n = len(n_true_per_s1)
    tp = np.bincount(s1[chosen], weights=label[chosen], minlength=n)
    npred = np.bincount(s1[chosen], minlength=n).astype(float)
    nt = n_true_per_s1.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(nt == 0, (npred == 0).astype(float), 1.25 * tp / (1.25 * tp + 0.25 * (nt - tp) + (npred - tp)))
    return float(np.nan_to_num(f).mean())


def write_matches(path, data, s1, tg, chosen):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sel = pd.DataFrame({'s1': s1[chosen], 'c': data.tg_ids[tg[chosen]]})
    lists = sel.groupby('s1')['c'].agg(lambda x: ','.join(sorted(set(x))))
    lists = lists.reindex(np.arange(len(data.s1)), fill_value='')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id, ids in zip(data.s1_ids, lists.to_numpy()):
            f.write(f'{s1_id}\t{ids}\n')
    log(f'wrote {path}: {int(chosen.sum()):,} matches')
    return path


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def prepare(args, cfg):
    use_translit(args.artifacts_dir)
    data = load_dir(args.data_dir)
    check_manifest(args.candidates_dir, data)
    prepare_data(data, cfg, n_workers(cfg['normalization'].get('workers', 0)))
    parts = debug_parts(args.candidates_dir)
    import duckdb
    tg_all = duckdb.sql(f"SELECT tg_idx FROM read_parquet({parts!r})").fetchnumpy()['tg_idx']
    hub = np.bincount(np.asarray(tg_all, np.int64), minlength=len(data.tg))
    return data, parts, load_record_arrays(data), hub


def fold_of(s1_ids, k, seed='lgbm-cv'):
    return np.array([int(hashlib.md5(f'{seed}:{u}'.encode()).hexdigest()[:8], 16) % k for u in s1_ids])


def run_fit(args, cfg):
    mcfg = cfg['matcher']
    folds_k = int(args.folds or mcfg.get('folds', 5))
    frac = float(args.train_fraction if args.train_fraction is not None else mcfg.get('train_fraction', 1.0))
    data, parts, rec, hub = prepare(args, cfg)
    if not data.has_truth:
        raise SystemExit('fit needs a labelled --data-dir')
    rng = np.random.default_rng(int(cfg.get('seed', 42)))
    s1_keep = rng.random(len(data.s1)) < frac if frac < 1 else np.ones(len(data.s1), bool)
    own = data.owner()
    Fs, meta = [], {'s1': [], 'tg': [], 'label': []}
    names = None
    for path in parts:
        part = read_part(path)
        m = s1_keep[part['s1_idx']]
        part = {k: v[m] for k, v in part.items()}
        if not len(part['s1_idx']):
            continue
        f = pair_features(part, rec, hub)
        names = names or sorted(f)
        Fs.append(feature_matrix(f, names))
        meta['s1'].append(part['s1_idx'].astype(np.int64))
        meta['tg'].append(part['tg_idx'].astype(np.int64))
        meta['label'].append((own[part['tg_idx']] == part['s1_idx']).astype(np.int8))
        log(f'  features {os.path.basename(path)}: {len(part["s1_idx"]):,} pairs')
    X = np.concatenate(Fs)
    del Fs
    s1, tg, y = (np.concatenate(meta[k]) for k in ('s1', 'tg', 'label'))
    src = rec['tg']['is_s3'][tg].astype(np.int8)
    log(f'{len(y):,} pairs, {len(names)} features, {y.mean():.2%} positive, '
        f'{int(s1_keep.sum()):,} Source 1 entities (train_fraction={frac})')

    fold = fold_of(data.s1_ids, folds_k)[s1]
    params = dict(mcfg.get('lgb', {}), num_threads=n_workers(cfg['runtime'].get('workers', 0)),
                  seed=int(cfg.get('seed', 42)))
    full = lgb.Dataset(X, y, feature_name=names, params=params, free_raw_data=False).construct()
    oof = np.zeros(len(y), np.float32)
    out_dir = os.path.join(args.artifacts_dir, 'matcher')
    os.makedirs(out_dir, exist_ok=True)
    for k in range(folds_k):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        es = fold_of(data.s1_ids, 10, 'lgbm-es')[s1[tr]] == 0          # early-stopping slice of the training folds
        if es.any() and not es.all():
            model = lgb.train(params, full.subset(tr[~es]), int(mcfg.get('num_boost_round', 2000)),
                              valid_sets=[full.subset(tr[es])],
                              callbacks=[lgb.early_stopping(int(mcfg.get('early_stopping', 50)), verbose=False)])
        else:                                                              # tiny data: no held-out slice
            model = lgb.train(params, full.subset(tr), min(200, int(mcfg.get('num_boost_round', 2000))))
        oof[te] = model.predict(X[te], num_threads=params['num_threads']) if len(te) else oof[te]
        model.save_model(os.path.join(out_dir, f'fold{k}.txt'))
        log(f'  fold {k + 1}/{folds_k}: {model.best_iteration} trees')

    from sklearn.metrics import roc_auc_score
    log(f'out-of-fold AUC {roc_auc_score(y, oof):.5f}')
    # F0.5 over the trained-on Source 1 entities only (all of them, with or without candidates)
    n_true = np.where(s1_keep, data.n_true(), 0)
    ent = np.flatnonzero(s1_keep)
    remap = np.full(len(data.s1), -1, np.int64)
    remap[ent] = np.arange(len(ent))
    rows = []
    for t in mcfg.get('threshold_grid', [0.5]):
        rows.append((float(t), macro_f05(n_true[ent], remap[s1], y.astype(float), decode(s1, tg, src, oof, t))))
    res = pd.DataFrame(rows, columns=['threshold', 'oof_f05'])
    print('\n== threshold sweep (out-of-fold F0.5, singletons included)\n' + res.to_string(index=False))
    best = res.loc[res['oof_f05'].idxmax()]
    write_json({'features': names, 'folds': folds_k, 'threshold': float(best['threshold']),
                'oof_f05': float(best['oof_f05']), 'trained_on': data.name, 'train_fraction': frac,
                'lgb_params': params}, os.path.join(out_dir, 'matcher.json'))
    chosen = decode(s1, tg, src, oof, float(best['threshold']))
    write_matches(os.path.join(args.candidates_dir, 'matching_results_oof.tsv'), data, s1, tg, chosen)
    imp = pd.Series(model.feature_importance('gain'), index=names).sort_values(ascending=False)
    print('\n== feature importance (gain share, top 20)\n' + (imp / imp.sum()).head(20).to_string())
    log(f'best threshold {best["threshold"]} (OOF F0.5 {best["oof_f05"]:.5f}); models in {out_dir}')


def run_predict(args, cfg):
    out_dir = os.path.join(args.artifacts_dir, 'matcher')
    mj = read_json(os.path.join(out_dir, 'matcher.json'))
    n_models = min(mj['folds'], args.max_folds) if args.max_folds else mj['folds']
    models = [lgb.Booster(model_file=os.path.join(out_dir, f'fold{k}.txt')) for k in range(n_models)]
    log(f'using {n_models} of {mj["folds"]} fold models')
    data, parts, rec, hub = prepare(args, cfg)
    nt = n_workers(cfg['runtime'].get('workers', 0))
    s1s, tgs, ps = [], [], []
    for path in parts:
        part = read_part(path)
        if not len(part['s1_idx']):
            continue
        X = feature_matrix(pair_features(part, rec, hub), mj['features'])
        ps.append(np.mean([m.predict(X, num_threads=nt) for m in models], axis=0).astype(np.float32))
        s1s.append(part['s1_idx'].astype(np.int64))
        tgs.append(part['tg_idx'].astype(np.int64))
        log(f'  scored {os.path.basename(path)}: {len(X):,} pairs')
    s1, tg, p = np.concatenate(s1s), np.concatenate(tgs), np.concatenate(ps)
    thr = float(args.threshold if args.threshold is not None else mj['threshold'])
    chosen = decode(s1, tg, rec['tg']['is_s3'][tg].astype(np.int8), p, thr)
    write_matches(os.path.join(args.output_dir, 'matching_results.tsv'), data, s1, tg, chosen)
    if data.has_truth:
        f = macro_f05(data.n_true(), s1, (data.owner()[tg] == s1).astype(float), chosen)
        log(f'F0.5 on {data.name} (labelled): {f:.5f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['fit', 'predict'])
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--candidates-dir', required=True, help='output directory of generate_candidates.py for --data-dir')
    ap.add_argument('--artifacts-dir', required=True)
    ap.add_argument('--output-dir', default=None, help='predict: where matching_results.tsv goes (default: candidates dir)')
    ap.add_argument('--train-fraction', type=float, default=None, help='fit: random fraction of Source 1 entities')
    ap.add_argument('--folds', type=int, default=None)
    ap.add_argument('--threshold', type=float, default=None, help='predict: override the tuned threshold')
    ap.add_argument('--max-folds', type=int, default=0,
                    help='predict: average only the first N fold models (0 = all); N=1 is ~N_folds x faster')
    add_config_args(ap)
    args = ap.parse_args(argv)
    args.output_dir = args.output_dir or args.candidates_dir
    cfg = load_config(args.config, args.set)
    (run_fit if args.mode == 'fit' else run_predict)(args, cfg)


if __name__ == '__main__':
    main()
