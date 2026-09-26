"""
Blocking engine: run the enabled pipelines on one data directory, union their
candidates, apply the budget, and write the outputs.

Order matters on macOS: the classical pipeline forks worker processes, so it
runs before any torch model is loaded; each dense model is freed after use.
"""

import gc
import os

import numpy as np

from blocker.pipelines import candidate_union as cu
from blocker.pipelines.common import Retrieval, country_groups
from blocker.pipelines.output import DebugWriter, write_candidate_tsv, write_manifest
from blocker.utils import log, n_workers, resolve_device

ALL_PIPELINES = ('classical', 'bert', 'jepa')


def enabled_pipelines(cfg, requested=None, artifacts_dir=None):
    out = []
    for p in (requested or ALL_PIPELINES):
        if p not in ALL_PIPELINES:
            raise ValueError(f'unknown pipeline {p!r}; choose from {ALL_PIPELINES}')
        if not cfg['pipelines'].get(p, {}).get('enabled', True):
            continue
        if p != 'classical':
            if not cfg['models'].get(p, {}).get('enabled', True):
                continue
            path = os.path.join(artifacts_dir or '', p)
            if not artifacts_dir or not os.path.exists(os.path.join(path, 'weights.pt')):
                if requested:
                    raise FileNotFoundError(f'pipeline {p}: no trained model in {path}; run train_blocker.py first')
                log(f'pipeline {p}: no trained model in {path}, skipped')
                continue
        out.append(p)
    return out


def _with_k(pcfg, k):
    """Copy of a pipeline config where every list length is k (for K sweeps)."""
    import copy
    c = copy.deepcopy(pcfg)
    for key in ('top_k_source2', 'top_k_source3'):
        if key in c and int(c[key]) > 0:
            c[key] = k
    for p in (c.get('passes') or {}).values():
        if int(p.get('top_k', 0)) > 0:
            p['top_k'] = k
    return c


def run_retrievals(data, cfg, pipelines, artifacts_dir=None, cache_dir=None, k_override=None):
    """{pipeline: [Retrieval]} for the given pipelines."""
    groups = country_groups(data, cfg['pipelines'].get('country_fallback', 'global'))
    log(f'{data.name}: {len(groups)} country groups: '
        + ', '.join(f'{c} ({len(a):,} x {len(b):,})' for c, a, b in groups))
    workers = n_workers(cfg['runtime'].get('workers', 0))
    out = {}
    for p in [x for x in ALL_PIPELINES if x in pipelines]:
        pcfg = cfg['pipelines'][p] if k_override is None else _with_k(cfg['pipelines'][p], k_override)
        if p == 'classical':
            from blocker.pipelines.classical_blocker import ClassicalBlocker
            out[p] = ClassicalBlocker(pcfg, workers).retrieve(data, groups)
        else:
            from blocker.models.jepa_encoder import load_dense_model
            from blocker.pipelines.embedding_blocker import DenseBlocker
            device = resolve_device(cfg['runtime'].get('device', 'auto'))
            model = load_dense_model(p, os.path.join(artifacts_dir, p), cfg['models'].get('max_params'), device)
            out[p] = DenseBlocker(model, pcfg, cfg.get('ann', {}), cfg['runtime'], cache_dir,
                                  int(cfg.get('seed', 0))).retrieve(data, groups)
            del model
            gc.collect()
        log(f'pipeline {p}: ' + ', '.join(f'{r.name} {len(r):,} pairs' for r in out[p]))
    return out


def truncate_to_config(retrievals, cfg, data):
    """Cut retrievals made with a larger k back to the configured list lengths."""
    src = data.tg['source'].to_numpy()
    out = {}
    for p, rets in retrievals.items():
        pcfg = cfg['pipelines'][p]
        cut = []
        for r in rets:
            if p == 'classical':
                cut.append(r.top(int(pcfg['passes'][r.name]['top_k'])))
            else:
                k2, k3 = int(pcfg.get('top_k_source2', 20)), int(pcfg.get('top_k_source3', 20))
                kk = np.where(src[r.c] == 2, k2, k3)
                m = r.rank <= kk
                cut.append(Retrieval(r.name, r.family, max(k2, k3), r.r[m], r.c[m], r.score[m], r.rank[m]))
        out[p] = cut
    return out


def flatten(retrievals):
    return [r for p in ALL_PIPELINES for r in retrievals.get(p, [])]


def build_candidates(data, retrievals, cfg, out_dir, chunk_rows=100_000, debug=True):
    """
    Union + budget over all Source 1 rows in chunks; writes candidate_pairs.tsv,
    debug_candidate_scores/ and candidate_manifest.json into out_dir.
    Returns the final (s1, tg, priority) arrays and summary stats.
    """
    ccfg = cfg['candidate_generation']
    rets = cu.sort_by_s1(flatten(retrievals))
    n_s1, n_tg = len(data.s1), len(data.tg)
    prio = ccfg.get('prioritizer', 'votes_rank')
    budget = int(ccfg.get('max_candidates_per_source1', 0) or 0)
    hard = bool(ccfg.get('hard_budget', True))
    protect = int(ccfg.get('protect_min_pipelines', 0) or 0)
    max_lists = int(ccfg.get('target_max_lists', 0) or 0)

    cuts = None
    if max_lists > 0:        # pass A: global per-target cut-offs after the per-S1 budget
        tg_all, s1_all, p_all = [], [], []
        for lo in range(0, n_s1, chunk_rows):
            u = cu.union_rows(rets, n_tg, lo, min(lo + chunk_rows, n_s1), prio)
            u = cu.subset(u, cu.budget_mask(u, budget, hard, protect))
            tg_all.append(u['tg']), s1_all.append(u['s1']), p_all.append(u['priority'])
        cuts = cu.target_cutoffs(np.concatenate(tg_all), np.concatenate(s1_all), np.concatenate(p_all),
                                 n_tg, max_lists)
        del tg_all, s1_all, p_all

    writer = DebugWriter(os.path.join(out_dir, 'debug_candidate_scores')) if debug else None
    fin_s1, fin_tg, fin_p = [], [], []
    n_union = n_cut = 0
    for lo in range(0, n_s1, chunk_rows):
        u = cu.union_rows(rets, n_tg, lo, min(lo + chunk_rows, n_s1), prio)
        n_union += u['n']
        keep = cu.budget_mask(u, budget, hard, protect)
        if cuts is not None:
            keep &= cu.target_mask(u, *cuts)
        n_cut += int((~keep).sum())
        u = cu.subset(u, keep)
        if writer:
            extra = cu.string_similarity(u, data) if ccfg.get('string_similarity', True) else None
            writer.write(u, data.s1_ids, data.tg_ids, extra)
        fin_s1.append(u['s1']), fin_tg.append(u['tg']), fin_p.append(u['priority'])
    s1 = np.concatenate(fin_s1) if fin_s1 else np.empty(0, np.int32)
    tg = np.concatenate(fin_tg) if fin_tg else np.empty(0, np.int32)
    pr = np.concatenate(fin_p) if fin_p else np.empty(0, np.float32)

    tsv = write_candidate_tsv(os.path.join(out_dir, 'candidate_pairs.tsv'), data.s1_ids, data.tg_ids, s1, tg, pr,
                              ccfg.get('output_order', 'priority'))
    counts = np.bincount(s1, minlength=n_s1)
    stats = {'union_pairs': n_union, 'cut_by_budget': n_cut, 'final_pairs': int(len(s1)),
             'mean_per_s1': float(counts.mean()) if n_s1 else 0.0, 'max_per_s1': int(counts.max()) if n_s1 else 0,
             'zero_candidate_s1': int((counts == 0).sum()),
             'passes': {r.name: {'family': r.family, 'k': r.k, 'pairs': len(r)} for r in rets}}
    write_manifest(os.path.join(out_dir, 'candidate_manifest.json'), data,
                   {k: cfg[k] for k in ('pipelines', 'candidate_generation', 'ann')}, stats)
    log(f'wrote {tsv}: {len(s1):,} pairs ({stats["mean_per_s1"]:.1f} per Source 1; union {n_union:,}, '
        f'cut by budget {n_cut:,}); zero-candidate Source 1: {stats["zero_candidate_s1"]:,}')
    return s1, tg, pr, stats
