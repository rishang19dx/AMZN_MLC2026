"""
Candidate union, deduplication, provenance and prioritised budget.

    C_final(s1) = C_bert(s1) U C_jepa(s1) U C_classical(s1)      (never an intersection)

Per (Source 1, target) pair the union keeps, for every pass that found it,
its similarity (`score_<pass>`) and rank (`rank_<pass>`, NaN = not retrieved),
plus `hit_<family>`, `n_pipelines` (families that found it, 0-3), `n_passes`
and a `priority`. Only if a Source 1 list is longer than the budget is it cut,
lowest priority first. Prioritisers (swappable, see PRIORITIZERS):

  votes_rank   n_pipelines + mean over the 3 families of the family's best
               normalised rank 1 - (rank-1)/k   (default: consensus first,
               then how high each pipeline ranked it)
  votes_score  n_pipelines + mean raw similarity of the passes that found it
  rrf          reciprocal-rank fusion: sum over passes of 1 / (60 + rank)

The union is processed in Source 1 row chunks, so memory stays bounded on the
1.7M-entity test set; pass rows are sorted by Source 1 once and sliced.
"""

import numpy as np

FAMILIES = ('bert', 'jepa', 'classical')


def _prio_votes_rank(u, rets):
    strength = {f: np.zeros(u['n'], np.float32) for f in FAMILIES}
    for ret in rets:
        rk = u[f'rank_{ret.name}']
        s = np.where(np.isnan(rk), 0, 1 - (rk - 1) / max(ret.k, 1)).astype(np.float32)
        strength[ret.family] = np.maximum(strength[ret.family], s)
    return u['n_pipelines'] + sum(strength.values()) / len(FAMILIES)


def _prio_votes_score(u, rets):
    tot = np.zeros(u['n'], np.float32)
    for ret in rets:
        tot += np.nan_to_num(u[f'score_{ret.name}'])
    return u['n_pipelines'] + tot / np.maximum(u['n_passes'], 1)


def _prio_rrf(u, rets):
    tot = np.zeros(u['n'], np.float32)
    for ret in rets:
        rk = u[f'rank_{ret.name}']
        tot += np.where(np.isnan(rk), 0, 1 / (60 + rk)).astype(np.float32)
    return tot


PRIORITIZERS = {'votes_rank': _prio_votes_rank, 'votes_score': _prio_votes_score, 'rrf': _prio_rrf}


def sort_by_s1(retrievals):
    out = []
    for ret in retrievals:
        o = np.argsort(ret.r, kind='stable')
        ret.r, ret.c, ret.score, ret.rank = ret.r[o], ret.c[o], ret.score[o], ret.rank[o]
        out.append(ret)
    return out


def union_rows(retrievals, n_tg, lo, hi, prioritizer='votes_rank'):
    """Union of all passes for Source 1 rows [lo, hi) (retrievals sorted by r).
    Returns a dict of equal-length arrays, pairs sorted by (s1, tg)."""
    keys, spans = [], []
    for ret in retrievals:
        a, b = np.searchsorted(ret.r, lo), np.searchsorted(ret.r, hi)
        spans.append((a, b))
        keys.append(ret.r[a:b].astype(np.int64) * n_tg + ret.c[a:b])
    allk = np.concatenate(keys) if keys else np.empty(0, np.int64)
    uniq, inv = np.unique(allk, return_inverse=True)
    n = len(uniq)
    u = {'n': n, 's1': (uniq // n_tg).astype(np.int32), 'tg': (uniq % n_tg).astype(np.int32)}
    hits = {f: np.zeros(n, bool) for f in FAMILIES}
    n_passes = np.zeros(n, np.int8)
    off = 0
    for ret, (a, b) in zip(retrievals, spans):
        sel = inv[off:off + (b - a)]
        off += b - a
        u[f'score_{ret.name}'] = np.full(n, np.nan, np.float32)
        u[f'rank_{ret.name}'] = np.full(n, np.nan, np.float32)
        u[f'score_{ret.name}'][sel] = ret.score[a:b]
        u[f'rank_{ret.name}'][sel] = ret.rank[a:b]
        hits[ret.family][sel] = True
        n_passes[sel] += 1
    for f in FAMILIES:
        u[f'hit_{f}'] = hits[f]
    u['n_pipelines'] = sum(h.astype(np.int8) for h in hits.values()).astype(np.int8)
    u['n_passes'] = n_passes
    u['priority'] = PRIORITIZERS[prioritizer](u, retrievals).astype(np.float32)
    return u


def _order_within(group, priority, tie):
    """Row order: by group, then priority desc, then tie asc."""
    return np.lexsort((tie, -priority, group))


def rank_within(group, priority, tie):
    order = _order_within(group, priority, tie)
    g = group[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1] if len(g) else np.empty(0, np.int64)
    sizes = np.diff(np.r_[starts, len(g)])
    rank = np.empty(len(g), np.int64)
    rank[order] = np.arange(len(g)) - np.repeat(starts, sizes) + 1
    return rank


def budget_mask(u, max_per_s1, hard=True, protect_min_pipelines=0):
    """Keep the top-`max_per_s1` pairs of each Source 1 by priority (ties: target
    row). Pairs found by >= protect_min_pipelines pipelines are never cut."""
    keep = np.ones(u['n'], bool)
    if not hard or not max_per_s1 or u['n'] == 0:
        return keep
    keep = rank_within(u['s1'], u['priority'], u['tg']) <= int(max_per_s1)
    if protect_min_pipelines:
        keep |= u['n_pipelines'] >= int(protect_min_pipelines)
    return keep


def target_cutoffs(tg, s1, priority, n_tg, max_lists):
    """For target-side pruning: each target keeps only its `max_lists` best
    Source 1 lists. Returns per-target (priority, s1) of the last kept pair."""
    r = rank_within(tg, priority, s1)
    last = r == max_lists
    cut_p = np.full(n_tg, -np.inf, np.float32)
    cut_s = np.full(n_tg, np.iinfo(np.int32).max, np.int64)
    cut_p[tg[last]] = priority[last]
    cut_s[tg[last]] = s1[last]
    return cut_p, cut_s


def target_mask(u, cut_p, cut_s):
    p, cp, cs = u['priority'], cut_p[u['tg']], cut_s[u['tg']]
    return (p > cp) | ((p == cp) & (u['s1'] <= cs))


def string_similarity(u, data):
    """RapidFuzz provenance scores on the union (cheap: linear in candidates)."""
    from rapidfuzz import fuzz, process
    out = {}
    for col, field in (('rf_name', 'name_n'), ('rf_addr', 'addr_n')):
        a = data.s1[field].to_numpy()[u['s1']]
        b = data.tg[field].to_numpy()[u['tg']]
        out[col] = (process.cpdist(list(a), list(b), scorer=fuzz.token_set_ratio, workers=-1,
                                   dtype=np.float32) / 100).astype(np.float32) if len(a) else np.empty(0, np.float32)
    return out


def subset(u, mask):
    out = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in u.items()}
    out['n'] = int(mask.sum())
    return out
