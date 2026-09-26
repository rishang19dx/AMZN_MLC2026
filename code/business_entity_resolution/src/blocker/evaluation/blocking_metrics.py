"""
Blocking quality metrics (everything computed from real labels; nothing assumed).

  pair recall (PC)       true pairs among the candidates / all true pairs
  macro recall           mean over Source 1 entities with >= 1 true match
  full-recall entities   share of those entities with every match retrieved
  F0.5 ceiling           leaderboard score of a perfect matcher on these candidates
  pair quality (PQ)      true pairs / candidate pairs: "of the blocked pairs, how many are real"
  reduction ratio (RR)   1 - candidates / (|S1| x |S2 + S3|), and vs the within-country product
  candidates per S1      mean / median / p95 / max; zero-candidate entities (and how many
                         of them actually have true matches = lost recall)
  singleton pollution    Source 1 entities without matches that still get candidates

Penalised scores (so a blocker cannot win by returning everything):
  penalized_recall        PC - lambda * (false candidates per S1) / ref_budget
  macro_penalized_recall  mean_i [ recall_i - lambda * fp_i / ref_budget ]   (recall_i = 1 for singletons)
  f1_pc_pq                harmonic mean of PC and PQ
  h_pc_rr                 harmonic mean of PC and RR
"""

import numpy as np

from blocker.pipelines import candidate_union as cu
from blocker.pipelines.common import Retrieval

FAMILY_ORDER = ('bert', 'jepa', 'classical')


def _keys(s1, tg, n_tg):
    return np.asarray(s1, np.int64) * n_tg + np.asarray(tg, np.int64)


def _hmean(a, b):
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def evaluate_candidates(data, s1, tg, penalty_lambda=0.1, ref_budget=100, per_country=True):
    n_s1, n_tg = len(data.s1), len(data.tg)
    pi, pj = data.positive_pairs()
    true_k = _keys(pi, pj, n_tg)
    cand_k = np.unique(_keys(s1, tg, n_tg))
    s1 = (cand_k // n_tg).astype(np.int64)
    is_true = np.isin(cand_k, true_k, assume_unique=False)
    found = np.isin(true_k, cand_k)

    n_true = np.bincount(pi, minlength=n_s1)
    n_cand = np.bincount(s1, minlength=n_s1)
    n_hit = np.bincount(s1[is_true], minlength=n_s1)
    fp = n_cand - n_hit
    has = n_true > 0
    rec_i = np.where(has, n_hit / np.maximum(n_true, 1), 1.0)
    # oracle matcher keeps exactly the true candidates: F0.5 = 1.25 r / (0.25 + r) with p = 1
    f_i = np.where(has, np.where(n_hit > 0, 1.25 * rec_i / (0.25 + rec_i), 0.0), 1.0)

    s1c = data.s1['country_n'].to_numpy()
    tgc = data.tg['country_n'].to_numpy()
    tg_per_c = {c: int((tgc == c).sum()) for c in np.unique(tgc)}
    within = sum(tg_per_c.get(c, n_tg) for c in s1c)
    n_c = len(cand_k)
    pc = float(found.mean()) if len(found) else float('nan')
    pq = float(is_true.mean()) if n_c else 0.0
    rr = 1 - n_c / max(n_s1 * n_tg, 1)
    rep = {
        'source1_entities': n_s1, 'targets': n_tg, 'true_pairs': int(len(true_k)),
        'candidate_pairs': int(n_c), 'true_pairs_found': int(found.sum()),
        'cands_mean': float(n_cand.mean()) if n_s1 else 0.0,
        'cands_median': float(np.median(n_cand)) if n_s1 else 0.0,
        'cands_p95': float(np.percentile(n_cand, 95)) if n_s1 else 0.0,
        'cands_max': int(n_cand.max()) if n_s1 else 0,
        'pair_recall': pc,
        'macro_recall': float(rec_i[has].mean()) if has.any() else float('nan'),
        'entities_full_recall': float((n_hit[has] == n_true[has]).mean()) if has.any() else float('nan'),
        'f05_ceiling': float(f_i.mean()) if n_s1 else float('nan'),
        'pair_quality': pq,
        'reduction_ratio': rr,
        'reduction_ratio_within_country': 1 - n_c / max(within, 1),
        'zero_candidate_entities': int((n_cand == 0).sum()),
        'zero_candidate_with_true_matches': int(((n_cand == 0) & has).sum()),
        'true_pairs_lost_to_empty_lists': int(n_true[(n_cand == 0)].sum()),
        'singletons': int((~has).sum()),
        'singletons_with_candidates': int(((~has) & (n_cand > 0)).sum()),
        'singleton_cands_mean': float(n_cand[~has].mean()) if (~has).any() else 0.0,
        'false_candidates_per_s1': float(fp.mean()) if n_s1 else 0.0,
        'penalty_lambda': penalty_lambda, 'penalty_ref_budget': ref_budget,
        'penalized_recall': pc - penalty_lambda * float(fp.mean()) / ref_budget if n_s1 else float('nan'),
        'macro_penalized_recall': float((rec_i - penalty_lambda * fp / ref_budget).mean()) if n_s1 else float('nan'),
        'f1_pc_pq': _hmean(pc, pq),
        'h_pc_rr': _hmean(pc, rr),
    }
    if per_country:
        rep['by_country'] = {}
        pic = s1c[pi]
        for c in np.unique(s1c):
            m = s1c == c
            tm = pic == c
            rep['by_country'][str(c)] = {
                'entities': int(m.sum()), 'true_pairs': int(tm.sum()),
                'pair_recall': float(found[tm].mean()) if tm.any() else float('nan'),
                'cands_mean': float(n_cand[m].mean()),
                'f05_ceiling': float(f_i[m].mean()),
                'pair_quality': float(n_hit[m].sum() / max(n_cand[m].sum(), 1)),
            }
    return rep


def pipeline_contributions(data, retrievals):
    """Recall of each pipeline alone and the Venn split of the true pairs found."""
    n_tg = len(data.tg)
    pi, pj = data.positive_pairs()
    true_k = _keys(pi, pj, n_tg)
    hit, cand = {}, {}
    for fam in FAMILY_ORDER:
        rets = retrievals.get(fam, [])
        if not rets:
            continue
        k = np.unique(np.concatenate([_keys(r.r, r.c, n_tg) for r in rets])) if rets else np.empty(0, np.int64)
        hit[fam] = np.isin(true_k, k)
        cand[fam] = len(k)
    out = {'recall': {f: float(h.mean()) for f, h in hit.items()},
           'candidates_per_s1': {f: n / max(len(data.s1), 1) for f, n in cand.items()}}
    for fam, rets in retrievals.items():
        for r in rets:
            if fam == 'classical':
                out['recall'][f'classical/{r.name}'] = float(np.isin(true_k, _keys(r.r, r.c, n_tg)).mean())
    if hit:
        union = np.logical_or.reduce(list(hit.values()))
        out['recall']['union'] = float(union.mean())
        fams = list(hit)
        venn = {}
        combo = np.zeros(len(true_k), np.int64)
        for b, f in enumerate(fams):
            combo |= hit[f].astype(np.int64) << b
        for code in range(1, 2 ** len(fams)):
            name = ' + '.join(f for b, f in enumerate(fams) if code >> b & 1)
            venn[name] = int((combo == code).sum())
        venn['none (missed by all)'] = int((combo == 0).sum())
        out['true_pair_venn'] = venn
    return out


def k_sweep(data, retrievals_max, ks):
    """Recall and candidates per S1 when every list is cut to K (union and per pipeline)."""
    n_tg, n_s1 = len(data.tg), len(data.s1)
    pi, pj = data.positive_pairs()
    true_k = _keys(pi, pj, n_tg)
    rows = []
    for K in ks:
        per, allk = {}, []
        for fam, rets in retrievals_max.items():
            k = np.unique(np.concatenate([_keys(t.r, t.c, n_tg) for t in (r.top(K) for r in rets)]))
            per[fam] = (float(np.isin(true_k, k).mean()), len(k) / max(n_s1, 1))
            allk.append(k)
        u = np.unique(np.concatenate(allk)) if allk else np.empty(0, np.int64)
        rows.append({'K': int(K), 'cands_mean': len(u) / max(n_s1, 1), 'recall': float(np.isin(true_k, u).mean()),
                     **{f'{f}_recall': v[0] for f, v in per.items()},
                     **{f'{f}_cands': v[1] for f, v in per.items()}})
    return rows


def budget_sweep(data, retrievals, budgets, prioritizer='votes_rank', chunk_rows=100_000):
    """Recall @ candidate budget: union cut to the top-B by priority per Source 1."""
    n_tg, n_s1 = len(data.tg), len(data.s1)
    pi, pj = data.positive_pairs()
    true_k = _keys(pi, pj, n_tg)
    rets = cu.sort_by_s1([Retrieval(r.name, r.family, r.k, r.r, r.c, r.score, r.rank)
                          for p in FAMILY_ORDER for r in retrievals.get(p, [])])
    s1s, tgs, prs = [], [], []
    for lo in range(0, n_s1, chunk_rows):
        u = cu.union_rows(rets, n_tg, lo, min(lo + chunk_rows, n_s1), prioritizer)
        s1s.append(u['s1']), tgs.append(u['tg']), prs.append(u['priority'])
    s1, tg, pr = (np.concatenate(x) if x else np.empty(0) for x in (s1s, tgs, prs))
    rank = cu.rank_within(s1, pr, tg) if len(s1) else np.empty(0, np.int64)
    keys = _keys(s1, tg, n_tg)
    rows = []
    for B in list(budgets) + [None]:
        m = np.ones(len(s1), bool) if B is None else rank <= B
        rows.append({'budget': 'none' if B is None else int(B), 'cands_mean': m.sum() / max(n_s1, 1),
                     'recall': float(np.isin(true_k, keys[m]).mean()) if len(true_k) else float('nan')})
    return rows


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _pct(x):
    return f'{100 * x:6.2f}%' if x == x else '   n/a'


def format_report(rep, contrib=None, sweep=None, bsweep=None, title='BLOCKING EVALUATION'):
    L = [f'=== {title} ===', '',
         f"Source 1 entities:          {rep['source1_entities']:>12,}",
         f"Targets (S2 + S3):          {rep['targets']:>12,}",
         f"True matches (pairs):       {rep['true_pairs']:>12,}",
         f"Candidate pairs:            {rep['candidate_pairs']:>12,}", '',
         'Candidate statistics (per Source 1):',
         f"  mean:                     {rep['cands_mean']:>12.1f}",
         f"  median:                   {rep['cands_median']:>12.1f}",
         f"  p95:                      {rep['cands_p95']:>12.1f}",
         f"  max:                      {rep['cands_max']:>12,}", '',
         'Candidate recall:',
         f"  pair recall (PC):         {_pct(rep['pair_recall']):>12}",
         f"  macro recall:             {_pct(rep['macro_recall']):>12}",
         f"  entities fully recalled:  {_pct(rep['entities_full_recall']):>12}",
         f"  F0.5 ceiling:             {rep['f05_ceiling']:>12.4f}", '',
         'Precision of the blocked set:',
         f"  pair quality (PQ):        {_pct(rep['pair_quality']):>12}   (true pairs / candidate pairs)",
         f"  false candidates per S1:  {rep['false_candidates_per_s1']:>12.1f}", '',
         f"Reduction ratio:            {rep['reduction_ratio']:>12.7f}",
         f"  vs within-country product:{rep['reduction_ratio_within_country']:>12.7f}", '',
         f"Entities with zero candidates: {rep['zero_candidate_entities']:,} "
         f"(of which with true matches: {rep['zero_candidate_with_true_matches']:,}, "
         f"true pairs lost: {rep['true_pairs_lost_to_empty_lists']:,})",
         f"Singletons: {rep['singletons']:,}; with >= 1 candidate: {rep['singletons_with_candidates']:,} "
         f"(mean {rep['singleton_cands_mean']:.1f} candidates)", '',
         f"Penalised scores (lambda={rep['penalty_lambda']}, ref budget={rep['penalty_ref_budget']}):",
         f"  penalized recall:         {rep['penalized_recall']:>12.4f}",
         f"  macro penalized recall:   {rep['macro_penalized_recall']:>12.4f}",
         f"  F1(PC, PQ):               {rep['f1_pc_pq']:>12.4f}",
         f"  H(PC, RR):                {rep['h_pc_rr']:>12.4f}"]
    if rep.get('by_country'):
        L += ['', f"{'country':<12}{'entities':>10}{'true':>10}{'recall':>9}{'cands':>8}{'PQ':>8}{'ceiling':>9}"]
        for c, r in rep['by_country'].items():
            L.append(f"{c:<12}{r['entities']:>10,}{r['true_pairs']:>10,}{_pct(r['pair_recall']):>9}"
                     f"{r['cands_mean']:>8.1f}{_pct(r['pair_quality']):>8}{r['f05_ceiling']:>9.4f}")
    if contrib:
        L += ['', 'Pipeline recall (true pairs found by the pipeline alone):']
        for k, v in contrib['recall'].items():
            L.append(f'  {k:<24}{_pct(v):>10}')
        L += ['', 'Candidates per S1 by pipeline (before union / budget):']
        for k, v in contrib['candidates_per_s1'].items():
            L.append(f'  {k:<24}{v:>10.1f}')
        if contrib.get('true_pair_venn'):
            tot = max(sum(contrib['true_pair_venn'].values()), 1)
            L += ['', 'Which pipelines found each true pair (exclusive):']
            for k, v in contrib['true_pair_venn'].items():
                L.append(f'  {k:<34}{v:>10,}  {_pct(v / tot)}')
    if sweep:
        fams = [f for f in FAMILY_ORDER if f'{f}_recall' in sweep[0]]
        L += ['', 'Recall vs K (every list cut to K):',
              f"{'K':>5}{'cands/S1':>10}{'recall':>9}" + ''.join(f'{f + " rec":>14}{f + " cand":>14}' for f in fams)]
        for r in sweep:
            L.append(f"{r['K']:>5}{r['cands_mean']:>10.1f}{_pct(r['recall']):>9}"
                     + ''.join(f"{_pct(r[f + '_recall']):>14}{r[f + '_cands']:>14.1f}" for f in fams))
    if bsweep:
        L += ['', 'Recall @ candidate budget (union cut by priority):', f"{'budget':>8}{'cands/S1':>10}{'recall':>9}"]
        for r in bsweep:
            L.append(f"{str(r['budget']):>8}{r['cands_mean']:>10.1f}{_pct(r['recall']):>9}")
    return '\n'.join(L)
