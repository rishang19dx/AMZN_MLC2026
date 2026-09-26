"""
Hard-negative mining for the learned encoders.

  1. embed a sample of training anchors (Source 1 with >= 1 match) and a pool of
     training targets (the anchors' true targets + random others) with the
     current model;
  2. retrieve each anchor's top_m nearest targets in its own country (ANN);
  3. drop the anchor's true matches, skip the first `skip_top` survivors
     (optional guard against label noise), keep `per_anchor` of the rest.

The next epochs attach these as extra negatives ("ABC Medical Store" vs
"ABC Medical Centre"), which random in-batch negatives almost never provide.
Uses training data only.
"""

import numpy as np

from blocker.pipelines.ann_index import search_jobs
from blocker.utils import log


def mine_hard_negatives(model, data, hcfg, ann_cfg, seed, batch_size=512, fp16=False):
    rng = np.random.default_rng(seed)
    own = data.owner()
    anchors = np.unique(own[own >= 0])
    max_a = int(hcfg.get('max_anchors', 200_000))
    if len(anchors) > max_a:
        anchors = np.sort(rng.choice(anchors, max_a, replace=False))
    is_anchor = np.zeros(len(data.s1), bool)
    is_anchor[anchors] = True
    pool = np.flatnonzero((own >= 0) & is_anchor[np.maximum(own, 0)])
    max_t = int(hcfg.get('max_targets', 1_000_000))
    others = np.setdiff1d(np.arange(len(data.tg)), pool)
    extra = max(0, max_t - len(pool))
    if extra and len(others):
        pool = np.union1d(pool, rng.choice(others, min(extra, len(others)), replace=False))
    log(f'mining hard negatives: {len(anchors):,} anchors, {len(pool):,} targets')

    q = model.embed(data.s1.iloc[anchors], 'query', batch_size, fp16)
    t = model.embed(data.tg.iloc[pool], 'target', batch_size, fp16)
    top_m, skip, keep = int(hcfg.get('top_m', 30)), int(hcfg.get('skip_top', 0)), int(hcfg.get('per_anchor', 3))
    s1c = data.s1['country_n'].to_numpy()[anchors]
    tgc = data.tg['country_n'].to_numpy()[pool]
    jobs, qa_list, tb_list = [], [], []
    for c in np.unique(s1c):
        qa, tb = np.flatnonzero(s1c == c), np.flatnonzero(tgc == c)
        if len(tb):
            jobs.append({'rows': tb, 'qrows': qa, 'k': top_m})
            qa_list.append(qa)
            tb_list.append(tb)
    out, sims = {}, []
    for qa, tb, (S, I, _) in zip(qa_list, tb_list, search_jobs(t, q, jobs, ann_cfg, seed)):
        for a_local, idx, sc in zip(qa, I, S):
            a = int(anchors[a_local])
            cand = [(int(pool[tb[x]]), s) for x, s in zip(idx, sc) if x >= 0 and own[pool[tb[x]]] != a]
            cand = cand[skip:skip + keep]
            if cand:
                out[a] = np.array([x for x, _ in cand], np.int32)
                sims.extend(s for _, s in cand)
    if sims:
        log(f'  mined {sum(len(v) for v in out.values()):,} hard negatives for {len(out):,} anchors '
            f'(mean cosine {np.mean(sims):.3f})')
    return out
