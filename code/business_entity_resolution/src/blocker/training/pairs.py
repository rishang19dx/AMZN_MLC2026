"""
Training pairs for the representation models.

  positives  (Source 1 row, matching Source 2/3 row) from the ground truth
  negatives  * in-batch: every other pair's target in the batch (masked when it
               is actually another true match of the same Source 1 entity)
             * random:   random_negative_pairs(), same-country non-matches
             * hard:     mined nearest-neighbour non-matches (hard_negative_mining.py)

Only the training universe is passed in here: validation entities and their
targets were removed by data.split_holdout() before any pair is built.
"""

import numpy as np


def positive_pairs(data):
    """(n, 2) int32 array of (s1_row, target_row)."""
    i, j = data.positive_pairs()
    return np.stack([i, j], axis=1).astype(np.int32)


def random_negative_pairs(data, per_anchor, seed, anchors=None, same_country=True):
    """
    (n, 2) array of (s1_row, target_row) where the target is NOT a true match of
    that Source 1 row. With same_country, targets are drawn from the anchor's
    country (harder, and matches never cross countries).
    """
    rng = np.random.default_rng(seed)
    own = data.owner()
    anchors = np.arange(len(data.s1)) if anchors is None else np.asarray(anchors)
    s1_country = data.s1['country_n'].to_numpy() if same_country else np.zeros(len(data.s1), object)
    tg_country = data.tg['country_n'].to_numpy() if same_country else np.zeros(len(data.tg), object)
    by_country = {c: np.flatnonzero(tg_country == c) for c in np.unique(tg_country)}
    everything = np.arange(len(data.tg))
    out = []
    for a in anchors:
        pool = by_country.get(s1_country[a], everything)
        if len(pool) == 0:
            pool = everything
        if len(pool) == 0:
            continue
        got, tries = 0, 0
        while got < per_anchor and tries < per_anchor * 10:
            t = pool[rng.integers(len(pool))]
            tries += 1
            if own[t] != a:
                out.append((a, t))
                got += 1
    return np.array(out, np.int32).reshape(-1, 2)


class PairBatcher:
    """
    Epoch iterator over positive pairs.

    * `max_pairs` caps pairs per epoch (a fresh random sample each epoch).
    * With `country_homogeneous`, each batch holds pairs of one country, so the
      in-batch negatives are same-country records (harder, like real blocking).
    * `hard_negs[s1_row]` -> array of mined negative target rows; `per_item`
      of them are attached to every pair.
    """

    def __init__(self, pairs, s1_country, batch_size, seed, max_pairs=0, country_homogeneous=True):
        self.pairs = pairs
        self.s1_country = s1_country
        self.batch_size = int(batch_size)
        self.seed = seed
        self.max_pairs = int(max_pairs or 0)
        self.country_homogeneous = country_homogeneous

    def n_batches(self):
        n = len(self.pairs) if not self.max_pairs else min(self.max_pairs, len(self.pairs))
        return max(1, -(-n // self.batch_size))

    def epoch(self, epoch, hard_negs=None, per_item=1):
        rng = np.random.default_rng(self.seed + 1000 * epoch)
        idx = rng.permutation(len(self.pairs))
        if self.max_pairs and len(idx) > self.max_pairs:
            idx = idx[:self.max_pairs]
        if self.country_homogeneous:
            idx = idx[np.argsort(self.s1_country[self.pairs[idx, 0]], kind='stable')]
            country = self.s1_country[self.pairs[idx, 0]]
            batches = []
            starts = np.r_[0, np.flatnonzero(country[1:] != country[:-1]) + 1, len(idx)]
            for a, b in zip(starts[:-1], starts[1:]):
                batches += [idx[s:min(s + self.batch_size, b)] for s in range(a, b, self.batch_size)]
        else:
            batches = [idx[s:s + self.batch_size] for s in range(0, len(idx), self.batch_size)]
        for bi in rng.permutation(len(batches)):
            b = self.pairs[batches[bi]]
            hn = None
            if hard_negs and per_item > 0:
                hn = []
                for a in b[:, 0]:
                    cand = hard_negs.get(int(a))
                    if cand is not None and len(cand):
                        hn.extend(rng.choice(cand, size=min(per_item, len(cand)), replace=False).tolist())
                hn = np.array(hn, np.int32)
            yield b, hn
