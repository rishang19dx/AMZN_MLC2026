"""
Pieces shared by all pipelines: country groups and the retrieval-result format.

Every pipeline returns a list of Retrieval objects (one per pass / per source
index), each holding flat arrays over the pairs it found:
  r      Source 1 row      c      target row
  score  similarity        rank   1-based rank in that pass's list for r
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Retrieval:
    name: str          # pass name, e.g. 'bert', 'jepa', 'addr', 'full', 'keys'
    family: str        # pipeline family: 'bert' | 'jepa' | 'classical'
    k: int             # list length asked for (used to normalise ranks)
    r: np.ndarray
    c: np.ndarray
    score: np.ndarray
    rank: np.ndarray

    def __len__(self):
        return len(self.r)

    def top(self, k):
        """The same retrieval truncated to rank <= k (for K sweeps)."""
        m = self.rank <= k
        return Retrieval(self.name, self.family, min(k, self.k), self.r[m], self.c[m], self.score[m], self.rank[m])

    @staticmethod
    def concat(parts, name, family, k):
        parts = [p for p in parts if len(p)]
        if not parts:
            e = np.empty(0, np.int32)
            return Retrieval(name, family, k, e, e.copy(), np.empty(0, np.float32), np.empty(0, np.float32))
        return Retrieval(name, family, k, *(np.concatenate([getattr(p, a) for p in parts])
                                            for a in ('r', 'c', 'score', 'rank')))


def country_groups(data, fallback='global'):
    """
    [(country, s1_rows, tg_rows)]: Source 1 is matched only against targets with
    the same normalised country (true matches never cross countries). Country is
    an open-set label: an unseen country is just another group. If a country has
    no targets at all, fallback='global' searches every target instead of
    returning nothing.
    """
    s1c = data.s1['country_n'].to_numpy()
    tgc = data.tg['country_n'].to_numpy()
    tg_by = {c: np.flatnonzero(tgc == c) for c in np.unique(tgc)} if len(tgc) else {}
    groups = []
    for c in sorted(np.unique(s1c)) if len(s1c) else []:
        rows = np.flatnonzero(s1c == c)
        t = tg_by.get(c, np.empty(0, np.int64))
        if len(t) == 0 and fallback == 'global':
            t = np.arange(len(tgc))
        groups.append((c, rows, t))
    return groups


def from_topk(q_rows, t_rows, S, I, name, family, k, min_similarity=None):
    """Flatten a (n_queries, k) ANN result into a Retrieval over global rows."""
    ranks = np.broadcast_to(np.arange(1, S.shape[1] + 1, dtype=np.float32), S.shape)
    m = I >= 0
    if min_similarity is not None:
        m &= S >= float(min_similarity)
    rr = np.broadcast_to(np.asarray(q_rows)[:, None], S.shape)[m]
    return Retrieval(name, family, k, rr.astype(np.int32), np.asarray(t_rows)[I[m]].astype(np.int32),
                     S[m].astype(np.float32), ranks[m].astype(np.float32))
