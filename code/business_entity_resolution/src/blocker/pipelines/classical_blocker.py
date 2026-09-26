"""
Pipeline C: classical (non-neural) candidate generation.

Each pass is a sparse TF-IDF top-K retrieval within the country, computed as
a chunked sparse matrix product in forked workers (src/blocking.py's
topk_sparse, the v1 blocker measured at 98.2% pair recall on local_val):

  addr   word tokens of the normalised address        house number + street, trade-name changes
  full   word tokens of name + address                joint evidence
  name   char 3-grams of name_core (off by default)   typos, spacing, transliteration
  keys   blocking keys (normalization.blocking_keys)  first significant name token, 4-char
                                                      name prefix, sorted token pair, last
                                                      token, house number + street word,
                                                      address token pair, postal code

The keys pass is an inverted index with a union over many keys, scored by
IDF-weighted overlap, so no single key has to match exactly; keys shared by
more than max_df of the targets (huge blocks) are dropped.

IDF is fitted per country on Source 1 + targets of the data being blocked
(unsupervised statistics only).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from blocking import auto_chunk, topk_sparse, within_rank
from blocker.pipelines.common import Retrieval
from blocker.utils import log


def tfidf_topk(s1_text, tg_text, k, analyzer='word', ngram=(1, 1), max_df=0.02, max_df_floor=50,
               workers=1, chunk=0):
    """(r, c, score) local-row arrays of the top-k targets per Source 1 by TF-IDF cosine."""
    empty = (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    if k <= 0 or len(s1_text) == 0 or len(tg_text) == 0:
        return empty
    kw = dict(analyzer='char_wb', ngram_range=tuple(ngram)) if analyzer == 'char_wb' else \
        dict(analyzer='word', token_pattern=r'\S+')
    vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=1 if len(tg_text) < 1000 else 2, **kw)
    try:
        vec.fit(pd.concat([s1_text, tg_text]))
    except ValueError:            # empty vocabulary (e.g. no keys at all)
        return empty
    X, Y = vec.transform(s1_text).tocsr(), vec.transform(tg_text).tocsr()
    df = np.bincount(Y.indices, minlength=Y.shape[1])
    keep = df <= max(max_df * Y.shape[0], max_df_floor)
    X, Y = X[:, keep].tocsr(), Y[:, keep].tocsr()
    return topk_sparse(X, Y, k, chunk=chunk or auto_chunk(Y.shape[0]), workers=workers)


class ClassicalBlocker:
    name = 'classical'

    def __init__(self, pcfg, workers=1):
        self.pcfg = pcfg
        self.workers = workers

    def passes(self):
        return {n: p for n, p in (self.pcfg.get('passes') or {}).items() if int(p.get('top_k', 0)) > 0}

    def retrieve(self, data, groups, k_scale=1.0):
        out = []
        for pname, p in self.passes().items():
            k = int(int(p['top_k']) * k_scale)
            parts = []
            for country, s1_rows, tg_rows in groups:
                r, c, v = tfidf_topk(data.s1[p['field']].iloc[s1_rows], data.tg[p['field']].iloc[tg_rows], k,
                                     p.get('analyzer', 'word'), (p.get('ngram_min', 1), p.get('ngram_max', 1)),
                                     float(p.get('max_df', self.pcfg.get('max_df', 0.02))),
                                     int(p.get('max_df_floor', self.pcfg.get('max_df_floor', 50))), self.workers)
                if len(r):
                    parts.append(Retrieval(pname, 'classical', k, s1_rows[r].astype(np.int32),
                                           tg_rows[c].astype(np.int32), v, within_rank(r, v)))
                log(f'  [classical/{pname}] {country:>10}: {len(s1_rows):,} x {len(tg_rows):,} -> {len(r):,} pairs')
            out.append(Retrieval.concat(parts, pname, 'classical', k))
        return out
