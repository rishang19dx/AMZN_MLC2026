"""
Dense retrieval for the two learned pipelines (A: bert, B: jepa).

  1. embed every Source 1 record (role 'query') and every target (role
     'target') independently: no pairwise model inference;
  2. per country, build one ANN index per source (S2, S3);
  3. every Source 1 retrieves its top_k_source2 and top_k_source3 targets.

Embeddings are written to float16 memmaps in the cache directory (~5 GB for
the 10M test targets at 256 dims) and indexed batch-wise, so RAM stays bounded.
"""

import os

import numpy as np

from blocker.pipelines.ann_index import search_jobs
from blocker.pipelines.common import Retrieval, from_topk
from blocker.utils import log


class DenseBlocker:
    def __init__(self, model, pcfg, ann_cfg, runtime, cache_dir, seed=0):
        self.model, self.pcfg, self.ann_cfg, self.runtime = model, pcfg, ann_cfg, runtime
        self.cache_dir, self.seed = cache_dir, seed
        self.name = model.kind

    def _embed(self, df, role, tag):
        bs = int(self.model.cfg.get('eval_batch_size', 512))
        fp16 = bool(self.runtime.get('fp16_inference', True))
        dim = self.model.cfg.get('embedding_dim') or 0
        if self.cache_dir and len(df) > 100_000 and dim:
            os.makedirs(self.cache_dir, exist_ok=True)
            path = os.path.join(self.cache_dir, f'{self.name}_{tag}_{role}.f16')
            out = np.lib.format.open_memmap(path + '.npy', mode='w+', dtype=np.float16, shape=(len(df), int(dim)))
            self.model.embed(df, role, bs, fp16, out=out)
            out.flush()
            return out
        return self.model.embed(df, role, bs, fp16)

    def retrieve(self, data, groups, k_scale=1.0):
        """[Retrieval] for this pipeline. k_scale lets a K sweep ask for longer lists."""
        ks = {2: int(self.pcfg.get('top_k_source2', 20) * k_scale),
              3: int(self.pcfg.get('top_k_source3', 20) * k_scale)}
        log(f'[{self.name}] embedding {len(data.s1):,} Source 1 + {len(data.tg):,} targets')
        q = self._embed(data.s1, 'query', data.name)
        t = self._embed(data.tg, 'target', data.name)
        src = data.tg['source'].to_numpy()
        minsim = self.pcfg.get('min_similarity')
        jobs, meta = [], []
        for country, s1_rows, tg_rows in groups:
            for s in (2, 3):
                rows = tg_rows[src[tg_rows] == s]
                if ks[s] <= 0 or len(rows) == 0 or len(s1_rows) == 0:
                    continue
                save = (os.path.join(self.cache_dir, 'indices', self.name, f'{country}_S{s}.faiss')
                        if self.ann_cfg.get('save_indices') and self.cache_dir else None)
                jobs.append({'rows': rows, 'qrows': s1_rows, 'k': ks[s], 'save_path': save})
                meta.append((country, s, s1_rows, rows))
        results = search_jobs(t, q, jobs, self.ann_cfg, self.seed, self.cache_dir)
        parts = {2: [], 3: []}
        for (country, s, s1_rows, rows), (S, I, kind) in zip(meta, results):
            res = from_topk(s1_rows, rows, S, I, self.name, self.name, ks[s], minsim)
            parts[s].append(res)
            log(f'  [{self.name}] {country:>10} S{s}: {len(s1_rows):,} x {len(rows):,} ({kind}) -> {len(res):,} pairs')
        # one retrieval per pipeline: a target is in exactly one source, so the two
        # per-source lists never share a pair; ranks stay per source (1..k_source)
        return [Retrieval.concat(parts[2] + parts[3], self.name, self.name, max(ks.values()))]
