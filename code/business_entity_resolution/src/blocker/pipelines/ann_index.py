"""
Approximate nearest-neighbour search over L2-normalised embeddings (inner
product = cosine), with FAISS when installed and an exact numpy fallback.

index_type:
  flat     exact (IndexFlatIP)
  ivf      inverted file, full vectors (IndexIVFFlat)
  ivf_sq8  inverted file, 8-bit scalar-quantised vectors: 4x less memory,
           the default for large target sets (10M x 256 dims ~ 2.6 GB)
  hnsw     graph index (IndexHNSWFlat)
  auto     flat below `exact_threshold` vectors, ivf_sq8 above

Vectors are added in batches, so a float16 np.memmap on disk can be indexed
without ever loading it fully as float32.
"""

import math
import os
import sys
import tempfile

import numpy as np

from blocker.utils import log

try:
    import faiss
except ImportError:          # pragma: no cover - exercised only without faiss
    faiss = None


def _torch_conflict():
    """macOS wheels: multi-threaded faiss next to torch segfaults (see blocker/__init__.py)."""
    return sys.platform == 'darwin' and 'torch' in sys.modules


def _guard_threads():
    if faiss is not None and _torch_conflict():
        faiss.omp_set_num_threads(1)


def exact_topk(queries, vectors, k, chunk=4096):
    """Exact inner-product top-k (numpy). Returns (scores, indices), -1 padded."""
    n = len(vectors)
    kk = min(k, n)
    S = np.full((len(queries), k), -np.inf, np.float32)
    I = np.full((len(queries), k), -1, np.int64)
    if kk == 0:
        return S, I
    V = np.asarray(vectors, np.float32)
    for s in range(0, len(queries), chunk):
        sim = np.asarray(queries[s:s + chunk], np.float32) @ V.T
        part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk] if kk < n else np.tile(np.arange(n), (len(sim), 1))
        ps = np.take_along_axis(sim, part, 1)
        o = np.argsort(-ps, axis=1, kind='stable')
        I[s:s + chunk, :kk] = np.take_along_axis(part, o, 1)
        S[s:s + chunk, :kk] = np.take_along_axis(ps, o, 1)
    return S, I


class AnnIndex:
    def __init__(self, dim, cfg=None):
        cfg = cfg or {}
        self.dim = int(dim)
        self.cfg = cfg
        self.kind = None
        self.index = None
        self._vectors = None      # numpy fallback
        self.n = 0

    def _choose(self, n):
        kind = self.cfg.get('index_type', 'auto')
        if faiss is None:
            return 'numpy'
        if kind == 'auto':
            kind = 'flat' if n <= int(self.cfg.get('exact_threshold', 200_000)) else 'ivf_sq8'
        return kind

    def build(self, vectors, seed=0, batch=500_000):
        _guard_threads()
        n = len(vectors)
        self.n = n
        self.kind = self._choose(n)
        if self.kind == 'numpy':
            self._vectors = np.asarray(vectors, np.float32)
            return self
        d = self.dim
        if self.kind == 'flat':
            index = faiss.IndexFlatIP(d)
        elif self.kind == 'hnsw':
            index = faiss.IndexHNSWFlat(d, int(self.cfg.get('hnsw_m', 32)), faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = max(40, int(self.cfg.get('ef_search', 128)))
        elif self.kind in ('ivf', 'ivf_sq8'):
            nlist = int(self.cfg.get('nlist', 0)) or int(4 * math.sqrt(max(n, 1)))
            nlist = max(1, min(nlist, n // 39 if n >= 39 else 1))
            quant = faiss.IndexFlatIP(d)
            if self.kind == 'ivf':
                index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexIVFScalarQuantizer(quant, d, nlist, faiss.ScalarQuantizer.QT_8bit,
                                                      faiss.METRIC_INNER_PRODUCT)
            rng = np.random.default_rng(seed)
            m = min(n, int(self.cfg.get('train_sample', 262_144)))
            sample = np.sort(rng.choice(n, size=m, replace=False))
            index.train(np.ascontiguousarray(np.asarray(vectors[sample], np.float32)))
            index.nprobe = min(nlist, int(self.cfg.get('nprobe', 48)))
            self._quant = quant
        else:
            raise ValueError(f'unknown index_type {self.kind!r}')
        for s in range(0, n, batch):
            index.add(np.ascontiguousarray(np.asarray(vectors[s:s + batch], np.float32)))
        if self.kind == 'hnsw':
            index.hnsw.efSearch = int(self.cfg.get('ef_search', 128))
        self.index = index
        return self

    def search(self, queries, k):
        """(scores, indices) of shape (len(queries), k); missing slots are (-inf, -1)."""
        k = int(k)
        if self.n == 0 or k <= 0 or len(queries) == 0:
            return (np.full((len(queries), max(k, 0)), -np.inf, np.float32),
                    np.full((len(queries), max(k, 0)), -1, np.int64))
        if self.kind == 'numpy':
            return exact_topk(queries, self._vectors, k)
        _guard_threads()
        bs = int(self.cfg.get('search_batch', 16384))
        S = np.empty((len(queries), k), np.float32)
        I = np.empty((len(queries), k), np.int64)
        for s in range(0, len(queries), bs):
            q = np.ascontiguousarray(np.asarray(queries[s:s + bs], np.float32))
            S[s:s + bs], I[s:s + bs] = self.index.search(q, k)
        S[I < 0] = -np.inf
        return S, I

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if self.kind == 'numpy':
            np.save(path + '.npy', self._vectors)
        else:
            faiss.write_index(self.index, path)
        log(f'  saved {self.kind} index ({self.n:,} vectors) -> {path}')

    @classmethod
    def load(cls, path, dim, cfg=None):
        obj = cls(dim, cfg)
        if os.path.exists(path + '.npy'):
            obj.kind, obj._vectors = 'numpy', np.load(path + '.npy')
            obj.n = len(obj._vectors)
        else:
            obj.index = faiss.read_index(path)
            obj.kind, obj.n = 'faiss', obj.index.ntotal
        return obj


# ---------------------------------------------------------------------------
# Batched jobs, optionally in a torch-free subprocess
# ---------------------------------------------------------------------------

def _as_path(arr, workdir, name):
    """File path of an embedding array: memmaps are reused, arrays are dumped."""
    fn = getattr(arr, 'filename', None)
    if fn:
        return str(fn)
    path = os.path.join(workdir, name + '.npy')
    np.save(path, np.asarray(arr))
    return path


def _run_jobs(specs, cfg, seed, out_dir):
    """Worker body: build one index per job and search it; results go to .npy files."""
    V_cache = {}
    out = []
    for n, job in enumerate(specs):
        V = V_cache.setdefault(job['vectors'], np.load(job['vectors'], mmap_mode='r'))
        Q = V_cache.setdefault(job['queries'], np.load(job['queries'], mmap_mode='r'))
        index = AnnIndex(V.shape[1], cfg).build(V[job['rows']], seed)
        S, I = index.search(Q[job['qrows']], job['k'])
        if job.get('save_path'):
            index.save(job['save_path'])
        sp, ip = os.path.join(out_dir, f'S{n}.npy'), os.path.join(out_dir, f'I{n}.npy')
        np.save(sp, S)
        np.save(ip, I)
        out.append((sp, ip, index.kind))
    return out


def search_jobs(vectors, queries, jobs, cfg, seed=0, workdir=None):
    """
    Run several (index, search) jobs over the same target / query embeddings.
    jobs: [{'rows': target rows to index, 'qrows': query rows, 'k': int, 'save_path': opt}]
    Returns [(scores, indices, index_kind)], indices local to each job's `rows`.

    ann.isolate_process: 'auto' (default) runs the jobs in a spawned subprocess
    that never imports torch whenever torch is loaded on macOS, so faiss keeps
    all its threads; 'true' / 'false' force it.
    """
    iso = cfg.get('isolate_process', 'auto')
    isolate = _torch_conflict() if iso == 'auto' else bool(iso)
    if not isolate:
        out = []
        for job in jobs:
            index = AnnIndex(vectors.shape[1], cfg).build(vectors[job['rows']], seed)
            S, I = index.search(queries[job['qrows']], job['k'])
            if job.get('save_path'):
                index.save(job['save_path'])
            out.append((S, I, index.kind))
        return out
    import multiprocessing as mp
    if workdir:
        os.makedirs(workdir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        specs = [dict(job, vectors=_as_path(vectors, tmp, 'vectors'), queries=_as_path(queries, tmp, 'queries'))
                 for job in jobs]
        with mp.get_context('spawn').Pool(1) as pool:
            res = pool.apply(_run_jobs, (specs, cfg, seed, tmp))
        return [(np.load(sp), np.load(ip), kind) for sp, ip, kind in res]
