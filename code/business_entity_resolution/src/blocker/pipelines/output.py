"""
Writers: candidate_pairs.tsv (challenge format), debug provenance parquet
parts, and a manifest tying integer rows back to the data directory.
"""

import os
import shutil

import numpy as np
import pandas as pd

from blocker.utils import log, write_json

TSV_HEADER = 'source1_entity_id\tcandidate_entity_ids\n'


def id_order(ids):
    """Rank of each id in Python string order (deterministic tie-breaks)."""
    r = np.empty(len(ids), np.int64)
    r[np.argsort(ids, kind='stable')] = np.arange(len(ids))
    return r


def write_candidate_tsv(path, s1_ids, tg_ids, s1, tg, priority=None, order='priority'):
    """
    One row per Source 1 entity (in file order), candidate ids comma-separated,
    deduplicated, ordered by priority (desc) then id, or by id only.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    s1 = np.asarray(s1, np.int64)
    tg = np.asarray(tg, np.int64)
    key = np.unique(s1 * len(tg_ids) + tg, return_index=True)[1]      # dedup (defensive)
    s1, tg = s1[key], tg[key]
    pr = np.asarray(priority, np.float32)[key] if priority is not None and order == 'priority' \
        else np.zeros(len(s1), np.float32)
    o = np.lexsort((id_order(tg_ids)[tg], -pr, s1))
    s1, tg = s1[o], tg[o]
    bounds = np.searchsorted(s1, np.arange(len(s1_ids) + 1))
    ids = np.asarray(tg_ids, dtype=object)[tg]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(TSV_HEADER)
        for i, s1_id in enumerate(s1_ids):
            f.write(f"{s1_id}\t{','.join(ids[bounds[i]:bounds[i + 1]])}\n")
    return path


class DebugWriter:
    """Provenance of every final candidate, as parquet parts (one per chunk)."""

    def __init__(self, directory):
        self.dir = directory
        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
        self.n_parts = 0

    def write(self, u, s1_ids, tg_ids, extra=None):
        import duckdb
        cols = {'s1_id': np.asarray(s1_ids, dtype=object)[u['s1']],
                'cand_id': np.asarray(tg_ids, dtype=object)[u['tg']],
                's1_idx': u['s1'], 'tg_idx': u['tg']}
        cols.update({k: v for k, v in u.items() if k not in ('n', 's1', 'tg') and isinstance(v, np.ndarray)})
        cols.update(extra or {})
        df = pd.DataFrame(cols)
        path = os.path.join(self.dir, f'part-{self.n_parts:05d}.parquet')
        duckdb.from_df(df).write_parquet(path, compression='zstd')
        self.n_parts += 1
        return path


def write_manifest(path, data, cfg_used, stats):
    write_json({
        'data': data.name, 'n_s1': len(data.s1), 'n_tg': len(data.tg),
        'first_s1': str(data.s1_ids[0]) if len(data.s1) else None,
        'last_s1': str(data.s1_ids[-1]) if len(data.s1) else None,
        'first_tg': str(data.tg_ids[0]) if len(data.tg) else None,
        'last_tg': str(data.tg_ids[-1]) if len(data.tg) else None,
        'row_order': 'Source 1 file order; targets = Source 2 file order then Source 3 file order',
        'stats': stats, 'config': cfg_used,
    }, path)
    log(f'wrote {path}')
