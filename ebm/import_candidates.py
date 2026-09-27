"""
Import an existing candidate_pairs.tsv (e.g. the main pipeline's blocking,
which produced the 0.959 submission) into an ebm cache, instead of ebm.block.

  --cache      ebm cache whose records contain every id in the file
               (train cache for scale_val candidates; test cache for test)
  --candidates candidate_pairs.tsv (submission format), may be .gz

Writes <cache>/candidates.npz (s1, tg rows; score columns NaN, unused by
train/predict) and copies the file to <cache>/candidate_pairs.tsv, which
ebm.predict ships next to matching_results.tsv.

Usage:
  python -m ebm.import_candidates --cache cache/train --candidates scale_val_candidate_pairs.tsv
  python -m ebm.import_candidates --cache cache/test  --candidates test_candidate_pairs.tsv
"""

import argparse
import gzip
import json
import os
import shutil
import time

import numpy as np


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:8.1f}s] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--candidates', required=True)
    args = ap.parse_args()
    ids = np.load(os.path.join(args.cache, 'ids.npy'))
    with open(os.path.join(args.cache, 'meta.json')) as f:
        meta = json.load(f)
    row = {e: i for i, e in enumerate(ids.tolist())}
    log(f'cache {args.cache}: {len(ids):,} records')

    opener = gzip.open if args.candidates.endswith('.gz') else open
    s1_rows, tg_rows, missing, n_lines = [], [], 0, 0
    with opener(args.candidates, 'rt', encoding='utf-8') as f:
        header = next(f).rstrip('\n').split('\t')
        assert header[0] == 'source1_entity_id', header
        for line in f:
            s1, _, cands = line.rstrip('\n').partition('\t')
            n_lines += 1
            a = row.get(s1)
            if a is None:
                missing += 1
                continue
            for c in cands.split(','):
                b = row.get(c) if c else None
                if b is not None:
                    s1_rows.append(a); tg_rows.append(b)
                elif c:
                    missing += 1
    s1 = np.array(s1_rows, np.int32); tg = np.array(tg_rows, np.int32)
    order = np.lexsort((tg, s1))
    s1, tg = s1[order], tg[order]
    nan = np.full(len(s1), np.nan, np.float32)
    np.savez(os.path.join(args.cache, 'candidates.npz'), s1=s1, tg=tg, score_addr=nan, score_full=nan)
    dst = os.path.join(args.cache, 'candidate_pairs.tsv')
    if args.candidates.endswith('.gz'):
        with gzip.open(args.candidates, 'rb') as fi, open(dst, 'wb') as fo:
            shutil.copyfileobj(fi, fo)
    elif os.path.abspath(args.candidates) != os.path.abspath(dst):
        shutil.copyfile(args.candidates, dst)
    per = np.bincount(s1, minlength=meta['n_s1'])
    log(f'{n_lines:,} Source 1 rows, {len(s1):,} pairs ({per[per > 0].mean():.1f} per blocked Source 1); '
        f'{missing:,} ids not in the cache')
    if meta.get('labelled'):
        owner = np.load(os.path.join(args.cache, 'owner.npy'))
        indptr = np.load(os.path.join(args.cache, 'gt_indptr.npy'))
        blocked = np.unique(s1)
        truth = int(np.diff(indptr)[blocked].sum())
        log(f'blocking recall on these Source 1: {int((owner[tg] == s1).sum()) / max(truth, 1):.4f} of {truth:,} true pairs')
    log(f'wrote {args.cache}/candidates.npz and candidate_pairs.tsv')


if __name__ == '__main__':
    main()
