"""
Preprocess one split once into memory-mappable arrays (streaming, parallel).

The original script re-tokenised every pair on every epoch in Python and
indexed everything through SQLite, so the GPU waited on the CPU. Here every
record is normalised and tokenised exactly once, by N worker processes, and
the result is a padded int32 array that training slices directly.

Outputs in --out:
  ids.npy            entity ids (row order: S1, then S2, then S3)
  tokens.npy         int32 [N, SEQ_LEN] hashed token ids (0 = padding)
  fields.npy         int8  [N, SEQ_LEN] field id per token
  source.npy         int8  [N]  1 / 2 / 3
  country.npy        int16 [N]  index into meta.json "countries" (open set)
  n_s1               in meta.json; S1 rows are 0 .. n_s1-1
  gt_indptr.npy, gt_targets.npy   CSR: S1 row -> matched target rows (labelled splits)
  owner.npy          int32 [N] target row -> its S1 row, or -1 (labelled splits)
  val.npy            bool [n_s1] validation Source 1 (stable hash of the id)
  translit.json      native-script word -> Latin word, learned from TRAIN pairs only

Usage:
  python -m ebm.preprocess --split-dir dataset/train --prefix train --out cache/train --learn-dictionary
  python -m ebm.preprocess --split-dir dataset/test  --prefix test  --out cache/test --dictionary cache/train/translit.json
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

from ebm import normalize as nz

_DICT = None


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:8.1f}s] {msg}', flush=True)


def lines(path):
    """Yield (entity_id, name, address, country) exactly as tab-split, no quote handling."""
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            if line == '\n':
                continue
            p = line.rstrip('\n').split('\t')
            p += [''] * (4 - len(p))
            yield p[0], p[1], p[2], p[3]


def chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def _init(dictionary):
    global _DICT
    _DICT = dictionary


def _tokenize_chunk(rows):
    tok = np.zeros((len(rows), nz.SEQ_LEN), np.int32)
    fld = np.zeros((len(rows), nz.SEQ_LEN), np.int8)
    for i, (_, name, addr, country) in enumerate(rows):
        t, f = nz.tokenize(name, addr, country, _DICT)
        tok[i, :len(t)] = t
        fld[i, :len(f)] = f
    return tok, fld


# ---------------------------------------------------------------------------
# native-script dictionary, learned from training pairs only
# ---------------------------------------------------------------------------

_INDIC_WORD = re.compile('[ऀ-෿]')


def learn_dictionary(paths, gt_path, val_fraction, min_count=2, min_share=0.5):
    """Position-aligned (native word -> S1 word) counts over train-split pairs whose
    two names have the same number of words; keep confident majorities."""
    s1_names = {eid: name for eid, name, _, _ in lines(paths[0]) if not is_val(eid, val_fraction)}
    native = {}
    for p in paths[1:]:
        for eid, name, _, _ in lines(p):
            if _INDIC_WORD.search(name):
                native[eid] = name
    counts = defaultdict(Counter)
    with open(gt_path, encoding='utf-8') as f:
        next(f)
        for line in f:
            s1, _, matched = line.rstrip('\n').partition('\t')
            a = s1_names.get(s1)
            if a is None:
                continue
            for t in matched.split(','):
                b = native.get(t)
                if b is None:
                    continue
                wa = re.findall(r'\w+', nz._strip_accents(a).lower())
                wb = b.split()
                if len(wa) == len(wb):
                    for x, y in zip(wb, wa):
                        if _INDIC_WORD.search(x):
                            counts[x][y] += 1
    out = {}
    for w, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            out[w] = best
    return out


def is_val(entity_id, fraction):
    h = int(hashlib.blake2b(entity_id.encode(), digest_size=8).hexdigest(), 16)
    return h / 2**64 < fraction


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-dir', required=True)
    ap.add_argument('--prefix', required=True, help='file prefix, e.g. train / test / local_val')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dictionary', help='translit.json from the train cache (for test)')
    ap.add_argument('--learn-dictionary', action='store_true', help='learn translit.json from this split (train only)')
    ap.add_argument('--val-fraction', type=float, default=0.1)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('EBM_WORKERS', min(32, os.cpu_count()))))
    ap.add_argument('--chunk', type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    paths = [os.path.join(args.split_dir, f'{args.prefix}_source{k}.tsv') for k in (1, 2, 3)]
    gt_path = os.path.join(args.split_dir, f'{args.prefix}_ground_truth.tsv')
    labelled = os.path.exists(gt_path)

    dictionary = {}
    if args.learn_dictionary:
        dictionary = learn_dictionary(paths, gt_path, args.val_fraction)
        log(f'learned native-script dictionary: {len(dictionary):,} words (train-split pairs only)')
    elif args.dictionary:
        with open(args.dictionary, encoding='utf-8') as f:
            dictionary = json.load(f)
        log(f'loaded dictionary: {len(dictionary):,} words')
    with open(os.path.join(args.out, 'translit.json'), 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False)

    # pass 1: ids, source, country (cheap, streaming) -> row counts for preallocation
    ids, source, country_names = [], [], []
    for k, p in enumerate(paths, 1):
        for eid, _, _, c in lines(p):
            ids.append(eid)
            source.append(k)
            country_names.append(c.strip())
        log(f'source {k}: {len(ids):,} rows so far')
    n = len(ids)
    n_s1 = source.count(1)
    countries = sorted(set(country_names))
    cidx = {c: i for i, c in enumerate(countries)}
    np.save(os.path.join(args.out, 'ids.npy'), np.array(ids))
    np.save(os.path.join(args.out, 'source.npy'), np.array(source, np.int8))
    np.save(os.path.join(args.out, 'country.npy'), np.array([cidx[c] for c in country_names], np.int16))
    del country_names, source

    # pass 2: tokens, in parallel, written straight into memory-mapped arrays
    tok = np.lib.format.open_memmap(os.path.join(args.out, 'tokens.npy'), 'w+', np.int32, (n, nz.SEQ_LEN))
    fld = np.lib.format.open_memmap(os.path.join(args.out, 'fields.npy'), 'w+', np.int8, (n, nz.SEQ_LEN))
    row = 0
    stream = (r for p in paths for r in lines(p))
    with Pool(args.workers, initializer=_init, initargs=(dictionary,)) as pool:
        for t, f in pool.imap(_tokenize_chunk, chunks(stream, args.chunk)):
            tok[row:row + len(t)] = t
            fld[row:row + len(f)] = f
            row += len(t)
            if row % (args.chunk * 50) < args.chunk:
                log(f'tokenised {row:,} / {n:,}')
    assert row == n, (row, n)
    tok.flush(); fld.flush()
    lengths = (fld != 0).sum(1)
    log(f'tokenised {n:,} records with {args.workers} workers; tokens per record mean {lengths.mean():.1f}, '
        f'max {lengths.max()}, empty {int((lengths == 0).sum()):,}')

    meta = {'n': n, 'n_s1': n_s1, 'countries': countries, 'seq_len': nz.SEQ_LEN,
            'hash_buckets': nz.HASH_BUCKETS, 'n_fields': nz.N_FIELDS, 'labelled': labelled,
            'val_fraction': args.val_fraction, 'prefix': args.prefix}

    s1_val = np.array([is_val(e, args.val_fraction) for e in ids[:n_s1]])
    np.save(os.path.join(args.out, 'val.npy'), s1_val)
    if labelled:
        index = {e: i for i, e in enumerate(ids)}
        indptr, targets = [0], []
        owner = np.full(n, -1, np.int32)
        with open(gt_path, encoding='utf-8') as f:
            next(f)
            gt = {}
            for line in f:
                s1, _, matched = line.rstrip('\n').partition('\t')
                gt[s1] = [index[t] for t in matched.split(',') if t in index]
        for i in range(n_s1):
            ts = gt.get(ids[i], [])
            targets.extend(ts)
            indptr.append(len(targets))
            owner[ts] = i
        np.save(os.path.join(args.out, 'gt_indptr.npy'), np.array(indptr, np.int64))
        np.save(os.path.join(args.out, 'gt_targets.npy'), np.array(targets, np.int32))
        np.save(os.path.join(args.out, 'owner.npy'), owner)
        meta['n_pairs'] = len(targets)
        log(f'ground truth: {len(targets):,} pairs; validation Source 1: {int(s1_val.sum()):,} of {n_s1:,}')
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    log(f'done -> {args.out}')


if __name__ == '__main__':
    main()
