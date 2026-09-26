"""
Local validation split for Business Entity Resolution.

The test set is a closed universe: test S1 is matched against test S2/S3 only,
and about 26% of S2/S3 records match nothing (distractors). A validation split
must look the same, otherwise blocking and threshold numbers measured on it are
misleading. So instead of scoring held-out S1 against the *full* train S2/S3
(where 90% of targets belong to S1 entities outside the split), each split is
its own closed universe:

  local_val   = held-out S1  +  every S2/S3 record matched to them
                              +  VAL_FRACTION of the unmatched S2/S3 records
  local_train = everything else

Every S2/S3 record matches at most one S1 entity, so this partitions S2/S3
exactly and keeps the S1 : target ratio of the test set.

Assignment uses a seeded md5 of the entity_id, so it is reproducible without
storing any state. Source lines are copied byte-for-byte: some fields contain
literal '"' characters that a quote-aware CSV writer would rewrite.

Usage:
    python src/data_loader.py
"""

import csv
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

GT_COLUMNS = ['source1_entity_id', 'matched_entity_ids']
SOURCE_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country']


def read_tsv(path, **kwargs):
    """
    Read a challenge TSV with pandas. Quote handling is disabled because the
    data contains literal '"' characters; with the default settings pandas
    would silently merge or mangle those rows.
    """
    import pandas as pd
    return pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False, **kwargs)


def in_holdout(entity_id, fraction=config.VAL_FRACTION, seed=config.SPLIT_SEED):
    """Deterministic, uniform assignment of an ID to the held-out side."""
    h = hashlib.md5(f'{seed}:{entity_id}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') < fraction * 2**64


def _check_header(line, expected, path):
    header = line.rstrip('\n').split('\t')
    if header != expected:
        raise ValueError(f'{path}: unexpected header {header}, expected {expected}')


def create_validation_split(fraction=config.VAL_FRACTION, seed=config.SPLIT_SEED):
    t0 = time.time()
    val, trn = config.split_paths('local_val'), config.split_paths('local_train')
    for p in (val, trn):
        os.makedirs(p['dir'], exist_ok=True)

    # 1. S1: split by hash.
    val_s1 = set()
    counts = {'val_s1': 0, 'train_s1': 0}
    with open(config.TRAIN_S1, encoding='utf-8') as f, \
         open(val['s1'], 'w', encoding='utf-8') as fv, \
         open(trn['s1'], 'w', encoding='utf-8') as ft:
        header = next(f)
        _check_header(header, SOURCE_COLUMNS, config.TRAIN_S1)
        fv.write(header)
        ft.write(header)
        for line in f:
            eid = line.split('\t', 1)[0]
            if in_holdout(eid, fraction, seed):
                val_s1.add(eid)
                fv.write(line)
                counts['val_s1'] += 1
            else:
                ft.write(line)
                counts['train_s1'] += 1

    # 2. Ground truth: follows its S1 entity. Remember which targets must go to
    #    val (matched to a val S1) and which hash-to-val targets are pinned to
    #    train (matched to a train S1). Both sets are ~10% of targets.
    val_targets, pinned_to_train = set(), set()
    with open(config.TRAIN_GT, encoding='utf-8') as f, \
         open(val['gt'], 'w', encoding='utf-8') as fv, \
         open(trn['gt'], 'w', encoding='utf-8') as ft:
        header = next(f)
        _check_header(header, GT_COLUMNS, config.TRAIN_GT)
        fv.write(header)
        ft.write(header)
        for line in f:
            s1_id, _, matched = line.rstrip('\n').partition('\t')
            targets = [t for t in matched.split(',') if t]
            if s1_id in val_s1:
                fv.write(line)
                val_targets.update(targets)
            else:
                ft.write(line)
                pinned_to_train.update(t for t in targets if in_holdout(t, fraction, seed))

    # 3. S2/S3: matched targets follow their S1; unmatched ones split by hash.
    for key in ('s2', 's3'):
        src = config.TRAIN_S2 if key == 's2' else config.TRAIN_S3
        n_val = n_trn = 0
        with open(src, encoding='utf-8') as f, \
             open(val[key], 'w', encoding='utf-8') as fv, \
             open(trn[key], 'w', encoding='utf-8') as ft:
            header = next(f)
            _check_header(header, SOURCE_COLUMNS, src)
            fv.write(header)
            ft.write(header)
            for line in f:
                eid = line.split('\t', 1)[0]
                if eid in val_targets or (eid not in pinned_to_train and in_holdout(eid, fraction, seed)):
                    fv.write(line)
                    n_val += 1
                else:
                    ft.write(line)
                    n_trn += 1
        counts[f'val_{key}'], counts[f'train_{key}'] = n_val, n_trn

    print(f"local_val:   S1={counts['val_s1']:,}  S2={counts['val_s2']:,}  S3={counts['val_s3']:,}")
    print(f"local_train: S1={counts['train_s1']:,}  S2={counts['train_s2']:,}  S3={counts['train_s3']:,}")
    print(f"Written to {config.SPLITS_DIR} in {time.time() - t0:.0f}s")
    return counts


def create_ce_train_split(fraction=0.05, seed='mlc26-ce'):
    """
    ce_train = a hash-sampled `fraction` of local_train's Source 1 (+ their
    ground-truth rows) against ALL local_train targets, so blocking produces
    realistic hard negatives. The target files are symlinks, not copies.
    Disjoint from local_val and test by construction.
    """
    src, dst = config.split_paths('local_train'), config.split_paths('ce_train')
    os.makedirs(dst['dir'], exist_ok=True)
    keep = set()
    with open(src['s1'], encoding='utf-8') as f, open(dst['s1'], 'w', encoding='utf-8') as o:
        o.write(next(f))
        for line in f:
            eid = line.split('\t', 1)[0]
            if in_holdout(eid, fraction, seed):
                keep.add(eid)
                o.write(line)
    with open(src['gt'], encoding='utf-8') as f, open(dst['gt'], 'w', encoding='utf-8') as o:
        o.write(next(f))
        o.writelines(line for line in f if line.split('\t', 1)[0] in keep)
    for key in ('s2', 's3'):
        if os.path.lexists(dst[key]):
            os.remove(dst[key])
        os.symlink(os.path.relpath(src[key], dst['dir']), dst[key])
    print(f'ce_train: {len(keep):,} Source 1 entities against all local_train targets -> {dst["dir"]}')


def create_scale_val_split(fraction=0.20, seed='mlc26-scale'):
    """
    scale_val = a hash-sampled `fraction` of local_train's Source 1 (excluding
    ce_train's) against ALL local_train targets. local_val is a ~9x smaller
    universe than test, and on test the candidates are far more similar to
    their Source 1 (address TF-IDF cosine median 0.46 vs 0.32; RapidFuzz
    address token-set 0.72 vs 0.57): a bigger pool has closer look-alikes.
    A matcher trained and scored here sees test-sized crowding and IDF.
    Same layout as ce_train (targets are symlinks).
    """
    src, dst = config.split_paths('local_train'), config.split_paths('scale_val')
    ce = config.split_paths('ce_train')['s1']
    exclude = set()
    if os.path.exists(ce):
        with open(ce, encoding='utf-8') as f:
            next(f)
            exclude = {line.split('\t', 1)[0] for line in f}
    os.makedirs(dst['dir'], exist_ok=True)
    keep = set()
    with open(src['s1'], encoding='utf-8') as f, open(dst['s1'], 'w', encoding='utf-8') as o:
        o.write(next(f))
        for line in f:
            eid = line.split('\t', 1)[0]
            if eid not in exclude and in_holdout(eid, fraction, seed):
                keep.add(eid)
                o.write(line)
    with open(src['gt'], encoding='utf-8') as f, open(dst['gt'], 'w', encoding='utf-8') as o:
        o.write(next(f))
        o.writelines(line for line in f if line.split('\t', 1)[0] in keep)
    for key in ('s2', 's3'):
        if os.path.lexists(dst[key]):
            os.remove(dst[key])
        os.symlink(os.path.relpath(src[key], dst['dir']), dst[key])
    print(f'scale_val: {len(keep):,} Source 1 entities against all local_train targets -> {dst["dir"]}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ce-train', action='store_true', help='also (only) build the cross-encoder training split')
    ap.add_argument('--scale-val', action='store_true', help='(only) build the test-scale matcher split')
    args = ap.parse_args()
    if args.scale_val:
        create_scale_val_split()
    elif args.ce_train:
        create_ce_train_split()
    else:
        create_validation_split()
