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
  local_fit   = the next FIT_FRACTION of S1 by the same hash, built the same
                way (its S1 + their matched targets + FIT_FRACTION of the
                unmatched records). A subset of local_train, disjoint from
                local_val: the matcher's training split on a 16 GB machine,
                where all of local_train does not fit in memory.

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


def _hash64(entity_id, seed):
    return int.from_bytes(hashlib.md5(f'{seed}:{entity_id}'.encode()).digest()[:8], 'big')


def in_holdout(entity_id, fraction=config.VAL_FRACTION, seed=config.SPLIT_SEED):
    """Deterministic, uniform assignment of an ID to the held-out side."""
    return _hash64(entity_id, seed) < fraction * 2**64


def in_fit(entity_id, fraction=config.VAL_FRACTION, fit_fraction=config.FIT_FRACTION, seed=config.SPLIT_SEED):
    """The hash band right after the held-out one: [fraction, fraction + fit_fraction)."""
    return fraction * 2**64 <= _hash64(entity_id, seed) < (fraction + fit_fraction) * 2**64


def _check_header(line, expected, path):
    header = line.rstrip('\n').split('\t')
    if header != expected:
        raise ValueError(f'{path}: unexpected header {header}, expected {expected}')


def create_validation_split(fraction=config.VAL_FRACTION, seed=config.SPLIT_SEED, fit_fraction=config.FIT_FRACTION):
    t0 = time.time()
    if not 0 <= fit_fraction <= 1 - fraction:
        raise ValueError(f'FIT_FRACTION={fit_fraction} must be in [0, {1 - fraction}]')
    val, trn, fit = (config.split_paths(n) for n in ('local_val', 'local_train', 'local_fit'))
    for p in (val, trn, fit):
        os.makedirs(p['dir'], exist_ok=True)
    is_fit = lambda eid: in_fit(eid, fraction, fit_fraction, seed)

    # 1. S1: split by hash. local_fit takes the next hash band out of local_train.
    val_s1, fit_s1 = set(), set()
    counts = {'val_s1': 0, 'train_s1': 0, 'fit_s1': 0}
    with open(config.TRAIN_S1, encoding='utf-8') as f, \
         open(val['s1'], 'w', encoding='utf-8') as fv, \
         open(trn['s1'], 'w', encoding='utf-8') as ft, \
         open(fit['s1'], 'w', encoding='utf-8') as ff:
        header = next(f)
        _check_header(header, SOURCE_COLUMNS, config.TRAIN_S1)
        fv.write(header)
        ft.write(header)
        ff.write(header)
        for line in f:
            eid = line.split('\t', 1)[0]
            if in_holdout(eid, fraction, seed):
                val_s1.add(eid)
                fv.write(line)
                counts['val_s1'] += 1
            else:
                ft.write(line)
                counts['train_s1'] += 1
                if is_fit(eid):
                    fit_s1.add(eid)
                    ff.write(line)
                    counts['fit_s1'] += 1

    # 2. Ground truth: follows its S1 entity. Remember which targets must go to
    #    val (matched to a val S1) and which hash-to-val targets are pinned to
    #    train (matched to a train S1). Both sets are ~10% of targets.
    #    Same for local_fit: fit_targets must go to fit, pinned_out_of_fit
    #    (fit-band hash, but matched to a non-fit S1) must not.
    val_targets, pinned_to_train = set(), set()
    fit_targets, pinned_out_of_fit = set(), set()
    with open(config.TRAIN_GT, encoding='utf-8') as f, \
         open(val['gt'], 'w', encoding='utf-8') as fv, \
         open(trn['gt'], 'w', encoding='utf-8') as ft, \
         open(fit['gt'], 'w', encoding='utf-8') as ff:
        header = next(f)
        _check_header(header, GT_COLUMNS, config.TRAIN_GT)
        fv.write(header)
        ft.write(header)
        ff.write(header)
        for line in f:
            s1_id, _, matched = line.rstrip('\n').partition('\t')
            targets = [t for t in matched.split(',') if t]
            if s1_id in val_s1:
                fv.write(line)
                val_targets.update(targets)
            else:
                ft.write(line)
                pinned_to_train.update(t for t in targets if in_holdout(t, fraction, seed))
            if s1_id in fit_s1:
                ff.write(line)
                fit_targets.update(targets)
            else:
                pinned_out_of_fit.update(t for t in targets if is_fit(t))

    # 3. S2/S3: matched targets follow their S1; unmatched ones split by hash.
    for key in ('s2', 's3'):
        src = config.TRAIN_S2 if key == 's2' else config.TRAIN_S3
        n_val = n_trn = n_fit = 0
        with open(src, encoding='utf-8') as f, \
             open(val[key], 'w', encoding='utf-8') as fv, \
             open(trn[key], 'w', encoding='utf-8') as ft, \
             open(fit[key], 'w', encoding='utf-8') as ff:
            header = next(f)
            _check_header(header, SOURCE_COLUMNS, src)
            fv.write(header)
            ft.write(header)
            ff.write(header)
            for line in f:
                eid = line.split('\t', 1)[0]
                if eid in val_targets or (eid not in pinned_to_train and in_holdout(eid, fraction, seed)):
                    fv.write(line)
                    n_val += 1
                else:
                    ft.write(line)
                    n_trn += 1
                    # (a fit-band target never takes the branch above: its hash
                    # is outside the val band and its S1, if any, is not in val)
                    if eid in fit_targets or (eid not in pinned_out_of_fit and is_fit(eid)):
                        ff.write(line)
                        n_fit += 1
        counts[f'val_{key}'], counts[f'train_{key}'], counts[f'fit_{key}'] = n_val, n_trn, n_fit

    print(f"local_val:   S1={counts['val_s1']:,}  S2={counts['val_s2']:,}  S3={counts['val_s3']:,}")
    print(f"local_train: S1={counts['train_s1']:,}  S2={counts['train_s2']:,}  S3={counts['train_s3']:,}")
    print(f"local_fit:   S1={counts['fit_s1']:,}  S2={counts['fit_s2']:,}  S3={counts['fit_s3']:,}  (subset of local_train)")
    print(f"Written to {config.SPLITS_DIR} in {time.time() - t0:.0f}s")
    return counts


if __name__ == '__main__':
    create_validation_split()
