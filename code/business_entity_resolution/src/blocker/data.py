"""
Directory-based data access for the blocker.

A data directory holds `*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv` and,
for labelled data, `*_ground_truth.tsv` (the official train/test folders and
the splits written by src/data_loader.py all follow this layout).

Records are kept in file order: Source 1 in `s1`; Source 2 then Source 3 in
`tg` ("targets"). Every candidate pair is stored as integer row positions into
these two frames, so the same directory always gives the same indices.

Splits are entity-level closed universes, exactly like src/data_loader.py:
a held-out Source 1 entity takes all of its matched targets with it, and the
unmatched targets (distractors) are split by the same hash. No positive pair of
a validation entity can leak into training.
"""

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_loader import SOURCE_COLUMNS, in_holdout, read_tsv


@dataclass
class ERData:
    s1: pd.DataFrame
    tg: pd.DataFrame                 # Source 2 rows then Source 3 rows; column `source` in {2, 3}
    gt: dict = None                  # s1 entity_id -> set of target ids (None when unlabelled)
    name: str = ''
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def has_truth(self):
        return self.gt is not None

    @property
    def s1_ids(self):
        return self.s1['entity_id'].to_numpy()

    @property
    def tg_ids(self):
        return self.tg['entity_id'].to_numpy()

    def positive_pairs(self):
        """(s1_idx, tg_idx) int32 arrays of every true pair whose ids exist here."""
        if 'pos' not in self._cache:
            if not self.has_truth:
                raise ValueError(f'{self.name}: no ground truth')
            s1_pos = pd.Index(self.s1_ids)
            a, b = [], []
            for s1_id, targets in self.gt.items():
                for t in targets:
                    a.append(s1_id)
                    b.append(t)
            i = s1_pos.get_indexer(pd.Index(a)) if a else np.empty(0, np.int64)
            j = pd.Index(self.tg_ids).get_indexer(pd.Index(b)) if b else np.empty(0, np.int64)
            ok = (i >= 0) & (j >= 0)
            self._cache['pos'] = (i[ok].astype(np.int32), j[ok].astype(np.int32))
        return self._cache['pos']

    def owner(self):
        """For each target row, the Source 1 row it matches (-1 = distractor)."""
        if 'owner' not in self._cache:
            own = np.full(len(self.tg), -1, np.int32)
            if self.has_truth:
                i, j = self.positive_pairs()
                own[j] = i
            self._cache['owner'] = own
        return self._cache['owner']

    def n_true(self):
        """Number of true matches per Source 1 row."""
        out = np.zeros(len(self.s1), np.int32)
        if self.has_truth:
            i, _ = self.positive_pairs()
            np.add.at(out, i, 1)
        return out

    def summary(self):
        s = f'{self.name}: S1 {len(self.s1):,}, targets {len(self.tg):,}'
        if self.has_truth:
            s += f', true pairs {len(self.positive_pairs()[0]):,}'
        return s


def find_files(directory):
    def one(pattern, required=True):
        hits = sorted(glob.glob(os.path.join(directory, pattern)))
        if not hits:
            if required:
                raise FileNotFoundError(f'no {pattern} in {directory}')
            return None
        if len(hits) > 1:
            raise ValueError(f'several {pattern} files in {directory}: {hits}')
        return hits[0]
    return {'s1': one('*_source1.tsv'), 's2': one('*_source2.tsv'), 's3': one('*_source3.tsv'),
            'gt': one('*_ground_truth.tsv', required=False)}


def read_ground_truth_file(path):
    gt = {}
    with open(path, encoding='utf-8') as f:
        header = next(f).rstrip('\n').split('\t')
        if header != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError(f'{path}: unexpected header {header}')
        for line in f:
            s1_id, _, ids = line.rstrip('\n').partition('\t')
            if s1_id:
                gt[s1_id] = {x.strip() for x in ids.split(',') if x.strip()}
    return gt


def load_dir(directory, name=None, with_truth=True):
    files = find_files(directory)
    s1 = read_tsv(files['s1'])
    parts = []
    for src in (2, 3):
        df = read_tsv(files[f's{src}'])
        df['source'] = np.int8(src)
        parts.append(df)
    tg = pd.concat(parts, ignore_index=True)
    for df in (s1, tg):
        missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f'{directory}: missing columns {missing}')
    gt = read_ground_truth_file(files['gt']) if with_truth and files['gt'] else None
    if gt is not None:
        for s1_id in s1['entity_id']:
            gt.setdefault(s1_id, set())
    return ERData(s1=s1, tg=tg, gt=gt, name=name or os.path.basename(os.path.normpath(directory)))


def restrict(data, s1_keep, tg_keep, name):
    """Sub-universe with the selected rows (order preserved)."""
    s1 = data.s1[s1_keep].reset_index(drop=True)
    tg = data.tg[tg_keep].reset_index(drop=True)
    gt = None
    if data.has_truth:
        ids = set(tg['entity_id'])
        gt = {s: {t for t in data.gt.get(s, ()) if t in ids} for s in s1['entity_id']}
    return ERData(s1=s1, tg=tg, gt=gt, name=name)


def _target_mask(data, s1_keep, unmatched_keep):
    own = data.owner()
    return np.where(own >= 0, s1_keep[np.maximum(own, 0)], unmatched_keep)


def split_holdout(data, val_fraction, seed):
    """(train, val) closed universes; with the default fraction/seed this is
    exactly splits/local_train and splits/local_val from data_loader.py."""
    s1_val = np.fromiter((in_holdout(e, val_fraction, seed) for e in data.s1_ids), bool, len(data.s1))
    un_val = np.fromiter((in_holdout(e, val_fraction, seed) for e in data.tg_ids), bool, len(data.tg))
    tg_val = _target_mask(data, s1_val, un_val)
    return (restrict(data, ~s1_val, ~tg_val, f'{data.name}-train'),
            restrict(data, s1_val, tg_val, f'{data.name}-val'))


def subsample(data, fraction, seed, name=None):
    """Random entity-level subset: `fraction` of Source 1 with all their targets,
    plus the same fraction of the unmatched targets (keeps the distractor ratio)."""
    if fraction >= 1.0:
        return data
    if not 0 < fraction < 1:
        raise ValueError(f'fraction must be in (0, 1], got {fraction}')
    rng = np.random.default_rng(seed)
    s1_keep = rng.random(len(data.s1)) < fraction
    un_keep = rng.random(len(data.tg)) < fraction
    tg_keep = _target_mask(data, s1_keep, un_keep) if data.has_truth else un_keep
    return restrict(data, s1_keep, tg_keep, name or f'{data.name}-sub{fraction:g}')
