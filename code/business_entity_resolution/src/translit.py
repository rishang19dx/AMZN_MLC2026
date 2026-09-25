"""
Learned native-script -> Latin word dictionary (docs/FINDINGS.md §3.1).

18% of India target names are written in Indic scripts (Kannada, Malayalam,
Devanagari, ...), while Source 1 is always Latin. Rule-based transliteration
(Unidecode, anyascii) spells them differently from Source 1 ("mai
propprttis" vs "my properties"). The vocabulary is small and closed, so we
learn the mapping from labelled pairs instead:

  for each true pair (Source 1 text, target text) whose target text is in an
  Indic script and has the same number of words as the Source 1 text, align
  words by position and count (native word -> Source 1 word). Keep, for each
  native word, its most frequent Latin word if it has >= --min-count support
  and >= --min-share of that word's alignments.

Learned from ground truth of a TRAINING split only (default local_train), so
local_val scores stay honest. For the final test run, build it from `train`.
Names and addresses are both used (addresses only when word counts match).

Output: $BER_CACHE_DIR/translit.json, read by normalize.norm().

Usage:
  python src/translit.py --split local_train
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from normalize import PUNCT, has_indic, norm_latin

DICT_PATH = os.path.join(config.CACHE_DIR, 'translit.json')


def native_words(text):
    return [w for w in (t.strip(PUNCT) for t in text.split()) if w]


def read_rows(path, keep=None):
    """entity_id -> (name, address), optionally only for ids where keep(name, addr)."""
    out = {}
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            eid, name, addr, _ = line.rstrip('\n').split('\t')
            if keep is None or keep(name, addr):
                out[eid] = (name, addr)
    return out


def build(split, min_count, min_share):
    t0 = time.time()
    p = config.split_paths(split)
    s1 = read_rows(p['s1'])
    native = {}
    for key in ('s2', 's3'):
        native.update(read_rows(p[key], keep=lambda n, a: has_indic(n) or has_indic(a)))
    print(f'{len(s1):,} Source 1 records, {len(native):,} targets with Indic script ({time.time() - t0:.0f}s)')

    counts = defaultdict(Counter)
    aligned = {'name': 0, 'addr': 0}
    with open(p['gt'], encoding='utf-8') as f:
        next(f)
        for line in f:
            s1_id, _, matched = line.rstrip('\n').partition('\t')
            for t in matched.split(','):
                if t not in native:
                    continue
                for field, (src, tgt) in (('name', (s1[s1_id][0], native[t][0])),
                                          ('addr', (s1[s1_id][1], native[t][1]))):
                    if not has_indic(tgt):
                        continue
                    lat, nat = norm_latin(src).split(), native_words(tgt)
                    if len(lat) != len(nat):
                        continue
                    aligned[field] += 1
                    for n, l in zip(nat, lat):
                        if has_indic(n):
                            counts[n][l] += 1

    table = {}
    for n, c in counts.items():
        (best, k), total = c.most_common(1)[0], sum(c.values())
        if k >= min_count and k / total >= min_share:
            table[n] = best
    print(f'aligned pairs: {aligned}; native words seen {len(counts):,}, kept {len(table):,} '
          f'(min_count={min_count}, min_share={min_share})')
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='local_train', choices=['local_train', 'train'])
    ap.add_argument('--min-count', type=int, default=3)
    ap.add_argument('--min-share', type=float, default=0.5)
    args = ap.parse_args()
    table = build(args.split, args.min_count, args.min_share)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    with open(DICT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'built_from': args.split, 'words': table}, f, ensure_ascii=False, indent=0)
    print(f'wrote {DICT_PATH}')


if __name__ == '__main__':
    main()
