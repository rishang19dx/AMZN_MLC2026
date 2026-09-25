"""Checks the matcher's decoding logic on hand-made cases.
Run with:  python tests/test_match.py   (or pytest)"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from match import assign_and_cap, decode_expected_f, group_stats
from normalize import norm


def frame(rows):
    """(s1_code, t_code, src, p) arrays, as the matcher holds them."""
    df = pd.DataFrame(rows, columns=['s1_id', 'cand_id', 'p'])
    return (pd.factorize(df['s1_id'])[0], pd.factorize(df['cand_id'])[0],
            df['cand_id'].str.startswith('S3-').to_numpy().astype(np.int8), df['p'].to_numpy(np.float32))


def test_group_stats():
    code = np.array([0, 0, 1, 1, 1])
    p = np.array([.2, .9, .5, .7, .1], np.float32)
    rank, top1, top2, gsum, n50 = group_stats(code, p)
    assert rank.tolist() == [2, 1, 2, 1, 3]
    assert np.allclose(top1, [.9, .9, .7, .7, .7]) and np.allclose(top2, [.2, .2, .5, .5, .5])
    assert np.allclose(gsum, [1.1, 1.1, 1.3, 1.3, 1.3]) and n50.tolist() == [1, 1, 1, 1, 1]


def test_target_goes_to_best_source1_only():
    assert assign_and_cap(*frame([('A', 'S2-1', .6), ('B', 'S2-1', .9), ('A', 'S2-2', .8)])).tolist() \
        == [False, True, True]


def test_per_source_caps():
    rows = [('A', f'S2-{k}', .99) for k in range(7)] + [('A', f'S3-{k}', .99) for k in range(7)]
    keep = assign_and_cap(*frame(rows))
    assert keep[:7].sum() == 5 and keep[7:].sum() == 6


def test_expected_f_decoding():
    s1_code, _, _, p = frame([
        ('sure2', 'S2-1', .99), ('sure2', 'S2-2', .98),      # both clearly matches -> k=2
        ('single', 'S2-3', .05),                              # likely singleton -> k=0
        ('one', 'S2-4', .95), ('one', 'S2-5', .30),           # second is below the F/1.25 bar -> k=1
    ])
    chosen = decode_expected_f(s1_code, p, np.ones(len(p), bool), samples=4096)
    assert chosen.tolist() == [True, True, False, True, False]


def test_norm_is_country_agnostic():
    assert norm('Café & Co., SARL') == 'cafe and co sarl'
    assert norm('  ') == '' and norm(None) == ''


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'ok  {name}')
