"""
Error analysis on out-of-fold matcher predictions (task T7).

Splits the F0.5 loss on a labelled split into buckets, so the next change goes
where the loss is. Needs `match.py --split <split> --cv` to have run (it saves
out-of-fold stage-2 probabilities to $BER_CACHE_DIR/<split>/oof_p2.npy, in
feature-part row order, and the decoding choice to models/decode.json).

  missed pairs     never retrieved by blocking | retrieved but lost the target
                   to another S1 | removed by a per-source cap | below the
                   decoding cut (by p2 band)
  false matches    on singletons | on matched entities; exact-name twins;
                   by p2 band
  per entity       F0.5 loss by country, true match count, native script
Plus a few examples per bucket.

Usage:
  python src/error_analysis.py --split local_val
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from evaluate import read_ground_truth
from match import MODEL_DIR, Pairs, assign_and_cap, decode, group_stats, split_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='local_val')
    ap.add_argument('--examples', type=int, default=8)
    args = ap.parse_args()

    pairs = Pairs(args.split)
    p2 = np.load(os.path.join(config.CACHE_DIR, args.split, 'oof_p2.npy'))
    assert len(p2) == len(pairs), 'oof_p2.npy does not match the feature parts; rerun match.py --cv'
    with open(os.path.join(MODEL_DIR, 'decode.json')) as f:
        choice = json.load(f)
    s1c, tc, src = pairs.s1_code, pairs.t_code, pairs.src
    chosen = decode(s1c, tc, src, p2, choice['method'], choice['param'])
    t_first = assign_and_cap(s1c, tc, src, p2)            # eligible after assignment and caps
    lost_target = group_stats(tc, p2)[0] > 1               # another S1 scored this target higher

    from data_loader import read_tsv
    p = config.split_paths(args.split)
    s1 = read_tsv(p['s1'])
    tg = pd.concat([read_tsv(p['s2']), read_tsv(p['s3'])], ignore_index=True)
    s1_ids, tg_ids = split_ids(args.split)
    gt = read_ground_truth(p['gt'])
    n_true = np.array([len(gt[s]) for s in s1_ids])
    country = s1['country'].to_numpy()
    from normalize import is_non_latin
    tg_nonlatin = tg['business_name'].map(is_non_latin).to_numpy()

    y = pairs.label.astype(bool)
    feats = {f: pairs.X[:, pairs.feats.index(f)] for f in ('name_exact', 'name_tset', 'addr_tset')}
    band = pd.cut(p2, [-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01], labels=['<.1', '.1-.3', '.3-.5', '.5-.7', '.7-.9', '>.9'])

    # ---- missed pairs
    n_gt_pairs = int(n_true.sum())
    retrieved_true = int(y.sum())
    fn = y & ~chosen
    rows = [('never retrieved by blocking', n_gt_pairs - retrieved_true)]
    rows.append(('retrieved, target went to another S1', int((fn & lost_target).sum())))
    rows.append(('retrieved, removed by per-source cap', int((fn & ~lost_target & ~t_first).sum())))
    below = fn & t_first
    rows.append(('retrieved, eligible, below decoding cut', int(below.sum())))
    miss = pd.DataFrame(rows, columns=['missed pairs', 'n'])
    miss['share'] = miss['n'] / miss['n'].sum()
    print(f'\n== missed pairs ({int(miss.n.sum()):,} of {n_gt_pairs:,} true pairs)\n' + miss.to_string(index=False, float_format='%.3f'))
    print('\n   eligible-but-below-cut misses by p2 band:\n' +
          pd.Series(band[below]).value_counts().sort_index().to_string())
    print('   ... of which target name is non-Latin: '
          f'{tg_nonlatin[tc[below]].mean():.1%} (vs {tg_nonlatin[tc[y]].mean():.1%} of all true retrieved)')

    # ---- false matches
    fp = chosen & ~y
    single = n_true[s1c] == 0
    fps = pd.DataFrame({
        'bucket': np.where(single[fp], 'on singleton S1', 'on matched S1'),
        'exact name twin': feats['name_exact'][fp] == 1,
        'p2 band': band[fp],
    })
    print(f'\n== false matches ({int(fp.sum()):,} pairs)\n' + fps.groupby('bucket').agg(
        n=('bucket', 'size'), exact_name_twin=('exact name twin', 'mean')).to_string(float_format='%.3f'))
    print('   by p2 band:\n' + fps['p2 band'].value_counts().sort_index().to_string())

    # ---- per-entity loss
    n = len(s1_ids)
    tp = np.bincount(s1c[chosen], weights=y[chosen], minlength=n)
    npred = np.bincount(s1c[chosen], minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(n_true == 0, (npred == 0).astype(float), 1.25 * tp / (1.25 * tp + 0.25 * (n_true - tp) + (npred - tp)))
    f = np.nan_to_num(f)
    has_native = np.zeros(n, bool)
    np.logical_or.at(has_native, s1c[y], tg_nonlatin[tc[y]])
    ent = pd.DataFrame({'country': country, 'n_true': np.minimum(n_true, 7), 'native': has_native, 'loss': 1 - f})
    total = ent['loss'].sum()
    for col in ('country', 'n_true', 'native'):
        g = ent.groupby(col)['loss'].agg(['size', 'mean', 'sum'])
        g['share_of_loss'] = g['sum'] / total
        print(f'\n== F0.5 loss by {col} (mean loss per entity; share of total loss)\n' + g.drop(columns='sum').to_string(float_format='%.4f'))
    print(f'\noverall F0.5 {f.mean():.5f}')

    # ---- examples
    def show(title, mask):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return
        idx = np.random.default_rng(0).choice(idx, min(args.examples, len(idx)), replace=False)
        print(f'\n-- {title}')
        for k in idx:
            a, b = s1.iloc[s1c[k]], tg.iloc[tc[k]]
            print(f'  p2={p2[k]:.3f}  S1: {a.business_name} | {a.business_address}\n'
                  f'            T : {b.business_name} | {b.business_address}')

    show('missed: eligible but below cut, p2 < 0.1', below & (p2 < 0.1))
    show('missed: target went to another S1', fn & lost_target)
    show('false match on a singleton', fp & single)
    show('false match on a matched entity', fp & ~single)


if __name__ == '__main__':
    main()
