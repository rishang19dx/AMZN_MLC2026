"""
Score every blocked test pair with the trained ranker, decode, write the submission.

  <cache>/candidates.npz  (ebm.block)  ->  p = sigmoid(-energy) per pair
  decode: one Source 1 per target, S2 <= 5 / S3 <= 6 per Source 1, threshold
          chosen on validation (stored in the checkpoint)
  ->  <out>/matching_results.tsv   (every Source 1 row; empty = no match)
      <out>/candidate_pairs.tsv    (copied from the blocking output)
      <out>/scores.npz             (s1, tg, p) for reuse, e.g. as a stage-2 feature

Usage:
  python -m ebm.predict --model artifacts/best.pt --cache cache/test --out artifacts/test
"""

import argparse
import os
import shutil
import time

import numpy as np
import torch

from ebm.model import PairEnergyModel
from ebm.train import autocast, decode


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:8.1f}s] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threshold', type=float, default=None, help='override the validation threshold')
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    a, m, v = ck['args'], ck['meta'], ck['validation']
    model = PairEnergyModel(m['hash_buckets'], m['n_fields'], d=a['d'], layers=a['layers']).to(device).eval()
    model.load_state_dict(ck['model'])
    thr = v['threshold'] if args.threshold is None else args.threshold
    log(f"model: step {v.get('step')}, validation macro F0.5 {v['macro_f05']:.4f} at threshold {v['threshold']:.2f}; using {thr:.2f}")

    c = args.cache
    tokens = np.load(os.path.join(c, 'tokens.npy'), mmap_mode='r')
    fields = np.load(os.path.join(c, 'fields.npy'), mmap_mode='r')
    source = np.load(os.path.join(c, 'source.npy'))
    ids = np.load(os.path.join(c, 'ids.npy'))
    z = np.load(os.path.join(c, 'candidates.npz'))
    s1, tg = z['s1'], z['tg']
    n_s1 = int((source == 1).sum())
    log(f'{len(s1):,} candidate pairs for {n_s1:,} Source 1')

    batch = lambda rows: (torch.from_numpy(np.asarray(tokens[rows])).to(device), torch.from_numpy(np.asarray(fields[rows])).to(device))
    p = np.empty(len(s1), np.float32)
    with torch.no_grad():
        for s in range(0, len(s1), args.batch):
            ta, fa = batch(s1[s:s + args.batch]); tb, fb = batch(tg[s:s + args.batch])
            with autocast(device):
                p[s:s + args.batch] = torch.sigmoid(-model(ta, fa, tb, fb)).float().cpu().numpy()
            if (s // args.batch) % 1000 == 0:
                log(f'scored {s + len(ta):,} / {len(s1):,}')
    np.savez(os.path.join(args.out, 'scores.npz'), s1=s1, tg=tg, p=p)

    keep = decode(s1, tg, p, source, thr)
    ks, kt = s1[keep], tg[keep]
    order = np.lexsort((kt, ks))
    ks, kt = ks[order], kt[order]
    starts = np.searchsorted(ks, np.arange(n_s1 + 1))
    with open(os.path.join(args.out, 'matching_results.tsv'), 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for i in range(n_s1):
            f.write(f"{ids[i]}\t{','.join(ids[kt[starts[i]:starts[i + 1]]])}\n")
    shutil.copy(os.path.join(c, 'candidate_pairs.tsv'), os.path.join(args.out, 'candidate_pairs.tsv'))
    empty = int((np.diff(starts) == 0).sum())
    log(f'{int(keep.sum()):,} matches, {empty:,} Source 1 with none ({empty / n_s1:.1%}) -> {args.out}/matching_results.tsv')


if __name__ == '__main__':
    main()
