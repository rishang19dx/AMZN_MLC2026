"""
Score candidate pairs with a trained pair energy model.

Input : a candidate_pairs.tsv (submission format: source1_entity_id, comma-separated
        candidate ids) and the preprocessed cache of the same split.
Output: TSV  source1_entity_id  candidate_entity_id  p_match   (p = sigmoid(-energy))

This is the integration point with the main pipeline: p_match can be joined
onto its candidates as a stage-2 feature, exactly like the cross-encoder score.

Usage:
  python -m ebm.score --model artifacts/ebm_v2/best.pt --cache cache/test \
      --candidates output/test/candidate_pairs.tsv --out artifacts/ebm_v2/test_scores.tsv
"""

import argparse
import os
import time

import numpy as np
import torch

from ebm.model import PairEnergyModel
from ebm.train import Data, autocast


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:8.1f}s] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--chunk', type=int, default=2_000_000, help='pairs held in memory at a time')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = torch.device(args.device)
    ck = torch.load(args.model, map_location=device, weights_only=False)
    a, m = ck['args'], ck['meta']
    model = PairEnergyModel(m['hash_buckets'], m['n_fields'], d=a['d'], layers=a['layers']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    log(f"model from step {ck['validation'].get('step')}, validation macro F0.5 {ck['validation']['macro_f05']:.4f}")

    ids = np.load(os.path.join(args.cache, 'ids.npy'))
    tokens = np.load(os.path.join(args.cache, 'tokens.npy'), mmap_mode='r')
    fields = np.load(os.path.join(args.cache, 'fields.npy'), mmap_mode='r')
    row = {e: i for i, e in enumerate(ids)}

    def batch(rows):
        return (torch.from_numpy(np.asarray(tokens[rows])).to(device), torch.from_numpy(np.asarray(fields[rows])).to(device))

    def flush(pa, pb, out):
        pa, pb = np.array(pa), np.array(pb)
        probs = []
        with torch.no_grad():
            for s in range(0, len(pa), args.batch):
                ta, fa = batch(pa[s:s + args.batch]); tb, fb = batch(pb[s:s + args.batch])
                with autocast(device):
                    probs.append(torch.sigmoid(-model(ta, fa, tb, fb)).float().cpu().numpy())
        p = np.concatenate(probs)
        out.writelines(f'{ids[x]}\t{ids[y]}\t{q:.6f}\n' for x, y, q in zip(pa, pb, p))
        return len(pa)

    n = 0
    with open(args.candidates, encoding='utf-8') as f, open(args.out, 'w', encoding='utf-8') as out:
        next(f)
        out.write('source1_entity_id\tcandidate_entity_id\tp_match\n')
        pa, pb = [], []
        for line in f:
            s1, _, cands = line.rstrip('\n').partition('\t')
            for c in cands.split(','):
                if c:
                    pa.append(row[s1]); pb.append(row[c])
            if len(pa) >= args.chunk:
                n += flush(pa, pb, out); pa, pb = [], []
                log(f'scored {n:,} pairs')
        if pa:
            n += flush(pa, pb, out)
    log(f'scored {n:,} pairs -> {args.out}')


if __name__ == '__main__':
    main()
