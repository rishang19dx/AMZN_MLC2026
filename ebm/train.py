"""
Train the pair energy model (v2) on a preprocessed cache (ebm.preprocess).

The model is a RANKER: it scores candidates produced by ebm.block, never
searches. Per step, for B anchor Source 1 records (train split, >= 1 match,
blocked):
  positive   one of its true targets (random)
  negatives  K from the anchor's OWN candidate pool (its blocked non-matches:
             exactly what it must reject at test time; the whole pool is
             used across steps), plus a --mined-share of same-country
             look-alikes from the inverted index. Every target has at most
             one Source 1, so "owner != anchor" rules out false negatives.
Loss:
  listwise   cross-entropy of -E over [positive, K hard negatives]
  BCE        on the same pairs (keeps sigmoid(-E) calibrated)
  in-batch   InfoNCE of the bi-encoder (anchor vs all positives in the batch)

Validation (end to end): up to --val-anchors held-out Source 1 (incl.
singletons) with their FULL candidate lists; true pairs that blocking missed
count as misses. Decoding = one Source 1 per target (among the validation
anchors) + per-source caps + a global threshold, chosen here and reused by
ebm.predict.

Usage:
  python -m ebm.train --cache cache/train --out artifacts/ebm_v2
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ebm.model import PairEnergyModel


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:8.1f}s] {msg}', flush=True)


class Data:
    def __init__(self, cache, mmap, max_df):
        load = lambda k: np.load(os.path.join(cache, f'{k}.npy'), mmap_mode='r' if mmap and k in ('tokens', 'fields') else None)
        with open(os.path.join(cache, 'meta.json')) as f:
            self.meta = json.load(f)
        self.tokens, self.fields = load('tokens'), load('fields')
        self.country, self.val = load('country'), load('val')
        self.source = load('source')
        self.indptr, self.targets, self.owner = load('gt_indptr'), load('gt_targets'), load('owner')
        self.n, self.n_s1 = self.meta['n'], self.meta['n_s1']
        counts = np.diff(self.indptr)
        s1 = np.arange(self.n_s1)
        self.train_anchors = s1[(~self.val) & (counts > 0)]
        self.val_anchors = s1[self.val]
        self._build_index(max_df)
        self._load_candidates(cache)

    def _load_candidates(self, cache):
        z = np.load(os.path.join(cache, 'candidates.npz'))
        order = np.lexsort((z['tg'], z['s1']))
        self.c_s1, self.c_tg = z['s1'][order], z['tg'][order]
        self.c_start = np.searchsorted(self.c_s1, np.arange(self.n_s1 + 1))
        blocked = np.diff(self.c_start) > 0
        self.train_anchors = self.train_anchors[blocked[self.train_anchors]]
        self.val_anchors = self.val_anchors[blocked[self.val_anchors]]
        log(f'candidates: {len(self.c_s1):,} pairs; {len(self.train_anchors):,} blocked train anchors, '
            f'{len(self.val_anchors):,} blocked validation Source 1')

    def candidate_negatives(self, anchors, k, mined_share, rng):
        n_pool = k - int(round(k * mined_share))
        out = np.empty((len(anchors), k), np.int64)
        mined = self.hard_negatives(anchors, k, rng)
        for i, a in enumerate(anchors):
            pool = self.c_tg[self.c_start[a]:self.c_start[a + 1]]
            pool = pool[self.owner[pool] != a]
            take = pool[rng.permutation(len(pool))[:n_pool]] if len(pool) else pool
            rest = [x for x in mined[i] if x not in set(take)][:k - len(take)]
            row = list(take) + rest
            while len(row) < k:
                row.append(mined[i][len(row) % k])
            out[i] = row[:k]
        return out

    def _keys(self, rows):
        """Mining keys: (country, token) for name-word and house-number tokens."""
        from ebm.normalize import F_NAME, F_HOUSE
        t = np.asarray(self.tokens[rows]).astype(np.int64)
        f = np.asarray(self.fields[rows])
        keep = (f == F_NAME) | (f == F_HOUSE)
        return np.where(keep, self.country[rows].astype(np.int64)[:, None] * (1 << 21) + t, -1)

    def _build_index(self, max_df):
        tg = np.arange(self.n_s1, self.n)
        keys, rows = [], []
        for s in range(0, len(tg), 2_000_000):                      # chunked: bounded memory
            k = self._keys(tg[s:s + 2_000_000])
            r = np.broadcast_to(tg[s:s + 2_000_000, None], k.shape)
            ok = k >= 0
            keys.append(k[ok]); rows.append(r[ok])
        keys, rows = np.concatenate(keys), np.concatenate(rows).astype(np.int32)
        order = np.argsort(keys, kind='stable')
        keys, rows = keys[order], rows[order]
        uk, start, cnt = np.unique(keys, return_index=True, return_counts=True)
        good = (cnt >= 2) & (cnt <= max_df)                          # rare-ish: informative look-alikes
        self.ukeys, self.kstart, self.kcount, self.postings = uk[good], start[good], cnt[good], rows
        # per-country target lists for the random fallback
        c = self.country[tg]
        self.cty_order = tg[np.argsort(c, kind='stable')]
        self.cty_start = np.searchsorted(np.sort(c), np.arange(len(self.meta['countries']) + 1))
        log(f'mining index: {len(self.ukeys):,} keys (df 2..{max_df}), {int(self.kcount.sum()):,} postings')

    def positives(self, anchors, rng):
        lo, hi = self.indptr[anchors], self.indptr[anchors + 1]
        return self.targets[lo + (rng.random(len(anchors)) * (hi - lo)).astype(np.int64)]

    def hard_negatives(self, anchors, k, rng):
        keys = self._keys(anchors)                                   # [B, L]
        pos = np.searchsorted(self.ukeys, keys)
        pos = np.clip(pos, 0, len(self.ukeys) - 1)
        valid = (keys >= 0) & (self.ukeys[pos] == keys)
        out = np.empty((len(anchors), k), np.int64)
        for i in range(len(anchors)):
            ks = pos[i][valid[i]]
            got = []
            for _ in range(4 * k):
                if len(got) == k or len(ks) == 0:
                    break
                j = ks[rng.integers(len(ks))]
                cand = self.postings[self.kstart[j] + rng.integers(self.kcount[j])]
                if self.owner[cand] != anchors[i] and cand not in got:
                    got.append(cand)
            while len(got) < k:                                      # random same-country fallback
                c = self.country[anchors[i]]
                lo, hi = self.cty_start[c], self.cty_start[c + 1]
                cand = self.cty_order[lo + rng.integers(max(hi - lo, 1))] if hi > lo else self.n_s1
                if self.owner[cand] != anchors[i]:
                    got.append(cand)
            out[i] = got
        return out

    def batch(self, rows, device):
        t = torch.from_numpy(np.asarray(self.tokens[rows])).to(device, non_blocking=True)
        f = torch.from_numpy(np.asarray(self.fields[rows])).to(device, non_blocking=True)
        return t, f


CAPS = {2: 5, 3: 6}          # max matches per Source 1 from S2 / S3 (train maxima)


def decode(s1, tg, p, source, thr):
    """One Source 1 per target (highest p), per-source caps, threshold -> kept mask."""
    keep = np.zeros(len(p), bool)
    order = np.lexsort((-p, tg))                          # per target, best first
    first = np.r_[True, tg[order][1:] != tg[order][:-1]]
    best = order[first]
    best = best[p[best] >= thr]
    keep[best] = True
    # caps: per (s1, source) keep the highest-p ones
    idx = np.flatnonzero(keep)
    o = idx[np.lexsort((-p[idx], source[tg[idx]], s1[idx]))]
    grp = s1[o].astype(np.int64) * 4 + source[tg[o]]
    rank = np.arange(len(o)) - np.searchsorted(grp, grp)   # grp is sorted within o
    cap = np.where(source[tg[o]] == 2, CAPS[2], CAPS[3])
    keep[o[rank >= cap]] = False
    return keep


def f05_per_entity(s1_rows, kept_s1, kept_tp, truth_count):
    """Macro F0.5 over s1_rows (singletons included), given kept pairs and their correctness."""
    n = len(s1_rows)
    pos = {r: i for i, r in enumerate(s1_rows)}
    g = np.array([pos[x] for x in kept_s1], np.int64) if len(kept_s1) else np.array([], np.int64)
    tp = np.bincount(g, weights=kept_tp, minlength=n)
    pp = np.bincount(g, minlength=n)
    ap = truth_count
    f = np.where(ap == 0, (pp == 0).astype(float), 0.0)
    ok = (ap > 0) & (tp > 0)
    pr, rc = tp[ok] / pp[ok], tp[ok] / ap[ok]
    f[ok] = 1.25 * pr * rc / (0.25 * pr + rc)
    return f.mean()


@torch.no_grad()
def validate(model, data, device, n_anchors, bs, seed):
    rng = np.random.default_rng(seed)
    anchors = data.val_anchors if not n_anchors or n_anchors >= len(data.val_anchors) else \
        np.sort(rng.choice(data.val_anchors, n_anchors, replace=False))
    lo, hi = data.c_start[anchors], data.c_start[anchors + 1]
    sel = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
    a_rows, b_rows = data.c_s1[sel], data.c_tg[sel]
    labels = (data.owner[b_rows] == a_rows)
    truth = np.diff(data.indptr)[anchors]                  # incl. pairs blocking missed
    model.eval()
    out = []
    for s in range(0, len(a_rows), bs):
        ta, fa = data.batch(a_rows[s:s + bs], device)
        tb, fb = data.batch(b_rows[s:s + bs], device)
        with autocast(device):
            out.append(torch.sigmoid(-model(ta, fa, tb, fb)).float().cpu().numpy())
    p = np.concatenate(out)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, p) if 0 < labels.mean() < 1 else float('nan')
    res = []
    for t in np.linspace(0.05, 0.95, 19):
        k = decode(a_rows, b_rows, p, data.source, t)
        res.append((f05_per_entity(anchors, a_rows[k], labels[k].astype(float), truth), t))
    best = max(res)
    ceiling = f05_per_entity(anchors, a_rows[labels], np.ones(int(labels.sum())), truth)
    model.train()
    return {'macro_f05': best[0], 'threshold': float(best[1]), 'auc': auc, 'ceiling': ceiling,
            'blocking_recall': float(labels.sum() / max(truth.sum(), 1)),
            'anchors': len(anchors), 'pairs': len(labels), 'pos_share': float(labels.mean())}


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else torch.autocast('cpu', enabled=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=512, help='anchors per step')
    ap.add_argument('--hard', type=int, default=15, help='negatives per anchor per step (from its candidate pool)')
    ap.add_argument('--mined-share', type=float, default=0.25, help='share of those from mined look-alikes')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--max-df', type=int, default=5000, help='mining index: max targets per key')
    ap.add_argument('--inbatch-weight', type=float, default=0.5)
    ap.add_argument('--val-anchors', type=int, default=50000, help='0 = all validation Source 1')
    ap.add_argument('--val-every', type=int, default=1000)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke tests')
    ap.add_argument('--mmap', action='store_true', help='memory-map tokens (low-RAM machines)')
    ap.add_argument('--seed', type=int, default=26)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = Data(args.cache, args.mmap, args.max_df)
    log(f'{len(data.train_anchors):,} train anchors, {len(data.val_anchors):,} validation Source 1, device {device}')

    m = data.meta
    model = PairEnergyModel(m['hash_buckets'], m['n_fields'], d=args.d, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'PairEnergyModel v2: {n_params / 1e6:.1f}M parameters')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(data.train_anchors) // args.batch
    total = args.max_steps or args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.05)
    rng = np.random.default_rng(args.seed)
    best, step = -1.0, 0
    hist = []
    model.train()
    for epoch in range(args.epochs):
        order = rng.permutation(data.train_anchors)
        for s in range(0, steps_per_epoch * args.batch, args.batch):
            A = order[s:s + args.batch]
            P = data.positives(A, rng)
            N = data.candidate_negatives(A, args.hard, args.mined_share, rng)   # [B, K]
            ta, fa = data.batch(A, device)
            tp, fp = data.batch(P, device)
            tn, fn = data.batch(N.reshape(-1), device)
            with autocast(device):
                ea, ha, ma = model.encode(ta, fa)
                ep, hp, mp = model.encode(tp, fp)
                en, hn, mn = model.encode(tn, fn)
                K = args.hard
                rep = lambda x: x.repeat_interleave(K, 0)
                e_pos = model.energy_from(ea, ha, ma, fa, ep, hp, mp, fp)                       # [B]
                e_neg = model.energy_from(rep(ea), rep(ha), rep(ma), rep(fa), en, hn, mn, fn).view(-1, K)
                logits = torch.cat([-e_pos[:, None], -e_neg], 1).float()                       # [B, 1+K]
                listwise = F.cross_entropy(logits, torch.zeros(len(A), dtype=torch.long, device=device))
                bce = F.binary_cross_entropy_with_logits(logits, torch.cat(
                    [torch.ones(len(A), 1, device=device), torch.zeros(len(A), K, device=device)], 1))
                za, zp = F.normalize(ea.float(), dim=-1), F.normalize(ep.float(), dim=-1)
                sims = za @ zp.T * model.logit_scale.exp().clamp(max=100)
                inbatch = F.cross_entropy(sims, torch.arange(len(A), device=device))
                loss = listwise + bce + args.inbatch_weight * inbatch
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                log(f'epoch {epoch} step {step}/{total} loss {loss.item():.4f} (list {listwise.item():.3f} '
                    f'bce {bce.item():.3f} inbatch {inbatch.item():.3f}) lr {sched.get_last_lr()[0]:.2e}')
            if step % args.val_every == 0 or step == total:
                v = validate(model, data, device, args.val_anchors, 4096, args.seed)
                v.update(step=step, epoch=epoch); hist.append(v)
                log(f'VALIDATION step {step}: macro F0.5 {v["macro_f05"]:.4f} @ {v["threshold"]:.2f} '
                    f'(ceiling {v["ceiling"]:.4f}, blocking recall {v["blocking_recall"]:.4f})  AUC {v["auc"]:.4f}  '
                    f'{v["anchors"]:,} Source 1, {v["pairs"]:,} pairs')
                if v['macro_f05'] > best:
                    best = v['macro_f05']
                    torch.save({'model': model.state_dict(), 'args': vars(args), 'meta': m, 'validation': v},
                               os.path.join(args.out, 'best.pt'))
                with open(os.path.join(args.out, 'history.json'), 'w') as f:
                    json.dump(hist, f, indent=2)
            if step >= total:
                break
        if step >= total:
            break
    log(f'done: best validation macro F0.5 {best:.4f} -> {args.out}/best.pt')


if __name__ == '__main__':
    main()
