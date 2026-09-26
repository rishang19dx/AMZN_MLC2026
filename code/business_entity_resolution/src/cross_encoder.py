"""
Cross-encoder member (docs/ENSEMBLE.md §3.2, re-budgeted for one Kaggle T4).

One microsoft/mdeberta-v3-base (MIT, multilingual: native scripts and French
are in its vocabulary) fine-tuned as a pair classifier on `name | address` of
both records. Its logit becomes a stage-2 feature (`ce`), plus its context
(rank within the Source 1 list, margin over the best competing Source 1).

Out-of-sample by construction: it is trained on the `ce_train` split (5% of
local_train Source 1 against all local_train targets, so hard negatives are
realistic), which is disjoint from local_val (stage-2 training data) and test.
So one model replaces ENSEMBLE.md's fold-0/fold-1 pair A/B at half the GPU
time, and rules 1-2 (out-of-fold scores only) still hold.

Only the uncertain band is scored: pairs with stage-1 probability p1 in
[BAND_LO, BAND_HI]. Outside the band, `ce` is NaN (LightGBM routes NaN
natively). The same band rule is used for local_val (out-of-fold p1) and test
(final stage-1 model), as the combiner requires.

Modes
  export   (laptop, CPU) text pairs -> $BER_CACHE_DIR/ce/<split>.parquet
           ce_train: band pairs + a sample of easy pairs, with labels;
           local_val / test: band pairs only.
  train    (GPU) fine-tune on ce/ce_train.parquet -> $BER_CACHE_DIR/ce/model
  score    (GPU) score ce/<split>.parquet -> $BER_CACHE_DIR/<split>/ce_scores.parquet
           (columns s1_idx, tg_idx, ce = logit)

Settings (docs/FINDINGS.md §5): fp16 autocast (never bf16: DeBERTa NaNs),
max_length 128, one epoch, random swap of the two records, lr 2e-5, linear
warmup/decay.

Loss (--loss, default listwise; docs/ACCURACY_PLAN.md): pairs are grouped by
Source 1 (all its positives + its hardest negatives by p1, up to GROUP_MAX),
and the loss is BCE + LIST_WEIGHT * listwise, where the listwise term is
-log(sum_pos e^s / sum_all e^s) over each group with a positive. A controlled
study of cross-encoder training (arXiv 2603.03010) found listwise/pairwise
objectives beat pointwise ones by about one backbone-size tier, and negative
quality mattered as much as the loss; candidates competing within a Source 1
list is also how the matcher decides. BCE is kept so singleton groups (no
positive) still train and the logit stays probability-like for stage 2.

Usage
  python src/cross_encoder.py export --split ce_train
  python src/cross_encoder.py export --split local_val
  python src/cross_encoder.py export --split test
  python src/cross_encoder.py train                 # GPU
  python src/cross_encoder.py score --split local_val   # GPU
  python src/cross_encoder.py score --split test        # GPU
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

MODEL_NAME = 'microsoft/mdeberta-v3-base'
BAND_LO, BAND_HI = 0.02, 0.98
EASY_SAMPLE = 0.10          # share of out-of-band ce_train pairs kept for training
GROUP_MAX = 16              # pairs per Source 1 group (all positives + hardest negatives)
LIST_WEIGHT = 1.0           # weight of the listwise term next to BCE
CE_DIR = os.path.join(config.CACHE_DIR, 'ce')


def log(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:7.1f}s] {msg}', flush=True)


def record_text(name, addr):
    return f'{name} | {addr}' if addr else name


# ---------------------------------------------------------------------------
# export (CPU): which pairs, and their text
# ---------------------------------------------------------------------------

def stage1_p1(split):
    """Stage-1 probabilities in feature-part row order: out-of-fold for the
    training split (saved by match.py --fit), final model otherwise."""
    from match import MODEL_DIR, predict, read_part
    from features import feature_parts
    with open(os.path.join(MODEL_DIR, 'features.json')) as f:
        fl = json.load(f)
    oof = os.path.join(config.CACHE_DIR, split, 'oof_p1.npy')
    metas, p1 = [], []
    import lightgbm as lgb
    m1 = None if os.path.exists(oof) else lgb.Booster(model_file=os.path.join(MODEL_DIR, 'stage1.txt'))
    for path in feature_parts(split):
        meta, X, _ = read_part(path, fl['stage1'])
        metas.append(meta)
        if m1 is not None:
            p1.append(predict(m1, X))
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}
    p1 = np.load(oof) if m1 is None else np.concatenate(p1)
    assert len(p1) == len(meta['s1_idx'])
    return meta, p1


def export(split):
    from data_loader import read_tsv
    meta, p1 = stage1_p1(split)
    band = (p1 >= BAND_LO) & (p1 <= BAND_HI)
    keep = band.copy()
    if split == 'ce_train':
        rng = np.random.default_rng(26)
        keep |= rng.random(len(p1)) < EASY_SAMPLE
    p = config.split_paths(split)
    s1 = read_tsv(p['s1'])
    tg = pd.concat([read_tsv(p['s2']), read_tsv(p['s3'])], ignore_index=True)
    i, j = meta['s1_idx'][keep], meta['tg_idx'][keep]
    out = pd.DataFrame({
        's1_idx': i, 'tg_idx': j, 'in_band': band[keep], 'p1': p1[keep],
        'text_a': [record_text(n, a) for n, a in zip(s1['business_name'].to_numpy()[i], s1['business_address'].to_numpy()[i])],
        'text_b': [record_text(n, a) for n, a in zip(tg['business_name'].to_numpy()[j], tg['business_address'].to_numpy()[j])],
    })
    if 'label' in meta:
        out['label'] = meta['label'][keep].astype(np.int8)
    os.makedirs(CE_DIR, exist_ok=True)
    path = os.path.join(CE_DIR, f'{split}.parquet')
    import duckdb
    duckdb.from_df(out).write_parquet(path, compression='zstd')
    msg = f'{split}: {len(p1):,} pairs, {int(band.sum()):,} in band [{BAND_LO}, {BAND_HI}], exported {len(out):,}'
    if 'label' in out:
        msg += f' ({out["label"].mean():.1%} positive)'
    log(f'{msg} -> {path}')


# ---------------------------------------------------------------------------
# train / score (GPU)
# ---------------------------------------------------------------------------

def _batches(df, tok, bs, shuffle, swap, rng):
    idx = rng.permutation(len(df)) if shuffle else np.arange(len(df))
    a, b = df['text_a'].to_numpy(), df['text_b'].to_numpy()
    y = df['label'].to_numpy(np.float32) if 'label' in df else None
    for s in range(0, len(idx), bs):
        k = idx[s:s + bs]
        ta, tb = a[k], b[k]
        if swap:  # matching is symmetric: randomly swap which record comes first
            flip = rng.random(len(k)) < 0.5
            ta, tb = np.where(flip, tb, ta), np.where(flip, ta, tb)
        enc = tok(list(ta), list(tb), truncation=True, max_length=128, padding=True, return_tensors='pt')
        yield enc, (None if y is None else y[k])


def make_groups(df, group_max=GROUP_MAX):
    """Row-index arrays, one per Source 1: all positives first, then negatives
    by descending p1 (hardest first), capped at group_max."""
    order = np.lexsort((-df['p1'].to_numpy(), -df['label'].to_numpy(), df['s1_idx'].to_numpy()))
    s1 = df['s1_idx'].to_numpy()[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s1)) + 1, len(s1)]
    return [order[a:min(b, a + group_max)] for a, b in zip(starts[:-1], starts[1:])]


def _group_batches(df, groups, tok, bs, rng):
    """Whole groups packed into batches of about `bs` pairs, random order and swap."""
    a, b = df['text_a'].to_numpy(), df['text_b'].to_numpy()
    y = df['label'].to_numpy(np.float32)
    batch, gid = [], []
    for g in rng.permutation(len(groups)):
        batch.append(groups[g])
        if sum(len(x) for x in batch) >= bs:
            yield _encode_groups(batch, a, b, y, tok, rng)
            batch = []
    if batch:
        yield _encode_groups(batch, a, b, y, tok, rng)


def _encode_groups(batch, a, b, y, tok, rng):
    k = np.concatenate(batch)
    gid = np.concatenate([np.full(len(x), n) for n, x in enumerate(batch)])
    ta, tb = a[k], b[k]
    flip = rng.random(len(k)) < 0.5
    ta, tb = np.where(flip, tb, ta), np.where(flip, ta, tb)
    enc = tok(list(ta), list(tb), truncation=True, max_length=128, padding=True, return_tensors='pt')
    return enc, y[k], gid


def listwise_loss(logits, y, gid):
    """Mean over groups with >=1 positive of -log(sum_pos e^s / sum_all e^s)."""
    import torch
    terms = []
    for g in torch.unique(gid):
        m = gid == g
        s, t = logits[m], y[m]
        if t.sum() > 0:
            terms.append(torch.logsumexp(s, 0) - torch.logsumexp(s[t > 0], 0))
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _autocast(dev):
    import torch
    # fp16 on GPU (never bf16 for DeBERTa); plain fp32 on CPU (smoke tests only)
    return torch.autocast('cuda', dtype=torch.float16) if dev == 'cuda' else torch.autocast('cpu', enabled=False)


def train(epochs=1, bs=32, lr=2e-5, max_pairs=None, model_name=MODEL_NAME, loss='listwise'):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    import duckdb
    df = duckdb.sql(f"SELECT s1_idx, p1, text_a, text_b, label FROM read_parquet('{os.path.join(CE_DIR, 'ce_train.parquet')}')").df()
    if max_pairs and len(df) > max_pairs:
        if loss == 'listwise':      # sample whole Source 1 groups, not pairs
            s1 = df['s1_idx'].unique()
            keep = np.random.default_rng(26).permutation(s1)[:max(1, int(len(s1) * max_pairs / len(df)))]
            df = df[df['s1_idx'].isin(keep)].reset_index(drop=True)
        else:
            df = df.sample(max_pairs, random_state=26).reset_index(drop=True)
    groups = make_groups(df) if loss == 'listwise' else None
    if groups is not None:
        log(f'{len(groups):,} Source 1 groups, {sum(len(g) for g in groups):,} pairs '
            f'(of {len(df):,}; cap {GROUP_MAX} per group)')
    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_name)
    # 1 output = match logit; replaces any existing classification head of the checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, ignore_mismatched_sizes=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_pairs = sum(len(g) for g in groups) if groups is not None else len(df)
    steps = epochs * ((n_pairs + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = (torch.amp.GradScaler('cuda') if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()) \
        if dev == 'cuda' else None
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(26)
    model.train()
    step, t0 = 0, time.time()
    for _ in range(epochs):
        it = _group_batches(df, groups, tok, bs, rng) if groups is not None else \
            ((e, t, None) for e, t in _batches(df, tok, bs, shuffle=True, swap=True, rng=rng))
        for enc, y, gid in it:
            enc = {k: v.to(dev) for k, v in enc.items()}
            with _autocast(dev):
                logits = model(**enc).logits.squeeze(-1)
            yt = torch.from_numpy(y).to(dev)
            loss = loss_fn(logits.float(), yt)
            if gid is not None:
                loss = loss + LIST_WEIGHT * listwise_loss(logits.float(), yt, torch.from_numpy(gid).to(dev))
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                log(f'step {step}/{steps}  loss {loss.item():.4f}  {step * bs / (time.time() - t0):.0f} pairs/s')
    out = os.path.join(CE_DIR, 'model')
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log(f'saved {out}')


def score(split, bs=256):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import duckdb
    src = os.path.join(CE_DIR, f'{split}.parquet')
    df = duckdb.sql(f"SELECT s1_idx, tg_idx, text_a, text_b FROM read_parquet('{src}') WHERE in_band").df()
    # length-sorted batches: far less padding, same scores
    order = np.argsort((df['text_a'].str.len() + df['text_b'].str.len()).to_numpy())
    df = df.iloc[order].reset_index(drop=True)
    mdir = os.path.join(CE_DIR, 'model')
    tok = AutoTokenizer.from_pretrained(mdir)
    dev = _device()
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to(dev).eval()
    out, t0 = [], time.time()
    with torch.no_grad():
        for enc, _ in _batches(df, tok, bs, shuffle=False, swap=False, rng=None):
            enc = {k: v.to(dev) for k, v in enc.items()}
            with _autocast(dev):
                out.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
    df['ce'] = np.concatenate(out)
    path = os.path.join(config.CACHE_DIR, split, 'ce_scores.parquet')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duckdb.from_df(df[['s1_idx', 'tg_idx', 'ce']]).write_parquet(path, compression='zstd')
    log(f'{split}: scored {len(df):,} band pairs in {time.time() - t0:.0f}s -> {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['export', 'train', 'score'])
    ap.add_argument('--split', choices=config.SPLIT_NAMES)
    ap.add_argument('--max-pairs', type=int, default=None, help='cap on training pairs (train mode)')
    ap.add_argument('--model', default=MODEL_NAME, help='base checkpoint (train mode); tiny models for smoke tests')
    ap.add_argument('--loss', choices=['listwise', 'bce'], default='listwise',
                    help='listwise: BCE + per-Source-1 listwise term (default); bce: pointwise only')
    args = ap.parse_args()
    if args.mode == 'export':
        export(args.split)
    elif args.mode == 'train':
        train(max_pairs=args.max_pairs, model_name=args.model, loss=args.loss)
    else:
        score(args.split)


if __name__ == '__main__':
    main()
