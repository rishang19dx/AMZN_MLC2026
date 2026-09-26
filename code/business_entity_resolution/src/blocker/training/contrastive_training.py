"""
Fine-tuning loop shared by both learned pipelines, and the Pipeline A
(BERT bi-encoder) objective.

Pipeline A objective: symmetric InfoNCE (Multiple Negatives Ranking loss)
between Source 1 and its matching Source 2/3 records, with
  * in-batch negatives (same-country batches, so they are plausible look-alikes),
  * mined hard negatives (hard_negative_mining.py) from round 2 on,
  * masking of other true matches of the same entity (never pushed apart).

Model selection: recall@k on a held-out *validation* sub-universe after every
epoch; the best epoch is saved. Validation entities never contribute pairs.
"""

import math
import os
import time

import numpy as np
import torch

from blocker.data import subsample
from blocker.pipelines.ann_index import exact_topk
from blocker.training.hard_negative_mining import mine_hard_negatives
from blocker.training.losses import info_nce
from blocker.training.pairs import PairBatcher, positive_pairs
from blocker.models.encoder import tokenize
from blocker.utils import data_parallel, log, n_gpus, set_seed, write_json


def evaluation_universe(val_data, max_entities, seed):
    """A smaller closed universe of the validation data (same distractor ratio)."""
    if val_data is None or not val_data.has_truth:
        return None
    frac = min(1.0, max_entities / max(len(val_data.s1), 1)) if max_entities else 1.0
    return subsample(val_data, frac, seed, name=f'{val_data.name}-eval') if frac < 1 else val_data


def recall_at_k(model, data, k, batch_size=512, fp16=False):
    """Share of true pairs whose target is in the top-k (same country) of its Source 1."""
    q = model.embed(data.s1, 'query', batch_size, fp16)
    t = model.embed(data.tg, 'target', batch_size, fp16)
    i_true, j_true = data.positive_pairs()
    s1c, tgc = data.s1['country_n'].to_numpy(), data.tg['country_n'].to_numpy()
    hit = 0
    for c in np.unique(s1c):
        qi, ti = np.flatnonzero(s1c == c), np.flatnonzero(tgc == c)
        if len(ti) == 0:
            continue
        _, I = exact_topk(q[qi], t[ti], k)
        found = set()
        for r, row in zip(qi, I):
            found.update((int(r), int(ti[x])) for x in row if x >= 0)
        hit += sum((int(a), int(b)) in found for a, b in zip(i_true, j_true) if s1c[a] == c)
    return hit / max(len(i_true), 1)


class Trainer:
    """Epoch loop, optimiser, schedule, AMP, evaluation, mining, checkpointing."""

    def __init__(self, model, train_data, val_data, mcfg, cfg, out_dir, device):
        self.model = model.to(device)
        self.train_data, self.mcfg, self.cfg, self.out_dir, self.device = train_data, mcfg, cfg, out_dir, device
        self.tcfg = cfg.get('training', {})
        self.seed = int(cfg.get('seed', 42))
        self.eval_data = evaluation_universe(val_data, self.tcfg.get('eval_max_entities', 20000), self.seed)
        self.q_texts = model.texts(train_data.s1, 'query')
        self.t_texts = model.texts(train_data.tg, 'target')
        self.owner = train_data.owner()
        self.amp = device.type == 'cuda' and mcfg.get('amp', True)
        # bf16 only on Ampere+ (T4 / P100 report "supported" via slow emulation): fp16 + loss scaling there
        self.amp_dtype = (torch.bfloat16 if self.amp and torch.cuda.get_device_capability(device)[0] >= 8
                          else torch.float16)
        self._dp = {}
        self.scaler = torch.amp.GradScaler('cuda') if self.amp and self.amp_dtype == torch.float16 else None

    # hooks ------------------------------------------------------------------
    def loss(self, b, hn):
        raise NotImplementedError

    def after_step(self, step, total):
        pass

    # helpers ----------------------------------------------------------------
    def par(self, name, module):
        """`module` split over all GPUs (DataParallel) when there are several; cached."""
        if name not in self._dp:
            self._dp[name] = data_parallel(module)
            if self._dp[name] is not module:
                log(f'  {name}: DataParallel over {n_gpus()} GPUs')
        return self._dp[name]

    def tok(self, texts):
        return tokenize(self.model.tokenizer, texts, self.model.max_length, self.device)

    def owners(self, rows_q, rows_k):
        q = torch.as_tensor(np.asarray(rows_q, np.int64), device=self.device)
        k = torch.as_tensor(self.owner[np.asarray(rows_k)].astype(np.int64), device=self.device)
        return q, k

    def keys(self, b, hn):
        return np.concatenate([b[:, 1], hn]) if hn is not None and len(hn) else b[:, 1]

    # main loop --------------------------------------------------------------
    def fit(self):
        set_seed(self.seed)
        mcfg, hcfg = self.mcfg, self.tcfg.get('hard_negatives', {})
        pairs = positive_pairs(self.train_data)
        if len(pairs) == 0:
            raise ValueError('no positive training pairs')
        batcher = PairBatcher(pairs, self.train_data.s1['country_n'].to_numpy(), mcfg.get('batch_size', 64),
                              self.seed, mcfg.get('max_pairs_per_epoch', 0), mcfg.get('country_homogeneous_batches', True))
        epochs = int(mcfg.get('epochs', 1))
        per_epoch = batcher.n_batches()
        total = epochs * per_epoch
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=float(mcfg.get('learning_rate', 3e-5)),
                                weight_decay=float(mcfg.get('weight_decay', 0.01)))
        warm = max(1, int(total * float(mcfg.get('warmup_ratio', 0.06))))
        sched = torch.optim.lr_scheduler.LambdaLR(      # linear warm-up, then linear decay
            opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
        log(f'{self.model.kind}: {len(pairs):,} positive pairs, {epochs} epochs x {per_epoch:,} batches '
            f'(batch {batcher.batch_size}), device {self.device}')

        hard_negs, history, best, step = None, [], -1.0, 0
        mine_after = set(int(e) for e in hcfg.get('mine_after_epochs', []) or [])
        for ep in range(1, epochs + 1):
            self.model.train()
            t0, run = time.time(), []
            for b, hn in batcher.epoch(ep, hard_negs, int(hcfg.get('per_batch_item', 1))):
                with torch.autocast('cuda', dtype=self.amp_dtype, enabled=self.amp):
                    loss = self.loss(b, hn)
                if not torch.isfinite(loss):
                    log(f'  non-finite loss at step {step}, batch skipped')
                    opt.zero_grad(set_to_none=True)
                    continue
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    self.scaler.step(opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                self.after_step(step, total)
                run.append(float(loss.detach()))
                if step % 200 == 0 or step == total:
                    log(f'  ep {ep} step {step:,}/{total:,} loss {np.mean(run[-200:]):.4f} '
                        f'lr {sched.get_last_lr()[0]:.2e} ({(time.time() - t0) / len(run):.2f}s/batch)')
            rec = {'epoch': ep, 'loss': float(np.mean(run)) if run else math.nan, 'seconds': time.time() - t0}
            if self.eval_data is not None and self.tcfg.get('eval_every_epoch', True):
                k = int(self.tcfg.get('eval_k', 30))
                rec[f'val_recall@{k}'] = recall_at_k(self.model, self.eval_data, k,
                                                     int(mcfg.get('eval_batch_size', 512)))
                log(f'  epoch {ep}: val recall@{k} = {rec[f"val_recall@{k}"]:.4f} '
                    f'({len(self.eval_data.s1):,} val entities)')
                if rec[f'val_recall@{k}'] > best:
                    best = rec[f'val_recall@{k}']
                    self.model.save(self.out_dir)
                    rec['saved'] = True
            else:
                self.model.save(self.out_dir)
                rec['saved'] = True
            history.append(rec)
            if hcfg.get('enabled', False) and ep in mine_after and ep < epochs:
                hard_negs = mine_hard_negatives(self.model, self.train_data, hcfg, self.cfg.get('ann', {}),
                                                self.seed + ep, int(mcfg.get('eval_batch_size', 512)))
        write_json({'history': history, 'best_val_recall': best if best >= 0 else None},
                   os.path.join(self.out_dir, 'training_history.json'))
        return history


class BertTrainer(Trainer):
    def loss(self, b, hn):
        m = self.model
        enc = self.par('encoder', m.encoder)
        q = enc(**self.tok(self.q_texts[b[:, 0]]))
        keys = self.keys(b, hn)
        k = enc(**self.tok(self.t_texts[keys]))
        q_owner, k_owner = self.owners(b[:, 0], keys)
        return info_nce(q, k, q_owner, k_owner, float(self.mcfg.get('temperature', 0.05)))


def train_bert(train_data, val_data, cfg, out_dir, device):
    from blocker.models.bert_encoder import BertBiEncoder
    mcfg = cfg['models']['bert']
    model = BertBiEncoder.build(mcfg, cfg['models'].get('max_params'))
    BertTrainer(model, train_data, val_data, mcfg, cfg, out_dir, device).fit()
    return out_dir
