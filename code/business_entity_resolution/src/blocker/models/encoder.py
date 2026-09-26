"""
Shared transformer text encoder: backbone -> pooling -> linear projection ->
L2-normalised embedding, plus batched inference.

Saving writes the backbone *config*, the tokenizer and one state dict, so
loading needs no network (AutoModel.from_config + load_state_dict).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocker.utils import log


def _pool(hidden, mask, how):
    if how == 'cls':
        return hidden[:, 0]
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1e-6)


class TextEncoder(nn.Module):
    def __init__(self, backbone, embedding_dim=None, pooling='mean'):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.embedding_dim = int(embedding_dim or hidden)
        self.pooling = pooling
        self.proj = nn.Linear(hidden, self.embedding_dim) if self.embedding_dim != hidden else nn.Identity()

    @classmethod
    def from_pretrained(cls, name, embedding_dim=None, pooling='mean'):
        from transformers import AutoModel
        return cls(AutoModel.from_pretrained(name), embedding_dim, pooling)

    @classmethod
    def from_config_dir(cls, backbone_dir, embedding_dim=None, pooling='mean'):
        from transformers import AutoConfig, AutoModel
        return cls(AutoModel.from_config(AutoConfig.from_pretrained(backbone_dir)), embedding_dim, pooling)

    def forward(self, input_ids, attention_mask, normalize=True, **_):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        z = self.proj(_pool(out.last_hidden_state, attention_mask, self.pooling))
        return F.normalize(z.float(), dim=-1) if normalize else z


def tokenize(tokenizer, texts, max_length, device):
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return {k: v.to(device) for k, v in enc.items() if k in ('input_ids', 'attention_mask')}


def _autocast(device, enabled):
    if enabled and device.type in ('cuda', 'mps'):
        try:
            return torch.autocast(device_type=device.type, dtype=torch.float16)
        except (RuntimeError, ValueError):
            pass
    import contextlib
    return contextlib.nullcontext()


@torch.inference_mode()
def encode_texts(fn, tokenizer, texts, max_length, batch_size, device, fp16=False, out=None, desc='encode'):
    """
    Embed `texts` with `fn(**tokens) -> (B, d)` tensor. Identical texts are
    embedded once; the rest is sorted by length so each batch pads little.
    `out` may be a preallocated (N, d) array or np.memmap (e.g. float16 on disk).
    """
    import pandas as pd
    inv, uniq_texts = pd.factorize(pd.Series(np.asarray(texts, dtype=object)), sort=False)
    uniq_texts = np.asarray(uniq_texts, dtype=object)
    order = np.argsort(np.fromiter((len(t) for t in uniq_texts), np.int64, len(uniq_texts)), kind='stable')
    emb_u = None
    n = len(uniq_texts)
    step = max(1, n // 20)
    next_log = step
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        with _autocast(device, fp16):
            z = fn(**tokenize(tokenizer, uniq_texts[idx], max_length, device))
        z = z.float().cpu().numpy()
        if emb_u is None:
            emb_u = np.empty((n, z.shape[1]), np.float32)
        emb_u[idx] = z
        if s + batch_size >= next_log:
            log(f'  {desc}: {min(s + batch_size, n):,}/{n:,} unique texts')
            next_log += step
    if emb_u is None:
        return np.empty((0, 0), np.float32) if out is None else out
    if out is None:
        return emb_u[inv]
    for s in range(0, len(inv), 1_000_000):
        out[s:s + 1_000_000] = emb_u[inv[s:s + 1_000_000]]
    return out


def save_encoder_parts(model, tokenizer, backbones, out_dir, meta):
    """backbones: {name: TextEncoder} whose backbone configs must be saved."""
    from blocker.utils import write_json
    os.makedirs(out_dir, exist_ok=True)
    for name, enc in backbones.items():
        enc.backbone.config.save_pretrained(os.path.join(out_dir, f'{name}_backbone'))
    tokenizer.save_pretrained(os.path.join(out_dir, 'tokenizer'))
    torch.save(model.state_dict(), os.path.join(out_dir, 'weights.pt'))
    write_json(meta, os.path.join(out_dir, 'meta.json'))


def load_tokenizer(path_or_name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path_or_name)
