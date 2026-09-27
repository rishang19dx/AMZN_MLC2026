"""
Pair energy model, v2 (scaled).

E(a, b) = energy of a record pair; low energy = same business. The match
logit is -E, so p(match) = sigmoid(-E), and within a candidate list
p(b | a) is proportional to exp(-E(a, b)).

Encoder (shared by both records):
  hashed token embedding (HASH_BUCKETS x d) + field embedding
  -> 2-layer pre-norm Transformer (records are short: <= 96 tokens)
  -> attention pooling + mean pooling -> entity vector (d_out)
Energy head:
  entity interactions [|a-b|, a*b, (a+b)/2]
  + late interaction: soft token alignment (cosine) between the two records,
    as coverage in both directions, overall and separately for name-like and
    address-like fields; matching evidence lives at token level ("same house
    number, same street, different name" must still be visible)
  -> MLP -> scalar energy
The encoder also serves as a bi-encoder (cosine of entity vectors), trained
with in-batch negatives; that makes it usable for retrieval / blocking too.

Default size (d=128, 1M buckets): ~136M parameters, far below the 8B limit.
"""

import torch
from torch import nn
import torch.nn.functional as F

from ebm.normalize import F_NAME, F_NAME3, F_LEGAL, F_ADDR, F_NUM, F_HOUSE, F_UNIT, F_ZIP

NAME_FIELDS = (F_NAME, F_NAME3, F_LEGAL)
ADDR_FIELDS = (F_ADDR, F_NUM, F_HOUSE, F_UNIT, F_ZIP)


class Encoder(nn.Module):
    def __init__(self, buckets, n_fields, d=128, layers=2, heads=4, d_out=256, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(buckets, d, padding_idx=0)
        self.field = nn.Embedding(n_fields, d, padding_idx=0)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True,
                                           activation='gelu')
        self.body = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(d) * 0.02)
        self.out = nn.Sequential(nn.Linear(2 * d, d_out), nn.GELU(), nn.LayerNorm(d_out))
        nn.init.normal_(self.tok.weight, std=0.02)

    def forward(self, tokens, fields):
        pad = tokens.eq(0)                                              # [B, L]
        h = self.tok(tokens) + self.field(fields.long())
        h = self.body(h, src_key_padding_mask=pad)                      # [B, L, d]
        keep = (~pad).unsqueeze(-1).float()
        att = (h @ self.query).masked_fill(pad, float('-inf')).softmax(-1).unsqueeze(-1)
        att = torch.nan_to_num(att)                                     # all-padding rows
        pooled = torch.cat([(att * h).sum(1), (h * keep).sum(1) / keep.sum(1).clamp_min(1)], -1)
        return self.out(pooled), h, ~pad


def _coverage(sim, mask_a, mask_b):
    """Mean over a's valid tokens of the best cosine to any valid token of b."""
    s = sim.masked_fill(~mask_b.unsqueeze(1), -1.0).amax(2)             # [B, La]
    return (s * mask_a).sum(1) / mask_a.sum(1).clamp_min(1)


class PairEnergyModel(nn.Module):
    def __init__(self, buckets, n_fields, d=128, layers=2, heads=4, d_out=256, hidden=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(buckets, n_fields, d, layers, heads, d_out, dropout)
        self.proj = nn.Linear(d, 64, bias=False)                        # token space for late interaction
        self.head = nn.Sequential(
            nn.Linear(3 * d_out + 6, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 1))
        self.logit_scale = nn.Parameter(torch.tensor(3.0))              # bi-encoder temperature (log)

    def encode(self, tokens, fields):
        return self.encoder(tokens, fields)

    def energy_from(self, ea, ha, ma, fa, eb, hb, mb, fb):
        ta, tb = F.normalize(self.proj(ha), dim=-1), F.normalize(self.proj(hb), dim=-1)
        sim = ta @ tb.transpose(1, 2)                                   # [B, La, Lb]
        feats = []
        for group in (None, NAME_FIELDS, ADDR_FIELDS):
            if group is None:
                ga, gb = ma, mb
            else:
                g = torch.tensor(group, device=fa.device)
                ga, gb = ma & torch.isin(fa, g), mb & torch.isin(fb, g)
            feats += [_coverage(sim, ga.float(), gb), _coverage(sim.transpose(1, 2), gb.float(), ga)]
        x = torch.cat([(ea - eb).abs(), ea * eb, (ea + eb) / 2, torch.stack(feats, 1)], 1)
        return self.head(x).squeeze(1)

    def forward(self, ta, fa, tb, fb):
        """Energy for aligned pairs (a_i, b_i)."""
        ea, ha, ma = self.encode(ta, fa)
        eb, hb, mb = self.encode(tb, fb)
        return self.energy_from(ea, ha, ma, fa, eb, hb, mb, fb)
