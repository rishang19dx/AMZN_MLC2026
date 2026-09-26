"""
Metric-learning losses.

info_nce: InfoNCE / Multiple-Negatives-Ranking loss with in-batch + hard
negatives. A Source 1 entity often has several true targets, and a mined or
in-batch "negative" may in fact be one of them: every such column is masked
out (owner of the key == anchor entity), so the loss never pushes true
matches apart.
"""

import torch
import torch.nn.functional as F

NEG_INF = -1e4


def info_nce(q, k, q_owner, k_owner, temperature, symmetric=True):
    """
    q: (B, d) anchors (Source 1), k: (M, d) keys, M >= B, with k[i] the positive
    of q[i] and k[B:] extra (hard) negatives. q_owner: (B,) Source 1 row of each
    anchor; k_owner: (M,) Source 1 row that owns each key (-1 = none).
    All embeddings are L2-normalised.
    """
    B = q.shape[0]
    logits = (q @ k.T) / temperature
    same = (k_owner[None, :] == q_owner[:, None])
    same[torch.arange(B), torch.arange(B)] = False          # keep the positive itself
    logits = logits.masked_fill(same, NEG_INF)
    target = torch.arange(B, device=q.device)
    loss = F.cross_entropy(logits, target)
    if symmetric:
        logits_t = (k[:B] @ q.T) / temperature                 # target -> which Source 1?
        same_t = (q_owner[None, :] == q_owner[:, None])
        same_t[torch.arange(B), torch.arange(B)] = False
        loss = 0.5 * (loss + F.cross_entropy(logits_t.masked_fill(same_t, NEG_INF), target))
    return loss


def cosine_regression(pred, target):
    """JEPA latent prediction loss: 1 - cos(prediction, stop-grad target)."""
    return (1 - (F.normalize(pred, dim=-1) * F.normalize(target.detach(), dim=-1)).sum(-1)).mean()


def variance_hinge(z, gamma=1.0, eps=1e-4):
    """VICReg variance term: keeps every embedding dimension's std above gamma/sqrt(d)
    (scaled for unit-norm vectors), which prevents representation collapse."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(gamma / (z.shape[1] ** 0.5) - std).mean() * (z.shape[1] ** 0.5)
