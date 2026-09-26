"""
Pipeline B objective (JEPA-style), per batch of (Source 1, target) pairs:

  z   = f_xi(S1)                       target (EMA) encoder, stop-grad
  p   = g(f_theta(mask(target)))       context encoder + predictor on the noisy record
  ps  = g(f_theta(mask(S1)))           the same on a masked view of Source 1 itself

  L = w_con  * InfoNCE(z, [p; p_hard])          latent prediction must single out
                                                the right entity among in-batch and
                                                mined hard negatives
    + w_pred * (1 - cos(p, z))                  JEPA latent regression
    + w_var  * variance_hinge(p)                VICReg term against collapse
    + w_self * [(1 - cos(ps, z)) + InfoNCE(z, ps)]   masked-view self-prediction

After every optimiser step the target encoder follows the context encoder by
EMA (momentum annealed ema_momentum -> ema_momentum_end), as in I-JEPA/BYOL.
Masking = random token dropout + dropping the whole address field, so the
predictor has to infer the entity from partial evidence (the JEPA idea of
predicting the representation of missing content).
"""

import numpy as np

from blocker.training.contrastive_training import Trainer
from blocker.training.losses import cosine_regression, info_nce, variance_hinge


def mask_texts(texts, rng, token_p, field_p):
    out = []
    for t in texts:
        parts = t.split(' ; ')
        if len(parts) >= 2 and rng.random() < field_p:
            parts[1] = ''
        toks = ' ; '.join(parts).split(' ')
        if token_p > 0 and len(toks) > 2:
            keep = rng.random(len(toks)) >= token_p
            if not keep.any():
                keep[rng.integers(len(toks))] = True
            toks = [w for w, k in zip(toks, keep) if k]
        out.append(' '.join(toks))
    return np.array(out, dtype=object)


class JEPATrainer(Trainer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rng = np.random.default_rng(self.seed + 17)

    def loss(self, b, hn):
        m, c = self.model, self.mcfg
        T = float(c.get('temperature', 0.07))
        tp, fp = float(c.get('token_mask_prob', 0.15)), float(c.get('field_drop_prob', 0.1))
        s1_txt = self.q_texts[b[:, 0]]
        z = m.reference(**self.tok(s1_txt))
        keys = self.keys(b, hn)
        p = m.predict(**self.tok(mask_texts(self.t_texts[keys], self.rng, tp, 0.0)))
        q_owner, k_owner = self.owners(b[:, 0], keys)
        B = len(b)
        loss = float(c.get('contrastive_weight', 1.0)) * info_nce(z, p, q_owner, k_owner, T)
        loss = loss + float(c.get('predictive_weight', 1.0)) * cosine_regression(p[:B], z)
        loss = loss + float(c.get('variance_weight', 0.5)) * variance_hinge(p[:B])
        ws = float(c.get('self_weight', 0.5))
        if ws > 0:
            ps = m.predict(**self.tok(mask_texts(s1_txt, self.rng, tp, fp)))
            loss = loss + ws * (cosine_regression(ps, z) + info_nce(z, ps, q_owner, q_owner, T, symmetric=False))
        return loss

    def after_step(self, step, total):
        m0 = float(self.mcfg.get('ema_momentum', 0.996))
        m1 = float(self.mcfg.get('ema_momentum_end', 1.0))
        self.model.update_target(m0 + (m1 - m0) * min(1.0, step / max(total, 1)))


def train_jepa(train_data, val_data, cfg, out_dir, device):
    from blocker.models.jepa_encoder import JEPAEncoder
    mcfg = cfg['models']['jepa']
    model = JEPAEncoder.build(mcfg, cfg['models'].get('max_params'))
    JEPATrainer(model, train_data, val_data, mcfg, cfg, out_dir, device).fit()
    return out_dir
