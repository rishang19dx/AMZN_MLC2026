"""
Pipeline B model: a JEPA-style (Joint-Embedding Predictive Architecture)
entity encoder.

No JEPA checkpoint fits the challenge rules: Meta's I-JEPA / V-JEPA weights
are vision models under non-commercial licences, and there is no text JEPA
release. So the JEPA *objective* is implemented on an Apache-2.0 multilingual
MiniLM (see docs/BLOCKING.md):

  context encoder  f_theta (trained)    reads a noisy observation (an S2/S3 record,
                                        or a masked Source 1 record)
  target encoder   f_xi = EMA(f_theta)  reads the clean Source 1 reference record;
                                        no gradients (stop-grad), like I-JEPA
  predictor        g_phi (MLP)          predicts, in latent space, the target
                                        encoder's embedding of the reference record

Retrieval is asymmetric:
  Source 1 (query)   -> f_xi(record)                 "what the clean entity looks like"
  Source 2/3 (index) -> g_phi(f_theta(record))       "which clean entity this noisy record predicts"

This differs from Pipeline A in input view (raw text in its original script,
not the transliterated/normalised template), backbone, objective (latent
prediction + contrastive + variance regulariser, masked views) and
architecture (asymmetric), so the two pipelines make different mistakes.

The token-embedding matrix is shared (and frozen) between the two encoders,
so the whole model stays below 200M unique parameters (~139M).
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocker.models.bert_encoder import DenseModel
from blocker.models.encoder import TextEncoder, load_tokenizer, save_encoder_parts
from blocker.normalization import raw_view
from blocker.utils import check_param_budget, count_parameters, read_json


def jepa_texts(df):
    name = df['business_name'].map(raw_view)
    addr = df['business_address'].map(raw_view)
    return (name + ' ; ' + addr + ' ; ' + df['country_n']).to_numpy(dtype=object)


class JEPAEncoder(DenseModel):
    kind = 'jepa'

    def __init__(self, context, target, tokenizer, cfg):
        super().__init__()
        self.context = context
        self.target = target
        self.tokenizer = tokenizer
        self.cfg = copy.deepcopy(cfg)
        self.max_length = int(cfg.get('max_length', 128))
        d = context.embedding_dim
        h = int(cfg.get('predictor_hidden', 768))
        self.predictor = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, d))
        if cfg.get('share_embeddings', True):
            emb = context.backbone.get_input_embeddings()
            emb.weight.requires_grad_(False)
            target.backbone.set_input_embeddings(emb)
        for p in self.target.parameters():
            if p is not self.context.backbone.get_input_embeddings().weight:
                p.requires_grad_(False)

    @classmethod
    def build(cls, cfg, max_params):
        context = TextEncoder.from_pretrained(cfg['model_name'], cfg.get('embedding_dim'), cfg.get('pooling', 'mean'))
        target = copy.deepcopy(context)
        model = cls(context, target, load_tokenizer(cfg['model_name']), cfg)
        check_param_budget(model, max_params, f"jepa ({cfg['model_name']}, context + EMA target + predictor)")
        return model

    # -- embeddings ---------------------------------------------------------
    def texts(self, df, role):
        return jepa_texts(df)

    def reference(self, **tok):
        """Target (EMA) encoder: Source 1 side, no gradients."""
        with torch.no_grad():
            return self.target(**tok)

    def predict(self, **tok):
        """Context encoder + predictor: Source 2/3 side (and masked Source 1 views)."""
        return F.normalize(self.predictor(self.context(**tok)).float(), dim=-1)

    def embed_role(self, role, **tok):
        return self.reference(**tok) if role == 'query' else self.predict(**tok)

    # -- EMA ----------------------------------------------------------------
    @torch.no_grad()
    def update_target(self, momentum):
        for pc, pt in zip(self.context.parameters(), self.target.parameters()):
            if pt is pc:          # shared (frozen) embedding
                continue
            pt.mul_(momentum).add_(pc.detach(), alpha=1 - momentum)

    # -- persistence --------------------------------------------------------
    def save(self, out_dir):
        save_encoder_parts(self, self.tokenizer, {'context': self.context, 'target': self.target}, out_dir,
                           {'kind': self.kind, 'config': self.cfg, 'n_params': count_parameters(self)})

    @classmethod
    def load(cls, out_dir, max_params=None, device='cpu'):
        meta = read_json(os.path.join(out_dir, 'meta.json'))
        cfg = meta['config']
        mk = lambda n: TextEncoder.from_config_dir(os.path.join(out_dir, f'{n}_backbone'),
                                                   cfg.get('embedding_dim'), cfg.get('pooling', 'mean'))
        model = cls(mk('context'), mk('target'), load_tokenizer(os.path.join(out_dir, 'tokenizer')), cfg)
        model.load_state_dict(torch.load(os.path.join(out_dir, 'weights.pt'), map_location='cpu'))
        if max_params:
            check_param_budget(model, max_params, 'jepa (loaded)')
        return model.to(device)


def load_dense_model(kind, out_dir, max_params=None, device='cpu'):
    from blocker.models.bert_encoder import BertBiEncoder
    return {'bert': BertBiEncoder, 'jepa': JEPAEncoder}[kind].load(out_dir, max_params, device)
