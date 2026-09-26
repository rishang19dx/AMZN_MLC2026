"""
Pipeline A model: a symmetric fine-tuned BERT-family bi-encoder.

Input view: the *normalised* fields in a fixed template
    "query: name: <name_n> | address: <addr_n> | country: <country_n>"
(transliterated, abbreviations unified). Source 1 and Source 2/3 records go
through the same encoder, so similarity = cosine of the two embeddings.
"""

import copy
import os

import torch
import torch.nn as nn

from blocker.models.encoder import TextEncoder, encode_texts, load_tokenizer, save_encoder_parts
from blocker.utils import check_param_budget, read_json


def bert_texts(df, prefix=''):
    return (prefix + 'name: ' + df['name_n'] + ' | address: ' + df['addr_n']
            + ' | country: ' + df['country_n']).to_numpy(dtype=object)


class DenseModel(nn.Module):
    """Interface shared by the two learned pipelines (trainer, miner, blocker)."""
    kind = ''

    def texts(self, df, role):
        raise NotImplementedError

    def embed_role(self, role, **tok):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    def embed(self, df, role, batch_size=512, fp16=False, out=None):
        was_training = self.training
        self.eval()
        try:
            return encode_texts(lambda **t: self.embed_role(role, **t), self.tokenizer, self.texts(df, role),
                                self.max_length, batch_size, self.device, fp16, out, desc=f'{self.kind}/{role}')
        finally:
            self.train(was_training)


class BertBiEncoder(DenseModel):
    kind = 'bert'

    def __init__(self, encoder, tokenizer, cfg):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.cfg = copy.deepcopy(cfg)
        self.max_length = int(cfg.get('max_length', 128))
        self.prefix = cfg.get('text_prefix', '') or ''

    @classmethod
    def build(cls, cfg, max_params):
        enc = TextEncoder.from_pretrained(cfg['model_name'], cfg.get('embedding_dim'), cfg.get('pooling', 'mean'))
        model = cls(enc, load_tokenizer(cfg['model_name']), cfg)
        check_param_budget(model, max_params, f"bert ({cfg['model_name']})")
        return model

    def texts(self, df, role):
        return bert_texts(df, self.prefix)

    def embed_role(self, role, **tok):
        return self.encoder(**tok)

    def save(self, out_dir):
        from blocker.utils import count_parameters
        save_encoder_parts(self, self.tokenizer, {'encoder': self.encoder}, out_dir,
                           {'kind': self.kind, 'config': self.cfg, 'n_params': count_parameters(self)})

    @classmethod
    def load(cls, out_dir, max_params=None, device='cpu'):
        meta = read_json(os.path.join(out_dir, 'meta.json'))
        cfg = meta['config']
        enc = TextEncoder.from_config_dir(os.path.join(out_dir, 'encoder_backbone'),
                                          cfg.get('embedding_dim'), cfg.get('pooling', 'mean'))
        model = cls(enc, load_tokenizer(os.path.join(out_dir, 'tokenizer')), cfg)
        model.load_state_dict(torch.load(os.path.join(out_dir, 'weights.pt'), map_location='cpu'))
        if max_params:
            check_param_budget(model, max_params, 'bert (loaded)')
        return model.to(device)

