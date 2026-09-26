"""Shared fixtures: a small synthetic challenge directory and tiny local encoders
(random-initialised BERT with a character vocabulary; nothing is downloaded)."""

import copy
import os
import string
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, SRC)
import blocker  # noqa: E402,F401  (environment for faiss/torch/lightgbm)

HEADER = 'entity_id\tbusiness_name\tbusiness_address\tcountry\n'

# (S1 name, S1 address, country, [(source, variant name, variant address)])
ENTITIES = [
    ('Sharma Medical Store', '12 MG Road, Pune, Maharashtra', 'India',
     [(2, 'SHARMA MED STORE', '12 M.G. Rd, Pune'), (3, 'Sharma Medical Stores Pvt Ltd', 'Pune, 12 MG Road')]),
    ('ABC Medical Store', '5 Station Road, Delhi', 'India',
     [(2, 'ABC Med Store', '5 Station Rd, New Delhi')]),
    ('ABC Medical Centre', '77 Lake Town, Kolkata', 'India', [(3, 'ABC Medical Center', 'Lake Town 77, Kolkata')]),
    ('Raj Investments LLP', '6 CIT Colony, Chennai 600004', 'India',
     [(2, 'Raj Investment', 'C.I.T. Colony 6, Chennai'), (3, 'RAJ INVESTMENTS', '6 CIT Colony Chennai 600004')]),
    ('Prime Money', '17560 Ellis Road, Tahlequah, OK', 'US',
     [(2, 'Prime Money Inc', '17560 ELLIS RD, TAHLEQUAH, OK'), (3, 'prime money', 'Ellis Road 17560, Tahlequah')]),
    ('Orelees Barbershop', '1795 Westchester Drive, High Point, NC', 'US',
     [(2, "Orelee's Barber Shop", '1795 WESTCHESTER DR, HIGH POINT, NC')]),
    ('Dahlia Power Reliable Scientific', '630 45th Terrace, Kansas City, MO', 'US',
     [(2, 'Dahlia Power Reliable', 'KANSAS CITY, MO, 630 45ND TERRACE, null'),
      (3, 'Dahlia Ponr Reliable Scientific LLC', 'Missouri, 630 45th Terrace, Kansas City')]),
    ('Vision Partners Corp', '1064 Newton Rd, Unit 11, Iowa City, IA', 'US',
     [(3, 'Vision Partners Corporation', 'IA, Iowa City, 1064 Newton Rd, Unit 11')]),
    ('Lonely Singleton Bakery', '1 Nowhere Lane, Austin, TX', 'US', []),
    ('Thermal & Fils SASU', '20 Rue Parmentier, Dunkerque', 'France',
     [(2, 'THERMAL ET FILS', '20 R. PARMENTIER, DUNKERQUE'), (3, 'Thermal and Fils S.A.S.U', 'Dunkerque, 20 Rue Parmentier')]),
    ('Grain & Fils', '329 Avenue de Dunkerque, Lille', 'France', [(2, 'Grain Fils SARL', '329 AV DE DUNKERQUE, LILLE')]),
    ('Maison de Sante Generation', '30 Rue Louis Thenard, Saint-Nazaire', 'France', []),
]
DISTRACTORS = [
    (2, 'Unrelated Hardware Mart', '99 Park Street, Mumbai', 'India'),
    (3, 'Sharma Electricals', '400 Ring Road, Surat', 'India'),
    (2, 'Prime Pizza', '12 Elm St, Dallas, TX', 'US'),
    (3, 'Club de Foot', '8 Rue Nationale, Lille', 'France'),
    (2, 'Zeta Consulting LLC', '5 Main St, Boston, MA', 'Germany'),   # unseen country with no S1
]


def write_dir(root, prefix='test', with_truth=True):
    os.makedirs(root, exist_ok=True)
    s1 = [HEADER]
    src = {2: [HEADER], 3: [HEADER]}
    gt = ['source1_entity_id\tmatched_entity_ids\n']
    tid = {2: 100, 3: 500}
    for n, (name, addr, country, variants) in enumerate(ENTITIES):
        sid = f'S1-{n + 1:04d}'
        s1.append(f'{sid}\t{name}\t{addr}\t{country}\n')
        ids = []
        for s, vn, va in variants:
            tid[s] += 1
            t = f'S{s}-{tid[s]:05d}'
            src[s].append(f'{t}\t{vn}\t{va}\t{country}\n')
            ids.append(t)
        gt.append(f"{sid}\t{','.join(ids)}\n")
    for s, vn, va, c in DISTRACTORS:
        tid[s] += 1
        src[s].append(f'S{s}-{tid[s]:05d}\t{vn}\t{va}\t{c}\n')
    files = {f'{prefix}_source1.tsv': s1, f'{prefix}_source2.tsv': src[2], f'{prefix}_source3.tsv': src[3]}
    if with_truth:
        files[f'{prefix}_ground_truth.tsv'] = gt
    for fn, lines in files.items():
        with open(os.path.join(root, fn), 'w', encoding='utf-8') as f:
            f.writelines(lines)
    return root


@pytest.fixture
def data_dir(tmp_path):
    return write_dir(str(tmp_path / 'data'))


@pytest.fixture
def cfg():
    from blocker.config import load_config
    c = load_config(overrides=['runtime.device=cpu', 'runtime.workers=1', 'normalization.workers=1',
                               'runtime.fp16_inference=false'])
    for k in ('bert', 'jepa'):
        c['models'][k].update(embedding_dim=16, max_length=48, batch_size=4, eval_batch_size=8, epochs=1)
        c['pipelines'][k].update(top_k_source2=3, top_k_source3=3)
    c['training']['hard_negatives'].update(max_anchors=50, max_targets=100, top_m=5, per_anchor=2)
    c['training']['eval_max_entities'] = 0
    for p in c['pipelines']['classical']['passes'].values():
        if p['top_k']:
            p['top_k'] = 4
    c['candidate_generation']['max_candidates_per_source1'] = 8
    c['validation']['k_sweep'] = [2, 4]
    c['validation']['budget_sweep'] = [2, 4]
    c['matcher']['folds'] = 2
    return c


def _tokenizer(tmp):
    from transformers import BertTokenizerFast
    chars = string.ascii_lowercase + string.digits + string.punctuation
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + list(chars) + ['##' + c for c in chars]
    path = os.path.join(tmp, 'vocab.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(vocab))
    return BertTokenizerFast(vocab_file=path, do_lower_case=True)


def _backbone(tok):
    import torch
    from transformers import BertConfig, BertModel
    torch.manual_seed(0)
    return BertModel(BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                intermediate_size=64, max_position_embeddings=128))


@pytest.fixture
def tiny_models(tmp_path, cfg):
    """Factory: kind -> freshly initialised tiny model of that pipeline."""
    from blocker.models.bert_encoder import BertBiEncoder
    from blocker.models.encoder import TextEncoder
    from blocker.models.jepa_encoder import JEPAEncoder

    def make(kind):
        tok = _tokenizer(str(tmp_path))
        mcfg = dict(cfg['models'][kind], model_name='tiny-test-bert')
        enc = TextEncoder(_backbone(tok), mcfg['embedding_dim'])
        if kind == 'bert':
            return BertBiEncoder(enc, tok, mcfg)
        return JEPAEncoder(enc, copy.deepcopy(enc), tok, mcfg)
    return make
