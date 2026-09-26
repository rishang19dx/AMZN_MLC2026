"""
Unit / integration tests of the multi-pipeline blocker on a synthetic
challenge directory. Run from code/business_entity_resolution/:

    .venv/bin/python -m pytest tests/test_blocker.py -q

The learned encoders here are tiny random-initialised BERTs (no download); the
two training tests run one epoch on ~20 synthetic pairs to check the loop.
"""

import filecmp
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from blocker.data import load_dir, split_holdout, subsample
from blocker.normalization import (extract_postal, name_core, normalize_address, normalize_country, normalize_name,
                                   prepare_data)
from blocker.pipelines import candidate_union as cu
from blocker.pipelines.ann_index import AnnIndex, exact_topk
from blocker.pipelines.common import Retrieval, country_groups
from blocker.training.losses import info_nce
from blocker.training.pairs import PairBatcher, positive_pairs, random_negative_pairs

from conftest import write_dir

VALIDATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'utils', 'validate_submission.py')


def prepared(d, cfg):
    return prepare_data(load_dir(d), cfg)


def read_tsv_rows(path):
    with open(path, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        rows = [line.rstrip('\n').split('\t') for line in f]
    return header, rows


# 1. TSV loading --------------------------------------------------------------

def test_load_dir(data_dir):
    d = load_dir(data_dir)
    assert list(d.s1.columns) == ['entity_id', 'business_name', 'business_address', 'country']
    assert d.tg['entity_id'].str[:3].isin(['S2-', 'S3-']).all()
    assert set(d.tg['source']) == {2, 3}
    assert d.gt['S1-0009'] == set()                               # singleton
    i, j = d.positive_pairs()
    assert len(i) == sum(len(v) for v in d.gt.values())
    assert (d.owner()[j] == i).all()
    assert "Orelee's Barber Shop" in set(d.tg['business_name'])   # quotes/apostrophes read verbatim


def test_holdout_and_subsample_are_closed_universes(data_dir):
    d = load_dir(data_dir)
    tr, va = split_holdout(d, 0.5, 'mlc26')
    assert len(tr.s1) + len(va.s1) == len(d.s1) and len(tr.tg) + len(va.tg) == len(d.tg)
    assert not set(tr.s1_ids) & set(va.s1_ids)
    for part in (tr, va):                                          # every match of an entity stays with it
        ids = set(part.tg_ids)
        assert all(t in ids for s in part.s1_ids for t in d.gt[s])
    sub = subsample(d, 0.5, 3)
    ids = set(sub.tg_ids)
    assert all(t in ids for s in sub.s1_ids for t in d.gt[s])


# 2. normalisation ---------------------------------------------------------------

def test_normalization():
    assert normalize_name('Pvt. EFS Print Ventures Ltd.') == 'private efs print ventures limited'
    assert name_core(normalize_name('Pvt. EFS Print Ventures Ltd.')) == 'efs print ventures'
    assert normalize_name('Fractales Amis Groupe S.A.S') == 'fractales amis groupe sas'
    assert normalize_name('Thermal & Fils') == normalize_name('Thermal and Fils')
    assert normalize_address('630 45ND TERRACE, null') == '630 45 terrace'
    assert normalize_address('12 M.G. Rd') == '12 mg road'
    assert normalize_address('63 R. DE DIEPPE') == '63 rue de dieppe'
    assert normalize_address('Flat 007, 2nd Floor') == 'flat 7 2 floor'
    assert extract_postal('Chennai 600004, TN') == ['600004']
    assert normalize_country(' USA ') == 'us' and normalize_country('France') == 'france'
    assert normalize_country('Deutschland') == 'deutschland'      # open set: passes through
    assert normalize_country('') == 'unknown'


def test_prepare_keeps_raw(data_dir, cfg):
    d = prepared(data_dir, cfg)
    for col in ('business_name', 'name_n', 'addr_n', 'country_n', 'name_core', 'full_n', 'keys'):
        assert col in d.s1 and col in d.tg
    assert d.s1.loc[0, 'business_name'] == 'Sharma Medical Store'


# 3-4. pair construction ---------------------------------------------------------

def test_positive_and_negative_pairs(data_dir, cfg):
    d = prepared(data_dir, cfg)
    pos = positive_pairs(d)
    own = d.owner()
    assert len(pos) == sum(len(v) for v in d.gt.values()) == 15 and (own[pos[:, 1]] == pos[:, 0]).all()
    neg = random_negative_pairs(d, 3, seed=0)
    assert len(neg) > 0 and (own[neg[:, 1]] != neg[:, 0]).all()
    s1c, tgc = d.s1['country_n'].to_numpy(), d.tg['country_n'].to_numpy()
    assert (s1c[neg[:, 0]] == tgc[neg[:, 1]]).all()               # same-country negatives


def test_batcher_country_homogeneous_and_hard_negs(data_dir, cfg):
    d = prepared(data_dir, cfg)
    pos = positive_pairs(d)
    s1c = d.s1['country_n'].to_numpy()
    b = PairBatcher(pos, s1c, 4, seed=1)
    seen = []
    for batch, hn in b.epoch(1, {int(pos[0, 0]): np.array([pos[-1, 1]])}, per_item=1):
        assert len(set(s1c[batch[:, 0]])) == 1
        seen.extend(map(tuple, batch))
    assert sorted(seen) == sorted(map(tuple, pos))


def test_info_nce_masks_same_entity():
    q = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    k = q.clone()
    owner = torch.tensor([0, 0, 1])                                # rows 0 and 1: same entity
    loss_masked = info_nce(q, k, owner, owner, 0.05)
    loss_plain = info_nce(q, k, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), 0.05)
    assert torch.isfinite(loss_masked) and loss_masked <= loss_plain + 1e-6


# 5. embeddings -----------------------------------------------------------------

@pytest.mark.parametrize('kind', ['bert', 'jepa'])
def test_embedding_generation_and_roundtrip(tmp_path, data_dir, cfg, tiny_models, kind):
    from blocker.models.jepa_encoder import load_dense_model
    d = prepared(data_dir, cfg)
    m = tiny_models(kind)
    q = m.embed(d.s1, 'query', batch_size=5)
    t = m.embed(d.tg, 'target', batch_size=5)
    assert q.shape == (len(d.s1), 16) and t.shape == (len(d.tg), 16)
    assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-4)
    out = np.lib.format.open_memmap(str(tmp_path / 'e.npy'), mode='w+', dtype=np.float16, shape=t.shape)
    m.embed(d.tg, 'target', batch_size=3, out=out)
    assert np.allclose(out, t, atol=2e-3)                          # batching/dedup/memmap do not change results
    m.save(str(tmp_path / kind))
    m2 = load_dense_model(kind, str(tmp_path / kind), max_params=2e8)
    assert np.allclose(m2.embed(d.s1, 'query'), q, atol=1e-5)


def test_jepa_shares_embeddings_and_ema(tiny_models):
    m = tiny_models('jepa')
    assert m.target.backbone.get_input_embeddings() is m.context.backbone.get_input_embeddings()
    assert not any(p.requires_grad for p in m.target.parameters())
    before = [p.clone() for p in m.target.parameters()]
    with torch.no_grad():
        for p in m.context.parameters():
            if p.requires_grad:
                p.add_(1.0)
    m.update_target(0.5)
    changed = [not torch.equal(a, b) for a, b in zip(before, m.target.parameters())]
    assert any(changed)


def test_param_budget_is_enforced(tiny_models):
    from blocker.utils import check_param_budget
    with pytest.raises(ValueError):
        check_param_budget(tiny_models('bert'), 1000, 'tiny')


# 6. ANN retrieval ----------------------------------------------------------------

@pytest.mark.parametrize('kind', ['flat', 'ivf', 'ivf_sq8', 'hnsw'])
def test_ann_retrieval_matches_exact(kind):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(2000, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[:50] + 0.01 * rng.normal(size=(50, 16)).astype(np.float32)
    idx = AnnIndex(16, {'index_type': kind, 'nprobe': 64, 'nlist': 16}).build(v)
    S, I = idx.search(q, 5)
    assert (I[:, 0] == np.arange(50)).mean() > 0.95
    _, Ie = exact_topk(q, v, 5)
    assert (Ie[:, 0] == np.arange(50)).all()
    S2, I2 = AnnIndex(16, {'index_type': 'flat'}).build(v[:3]).search(q[:2], 5)
    assert (I2[:, 3:] == -1).all()                                # fewer vectors than k


# 7. classical blocking -----------------------------------------------------------

def test_classical_blocker_finds_noisy_matches(data_dir, cfg):
    from blocker.pipelines.classical_blocker import ClassicalBlocker
    d = prepared(data_dir, cfg)
    rets = ClassicalBlocker(cfg['pipelines']['classical'], 1).retrieve(d, country_groups(d))
    assert {r.name for r in rets} == {'addr', 'full', 'keys'}
    got = set()
    for r in rets:
        got |= set(zip(r.r.tolist(), r.c.tolist()))
        assert (r.rank >= 1).all() and (r.rank <= r.k).all()
    i, j = d.positive_pairs()
    recall = np.mean([(a, b) in got for a, b in zip(i.tolist(), j.tolist())])
    assert recall >= 0.9
    s1c, tgc = d.s1['country_n'].to_numpy(), d.tg['country_n'].to_numpy()
    assert all(s1c[a] == tgc[b] for a, b in got)                  # never crosses countries


def test_country_groups_open_set(data_dir, cfg):
    d = prepared(data_dir, cfg)
    d.s1.loc[0, 'country_n'] = 'atlantis'                          # a country with no targets at all
    groups = {c: (a, b) for c, a, b in country_groups(d, 'global')}
    assert len(groups['atlantis'][1]) == len(d.tg)                # falls back to every target
    assert 'france' in groups


# 8-9. candidate union + duplicate removal ----------------------------------------

def _ret(name, fam, pairs, k=5):
    r = np.array([p[0] for p in pairs], np.int32)
    c = np.array([p[1] for p in pairs], np.int32)
    return Retrieval(name, fam, k, r, c, np.linspace(0.9, 0.5, len(pairs)).astype(np.float32),
                     np.arange(1, len(pairs) + 1, dtype=np.float32))


def test_union_dedups_and_prioritises():
    rets = cu.sort_by_s1([_ret('bert', 'bert', [(0, 1), (0, 2), (1, 3)]),
                          _ret('jepa', 'jepa', [(0, 2), (1, 4)]),
                          _ret('addr', 'classical', [(0, 2), (0, 5), (1, 3)])])
    u = cu.union_rows(rets, 10, 0, 2)
    pairs = list(zip(u['s1'].tolist(), u['tg'].tolist()))
    assert pairs == sorted(set(pairs)) == [(0, 1), (0, 2), (0, 5), (1, 3), (1, 4)]
    npipe = dict(zip(pairs, u['n_pipelines'].tolist()))
    assert npipe[(0, 2)] == 3 and npipe[(1, 3)] == 2 and npipe[(0, 5)] == 1
    pr = dict(zip(pairs, u['priority'].tolist()))
    assert pr[(0, 2)] > pr[(0, 1)] and pr[(1, 3)] > pr[(1, 4)]    # consensus first
    keep = cu.budget_mask(u, 2)
    kept = {p for p, k in zip(pairs, keep) if k}
    assert {p for p in kept if p[0] == 0} == {(0, 2), (0, 1)}      # (0, 5): single pipeline, rank 2
    keep_prot = cu.budget_mask(u, 1, protect_min_pipelines=2)
    assert {p for p, k in zip(pairs, keep_prot) if k} >= {(0, 2), (1, 3)}
    assert not np.isnan(u['score_bert'][pairs.index((0, 1))]) and np.isnan(u['score_jepa'][pairs.index((0, 1))])


def test_target_side_pruning():
    rets = cu.sort_by_s1([_ret('bert', 'bert', [(0, 7), (1, 7), (2, 7)])])
    u = cu.union_rows(rets, 10, 0, 3)
    cuts = cu.target_cutoffs(u['tg'], u['s1'], u['priority'], 10, 2)
    assert cu.target_mask(u, *cuts).sum() == 2


# 10-14. end-to-end outputs --------------------------------------------------------

@pytest.fixture
def artifacts(tmp_path, tiny_models):
    a = tmp_path / 'artifacts'
    for kind in ('bert', 'jepa'):
        tiny_models(kind).save(str(a / kind))
    return str(a)


def _run(data_dir, cfg, artifacts, out, sweep=False):
    from blocker.run import block_and_report
    d = prepared(data_dir, cfg)
    return block_and_report(d, cfg, out, artifacts, None, evaluate=True, sweep=sweep)


def test_end_to_end_output_format(tmp_path, data_dir, cfg, artifacts):
    out = str(tmp_path / 'out')
    rep = _run(data_dir, cfg, artifacts, out, sweep=True)
    header, rows = read_tsv_rows(os.path.join(out, 'candidate_pairs.tsv'))
    assert header == ['source1_entity_id', 'candidate_entity_ids']
    d = load_dir(data_dir)
    s1_ids, tg_ids = list(d.s1_ids), set(d.tg_ids)
    assert [r[0] for r in rows] == s1_ids                           # 11. every S1 exactly once, in order
    for s1_id, ids in rows:
        lst = [x for x in ids.split(',') if x]
        assert len(lst) == len(set(lst))                             # 9. no duplicates
        assert not any(x.startswith('S1-') for x in lst)            # 12. no Source 1 ids
        assert set(lst) <= tg_ids                                    # 13. only existing S2/S3 ids
        assert len(lst) <= cfg['candidate_generation']['max_candidates_per_source1']
    assert rep['pair_recall'] > 0.5 and rep['reduction_ratio'] > 0
    assert os.path.exists(os.path.join(out, 'blocking_report.txt'))
    assert os.listdir(os.path.join(out, 'debug_candidate_scores'))
    # every recovered true pair is in candidate_pairs.tsv (and nothing else was lost)
    cands = {(s, x) for s, ids in rows for x in ids.split(',') if x}
    true_found = sum((s, t) in cands for s, ts in d.gt.items() for t in ts)
    assert true_found == rep['true_pairs_found']


def test_deterministic_output(tmp_path, data_dir, cfg, artifacts):
    a, b = str(tmp_path / 'a'), str(tmp_path / 'b')
    _run(data_dir, cfg, artifacts, a)
    _run(data_dir, cfg, artifacts, b)
    assert filecmp.cmp(os.path.join(a, 'candidate_pairs.tsv'), os.path.join(b, 'candidate_pairs.tsv'), shallow=False)


def test_empty_lists_allowed(tmp_path, data_dir, cfg):
    from blocker.run import block_and_report
    cfg['pipelines']['classical']['passes'] = {'keys': {'field': 'keys', 'analyzer': 'word', 'top_k': 1}}
    d = prepared(data_dir, cfg)
    out = str(tmp_path / 'o')
    block_and_report(d, cfg, out, None, ['classical'], evaluate=False)
    _, rows = read_tsv_rows(os.path.join(out, 'candidate_pairs.tsv'))
    assert len(rows) == len(d.s1)


@pytest.mark.skipif(not os.path.exists(VALIDATOR), reason='official validator not found')
def test_official_validator(tmp_path, cfg, artifacts):
    test_dir = write_dir(str(tmp_path / 'test'), prefix='test', with_truth=False)
    out = str(tmp_path / 'out')
    from blocker.run import block_and_report
    d = prepare_data(load_dir(test_dir), cfg)
    block_and_report(d, cfg, out, artifacts, None, evaluate=False)
    match = os.path.join(out, 'matching_results.tsv')
    with open(match, 'w') as f:                                    # empty matches: format check only
        f.write('source1_entity_id\tmatched_entity_ids\n' + ''.join(f'{s}\t\n' for s in d.s1_ids))
    res = subprocess.run([sys.executable, VALIDATOR, '--matching', match, '--candidate',
                          os.path.join(out, 'candidate_pairs.tsv'), '--test-dir', test_dir, '--check-ids'],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr


# training loops (tiny, synthetic) ---------------------------------------------------

@pytest.mark.parametrize('kind', ['bert', 'jepa'])
def test_training_loop_smoke(tmp_path, data_dir, cfg, tiny_models, kind):
    from blocker.training.contrastive_training import BertTrainer
    from blocker.training.jepa_training import JEPATrainer
    d = prepared(data_dir, cfg)
    tr, va = split_holdout(d, 0.3, 'mlc26')
    cfg['models'][kind]['epochs'] = 2
    cfg['training']['hard_negatives']['mine_after_epochs'] = [1]
    trainer = (BertTrainer if kind == 'bert' else JEPATrainer)(
        tiny_models(kind), tr, va, cfg['models'][kind], cfg, str(tmp_path / kind), torch.device('cpu'))
    hist = trainer.fit()
    assert len(hist) == 2 and all(np.isfinite(h['loss']) for h in hist)
    assert os.path.exists(tmp_path / kind / 'weights.pt')


def test_hard_negative_mining(data_dir, cfg, tiny_models):
    from blocker.training.hard_negative_mining import mine_hard_negatives
    d = prepared(data_dir, cfg)
    hn = mine_hard_negatives(tiny_models('bert'), d, cfg['training']['hard_negatives'], {}, seed=0, batch_size=8)
    own = d.owner()
    assert hn and all((own[v] != a).all() for a, v in hn.items())


# checker + matcher ------------------------------------------------------------------

def test_check_blocking_and_matcher(tmp_path, data_dir, cfg, artifacts):
    import yaml
    import check_blocking
    import lgbm_matcher
    out = str(tmp_path / 'out')
    _run(data_dir, cfg, artifacts, out)
    res = check_blocking.main(['--candidates', os.path.join(out, 'candidate_pairs.tsv'), '--data-dir', data_dir,
                               '--debug', os.path.join(out, 'debug_candidate_scores')])
    m = res['metrics']
    assert 0 < m['pair_quality'] <= 1 and m['penalized_recall'] < m['pair_recall']
    cfg_path = str(tmp_path / 'cfg.yaml')
    cfg['matcher']['lgb'].update(min_data_in_leaf=1, num_leaves=4)
    cfg['matcher']['num_boost_round'] = 5
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    common = ['--data-dir', data_dir, '--candidates-dir', out, '--artifacts-dir', artifacts, '--config', cfg_path]
    lgbm_matcher.main(['fit'] + common)
    lgbm_matcher.main(['predict'] + common + ['--output-dir', str(tmp_path / 'pred')])
    header, rows = read_tsv_rows(str(tmp_path / 'pred' / 'matching_results.tsv'))
    assert header == ['source1_entity_id', 'matched_entity_ids'] and len(rows) == len(load_dir(data_dir).s1)
    _, crows = read_tsv_rows(os.path.join(out, 'candidate_pairs.tsv'))
    cand = {s: set(x.split(',')) for s, x in crows}
    assert all(set(filter(None, x.split(','))) <= cand[s] for s, x in rows)   # matches subset of candidates
