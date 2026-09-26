"""
High-level steps shared by the CLIs: load + normalise a directory, run the
blocker on it, evaluate it when labels exist.
"""

import os

from blocker.data import load_dir, subsample
from blocker.evaluation import blocking_metrics as bm
from blocker.normalization import prepare_data, set_translit_dict
from blocker.pipelines.engine import build_candidates, enabled_pipelines, run_retrievals, truncate_to_config
from blocker.utils import log, n_workers, read_json, write_json

# Documented for the model/license audit (verified on huggingface.co, Sep 2026).
KNOWN_MODELS = {
    'intfloat/multilingual-e5-small': {'license': 'MIT', 'params': 117_654_272,
                                       'source': 'https://huggingface.co/intfloat/multilingual-e5-small'},
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': {
        'license': 'Apache-2.0', 'params': 117_654_272,
        'source': 'https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'},
}


def use_translit(artifacts_dir=None, explicit=None):
    """Select the native-script dictionary: explicit path > artifacts/translit.json >
    src/normalize.py's default ($BER_CACHE_DIR/translit.json, if present)."""
    path = explicit or (os.path.join(artifacts_dir, 'translit.json') if artifacts_dir else None)
    if path and os.path.exists(path):
        words = read_json(path).get('words', {})
        set_translit_dict(words)
        log(f'transliteration dictionary: {len(words):,} words from {path}')
    elif explicit:
        raise FileNotFoundError(explicit)
    else:
        log('transliteration dictionary: none in artifacts; using src/normalize.py default (if any)')


def load_prepared(directory, cfg, fraction=1.0, seed=0, with_truth=True):
    data = load_dir(directory, with_truth=with_truth)
    if fraction < 1.0:
        data = subsample(data, fraction, seed)
    log(f'loaded {data.summary()}')
    prepare_data(data, cfg, n_workers(cfg['normalization'].get('workers', 0)))
    log('normalised')
    return data


def block_and_report(data, cfg, out_dir, artifacts_dir=None, pipelines=None, evaluate=True, sweep=False,
                     debug=True):
    """Run the blocker on prepared data; write outputs (+ reports when labelled)."""
    os.makedirs(out_dir, exist_ok=True)
    pipes = enabled_pipelines(cfg, pipelines, artifacts_dir)
    if not pipes:
        raise SystemExit('no pipeline enabled / available')
    cache_dir = cfg['runtime'].get('cache_dir') or os.path.join(out_dir, 'cache')
    vcfg = cfg.get('validation', {})
    k_max = max(vcfg.get('k_sweep') or [0]) if sweep and data.has_truth else None
    rets = run_retrievals(data, cfg, pipes, artifacts_dir, cache_dir, k_override=k_max)
    rets_op = truncate_to_config(rets, cfg, data) if k_max else rets
    s1, tg, _, stats = build_candidates(data, rets_op, cfg, out_dir, debug=debug)
    if not (evaluate and data.has_truth):
        return stats
    lam, ref = float(vcfg.get('penalty_lambda', 0.1)), int(vcfg.get('penalty_ref_budget', 100))
    rep = bm.evaluate_candidates(data, s1, tg, lam, ref)
    contrib = bm.pipeline_contributions(data, rets_op)
    ks = bm.k_sweep(data, rets, vcfg.get('k_sweep', [])) if k_max else None
    bs = bm.budget_sweep(data, rets_op, vcfg.get('budget_sweep', []),
                         cfg['candidate_generation'].get('prioritizer', 'votes_rank'))
    text = bm.format_report(rep, contrib, ks, bs, title=f'BLOCKING EVALUATION ({data.name}; {", ".join(pipes)})')
    print('\n' + text + '\n', flush=True)
    with open(os.path.join(out_dir, 'blocking_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    write_json({'metrics': rep, 'pipelines': contrib, 'k_sweep': ks, 'budget_sweep': bs, 'build': stats},
               os.path.join(out_dir, 'blocking_report.json'))
    return rep
