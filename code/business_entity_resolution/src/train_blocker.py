"""
Train the learned blocking representations (Pipeline A: BERT bi-encoder,
Pipeline B: JEPA-style encoder) and validate the full multi-pipeline blocker.

Data handling (no leakage):
  * validation = --val-dir, or an entity-level closed-universe holdout of
    --train-dir (same md5 hash as src/data_loader.py, so the default holdout is
    exactly splits/local_val); its entities and their targets never produce
    training pairs, and the transliteration dictionary is learned without them;
  * --train-fraction F trains on a random F of the remaining Source 1 entities
    (with their targets and the same share of distractors) - for fast runs.

Artifacts written to --output-dir:
  translit.json                     native-script dictionary (training pairs only)
  bert/, jepa/                      model config, tokenizer, weights, meta, training history
  config.resolved.yaml              the exact configuration used
  model_card.json                   model names, parameter counts, licences, sources
  validation/                       candidates + blocking_report.{txt,json} on the holdout

Usage (from code/business_entity_resolution/):
  python src/train_blocker.py --train-dir ../../../student_resource/dataset/train \
      --output-dir ../../artifacts --train-fraction 0.2
  python src/train_blocker.py ... --pipelines bert          # one pipeline
  python src/train_blocker.py ... --eval-only               # re-validate saved artifacts
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocker  # noqa: E402,F401  (sets macOS/OpenMP environment first)
from blocker.config import add_config_args, load_config, save_config
from blocker.data import load_dir, split_holdout, subsample
from blocker.normalization import learn_translit_dict, prepare_data, set_translit_dict
from blocker.run import KNOWN_MODELS, block_and_report, use_translit
from blocker.utils import log, n_workers, read_json, resolve_device, set_seed, write_json


def model_card(cfg, out_dir):
    card = {}
    for kind in ('bert', 'jepa'):
        meta_path = os.path.join(out_dir, kind, 'meta.json')
        if not os.path.exists(meta_path):
            continue
        meta = read_json(meta_path)
        name = meta['config']['model_name']
        card[kind] = {'base_model': name, 'trained_params': meta.get('n_params'),
                      'limit': cfg['models'].get('max_params'), **KNOWN_MODELS.get(name, {'license': 'CHECK'})}
    write_json(card, os.path.join(out_dir, 'model_card.json'))
    return card


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-dir', required=True, help='labelled directory (*_source{1,2,3}.tsv + *_ground_truth.tsv)')
    ap.add_argument('--val-dir', default=None, help='separate labelled validation directory (else a holdout of --train-dir)')
    ap.add_argument('--output-dir', required=True, help='artifacts directory')
    ap.add_argument('--pipelines', default='bert,jepa', help='which learned pipelines to train')
    ap.add_argument('--train-fraction', type=float, default=None,
                    help='random entity-level fraction of the training part to train on (overrides data.train_fraction)')
    ap.add_argument('--val-fraction', type=float, default=None, help='holdout fraction (overrides data.val_fraction)')
    ap.add_argument('--val-subsample', type=float, default=None,
                    help='fraction of validation entities for the final blocking report (overrides data.val_subsample)')
    ap.add_argument('--epochs', type=int, default=None, help='override epochs for every trained model')
    ap.add_argument('--device', default=None, help='cuda | mps | cpu (overrides runtime.device)')
    ap.add_argument('--no-eval', action='store_true', help='skip the final validation blocking run')
    ap.add_argument('--eval-only', action='store_true', help='no training: validate existing artifacts')
    add_config_args(ap)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    d = cfg['data']
    if args.train_fraction is not None:
        d['train_fraction'] = args.train_fraction
    if args.val_fraction is not None:
        d['val_fraction'] = args.val_fraction
    if args.val_subsample is not None:
        d['val_subsample'] = args.val_subsample
    if args.device:
        cfg['runtime']['device'] = args.device
    pipes = [p.strip() for p in args.pipelines.split(',') if p.strip()]
    for p in pipes:
        if p not in ('bert', 'jepa'):
            ap.error(f'--pipelines: {p!r} is not a learned pipeline (bert, jepa)')
        if args.epochs is not None:
            cfg['models'][p]['epochs'] = args.epochs
    seed = int(cfg.get('seed', 42))
    set_seed(seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(cfg, os.path.join(args.output_dir, 'config.resolved.yaml'))

    # -- data ---------------------------------------------------------------
    full = load_dir(args.train_dir)
    if not full.has_truth:
        raise SystemExit(f'{args.train_dir}: no ground truth file')
    if args.val_dir:
        train, val = full, load_dir(args.val_dir)
    else:
        train, val = split_holdout(full, float(d['val_fraction']), str(d['split_seed']))
    del full
    log(f'train part {train.summary()}')
    log(f'validation {val.summary()}')

    workers = n_workers(cfg['normalization'].get('workers', 0))
    if args.eval_only:
        use_translit(args.output_dir)
    elif cfg['normalization'].get('learn_translit', True):
        words = learn_translit_dict(train, int(cfg['normalization']['translit_min_count']),
                                    float(cfg['normalization']['translit_min_share']))
        write_json({'built_from': train.name, 'words': words}, os.path.join(args.output_dir, 'translit.json'))
        set_translit_dict(words)
    prepare_data(val, cfg, workers)

    if not args.eval_only:
        fit = subsample(train, float(d['train_fraction']), int(d['subset_seed']))
        log(f'training on {fit.summary()} (train_fraction={d["train_fraction"]})')
        prepare_data(fit, cfg, workers)
        device = resolve_device(cfg['runtime'].get('device', 'auto'))
        for p in pipes:
            if not cfg['models'][p].get('enabled', True):
                log(f'{p}: disabled in config, skipped')
                continue
            out = os.path.join(args.output_dir, p)
            log(f'=== training pipeline {p} -> {out}')
            if p == 'bert':
                from blocker.training.contrastive_training import train_bert
                train_bert(fit, val, cfg, out, device)
            else:
                from blocker.training.jepa_training import train_jepa
                train_jepa(fit, val, cfg, out, device)
        del fit
        card = model_card(cfg, args.output_dir)
        log(f'model card: {card}')

    # -- validation of the whole blocker ----------------------------------------
    if cfg['validation'].get('enabled', True) and not args.no_eval:
        v = subsample(val, float(d.get('val_subsample', 1.0)), seed, name=f'{val.name}-eval')
        block_and_report(v, cfg, os.path.join(args.output_dir, 'validation'), args.output_dir,
                         pipelines=None, evaluate=True, sweep=True)
    log('done')


if __name__ == '__main__':
    main()
