"""
Candidate generation (inference): run the multi-pipeline blocker on one data
directory and write

  <output-dir>/candidate_pairs.tsv            challenge format, one row per Source 1
  <output-dir>/debug_candidate_scores/        per-pair provenance (parquet parts)
  <output-dir>/candidate_manifest.json        row-index contract + build stats
  <output-dir>/blocking_report.{txt,json}     only when the directory has ground truth

Usage (from code/business_entity_resolution/):
  # test set, all pipelines trained by train_blocker.py
  python src/generate_candidates.py --data-dir ../../dataset/test --artifacts-dir ../../artifacts \
      --output-dir ../../output/test

  # labelled holdout, with recall-vs-K sweep (report written next to the candidates)
  python src/generate_candidates.py --data-dir ../../dataset/splits/local_val \
      --artifacts-dir ../../artifacts --output-dir ../../output/local_val --k-sweep

  # classical pipeline only (no trained models needed)
  python src/generate_candidates.py --data-dir ... --output-dir ... --pipelines classical
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocker  # noqa: E402,F401  (sets macOS/OpenMP environment first)
from blocker.config import add_config_args, load_config, save_config
from blocker.run import block_and_report, load_prepared, use_translit
from blocker.utils import apply_runtime, log, set_seed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', '--test-dir', dest='data_dir', required=True,
                    help='directory with *_source{1,2,3}.tsv (and optionally *_ground_truth.tsv)')
    ap.add_argument('--artifacts-dir', default=None, help='output of train_blocker.py (models, translit.json)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--pipelines', default=None,
                    help='comma list of classical,bert,jepa (default: every enabled pipeline with a trained model)')
    ap.add_argument('--subsample', type=float, default=1.0,
                    help='entity-level random fraction of Source 1 (+ its targets) for quick runs')
    ap.add_argument('--translit', default=None, help='explicit transliteration dictionary JSON')
    ap.add_argument('--k-sweep', action='store_true', help='labelled data: also report recall vs K')
    ap.add_argument('--no-eval', action='store_true')
    ap.add_argument('--no-debug', action='store_true', help='skip the provenance parquet')
    add_config_args(ap)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    apply_runtime(cfg)
    set_seed(int(cfg.get('seed', 42)))
    os.makedirs(args.output_dir, exist_ok=True)
    save_config(cfg, os.path.join(args.output_dir, 'blocking_config.resolved.yaml'))
    use_translit(args.artifacts_dir, args.translit)
    data = load_prepared(args.data_dir, cfg, args.subsample, int(cfg['data'].get('subset_seed', 7)))
    pipelines = [p.strip() for p in args.pipelines.split(',')] if args.pipelines else None
    block_and_report(data, cfg, args.output_dir, args.artifacts_dir, pipelines,
                     evaluate=not args.no_eval, sweep=args.k_sweep, debug=not args.no_debug)
    log('done')


if __name__ == '__main__':
    main()
