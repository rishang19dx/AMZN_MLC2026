"""
Check a blocking strategy against ground truth: of the blocked pairs, how many
are real matches, how many real matches were kept, and a penalised score that
charges for every wasted candidate.

Works on any candidate_pairs.tsv (this blocker's, blocking.py v1's, a teammate's).
With --debug (the debug_candidate_scores/ directory written next to it) it also
splits recall and pair quality by pipeline.

Metrics (src/blocker/evaluation/blocking_metrics.py):
  pair_quality (PQ)        true pairs in candidates / candidate pairs       <- "how many were in ground truth"
  pair_recall (PC)         true pairs in candidates / true pairs
  penalized_recall         PC - lambda * false_candidates_per_S1 / ref_budget
  macro_penalized_recall   mean over S1 of recall_i - lambda * fp_i / ref_budget
  f1_pc_pq, h_pc_rr, f05_ceiling, reduction ratio, singleton pollution, ...

Usage:
  python src/check_blocking.py --candidates ../../output/local_val/candidate_pairs.tsv \
      --data-dir ../../../student_resource/dataset/splits/local_val [--debug ../../output/local_val/debug_candidate_scores]
      [--lambda 0.1 --ref-budget 100] [--json report.json]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocker  # noqa: F401
from blocker.data import load_dir
from blocker.evaluation import blocking_metrics as bm
from blocker.normalization import normalize_country
from blocker.utils import write_json
from evaluate import CANDIDATE_HEADER, read_id_lists


def candidate_rows(path, data):
    lists, issues = read_id_lists(path, CANDIDATE_HEADER)
    s1_pos = pd.Index(data.s1_ids)
    tg_pos = pd.Index(data.tg_ids)
    a, b = [], []
    for s1_id, ids in lists.items():
        a.extend([s1_id] * len(ids))
        b.extend(ids)
    i = s1_pos.get_indexer(pd.Index(a)) if a else np.empty(0, np.int64)
    j = tg_pos.get_indexer(pd.Index(b)) if b else np.empty(0, np.int64)
    missing_s1 = len(set(data.s1_ids) - set(lists))
    unknown = int(((i < 0) | (j < 0)).sum())
    if missing_s1:
        issues.append(f'{missing_s1:,} Source 1 entities have no row')
    if unknown:
        issues.append(f'{unknown:,} candidate ids (or Source 1 ids) do not exist in {data.name}')
    ok = (i >= 0) & (j >= 0)
    return i[ok], j[ok], issues


def by_pipeline(debug_dir, data):
    import duckdb
    parts = os.path.join(debug_dir, 'part-*.parquet')
    cols = [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{parts}')").fetchall()]
    fams = [c for c in cols if c.startswith('hit_')]
    d = duckdb.sql(f"SELECT s1_idx, tg_idx, n_pipelines, {', '.join(fams)} FROM read_parquet('{parts}')").fetchnumpy()
    own = data.owner()
    true = own[d['tg_idx']] == d['s1_idx']
    n_true = len(data.positive_pairs()[0])
    rows = []
    for f in fams:
        m = np.asarray(d[f], bool)
        rows.append((f[4:], int(m.sum()), float(true[m].mean()) if m.any() else 0.0, true[m].sum() / max(n_true, 1)))
    for k in (1, 2, 3):
        m = np.asarray(d['n_pipelines']) == k
        rows.append((f'found by exactly {k}', int(m.sum()), float(true[m].mean()) if m.any() else 0.0,
                     true[m].sum() / max(n_true, 1)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--data-dir', required=True, help='labelled directory the candidates were generated for')
    ap.add_argument('--debug', default=None, help='debug_candidate_scores/ directory for a per-pipeline split')
    ap.add_argument('--lambda', dest='lam', type=float, default=0.1, help='penalty per ref-budget false candidates')
    ap.add_argument('--ref-budget', type=int, default=100)
    ap.add_argument('--json', default=None)
    args = ap.parse_args(argv)

    data = load_dir(args.data_dir)
    if not data.has_truth:
        raise SystemExit(f'{args.data_dir}: no ground truth')
    data.s1['country_n'] = data.s1['country'].map(normalize_country)
    data.tg['country_n'] = data.tg['country'].map(normalize_country)
    i, j, issues = candidate_rows(args.candidates, data)
    rep = bm.evaluate_candidates(data, i, j, args.lam, args.ref_budget)
    print(bm.format_report(rep, title=f'BLOCKING CHECK: {args.candidates}'))
    out = {'metrics': rep, 'issues': issues}
    if args.debug:
        rows = by_pipeline(args.debug, data)
        print('\nBy pipeline (pairs in the final candidate set):')
        print(f"  {'pipeline':<22}{'pairs':>14}{'PQ (share true)':>18}{'share of all true':>20}")
        for name, n, pq, share in rows:
            print(f'  {name:<22}{n:>14,}{100 * pq:>17.2f}%{100 * share:>19.2f}%')
        out['by_pipeline'] = [dict(zip(('pipeline', 'pairs', 'pair_quality', 'recall_share'), r)) for r in rows]
    for m in issues:
        print(f'WARNING: {m}')
    if args.json:
        write_json(out, args.json)
    return out


if __name__ == '__main__':
    main()
