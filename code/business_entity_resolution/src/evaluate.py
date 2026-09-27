"""
Local scorer for Business Entity Resolution.

Reproduces the leaderboard metric: F_0.5 computed per Source 1 entity and
macro-averaged over all S1 entities, singletons included
  - truth empty: 1.0 if the prediction is empty, else 0.0
  - truth non-empty: F_0.5 of predicted vs true IDs (0.0 if nothing correct)

Optionally also scores a candidate_pairs.tsv (blocking quality): pair recall,
reduction ratio, candidate counts, and the F_0.5 *ceiling*, i.e. the score a
perfect matcher would reach given these candidates.

Usage:
    python src/evaluate.py --split local_val \
        --matching ../../output/local_val/matching_results.tsv \
        --candidates ../../output/local_val/candidate_pairs.tsv
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

BETA = 0.5
MATCHING_HEADER = ['source1_entity_id', 'matched_entity_ids']
CANDIDATE_HEADER = ['source1_entity_id', 'candidate_entity_ids']
GT_HEADER = ['source1_entity_id', 'matched_entity_ids']


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def f_beta(pred, truth, beta=BETA):
    """Per-entity F_beta exactly as the leaderboard defines it."""
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_id_lists(path, header):
    """
    Read a two-column TSV of (source1_entity_id, comma-separated IDs).
    Returns (dict s1_id -> set of IDs, list of format issues found).
    """
    lists, issues = {}, []
    dup_rows = dup_ids = bad_prefix = 0
    with open(path, encoding='utf-8') as f:
        got = next(f, '').rstrip('\n').split('\t')
        if got != header:
            issues.append(f'header is {got}, expected {header}')
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            s1_id, _, ids = line.partition('\t')
            items = [x.strip() for x in ids.split(',') if x.strip()]
            unique = set(items)
            dup_ids += len(items) - len(unique)
            bad_prefix += sum(1 for x in unique if not x.startswith(('S2-', 'S3-')))
            if s1_id in lists:
                dup_rows += 1
                lists[s1_id] |= unique
            else:
                lists[s1_id] = unique
    if dup_rows:
        issues.append(f'{dup_rows} duplicate source1_entity_id rows (merged)')
    if dup_ids:
        issues.append(f'{dup_ids} duplicate IDs inside ID lists')
    if bad_prefix:
        issues.append(f'{bad_prefix} IDs without an S2-/S3- prefix')
    return lists, issues


def read_ground_truth(path):
    gt, issues = read_id_lists(path, GT_HEADER)
    if issues:
        raise ValueError(f'{path}: ' + '; '.join(issues))
    return gt


def _open_source(path):
    """Source TSVs may be gzipped (config.split_paths falls back to .tsv.gz)."""
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, encoding='utf-8')


def read_country_map(path):
    """entity_id -> country for one source file (streams; no pandas needed)."""
    out = {}
    with _open_source(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            out[parts[0]] = parts[3] if len(parts) > 3 else ''
    return out


def count_countries(*paths):
    counts = Counter()
    for path in paths:
        with _open_source(path) as f:
            next(f)
            for line in f:
                counts[line.rstrip('\n').rsplit('\t', 1)[-1]] += 1
    return counts


def align(gt, preds, name):
    """Warn about S1 rows that are missing or unknown; missing rows count as empty."""
    missing = len(gt.keys() - preds.keys())
    extra = len(preds.keys() - gt.keys())
    notes = []
    if missing:
        notes.append(f'{name}: {missing} S1 entities missing (scored as empty; the leaderboard would reject this)')
    if extra:
        notes.append(f'{name}: {extra} S1 IDs not in ground truth (ignored)')
    return notes


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def _percentile(sorted_xs, q):
    if not sorted_xs:
        return float('nan')
    return sorted_xs[min(len(sorted_xs) - 1, int(q * len(sorted_xs)))]


def score_matching(gt, preds, countries=None):
    """
    Returns a dict of metrics, overall and per country. Besides the headline
    F_0.5 it reports where the loss comes from: singletons wrongly matched,
    precision and recall on entities that do have matches.
    """
    groups = defaultdict(list)
    for s1_id, truth in gt.items():
        pred = preds.get(s1_id, set())
        country = countries.get(s1_id, '?') if countries else 'all'
        groups['ALL'].append((pred, truth))
        if countries:
            groups[country].append((pred, truth))

    report = {}
    for group, rows in groups.items():
        single = [(p, t) for p, t in rows if not t]
        multi = [(p, t) for p, t in rows if t]
        tp = sum(len(p & t) for p, t in rows)
        n_pred = sum(len(p) for p, _ in rows)
        n_true = sum(len(t) for _, t in rows)
        report[group] = {
            'entities': len(rows),
            'f05': _mean([f_beta(p, t) for p, t in rows]),
            'singletons': len(single),
            'singleton_acc': _mean([float(not p) for p, _ in single]),
            'f05_non_singleton': _mean([f_beta(p, t) for p, t in multi]),
            'macro_precision': _mean([len(p & t) / len(p) for p, t in multi if p]),
            'macro_recall': _mean([len(p & t) / len(t) for p, t in multi]),
            'pair_precision': tp / n_pred if n_pred else float('nan'),
            'pair_recall': tp / n_true if n_true else float('nan'),
            'false_pos_pairs': n_pred - tp,
            'false_neg_pairs': n_true - tp,
            'entities_with_fp': sum(1 for p, t in rows if p - t),
        }
    return report


def score_candidates(gt, cands, countries=None, target_counts=None):
    """
    Blocking quality. f05_ceiling = score of an oracle matcher that keeps
    exactly the true IDs among the candidates (and predicts empty for
    singletons), i.e. the best any matcher can do on this candidate set.
    """
    groups = defaultdict(list)
    for s1_id, truth in gt.items():
        cand = cands.get(s1_id, set())
        country = countries.get(s1_id, '?') if countries else 'all'
        groups['ALL'].append((cand, truth))
        if countries:
            groups[country].append((cand, truth))

    report = {}
    for group, rows in groups.items():
        sizes = sorted(len(c) for c, _ in rows)
        found = sum(len(c & t) for c, t in rows)
        n_true = sum(len(t) for _, t in rows)
        multi = [(c, t) for c, t in rows if t]
        r = {
            'entities': len(rows),
            'pairs': sum(sizes),
            'cands_mean': _mean(sizes),
            'cands_p50': _percentile(sizes, 0.50),
            'cands_p99': _percentile(sizes, 0.99),
            'cands_max': sizes[-1] if sizes else 0,
            'pair_recall': found / n_true if n_true else float('nan'),
            'entities_full_recall': _mean([float(t <= c) for c, t in multi]),
            'f05_ceiling': _mean([f_beta(c & t, t) for c, t in rows]),
        }
        # Reduction ratio against the within-country Cartesian product (true
        # matches never cross countries, so that is the natural baseline).
        if target_counts is not None:
            if group == 'ALL':
                s1_by_country = Counter(countries.values()) if countries else None
                full = sum(n * target_counts.get(c, 0) for c, n in s1_by_country.items()) if s1_by_country else None
            else:
                full = len(rows) * target_counts.get(group, 0)
            if full:
                r['reduction_ratio'] = 1 - r['pairs'] / full
        report[group] = r
    return report


def print_report(title, report, keys):
    print(f'\n== {title}')
    groups = ['ALL'] + sorted(g for g in report if g != 'ALL')
    print(f"{'':24}" + ''.join(f'{g:>14}' for g in groups))
    for k in keys:
        row = f'{k:24}'
        for g in groups:
            v = report[g].get(k, float('nan'))
            # the reduction ratio is always ~0.9999..., so show enough digits to compare runs
            row += f'{v:>14,}' if isinstance(v, int) else f'{v:>14.7f}' if k == 'reduction_ratio' else f'{v:>14.4f}'
        print(row)


MATCHING_KEYS = ['entities', 'f05', 'singletons', 'singleton_acc', 'f05_non_singleton',
                 'macro_precision', 'macro_recall', 'pair_precision', 'pair_recall',
                 'false_pos_pairs', 'false_neg_pairs', 'entities_with_fp']
CANDIDATE_KEYS = ['entities', 'pairs', 'cands_mean', 'cands_p50', 'cands_p99', 'cands_max',
                  'pair_recall', 'entities_full_recall', 'f05_ceiling', 'reduction_ratio']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='local_val', choices=[s for s in config.SPLIT_NAMES if s != 'test'],
                    help='which ground truth / source files to score against')
    ap.add_argument('--matching', help='matching_results.tsv to score')
    ap.add_argument('--candidates', help='candidate_pairs.tsv to score (blocking quality)')
    ap.add_argument('--json', help='also write the metrics to this JSON file')
    args = ap.parse_args()
    if not args.matching and not args.candidates:
        ap.error('pass --matching and/or --candidates')

    paths = config.split_paths(args.split)
    gt = read_ground_truth(paths['gt'])
    countries = read_country_map(paths['s1'])
    out, notes = {'split': args.split}, []

    if args.matching:
        preds, issues = read_id_lists(args.matching, MATCHING_HEADER)
        notes += [f'matching: {i}' for i in issues] + align(gt, preds, 'matching')
        out['matching'] = score_matching(gt, preds, countries)
        print_report(f'Matching ({args.split}): {args.matching}', out['matching'], MATCHING_KEYS)

    if args.candidates:
        cands, issues = read_id_lists(args.candidates, CANDIDATE_HEADER)
        notes += [f'candidates: {i}' for i in issues] + align(gt, cands, 'candidates')
        target_counts = count_countries(paths['s2'], paths['s3'])
        out['candidates'] = score_candidates(gt, cands, countries, target_counts)
        print_report(f'Candidates ({args.split}): {args.candidates}', out['candidates'], CANDIDATE_KEYS)
        if args.matching:
            outside = sum(len(p - cands.get(s, set())) for s, p in preds.items())
            if outside:
                notes.append(f'{outside} matched IDs are not in the candidate set (pipeline bug)')

    for n in notes:
        print(f'WARNING: {n}')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == '__main__':
    main()
