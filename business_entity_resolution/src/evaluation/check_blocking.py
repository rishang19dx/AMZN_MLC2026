"""
CLI utility to evaluate blocked candidates against ground truth.

Usage:
    python -m business_entity_resolution.src.evaluation.check_blocking \
        --candidates output/candidate_pairs.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv \
        [--provenance output/debug_candidate_scores.tsv] \
        [--output-json output/blocking_metrics.json]
"""

import argparse
import json
import os
import sys
import pandas as pd

from business_entity_resolution.src.evaluation.blocking_metrics import (
    compute_blocking_metrics,
    format_metrics_report,
)


def parse_candidate_tsv(path: str):
    """Load candidate pairs TSV: {source1_entity_id: [candidate_ids]}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Candidate file not found: {path}")

    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header[0] == "source1_entity_id", f"Invalid candidate header: {header}"
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1 = parts[0].strip()
            cands_str = parts[1].strip() if len(parts) > 1 else ""
            cands = [c.strip() for c in cands_str.split(",") if c.strip()]
            mapping[s1] = cands
    return mapping


def parse_ground_truth_tsv(path: str):
    """Load ground truth TSV: {source1_entity_id: [matched_ids]}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header[0] == "source1_entity_id", f"Invalid ground truth header: {header}"
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1 = parts[0].strip()
            matches_str = parts[1].strip() if len(parts) > 1 else ""
            matches = [m.strip() for m in matches_str.split(",") if m.strip()]
            mapping[s1] = matches
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Evaluate Candidate Blocking against Ground Truth")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground_truth.tsv")
    parser.add_argument("--provenance", type=str, default=None, help="Optional debug provenance TSV/parquet")
    parser.add_argument("--candidate-budget", type=float, default=100.0, help="Budget penalty scale K")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")

    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidate_mapping = parse_candidate_tsv(args.candidates)
    print(f"Loaded candidates for {len(candidate_mapping)} Source 1 entities.")

    print(f"Loading ground truth from {args.ground-truth}...")
    gt_mapping = parse_ground_truth_tsv(args.ground-truth)
    print(f"Loaded ground truth for {len(gt_mapping)} Source 1 entities.")

    # Filter GT to entities present in candidate_mapping if candidate_mapping is a validation subset
    relevant_gt = {s1: gt_mapping[s1] for s1 in candidate_mapping.keys() if s1 in gt_mapping}

    prov_df = None
    if args.provenance and os.path.exists(args.provenance):
        print(f"Loading provenance from {args.provenance}...")
        if args.provenance.endswith(".parquet"):
            prov_df = pd.read_parquet(args.provenance)
        else:
            prov_df = pd.read_csv(args.provenance, sep="\t")

    metrics = compute_blocking_metrics(
        candidate_mapping=candidate_mapping,
        ground_truth=relevant_gt,
        provenance_df=prov_df,
        candidate_budget_penalty_k=args.candidate_budget,
    )

    report = format_metrics_report(metrics)
    print("\n" + report)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        # Convert non-serializable types
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool, list, dict)):
                clean_metrics[k] = v
            else:
                clean_metrics[k] = str(v)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(clean_metrics, f, indent=2)
        print(f"Metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
