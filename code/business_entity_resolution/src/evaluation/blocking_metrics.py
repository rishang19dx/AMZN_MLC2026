"""
Evaluation metrics module for candidate generation / blocking.

Computes:
- Candidate Recall
- Missed true matches
- Candidate Precision and wasted candidate count
- Candidate volume distribution (mean, median, p95, max)
- Zero-candidate entity counts
- Reduction ratio over full Cartesian space
- Composite recall-vs-candidate penalty diagnostic
- Pipeline complementarity breakdown (Venn partition)
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd


def compute_blocking_metrics(
    candidate_mapping: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    num_total_targets: Optional[int] = None,
    provenance_df: Optional[pd.DataFrame] = None,
    candidate_budget_penalty_k: float = 100.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive blocking metrics against ground truth.
    
    Args:
        candidate_mapping: {s1_id: [cand_id_1, ...]}
        ground_truth: {s1_id: [true_match_1, ...]}
        num_total_targets: total |S2| + |S3| for reduction ratio calculation
        provenance_df: optional debug DataFrame containing pipeline source flags
        candidate_budget_penalty_k: scale factor for candidate size penalty
        
    Returns:
        dict of metric names to values
    """
    # 1. Build pair sets
    gt_pairs: Set[Tuple[str, str]] = set()
    for s1, matches in ground_truth.items():
        for m in matches:
            gt_pairs.add((s1, m))

    cand_pairs: Set[Tuple[str, str]] = set()
    cand_counts_per_entity: List[int] = []
    zero_candidate_s1: List[str] = []

    for s1 in ground_truth.keys():
        cands = candidate_mapping.get(s1, [])
        cand_counts_per_entity.append(len(cands))
        if len(cands) == 0:
            zero_candidate_s1.append(s1)
        for c in cands:
            cand_pairs.add((s1, c))

    # 2. Recall and Precision
    recovered_pairs = gt_pairs.intersection(cand_pairs)
    missed_pairs = gt_pairs - cand_pairs

    total_gt = len(gt_pairs)
    recovered_count = len(recovered_pairs)
    missed_count = len(missed_pairs)
    total_candidates = len(cand_pairs)

    recall = (recovered_count / total_gt) if total_gt > 0 else 1.0
    precision = (recovered_count / total_candidates) if total_candidates > 0 else 0.0
    wasted_candidates = total_candidates - recovered_count

    # 3. Distribution of candidate counts per S1 entity
    counts_arr = np.array(cand_counts_per_entity) if cand_counts_per_entity else np.array([0])
    avg_cands = float(np.mean(counts_arr))
    median_cands = float(np.median(counts_arr))
    p95_cands = float(np.percentile(counts_arr, 95))
    max_cands = int(np.max(counts_arr))
    zero_cand_count = len(zero_candidate_s1)
    zero_cand_pct = (zero_cand_count / len(ground_truth)) * 100.0 if ground_truth else 0.0

    # Check how many ground truth matches were lost due to zero-candidate entities
    lost_on_zero = sum(1 for s1, _ in missed_pairs if s1 in set(zero_candidate_s1))

    # 4. Reduction Ratio
    if num_total_targets is not None and num_total_targets > 0 and len(ground_truth) > 0:
        cartesian_space = float(len(ground_truth)) * float(num_total_targets)
        reduction_ratio = 1.0 - (float(total_candidates) / cartesian_space)
    else:
        reduction_ratio = 0.0

    # 5. Composite Recall-vs-Candidate Penalty Diagnostic
    # Penalizes bloated candidate sizes: Diagnostic = Recall * exp(-avg_candidates / candidate_budget_penalty_k)
    # Alternatively: candidate-weighted F_0.05
    size_penalty_factor = max(0.0, 1.0 - (avg_cands / (2.0 * max(1.0, candidate_budget_penalty_k))))
    composite_diagnostic = recall * size_penalty_factor

    metrics: Dict[str, Any] = {
        "total_true_matches": total_gt,
        "true_matches_recovered": recovered_count,
        "candidate_recall": recall,
        "missed_true_matches": missed_count,
        "total_candidate_pairs": total_candidates,
        "wasted_candidates": wasted_candidates,
        "candidate_precision": precision,
        "average_candidates_per_entity": avg_cands,
        "median_candidates_per_entity": median_cands,
        "p95_candidates_per_entity": p95_cands,
        "max_candidates_per_entity": max_cands,
        "zero_candidate_entities": zero_cand_count,
        "zero_candidate_percentage": zero_cand_pct,
        "true_matches_lost_to_zero_candidates": lost_on_zero,
        "reduction_ratio": reduction_ratio,
        "composite_diagnostic_score": composite_diagnostic,
    }

    # 6. Pipeline Complementarity (if provenance is available)
    if provenance_df is not None and len(provenance_df) > 0:
        comp = analyze_pipeline_complementarity(provenance_df, gt_pairs)
        metrics["pipeline_complementarity"] = comp

    return metrics


def analyze_pipeline_complementarity(
    provenance_df: pd.DataFrame,
    gt_pairs: Set[Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Computes Venn breakdown of candidate retrieval and recall across pipelines.
    """
    # Create unique pair key
    provenance_df["_pair_key"] = list(zip(provenance_df["source1_id"], provenance_df["candidate_id"]))
    provenance_df["_is_true_match"] = [k in gt_pairs for k in provenance_df["_pair_key"]]

    b = provenance_df["in_bert"].astype(bool)
    l = provenance_df["in_learned"].astype(bool)
    c = provenance_df["in_classical"].astype(bool)

    categories = {
        "bert_only": b & (~l) & (~c),
        "learned_only": (~b) & l & (~c),
        "classical_only": (~b) & (~l) & c,
        "bert_plus_learned": b & l & (~c),
        "bert_plus_classical": b & (~l) & c,
        "learned_plus_classical": (~b) & l & c,
        "all_three": b & l & c,
    }

    breakdown = {}
    for cat_name, mask in categories.items():
        subset = provenance_df[mask]
        cand_count = len(subset)
        true_matches = subset["_is_true_match"].sum()
        breakdown[cat_name] = {
            "candidate_count": int(cand_count),
            "true_matches_captured": int(true_matches),
        }

    return breakdown


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary into a readable markdown table."""
    lines = [
        "### Blocking Evaluation Report",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total True Matches** | {metrics['total_true_matches']} |",
        f"| **True Matches Recovered** | {metrics['true_matches_recovered']} |",
        f"| **Candidate Recall** | {metrics['candidate_recall']:.4f} ({metrics['candidate_recall']*100:.2f}%) |",
        f"| **Missed True Matches** | {metrics['missed_true_matches']} |",
        f"| **Total Blocked Pairs** | {metrics['total_candidate_pairs']} |",
        f"| **Wasted (Non-Match) Pairs** | {metrics['wasted_candidates']} |",
        f"| **Candidate Precision** | {metrics['candidate_precision']:.6f} |",
        f"| **Average Candidates / S1** | {metrics['average_candidates_per_entity']:.2f} |",
        f"| **Median Candidates / S1** | {metrics['median_candidates_per_entity']:.1f} |",
        f"| **P95 Candidates / S1** | {metrics['p95_candidates_per_entity']:.1f} |",
        f"| **Max Candidates / S1** | {metrics['max_candidates_per_entity']} |",
        f"| **Zero-Candidate Entities** | {metrics['zero_candidate_entities']} ({metrics['zero_candidate_percentage']:.2f}%) |",
        f"| **Reduction Ratio** | {metrics['reduction_ratio']:.6f} |",
        f"| **Composite Diagnostic Score** | {metrics['composite_diagnostic_score']:.4f} |",
        "",
    ]

    if "pipeline_complementarity" in metrics:
        lines.extend([
            "### Pipeline Complementarity Breakdown",
            "",
            "| Pipeline Combination | Candidates Generated | True Matches Captured |",
            "| :--- | :--- | :--- |",
        ])
        for cat, data in metrics["pipeline_complementarity"].items():
            lines.append(f"| {cat} | {data['candidate_count']} | {data['true_matches_captured']} |")
        lines.append("")

    return "\n".join(lines)
