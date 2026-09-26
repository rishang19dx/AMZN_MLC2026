"""
Candidate Union and Prioritization module.

Combines candidates from:
- C_bert (BERT semantic retrieval)
- C_learned (JEPA / second learned representation)
- C_classical (Multi-key inverted indexing + string similarity)

Guarantees:
- Union only (C_final = C_bert ∪ C_learned ∪ C_classical), never intersection.
- Deduplication of candidate IDs per Source 1 entity.
- Full provenance tracking (ranks, similarities, blocking keys, pipeline counts).
- Modular candidate budget prioritization (num_pipelines + normalized similarity).
- Generates compliant output/candidate_pairs.tsv and output/debug_candidate_scores.tsv.
"""

import os
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd


def compute_candidate_priority(
    num_pipelines: int,
    bert_sim: float,
    learned_sim: float,
    classical_score: float,
) -> float:
    """
    Modular prioritization function for candidate pruning:
    Prioritizes candidates confirmed by multiple pipelines first,
    broken down by total normalized retrieval similarity.
    """
    norm_bert = max(0.0, float(bert_sim))
    norm_learned = max(0.0, float(learned_sim))
    norm_classical = max(0.0, float(classical_score))
    
    # Base priority from pipeline consensus
    priority = (float(num_pipelines) * 100.0) + (norm_bert + norm_learned + norm_classical)
    return priority


class CandidateUnion:
    """
    Merges, deduplicates, prioritizes, and exports multi-pipeline candidates.
    """

    def __init__(self, max_candidates_per_entity: Optional[int] = 100):
        self.max_candidates_per_entity = max_candidates_per_entity

    def merge_candidates(
        self,
        all_s1_ids: List[str],
        bert_candidates: Optional[Dict[str, List[Tuple[str, int, float]]]] = None,
        learned_candidates: Optional[Dict[str, List[Tuple[str, int, float]]]] = None,
        classical_candidates: Optional[Dict[str, List[Tuple[str, int, float, str]]]] = None,
    ) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
        """
        Merge candidate lists from all active pipelines.
        
        Args:
            all_s1_ids: complete list of S1 entity IDs that must be present in output
            bert_candidates: {s1_id: [(cand_id, rank, sim)]}
            learned_candidates: {s1_id: [(cand_id, rank, sim)]}
            classical_candidates: {s1_id: [(cand_id, rank, score, blocking_keys)]}
            
        Returns:
            final_mapping: {s1_id: [deduplicated, prioritized candidate IDs]}
            debug_df: DataFrame with full provenance records
        """
        bert_candidates = bert_candidates or {}
        learned_candidates = learned_candidates or {}
        classical_candidates = classical_candidates or {}

        debug_records = []
        final_mapping: Dict[str, List[str]] = {}

        for s1_id in all_s1_ids:
            # Map candidate_id -> dict of properties
            cand_map: Dict[str, Dict[str, Any]] = {}

            # 1. Process BERT candidates
            for cand_id, rank, sim in bert_candidates.get(s1_id, []):
                if cand_id.startswith("S1-"):
                    continue  # Guard against self-match
                if cand_id not in cand_map:
                    cand_map[cand_id] = {
                        "source1_id": s1_id,
                        "candidate_id": cand_id,
                        "bert_rank": rank,
                        "bert_sim": float(sim),
                        "learned_rank": None,
                        "learned_sim": 0.0,
                        "classical_score": 0.0,
                        "blocking_keys": "",
                        "in_bert": True,
                        "in_learned": False,
                        "in_classical": False,
                    }
                else:
                    cand_map[cand_id]["bert_rank"] = rank
                    cand_map[cand_id]["bert_sim"] = float(sim)
                    cand_map[cand_id]["in_bert"] = True

            # 2. Process Learned candidates
            for cand_id, rank, sim in learned_candidates.get(s1_id, []):
                if cand_id.startswith("S1-"):
                    continue
                if cand_id not in cand_map:
                    cand_map[cand_id] = {
                        "source1_id": s1_id,
                        "candidate_id": cand_id,
                        "bert_rank": None,
                        "bert_sim": 0.0,
                        "learned_rank": rank,
                        "learned_sim": float(sim),
                        "classical_score": 0.0,
                        "blocking_keys": "",
                        "in_bert": False,
                        "in_learned": True,
                        "in_classical": False,
                    }
                else:
                    cand_map[cand_id]["learned_rank"] = rank
                    cand_map[cand_id]["learned_sim"] = float(sim)
                    cand_map[cand_id]["in_learned"] = True

            # 3. Process Classical candidates
            for cand_id, rank, score, bkeys in classical_candidates.get(s1_id, []):
                if cand_id.startswith("S1-"):
                    continue
                if cand_id not in cand_map:
                    cand_map[cand_id] = {
                        "source1_id": s1_id,
                        "candidate_id": cand_id,
                        "bert_rank": None,
                        "bert_sim": 0.0,
                        "learned_rank": None,
                        "learned_sim": 0.0,
                        "classical_score": float(score),
                        "blocking_keys": bkeys,
                        "in_bert": False,
                        "in_learned": False,
                        "in_classical": True,
                    }
                else:
                    cand_map[cand_id]["classical_score"] = float(score)
                    cand_map[cand_id]["blocking_keys"] = bkeys
                    cand_map[cand_id]["in_classical"] = True

            if not cand_map:
                final_mapping[s1_id] = []
                continue

            # Compute pipeline counts and priority score
            prioritized_list = []
            for cid, info in cand_map.items():
                num_p = (
                    (1 if info["in_bert"] else 0)
                    + (1 if info["in_learned"] else 0)
                    + (1 if info["in_classical"] else 0)
                )
                info["num_pipelines"] = num_p
                info["priority"] = compute_candidate_priority(
                    num_p,
                    info["bert_sim"],
                    info["learned_sim"],
                    info["classical_score"],
                )
                debug_records.append(info)
                prioritized_list.append((cid, info["priority"]))

            # Sort deterministically: highest priority first, then lexicographical on candidate_id
            prioritized_list.sort(key=lambda x: (-x[1], x[0]))

            # Apply candidate budget if configured
            if self.max_candidates_per_entity is not None and self.max_candidates_per_entity > 0:
                selected_ids = [cid for cid, _ in prioritized_list[: self.max_candidates_per_entity]]
            else:
                selected_ids = [cid for cid, _ in prioritized_list]

            final_mapping[s1_id] = selected_ids

        debug_df = pd.DataFrame(debug_records)
        return final_mapping, debug_df

    @staticmethod
    def export_candidate_pairs(
        candidate_mapping: Dict[str, List[str]],
        output_filepath: str,
    ):
        """
        Write candidate_pairs.tsv in strictly required format:
        source1_entity_id \t candidate_entity_ids (comma-separated, tab-separated).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id, cands in candidate_mapping.items():
                cand_str = ",".join(cands) if cands else ""
                f.write(f"{s1_id}\t{cand_str}\n")

    @staticmethod
    def export_debug_scores(
        debug_df: pd.DataFrame,
        output_filepath: str,
    ):
        """
        Export detailed provenance scores to TSV or Parquet.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        if output_filepath.endswith(".parquet"):
            try:
                debug_df.to_parquet(output_filepath, index=False)
                return
            except Exception:
                # Fallback to TSV
                output_filepath = output_filepath.replace(".parquet", ".tsv")
        debug_df.to_csv(output_filepath, sep="\t", index=False)
