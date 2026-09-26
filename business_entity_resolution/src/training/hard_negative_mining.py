"""
Hard Negative Mining module for Entity Resolution.

Iteratively mines non-matching records that exhibit high semantic similarity
under the current encoder representations, creating informative training signals.
"""

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

from business_entity_resolution.src.blocking.ann_index import VectorIndex


def mine_hard_negatives(
    encoder,
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth: Dict[str, List[str]],
    top_k_search: int = 20,
    max_hard_negatives_per_anchor: int = 5,
    batch_size: int = 64,
    device: str = "auto",
) -> Dict[str, List[str]]:
    """
    Mine hard negatives:
    1. Embed target candidate pool.
    2. Build vector index.
    3. Query with S1 anchors.
    4. Exclude ground truth matches to identify deceptive non-matches.
    """
    print(f"Mining hard negatives for {len(s1_df)} anchors against {len(target_df)} candidates...")

    target_texts = target_df["combined_text"].tolist()
    target_ids = target_df["entity_id"].tolist()

    target_emb = encoder.encode_texts(
        target_texts, batch_size=batch_size, device=device, show_progress=False
    )
    dim = encoder.embedding_dim

    index = VectorIndex(dim=dim, device=device)
    index.build(target_emb, target_ids)

    s1_texts = s1_df["combined_text"].tolist()
    s1_ids = s1_df["entity_id"].tolist()

    s1_emb = encoder.encode_texts(
        s1_texts, batch_size=batch_size, device=device, show_progress=False
    )

    scores, idxs = index.search(s1_emb, top_k=top_k_search)

    hard_negatives: Dict[str, List[str]] = {}

    for i, s1_id in enumerate(s1_ids):
        true_matches: Set[str] = set(ground_truth.get(s1_id, []))
        mined_for_s1: List[str] = []

        for candidate_idx in idxs[i]:
            cand_id = index.ids[candidate_idx]
            # Must NOT be a true match
            if cand_id not in true_matches and not cand_id.startswith("S1-"):
                mined_for_s1.append(cand_id)
                if len(mined_for_s1) >= max_hard_negatives_per_anchor:
                    break

        if mined_for_s1:
            hard_negatives[s1_id] = mined_for_s1

    print(f"Mined hard negatives for {len(hard_negatives)} anchors.")
    return hard_negatives
