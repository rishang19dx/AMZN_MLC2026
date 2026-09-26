"""
BERT Semantic Blocker module.

Generates dense embeddings for S1, S2, and S3 records and performs top-K
cosine similarity search via VectorIndex (FAISS or GPU/CPU chunked matrix multiplication).
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from business_entity_resolution.src.blocking.ann_index import VectorIndex
from business_entity_resolution.src.models.bert_encoder import BERTDualEncoder


class BERTBlocker:
    """
    BERT-based semantic blocker for Source 1 vs Source 2 and Source 3.
    """

    def __init__(
        self,
        encoder: BERTDualEncoder,
        top_k_source2: int = 30,
        top_k_source3: int = 30,
        batch_size: int = 64,
        device: str = "auto",
        use_faiss: bool = True,
    ):
        self.encoder = encoder
        self.top_k_source2 = top_k_source2
        self.top_k_source3 = top_k_source3
        self.batch_size = batch_size
        self.device = device
        self.use_faiss = use_faiss

        self.index_s2: Optional[VectorIndex] = None
        self.index_s3: Optional[VectorIndex] = None

    def fit_targets(
        self,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        show_progress: bool = True,
    ) -> "BERTBlocker":
        """
        Encode S2 and S3 records and index them in separate VectorIndex instances.
        """
        dim = self.encoder.embedding_dim

        # 1. Index Source 2
        if len(s2_df) > 0:
            texts_s2 = s2_df["combined_text"].tolist()
            ids_s2 = s2_df["entity_id"].tolist()
            emb_s2 = self.encoder.encode_texts(
                texts_s2,
                batch_size=self.batch_size,
                device=self.device,
                show_progress=show_progress,
            )
            self.index_s2 = VectorIndex(
                dim=dim, use_faiss=self.use_faiss, device=self.device
            )
            self.index_s2.build(emb_s2, ids_s2)
        else:
            self.index_s2 = None

        # 2. Index Source 3
        if len(s3_df) > 0:
            texts_s3 = s3_df["combined_text"].tolist()
            ids_s3 = s3_df["entity_id"].tolist()
            emb_s3 = self.encoder.encode_texts(
                texts_s3,
                batch_size=self.batch_size,
                device=self.device,
                show_progress=show_progress,
            )
            self.index_s3 = VectorIndex(
                dim=dim, use_faiss=self.use_faiss, device=self.device
            )
            self.index_s3.build(emb_s3, ids_s3)
        else:
            self.index_s3 = None

        return self

    def retrieve_candidates(
        self,
        s1_df: pd.DataFrame,
        show_progress: bool = True,
    ) -> Dict[str, List[Tuple[str, int, float]]]:
        """
        Encode S1 queries and retrieve top-K candidates from S2 and S3.
        
        Returns:
            Dict mapping s1_id -> [(candidate_id, rank, similarity)]
        """
        if len(s1_df) == 0:
            return {}

        texts_s1 = s1_df["combined_text"].tolist()
        ids_s1 = s1_df["entity_id"].tolist()

        emb_s1 = self.encoder.encode_texts(
            texts_s1,
            batch_size=self.batch_size,
            device=self.device,
            show_progress=show_progress,
        )

        results: Dict[str, List[Tuple[str, int, float]]] = {eid: [] for eid in ids_s1}

        # Search S2
        if self.index_s2 is not None and self.top_k_source2 > 0:
            scores_s2, idxs_s2 = self.index_s2.search(emb_s1, top_k=self.top_k_source2)
            for i, s1_id in enumerate(ids_s1):
                row_scores = scores_s2[i]
                row_idxs = idxs_s2[i]
                for rank, (score, idx) in enumerate(zip(row_scores, row_idxs), start=1):
                    cand_id = self.index_s2.ids[idx]
                    results[s1_id].append((cand_id, rank, float(score)))

        # Search S3
        if self.index_s3 is not None and self.top_k_source3 > 0:
            scores_s3, idxs_s3 = self.index_s3.search(emb_s1, top_k=self.top_k_source3)
            for i, s1_id in enumerate(ids_s1):
                row_scores = scores_s3[i]
                row_idxs = idxs_s3[i]
                for rank, (score, idx) in enumerate(zip(row_scores, row_idxs), start=1):
                    cand_id = self.index_s3.ids[idx]
                    results[s1_id].append((cand_id, rank, float(score)))

        return results
