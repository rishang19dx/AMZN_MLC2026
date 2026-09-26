"""
Approximate Nearest Neighbor (ANN) and Exact Cosine Index module.

Supports:
1. FAISS index (IndexFlatIP or IndexIVFFlat) when faiss is available.
2. GPU/CPU Chunked PyTorch Top-K Matrix Multiplication (fast, zero memory bloat, CUDA-accelerated).
3. NumPy chunked dot product fallback.
"""

import os
import pickle
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VectorIndex:
    """
    High-performance vector index for cosine similarity search on normalized vectors.
    """

    def __init__(
        self,
        dim: int,
        use_faiss: bool = True,
        use_gpu: bool = True,
        device: str = "auto",
    ):
        self.dim = dim
        self.use_faiss = use_faiss and HAS_FAISS
        self.device = device
        if self.device == "auto":
            if HAS_TORCH and torch.cuda.is_available() and use_gpu:
                self.device = "cuda"
            else:
                self.device = "cpu"

        self.index = None
        self.embeddings: Optional[np.ndarray] = None
        self.ids: List[str] = []
        self._torch_embeddings = None

    def build(self, embeddings: np.ndarray, entity_ids: List[str]):
        """
        Build index over normalized embeddings (shape: N x dim).
        """
        assert len(embeddings) == len(entity_ids), "Embeddings and IDs count mismatch!"
        n, d = embeddings.shape
        assert d == self.dim, f"Embedding dimension mismatch: expected {self.dim}, got {d}"

        # Ensure float32 and L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = (embeddings / norms).astype(np.float32)

        self.ids = list(entity_ids)
        self.embeddings = normed

        if self.use_faiss:
            # Use FAISS Inner Product index (equivalent to cosine for normalized vectors)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(normed)
        elif HAS_TORCH and self.device == "cuda":
            # Pre-load to GPU tensor for ultra-fast chunked search
            self._torch_embeddings = torch.from_numpy(normed).to("cuda")

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 30,
        query_batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search top_k nearest neighbors for query_embeddings.
        
        Returns:
            scores: (num_queries x top_k) cosine similarity scores
            indices: (num_queries x top_k) candidate indices in self.ids
        """
        if self.embeddings is None or len(self.ids) == 0:
            return np.empty((len(query_embeddings), 0)), np.empty((len(query_embeddings), 0), dtype=int)

        # Ensure queries are L2-normalized float32
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q_normed = (query_embeddings / norms).astype(np.float32)
        k = min(top_k, len(self.ids))

        # 1. FAISS Backend
        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(q_normed, k)
            return scores, indices

        # 2. PyTorch GPU / CPU Chunked Matrix Multiplication
        if HAS_TORCH:
            q_tensor = torch.from_numpy(q_normed)
            target_tensor = (
                self._torch_embeddings
                if self._torch_embeddings is not None
                else torch.from_numpy(self.embeddings)
            )

            dev = (
                torch.device("cuda")
                if (self.device == "cuda" and torch.cuda.is_available())
                else torch.device("cpu")
            )
            target_tensor = target_tensor.to(dev)

            all_scores = []
            all_indices = []

            for i in range(0, len(q_tensor), query_batch_size):
                batch_q = q_tensor[i : i + query_batch_size].to(dev)
                # Cosine similarity matrix: (batch_size x N)
                sim_matrix = torch.matmul(batch_q, target_tensor.T)
                top_scores, top_idx = torch.topk(sim_matrix, k=k, dim=1, largest=True)
                all_scores.append(top_scores.cpu().numpy())
                all_indices.append(top_idx.cpu().numpy())

            return np.vstack(all_scores), np.vstack(all_indices)

        # 3. NumPy Fallback
        all_scores = []
        all_indices = []
        target = self.embeddings

        for i in range(0, len(q_normed), query_batch_size):
            batch_q = q_normed[i : i + query_batch_size]
            sim_matrix = np.matmul(batch_q, target.T)
            # Partition top_k
            part_idx = np.argpartition(-sim_matrix, k - 1, axis=1)[:, :k]
            row_idx = np.arange(len(batch_q))[:, None]
            part_scores = sim_matrix[row_idx, part_idx]
            sort_order = np.argsort(-part_scores, axis=1)
            sorted_idx = part_idx[row_idx, sort_order]
            sorted_scores = part_scores[row_idx, sort_order]
            all_scores.append(sorted_scores)
            all_indices.append(sorted_idx)

        return np.vstack(all_scores), np.vstack(all_indices)

    def save(self, filepath: str):
        """Save index and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if self.use_faiss and self.index is not None:
            faiss_file = f"{filepath}.faiss"
            faiss.write_index(self.index, faiss_file)
            meta = {"dim": self.dim, "ids": self.ids, "faiss_file": os.path.basename(faiss_file)}
            with open(filepath, "wb") as f:
                pickle.dump(meta, f)
        else:
            data = {"dim": self.dim, "ids": self.ids, "embeddings": self.embeddings}
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str, device: str = "auto") -> "VectorIndex":
        """Load index from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        idx = cls(dim=data["dim"], device=device)
        idx.ids = data["ids"]

        if "faiss_file" in data and HAS_FAISS:
            faiss_path = os.path.join(os.path.dirname(filepath), data["faiss_file"])
            idx.index = faiss.read_index(faiss_path)
            idx.use_faiss = True
        else:
            embeddings = data.get("embeddings")
            if embeddings is not None:
                idx.build(embeddings, idx.ids)

        return idx
