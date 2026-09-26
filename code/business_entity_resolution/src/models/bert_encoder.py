"""
BERT Dual Encoder module for dense semantic retrieval.

Constraint Checklist:
- Parameter count: <200M parameters strictly enforced.
  Default: sentence-transformers/all-MiniLM-L6-v2 (22.7M parameters, Apache 2.0 license).
  Alternative: distilbert-base-uncased (66M parameters, Apache 2.0 license).
- Dual encoder: independent inference on Source 1, Source 2, and Source 3 records.
- Output: L2-normalized embedding vectors for cosine similarity via inner product.
"""

import os
from typing import List, Optional, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False


class MeanPooling(nn.Module if HAS_TORCH_TRANSFORMERS else object):
    """Mean pooling over token representations weighted by attention mask."""

    def forward(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class BERTDualEncoder(nn.Module if HAS_TORCH_TRANSFORMERS else object):
    """
    BERT Dual Encoder for business entity resolution.
    Encodes records independently into dense L2-normalized vector representations.
    """

    # Documentation metadata
    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_ALLOWED_PARAMS = 200_000_000

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 128,
        projection_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
    ):
        if not HAS_TORCH_TRANSFORMERS:
            raise ImportError(
                "PyTorch and HuggingFace Transformers must be installed to use BERTDualEncoder."
            )
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPooling()

        # Check parameter count
        total_params = sum(p.numel() for p in self.transformer.parameters())
        if total_params >= self.MAX_ALLOWED_PARAMS:
            raise ValueError(
                f"Model {model_name} has {total_params} parameters, exceeding the 200M limit!"
            )
        self.param_count = total_params

        # Optional linear projection layer
        hidden_dim = self.transformer.config.hidden_size
        if projection_dim is not None and projection_dim != hidden_dim:
            self.projection = nn.Linear(hidden_dim, projection_dim)
            self.embedding_dim = projection_dim
        else:
            self.projection = nn.Identity()
            self.embedding_dim = hidden_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass generating normalized pooled embeddings."""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs[0]  # First element is hidden states
        pooled = self.pooler(token_embeddings, attention_mask)
        projected = self.projection(pooled)
        if self.normalize_embeddings:
            projected = F.normalize(projected, p=2, dim=1)
        return projected

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        device: str = "auto",
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode list of strings into normalized NumPy embeddings array.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.eval()

        all_embeddings = []
        n = len(texts)

        iterator = range(0, n, batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding with BERT")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + batch_size]
                # Fallback for empty strings
                batch_texts = [t if t.strip() else "empty" for t in batch_texts]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)

                emb = self.forward(encoded["input_ids"], encoded["attention_mask"])
                all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.embedding_dim), dtype=np.float32)

    def save_pretrained(self, save_directory: str):
        """Save encoder checkpoint and tokenizer."""
        os.makedirs(save_directory, exist_ok=True)
        self.transformer.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        meta = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim,
            "param_count": self.param_count,
        }
        import json
        with open(os.path.join(save_directory, "encoder_config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load fine-tuned checkpoint."""
        return cls(model_name=load_directory, **kwargs)
