"""
Independent Learned Representation / JEPA-style Encoder module.

Constraint Checklist:
- Parameter count: <200M parameters strictly enforced.
  Default: google/electra-small-discriminator (14M parameters, Apache 2.0 license)
  or sentence-transformers/paraphrase-MiniLM-L3-v2 (17M parameters, Apache 2.0 license).
- Architecture:
  Joint-Embedding Predictive Architecture (JEPA) consisting of:
  1. Context Encoder E_c: maps partial / masked entity context to latent representation.
  2. Target Encoder E_t: maps complete target entity to latent space (updated via EMA / stop-gradient).
  3. Predictor Network P: MLP / Transformer predicting target embedding from context representation.
- Provides an independent, genuinely distinct candidate generation signal from Pipeline A.
"""

import copy
import os
from typing import List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    HAS_TORCH_TRANSFORMERS = True
except ImportError:
    HAS_TORCH_TRANSFORMERS = False


class JEPAPredictor(nn.Module if HAS_TORCH_TRANSFORMERS else object):
    """Predictor MLP predicting target latent vector from context representation."""

    def __init__(self, hidden_dim: int, predictor_hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(predictor_hidden_dim),
            nn.Linear(predictor_hidden_dim, hidden_dim),
        )

    def forward(self, context_rep: torch.Tensor) -> torch.Tensor:
        return self.net(context_rep)


class JEPAEntityEncoder(nn.Module if HAS_TORCH_TRANSFORMERS else object):
    """
    Joint-Embedding Predictive Architecture (JEPA) for Business Entities.
    """

    DEFAULT_BACKBONE = "google/electra-small-discriminator"
    MAX_ALLOWED_PARAMS = 200_000_000

    def __init__(
        self,
        model_name: str = DEFAULT_BACKBONE,
        max_length: int = 128,
        ema_decay: float = 0.996,
        projection_dim: Optional[int] = None,
    ):
        if not HAS_TORCH_TRANSFORMERS:
            raise ImportError("PyTorch and HuggingFace Transformers required for JEPAEntityEncoder.")
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.ema_decay = ema_decay

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Context Encoder
        self.context_encoder = AutoModel.from_pretrained(model_name)
        
        # Target Encoder (EMA copy)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False  # Target encoder updated strictly via EMA

        # Verify parameters (<200M)
        total_params = sum(p.numel() for p in self.context_encoder.parameters())
        if total_params >= self.MAX_ALLOWED_PARAMS:
            raise ValueError(f"Model {model_name} exceeds 200M parameter limit: {total_params}")
        self.param_count = total_params

        hidden_dim = self.context_encoder.config.hidden_size
        self.embedding_dim = projection_dim if projection_dim else hidden_dim

        # Predictor Network
        self.predictor = JEPAPredictor(hidden_dim=hidden_dim, predictor_hidden_dim=hidden_dim * 2)

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_emb = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_emb / sum_mask

    def encode_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode context through trainable context encoder."""
        out = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled = self._mean_pool(out, attention_mask)
        return pooled

    @torch.no_grad()
    def encode_target(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode complete target through EMA target encoder."""
        out = self.target_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled = self._mean_pool(out, attention_mask)
        return pooled

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass during training:
        Computes predicted target representation from context and returns actual target representation.
        """
        context_rep = self.encode_context(context_ids, context_mask)
        pred_target_rep = self.predictor(context_rep)

        actual_target_rep = None
        if target_ids is not None and target_mask is not None:
            actual_target_rep = self.encode_target(target_ids, target_mask)

        return pred_target_rep, actual_target_rep

    @torch.no_grad()
    def update_target_encoder(self):
        """Update EMA target encoder weights."""
        for c_param, t_param in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            t_param.data = self.ema_decay * t_param.data + (1.0 - self.ema_decay) * c_param.data

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        device: str = "auto",
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate L2-normalized representations for ANN retrieval using target encoder.
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
            iterator = tqdm(iterator, desc="Encoding with JEPA/Learned Model")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + batch_size]
                batch_texts = [t if t.strip() else "empty" for t in batch_texts]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)

                emb = self.encode_target(encoded["input_ids"], encoded["attention_mask"])
                normed = F.normalize(emb, p=2, dim=1)
                all_embeddings.append(normed.cpu().numpy())

        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.embedding_dim), dtype=np.float32)

    def save_pretrained(self, save_directory: str):
        """Save model weights and tokenizer."""
        os.makedirs(save_directory, exist_ok=True)
        self.target_encoder.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        torch.save(self.predictor.state_dict(), os.path.join(save_directory, "predictor.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load pretrained JEPA encoder."""
        return cls(model_name=load_directory, **kwargs)
