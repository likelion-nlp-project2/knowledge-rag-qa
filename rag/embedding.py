"""SentenceTransformer 로드 + 임베딩 (query/passage prefix 지원, e5류 대비)."""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def load_embedder(model_name: str, device: str, max_seq_len: int) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_len
    return model


def embed(
    model: SentenceTransformer,
    texts: List[str],
    prefix: str = "",
    batch_size: int = 128,
    show_progress: bool = False,
) -> np.ndarray:
    prefixed = [prefix + t for t in texts]
    return model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
