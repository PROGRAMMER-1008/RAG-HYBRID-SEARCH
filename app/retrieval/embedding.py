"""Small, CPU-only dense encoder used by the public workspace API."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class FastEmbedEncoder:
    """Adapt FastEmbed's generator API to the encoder interface used by retrieval."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str | Path,
        threads: int = 1,
        model=None,
    ) -> None:
        if model is None:
            from fastembed import TextEmbedding

            model = TextEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir),
                threads=threads,
                cuda=False,
            )
        self._model = model

    def encode(
        self,
        texts: Iterable[str],
        *,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        del show_progress_bar  # FastEmbed intentionally has no progress UI in the API.
        vectors = np.asarray(
            list(self._model.embed(texts, batch_size=8, parallel=None)),
            dtype=np.float32,
        )
        if vectors.ndim != 2:
            raise RuntimeError("The embedding backend returned an invalid vector matrix.")
        if normalize_embeddings and vectors.size:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.divide(vectors, norms, out=vectors, where=norms > 0)
        return vectors
