"""
Phase 1c: Deduplication.

Job: given chunks with embeddings, drop near-duplicates before they
reach the vector store. A near-duplicate wastes a retrieval slot on
redundant content instead of a second useful fact.

Deliberately embedding-agnostic: this module takes pre-computed vectors
in, so it can be unit-tested with synthetic vectors (see test block)
without needing a real embedding model loaded. The real model gets
wired in by indexer.py.
"""

import numpy as np
from dataclasses import dataclass

from app.ingestion.chunker import Chunk

DEFAULT_SIMILARITY_THRESHOLD = 0.95


@dataclass
class DedupResult:
    kept_chunks: list[Chunk]
    kept_embeddings: list[np.ndarray]
    dropped: list[tuple[Chunk, str, float]]   # (dropped_chunk, kept_against_chunk_id, similarity)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def deduplicate(
    chunks: list[Chunk],
    embeddings: list[np.ndarray],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> DedupResult:
    """O(n^2) pairwise comparison against already-accepted chunks.

    Fine at our corpus scale (~100 chunks -> ~5k comparisons). If this
    corpus grows into the thousands, swap for an approximate nearest-
    neighbor index (e.g. query ChromaDB itself before insert) rather
    than optimizing this loop directly.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    kept_chunks: list[Chunk] = []
    kept_embeddings: list[np.ndarray] = []
    dropped: list[tuple[Chunk, str, float]] = []

    for chunk, emb in zip(chunks, embeddings):
        is_dupe = False
        for kept_chunk, kept_emb in zip(kept_chunks, kept_embeddings):
            sim = _cosine_sim(emb, kept_emb)
            if sim >= threshold:
                dropped.append((chunk, kept_chunk.chunk_id, sim))
                is_dupe = True
                break
        if not is_dupe:
            kept_chunks.append(chunk)
            kept_embeddings.append(emb)

    return DedupResult(kept_chunks=kept_chunks, kept_embeddings=kept_embeddings, dropped=dropped)


if __name__ == "__main__":
    # Algorithm test with synthetic vectors — no real embedding model needed.
    # This proves the dedup LOGIC works; indexer.py will feed it real
    # bge-small embeddings later, when running with actual Hub access.
    from app.ingestion.chunker import Chunk

    fake_chunks = [
        Chunk(text="ROC-AUC is 0.681", source_file="a.md", heading_path="Results", chunk_index=0),
        Chunk(text="ROC AUC score: 0.681", source_file="a.md", heading_path="Results (copy)", chunk_index=0),  # near-dupe of #1
        Chunk(text="The pipeline uses FastAPI for the backend", source_file="b.md", heading_path="Stack", chunk_index=0),
        Chunk(text="Dice score is 0.691 for segmentation", source_file="c.md", heading_path="Results", chunk_index=0),
    ]

    # Synthetic embeddings: #0 and #1 nearly identical vectors, #2 and #3 distinct
    fake_embeddings = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.99, 0.02, 0.0]),   # cosine sim ~0.999 with #0 -> should be dropped
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    result = deduplicate(fake_chunks, fake_embeddings, threshold=0.95)

    print(f"Kept: {len(result.kept_chunks)} / {len(fake_chunks)}")
    for c in result.kept_chunks:
        print(f"  KEPT: {c.text}")
    for chunk, kept_against, sim in result.dropped:
        print(f"  DROPPED: '{chunk.text}' (sim={sim:.3f} vs {kept_against})")
