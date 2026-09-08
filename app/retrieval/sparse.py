"""
Phase 2b: Sparse retrieval.

Job: given a user question, tokenize it the same way the index was
built (lowercase + whitespace split) and query the persisted BM25
index for the top-k chunks by BM25 score.

No embedding model needed here at all — this is why it's testable
without Hub/network access, unlike dense.py.
"""

import pickle
from dataclasses import dataclass

from app.config import SPARSE_TOP_K
from app.ingestion.indexer import BM25_INDEX_PATH


@dataclass
class SparseResult:
    chunk_id: str
    text: str
    source_file: str
    heading_path: str
    score: float   # raw BM25 score, unbounded, higher = more relevant


class SparseRetriever:
    def __init__(self, bm25_path=None):
        # See DenseRetriever's __init__ comment for why this is a
        # constructor parameter rather than a module-attribute override.
        with open(bm25_path or BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]   # same order the BM25 index was built over

    def search(self, query: str, top_k: int = SPARSE_TOP_K) -> list[SparseResult]:
        # Must match the exact tokenization used at index time
        # (indexer.py: c.text.lower().split()) or scores are meaningless.
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each chunk with its score, sort descending, take top_k
        scored = list(zip(self.chunks, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        out = []
        for chunk, score in scored[:top_k]:
            out.append(
                SparseResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_file=chunk.source_file,
                    heading_path=chunk.heading_path,
                    score=float(score),
                )
            )
        return out


if __name__ == "__main__":
    retriever = SparseRetriever()
    query = "ResUNet Tversky loss segmentation"
    print(f"Query: {query}\n")
    for r in retriever.search(query, top_k=5):
        print(f"[{r.score:.3f}] {r.source_file} | {r.heading_path}")
        print(f"        {r.text[:100]}...")
