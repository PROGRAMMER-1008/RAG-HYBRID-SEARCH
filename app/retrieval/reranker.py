"""
Phase 2d: Reranking.

Job: take the FUSION_TOP_N candidates from RRF and re-score each one
with a cross-encoder that sees the (query, chunk) PAIR directly —
strictly more accurate than dense/sparse's independent-vector
comparison, but too slow to run over the whole corpus, which is why
it only runs on the already-narrowed fusion shortlist.

bge-reranker-base outputs a single relevance logit per (query, chunk)
pair, not a 0-1 probability by default — we keep the raw score and
only care about relative ordering, not its absolute scale.
"""

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.config import RERANK_TOP_N, RERANKER_MODEL
from app.retrieval.fusion import FusedResult


@dataclass
class RerankedResult:
    chunk_id: str
    text: str
    source_file: str
    heading_path: str
    rerank_score: float
    # Carried through for debugging/eval — lets you see how much the
    # reranker's opinion diverged from the fusion stage's ordering.
    rrf_score: float


class Reranker:
    def __init__(self, model=None):
        self.model = model or CrossEncoder(RERANKER_MODEL)

    def rerank(self, query: str, candidates: list[FusedResult], top_n: int = RERANK_TOP_N) -> list[RerankedResult]:
        if not candidates:
            return []

        # CrossEncoder.predict wants a list of (query, doc) pairs, scores all at once
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)

        reranked = [
            RerankedResult(
                chunk_id=c.chunk_id,
                text=c.text,
                source_file=c.source_file,
                heading_path=c.heading_path,
                rerank_score=float(score),
                rrf_score=c.rrf_score,
            )
            for c, score in zip(candidates, scores)
        ]

        reranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return reranked[:top_n]


if __name__ == "__main__":
    # Real end-to-end smoke test: needs the real dense+sparse+fusion
    # pipeline, so this only runs where the actual indexes exist (your
    # machine), not in the dev sandbox.
    from app.retrieval.dense import DenseRetriever
    from app.retrieval.sparse import SparseRetriever
    from app.retrieval.fusion import reciprocal_rank_fusion

    query = "What loss function handles class imbalance in the tumor segmentation model?"

    dense_hits = DenseRetriever().search(query)
    sparse_hits = SparseRetriever().search(query)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits)

    print(f"Query: {query}\n")
    print(f"Fusion shortlist: {len(fused)} candidates\n")

    reranked = Reranker().rerank(query, fused)
    print(f"After reranking, top {len(reranked)}:\n")
    for r in reranked:
        print(f"[rerank={r.rerank_score:.3f}  rrf={r.rrf_score:.5f}] {r.source_file} | {r.heading_path}")
        print(f"        {r.text[:100]}...")
