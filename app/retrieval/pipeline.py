"""
Phase 2e: Pipeline orchestrator.

Job: wire dense -> sparse -> fusion -> rerank into one object that
loads its models/indexes ONCE (expensive) and exposes a single
retrieve(query) method (cheap per call). Without this, every caller
(API endpoint, eval script) would need to remember the correct
4-step order and re-load the embedding + reranker models on every
single call, which is both error-prone and slow.
"""

from dataclasses import dataclass, field

from app.retrieval.dense import DenseRetriever, DenseResult
from app.retrieval.sparse import SparseRetriever, SparseResult
from app.retrieval.fusion import reciprocal_rank_fusion, FusedResult
from app.retrieval.reranker import Reranker, RerankedResult


@dataclass
class RetrievalTrace:
    """Full evidence trail for one query — every intermediate stage,
    not just the final answer. This is what makes the system debuggable:
    if the final result looks wrong, you can inspect exactly which
    stage (dense, sparse, fusion, or rerank) is responsible, the same
    diagnostic instinct as the tracing layer in Failure Forensics.

    dense_hits/sparse_hits/fused_hits carry the ACTUAL per-stage results
    (not just counts) — added for the dashboard's stage-by-stage
    visualization, so it can show a chunk's rank at every stage, not
    only its final position. The *_count fields are kept alongside for
    existing callers (evaluator.py, pipeline.py's own __main__ block)
    that only ever needed the aggregate number.
    """
    query: str
    dense_count: int
    sparse_count: int
    fused_count: int
    final_results: list[RerankedResult]
    dense_hits: list[DenseResult] = field(default_factory=list)
    sparse_hits: list[SparseResult] = field(default_factory=list)
    fused_hits: list[FusedResult] = field(default_factory=list)


class HybridRetrievalPipeline:
    def __init__(self, chroma_dir=None, collection_name=None, bm25_path=None,
                 embedding_model=None, reranker_model=None):
        print("Loading retrieval pipeline (embedding model, indexes, reranker)...")
        self.dense = DenseRetriever(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            model=embedding_model,
        )
        self.sparse = SparseRetriever(bm25_path=bm25_path)
        self.reranker = Reranker(model=reranker_model)
        print("Pipeline ready.")

    def retrieve(self, query: str) -> RetrievalTrace:
        dense_hits = self.dense.search(query)
        sparse_hits = self.sparse.search(query)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)
        final = self.reranker.rerank(query, fused)

        return RetrievalTrace(
            query=query,
            dense_count=len(dense_hits),
            sparse_count=len(sparse_hits),
            fused_count=len(fused),
            final_results=final,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused_hits=fused,
        )


if __name__ == "__main__":
    pipeline = HybridRetrievalPipeline()

    test_queries = [
        "What loss function handles class imbalance in tumor segmentation?",
        "What ROC-AUC does the corruption detector achieve?",
        "What formula does the Delhi Digital Twin use for traffic congestion?",
        "How does Failure Forensics do backward root-cause analysis?",
    ]

    for q in test_queries:
        trace = pipeline.retrieve(q)
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        print(f"  dense={trace.dense_count} sparse={trace.sparse_count} fused={trace.fused_count}")
        for r in trace.final_results[:3]:
            print(f"  [{r.rerank_score:.3f}] {r.source_file} | {r.heading_path}")
