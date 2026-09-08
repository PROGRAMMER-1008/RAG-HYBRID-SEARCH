"""
Phase 2c: Fusion via Reciprocal Rank Fusion (RRF).

Job: combine a dense result list and a sparse result list into one
ranked list, without needing to reconcile their incompatible score
scales (bounded cosine similarity vs. unbounded BM25 score).

RRF sidesteps the scale problem entirely by using each chunk's RANK
POSITION in each list, not its raw score:

    rrf_score(chunk) = sum over lists it appears in of  1 / (k + rank)

A chunk ranked #1 in both lists scores highest. A chunk that only
shows up in one list still contributes, just less. k=60 (from config,
the standard value from the original RRF paper) dampens how much
being #1 vs #2 matters — without it, rank 1 vs rank 2 would swing
the score disproportionately for small k.
"""

from dataclasses import dataclass

from app.config import FUSION_TOP_N, RRF_K
from app.retrieval.dense import DenseResult
from app.retrieval.sparse import SparseResult


@dataclass
class FusedResult:
    chunk_id: str
    text: str
    source_file: str
    heading_path: str
    rrf_score: float
    dense_rank: int | None    # 1-indexed rank in dense list, None if absent
    sparse_rank: int | None   # 1-indexed rank in sparse list, None if absent


def reciprocal_rank_fusion(
    dense_results: list[DenseResult],
    sparse_results: list[SparseResult],
    k: int = RRF_K,
    top_n: int = FUSION_TOP_N,
) -> list[FusedResult]:
    # chunk_id -> accumulated data as we walk both lists
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    # chunk_id -> the actual content/metadata, so we only need to store it once
    chunk_data: dict[str, dict] = {}

    for rank, r in enumerate(dense_results, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
        dense_ranks[r.chunk_id] = rank
        chunk_data[r.chunk_id] = {
            "text": r.text,
            "source_file": r.source_file,
            "heading_path": r.heading_path,
        }

    for rank, r in enumerate(sparse_results, start=1):
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
        sparse_ranks[r.chunk_id] = rank
        # Only set chunk_data if dense didn't already provide it —
        # both lists describe the same chunk_id identically, so either source is fine.
        chunk_data.setdefault(
            r.chunk_id,
            {"text": r.text, "source_file": r.source_file, "heading_path": r.heading_path},
        )

    fused = [
        FusedResult(
            chunk_id=cid,
            text=chunk_data[cid]["text"],
            source_file=chunk_data[cid]["source_file"],
            heading_path=chunk_data[cid]["heading_path"],
            rrf_score=score,
            dense_rank=dense_ranks.get(cid),
            sparse_rank=sparse_ranks.get(cid),
        )
        for cid, score in scores.items()
    ]

    fused.sort(key=lambda r: r.rrf_score, reverse=True)
    return fused[:top_n]


if __name__ == "__main__":
    # Synthetic test: no model or index needed, pure fusion-logic check.
    dense = [
        DenseResult(chunk_id="A", text="dense text A", source_file="f.md", heading_path="H", score=0.91),
        DenseResult(chunk_id="B", text="dense text B", source_file="f.md", heading_path="H", score=0.85),
        DenseResult(chunk_id="C", text="dense text C", source_file="f.md", heading_path="H", score=0.80),
    ]
    sparse = [
        SparseResult(chunk_id="C", text="sparse text C", source_file="f.md", heading_path="H", score=9.5),
        SparseResult(chunk_id="A", text="sparse text A", source_file="f.md", heading_path="H", score=7.2),
        SparseResult(chunk_id="D", text="sparse text D", source_file="f.md", heading_path="H", score=6.0),
    ]

    # Expectation: A appears rank 1 in dense, rank 2 in sparse -> strong combined score
    #              C appears rank 3 in dense, rank 1 in sparse -> also strong
    #              B only in dense (rank 2), D only in sparse (rank 3) -> weaker, single-list only
    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_n=10)

    for r in fused:
        print(f"{r.chunk_id}: rrf={r.rrf_score:.5f}  dense_rank={r.dense_rank}  sparse_rank={r.sparse_rank}")
