"""Per-upload, in-memory hybrid retrieval without shared user indexes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import (
    BGE_QUERY_PREFIX,
    DENSE_TOP_K,
    FUSION_TOP_N,
    RERANK_TOP_N,
    RRF_K,
    SPARSE_TOP_K,
    settings,
)
from app.ingestion.chunker import Chunk, chunk_corpus
from app.ingestion.loader import RawDocument
from app.retrieval.dense import DenseResult
from app.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from app.retrieval.pipeline import RetrievalTrace
from app.retrieval.reranker import RerankedResult
from app.retrieval.sparse import SparseResult
from app.retrieval.tokenize import tokenize_technical_text


@dataclass
class WorkspacePipeline:
    """Hybrid retrieval state scoped to one short-lived user workspace."""

    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: BM25Okapi
    tokenized_documents: list[list[str]]
    embedding_model: object
    reranker_model: object | None

    @classmethod
    def build(
        cls,
        documents: list[RawDocument],
        embedding_model: object,
        reranker_model: object | None,
    ) -> "WorkspacePipeline":
        chunks = chunk_corpus(documents)
        if not chunks:
            raise ValueError("No searchable text chunks were extracted from these files.")
        if len(chunks) > settings.max_chunks_per_workspace:
            raise ValueError(
                f"The uploaded files create {len(chunks)} chunks; the limit is "
                f"{settings.max_chunks_per_workspace}. Split the files or upload fewer at a time."
            )

        # Include source context in the retrieval representation. Keep Chunk.text
        # unchanged so citations display the actual extracted passage.
        index_texts = [
            f"Document: {chunk.source_file}\nSection: {chunk.heading_path}\n{chunk.text}"
            for chunk in chunks
        ]
        tokenized = [tokenize_technical_text(text) for text in index_texts]
        if not any(tokenized):
            raise ValueError("No searchable words were found in the uploaded files.")
        embeddings = np.asarray(
            embedding_model.encode(index_texts, show_progress_bar=False, normalize_embeddings=True),
            dtype=np.float32,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise RuntimeError("The embedding model returned an invalid index matrix.")
        return cls(
            chunks=chunks,
            embeddings=embeddings,
            bm25=BM25Okapi(tokenized),
            tokenized_documents=tokenized,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
        )

    def retrieve(self, query: str) -> RetrievalTrace:
        query_embedding = np.asarray(
            self.embedding_model.encode(
                [BGE_QUERY_PREFIX + query], show_progress_bar=False, normalize_embeddings=True
            )[0],
            dtype=np.float32,
        )
        similarities = self.embeddings @ query_embedding
        dense_order = np.argsort(-similarities, kind="stable")[: min(DENSE_TOP_K, len(self.chunks))]
        dense_hits = [self._dense_result(int(index), float(similarities[index])) for index in dense_order]

        sparse_scores = self.bm25.get_scores(tokenize_technical_text(query))
        sparse_order = sorted(
            range(len(self.chunks)), key=lambda index: (-float(sparse_scores[index]), index)
        )[: min(SPARSE_TOP_K, len(self.chunks))]
        sparse_hits = [self._sparse_result(index, float(sparse_scores[index])) for index in sparse_order]

        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=RRF_K, top_n=FUSION_TOP_N)
        final = self._rerank(query, fused)
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

    def _dense_result(self, index: int, score: float) -> DenseResult:
        chunk = self.chunks[index]
        return DenseResult(chunk.chunk_id, chunk.text, chunk.source_file, chunk.heading_path, score)

    def _sparse_result(self, index: int, score: float) -> SparseResult:
        chunk = self.chunks[index]
        return SparseResult(chunk.chunk_id, chunk.text, chunk.source_file, chunk.heading_path, score)

    def _rerank(self, query: str, candidates: list[FusedResult]) -> list[RerankedResult]:
        if not candidates:
            return []
        if self.reranker_model is not None:
            pairs = [(query, candidate.text) for candidate in candidates]
            scores = self.reranker_model.predict(pairs)
            reranked = [
                RerankedResult(
                    candidate.chunk_id,
                    candidate.text,
                    candidate.source_file,
                    candidate.heading_path,
                    float(score),
                    candidate.rrf_score,
                )
                for candidate, score in zip(candidates, scores)
            ]
            reranked.sort(key=lambda result: result.rerank_score, reverse=True)
            return reranked[:RERANK_TOP_N]
        return [
            RerankedResult(
                candidate.chunk_id,
                candidate.text,
                candidate.source_file,
                candidate.heading_path,
                candidate.rrf_score,
                candidate.rrf_score,
            )
            for candidate in candidates[:RERANK_TOP_N]
        ]
