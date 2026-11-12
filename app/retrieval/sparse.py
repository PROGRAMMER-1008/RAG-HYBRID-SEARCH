"""Sparse retrieval backed by a safe JSON corpus, never executable pickle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import CHROMA_DIR, INDEX_MANIFEST_PATH, SPARSE_TOP_K
from app.ingestion.chunker import Chunk
from app.retrieval.tokenize import tokenize_technical_text

BM25_INDEX_PATH = CHROMA_DIR.parent / "bm25_index.json"


@dataclass
class SparseResult:
    chunk_id: str
    text: str
    source_file: str
    heading_path: str
    score: float


def _resolve_index_path(explicit_path: str | Path | None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    if INDEX_MANIFEST_PATH.exists():
        try:
            manifest = json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
            file_name = manifest["bm25_path"]
            if not isinstance(file_name, str) or Path(file_name).name != file_name or not file_name.endswith(".json"):
                raise ValueError("Invalid sparse index path in manifest.")
            return INDEX_MANIFEST_PATH.parent / file_name
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("The active index manifest is malformed. Rebuild the index.") from exc
    return BM25_INDEX_PATH


class SparseRetriever:
    def __init__(self, bm25_path: str | Path | None = None):
        index_path = _resolve_index_path(bm25_path)
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            if payload.get("format_version") != 1 or not isinstance(payload.get("documents"), list):
                raise ValueError("Unsupported sparse index format. Rebuild the index.")
            documents = payload["documents"]
            chunks: list[Chunk] = []
            tokenized: list[list[str]] = []
            for item in documents:
                chunk_data = item["chunk"]
                tokens = item["tokens"]
                if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
                    raise ValueError("The sparse index contains invalid tokens.")
                chunks.append(Chunk(**chunk_data))
                tokenized.append(tokens)
            if not chunks or len(chunks) != len(tokenized):
                raise ValueError("The sparse index is empty or inconsistent.")
            if not any(tokenized):
                raise ValueError("The sparse index contains no searchable tokens. Rebuild it from text documents.")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Sparse index not found at {index_path}. Rebuild it with `python -m app.ingestion.indexer`."
            ) from exc
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("The sparse index is malformed. Rebuild it from trusted source files.") from exc

        self.chunks = chunks
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = SPARSE_TOP_K) -> list[SparseResult]:
        scores = self.bm25.get_scores(tokenize_technical_text(query))
        scored = list(zip(self.chunks, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            SparseResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_file=chunk.source_file,
                heading_path=chunk.heading_path,
                score=float(score),
            )
            for chunk, score in scored[:top_k]
        ]
