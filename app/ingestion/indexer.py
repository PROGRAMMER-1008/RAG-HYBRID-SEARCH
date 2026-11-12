"""
Phase 1d / Ingestion entrypoint: build both indexes from the raw corpus.

This is the script you run locally (needs Hugging Face access for the
embedding model, which the sandbox this was developed in does not have).

Usage:
    python -m app.ingestion.indexer

What it does, in order:
    1. Load + chunk the corpus (loader.py, chunker.py — pure logic, no model needed)
    2. Embed every chunk with bge-small-en-v1.5 (NO query prefix — see note below)
    3. Deduplicate near-identical chunks (dedup.py)
    4. Write surviving chunks + embeddings into ChromaDB (dense index)
    5. Build a BM25 index over the same surviving chunks and persist its
       tokenized source data as JSON, then atomically switch the active manifest

IMPORTANT bge asymmetry: documents being indexed get embedded RAW, with no
instruction prefix. Only user QUERIES get the BGE_QUERY_PREFIX prepended,
at retrieval time (see retrieval/dense.py in Phase 2). Getting this
backwards silently degrades retrieval quality without erroring — there's
no crash to warn you, so this comment is the warning.
"""

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    INDEX_MANIFEST_PATH,
    RAW_DOCS_DIR,
)
from app.ingestion.chunker import chunk_corpus
from app.ingestion.dedup import deduplicate
from app.ingestion.loader import load_corpus
from app.retrieval.tokenize import tokenize_technical_text

BM25_INDEX_PATH = CHROMA_DIR.parent / "bm25_index.json"


def build_indexes(chunks=None, chroma_dir=None, bm25_path=None, collection_name=None, quiet=False):
    """Build both indexes from a chunk list.

    Parameterized (rather than hardcoded to chunk_corpus()'s output) so
    the Phase 4 chunking-strategy comparison can reuse this exact,
    already-tested embed/dedup/index logic for fixed-size and semantic
    chunks too, instead of forking a second copy of it. Calling with no
    arguments preserves the original production behavior exactly:
    load the real corpus, chunk it with the production (structure-aware)
    chunker, and write a versioned dense+sparse index under the real data directory.
    """
    is_default_index = chunks is None and chroma_dir is None and bm25_path is None and collection_name is None
    index_version = uuid.uuid4().hex[:12]
    _chroma_dir = chroma_dir or CHROMA_DIR
    _bm25_path = bm25_path or (CHROMA_DIR.parent / f"bm25_index-{index_version}.json" if is_default_index else BM25_INDEX_PATH)
    _collection_name = collection_name or (f"{COLLECTION_NAME}_{index_version}" if is_default_index else COLLECTION_NAME)

    def log(msg):
        if not quiet:
            print(msg)

    if chunks is None:
        log(f"Loading corpus from {RAW_DOCS_DIR}...")
        docs = load_corpus(RAW_DOCS_DIR)
        log(f"  {len(docs)} documents loaded")
        log("Chunking...")
        chunks = chunk_corpus(docs)
    log(f"  {len(chunks)} chunks to index")

    log(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    log("Embedding chunks (no query prefix — these are documents, not queries)...")
    indexed_texts = [
        f"Document: {chunk.source_file}\nSection: {chunk.heading_path}\n{chunk.text}"
        for chunk in chunks
    ]
    embeddings = model.encode(indexed_texts, show_progress_bar=not quiet, normalize_embeddings=True)

    log("Deduplicating...")
    result = deduplicate(chunks, list(embeddings))
    log(f"  Kept {len(result.kept_chunks)}, dropped {len(result.dropped)} near-duplicates")
    for chunk, kept_against, sim in result.dropped:
        log(f"    DROPPED (sim={sim:.3f} vs {kept_against}): {chunk.text[:60]}...")

    log(f"Writing dense index to {_chroma_dir}...")
    _chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(_chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.create_collection(_collection_name, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c.chunk_id for c in result.kept_chunks],
        embeddings=[e.tolist() for e in result.kept_embeddings],
        documents=[c.text for c in result.kept_chunks],
        metadatas=[
            {
                "source_file": c.source_file,
                "heading_path": c.heading_path,
                "strategy": c.strategy,
            }
            for c in result.kept_chunks
        ],
    )
    log(f"  {collection.count()} chunks indexed in ChromaDB")

    log(f"Building BM25 index...")
    tokenized = [
        tokenize_technical_text(f"Document: {chunk.source_file}\nSection: {chunk.heading_path}\n{chunk.text}")
        for chunk in result.kept_chunks
    ]
    if not any(tokenized):
        raise ValueError("The corpus contains no searchable words. Add text documents and rebuild the index.")
    index_payload = {
        "format_version": 1,
        "documents": [
            {"chunk": asdict(chunk), "tokens": tokens}
            for chunk, tokens in zip(result.kept_chunks, tokenized)
        ],
    }
    _bm25_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_sparse_path = _bm25_path.with_name(f"{_bm25_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_sparse_path.write_text(json.dumps(index_payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_sparse_path, _bm25_path)
    log(f"  BM25 index persisted to {_bm25_path}")

    if is_default_index:
        manifest = {
            "format_version": 1,
            "version": index_version,
            "collection_name": _collection_name,
            "bm25_path": _bm25_path.name,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_count": len(result.kept_chunks),
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary_manifest_path = INDEX_MANIFEST_PATH.with_name(f"{INDEX_MANIFEST_PATH.name}.{uuid.uuid4().hex}.tmp")
        temporary_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary_manifest_path, INDEX_MANIFEST_PATH)
        log(f"  Active index switched atomically to version {index_version}")

    log(f"\nDone. Both indexes built over the same {len(result.kept_chunks)} chunks.")
    return result


if __name__ == "__main__":
    build_indexes()
