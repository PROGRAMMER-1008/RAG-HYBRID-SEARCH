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
    5. Build a BM25 index over the same surviving chunks' raw text (sparse index),
       and persist it to disk with pickle since BM25 has no native persistence

IMPORTANT bge asymmetry: documents being indexed get embedded RAW, with no
instruction prefix. Only user QUERIES get the BGE_QUERY_PREFIX prepended,
at retrieval time (see retrieval/dense.py in Phase 2). Getting this
backwards silently degrades retrieval quality without erroring — there's
no crash to warn you, so this comment is the warning.
"""

import pickle

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RAW_DOCS_DIR,
)
from app.ingestion.chunker import chunk_corpus
from app.ingestion.dedup import deduplicate
from app.ingestion.loader import load_corpus

BM25_INDEX_PATH = CHROMA_DIR.parent / "bm25_index.pkl"


def build_indexes(chunks=None, chroma_dir=None, bm25_path=None, collection_name=None, quiet=False):
    """Build both indexes from a chunk list.

    Parameterized (rather than hardcoded to chunk_corpus()'s output) so
    the Phase 4 chunking-strategy comparison can reuse this exact,
    already-tested embed/dedup/index logic for fixed-size and semantic
    chunks too, instead of forking a second copy of it. Calling with no
    arguments preserves the original production behavior exactly:
    load the real corpus, chunk it with the production (structure-aware)
    chunker, and write to the real CHROMA_DIR/BM25_INDEX_PATH.
    """
    _chroma_dir = chroma_dir or CHROMA_DIR
    _bm25_path = bm25_path or BM25_INDEX_PATH
    _collection_name = collection_name or COLLECTION_NAME

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
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=not quiet, normalize_embeddings=True)

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
    try:
        client.delete_collection(_collection_name)
    except Exception:
        pass
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
    tokenized = [c.text.lower().split() for c in result.kept_chunks]
    bm25 = BM25Okapi(tokenized)

    with open(_bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": result.kept_chunks}, f)
    log(f"  BM25 index persisted to {_bm25_path}")

    log(f"\nDone. Both indexes built over the same {len(result.kept_chunks)} chunks.")
    return result


if __name__ == "__main__":
    build_indexes()
