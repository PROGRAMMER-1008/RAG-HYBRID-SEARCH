"""
Phase 2a: Dense retrieval.

Job: given a user question, embed it and query ChromaDB for the top-k
chunks by cosine similarity.

Critical bge detail (see indexer.py's note): documents were embedded
RAW at index time. Queries must be embedded WITH the instruction
prefix (BGE_QUERY_PREFIX). This asymmetry is how bge models were
trained to be used — getting it backwards doesn't crash, it just
quietly retrieves worse results, so this is enforced here in one
place rather than left to whoever calls this function to remember.
"""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.config import (
    BGE_QUERY_PREFIX,
    CHROMA_DIR,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
)


@dataclass
class DenseResult:
    chunk_id: str
    text: str
    source_file: str
    heading_path: str
    score: float   # cosine similarity, higher = more relevant


class DenseRetriever:
    def __init__(self, chroma_dir=None, collection_name=None):
        # Defaults preserve exact original behavior for every existing
        # caller. Explicit overrides exist so the Phase 4 chunking
        # comparison (chunking_comparison.py) can point at an isolated
        # index without touching production data — passing these in is
        # the correct fix; module-attribute reassignment doesn't work
        # here because CHROMA_DIR/COLLECTION_NAME are bound to local
        # names at import time via `from app.config import ...`.
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(
            path=str(chroma_dir or CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(collection_name or COLLECTION_NAME)

    def search(self, query: str, top_k: int = DENSE_TOP_K) -> list[DenseResult]:
        # The one line that matters most in this file: prefix applied
        # to the QUERY only, never to indexed documents.
        prefixed_query = BGE_QUERY_PREFIX + query
        query_embedding = self.model.encode([prefixed_query], normalize_embeddings=True)[0]

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        out = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]   # cosine DISTANCE (0=identical), not similarity

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            similarity = 1 - dist   # ChromaDB's cosine space returns distance; convert back
            out.append(
                DenseResult(
                    chunk_id=cid,
                    text=doc,
                    source_file=meta["source_file"],
                    heading_path=meta["heading_path"],
                    score=similarity,
                )
            )
        return out


if __name__ == "__main__":
    retriever = DenseRetriever()
    query = "What is the accuracy of the brain tumor segmentation model?"
    print(f"Query: {query}\n")
    for r in retriever.search(query, top_k=5):
        print(f"[{r.score:.3f}] {r.source_file} | {r.heading_path}")
        print(f"        {r.text[:100]}...")
