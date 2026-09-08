"""
Phase 4 addendum: alternate chunking strategies for comparison.

The guide's Phase 4 spec asks for a comparison across three chunking
strategies, scored on the same eval suite, as evidence for which
strategy actually wins — not just an assumption. chunker.py's
structure-aware (header-based) splitting is the ALREADY-PRODUCTION
strategy, extensively tested through Phases 1-4. This file adds the
other two strategies PURELY for comparison; nothing here is used by
the live retrieval pipeline.

Kept as a separate file rather than modifying chunker.py so the
already-verified production chunker carries zero risk from this
addition — a bug here can't touch what's actually indexed and served.
"""

import re
from dataclasses import dataclass

from app.ingestion.chunker import Chunk
from app.ingestion.loader import RawDocument


def fixed_size_chunk_document(doc: RawDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    """Baseline strategy: cut every `chunk_size` characters across the
    WHOLE document, ignoring markdown structure entirely. This is
    deliberately the naive approach the guide's spec calls "baseline" —
    it WILL cut through headers, tables, and code blocks mid-structure,
    which is exactly what structure-aware chunking exists to avoid.
    Kept simple on purpose: no bullet-boundary awareness, no fence
    masking — the point of a baseline is to be the thing you're
    comparing against, not a second good option.
    """
    text = doc.text.strip()
    if len(text) <= chunk_size:
        return [Chunk(text=text, source_file=doc.source_path, heading_path="(fixed-size, no headers)",
                       chunk_index=0, strategy="fixed_size_baseline")]

    chunks = []
    start = 0
    idx = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(
            text=text[start:end],
            source_file=doc.source_path,
            heading_path="(fixed-size, no headers)",
            chunk_index=idx,
            strategy="fixed_size_baseline",
        ))
        idx += 1
        if end == len(text):
            break
        start += step
    return chunks


def fixed_size_chunk_corpus(docs: list[RawDocument], chunk_size: int, overlap: int) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(fixed_size_chunk_document(doc, chunk_size, overlap))
    return all_chunks


def _split_sentences(text: str) -> list[str]:
    """Rough sentence splitter for semantic chunking's candidate boundaries.
    Same known simplification as citation.py's claim splitter (won't
    handle 'Dr. Smith' correctly) — acceptable for technical READMEs.
    """
    # Split on paragraph breaks first (a stronger signal than sentence
    # punctuation for topic shifts in markdown prose), then sentences
    # within a paragraph.
    paragraphs = re.split(r"\n\s*\n", text)
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = re.split(r"(?<=[.!?])\s+", para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def semantic_chunk_document(doc: RawDocument, embedding_model, chunk_size: int,
                             similarity_threshold: float = 0.55) -> list[Chunk]:
    """Semantic strategy: split into sentences, embed each one, and cut
    a new chunk whenever consecutive sentences' cosine similarity drops
    below `similarity_threshold` — i.e. wherever the topic actually
    shifts, rather than at a fixed character count or a markdown header.

    Requires a loaded SentenceTransformer (passed in, not loaded here)
    since this needs the exact same bge-small model already used for
    the production embedding index — reusing embeddings this way is
    also why this function can't run standalone the way fixed_size can;
    it always needs Hub access (see indexer.py's note on this).

    similarity_threshold=0.55 is a starting point, not a tuned constant —
    unlike CHUNK_SIZE/CHUNK_OVERLAP in config.py, this hasn't been
    validated against real eval numbers the way the production chunker's
    constants have; treat it as a reasonable default for this comparison
    run, not a claim of optimality.
    """
    import numpy as np

    sentences = _split_sentences(doc.text)
    if len(sentences) <= 1:
        return [Chunk(text=doc.text.strip(), source_file=doc.source_path,
                       heading_path="(semantic, single unit)", chunk_index=0,
                       strategy="semantic")]

    embeddings = embedding_model.encode(sentences, normalize_embeddings=True)

    boundaries = [0]  # sentence indices where a new chunk starts
    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))  # normalized vectors -> dot = cosine
        if sim < similarity_threshold:
            boundaries.append(i)
    boundaries.append(len(sentences))

    # Merge sentence-groups into chunks, respecting chunk_size as an
    # upper bound so one topic segment doesn't become one enormous chunk.
    chunks = []
    chunk_idx = 0
    for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
        group = sentences[b_start:b_end]
        buffer = ""
        for sent in group:
            candidate = (buffer + " " + sent).strip() if buffer else sent
            if len(candidate) > chunk_size and buffer:
                chunks.append(Chunk(text=buffer, source_file=doc.source_path,
                                     heading_path="(semantic boundary)", chunk_index=chunk_idx,
                                     strategy="semantic"))
                chunk_idx += 1
                buffer = sent
            else:
                buffer = candidate
        if buffer:
            chunks.append(Chunk(text=buffer, source_file=doc.source_path,
                                 heading_path="(semantic boundary)", chunk_index=chunk_idx,
                                 strategy="semantic"))
            chunk_idx += 1

    return chunks


def semantic_chunk_corpus(docs: list[RawDocument], embedding_model, chunk_size: int) -> list[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(semantic_chunk_document(doc, embedding_model, chunk_size))
    return all_chunks


if __name__ == "__main__":
    # Fixed-size test: pure logic, no model needed, runs anywhere.
    from app.ingestion.loader import load_corpus
    from app.config import RAW_DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

    docs = load_corpus(RAW_DOCS_DIR)
    fixed_chunks = fixed_size_chunk_corpus(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Fixed-size baseline: {len(fixed_chunks)} chunks")
    lengths = [len(c.text) for c in fixed_chunks]
    print(f"  length: min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.0f}")

    # Semantic chunking needs the real embedding model — cannot run in
    # a sandbox without Hugging Face Hub access. Left commented as the
    # exact call to make on a machine that has it:
    #
    # from sentence_transformers import SentenceTransformer
    # from app.config import EMBEDDING_MODEL
    # model = SentenceTransformer(EMBEDDING_MODEL)
    # semantic_chunks = semantic_chunk_corpus(docs, model, CHUNK_SIZE)
    # print(f"Semantic: {len(semantic_chunks)} chunks")
