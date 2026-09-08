"""
Central config for the whole pipeline.

Why this file exists: every "why did you choose X" interview question
should have an answer traceable to a single tunable constant here,
not a magic number buried in some function three folders deep.
"""
import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
GOLDEN_SET_PATH = BASE_DIR / "app" / "eval" / "golden_set.json"

# --- Embeddings ---
# bge-small-en-v1.5: 384-dim, free, local, outperforms MiniLM on MTEB retrieval.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained with an asymmetric instruction prefix for queries
# (not for the documents being indexed) — this measurably improves retrieval.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Chunking ---
CHUNK_SIZE = 500          # characters, tuned in Phase 1
CHUNK_OVERLAP = 50

# --- Retrieval ---
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
RRF_K = 60                # standard RRF damping constant
FUSION_TOP_N = 20         # candidates passed into reranker
RERANK_TOP_N = 5          # final chunks passed to the LLM

# --- Reranker ---
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Generation ---
# Local dev default. Swap to Groq for the hosted portfolio demo.
GENERATION_PROVIDER = "ollama"   # "ollama" | "groq"
OLLAMA_MODEL = "llama3.1:8b"
# NOTE: Groq deprecated llama-3.3-70b-versatile on 2026-06-17 and shut it
# down on 2026-08-16. openai/gpt-oss-120b is Groq's own recommended
# successor as of this writing. Groq model availability changes over
# time — if this model is also gone by the time you read this, check
# https://console.groq.com/docs/models for the current list before
# assuming the code is broken.
GROQ_MODEL = "openai/gpt-oss-120b"

# --- Judge (citation verification + eval correctness scoring) ---
# Deliberately a SEPARATE knob from GENERATION_PROVIDER, not the same
# setting reused. EVAL_FINDINGS.md documents a real failure: when the
# judge is the same small model that generated the answer (llama3.1:8b
# judging llama3.1:8b), it can get an easy call wrong — a run 4 case
# (q02) scored a plainly-correct citation as UNSUPPORTED. Defaulting
# the judge to Groq's larger model, independent of whatever generates
# the answer, is the fix: a stronger, separate reviewer catches errors
# a same-class self-judge misses. This mirrors the guide's Project 2
# "async quality verification loop" pattern — route verification to
# the highest-tier model available, not necessarily the same one that
# produced the output being checked.
JUDGE_PROVIDER = "groq"   # "ollama" | "groq" — independent of GENERATION_PROVIDER

# --- Vector store collection name ---
COLLECTION_NAME = "hybrid_rag_docs"
