"""
Phase 5: FastAPI service.

Job: expose the hybrid RAG pipeline over HTTP.

Scope decision: this API is READ-ONLY (query + list documents), not
a live ingestion endpoint. Re-indexing (loader -> chunker -> dedup ->
embed -> write to ChromaDB/BM25) is a batch operation that mutates two
persistent stores and takes real wall-clock time — doing that safely
inside a single request (partial failures, concurrent readers during
a rebuild, no job queue/locking) is a different engineering problem
than serving queries, and out of scope here. Re-indexing is a CLI
command: `python -m app.ingestion.indexer`.

Startup cost matters: DenseRetriever, SparseRetriever, and Reranker
each load real models (bge-small, bge-reranker-base) and open the
ChromaDB connection — expensive (seconds, hundreds of MB). This MUST
happen once at process startup, not per-request, or every API call
would reload everything. FastAPI's lifespan context manager is the
tool for exactly this: run setup once before the server starts
accepting requests, and clean up once when it shuts down.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import RAW_DOCS_DIR
from app.generation.citation import CITATION_RE, citation_accuracy, verify_citations
from app.generation.llm_client import generate
from app.generation.prompt import build_generation_prompt
from app.retrieval.pipeline import HybridRetrievalPipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Populated once at startup, read (never mutated) by request handlers.
_pipeline: HybridRetrievalPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    print("Starting up: loading embedding model, indexes, and reranker...")
    _pipeline = HybridRetrievalPipeline()
    print("Startup complete. Ready to serve requests.")
    yield
    # Nothing to explicitly tear down: ChromaDB's PersistentClient and
    # in-memory models don't hold resources that need manual closing.
    print("Shutting down.")


app = FastAPI(
    title="RAG Hybrid Search API",
    description="Hybrid (dense + sparse) retrieval-augmented generation over a personal project corpus.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request/response schemas ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask.")


class CitationInfo(BaseModel):
    index: int                # the [N] number as shown to the user
    source_file: str
    heading_path: str
    rerank_score: float
    was_cited: bool            # did the generated answer actually cite this [N]?
    # All 5 reranked chunks are returned (not just cited ones) so the
    # full retrieval evidence trail stays visible — e.g. a citation like
    # [5] scoring 0.00004 while [1] scores 0.876 is a real signal about
    # how confidently the reranker discriminated relevant from irrelevant
    # content for this specific question, and hiding the low-score
    # chunks would hide that signal. was_cited answers "did the model
    # actually use this" without deleting the ones it didn't.


class StageHit(BaseModel):
    """One chunk's result at a single pipeline stage — used for the
    dashboard's stage-by-stage relay view, not for the final answer."""
    source_file: str
    heading_path: str
    score: float


class PipelineStages(BaseModel):
    """Counts + top few results at each retrieval stage, so a caller
    (the dashboard) can render the dense/sparse -> fusion -> rerank
    relay without needing to re-run retrieval itself. Capped at 5 per
    stage rather than returning all 10 dense/10 sparse candidates —
    enough to show the shape of each stage without bloating every
    response with candidates that never influenced the final answer."""
    dense_count: int
    sparse_count: int
    fused_count: int
    dense_top: list[StageHit]
    sparse_top: list[StageHit]
    fused_top: list[StageHit]


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationInfo]
    citation_accuracy: float  # fraction of citations verified as SUPPORTED, see citation.py
    retrieval_confidence: float  # top reranked chunk's score — see note in retrieve_and_answer
    pipeline_stages: PipelineStages


class DocumentInfo(BaseModel):
    filename: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    count: int


# --- Endpoints ---

@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Run a question through the full pipeline: retrieve -> generate -> verify citations."""
    if _pipeline is None:
        # Defensive only — lifespan guarantees this is set before the
        # server accepts traffic, so this branch should be unreachable
        # in practice, but failing loudly beats a confusing AttributeError.
        raise HTTPException(status_code=503, detail="Pipeline not yet initialized.")

    trace = _pipeline.retrieve(request.question)
    prompt = build_generation_prompt(request.question, trace.final_results)
    answer = generate(prompt)

    checks = verify_citations(answer, trace.final_results)
    cit_acc = citation_accuracy(checks)

    # Which [N] indices did the model actually cite anywhere in the
    # answer text? Reuses the exact same regex citation.py's verifier
    # uses, so "was this cited" and "was this citation supported" stay
    # consistent with each other rather than computed by two different
    # ad-hoc parsers that could disagree.
    cited_indices = {int(n) for n in CITATION_RE.findall(answer)}

    # retrieval_confidence: the TOP reranked chunk's cross-encoder score,
    # not an average across all 5. A weak top score is the strongest
    # single signal that retrieval didn't find great supporting evidence
    # for this specific question, even if generation still produced a
    # plausible-sounding answer — exactly the mismatch we saw in Phase 2
    # (a 0.016-scored chunk that still happened to hold the right facts).
    # Averaging across all 5 would dilute that signal, not preserve it.
    retrieval_confidence = trace.final_results[0].rerank_score if trace.final_results else 0.0

    citations = [
        CitationInfo(
            index=i,
            source_file=r.source_file,
            heading_path=r.heading_path,
            rerank_score=r.rerank_score,
            was_cited=(i in cited_indices),
        )
        for i, r in enumerate(trace.final_results, start=1)
    ]

    stages = PipelineStages(
        dense_count=trace.dense_count,
        sparse_count=trace.sparse_count,
        fused_count=trace.fused_count,
        dense_top=[
            StageHit(source_file=r.source_file, heading_path=r.heading_path, score=r.score)
            for r in trace.dense_hits[:5]
        ],
        sparse_top=[
            StageHit(source_file=r.source_file, heading_path=r.heading_path, score=r.score)
            for r in trace.sparse_hits[:5]
        ],
        fused_top=[
            StageHit(source_file=r.source_file, heading_path=r.heading_path, score=r.rrf_score)
            for r in trace.fused_hits[:5]
        ],
    )

    return AskResponse(
        question=request.question,
        answer=answer,
        citations=citations,
        citation_accuracy=cit_acc,
        retrieval_confidence=retrieval_confidence,
        pipeline_stages=stages,
    )


@app.get("/v1/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    """List the source documents currently indexed (read from disk, not the index itself —
    see note below for why that distinction matters)."""
    # Reading RAW_DOCS_DIR rather than querying ChromaDB's metadata is a
    # deliberate simplification: it reports what's on disk, which may in
    # principle drift from what's actually indexed if someone edits
    # data/raw/ without re-running the indexer. Acceptable for this
    # project's scope (a single operator, not a multi-user system) —
    # worth knowing about, not worth over-engineering around here.
    md_files = sorted(RAW_DOCS_DIR.glob("*.md"))
    return DocumentsResponse(
        documents=[DocumentInfo(filename=f.name) for f in md_files],
        count=len(md_files),
    )


@app.get("/v1/health")
def health() -> dict:
    """Basic liveness/readiness check — is the pipeline loaded and ready."""
    return {"status": "ok" if _pipeline is not None else "starting"}


@app.get("/")
def dashboard() -> FileResponse:
    """Serve the retrieval console UI at the app's root, so the whole
    project (API + frontend) is reachable from one URL — important for
    a portfolio demo link where a recruiter shouldn't need to know
    /docs exists to see anything."""
    return FileResponse(STATIC_DIR / "dashboard.html")
