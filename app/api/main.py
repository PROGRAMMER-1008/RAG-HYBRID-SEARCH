"""FastAPI service for temporary, isolated, multi-format document workspaces."""

from __future__ import annotations

import logging
import re
import threading
import uuid
import asyncio
from contextlib import suppress
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import CHROMA_DIR, INDEX_MANIFEST_PATH, RAW_DOCS_DIR, settings
from app.generation.citation import (
    CITATION_RE,
    citation_accuracy,
    citation_check_count,
    citations_cover_every_claim,
    verify_citations,
)
from app.generation.llm_client import generate
from app.generation.prompt import GenerationPrompt, build_generation_prompt
from app.ingestion.loader import (
    RawDocument,
    extension_for_path,
    load_document,
    sanitize_source_name,
    supported_extensions,
)
from app.retrieval.pipeline import HybridRetrievalPipeline
from app.retrieval.embedding import FastEmbedEncoder
from app.retrieval.workspace import WorkspacePipeline
from app.api.middleware import RequestBodyLimitMiddleware
from app.api.workspaces import Workspace, WorkspaceStore

logger = logging.getLogger("rag.api")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"

_model_lock = threading.Lock()
_embedding_model = None
_reranker_model = None
_legacy_pipeline: HybridRetrievalPipeline | None = None
_legacy_pipeline_lock = threading.Lock()


class _LockedModel:
    """Serialize inference on shared models to bound memory and thread pressure."""

    def __init__(self, model):
        self._model = model
        self._lock = threading.Lock()

    def encode(self, *args, **kwargs):
        with self._lock:
            return self._model.encode(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with self._lock:
            return self._model.predict(*args, **kwargs)


def _get_models():
    global _embedding_model, _reranker_model
    if _embedding_model is not None:
        return _embedding_model, _reranker_model
    with _model_lock:
        if _embedding_model is None:
            try:
                embedding = FastEmbedEncoder(
                    settings.embedding_model,
                    cache_dir=settings.fastembed_cache_dir,
                    threads=1,
                )
                reranker = None
                if settings.enable_reranker:
                    from sentence_transformers import CrossEncoder

                    reranker = CrossEncoder(settings.reranker_model, device="cpu")
            except Exception as exc:
                _log_failure("Failed to load retrieval models.", exc)
                raise RuntimeError("Retrieval models could not be loaded. Check model access and backend memory.") from exc
            _embedding_model = _LockedModel(embedding)
            _reranker_model = _LockedModel(reranker) if reranker is not None else None
    return _embedding_model, _reranker_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.workspaces = WorkspaceStore()
    app.state.started_at = datetime.now(timezone.utc)
    cleanup_task = asyncio.create_task(_cleanup_expired_workspaces(app.state.workspaces))
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        app.state.workspaces.close()


async def _cleanup_expired_workspaces(store: WorkspaceStore) -> None:
    while True:
        await asyncio.sleep(60)
        store.cleanup_expired()


app = FastAPI(
    title="RAG Hybrid Search",
    description="Upload common documents, search them with hybrid retrieval, and ask grounded or general AI questions.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_env != "production" else None,
    redoc_url=None,
)
app.add_middleware(RequestBodyLimitMiddleware, max_bytes=settings.max_upload_request_bytes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)
app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[host.strip() for host in settings.trusted_hosts.split(",") if host.strip()],
)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=1, max_length=settings.max_question_chars)


class WorkspaceAskRequest(AskRequest):
    mode: Literal["documents", "general"] = "documents"


class CitationInfo(BaseModel):
    index: int
    source_file: str
    heading_path: str
    rerank_score: float
    was_cited: bool
    text_excerpt: str = ""
    verified_claims: list[dict] = Field(default_factory=list)


class StageHit(BaseModel):
    source_file: str
    heading_path: str
    score: float


class PipelineStages(BaseModel):
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
    citation_accuracy: float
    retrieval_confidence: float
    pipeline_stages: PipelineStages


class WorkspaceInfo(BaseModel):
    workspace_id: str
    access_token: str
    expires_at: datetime
    supported_extensions: list[str]


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    content_hash: str


class WorkspaceDocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_bytes: int
    chunk_count: int
    errors: list[dict]


class WorkspaceAskResponse(BaseModel):
    question: str
    mode: Literal["documents", "general"]
    answer: str
    notice: str
    grounded: bool
    citations: list[CitationInfo]
    citation_accuracy: float | None
    citation_verification_available: bool
    retrieval_score: float | None
    pipeline_stages: PipelineStages | None


def _get_workspace(request: Request, workspace_id: str, authorization: str | None) -> Workspace:
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    try:
        return request.app.state.workspaces.authorize(workspace_id, token)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _enforce_rate_limit(request: Request) -> None:
    if not request.app.state.workspaces.allow_request():
        raise HTTPException(status_code=429, detail="The public demo is busy. Try again in a minute.")


def _unique_source_name(existing: list[RawDocument], name: str) -> str:
    names = {document.source_path.casefold() for document in existing}
    if name.casefold() not in names:
        return name
    path = Path(name)
    suffix = path.suffix
    stem = path.stem
    number = 2
    candidate = (path.parent / f"{stem} ({number}){suffix}").as_posix()
    while candidate.casefold() in names:
        number += 1
        candidate = (path.parent / f"{stem} ({number}){suffix}").as_posix()
    return candidate


def _log_failure(message: str, exc: Exception) -> None:
    """Avoid writing user prompts, document excerpts, and provider payloads to prod logs."""
    if settings.app_env == "production":
        logger.error("%s error_type=%s", message, type(exc).__name__)
    else:
        logger.exception(message)


def _stages(trace) -> PipelineStages:
    return PipelineStages(
        dense_count=trace.dense_count,
        sparse_count=trace.sparse_count,
        fused_count=trace.fused_count,
        dense_top=[StageHit(source_file=r.source_file, heading_path=r.heading_path, score=r.score)
                   for r in trace.dense_hits[:5]],
        sparse_top=[StageHit(source_file=r.source_file, heading_path=r.heading_path, score=r.score)
                    for r in trace.sparse_hits[:5]],
        fused_top=[StageHit(source_file=r.source_file, heading_path=r.heading_path, score=r.rrf_score)
                   for r in trace.fused_hits[:5]],
    )


def _citation_info(answer: str, trace, checks: list) -> list[CitationInfo]:
    cited_indices = {int(number) for number in CITATION_RE.findall(answer)}
    claim_details: dict[int, list[dict]] = {}
    for check in checks:
        for index, verdict in check.verdicts.items():
            claim_details.setdefault(index, []).append({"claim": check.claim_sentence, "verdict": verdict})
    return [
        CitationInfo(
            index=index,
            source_file=result.source_file,
            heading_path=result.heading_path,
            rerank_score=result.rerank_score,
            was_cited=index in cited_indices,
            text_excerpt=result.text[:1800],
            verified_claims=claim_details.get(index, []),
        )
        for index, result in enumerate(trace.final_results, start=1)
    ]


def _load_legacy_pipeline() -> HybridRetrievalPipeline:
    global _legacy_pipeline
    if _legacy_pipeline is not None:
        return _legacy_pipeline
    with _legacy_pipeline_lock:
        if _legacy_pipeline is None:
            if not CHROMA_DIR.exists() or not INDEX_MANIFEST_PATH.exists():
                raise HTTPException(
                    status_code=409,
                    detail="No built-in index is available. Create a temporary workspace and upload documents first.",
                )
            _legacy_pipeline = HybridRetrievalPipeline()
    return _legacy_pipeline


@app.post("/v1/workspaces", response_model=WorkspaceInfo, status_code=status.HTTP_201_CREATED)
def create_workspace(request: Request) -> WorkspaceInfo:
    """Create an isolated, temporary workspace; its token is shown only once."""
    _enforce_rate_limit(request)
    try:
        workspace, token = request.app.state.workspaces.create()
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    expires_at = datetime.fromtimestamp(workspace.expires_at, tz=timezone.utc)
    return WorkspaceInfo(
        workspace_id=workspace.workspace_id,
        access_token=token,
        expires_at=expires_at,
        supported_extensions=sorted(supported_extensions()),
    )


@app.get("/v1/formats")
def formats() -> dict:
    return {
        "extensions": sorted(supported_extensions()),
        "limits": {
            "per_file_bytes": settings.max_upload_file_bytes,
            "per_workspace_bytes": settings.max_workspace_bytes,
            "files_per_workspace": settings.max_files_per_workspace,
            "pages_or_slides_per_file": settings.max_pages_per_file,
            "workspace_ttl_minutes": settings.workspace_ttl_minutes,
        },
        "notes": [
            "Images and scanned PDF pages use OCR; Tesseract is required by the server image.",
            "Legacy .doc and .xls formats are not supported; save them as .docx or .xlsx.",
            "Unknown extensions are accepted only when their contents are safely readable as plain text.",
        ],
        "accept_any_text": True,
    }


@app.post("/v1/workspaces/{workspace_id}/documents", response_model=WorkspaceDocumentsResponse)
def upload_documents(
    request: Request,
    workspace_id: str,
    files: list[UploadFile] = File(...),
    authorization: str | None = Header(default=None),
) -> WorkspaceDocumentsResponse:
    _enforce_rate_limit(request)
    workspace = _get_workspace(request, workspace_id, authorization)
    if not files:
        raise HTTPException(status_code=422, detail="Choose one or more files to upload.")

    with workspace.lock:
        if len(workspace.documents) + len(files) > settings.max_files_per_workspace:
            raise HTTPException(status_code=413, detail="Too many files for this workspace.")
        current_bytes = sum(document.size_bytes for document in workspace.documents)
        current_text_bytes = sum(len(document.text.encode("utf-8")) for document in workspace.documents)
        new_text_bytes = 0
        new_documents: list[RawDocument] = []
        errors: list[dict] = []
        for upload in files:
            try:
                safe_name = sanitize_source_name(upload.filename or "")
            except ValueError:
                errors.append({"filename": "(unnamed)", "error": "A valid filename is required."})
                continue
            extension = extension_for_path(Path(safe_name))
            try:
                payload = upload.file.read(settings.max_upload_file_bytes + 1)
                if len(payload) > settings.max_upload_file_bytes:
                    raise ValueError("File exceeds the configured per-file upload limit.")
                if current_bytes + sum(doc.size_bytes for doc in new_documents) + len(payload) > settings.max_workspace_bytes:
                    raise ValueError("Workspace exceeds the total upload limit.")
                if not payload:
                    raise ValueError("Empty files cannot be indexed.")
                safe_extension = extension if re.fullmatch(r"[a-z0-9]{1,16}", extension) else "upload"
                temporary_path = workspace.root / f"{uuid.uuid4().hex}.{safe_extension}"
                temporary_path.write_bytes(payload)
                try:
                    parsed = load_document(temporary_path, source_name=safe_name)
                finally:
                    temporary_path.unlink(missing_ok=True)
                if any(doc.content_hash == parsed.content_hash for doc in workspace.documents + new_documents):
                    errors.append({"filename": safe_name, "error": "This file is already in the workspace."})
                    continue
                parsed_text_bytes = len(parsed.text.encode("utf-8"))
                if current_text_bytes + new_text_bytes + parsed_text_bytes > settings.max_workspace_bytes:
                    raise ValueError("Extracted text exceeds the configured workspace processing limit.")
                parsed.source_path = _unique_source_name(workspace.documents + new_documents, safe_name)
                new_documents.append(parsed)
                new_text_bytes += parsed_text_bytes
            except ValueError as exc:
                errors.append({"filename": safe_name, "error": str(exc)})
            except OSError:
                logger.warning("Temporary upload processing failed (error_type=%s).", "OSError")
                errors.append({"filename": safe_name, "error": "The file could not be processed safely."})
            finally:
                upload.file.close()

        if not new_documents:
            raise HTTPException(status_code=422, detail={"message": "No files were added.", "errors": errors})

        candidate_documents = workspace.documents + new_documents
        try:
            embedding_model, reranker_model = _get_models()
            candidate_pipeline = WorkspacePipeline.build(candidate_documents, embedding_model, reranker_model)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            _log_failure("Workspace indexing failed.", exc)
            raise HTTPException(status_code=503, detail="Could not build the document index. Try a smaller upload.") from exc

        workspace.documents = candidate_documents
        workspace.pipeline = candidate_pipeline
        return WorkspaceDocumentsResponse(
            documents=[
                DocumentInfo(filename=doc.source_path, size_bytes=doc.size_bytes, content_hash=doc.content_hash)
                for doc in workspace.documents
            ],
            total_bytes=sum(doc.size_bytes for doc in workspace.documents),
            chunk_count=len(workspace.pipeline.chunks),
            errors=errors,
        )


@app.get("/v1/workspaces/{workspace_id}/documents", response_model=WorkspaceDocumentsResponse)
def list_workspace_documents(
    request: Request,
    workspace_id: str,
    authorization: str | None = Header(default=None),
) -> WorkspaceDocumentsResponse:
    workspace = _get_workspace(request, workspace_id, authorization)
    return WorkspaceDocumentsResponse(
        documents=[DocumentInfo(filename=doc.source_path, size_bytes=doc.size_bytes, content_hash=doc.content_hash)
                   for doc in workspace.documents],
        total_bytes=sum(doc.size_bytes for doc in workspace.documents),
        chunk_count=len(workspace.pipeline.chunks) if workspace.pipeline else 0,
        errors=[],
    )


@app.delete(
    "/v1/workspaces/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_workspace(
    request: Request,
    workspace_id: str,
    authorization: str | None = Header(default=None),
) -> Response:
    try:
        request.app.state.workspaces.remove(workspace_id, _bearer(authorization))
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _bearer(authorization: str | None) -> str | None:
    return authorization[7:].strip() if authorization and authorization.startswith("Bearer ") else None


@app.post("/v1/workspaces/{workspace_id}/ask", response_model=WorkspaceAskResponse)
def ask_workspace(
    request: Request,
    workspace_id: str,
    body: WorkspaceAskRequest,
    authorization: str | None = Header(default=None),
) -> WorkspaceAskResponse:
    _enforce_rate_limit(request)
    workspace = _get_workspace(request, workspace_id, authorization)
    if not request.app.state.workspaces.allow_question(workspace):
        raise HTTPException(status_code=429, detail="The public demo reached its question limit. Try again later.")

    if body.mode == "general":
        prompt = GenerationPrompt(
            system_prompt=(
                "You are a general-purpose AI assistant. Answer the user's question clearly and helpfully. "
                "You have not searched the web and must not claim that you have. State uncertainty when appropriate. "
                "This answer is not grounded in or verified against the user's uploaded documents."
            ),
            user_prompt=body.question,
            numbered_chunks=[],
        )
        try:
            answer = generate(prompt)
        except Exception as exc:
            _log_failure("General answer provider call failed.", exc)
            raise HTTPException(status_code=502, detail="The AI provider did not return an answer. Try again later.") from exc
        return WorkspaceAskResponse(
            question=body.question,
            mode="general",
            answer=answer,
            notice="General AI answer; it is not checked against uploaded files or live web sources.",
            grounded=False,
            citations=[],
            citation_accuracy=None,
            citation_verification_available=False,
            retrieval_score=None,
            pipeline_stages=None,
        )

    if not workspace.documents or workspace.pipeline is None:
        raise HTTPException(status_code=409, detail="Upload at least one supported file before asking document questions.")

    with workspace.lock:
        try:
            trace = workspace.pipeline.retrieve(body.question)
            if not trace.final_results:
                return WorkspaceAskResponse(
                    question=body.question,
                    mode="documents",
                    answer="The uploaded documents do not provide enough supported information to answer this question.",
                    notice="No sufficiently relevant passages were found, so the response was withheld.",
                    grounded=False,
                    citations=[],
                    citation_accuracy=None,
                    citation_verification_available=False,
                    retrieval_score=None,
                    pipeline_stages=_stages(trace),
                )
            prompt = build_generation_prompt(body.question, trace.final_results)
            answer = generate(prompt)
        except Exception as exc:
            _log_failure("Grounded answer provider or retrieval call failed.", exc)
            raise HTTPException(status_code=502, detail="The retrieval or AI provider failed. Try again later.") from exc

        if not citations_cover_every_claim(answer, len(trace.final_results)):
            return WorkspaceAskResponse(
                question=body.question,
                mode="documents",
                answer="The uploaded documents do not provide enough supported information to answer this question.",
                notice="The model did not return a verifiable answer, so the response was withheld.",
                grounded=False,
                citations=_citation_info(answer, trace, []),
                citation_accuracy=None,
                citation_verification_available=False,
                retrieval_score=trace.final_results[0].rerank_score if trace.final_results else None,
                pipeline_stages=_stages(trace),
            )

        checks = []
        verification_available = False
        verification_skipped = (
            settings.enable_citation_verification
            and citation_check_count(answer) > settings.max_citation_checks_per_answer
        )
        if settings.enable_citation_verification and not verification_skipped:
            try:
                checks = verify_citations(answer, trace.final_results)
                verification_available = True
            except Exception:
                # Preserve the generated answer, but never represent failed
                # verification as a successful citation check.
                logger.warning("Citation verification provider failed.")
        accuracy = citation_accuracy(checks) if checks else None
        grounded = True
        if checks and any(verdict != "SUPPORTED" for check in checks for verdict in check.verdicts.values()):
            answer = "The uploaded documents do not provide enough supported information to answer this question."
            notice = "The generated answer was withheld because at least one cited claim was not fully supported."
            grounded = False
        elif verification_available:
            notice = "Answer grounded in retrieved passages. Check the cited excerpts below."
        elif verification_skipped:
            notice = "Answer grounded in retrieved passages; automated verification was skipped because the answer cited too many claims. Check each excerpt below."
        else:
            notice = "Answer grounded in retrieved passages; automated citation verification was unavailable. Check each excerpt below."
        return WorkspaceAskResponse(
            question=body.question,
            mode="documents",
            answer=answer,
            notice=notice,
            grounded=grounded,
            citations=_citation_info(answer, trace, checks),
            citation_accuracy=accuracy,
            citation_verification_available=verification_available,
            retrieval_score=trace.final_results[0].rerank_score if trace.final_results else None,
            pipeline_stages=_stages(trace),
        )


@app.get("/v1/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "mode": settings.app_env,
        "models_loaded": _embedding_model is not None,
        "generation_provider": settings.generation_provider,
        "uploads_enabled": True,
    }


@app.get("/v1/ready")
def ready(request: Request) -> JSONResponse:
    models_ready = _embedding_model is not None and (
        not settings.enable_reranker or _reranker_model is not None
    )
    return JSONResponse(
        status_code=200 if models_ready else 503,
        content={
            "status": "ready" if models_ready else "warming",
            "models_loaded": models_ready,
            "workspace_capacity": settings.max_active_workspaces,
        },
    )


@app.get("/v1/documents")
def list_documents() -> dict:
    """Legacy endpoint listing files in the optional local starter corpus."""
    files = sorted(
        path.relative_to(RAW_DOCS_DIR).as_posix()
        for path in RAW_DOCS_DIR.rglob("*")
        if path.is_file() and extension_for_path(path) in supported_extensions()
    ) if RAW_DOCS_DIR.exists() else []
    return {"documents": [{"filename": name} for name in files], "count": len(files)}


@app.post("/v1/ask", response_model=AskResponse)
def ask_legacy(request: AskRequest) -> AskResponse:
    """Backward-compatible query against the optional prebuilt local index."""
    pipeline = _load_legacy_pipeline()
    try:
        trace = pipeline.retrieve(request.question)
        answer = generate(build_generation_prompt(request.question, trace.final_results))
        checks = verify_citations(answer, trace.final_results) if settings.enable_citation_verification else []
    except HTTPException:
        raise
    except Exception as exc:
        _log_failure("Legacy ask endpoint failed.", exc)
        raise HTTPException(status_code=502, detail="The retrieval or AI provider failed. Try again later.") from exc
    return AskResponse(
        question=request.question,
        answer=answer,
        citations=_citation_info(answer, trace, checks),
        citation_accuracy=citation_accuracy(checks),
        retrieval_confidence=trace.final_results[0].rerank_score if trace.final_results else 0.0,
        pipeline_stages=_stages(trace),
    )


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/config.js")
def frontend_config() -> Response:
    return Response(
        content="window.RAG_API_BASE = window.location.origin;\n",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = uuid.uuid4().hex[:12]
    if settings.app_env == "production":
        logger.error("Unhandled request failure (request_id=%s, error_type=%s).", request_id, type(exc).__name__)
    else:
        logger.exception("Unhandled request failure (request_id=%s).", request_id)
    return JSONResponse(status_code=500, content={"detail": "Internal server error.", "request_id": request_id})
