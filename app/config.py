"""Typed, environment-driven application configuration."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)


class Settings(BaseModel):
    """Settings shared by the API, ingestion CLI, and deployment."""

    model_config = ConfigDict(extra="ignore")

    app_env: str = "development"
    generation_provider: str = "ollama"
    judge_provider: str = "ollama"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = 30.0
    max_answer_tokens: int = 1024
    max_judge_tokens: int = 256
    max_citation_checks_per_answer: int = 8
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_seconds: float = 90.0

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    enable_reranker: bool = True
    enable_citation_verification: bool = True
    enable_ocr: bool = True

    max_upload_file_bytes: int = 15 * 1024 * 1024
    max_workspace_bytes: int = 50 * 1024 * 1024
    max_upload_request_bytes: int = 60 * 1024 * 1024
    max_files_per_workspace: int = 100
    max_pages_per_file: int = 500
    max_ocr_pages_per_file: int = 30
    max_chunks_per_workspace: int = 5000
    max_active_workspaces: int = 3
    workspace_ttl_minutes: int = 45
    max_question_chars: int = 4000
    max_questions_per_workspace: int = 10
    max_daily_questions: int = 25
    max_requests_per_minute: int = 20

    cors_origins: str = "http://localhost:5173,http://localhost:8001"
    workspace_root: Path = Path(tempfile.gettempdir()) / "rag-workspaces"
    trusted_hosts: str = "localhost,127.0.0.1,testserver"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load .env locally, then validate typed values from the environment."""
        load_dotenv(override=False)
        values = {
            name: os.environ[name.upper()]
            for name in cls.model_fields
            if name.upper() in os.environ
        }
        return cls(**values)

    @field_validator("generation_provider", "judge_provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"groq", "ollama"}:
            raise ValueError("Provider must be 'groq' or 'ollama'.")
        return value

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, value: str) -> str:
        origins = [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]
        for origin in origins:
            parsed = urlsplit(origin)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
                    or "*" in parsed.netloc):
                raise ValueError("CORS_ORIGINS must be a comma-separated list of exact HTTP(S) origins.")
        return ",".join(origins)

    @field_validator("trusted_hosts")
    @classmethod
    def validate_trusted_hosts(cls, value: str) -> str:
        hosts = [host.strip().lower() for host in value.split(",") if host.strip()]
        if any("/" in host or "*" in host and not host.startswith("*.") for host in hosts):
            raise ValueError("TRUSTED_HOSTS permits exact hostnames and leading '*.' wildcards only.")
        if not hosts:
            raise ValueError("TRUSTED_HOSTS must include at least one hostname.")
        return ",".join(hosts)

    @field_validator("max_answer_tokens", "max_judge_tokens", "max_citation_checks_per_answer",
                     "max_upload_file_bytes", "max_workspace_bytes", "max_upload_request_bytes", "max_files_per_workspace",
                     "max_pages_per_file", "max_ocr_pages_per_file", "max_chunks_per_workspace", "max_active_workspaces",
                     "workspace_ttl_minutes", "max_question_chars", "max_questions_per_workspace",
                     "max_daily_questions", "max_requests_per_minute")
    @classmethod
    def positive_limits(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Resource limits must be positive integers.")
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip().rstrip("/") for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings.from_environment()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
INDEX_MANIFEST_PATH = BASE_DIR / "data" / "index_manifest.json"
GOLDEN_SET_PATH = BASE_DIR / "app" / "eval" / "golden_set.json"

EMBEDDING_MODEL = settings.embedding_model
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
DENSE_TOP_K = 12
SPARSE_TOP_K = 12
RRF_K = 60
FUSION_TOP_N = 24
RERANK_TOP_N = 6
RERANKER_MODEL = settings.reranker_model
GENERATION_PROVIDER = settings.generation_provider
JUDGE_PROVIDER = settings.judge_provider
GROQ_MODEL = settings.groq_model
OLLAMA_MODEL = settings.ollama_model
COLLECTION_NAME = "hybrid_rag_docs"
