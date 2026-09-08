# Dockerfile for the RAG Hybrid Search API + dashboard.
#
# SCOPE (deliberate, not an oversight): this container runs ONLY the
# FastAPI app. It does NOT run Ollama or build the retrieval index.
#
# Why: this project's local generation model (Ollama, llama3.1:8b) is
# an 8B-parameter model meant to use your host machine's GPU.
# Containerizing an LLM runtime means configuring GPU passthrough into
# Docker (host-specific, often fragile) and re-downloading multi-GB
# model weights on every image build — worse than just running Ollama
# on the host, which already works. Real production RAG systems
# usually split this the same way: the application is containerized,
# the model-serving layer is a separate, independently-run service
# reached over the network (exactly how Groq already works here).
#
# Similarly, this container does NOT build data/chroma_db or
# data/bm25_index.pkl at build time — that needs Hugging Face Hub
# access for the embedding model, which you don't want re-fetched on
# every image rebuild. Build the index once locally
# (`python -m app.ingestion.indexer`), then mount `data/` as a volume
# at container run time (see docker-compose.yml) so the container
# reads the index you already built, rather than rebuilding it.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8001

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8001"]