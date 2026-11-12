FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/tmp/huggingface \
    FASTEMBED_CACHE_DIR=/opt/fastembed \
    PORT=8001

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app

ARG DEPENDENCY_FILE=requirements-runtime.txt
COPY requirements.txt requirements-runtime.txt ./
RUN pip install --no-cache-dir -r "${DEPENDENCY_FILE}"
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/fastembed', threads=1, cuda=False)"

COPY app/ ./app/
COPY web/ ./web/

RUN mkdir -p /app/data /tmp/rag-workspaces /tmp/huggingface /opt/fastembed \
    && chown -R app:app /app/data /tmp/rag-workspaces /tmp/huggingface /opt/fastembed
USER app

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8001')+'/v1/health',timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8001} --proxy-headers"]
