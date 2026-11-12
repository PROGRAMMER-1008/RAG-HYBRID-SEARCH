# RAG Hybrid Search

An open-ended document question answering workbench. Upload files or a project folder, ask questions in natural language, and inspect the retrieved passages behind grounded answers.

## What it does

- Extracts text from PDF, DOCX, PPTX, XLSX/XLSM, CSV/TSV, EML, HTML, JSON, XML, notebooks, Markdown, common code files, and OCR-capable images. Unknown file extensions are accepted when their contents are clearly readable as text. Binary files are never executed.
- Preserves relative folder names when a folder is uploaded, so citations can distinguish files with the same basename.
- Uses BGE dense retrieval, BM25 lexical retrieval, reciprocal rank fusion, and an optional BGE cross-encoder reranker.
- Answers document questions from retrieved passages, shows excerpts and pipeline rankings, and withholds answers that have no usable citations or whose cited claims fail automated verification.
- Offers a separate general AI mode for questions that do not depend on uploaded files. It does not browse the web.
- Gives each browser session a random workspace ID and bearer token. Workspaces are process-local, expire after 45 minutes of inactivity by default, and disappear on service restart. Uploaded bytes are deleted after extraction; extracted text is held in memory until the workspace expires or is cleared.
- Includes per-file, per-workspace, per-session, daily, and per-minute limits intended for a small public demo.

Uploaded document text and questions are sent to the configured generation provider. In grounded mode, cited claims may also be sent to the configured judge provider. Do not upload confidential files unless you have permission to send them to those providers.

## Run locally

Requirements: Python 3.11, Node.js 20+, and Tesseract OCR if you need scanned PDF or image extraction.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

The default provider is Ollama. Start Ollama separately and pull the configured model, or edit `.env` to use Groq and set `GROQ_API_KEY` in that file. Keep `.env` out of Git.

`requirements-dev.txt` installs the upload API and test tooling. Install `requirements.txt` as well only if you use the optional local persistent-index command; it adds ChromaDB for that legacy workflow.

```powershell
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8001
```

Open <http://127.0.0.1:8001>. The server downloads the embedding and reranker models from Hugging Face the first time you upload a document. You can also build the optional built-in corpus index with `python -m app.ingestion.indexer`.

To run the web build separately:

```powershell
cd web
npm run build
```

## Supported inputs and limits

The live API returns the exact formats and active limits from `GET /v1/formats`. Common formats include PDF, DOCX, PPTX, XLSX/XLSM, CSV/TSV, EML, images (PNG, JPEG, TIFF, BMP, WEBP), HTML, Markdown, JSON/JSONL, XML, Jupyter notebooks, and source/configuration text. Legacy `.doc` and `.xls` files are not supported; convert them to Open XML formats. ZIP archives are not unpacked. A folder can be selected in supported browsers, or files can be selected and dropped individually.

Local defaults allow 15 MiB per file, 50 MiB total per workspace, 100 files, 5,000 chunks, 3 active workspaces, and 10 questions per workspace. The Render Blueprint uses a smaller 25 MiB total workspace cap and 2,500 chunks to fit its configured instance. Limits can be changed with environment variables, but raise them only after reviewing memory and request costs.

OCR of scanned PDFs and images requires the Tesseract executable. OCR and document parsers are bounded by page, image, archive, row, and file-size limits. A parser error is returned per file; secret-like content and common credential files are rejected without echoing the detected value.

## API overview

- `GET /v1/health` — liveness and non-secret runtime metadata.
- `GET /v1/formats` — format notes and active upload limits.
- `POST /v1/workspaces` — create a workspace; the bearer token is returned once.
- `POST /v1/workspaces/{id}/documents` — upload multipart field `files` (requires `Authorization: Bearer <token>`).
- `GET /v1/workspaces/{id}/documents` — list indexed files.
- `POST /v1/workspaces/{id}/ask` — JSON body `{"question":"...","mode":"documents"}` or `{"question":"...","mode":"general"}`.
- `DELETE /v1/workspaces/{id}` — clear a workspace.

The older `/v1/ask` endpoint remains available for a locally built shared index. The demo UI uses isolated temporary workspaces.

## Deploy: Vercel frontend + Render API

The repository is prepared for a static Vercel frontend and a Docker-based Render API. The Render service serves the UI too, so the backend remains usable on its own. The Render Docker build installs `requirements-runtime.txt`; the upload API uses an in-memory NumPy dense index and does not install ChromaDB.

### Vercel

Create a Vercel project from this repository and set its **Root Directory** to `web`. The build command (`npm run build`) and output directory (`dist`) are also declared in `web/vercel.json`. Configure the production environment variable `VITE_API_BASE_URL` as the public Render API URL, for example `https://rag-hybrid-search-api.onrender.com`. Redeploy after changing it because the value is embedded at build time. The Vercel project must use a stable production domain; update the Render service's `CORS_ORIGINS` if that domain differs from the default in `render.yaml`.

The static project uses security headers from `web/vercel.json`. The frontend can also be served by Render at the API origin.

### Render

`render.yaml` defines one public Docker web service in Singapore on the `1c-2g` plan, with a `/v1/health` health check. It keeps uploaded workspaces in memory and temporary files under `/tmp`; no persistent disk is required. The API image includes Tesseract and runs as a non-root user.

Set `GROQ_API_KEY` in the Render Dashboard as a secret before using AI modes. The Blueprint defaults both generation and citation checking to Groq. Never commit an API key. Confirm your Groq account has an appropriate usage limit; the app's in-memory request limits are only a demo guard and reset when the service restarts.

The Blueprint expects the Vercel production origin `https://rag-hybrid-search.vercel.app`. Change `CORS_ORIGINS` in `render.yaml` to the actual Vercel domain before the first sync if the project receives a different domain. Preview deployments are not included by default.

Render plan availability and prices can change. Review the plan and recurring charge in the Render Dashboard before applying the Blueprint. This single-process prototype uses in-memory sessions and rate counters; do not add multiple Render instances without moving session state, quotas, and indexing to shared services.

## Quality checks

```powershell
python -m ruff check --select E9,F63,F7,F82 app tests
python -m compileall -q app tests
python -m pytest
cd web
npm run build
```

GitHub Actions audits pinned runtime dependencies, runs the Python checks and static frontend build on pushes and pull requests. Dependabot opens dependency update pull requests. Model downloads and external LLM calls are not part of the unit tests.

## Security and operational limits

- Workspace IDs and tokens are generated with a cryptographic random source; only a hash of each token is retained by the server.
- Uploads are size-limited, parsed as data, scanned for common credentials, and never passed to a shell or imported as code. Office ZIP archives and images have expansion and pixel limits.
- The API uses a streaming upload body cap, CORS allowlist, Trusted Host checks, a restrictive frontend content security policy, and generic production error responses.
- The public runtime excludes ChromaDB because its standalone multi-tenant server has current unpatched security advisories. The optional local full `requirements.txt` still includes ChromaDB for the existing embedded persistent-index CLI; do not expose a Chroma server or use that optional path for untrusted multi-tenant data.
- Public demo quotas are process-local, not IP-based or globally distributed. A restart clears quotas and workspaces. They reduce accidental spend but are not a substitute for provider spending limits, authentication, a shared rate-limit store, or monitoring.
- The current deployment parses untrusted files inside the API container and is a learning/demo architecture. For real multi-user or sensitive workloads, move parsing into isolated workers or sandboxes, add durable job processing, shared storage, distributed quotas, observability, account-level authorization, and a reviewed data retention policy.
