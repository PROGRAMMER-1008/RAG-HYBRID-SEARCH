# Failure Forensics

An observability and root-cause analysis tool for multi-step AI pipelines — built entirely on free, local infrastructure. When a multi-step LLM pipeline produces a bad output, most teams have no idea which step broke it. This tool traces every step, runs independent LLM-as-judge review, performs backward root-cause analysis, and grows a permanent, human-verified eval dataset from its own real failures.

Runs 100% locally on [Ollama](https://ollama.com) — no paid API keys required.

---

## The problem

Imagine a document goes through a 4-step pipeline: intake, entity extraction, classification, summarization. The final output is wrong. **Which step broke it?**

In traditional software you'd read a stack trace. In an LLM pipeline, nothing crashes — the code "worked," the output is just silently wrong, and any step could be the culprit. This project answers "where did this go wrong?" the way a proper observability system would: by recording every step ("spans"), independently reviewing each one, and walking backward from the bad output to find the first genuine point of failure.

## What it does

- **Traced 3-step pipeline** — extraction → classification → summarization, each step isolated and typed (Pydantic), each call recorded as a span (input, output, confidence, latency, raw response)
- **Independent LLM-as-judge review** — a separate model call reviews each span's input/output pair without seeing the step's own reasoning, checking specifically for invented content, placeholder-as-data errors, missing information, contradictions, and over-confident framing
- **Backward root-cause analysis** — walks a trace backward from the final step, using judge scores and self/judge disagreement as evidence, to find the first genuine point of failure and classify it into a failure taxonomy (extraction hallucination, misclassification, propagation error, prompt failure, context loss, placeholder-as-data)
- **Human-in-the-loop review** — every analyzer diagnosis is a draft, not a verdict. A human confirms, corrects, or marks it unclear directly in the web UI; confirmed/corrected cases become permanent eval cases
- **Growing eval dataset** — every human-reviewed failure becomes a labeled regression test case, generated from real pipeline behavior rather than hand-written guesses
- **Live document analysis** — upload a `.txt`, `.pdf`, or `.docx` file and watch it move through the full pipeline in real time, rendered as an animated node graph
- **Full web UI** — case overview, trace explorer with per-span evidence, live analysis, and the eval dataset, all reading from the same SQLite + JSON storage layer

## A real bug I found and fixed

While testing on a real invoice (not a synthetic sample), the judge confidently — and wrongly — flagged genuine invoice fields as "invented," including the country name and the invoice's own total amount, even scoring one span as low as 2/5 with a fabricated justification.

Root cause, on inspection: the judge was never shown the original source document during summarization review — only a truncated, comma-joined string of previously extracted entities. It had no way to verify a claim against the real text, only against an already-lossy secondhand summary. A second bug compounded this: extraction's output was silently truncated at 200 characters before being shown to the judge at all, meaning entities from longer documents were being cut out of the judge's view entirely.

Fix: the judge now receives the full, untruncated original document alongside a clearly structured (not truncated, not comma-joined) list of extracted entities, plus an explicit instruction to treat table/form/address-block content as equally "stated" as prose. Re-tested against the exact invoice that exposed the bug — the false accusations disappeared and the judge correctly began catching a genuine issue instead (a placeholder bank sort code, `00-00-00`, being treated as real data).

## Known limitations

- **The judge runs on the same model class as the pipeline itself** (`llama3.1:8b` reviewing `llama3.1:8b`), not a stronger reviewer model. This is a real, measured limitation, not a hypothetical one — self-reported confidence has been observed to vary between identical runs of the same document, and the judge has been caught making a confident, well-articulated diagnosis that didn't match its own numeric score. Human review exists specifically because of this.
- **Category correction is single-label.** Some human corrections are really about judge-reasoning accuracy rather than the pipeline step itself; the current taxonomy doesn't fully capture that nuance.
- **No OCR.** Scanned or image-based PDFs won't extract text and will return a clear error rather than a silent failure.
- **Live analysis progress is an honest approximation.** A single HTTP request can't report which server-side step is currently running without websocket/SSE plumbing; the UI advances an estimated timeline rather than true step-by-step telemetry, and only reaches its real final state once the actual response returns.

## Architecture

```
Document (.txt / .pdf / .docx)
        │
        ▼
   Intake ──▶ Extraction ──▶ Classification ──▶ Summarization
        │           │               │                 │
        └───────────┴───────────────┴─────────────────┘
                          │
                    Traced Spans (SQLite + JSON)
                          │
                 Independent LLM Judge
                          │
              Backward Root-Cause Analyzer
                          │
                 Human Review (confirm / correct / unclear)
                          │
                 Growing Eval Dataset
```

## Tech stack

| Component | Choice |
|---|---|
| LLM runtime | Ollama, `llama3.1:8b` (local, free) |
| Pipeline & typing | Python, Pydantic |
| Storage | SQLite (index) + JSON (full trace detail) |
| Backend API | FastAPI |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| File parsing | `pypdf`, `python-docx` |

## Setup

**1. Install Ollama** from [ollama.com](https://ollama.com), then pull the model:
```bash
ollama pull llama3.1:8b
```

**2. Install Python dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Run the web app:**
```bash
uvicorn webui.server:app --reload --port 8000
```
Open **http://localhost:8000**

**4. (Optional) Run the pipeline from the command line** instead of the UI:
```bash
python test_judge.py            # run pipeline + judge on sample docs
python test_root_cause.py       # backward root-cause analysis on saved traces
python review_traces.py         # interactive human review (CLI version)
python view_eval_dataset.py     # view the growing eval dataset
```

## Project structure

```
failure-forensics/
├── pipeline/           # core pipeline: models, LLM client, steps, tracing, storage
├── analysis/           # judge + backward root-cause analyzer
├── webui/              # FastAPI backend + single-page frontend
├── data/                # SQLite db + JSON traces (generated at runtime)
└── test_*.py            # standalone test/demo scripts for each component
```

## Why this project exists

Most AI portfolio projects generate answers. This one catches bad ones — and is honest about the fact that its own quality-control layer has real, documented blind spots, which is exactly why a human review step is built into the architecture rather than bolted on as an afterthought.
