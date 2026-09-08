# 🔍 AI Corruption Pattern Detector

> Exposing fraud in government procurement through Machine Learning & Graph Intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What It Does

An end-to-end unsupervised anomaly detection system that ingests real government
procurement data, engineers 30+ behavioral features, runs a four-model ML ensemble,
maps corruption networks using graph analytics, and delivers a **0–100 corruption
risk score** for every contract with a plain-English explanation.

## Corruption Patterns Detected

| Pattern | Detection Method |
|---|---|
| Bid Rigging | Isolation Forest + tight bid spread flag |
| Contract Splitting | Same-day award clustering |
| Shell Company Networks | Shared address detection + graph communities |
| Sole Source Abuse | Agency-level baseline deviation |
| Inflated Pricing | NAICS category price deviation |
| Kickback Networks | Louvain community detection on vendor graph |

## Tech Stack

- **ML:** Scikit-learn (Isolation Forest, LOF, One-Class SVM), PyTorch Autoencoder
- **Graph:** NetworkX, python-louvain, PageRank, Betweenness Centrality
- **API:** FastAPI, Uvicorn
- **Dashboard:** D3.js, Plotly, HTML5/CSS3
- **Data:** USASpending.gov API, World Bank Contracts API

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Collect data (hits real APIs + generates synthetic)
python src/ingestion/data_collector.py

# 3. Run the full analysis pipeline
jupyter notebook notebooks/corruption_detector.ipynb

# 4. Start the API server (new terminal)
uvicorn src.api.main:app --reload --port 8000

# 5. Open the dashboard
open dashboard/index.html
```

## Project Structure
```
├── notebooks/          # Main Jupyter analysis pipeline
├── src/
│   ├── ingestion/      # Data collection from real APIs
│   ├── analytics/      # Feature engineering, graph analytics, risk scoring
│   ├── models/         # ML ensemble + PyTorch autoencoder
│   └── api/            # FastAPI backend (7 endpoints)
├── dashboard/          # Live investigator dashboard
├── data/sample/        # Sample synthetic dataset
└── requirements.txt
```

## Model Performance

- **ROC-AUC: 0.681–0.683** validated against ground-truth injected corruption patterns (verified against actual model output, not an earlier README estimate)
- **$0 infrastructure cost** — runs entirely on public data and open-source tools
- **< 10 minutes** full pipeline on a standard laptop

## Data Sources

- [USASpending.gov API](https://api.usaspending.gov) — US Federal contracts (no key needed)
- [World Bank Contracts](https://finances.worldbank.org/resource/kdui-wcs3.json) — International procurement
- [FPDS Bulk Data](https://www.fpds.gov) — Full US procurement dataset
- [EU TED Database](https://data.europa.eu/data/datasets/ted-csv) — European tenders