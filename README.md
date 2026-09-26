# Enterprise Hybrid-RAG — Production Retrieval-Augmented Generation with Evaluation

Hybrid RAG pipeline combining NVIDIA NeMo Retriever + custom embeddings,
multi-modal retrieval (text + image + structured data), production deployment
with Docker Compose, and a continuous evaluation layer.

## Features

- NeMo Retriever — GPU-accelerated embedding and retrieval via NVIDIA NIM
- Multi-modal retrieval — text, chart, and structured-data sources in one pipeline
- Evaluation Layer — continuous RAG metrics (precision, recall, MRR) with alerts
- Production deployment — Docker Compose and Kubernetes manifests

## Quickstart

```bash
git clone https://github.com/we-do-care-global/enterprise-hybrid-rag.git
cd enterprise-hybrid-rag
docker compose up -d
# API at http://localhost:8000
```

For Python-only mode:

```bash
pip install -r requirements.txt
python -m app.main
```

## Architecture

- Python 3.11 / 3.12
- FastAPI service (`app/`)
- NVIDIA NeMo Retriever + custom embeddings
- Retrieval: BM25 + RRF fusion, Qdrant / Milvus backend
- Multi-modal sources: text, chart, structured
- Continuous evaluation with `ragas`

## Testing / CI

```bash
pytest
docker compose build
```

- CI workflow: `.github/workflows/ci.yml`
- Pages: `docs/` deployed from `main`

## Deployment

- Docker Compose (`docker-compose.yml` in repo root — verified in docs/index.html reference)
- Kubernetes manifests available
- Docker image reference: `hub.docker.com/r/we-do-care-global/enterprise-hybrid-rag`

## Metadata / Publication

- License: Apache 2.0 (`LICENSE`)
- ORCID: https://orcid.org/0009-0009-8515-2727
- Citation: `citation.cff` (CFF 1.2.0)
- Zenodo archive: `.zenodo.json` (creator ORCID embedded for DataCite → ORCID sync)
- GitHub Pages: https://we-do-care-global.github.io/enterprise-hybrid-rag/
- Latest release tag: v0.1.1
