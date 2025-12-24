# Sentiment Analysis & Semantic Search System (Dockerized)

This project is the **final submission** for the *Professional Training* course.  
It implements a **production-style microservice architecture** with strict separation of concerns, exactly as required by the instructor.

---

## System Architecture

The application is composed of **three independent Docker services**:

| Service | Port | Responsibility |
|------|------|----------------|
| **sentiment-api** | `7860` | REST API (FastAPI) for sentiment inference |
| **sentiment-ui** | `7861` | Frontend UI (Gradio) – calls API over HTTP |
| **chroma** | `8000` | Vector Database (ChromaDB) |

### Key Design Guarantees
- UI **never loads ML models**
- API **never renders UI**
- Vector DB runs in a **separate container**
- All communication happens via **Docker network + HTTP**
---

## Project Structure

```
sentiment-analysis-project/
│
├── app/
│   ├── api.py           # FastAPI backend (REST API)
│   ├── ui.py            # Gradio UI (calls API)
│   ├── model.py         # Sentiment model loader
│   ├── predict.py       # Inference logic
│   ├── preprocess.py   # Text cleaning
│   ├── vector_store.py # ChromaDB + embedding logic
│   └── __init__.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── venv/ (local environment)
```

---

## Models Used

### 1 Sentiment Classification Model
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Library**: Hugging Face Transformers
- **Used in**: `sentiment-api`

### 2 Embedding Model (Semantic Search)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Library**: Sentence-Transformers
- **Used in**: API + UI (via Vector DB)

---

## Prerequisites

Ensure the following are installed:

- Docker (v20+)
- Docker Compose v2
- Python 3.10
- Git

Verify:
```bash
docker --version
docker compose version
```

---

## How to Run

### 1 Clone the Repository
```bash
git clone https://github.com/ahtbd/Sentiment-Analysis.git
cd sentiment-analysis-project
```

---

### 2 Create Virtual Environment (Host)
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3 Cache Required Models (ONE TIME ONLY)

#### Cache sentiment model
```bash
python - <<EOF
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
AutoTokenizer.from_pretrained(MODEL)
AutoModelForSequenceClassification.from_pretrained(MODEL)
print("Sentiment model cached")
EOF
```

#### Cache embedding model
```bash
python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model cached")
EOF
```

Models are stored at:
```
~/.cache/huggingface/
```

---

### 4 Verify Docker Volume Path

Open `docker-compose.yml` and ensure **absolute path** is used:

```yaml
- /home/USERNAME/.cache/huggingface:/hf:ro
```

 `~` is NOT allowed in Docker Compose.

---

### 5 Build & Run Containers

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

---

## Access the Application

| Component | URL |
|-------|-----|
| **API (Swagger UI)** | http://localhost:7860/docs |
| **UI (Gradio)** | http://localhost:7861 |
| **ChromaDB** | http://localhost:8000 |

---

## Development Guide

### Run API Locally (Without Docker)
```bash
source venv/bin/activate
uvicorn app.api:app --reload --port 7860
```

### Run UI Locally (Without Docker)
```bash
python -m app.ui
```

---

## Docker Services Summary

| Service | Role |
|------|------|
| sentiment-api | REST API (FastAPI + Transformers) |
| sentiment-ui | Frontend UI (Gradio) |
| chroma | Vector database (semantic search) |

---

## Offline & Reproducibility

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `SENTENCE_TRANSFORMERS_HOME` explicitly set
- All Hugging Face models loaded with `local_files_only=True`

 The system runs **fully offline** after initial caching.

---

## Cleanup Commands

```bash
docker compose down
docker rm -f sentiment-api sentiment-ui 2>/dev/null
```

---

> This project demonstrates a complete, offline-safe, Dockerized ML system with proper microservice separation, suitable for real-world deployment and academic evaluation.
