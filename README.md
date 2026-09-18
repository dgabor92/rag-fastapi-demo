# RAG FastAPI Demo

A fully containerized Retrieval-Augmented Generation (RAG) demo built with FastAPI, pgvector, Ollama, and Claude. Designed as a learning project to demonstrate how the RAG pipeline works in practice.

---

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a pattern where, instead of relying solely on an LLM's training data, you:

1. Store your own documents as vector embeddings in a database
2. At query time, find the most semantically similar chunks to the user's question
3. Pass those chunks as context to the LLM
4. The LLM answers **only based on your provided context**, not from its general knowledge

This prevents hallucination and keeps answers grounded in your actual data.

---

## Architecture

```
User
 │
 ▼
┌──────────────────────────────────────────┐
│  Frontend (React + Vite + Tailwind)      │  :3000
│  3 tabs: Ingest / Semantic / Hybrid      │
└──────────────┬───────────────────────────┘
               │ /api/* (nginx proxy)
               ▼
┌──────────────────────────────────────────┐
│  FastAPI (Python)                        │  :8000 (internal)
│  /ingest  /query  /hybrid-query  /health │
└──────┬───────────────────┬───────────────┘
       │                   │
       ▼                   ▼
┌────────────┐    ┌─────────────────────┐
│ PostgreSQL │    │  Ollama (host)      │
│ + pgvector │    │  nomic-embed-text   │
│  :5432     │    │  :11434             │
└────────────┘    └─────────────────────┘
                           │
                           ▼
                  ┌────────────────────┐
                  │  Anthropic Claude  │
                  │  claude-haiku      │
                  └────────────────────┘
```

**Key design decisions:**
- Nginx in the frontend container proxies `/api/*` to the API container — the browser never talks to the API directly
- Ollama runs on the **host machine** (not in Docker); the API container reaches it via `host.docker.internal:11434`
- PostgreSQL data persists in a named Docker volume (`pgdata`)

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker + Docker Compose | Any recent version |
| Ollama | Must be running on the host with `nomic-embed-text` pulled |
| Anthropic API key | Get one at [console.anthropic.com](https://console.anthropic.com) |

Install and start Ollama, then pull the embedding model:

```bash
ollama pull nomic-embed-text
```

---

## Setup

**1. Clone the repo**

```bash
git clone <repo-url>
cd rag-fastapi-demo
```

**2. Create your `.env` file**

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```env
DATABASE_URL="postgresql://rag:rag@localhost:5432/rag"
ANTHROPIC_API_KEY="sk-ant-..."
OLLAMA_URL="http://localhost:11434"
```

The `DATABASE_URL` and `OLLAMA_URL` values are only used when running the API outside Docker. Inside Docker Compose these are overridden automatically.

**3. Build and start all containers**

```bash
docker compose up --build -d
```

This builds both images and starts three containers: `db`, `api`, and `frontend`.

**4. Open the UI**

```
http://localhost:3000
```

---

## Running (after first build)

```bash
# Start
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f api

# Rebuild after code changes
docker compose up --build -d
```

---

## API Endpoints

All endpoints are reachable from the host via the nginx proxy at `http://localhost:3000/api/`.

### `GET /health`

Returns the API and database status.

```json
{ "status": "ok", "db": "ok" }
```

---

### `POST /ingest`

Ingests text into the vector database. The text is split into overlapping chunks (500 words, 50-word overlap), each chunk is embedded via Ollama and stored in PostgreSQL with pgvector.

**Request:**
```json
{ "text": "Your document content here..." }
```

**Response:**
```json
{ "chunks_stored": 3 }
```

**What happens internally:**
1. `chunk_text()` splits the input into word-based chunks
2. Each chunk is sent to Ollama's `/api/embeddings` endpoint → returns a 768-dimensional float vector
3. The vector and original text are stored in the `documents` table

---

### `POST /query`

Semantic (vector) search + Claude answer.

**Request:**
```json
{
  "question": "What is FastAPI?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "FastAPI is a modern Python web framework...",
  "sources": [
    { "content": "FastAPI is...", "similarity": 0.8923 },
    ...
  ]
}
```

**What happens internally:**
1. The question is embedded via Ollama (same model as ingest)
2. pgvector finds the `top_k` closest vectors using cosine distance (`<=>` operator)
3. The matching chunks are assembled into a context string
4. Claude Haiku receives `Context: ... \n\nQuestion: ...` and answers **only from the context**
5. The response includes both the answer and the source chunks with similarity scores

**Similarity score:** `1 - cosine_distance`. Range 0-1, where 1 = identical meaning.

---

### `POST /hybrid-query`

Hybrid search (semantic + full-text) using Reciprocal Rank Fusion (RRF).

**Request:**
```json
{
  "question": "What is FastAPI?",
  "top_k": 5,
  "rrf_k": 60
}
```

**Response:**
```json
{
  "answer": "FastAPI is...",
  "sources": [
    { "content": "FastAPI is...", "rrf_score": 0.016393 },
    ...
  ]
}
```

**What happens internally:**
1. **Semantic search:** finds `top_k * 2` candidates by vector similarity
2. **Full-text search:** finds `top_k * 2` candidates using PostgreSQL `tsvector` + `websearch_to_tsquery`
3. **RRF fusion:** each document scores `1 / (rrf_k + rank)` from each list it appears in; scores are summed
4. Documents appearing in **both** lists rank higher than those in only one
5. Top `top_k` results by combined score go to Claude

**When to use hybrid vs semantic:**
- Semantic: better for conceptual/paraphrased questions
- Hybrid: better when the question contains specific keywords or product names

---

## Project Structure

```
rag-fastapi-demo/
├── app/
│   ├── config.py          # Settings (loaded from .env)
│   ├── db.py              # asyncpg connection pool + table init
│   ├── embed.py           # Ollama embedding + text chunking
│   ├── llm.py             # Anthropic client
│   ├── models.py          # Pydantic request/response schemas
│   └── routers/
│       ├── health.py
│       ├── ingest.py
│       ├── query.py
│       └── hybrid_query.py
├── frontend/
│   └── src/
│       └── App.tsx        # React UI (3 tabs: Ingest, Semantic, Hybrid)
├── Dockerfile.api          # Python 3.11-slim + uvicorn
├── Dockerfile.frontend     # Node 20 multi-stage build + nginx serve
├── docker-compose.yml      # db + api + frontend with healthchecks
├── nginx.conf              # Proxy /api/ to api container
├── requirements.txt
└── .env.example
```

---

## Database Schema

The `documents` table is created automatically on startup:

```sql
CREATE TABLE IF NOT EXISTS documents (
    id        SERIAL PRIMARY KEY,
    content   TEXT NOT NULL,
    embedding VECTOR(768),            -- pgvector column (nomic-embed-text = 768 dims)
    content_tsv TSVECTOR              -- full-text search index
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops);
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | Python 3.11, FastAPI, asyncpg |
| Embeddings | Ollama + nomic-embed-text (local, free) |
| Vector DB | PostgreSQL 16 + pgvector extension |
| LLM | Anthropic Claude Haiku |
| Frontend | React 18, Vite, Tailwind CSS |
| Containerization | Docker Compose, nginx |

---

## Troubleshooting

**Ollama not reachable from the API container**

Ensure Ollama is running on the host (`ollama serve`) and that the `nomic-embed-text` model is pulled. The API container connects to it via `host.docker.internal:11434`.

**API returns 500 on /query**

Check that `ANTHROPIC_API_KEY` is set correctly in `.env`. The key is read by Docker Compose from `.env` automatically.

**Database tables not created**

The API creates tables on startup via the lifespan hook. If the DB was not yet healthy when the API started, restart the API container:

```bash
docker compose restart api
```

**Rebuild after changing Python or frontend code**

```bash
docker compose up --build -d
```
