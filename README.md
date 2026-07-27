# RAG FastAPI Demo

A REST API wrapper around a Retrieval-Augmented Generation (RAG) pipeline, built with FastAPI and Python.

This is a learning project — the goal is to take the core RAG logic from [rag-demo-python](https://github.com/dgabor92/rag-demo-python) and expose it as a proper HTTP API with validation, error handling, and production-ready patterns.

## What We're Building

A FastAPI service with three endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Accept a text document, chunk it, embed it, store in pgvector |
| `POST` | `/query` | Accept a question, return a RAG answer + top-K matching chunks with similarity scores |
| `GET` | `/health` | Check if the API is running and the database connection is alive |

## Stack

- **API**: FastAPI + uvicorn
- **Validation**: Pydantic v2
- **Embedding**: Ollama (`nomic-embed-text`, 768 dim, runs locally)
- **Database**: PostgreSQL 16 + pgvector (Docker)
- **LLM**: Claude Haiku (Anthropic API)
- **Python**: 3.11+

## Learning Goals

- FastAPI routing, path/query/body parameters
- Pydantic request and response models
- Lifespan events for managing DB connection pool on startup/shutdown
- HTTP status codes and error handling with `HTTPException`
- Running the API with uvicorn and exploring the auto-generated Swagger UI (`/docs`)

## Planned Project Structure

```
rag-fastapi-demo/
├── app/
│   ├── main.py        # FastAPI app + lifespan
│   ├── routers/
│   │   ├── ingest.py  # POST /ingest
│   │   └── query.py   # POST /query
│   ├── models.py      # Pydantic schemas
│   ├── embed.py       # Ollama embedding
│   └── db.py          # Connection pool
├── docker-compose.yml
├── requirements.txt
└── .env
```

## Prerequisites

- Docker
- Ollama with `nomic-embed-text` pulled (`ollama pull nomic-embed-text`)
- Python 3.11+
- Anthropic API key
