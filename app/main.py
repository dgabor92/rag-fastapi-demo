from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app import db
from app.routers import ingest, query, health, hybrid_query

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect(settings.database_url)
    yield
    await db.disconnect()


app = FastAPI(title='RAG FastAPI Demo', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(hybrid_query.router)