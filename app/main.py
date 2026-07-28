from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings
from app import db
from app.routers import ingest, query, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect(settings.database_url)
    yield
    await db.disconnect()


app = FastAPI(title='RAG FastAPI Demo', lifespan=lifespan)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)