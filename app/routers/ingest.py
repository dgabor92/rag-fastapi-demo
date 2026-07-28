from fastapi import APIRouter, HTTPException
from app.models import IngestRequest, IngestResponse
from app.embed import embed, chunk_text
from app.db import get_pool

router = APIRouter()


@router.post('/ingest', response_model=IngestResponse)
async def ingest(body: IngestRequest) -> IngestResponse:
    if not body.text.strip():
        raise HTTPException(status_code=422, detail='text cannot be empty')

    chunks = chunk_text(body.text)
    pool = get_pool()

    async with pool.acquire() as conn:
        for chunk in chunks:
            vector = await embed(chunk)
            await conn.execute(
                'INSERT INTO documents (content, embedding) VALUES ($1, $2)',
                chunk,
                vector,
            )

    return IngestResponse(chunks_stored=len(chunks))
