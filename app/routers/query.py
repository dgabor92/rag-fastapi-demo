from fastapi import APIRouter, HTTPException
import anthropic
from app.models import QueryRequest, QueryResponse, ChunkMatch
from app.embed import embed
from app.db import get_pool
from app.config import settings

router = APIRouter()

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


@router.post('/query', response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    if not body.question.strip():
        raise HTTPException(status_code=422, detail='question cannot be empty')

    vector = await embed(body.question)
    pool = get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            '''
            SELECT content, 1 - (embedding <=> $1) AS similarity
            FROM documents
            ORDER BY embedding <=> $1
            LIMIT $2
            ''',
            vector,
            body.top_k,
        )

    if not rows:
        raise HTTPException(status_code=404, detail='No documents ingested yet')

    context = '\n\n'.join(row['content'] for row in rows)
    sources = [ChunkMatch(content=row['content'], similarity=round(row['similarity'], 4)) for row in rows]

    message = await _client.messages.create(
        model='claude-haiku-4-5-20251001',
        max_tokens=1024,
        messages=[
            {
                'role': 'user',
                'content': f'Context:\n{context}\n\nQuestion: {body.question}\n\nAnswer based only on the context above.',
            }
        ],
    )

    answer = message.content[0].text
    return QueryResponse(answer=answer, sources=sources)
